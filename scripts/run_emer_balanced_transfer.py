#!/usr/bin/env python3
"""Exploratory controlled sparse-task training; preserve completed CV evidence."""
import argparse
import copy
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from paretojev.data.synthetic import collate_slates
from paretojev.models.emer import EMERReranker,pairwise_losses,utility,advantage_weights
from paretojev.experiments.comparison import _save_checkpoint,_source_identity,_output_lock
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from paretojev.utils.seed import seed_everything
from run_emer_jev import dataset,load_model,predict,validation,paired_report
from run_emer_scale import immutable_json


def warm_start(anchor,use_jev):
    model=EMERReranker(replace(anchor.config,use_jev=use_jev))
    model.load_state_dict(anchor.state_dict())
    start=model.config.feature_dim+2*model.config.objectives
    with torch.no_grad():model.projection[0].weight[:,start:start+4*model.config.objectives].zero_()
    return model.eval()


def active_goal_rates(labels,mask,weights):
    pairs=(labels[:,:,None,:]>labels[:,None,:,:]) & mask[:,:,None,None] & mask[:,None,:,None]
    active=pairs.any((1,2)).to(torch.float64)
    w=torch.as_tensor(weights,dtype=torch.float64)
    return (active*w[:,None]).sum(0)/w.sum()


def balanced_posterior(scores,labels,mask,row_weights,rates):
    losses=pairwise_losses(scores,utility(labels),mask)
    inverse=torch.where(rates>0,1/rates.clamp_min(1e-30),torch.zeros_like(rates)).to(scores.dtype)
    return (losses*inverse*row_weights[:,None]).mean()


def admissible(points,anchor,c):
    delta=np.asarray(points)-np.asarray(anchor)
    limits=np.array([c['evaluation']['maximum_positive_goal_absolute_point_drop']]*6+
        [c['evaluation']['maximum_hate_risk_absolute_point_increase']])
    return bool(np.isfinite(points).all() and (delta>=-limits).all())


def best_epoch(anchor,history,c):
    best=0;value=float(np.mean(anchor[:6]))
    for row in history:
        if admissible(row['objectives'],anchor,c) and row['validation']>value:
            best=row['epoch'];value=row['validation']
    return best


def registration(a,c):
    parent=read_json(a.run/'registration.json');selection=read_json(a.run/'selection.json')
    if parent['model_source_sha256']!=_source_identity()['source_sha256']:raise ValueError('Parent model code changed')
    for p,h in parent['files'].items():
        if sha256_file(p)!=h:raise ValueError('Parent source/data changed')
    replay=read_json(a.run/'replay.json')
    if not replay['exact_predictions_and_metrics']:raise ValueError('Parent replay missing')
    for name,key in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(a.run/name)!=replay[key]:raise ValueError('Parent result changed')
    paths=[Path(__file__),a.config,a.run/'registration.json',a.run/'selection.json',a.run/'evaluation.json',
        a.run/'replay.json',a.run/'predictions.npz',a.run/'preparation.json',a.run/'data_audit.json']
    for f in c['data']['folds']:
        folder=a.run/f'fold-{f}';manifest=read_json(folder/'data_manifest.json');audit=read_json(folder/'selection_audit.json')
        if audit['selection_sha256']!=sha256_file(folder/'selection.json'):raise ValueError('Parent selection audit differs')
        paths.extend([folder/'data_manifest.json',folder/'selection_audit.json'])
        for name,h in manifest['files'].items():
            p=folder/name
            if sha256_file(p)!=h:raise ValueError('Saved fold changed')
            paths.append(p)
        for seed in c['training']['seeds']:
            spec=selection['folds'][str(f)][str(seed)]['emer'];p=folder/spec['path']
            if sha256_file(p)!=spec['checkpoint_sha256']:raise ValueError('Anchor changed')
            paths.append(p)
    result=dict(config=c,config_hash=canonical_hash(c),inputs={str(p):sha256_file(p) for p in paths},
        model_source_sha256=parent['model_source_sha256'],parent_outer_results_already_opened=True,
        paid_gpu=False,new_api_requests=0)
    immutable_json(a.output/'registration.json',result)
    return selection,parent['config']


def fit(data,c,anchor_path,seed,use_jev,path):
    seed_everything(seed);anchor=load_model(anchor_path);model=warm_start(anchor,use_jev)
    for split in ('train','validation'):
        if not np.array_equal(predict(anchor,data[split],c['training']['batch_size']),predict(model,data[split],c['training']['batch_size'])):
            raise ValueError('Warm start differs from anchor')
    initial,anchor_points=validation(anchor,data['validation'],c)
    best=initial;epoch_best=0;points_best=anchor_points;state=copy.deepcopy(model.state_dict());history=[]
    rates=active_goal_rates(data['train'].arrays['labels'],data['train'].arrays['mask'],data['train'].sample_weights)
    previous=copy.deepcopy(model).eval()
    for p in previous.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay'])
    weights=data['train'].sample_weights/data['train'].sample_weights.mean()
    rows=[dict(data['train'][i],row_weight=torch.tensor(weights[i],dtype=torch.float32)) for i in range(len(weights))]
    batches=DataLoader(rows,batch_size=c['training']['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda rows:(collate_slates([{k:v for k,v in r.items() if k!='row_weight'} for r in rows]),torch.stack([r['row_weight'] for r in rows])))
    for epoch in range(1,c['training']['epochs']+1):
        snapshot=copy.deepcopy(model.state_dict());losses=[]
        for batch,w in batches:
            model.eval()
            with torch.no_grad():adv=advantage_weights(model(batch)['score_mean'],previous(batch)['score_mean'],utility(batch.behavior_scores),batch.mask,3)
            model.train();optimizer.zero_grad();scores=model(batch)['score_mean']
            posterior=balanced_posterior(scores,batch.labels,batch.mask,w,rates)
            prior=(pairwise_losses(scores,utility(batch.behavior_scores),batch.mask)*adv).mean(-1)
            loss=posterior+c['training']['prior_coefficient']*(prior*w).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip_norm']);optimizer.step();losses.append(float(loss.detach()))
        previous.load_state_dict(snapshot);point,goals=validation(model,data['validation'],c)
        allowed=admissible(goals,anchor_points,c)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=point,objectives=goals,admissible=allowed))
        if allowed and point>best:best=point;epoch_best=epoch;points_best=goals;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    _save_checkpoint(path,dict(model_config=asdict(model.config),state=state,seed=seed,best_epoch=epoch_best))
    row=dict(seed=seed,use_jev=use_jev,parameters=sum(p.numel() for p in model.parameters()),best_epoch=epoch_best,
        validation=best,validation_objectives=points_best,anchor_validation=initial,anchor_objectives=anchor_points,
        anchor_sha256=sha256_file(anchor_path),initial_predictions_exact=True,training_goal_rates=rates.tolist(),history=history,
        checkpoint_sha256=sha256_file(path))
    atomic_write_json(path.with_suffix('.json'),row);return row


def train(a,c,parent):
    chosen={};completed=0
    for f in c['data']['folds']:
        folder=a.output/f'fold-{f}';folder.mkdir(exist_ok=True)
        data={s:dataset(a.run/f'fold-{f}'/f'{s}.npz') for s in ('train','validation')};chosen[str(f)]={}
        for seed in c['training']['seeds']:
            anchor=a.run/f'fold-{f}'/parent['folds'][str(f)][str(seed)]['emer']['path'];chosen[str(f)][str(seed)]={}
            for arm,use_jev in [('emer',False),('emer_jev',True)]:
                path=folder/f'{arm}-seed{seed}.pt'
                if path.exists():
                    row=read_json(path.with_suffix('.json'))
                    if row['checkpoint_sha256']!=sha256_file(path) or len(row['history'])!=c['training']['epochs']:raise ValueError('Incomplete or changed fit')
                    if row['seed']!=seed or row['use_jev']!=use_jev or row['anchor_sha256']!=sha256_file(anchor):raise ValueError('Fit identity differs')
                else:row=fit(data,c,anchor,seed,use_jev,path)
                chosen[str(f)][str(seed)][arm]=dict(path=str(path.relative_to(a.output)),checkpoint_sha256=row['checkpoint_sha256'],best_epoch=row['best_epoch'])
                completed+=1;atomic_write_json(a.output/'progress.json',dict(completed_fits=completed,planned_fits=18,new_outer_metrics_computed=False))
                print(f'fit {completed}/18 fold={f} seed={seed} arm={arm} selected_epoch={row["best_epoch"]}',flush=True)
        del data
    if completed!=18:raise ValueError('Unequal training budget')
    immutable_json(a.output/'selection.json',dict(folds=chosen,completed_fits=completed,config_hash=canonical_hash(read_json(a.config))))


def audit(a,c,parent):
    selection=read_json(a.output/'selection.json');models=[]
    if selection['completed_fits']!=18 or selection['config_hash']!=canonical_hash(read_json(a.config)):raise ValueError('All18 registered fits required')
    for f in c['data']['folds']:
        data={s:dataset(a.run/f'fold-{f}'/f'{s}.npz') for s in ('train','validation')}
        rates=active_goal_rates(data['train'].arrays['labels'],data['train'].arrays['mask'],data['train'].sample_weights).tolist()
        for seed in c['training']['seeds']:
            anchor_path=a.run/f'fold-{f}'/parent['folds'][str(f)][str(seed)]['emer']['path'];anchor=load_model(anchor_path)
            initial,points=validation(anchor,data['validation'],c)
            for arm,use_jev in [('emer',False),('emer_jev',True)]:
                spec=selection['folds'][str(f)][str(seed)][arm];path=a.output/spec['path'];r=read_json(path.with_suffix('.json'))
                if r['checkpoint_sha256']!=spec['checkpoint_sha256'] or sha256_file(path)!=spec['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
                if r['anchor_sha256']!=sha256_file(anchor_path) or r['training_goal_rates']!=rates:raise ValueError('Training input mismatch')
                if r['seed']!=seed or r['use_jev']!=use_jev or r['parameters']!=43345:raise ValueError('Capacity or arm differs')
                if len(r['history'])!=c['training']['epochs'] or [h['epoch'] for h in r['history']]!=list(range(1,c['training']['epochs']+1)):raise ValueError('Budget differs')
                if r['anchor_validation']!=initial or r['anchor_objectives']!=points or not r['initial_predictions_exact']:raise ValueError('Initial reference differs')
                if any(h['admissible']!=admissible(h['objectives'],points,c) for h in r['history']):raise ValueError('Guard record differs')
                if any(not np.isclose(h['validation'],np.mean(h['objectives'][:6]),rtol=0,atol=1e-12) for h in r['history']):raise ValueError('History mean differs')
                if best_epoch(points,r['history'],c)!=r['best_epoch'] or spec['best_epoch']!=r['best_epoch']:raise ValueError('Guarded selection differs')
                expected=points if r['best_epoch']==0 else r['history'][r['best_epoch']-1]['objectives']
                if expected!=r['validation_objectives']:raise ValueError('Selected history metrics differ')
                model=load_model(path)
                if model.config.use_jev!=use_jev:raise ValueError('Checkpoint arm differs')
                value,goals=validation(model,data['validation'],c)
                if value!=r['validation'] or goals!=r['validation_objectives'] or not admissible(goals,points,c):raise ValueError('Selected validation replay failed')
                models.append(dict(path=spec['path'],checkpoint_sha256=spec['checkpoint_sha256'],best_epoch=r['best_epoch'],validation_replayed=True))
    atomic_write_json(a.output/'audit.json',dict(models=models,selection_sha256=sha256_file(a.output/'selection.json'),all18_verified=True))
    print('All18 trained checkpoints and guarded inner selections verified',flush=True)


def screen(pooled,folds,c):
    limits=c['evaluation']
    return dict(primary_conditional_ci_positive=pooled['positive_mean']['confidence_interval'][0]>0,
        primary_positive_each_seed=all(np.mean(r['emer_jev'][:6])>np.mean(r['emer'][:6]) for r in pooled['per_seed'].values()),
        primary_positive_each_fold=all(r['positive_mean']['absolute_gain']>0 for r in folds.values()),
        positive_point_guards=all(r['absolute_gain']>=-limits['maximum_positive_goal_absolute_point_drop'] for n,r in pooled['objectives'].items() if n!='is_hate'),
        hate_point_guard=pooled['objectives']['is_hate']['absolute_gain']>=-limits['maximum_hate_risk_absolute_point_increase'])


def evaluate(a,c):
    selected=read_json(a.output/'selection.json');checked=read_json(a.output/'audit.json')
    if not checked['all18_verified'] or checked['selection_sha256']!=sha256_file(a.output/'selection.json'):raise ValueError('Full selection audit required')
    n=read_json(a.run/'preparation.json')['slates'];seen=np.zeros(n,bool)
    preds={f'{arm}/{seed}':np.zeros((n,10),np.float32) for arm in ('emer','emer_jev','anchor') for seed in c['training']['seeds']}
    labels=np.zeros((n,10,7),np.float32);mask=np.zeros((n,10),bool);weights=np.zeros(n);users=np.zeros(n,np.int64)
    with np.load(a.run/'predictions.npz') as z:anchors={str(seed):z[f'emer/{seed}'] for seed in c['training']['seeds']}
    by_fold={name:{} for name in ('signal','original_reference','training_control')}
    comparisons={'signal':('emer','emer_jev'),'original_reference':('anchor','emer_jev'),'training_control':('anchor','emer')}
    def compare(values,data):
        reports={}
        for name,(base,full) in comparisons.items():
            p={f'{arm}/{seed}':values[f'{source}/{seed}'] for arm,source in [('emer',base),('emer_jev',full)] for seed in c['training']['seeds']}
            r=paired_report(p,data,c);r.pop('gates');r.pop('all_effect_gates');reports[name]=r
        return reports
    for f in c['data']['folds']:
        path=a.run/f'fold-{f}'/'test.npz';d=dataset(path)
        with np.load(path) as z:ids=z['slate_ids']
        if seen[ids].any():raise ValueError('Overlapping outer rows')
        local={}
        for seed in c['training']['seeds']:
            local[f'anchor/{seed}']=anchors[str(seed)][ids]
            for arm in ('emer','emer_jev'):
                s=selected['folds'][str(f)][str(seed)][arm];p=a.output/s['path']
                if sha256_file(p)!=s['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
                local[f'{arm}/{seed}']=predict(load_model(p),d,c['training']['batch_size'])
        for name,v in local.items():preds[name][ids]=v
        labels[ids]=d.arrays['labels'].numpy();mask[ids]=d.arrays['mask'].numpy();weights[ids]=d.sample_weights;users[ids]=d.users;seen[ids]=True
        for name,r in compare(local,d).items():by_fold[name][str(f)]=r
    if not seen.all():raise ValueError('Missing outer rows')
    d=SimpleNamespace(arrays={'labels':torch.from_numpy(labels),'mask':torch.from_numpy(mask)},users=users,sample_weights=weights)
    pooled=compare(preds,d);conditions={name:screen(pooled[name],by_fold[name],c) for name in ('signal','original_reference')}
    result=dict(role=c['role'],pooled=pooled,by_fold=by_fold,conditions=conditions,screen_passed=all(all(v.values()) for v in conditions.values()),
        slates=n,users=len(np.unique(users)),selection_sha256=sha256_file(a.output/'selection.json'),
        fallback_models=[m['path'] for m in checked['models'] if m['best_epoch']==0],limitations=c['limitations'],new_jev_calls=0,paid_gpu=False)
    if a.stage=='replay':
        with np.load(a.output/'predictions.npz') as old:
            if set(old.files)!=set(preds) or any(not np.array_equal(old[k],v) for k,v in preds.items()):raise ValueError('Prediction replay failed')
        if read_json(a.output/'evaluation.json')!=result:raise ValueError('Metric replay failed')
        atomic_write_json(a.output/'replay.json',dict(exact_predictions_and_metrics=True,evaluation_sha256=sha256_file(a.output/'evaluation.json'),predictions_sha256=sha256_file(a.output/'predictions.npz')))
    else:
        if (a.output/'evaluation.json').exists():raise ValueError('Use replay for existing result')
        np.savez_compressed(a.output/'predictions.npz',**preds);immutable_json(a.output/'evaluation.json',result)
    print('Exploratory balanced-training screen:',conditions,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_balanced_transfer_v1.json'));p.add_argument('--stage',choices=['prepare','train','audit','evaluate','replay'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=read_json(a.config)
    torch.set_num_threads(c['training']['cpu_threads']);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=c['training']['cpu_threads']):
        parent,original=registration(a,c)
        c=dict(c,objectives=original['objectives'],directions=original['directions'],evaluation=dict(c['evaluation'],click_maximum_absolute_ndcg_drop=.005,hate_maximum_absolute_risk_increase=.005))
        if a.stage=='prepare':print('Exploratory protocol and input/source hashes frozen;18 fits planned',flush=True)
        elif a.stage=='train':train(a,c,parent)
        elif a.stage=='audit':audit(a,c,parent)
        else:evaluate(a,c)
