#!/usr/bin/env python3
"""Fixed half-original/half-balanced posterior; preserve both earlier studies.

The training loop is retained from the immutable balanced-transfer driver so
that its recorded source hashes and exact replay remain valid. Shared model,
initialization, guard, audit and evaluation helpers are reused unchanged.
"""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from paretojev.data.synthetic import collate_slates
from paretojev.models.emer import pairwise_losses,utility,advantage_weights
from paretojev.experiments.comparison import _save_checkpoint,_source_identity,_output_lock
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from paretojev.utils.seed import seed_everything
from run_emer_jev import dataset,load_model,predict,validation
from run_emer_scale import immutable_json
from run_emer_balanced_transfer import warm_start,active_goal_rates,balanced_posterior,admissible,audit,evaluate


def blended_posterior(scores,labels,mask,row_weights,rates,mix):
    if not 0<=mix<=1:raise ValueError('Invalid posterior mixture')
    satisfaction=labels[...,:-1].sum(-1)-labels[...,-1]
    original=(pairwise_losses(scores,satisfaction[...,None],mask).squeeze(-1)*row_weights).mean()
    balanced=balanced_posterior(scores,labels,mask,row_weights,rates)
    return (1-mix)*original+mix*balanced


def registration(a,c):
    previous=read_json(a.balanced_run/'registration.json')
    if previous['model_source_sha256']!=_source_identity()['source_sha256']:raise ValueError('Frozen model code changed')
    for p,h in previous['inputs'].items():
        if sha256_file(p)!=h:raise ValueError('Previous experiment input/source changed')
    replay=read_json(a.balanced_run/'replay.json')
    if not replay['exact_predictions_and_metrics']:raise ValueError('Completed balanced replay required')
    for n,k in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(a.balanced_run/n)!=replay[k]:raise ValueError('Balanced evidence changed')
    if str(a.run/'selection.json') not in previous['inputs']:raise ValueError('Wrong original CV run')
    paths=[Path(__file__),a.config,Path('scripts/run_emer_balanced_transfer.py'),
        Path('reports/emer-balanced-fallback-diagnosis/diagnosis.json')]+[a.balanced_run/n for n in
        ('registration.json','selection.json','audit.json','evaluation.json','replay.json','predictions.npz')]
    identity=dict(config=c,config_hash=canonical_hash(c),inputs=dict(previous['inputs'],**{str(p):sha256_file(p) for p in paths}),
        model_source_sha256=previous['model_source_sha256'],previous_outer_results_already_opened=True,new_api_requests=0,paid_gpu=False)
    immutable_json(a.output/'registration.json',identity)
    return read_json(a.run/'selection.json'),read_json(a.run/'registration.json')['config']


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
            posterior=blended_posterior(scores,batch.labels,batch.mask,w,rates,c['training']['posterior_mix'])
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
    row=dict(posterior_mix=c['training']['posterior_mix'],seed=seed,use_jev=use_jev,parameters=sum(p.numel() for p in model.parameters()),best_epoch=epoch_best,
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
                    if row['posterior_mix']!=c['training']['posterior_mix']:raise ValueError('Posterior mix differs')
                    if row['seed']!=seed or row['use_jev']!=use_jev or row['anchor_sha256']!=sha256_file(anchor):raise ValueError('Fit identity differs')
                else:row=fit(data,c,anchor,seed,use_jev,path)
                chosen[str(f)][str(seed)][arm]=dict(path=str(path.relative_to(a.output)),checkpoint_sha256=row['checkpoint_sha256'],best_epoch=row['best_epoch'])
                completed+=1;atomic_write_json(a.output/'progress.json',dict(completed_fits=completed,planned_fits=18,new_outer_metrics_computed=False))
                print(f'fit {completed}/18 fold={f} seed={seed} arm={arm} selected_epoch={row["best_epoch"]}',flush=True)
        del data
    if completed!=18:raise ValueError('Unequal training budget')
    immutable_json(a.output/'selection.json',dict(folds=chosen,completed_fits=completed,config_hash=canonical_hash(read_json(a.config))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--balanced-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/emer_blended_transfer_v1.json'))
    p.add_argument('--stage',choices=['prepare','train','audit','evaluate','replay'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=read_json(a.config)
    torch.set_num_threads(c['training']['cpu_threads']);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=c['training']['cpu_threads']):
        parent,original=registration(a,c)
        c=dict(c,objectives=original['objectives'],directions=original['directions'],evaluation=dict(c['evaluation'],click_maximum_absolute_ndcg_drop=.005,hate_maximum_absolute_risk_increase=.005))
        if a.stage=='prepare':print('Fixed50/50 protocol frozen;18 matched fits planned',flush=True)
        elif a.stage=='train':train(a,c,parent)
        elif a.stage=='audit':
            for f in c['data']['folds']:
                for seed in c['training']['seeds']:
                    for arm in ('emer','emer_jev'):
                        r=read_json(a.output/f'fold-{f}'/f'{arm}-seed{seed}.json')
                        if r['posterior_mix']!=c['training']['posterior_mix']:raise ValueError('Posterior mix differs')
            audit(a,c,parent)
        else:evaluate(a,c)
