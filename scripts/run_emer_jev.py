#!/usr/bin/env python3
"""Equal-budget EMER / EMER+Jev development, locked confirmation and replay."""
import argparse
import copy
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from paretojev.data.emer import encode_features, fit_feature_spec
from paretojev.data.precomputed import PrecomputedSlateDataset
from paretojev.data.synthetic import collate_slates
from paretojev.evaluation.comparison import _contributions, comparison_objective_metrics
from paretojev.experiments.comparison import _output_lock, _save_checkpoint, _source_identity
from paretojev.models.emer import EMERConfig, EMERReranker, emer_loss, advantage_weights, utility
from paretojev.training.behavior_calibration import apply_calibration
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash, sha256_file, runtime_provenance
from paretojev.utils.seed import seed_everything


def development_data(root, out):
    core = root/'core-choice-v1'; raw = np.load(core/'raw_slates.npz')
    train_mask = raw['split'] == 0
    expected_spec = fit_feature_spec(raw['x'][train_mask], raw['weight'][train_mask])
    if (out/'feature_spec.json').exists() and read_json(out/'feature_spec.json') != expected_spec:
        raise ValueError('Feature specification changed')
    atomic_write_json(out/'feature_spec.json', expected_spec)
    spec = expected_spec; result = {}
    for i, name in enumerate(('train','validation')):
        with np.load(core/f'full/{name}.npz') as saved: arrays = {key:saved[key].copy() for key in saved.files}
        arrays['candidate_features'] = encode_features(raw['x'][raw['split']==i],spec)
        path = out/f'dev-{name}.npz'
        if not path.exists(): np.savez_compressed(path,**arrays)
        else:
            with np.load(path) as previous:
                if set(previous.files)!=set(arrays) or any(not np.array_equal(previous[k],v) for k,v in arrays.items()):
                    raise ValueError('Development tensors changed')
        result[name] = dataset(path)
    return result


def dataset(path):
    data = PrecomputedSlateDataset(path)
    with np.load(path) as z:
        data.sample_weights = z['slate_weights'].astype('float64')
        data.users = z['user_ids'].copy()
    return data


def loader(data, batch, seed=None):
    return DataLoader(data,batch_size=batch,shuffle=seed is not None,collate_fn=collate_slates,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None)


@torch.no_grad()
def predict(model, data, batch):
    model.eval()
    return np.concatenate([model(b)['score_mean'].numpy() for b in loader(data,batch)])


def validation(model, data, c):
    scores = predict(model,data,c['training']['batch_size'])
    points,_ = comparison_objective_metrics(scores,data.arrays['labels'].numpy(),data.arrays['mask'].numpy(),
        c['directions'],top_k=c['evaluation']['top_k'],sample_weights=data.sample_weights)
    if not np.isfinite(points).all(): raise ValueError('Missing validation goal coverage')
    return float(points[:6].mean()),points.tolist()


def fit(data,c,seed,use_jev,lr,prior_coeff,path):
    seed_everything(seed); model = EMERReranker(EMERConfig(**c['model'],use_jev=use_jev))
    previous = copy.deepcopy(model).eval()
    for param in previous.parameters():param.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=c['training']['weight_decay'])
    score,points = validation(model,data['validation'],c)
    best,best_epoch,best_points = score,0,points
    state = copy.deepcopy(model.state_dict()); history=[]; start=time.perf_counter()
    train = data['train']; normalized_weights = train.sample_weights/train.sample_weights.mean()
    # Loader includes row weights explicitly; all other data use the established
    # immutable SlateBatch schema, whose labels are inaccessible to inference.
    train_rows = [dict(train[i],row_weight=torch.tensor(normalized_weights[i],dtype=torch.float32)) for i in range(len(train))]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_rows,batch_size=c['training']['batch_size'],shuffle=True,generator=generator,
        collate_fn=lambda rows:(collate_slates([{k:v for k,v in row.items() if k!='row_weight'} for row in rows]),
            torch.stack([row['row_weight'] for row in rows])))
    for epoch in range(1,c['training']['epochs']+1):
        current_snapshot = copy.deepcopy(model.state_dict()); losses=[]
        for batch,weights in train_loader:
            model.eval()
            with torch.no_grad():
                w = advantage_weights(model(batch)['score_mean'],previous(batch)['score_mean'],
                    utility(batch.behavior_scores),batch.mask,c['evaluation']['top_k'])
            model.train(); optimizer.zero_grad()
            # Weight complete per-slate posterior and prior losses, not arbitrary
            # minibatch means. A constant normalization preserves the population.
            scores=model(batch)['score_mean']
            loss=emer_loss(scores,batch,w,prior_coeff,row_weights=weights)['loss']
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        previous.load_state_dict(current_snapshot)
        score,points=validation(model,data['validation'],c)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=score,objectives=points))
        if score>best:
            best,best_epoch,best_points=score,epoch,points;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    payload=dict(model_config=asdict(model.config),state=state,seed=seed,lr=lr,prior_coefficient=prior_coeff,best_epoch=best_epoch)
    _save_checkpoint(path,payload)
    receipt=dict(seed=seed,use_jev=use_jev,lr=lr,prior_coefficient=prior_coeff,best_epoch=best_epoch,
        validation=best,validation_objectives=best_points,parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.perf_counter()-start,history=history,checkpoint_sha256=sha256_file(path))
    atomic_write_json(path.with_suffix('.json'),receipt)
    return model,receipt


def load_model(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    model=EMERReranker(EMERConfig(**payload['model_config']));model.load_state_dict(payload['state']);model.eval()
    return model


def develop(a,c):
    data=development_data(a.root,a.output)
    registration=dict(config=c,inputs={name:sha256_file(a.output/f'dev-{name}.npz') for name in data},
        feature_spec_sha256=sha256_file(a.output/'feature_spec.json'),source=_source_identity(),
        script_sha256=sha256_file(Path(__file__)),old_test_used=False)
    reg_path=a.output/'development_registration.json'
    if reg_path.exists() and read_json(reg_path)!=registration:raise ValueError('Development registration changed')
    atomic_write_json(reg_path,registration)
    selected={}; trials=[]
    for seed in c['training']['seeds']:
        selected[str(seed)]={}
        for arm,use_jev in (('emer',False),('emer_jev',True)):
            options=[]
            for trial,(lr,coef) in enumerate((lr,coef) for lr in c['training']['learning_rates'] for coef in c['training']['prior_coefficients']):
                path=a.output/f'{arm}-seed{seed}-trial{trial}.pt'
                if path.exists():
                    r=read_json(path.with_suffix('.json'))
                    if sha256_file(path)!=r['checkpoint_sha256']:raise ValueError('Trial changed')
                else:
                    _,r=fit(data,c,seed,use_jev,lr,coef,path)
                r=dict(r,path=path.name);options.append(r);trials.append(r)
                print(f'{arm} seed={seed} trial={trial} best_epoch={r["best_epoch"]} validation={r["validation"]:.6f}',flush=True)
            best=max(options,key=lambda x:x['validation'])
            selected[str(seed)][arm]={key:best[key] for key in ('path','checkpoint_sha256','validation','best_epoch','lr','prior_coefficient','parameters')}
    frozen=dict(config_hash=canonical_hash(c),development_registration_sha256=sha256_file(reg_path),
        selected=selected,trials=trials,selection='validation only; complete equal-budget search',test_opened=False)
    path=a.output/'selection.json'
    if path.exists() and read_json(path)!=frozen:raise ValueError('Frozen selection changed')
    atomic_write_json(path,frozen);print('Both arms and all seeds frozen before fresh-test evaluation',flush=True)


def materialize_test(a,c):
    raw=np.load(a.output/'raw_slates.npz');complete=read_json(a.output/'jev/complete.json')
    if not complete['complete'] or complete['scores_sha256']!=sha256_file(a.output/'jev/scores.npz'):
        raise ValueError('Missing real scores')
    with np.load(a.output/'jev/scores.npz') as scores:
        if not np.array_equal(scores['candidate_ids'],raw['items']) or scores['objective_names'].tolist()!=c['objectives']:
            raise ValueError('Jev identities changed')
        if str(scores['model'])!=c['jev']['model'] or str(scores['revision'])!=c['jev']['revision']:
            raise ValueError('Jev model changed')
        jev={key:scores[key].astype('float32') for key in ('probabilities','confidence','entropy')}
    tree_path=a.root/'behavior-v2/run-v2/upload_metadata_assumed_static.joblib'
    tree_result=read_json(tree_path.parent/'results.json')['models']['upload_metadata_assumed_static']
    if sha256_file(tree_path)!=tree_result['model_sha256']:raise ValueError('Behavior model changed')
    tree=joblib.load(tree_path);flat=raw['x'].reshape(-1,26)
    behavior=apply_calibration(np.column_stack([m.predict_proba(flat)[:,1] for m in tree['models']]),tree['calibration']).reshape(-1,10,7)
    features=encode_features(raw['x'],read_json(a.output/'feature_spec.json'))
    arrays=dict(candidate_features=features,behavior_scores=behavior.astype('float32'),jev_scores=jev['probabilities'],
        jev_confidence=jev['confidence'],jev_entropy=jev['entropy'],labels=raw['labels'].astype('float32'),
        mask=np.ones(raw['items'].shape,dtype=bool),popularity=raw['popularity'][...,None].astype('float32'),
        history_length=raw['history_count'][:,None].astype('float32'),slate_weights=raw['weight'],user_ids=raw['user'])
    path=a.output/'test.npz'
    if path.exists():
        with np.load(path) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):raise ValueError('Test tensors changed')
    else:np.savez_compressed(path,**arrays)
    return dataset(path)


def paired_report(scores,data,c):
    mask=data.arrays['mask'].numpy();labels=data.arrays['labels'].numpy();groups,codes=np.unique(data.users,return_inverse=True)
    stats={arm:[] for arm in ('emer','emer_jev')}
    for arm in stats:
        for seed in c['training']['seeds']:
            num,den,*_= _contributions(scores[f'{arm}/{seed}'],labels,mask,np.array(c['directions']),c['evaluation']['top_k'],np.zeros_like(mask))
            num*=data.sample_weights[:,None];den*=data.sample_weights[:,None]
            gn=np.zeros((len(groups),7));gd=np.zeros_like(gn)
            np.add.at(gn,codes,num);np.add.at(gd,codes,den);stats[arm].append((gn,gd))
    def metrics(arm,counts):
        points=[]
        for gn,gd in stats[arm]:
            den=counts@gd;num=counts@gn
            points.append(np.divide(num,den,out=np.full(7,np.nan),where=den>0))
        return np.mean(points,0)
    ones=np.ones(len(groups));base=metrics('emer',ones);full=metrics('emer_jev',ones)
    rng=np.random.default_rng(c['evaluation']['bootstrap_seed']);delta=[]
    for _ in range(c['evaluation']['bootstrap_samples']):
        counts=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))
        delta.append(metrics('emer_jev',counts)-metrics('emer',counts))
    delta=np.array(delta);per_goal={}
    for i,name in enumerate(c['objectives']):
        valid=delta[:,i][np.isfinite(delta[:,i])]
        per_goal[name]=dict(emer=float(base[i]),emer_jev=float(full[i]),absolute_gain=float(full[i]-base[i]),
            relative_gain=float((full[i]-base[i])/base[i]) if base[i] else None,
            confidence_interval=np.quantile(valid,[.025,.975]).tolist(),valid_bootstrap_replicates=len(valid))
    mean_delta=np.mean(delta[:,:6],1);ci=np.quantile(mean_delta[np.isfinite(mean_delta)],[.025,.975]).tolist()
    per_seed={str(seed):{arm:comparison_objective_metrics(scores[f'{arm}/{seed}'],labels,mask,c['directions'],
        top_k=c['evaluation']['top_k'],sample_weights=data.sample_weights)[0].tolist() for arm in stats} for seed in c['training']['seeds']}
    gates=dict(primary_ci_positive=per_goal['long_view']['confidence_interval'][0]>0,
        primary_positive_every_seed=all(row['emer_jev'][1]>row['emer'][1] for row in per_seed.values()),
        positive_mean_noninferior=ci[0]>=0,
        click_noninferior=per_goal['is_click']['confidence_interval'][0]>=-c['evaluation']['click_maximum_absolute_ndcg_drop'],
        hate_noninferior=per_goal['is_hate']['confidence_interval'][0]>=-c['evaluation']['hate_maximum_absolute_risk_increase'])
    return dict(objectives=per_goal,positive_mean=dict(emer=float(base[:6].mean()),emer_jev=float(full[:6].mean()),
        absolute_gain=float((full-base)[:6].mean()),confidence_interval=ci),per_seed=per_seed,gates=gates,
        all_effect_gates=all(gates.values()),bootstrap_users=len(groups),
        note='Paired user bootstrap conditional on fitted seeds; seed average is not a training-population CI. Hate is no-hate utility.',
        online_business_success=False)


def latency(models,data,c):
    batch=next(iter(loader(data,1)));model_times={arm:[] for arm in models};protocol=c['latency']
    with torch.no_grad():
        for arm in models:
            for _ in range(protocol['warmup']):models[arm](batch)['score_mean'].argsort(dim=1,descending=True)
        for i in range(protocol['iterations']):
            for arm in list(models)[::1 if i%2 else -1]:
                start=time.perf_counter();models[arm](batch)['score_mean'].argsort(dim=1,descending=True)
                model_times[arm].append((time.perf_counter()-start)*1000)
    return dict(protocol=protocol,milliseconds={arm:{key:float(np.quantile(v,q)) for key,q in
        (('p50',.5),('p95',.95),('p99',.99))} for arm,v in model_times.items()},
        cached_signal_only=True,complete_recommendation_rt_measured=False)


def confirm(a,c):
    selection=read_json(a.output/'selection.json')
    if selection['config_hash']!=canonical_hash(c):raise ValueError('Frozen design changed')
    if read_json(a.output/'development_registration.json')['source']['source_sha256']!=_source_identity()['source_sha256']:
        raise ValueError('Package changed after fitting')
    test=materialize_test(a,c);scores={};timing_models={}
    for seed,arms in selection['selected'].items():
        for arm,spec in arms.items():
            path=a.output/spec['path']
            if sha256_file(path)!=spec['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
            model=load_model(path);scores[f'{arm}/{seed}']=predict(model,test,c['training']['batch_size'])
            if int(seed)==c['training']['seeds'][0]:timing_models[arm]=model
    pred=a.output/'predictions.npz'
    if a.stage=='replay':
        with np.load(pred) as old:
            if set(old.files)!=set(scores) or any(not np.array_equal(old[k],v) for k,v in scores.items()):raise ValueError('Prediction replay differs')
    else:
        if pred.exists():raise ValueError('Frozen confirmation exists; use replay')
        np.savez_compressed(pred,**scores)
    report=paired_report(scores,test,c)
    if a.stage=='replay':
        if read_json(a.output/'evaluation.json')['results']!=report:raise ValueError('Metric or interval replay differs')
        atomic_write_json(a.output/'replay.json',dict(exact_predictions_and_metrics=True,provenance=runtime_provenance()))
    else:
        atomic_write_json(a.output/'evaluation.json',dict(results=report,latency=latency(timing_models,test,c),
            config=c,predictions_sha256=sha256_file(pred),test_sha256=sha256_file(a.output/'test.npz'),
            selection_sha256=sha256_file(a.output/'selection.json'),provenance=runtime_provenance(),
            implementation='EMER paper-aligned public-data adaptation, not industrial reproduction'))
    print('Confirmation complete:',report['gates'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev.json'));p.add_argument('--stage',choices=['develop','confirm','replay'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=read_json(a.config)
    torch.set_num_threads(c['training']['cpu_threads']);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output/'training'),threadpool_limits(limits=c['training']['cpu_threads']):
        develop(a,c) if a.stage=='develop' else confirm(a,c)
