#!/usr/bin/env python3
"""Fixed-budget tabular controls and time-separated calibration; no Jev requests."""
import argparse
import fcntl
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from paretojev.data.kuairand_27k import BINARY_LABELS
from paretojev.data.kuairand_features_v2 import CORE_FIELDS, EXTRA_FIELDS
from paretojev.evaluation.behavior_metrics import metrics
from paretojev.training.behavior_calibration import fit_calibration, apply_calibration, cluster_logloss_difference
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash, sha256_file, runtime_provenance


def load_split(root,manifest,split):
    spec=manifest['splits'][split];result={}
    for key,digest in spec['files'].items():
        path=root/f'{split}-{key}.npy'
        if sha256_file(path)!=digest: raise ValueError(f'Changed data: {path}')
        result[key]=np.load(path,mmap_mode='r')[:spec['rows']]
    if result['time'].min()!=spec['min_time'] or result['time'].max()!=spec['max_time']:
        raise ValueError('Recorded period mismatch')
    return result


def probabilities(models,x,columns):
    return np.column_stack([model.predict_proba(x[:,:columns])[:,1] for model in models])


def run(a):
    if (a.output/'results.json').exists() or list(a.output.glob('*.joblib')): raise ValueError('Existing run: choose a fresh directory')
    config=read_json(a.config);manifest=read_json(a.data/'samples.json')
    if config['research_target_mode'] or config['jev_requests'] or tuple(config['objectives'])!=BINARY_LABELS:
        raise ValueError('Unsupported protocol')
    if canonical_hash(config)!=manifest['config_hash']: raise ValueError('Config mismatch')
    if config['regimes']!=['exposure_context','upload_metadata_assumed_static']:
        raise ValueError('Unexpected feature regimes')
    if manifest['core_fields']!=CORE_FIELDS or manifest['extra_fields']!=EXTRA_FIELDS:
        raise ValueError('Feature schema mismatch')
    bounds=[manifest['splits'][s] for s in ('train','calibration','evaluation')]
    if not bounds[0]['max_time'] < config['train_end_ms'] <= bounds[1]['min_time'] <= bounds[1]['max_time'] < config['calibration_end_ms'] <= bounds[2]['min_time'] <= bounds[2]['max_time'] < config['evaluation_end_ms']:
        raise ValueError('Period leakage')
    train=load_split(a.data,manifest,'train');cal=load_split(a.data,manifest,'calibration')
    weights=np.asarray(train['weight'],dtype='float64');weights/=weights.mean()
    prior=np.array([np.average(train['y'][:,j],weights=weights) for j in range(7)]).clip(1e-7,1-1e-7)
    raw_weights=np.asarray(train['weight'],dtype='float64');tab_prior=np.tile(prior,(15,1))
    for tab in range(15):
        mask=train['x'][:,0]==tab;ww=raw_weights[mask]
        tab_prior[tab]=((train['y'][mask]*ww[:,None]).sum(axis=0)+100*prior)/(ww.sum()+100)
    frozen={};start=time.monotonic()
    result=dict(scope=config['scope'],config=config,samples=manifest,models={},
        feature_fields=CORE_FIELDS+EXTRA_FIELDS,
        provenance=runtime_provenance(input_paths=[a.config,a.data/'samples.json',Path(__file__),
          Path('src/paretojev/data/kuairand_features_v2.py'),Path('src/paretojev/training/behavior_calibration.py'),
          Path('src/paretojev/evaluation/behavior_metrics.py')]))
    atomic_write_json(a.output/'running.json',result)
    for regime in config['regimes']:
        columns=len(CORE_FIELDS) if regime=='exposure_context' else len(CORE_FIELDS)+len(EXTRA_FIELDS)
        models=[]
        for j,name in enumerate(BINARY_LABELS):
            spec=config['estimator'];task_start=time.monotonic()
            model=HistGradientBoostingClassifier(max_iter=spec['max_iter'],learning_rate=spec['learning_rate'],
                max_leaf_nodes=spec['max_leaf_nodes'],max_depth=spec['max_depth'],
                min_samples_leaf=spec['min_samples_leaf'],max_bins=spec['max_bins'],
                l2_regularization=spec['l2_prevalence_factor']*prior[j]*(1-prior[j]),
                categorical_features=[True]+[False]*(columns-1),early_stopping=False,
                random_state=config['seed']+j)
            print(f'fit {regime} {name}',flush=True)
            model.fit(train['x'][:,:columns],train['y'][:,j],sample_weight=weights)
            models.append(model)
            print(f'fitted {regime} {name} seconds={time.monotonic()-task_start:.1f}',flush=True)
        cal_pred=probabilities(models,cal['x'],columns)
        parameters=fit_calibration(cal['y'],cal_pred,cal['weight'],config['calibration'])
        payload=dict(models=models,calibration=parameters,columns=columns,config_hash=canonical_hash(config),
            sample_manifest_sha256=sha256_file(a.data/'samples.json'))
        path=a.output/f'{regime}.joblib';joblib.dump(payload,path)
        frozen[regime]=dict(model_sha256=sha256_file(path),calibration=parameters,columns=columns)
        atomic_write_json(a.output/'frozen_models.json',frozen)
        print(f'frozen {regime}',flush=True)
    # Evaluation labels are loaded only after all models and calibrators are frozen.
    del train,cal,weights,raw_weights,models,model
    evaluation=load_split(a.data,manifest,'evaluation')
    y=evaluation['y'];w=evaluation['weight']
    result['baselines']={'global':metrics(y,np.broadcast_to(prior,y.shape),w),
        'tab':metrics(y,tab_prior[evaluation['x'][:,0].astype('int64')],w)}
    calibrated={}
    for regime in config['regimes']:
        path=a.output/f'{regime}.joblib'
        if sha256_file(path)!=frozen[regime]['model_sha256']: raise ValueError('Frozen model changed')
        payload=joblib.load(path)
        raw=probabilities(payload['models'],evaluation['x'],payload['columns'])
        p=apply_calibration(raw,payload['calibration']);calibrated[regime]=p
        np.save(a.output/f'{regime}-raw.npy',raw);np.save(a.output/f'{regime}-calibrated.npy',p)
        score=metrics(y,p,w);gates={}
        for name,v in score.items():
            baseline=result['baselines']['tab'][name]
            gates[name]=v['sampled_positives']>=20 and v['auc'] is not None and v['auc']>.5 and v['logloss']<baseline['logloss'] and v['brier']<baseline['brier']
        result['models'][regime]=dict(raw=metrics(y,raw,w),calibrated=score,gates=gates,
            all_gates_pass=all(gates.values()),**frozen[regime],
            raw_sha256=sha256_file(a.output/f'{regime}-raw.npy'),
            calibrated_sha256=sha256_file(a.output/f'{regime}-calibrated.npy'))
        print(regime,'gates',sum(gates.values()),'/7',flush=True)
    result['metadata_minus_context_logloss_bootstrap']=cluster_logloss_difference(y,
        calibrated['upload_metadata_assumed_static'],calibrated['exposure_context'],w,evaluation['user'],
        replicates=config['bootstrap']['replicates'],seed=config['seed'])
    result['elapsed_seconds']=time.monotonic()-start
    result['research_targets_achieved']=False
    atomic_write_json(a.output/'results.json',result)
    print('Experiment complete',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/kuairand_27k_behavior_v2.json'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.lock').open('a') as lock, threadpool_limits(limits=read_json(a.config)['threads']):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)


if __name__=='__main__':main()
