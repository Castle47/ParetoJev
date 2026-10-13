#!/usr/bin/env python3
"""Expanded development only. No test reading, evaluation or test-based choices."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from paretojev.data.emer import fit_feature_spec,encode_features
from paretojev.experiments.comparison import _output_lock,_source_identity
from paretojev.training.behavior_calibration import apply_calibration
from paretojev.utils.io import atomic_write_json,read_json
from paretojev.utils.provenance import canonical_hash,sha256_file
from run_emer_jev import dataset,fit


def immutable_json(path,value):
    if path.exists():
        if read_json(path)!=value:raise ValueError(f'Frozen manifest changed: {path}')
    else:atomic_write_json(path,value)


def materialize(root,data,out,c):
    prep=read_json(data/'preparation.json')
    if prep['config_hash']!=canonical_hash(c):raise ValueError('Preparation protocol changed')
    raw_path=data/'raw_slates.npz';raw_hash=sha256_file(raw_path)
    if raw_hash!=prep['features']['raw_slates_sha256']:raise ValueError('Cohort changed')
    tree_path=root/'behavior-v2/run-v2/upload_metadata_assumed_static.joblib'
    tree_hash=sha256_file(tree_path)
    if tree_hash!=read_json(tree_path.parent/'results.json')['models']['upload_metadata_assumed_static']['model_sha256']:
        raise ValueError('Behavior checkpoint changed')
    identity=dict(config_hash=canonical_hash(c),raw_sha256=raw_hash,behavior_sha256=tree_hash,
        source_sha256=_source_identity()['source_sha256'],script_sha256=sha256_file(Path(__file__)),
        test_accessed=False,no_jev_structural_zero_block='excluded by use_jev=False, not synthetic model scores')
    manifest_path=out/'common_manifest.json'
    if manifest_path.exists():
        old=read_json(manifest_path)
        if old['identity']!=identity:raise ValueError('Materialization identity changed')
        for name,digest in old['files'].items():
            if sha256_file(out/name)!=digest:raise ValueError('Common tensors changed')
        return
    raw=dict(np.load(raw_path));train=raw['split']==0
    if c['split_names']!=['train','validation']:raise ValueError('No test may enter development')
    spec=fit_feature_spec(raw['x'][train],raw['weight'][train])
    features=encode_features(raw['x'],spec)
    tree=joblib.load(tree_path);flat=raw['x'].reshape(-1,26)
    behavior=apply_calibration(np.column_stack([m.predict_proba(flat)[:,1] for m in tree['models']]),
        tree['calibration']).reshape(-1,c['slate_size'],7).astype('float32')
    atomic_write_json(out/'feature_spec.json',spec);files={}
    for i,name in enumerate(c['split_names']):
        use=raw['split']==i;b=behavior[use]
        if use.sum()!=sum(c['sample_budgets'][name]):raise ValueError('Requested sample count not reached')
        arrays=dict(candidate_features=features[use],behavior_scores=b,jev_scores=np.zeros_like(b),
            jev_entropy=np.zeros_like(b),jev_confidence=np.zeros_like(b),labels=raw['labels'][use].astype('float32'),
            mask=np.ones(raw['items'][use].shape,dtype=bool),popularity=raw['popularity'][use,:,None].astype('float32'),
            history_length=raw['history_count'][use,None].astype('float32'),slate_weights=raw['weight'][use],
            user_ids=raw['user'][use],candidate_ids=raw['items'][use],slate_ids=np.flatnonzero(use))
        p=out/f'{name}.npz';np.savez_compressed(p,**arrays);files[p.name]=sha256_file(p)
    files['feature_spec.json']=sha256_file(out/'feature_spec.json')
    atomic_write_json(manifest_path,dict(identity=identity,files=files))
    print('Expanded common features and frozen behavior scores ready; Jev absent from baseline',flush=True)


def development_data(data,out,c,arm):
    result={name:dataset(out/f'{name}.npz') for name in c['split_names']}
    score_hash=None
    if arm=='emer_jev':
        folder=data/'jev';complete=read_json(folder/'complete.json')
        if not complete['complete'] or complete['registration']['config_hash']!=canonical_hash(c):
            raise ValueError('Complete real Jev scores required')
        score_hash=sha256_file(folder/'scores.npz')
        if score_hash!=complete['scores_sha256']:raise ValueError('Jev scores changed')
        with np.load(folder/'scores.npz') as z:
            if str(z['model'])!=c['jev']['model'] or str(z['revision'])!=c['jev']['revision'] or z['objective_names'].tolist()!=c['objectives']:
                raise ValueError('Jev model, revision or goals changed')
            if not np.array_equal(z['slate_ids'],np.arange(len(z['candidate_ids']))):raise ValueError('Score row identities changed')
            for name,d in result.items():
                with np.load(out/f'{name}.npz') as common:
                    ids=common['slate_ids']
                    if not np.array_equal(common['candidate_ids'],z['candidate_ids'][ids]):raise ValueError('Candidate alignment changed')
                for key,source in [('jev_scores','probabilities'),('jev_entropy','entropy'),('jev_confidence','confidence')]:
                    values=z[source][ids].astype('float32')
                    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('Invalid real scores')
                    d.arrays[key]=torch.from_numpy(values)
                if not np.allclose(d.arrays['jev_scores'].sum(1).numpy(),1,atol=.001):raise ValueError('Invalid Choice normalization')
    return result,score_hash


def develop(a,c):
    materialize(a.root,a.data,a.output,c)
    data,score_hash=development_data(a.data,a.output,c,a.arm)
    model_config=read_json('configs/emer_jev.json')['model']
    train_config=dict(c,model=model_config)
    identity=dict(config_hash=canonical_hash(c),model=model_config,
        common_manifest_sha256=sha256_file(a.output/'common_manifest.json'),real_jev_scores_sha256=score_hash,
        source_sha256=_source_identity()['source_sha256'],fit_script_sha256=sha256_file(Path('scripts/run_emer_jev.py')),
        driver_sha256=sha256_file(Path(__file__)),arm=a.arm,selection='validation only; same fixed grid and training seeds',test_opened=False)
    immutable_json(a.output/f'{a.arm}_registration.json',identity)
    selected={};trials=[]
    for seed in c['training']['seeds']:
        options=[]
        for trial,(lr,coef) in enumerate((lr,coef) for lr in c['training']['learning_rates'] for coef in c['training']['prior_coefficients']):
            p=a.output/f'{a.arm}-seed{seed}-trial{trial}.pt'
            if p.exists() and p.with_suffix('.json').exists():
                r=read_json(p.with_suffix('.json'))
                if sha256_file(p)!=r['checkpoint_sha256']:raise ValueError('Saved trial changed')
            else:
                _,r=fit(data,train_config,seed,a.arm=='emer_jev',lr,coef,p)
            r=dict(r,path=p.name);trials.append(r);options.append(r)
            print(f'{a.arm} seed={seed} trial={trial} epoch={r["best_epoch"]} validation={r["validation"]:.6f}',flush=True)
            atomic_write_json(a.output/f'{a.arm}_progress.json',dict(completed_trials=len(trials),planned_trials=12,test_opened=False))
        best=max(options,key=lambda r:r['validation'])
        selected[str(seed)]={key:best[key] for key in ['path','checkpoint_sha256','validation','best_epoch','lr','prior_coefficient','parameters']}
    immutable_json(a.output/f'{a.arm}_selection.json',dict(registration=identity,selected=selected,trials=trials,test_opened=False))
    print(a.arm,'development complete, independent test still unopened',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev_scale.json'))
    p.add_argument('--arm',choices=['emer','emer_jev'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=read_json(a.config)
    torch.set_num_threads(c['training']['cpu_threads']);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=c['training']['cpu_threads']):develop(a,c)
