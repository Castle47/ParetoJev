#!/usr/bin/env python3
"""Freeze common core tensors; missing Jev is legal only for explicit no-Jev arms."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from paretojev.models.kuairand_aligned import fit_normalization
from paretojev.training.behavior_calibration import apply_calibration
from paretojev.data.comparison import ComparisonSlateDataset
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file,runtime_provenance


def materialize(a):
    if (a.output/'manifest.json').exists():raise ValueError('Completed tensors are immutable')
    c=read_json(a.config);selection=read_json(a.data/'selection.json');f=read_json(a.data/'features.json')
    if selection['config_hash']!=canonical_hash(c) or sha256_file(a.data/'raw_slates.npz')!=f['raw_slates_sha256']:
        raise ValueError('Frozen cohort changed')
    raw=dict(np.load(a.data/'raw_slates.npz')); n,k,_=raw['x'].shape
    train=raw['split']==0; flat=raw['x'].reshape(-1,26)
    mean,scale=fit_normalization(raw['x'][train,:,1:].reshape(-1,25),np.repeat(raw['weight'][train],k))
    valid=np.isfinite(flat[:,1:]); numeric=np.where(valid,(flat[:,1:]-mean)/scale,0)
    tabs=flat[:,0].astype('int64')
    if (tabs<0).any() or (tabs>=15).any():raise ValueError('Unknown tab code')
    features=np.concatenate([numeric,(~valid).astype('float32'),np.eye(15,dtype='float32')[tabs]],axis=1).reshape(n,k,-1)
    names=['z_'+s for s in f['fields'][1:]]+['missing_'+s for s in f['fields'][1:]]+[f'tab_{i}' for i in range(15)]
    feature_spec=dict(names=names,mean=mean.tolist(),scale=scale.tolist(),fit_split='train only; inverse-inclusion-weighted moments',
        snapshot_assumption='same static-at-upload metadata assumption as v2',common_decision_time=True)
    tree_result=read_json(a.tree/'results.json');tree_spec=tree_result['models']['upload_metadata_assumed_static']
    checkpoint=a.tree/'upload_metadata_assumed_static.joblib'
    if sha256_file(checkpoint)!=tree_spec['model_sha256']:raise ValueError('Behavior model changed')
    tree=joblib.load(checkpoint)
    if tree['columns']!=26:raise ValueError('Wrong behavior features')
    p=np.column_stack([model.predict_proba(flat)[:,1] for model in tree['models']])
    behavior=apply_calibration(p,tree['calibration']).reshape(n,k,7).astype('float32')
    if a.no_jev:
        decision=dict(status='absent; excluded by explicit no-Jev model path',real_jev=False,
            structural_constants_are_not_model_scores=True)
        jev=np.full_like(behavior,.5);confidence=np.zeros_like(behavior);entropy=np.ones_like(behavior)
    else:
        if a.scores is None:raise ValueError('Real Jev scores required')
        complete=read_json(a.scores/'complete.json')
        if not complete['complete'] or sha256_file(a.scores/'scores.npz')!=complete['scores_sha256']:
            raise ValueError('Jev scoring incomplete or changed')
        if complete['registration']['config_hash']!=canonical_hash(c):raise ValueError('Score protocol changed')
        scored=dict(np.load(a.scores/'scores.npz'))
        if not np.array_equal(scored['candidate_ids'],raw['items']) or not np.array_equal(scored['slate_ids'],np.arange(n)) or scored['objective_names'].tolist()!=c['objectives']:
            raise ValueError('Jev candidate/objective identity mismatch')
        if str(scored['revision'])!=c['jev']['revision'] or str(scored['model'])!=c['jev']['model']:
            raise ValueError('Jev weights changed')
        jev,confidence,entropy=[scored[key].astype('float32') for key in ('probabilities','confidence','entropy')]
        decision=dict(status='complete real Choice scores',real_jev=True,model=c['jev']['model'],revision=c['jev']['revision'],
            semantics='per-objective within-slate choice probabilities, not event pXTR',complete_sha256=sha256_file(a.scores/'complete.json'))
    a.output.mkdir(parents=True,exist_ok=True)
    atomic_write_json(a.output/'feature_spec.json',feature_spec)
    atomic_write_json(a.output/'behavior_provenance.json',dict(checkpoint_sha256=sha256_file(checkpoint),
        source_results_sha256=sha256_file(a.tree/'results.json'),fit_end_exclusive_ms=1650211200000,
        calibration_end_exclusive_ms=1650297600000,reranker_start_ms=c['boundaries_ms'][0],no_behavior_refit=True))
    atomic_write_json(a.output/'decision_provenance.json',decision)
    atomic_write_json(a.output/'evaluation_protocol.json',dict(config=c,selection=selection,
        interpretation='weighted retrospective logged pseudo-slates; not causal online policy value'))
    manifest=dict(protocol_version='baseline-harness-v1',dataset='kuairand',data_kind='real',
        core_config_hash=canonical_hash(c),real_jev=decision['real_jev'],splits={},upstream={},provenance=runtime_provenance())
    for split,name in enumerate(c['split_names']):
        mask=raw['split']==split
        data=dict(candidate_features=features[mask].astype('float32'),behavior_scores=behavior[mask],jev_scores=jev[mask],
            labels=raw['labels'][mask].astype('float32'),mask=np.ones((int(mask.sum()),k),dtype=bool),
            jev_confidence=confidence[mask],jev_entropy=entropy[mask],popularity=raw['popularity'][mask].astype('float32'),
            history_length=raw['history_count'][mask].astype('float32'),
            slate_ids=np.flatnonzero(mask),candidate_ids=raw['items'][mask],user_ids=raw['user'][mask],
            timestamps=np.repeat(raw['decision'][mask,None],k,axis=1),
            objective_names=np.array(c['objectives']),directions=np.array(c['directions']),feature_names=np.array(names),
            feature_spec_hash=np.array(canonical_hash(feature_spec)),data_kind=np.array('real'),slate_weights=raw['weight'][mask],
            source_records=raw['records'][mask],source_exposure_times=raw['exposure_times'][mask])
        path=a.output/f'{name}.npz';np.savez_compressed(path,**data);ComparisonSlateDataset(path)
        manifest['splits'][name]=dict(path=path.name,sha256=sha256_file(path),slates=int(mask.sum()),
            weighted_population=float(raw['weight'][mask].sum()))
    for key in ('feature_spec','behavior_provenance','decision_provenance','evaluation_protocol'):
        path=a.output/f'{key}.json';manifest['upstream'][key]=dict(path=path.name,sha256=sha256_file(path))
    atomic_write_json(a.output/'manifest.json',manifest)
    print('Frozen real core tensors',a.output,'real_jev=',manifest['real_jev'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','tree','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scores',type=Path);p.add_argument('--no-jev',action='store_true')
    p.add_argument('--config',type=Path,default=Path('configs/kuairand_core.json'));a=p.parse_args()
    with threadpool_limits(limits=4):materialize(a)
