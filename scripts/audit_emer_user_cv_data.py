#!/usr/bin/env python3
"""Audit actual saved CV tensors without predicting or selecting any model."""
import argparse
from pathlib import Path
import numpy as np
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from paretojev.data.emer import encode_features
from run_emer_user_cv import user_bucket


def audit(a):
    c=read_json(a.config);reg=read_json(a.run/'registration.json')
    if reg['config_hash']!=canonical_hash(c):raise ValueError('CV protocol differs')
    for path,h in reg['files'].items():
        if sha256_file(path)!=h:raise ValueError('Registered source changed')
    with np.load(a.data/'raw_slates.npz') as z:raw={k:z[k] for k in ('x','user','items','labels','weight','history_count')}
    common={}
    for split in ('train','validation'):
        with np.load(a.data/'training'/f'{split}.npz') as z:
            for key in ('slate_ids','behavior_scores','popularity'):
                common.setdefault(key,[]).append(z[key])
    common={k:np.concatenate(v) for k,v in common.items()};order=np.argsort(common['slate_ids'])
    common={k:v[order] for k,v in common.items()}
    if not np.array_equal(common['slate_ids'],np.arange(36000)):raise ValueError('Common IDs differ')
    with np.load(a.data/'jev-native-v9-fp16/scores.npz') as z:
        signals={key:z[key] for key in ('probabilities','entropy','confidence','candidate_ids')}
    coverage=np.zeros(36000,dtype=int);folds=[]
    for fold in range(3):
        directory=a.run/f'fold-{fold}';manifest=read_json(directory/'data_manifest.json');spec=read_json(directory/'feature_spec.json')
        saved={}
        for split in ('train','validation','test'):
            path=directory/f'{split}.npz'
            if sha256_file(path)!=manifest['files'][path.name]:raise ValueError('Fold file hash differs')
            with np.load(path) as z:saved[split]={k:z[k] for k in z.files}
        sets={k:set(d['user_ids'].tolist()) for k,d in saved.items()}
        if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']:
            raise ValueError('User leakage between saved files')
        ids=np.concatenate([d['slate_ids'] for d in saved.values()])
        if not np.array_equal(np.sort(ids),np.arange(36000)):raise ValueError('Incomplete/disjoint row partition')
        # Recompute weighted moments column by column, separately from fit_feature_spec.
        train=saved['train']['slate_ids'];x=raw['x'][train,:,1:].reshape(-1,25).astype('float64')
        weights=np.repeat(raw['weight'][train],10);means=[];scales=[]
        for j in range(25):
            valid=np.isfinite(x[:,j]);v=x[valid,j];w=weights[valid];den=max(w.sum(),1.)
            mean=float(np.sum(v*w)/den);scale=float(np.sqrt(np.sum(w*(v-mean)**2)/den))
            means.append(mean);scales.append(1. if scale<1e-5 else scale)
        np.testing.assert_allclose(spec['mean'],means,rtol=1e-11,atol=1e-10)
        np.testing.assert_allclose(spec['scale'],scales,rtol=1e-11,atol=1e-10)
        for split,d in saved.items():
            ids=d['slate_ids']
            for key,expected in [('user_ids',raw['user'][ids]),('candidate_ids',raw['items'][ids]),
                ('labels',raw['labels'][ids]),('slate_weights',raw['weight'][ids]),
                ('history_length',raw['history_count'][ids,None].astype('float32')),
                ('behavior_scores',common['behavior_scores'][ids]),('popularity',common['popularity'][ids]),
                ('jev_scores',signals['probabilities'][ids].astype('float32')),
                ('jev_entropy',signals['entropy'][ids].astype('float32')),
                ('jev_confidence',signals['confidence'][ids].astype('float32')),
                ('candidate_features',encode_features(raw['x'][ids],spec))]:
                if not np.array_equal(d[key],expected,equal_nan=True):raise ValueError(f'Fold{fold}/{split} differs: {key}')
            if not d['mask'].all():raise ValueError('Unexpected padding')
            for user in sets[split]:
                outer=user_bucket(user,'outer',c['partition_seed'],3)
                inner=user_bucket(user,f'inner-{fold}',c['partition_seed'],c['inner_validation_modulus'])
                expected='test' if outer==fold else 'validation' if inner==0 else 'train'
                if split!=expected:raise ValueError('Saved user assignment differs from frozen hash')
            if split=='test':coverage[ids]+=1
        folds.append(dict(fold=fold,users={k:len(v) for k,v in sets.items()},disjoint_users=True,
            all_saved_inputs_match=True,training_only_moments_recomputed=True))
        print('Audited actual fold tensors',fold,flush=True)
        del saved
    if not (coverage==1).all():raise ValueError('Repeated/missing outer test row')
    result=dict(folds=folds,outer_rows=36000,every_row_tested_once=True,model_performance_computed=False,
        registration_sha256=sha256_file(a.run/'registration.json'),script_sha256=sha256_file(Path(__file__)))
    atomic_write_json(a.run/'data_audit.json',result);print('Actual CV data audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_user_cv_v1.json'));audit(p.parse_args())
