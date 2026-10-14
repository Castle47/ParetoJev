#!/usr/bin/env python3
"""Fresh equal-budget reranker fits with nested user-disjoint holdouts.

Existing real native scores only. The original future-date confirmation remains
untouched. Out-of-fold statistics are retrospective, not prospective evidence.
"""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paretojev.data.emer import fit_feature_spec,encode_features
from paretojev.experiments.comparison import _output_lock,_source_identity
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from run_emer_jev import dataset,fit,load_model,predict,paired_report
from run_emer_scale import immutable_json


def user_bucket(user,namespace,seed,modulus):
    message=f'{namespace}|{seed}|{int(user)}'.encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8],'big')%modulus


def roles_for_fold(users,fold,config):
    """0=train,1=inner validation,2=outer test; no labels enter this function."""
    unique=np.unique(users);role={}
    for user in unique:
        if user_bucket(user,'outer',config['partition_seed'],config['outer_folds'])==fold:role[int(user)]=2
        elif user_bucket(user,f'inner-{fold}',config['partition_seed'],config['inner_validation_modulus'])==0:role[int(user)]=1
        else:role[int(user)]=0
    return np.array([role[int(u)] for u in users],dtype='int8')


def train_only_encoding(x,weights,roles):
    spec=fit_feature_spec(x[roles==0],weights[roles==0])
    return encode_features(x,spec),spec


def source_identity(a,c):
    original=read_json(a.data/'training/emer_selection.json')
    common=read_json(a.data/'training/common_manifest.json')
    complete=read_json(a.data/'jev-native-v9-fp16/complete.json')
    audit=read_json(a.data/'training-native/native_response_audit.json')
    if original['registration']['source_sha256']!=_source_identity()['source_sha256']:
        raise ValueError('Original model implementation changed')
    if original['registration']['fit_script_sha256']!=sha256_file(Path('scripts/run_emer_jev.py')):
        raise ValueError('Original fit implementation changed')
    if common['identity']['raw_sha256']!=sha256_file(a.data/'raw_slates.npz'):
        raise ValueError('Raw development cohort changed')
    for name,h in common['files'].items():
        if sha256_file(a.data/'training'/name)!=h:raise ValueError('Common development input changed')
    if not complete['complete'] or complete['valid_decisions']!=252000 or complete['public_cache_mixed']:
        raise ValueError('Incomplete native scores')
    if sha256_file(a.data/'jev-native-v9-fp16/scores.npz')!=complete['scores_sha256']:
        raise ValueError('Native tensor changed')
    if not audit['all_raw_responses_match'] or audit['decisions']!=252000:
        raise ValueError('Real response audit required')
    files=[a.config,Path(__file__),Path('scripts/run_emer_jev.py'),a.data/'raw_slates.npz',
        a.data/'training/common_manifest.json',a.data/'training/train.npz',a.data/'training/validation.npz',
        a.data/'jev-native-v9-fp16/scores.npz',a.data/'jev-native-v9-fp16/registration.json',
        a.data/'jev-native-v9-fp16/complete.json',a.data/'training-native/native_response_audit.json']
    return dict(config=c,config_hash=canonical_hash(c),files={str(p):sha256_file(p) for p in files},
        model_source_sha256=_source_identity()['source_sha256'],paid_gpu=False,new_api_requests=0,
        scope='Retrospective nested user holdouts; not new chronological confirmation')


def common_arrays(a):
    splits=[]
    for name in ('train','validation'):
        with np.load(a.data/'training'/f'{name}.npz') as z:splits.append({k:z[k] for k in z.files})
    merged={k:np.concatenate([s[k] for s in splits]) for k in splits[0]}
    order=np.argsort(merged['slate_ids']);merged={k:v[order] for k,v in merged.items()}
    if not np.array_equal(merged['slate_ids'],np.arange(36000)):raise ValueError('Expected all36000 original IDs exactly once')
    with np.load(a.data/'raw_slates.npz') as z:
        x=z['x'];decision=z['decision'];users=z['user'];items=z['items'];labels=z['labels'];weights=z['weight']
    if not np.array_equal(merged['user_ids'],users) or not np.array_equal(merged['candidate_ids'],items) or not np.array_equal(merged['labels'],labels) or not np.array_equal(merged['slate_weights'],weights):
        raise ValueError('Original cohort alignment differs')
    bounds=read_json('configs/emer_jev_scale.json')['boundaries_ms']
    if not ((decision>=bounds[0])&(decision<bounds[-1])).all():raise ValueError('Non-development date entered CV')
    with np.load(a.data/'jev-native-v9-fp16/scores.npz') as z:
        if not np.array_equal(z['candidate_ids'],items):raise ValueError('Jev candidates differ')
        for key,name in [('jev_scores','probabilities'),('jev_entropy','entropy'),('jev_confidence','confidence')]:
            merged[key]=z[name].astype('float32')
    return merged,x


def prepare(a,c):
    identity=source_identity(a,c);immutable_json(a.output/'registration.json',identity)
    common,x=common_arrays(a);inventory=[]
    for fold in range(c['outer_folds']):
        out=a.output/f'fold-{fold}';out.mkdir(exist_ok=True)
        roles=roles_for_fold(common['user_ids'],fold,c)
        sets=[set(common['user_ids'][roles==i].tolist()) for i in range(3)]
        if any(not s for s in sets) or any(sets[i]&sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Missing or overlapping user partition')
        if (out/'data_manifest.json').exists():
            manifest=read_json(out/'data_manifest.json')
            if manifest['roles_sha256']!=hashlib.sha256(roles.tobytes()).hexdigest():raise ValueError('Fold changed')
            for name,h in manifest['files'].items():
                if sha256_file(out/name)!=h:raise ValueError('Fold data changed')
        else:
            encoded,spec=train_only_encoding(x,common['slate_weights'],roles)
            atomic_write_json(out/'feature_spec.json',spec);files={'feature_spec.json':sha256_file(out/'feature_spec.json')}
            sizes={}
            for i,name in enumerate(('train','validation','test')):
                use=roles==i;arrays={k:v[use] for k,v in common.items()};arrays['candidate_features']=encoded[use]
                path=out/f'{name}.npz';np.savez_compressed(path,**arrays);files[path.name]=sha256_file(path)
                sizes[name]=dict(slates=int(use.sum()),users=len(sets[i]))
            manifest=dict(fold=fold,roles_sha256=hashlib.sha256(roles.tobytes()).hexdigest(),files=files,sizes=sizes,
                user_disjoint=True,feature_fit='fold training only',outer_metrics_computed=False)
            immutable_json(out/'data_manifest.json',manifest)
        inventory.append(manifest)
    coverage=np.zeros(len(common['user_ids']),dtype=int)
    for fold in range(c['outer_folds']):coverage+=roles_for_fold(common['user_ids'],fold,c)==2
    if not np.array_equal(coverage,np.ones(len(coverage),dtype=int)):raise ValueError('Outer coverage differs')
    immutable_json(a.output/'preparation.json',dict(config_hash=canonical_hash(c),folds=inventory,all_slates_tested_once=True,
        slates=len(coverage),users=len(np.unique(common['user_ids'])),prospective_test_read=False))
    print('CV fold data prepared:',[m['sizes'] for m in inventory],flush=True)


def train(a,c):
    prepare(a,c)
    existing=read_json(a.output/'registration.json')
    if existing!=source_identity(a,c):raise ValueError('Registered source changed')
    selections={};completed=0
    grid=[(lr,coef) for lr in c['training']['learning_rates'] for coef in c['training']['prior_coefficients']]
    for fold in range(c['outer_folds']):
        directory=a.output/f'fold-{fold}'
        data={name:dataset(directory/f'{name}.npz') for name in ('train','validation')}
        selections[str(fold)]={}
        for seed in c['training']['seeds']:
            selections[str(fold)][str(seed)]={}
            for arm,use_jev in [('emer',False),('emer_jev',True)]:
                options=[]
                for trial,(lr,coef) in enumerate(grid):
                    path=directory/f'{arm}-seed{seed}-trial{trial}.pt'
                    if path.exists():
                        if not path.with_suffix('.json').exists():raise ValueError('Incomplete trial checkpoint; preserve before deterministic retry')
                        row=read_json(path.with_suffix('.json'))
                        if row['checkpoint_sha256']!=sha256_file(path):raise ValueError('Checkpoint changed')
                        if row['seed']!=seed or row['use_jev']!=use_jev or row['lr']!=lr or row['prior_coefficient']!=coef or len(row['history'])!=c['training']['epochs']:
                            raise ValueError('Trial protocol differs')
                    else:
                        _,row=fit(data,c,seed,use_jev,lr,coef,path)
                    options.append(dict(row,path=path.name));completed+=1
                    atomic_write_json(a.output/'progress.json',dict(completed_fits=completed,planned_fits=c['training']['total_complete_fits'],
                        fold=fold,seed=seed,arm=arm,trial=trial,outer_metrics_computed=False))
                    print(f'fit {completed}/72 fold={fold} seed={seed} arm={arm} trial={trial} inner_mean={row["validation"]:.6f}',flush=True)
                best=max(options,key=lambda r:r['validation'])
                selections[str(fold)][str(seed)][arm]={k:best[k] for k in ('path','checkpoint_sha256','validation','best_epoch','lr','prior_coefficient','parameters')}
        immutable_json(directory/'selection.json',selections[str(fold)])
        del data
    if completed!=72:raise ValueError('Equal-budget grid is incomplete')
    immutable_json(a.output/'selection.json',dict(config_hash=canonical_hash(c),folds=selections,completed_fits=completed,
        selection='inner user validation only',outer_metrics_computed=False))
    print('All72 fits selected; only now may outer performance be evaluated',flush=True)


def evaluate(a,c):
    if read_json(a.output/'registration.json')!=source_identity(a,c):raise ValueError('Registered source changed')
    selection=read_json(a.output/'selection.json')
    if selection['completed_fits']!=72 or selection['config_hash']!=canonical_hash(c):raise ValueError('Full frozen selection required')
    common,_=common_arrays(a);n=len(common['user_ids']);seen=np.zeros(n,bool)
    predictions={f'{arm}/{seed}':np.zeros((n,10),dtype='float32') for arm in ('emer','emer_jev') for seed in c['training']['seeds']}
    c_metric=dict(c,objectives=read_json('configs/emer_jev_scale.json')['objectives'],directions=[1]*6+[-1],
        evaluation=dict(c['evaluation'],click_maximum_absolute_ndcg_drop=.005,hate_maximum_absolute_risk_increase=.005))
    by_fold={}
    for fold in range(c['outer_folds']):
        directory=a.output/f'fold-{fold}';manifest=read_json(directory/'data_manifest.json')
        for name,h in manifest['files'].items():
            if sha256_file(directory/name)!=h:raise ValueError('Fold data changed before evaluation')
        test=dataset(directory/'test.npz')
        with np.load(directory/'test.npz') as z:ids=z['slate_ids']
        expected=roles_for_fold(common['user_ids'],fold,c)==2
        if seen[ids].any() or not np.array_equal(np.flatnonzero(expected),ids):raise ValueError('Outer assignment differs')
        local={}
        for seed in c['training']['seeds']:
            for arm in ('emer','emer_jev'):
                spec=selection['folds'][str(fold)][str(seed)][arm];path=directory/spec['path']
                if sha256_file(path)!=spec['checkpoint_sha256']:raise ValueError('Selected model changed')
                scores=predict(load_model(path),test,c['training']['batch_size'])
                predictions[f'{arm}/{seed}'][ids]=scores;local[f'{arm}/{seed}']=scores
        seen[ids]=True;by_fold[str(fold)]=paired_report(local,test,c_metric)
    if not seen.all():raise ValueError('Missing outer predictions')
    metric_data=SimpleNamespace(arrays={'labels':torch.from_numpy(common['labels']),'mask':torch.from_numpy(common['mask'])},
        users=common['user_ids'],sample_weights=common['slate_weights'])
    pooled=paired_report(predictions,metric_data,c_metric)
    guards={name:row['absolute_gain']>=-c['evaluation']['maximum_positive_goal_absolute_point_drop'] for name,row in list(pooled['objectives'].items())[:6]}
    conditions=dict(primary_conditional_ci_positive=pooled['positive_mean']['confidence_interval'][0]>0,
        primary_positive_each_seed=all(np.mean(r['emer_jev'][:6])>np.mean(r['emer'][:6]) for r in pooled['per_seed'].values()),
        primary_positive_each_fold=all(r['positive_mean']['absolute_gain']>0 for r in by_fold.values()),
        positive_point_guards=all(guards.values()),
        hate_point_guard=pooled['objectives']['is_hate']['absolute_gain']>=-c['evaluation']['maximum_hate_risk_absolute_point_increase'])
    # Keep legacy long-view diagnostics separate from the user's aggregate goal.
    for r in [pooled,*by_fold.values()]:r['legacy_longview_diagnostics']=r.pop('gates');r.pop('all_effect_gates')
    result=dict(role='retrospective user-disjoint out-of-fold comparison; not prospective chronological confirmation',
        pooled=pooled,by_fold=by_fold,conditions={k:bool(v) for k,v in conditions.items()},screen_passed=bool(all(conditions.values())),
        selection_sha256=sha256_file(a.output/'selection.json'),slates=n,users=len(np.unique(common['user_ids'])),
        every_user_excluded_from_predicting_model_training_and_selection=True,all_outer_rows_predicted_once=True,
        no_prospective_test_data_read=True,paid_gpu=False,new_api_requests=0,limitations=c['limitations'],
        interval_caveat=c['evaluation']['uncertainty'])
    if a.stage=='replay':
        with np.load(a.output/'predictions.npz') as old:
            if set(old.files)!=set(predictions) or any(not np.array_equal(old[k],v) for k,v in predictions.items()):raise ValueError('OOF prediction replay failed')
        if read_json(a.output/'evaluation.json')!=result:raise ValueError('OOF metric replay failed')
        atomic_write_json(a.output/'replay.json',dict(exact_predictions_and_metrics=True,
            evaluation_sha256=sha256_file(a.output/'evaluation.json'),predictions_sha256=sha256_file(a.output/'predictions.npz')))
    else:
        if (a.output/'evaluation.json').exists():raise ValueError('Use replay for existing evaluation')
        np.savez_compressed(a.output/'predictions.npz',**predictions);immutable_json(a.output/'evaluation.json',result)
    print('Retrospective OOF screen:',result['conditions'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_user_cv_v1.json'));p.add_argument('--stage',choices=['prepare','train','evaluate','replay'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=read_json(a.config)
    torch.set_num_threads(c['training']['cpu_threads']);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=c['training']['cpu_threads']):
        prepare(a,c) if a.stage=='prepare' else train(a,c) if a.stage=='train' else evaluate(a,c)
