#!/usr/bin/env python3
"""Fixed-date evaluation of the exported candidate; no performance without real scores."""
import argparse
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'deployment/emer_candidate'))
from candidate_runtime import Candidate,user_routes
from paretojev.data.emer import encode_features
from paretojev.experiments.comparison import _output_lock
from paretojev.evaluation.comparison import comparison_objective_metrics
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from run_emer_scale import immutable_json
from run_emer_jev import paired_report
from run_emer_balanced_transfer import screen
from evaluate_multiobjective_confirmation import audit_scores,materialize

SCORE_FILES=('complete.json','registration.json','scores.npz','native.sqlite3')


def require_scores(path):
    missing=list(SCORE_FILES) if path is None else [n for n in SCORE_FILES if not (path/n).is_file()]
    if missing:raise ValueError('Real score directory required before test inference; missing: '+', '.join(missing))
    complete=read_json(path/'complete.json')
    if not complete.get('complete') or complete.get('public_cache_mixed') or complete.get('valid_decisions')!=350000 or complete.get('http_groups')!=50000:
        raise ValueError('All350000 real decisions/50000 groups are required before test inference')


def paths(a,c):
    development=a.root/'emer-jev-scale-v3';cohort=a.root/'emer-multiobjective-confirmation-v1'
    bundle=a.bundle or development/'exports/emer-matched-blended-v1';old_config=REPO/c['score_protocol']['path']
    return development,cohort,bundle,old_config


def verify_upstream_runtime(candidate):
    for name in ('scikit-learn','joblib','pandas','scipy'):
        if importlib.metadata.version(name)!=candidate.manifest['runtime']['packages'][name]:raise ValueError('Upstream inference dependency changed: '+name)


def prepare(a,c):
    development,cohort,bundle,old_config=paths(a,c)
    if sha256_file(old_config)!=c['score_protocol']['sha256'] or sha256_file(bundle/'manifest.json')!=c['candidate_manifest_sha256']:raise ValueError('Frozen scoring protocol or candidate differs')
    candidate=Candidate(bundle);verify_upstream_runtime(candidate);manifest=candidate.manifest
    if any(manifest['routing'][k]!=c['routing'][k] for k in ('partition_seed','folds','namespace')):raise ValueError('Routing policy differs')
    expected={'preparation.json':'preparation_sha256','cohort_audit.json':'cohort_audit_sha256','sampling_replay.json':'sampling_replay_sha256',
        'transfer/transfer.json':'transfer_sha256','raw_slates.npz':'raw_sha256','transfer/jev_input.jsonl.gz':'compressed_input_sha256','jev_input.jsonl':'source_input_sha256'}
    inputs={str(a.config):sha256_file(a.config),str(old_config):sha256_file(old_config),str(bundle/'manifest.json'):sha256_file(bundle/'manifest.json')}
    for name,key in expected.items():
        p=cohort/name
        if sha256_file(p)!=c['cohort'][key]:raise ValueError('Reserved cohort changed: '+name)
        inputs[str(p)]=c['cohort'][key]
    prep=read_json(cohort/'preparation.json');audit=read_json(cohort/'cohort_audit.json');sampling=read_json(cohort/'sampling_replay.json')
    old=read_json(old_config)
    if prep['config_hash']!=canonical_hash(old) or not sampling['exact_selection_replayed'] or sampling['performance_evaluated']:raise ValueError('Untuned frozen sampling required')
    if not all(audit[k] for k in ('source_labels_features_and_histories_match','weights_verified','fresh_time_window_verified','exact_prompt_alignment')):raise ValueError('Full source audit required')
    behavior=a.root/'behavior-v2/run-v2/upload_metadata_assumed_static.joblib'
    if sha256_file(behavior)!=manifest['upstream']['behavior_checkpoint_sha256']:raise ValueError('Behavior model changed')
    inputs[str(behavior)]=sha256_file(behavior)
    for p in (cohort/'feature_spec.json',development/'training/feature_spec.json',development/'training/common_manifest.json',
        development/'training/emer_selection.json',development/'prior-popularity.npy',development/'jev-native-v9-fp16/registration.json'):
        inputs[str(p)]=sha256_file(p)
    for relative in ('scripts/evaluate_multiobjective_confirmation.py','scripts/run_emer_jev.py','scripts/run_emer_balanced_transfer.py',
        'deployment/emer_candidate/candidate_runtime.py','deployment/autodl/score_confirmation.py'):
        p=REPO/relative;inputs[str(p)]=sha256_file(p)
    inputs[str(Path(__file__))]=sha256_file(Path(__file__))
    with np.load(cohort/'raw_slates.npz',allow_pickle=False) as z:
        users=z['user'];x=z['x'];items=z['items'];times=z['decision']
    if len(users)!=50000 or len(np.unique(users))!=c['cohort']['users'] or items.shape!=(50000,10) or x.shape!=(50000,10,26):raise ValueError('Cohort size/schema differs')
    if not ((times>=c['cohort']['start_ms'])&(times<c['cohort']['end_ms'])).all():raise ValueError('Wrong date window')
    routes=user_routes(users,c['routing']['partition_seed'],c['routing']['folds']);records=[]
    for f in range(3):
        ids=np.flatnonzero(routes==f);encoded=encode_features(x[ids],candidate.specs[f])
        if not len(ids) or not np.isfinite(encoded).all() or encoded.shape!=(len(ids),10,65):raise ValueError('Per-route encoder preflight failed')
        records.append(dict(fold=f,slates=len(ids),users=len(np.unique(users[ids])),feature_shape=list(encoded.shape),
            feature_spec_sha256=sha256_file(candidate.path(manifest['routes'][str(f)]['feature_spec']))))
    preflight=dict(slates=len(users),users=len(np.unique(users)),route_encoders=records,feedback_labels_read=False,model_performance_computed=False)
    immutable_json(a.output/'input_preflight.json',preflight);inputs[str(a.output/'input_preflight.json')]=sha256_file(a.output/'input_preflight.json')
    immutable_json(a.output/'registration.json',dict(config=c,inputs=inputs,data_root=str(a.root.resolve()),candidate_manifest_sha256=c['candidate_manifest_sha256'],
        scoring_protocol_unchanged=True,new_training=False,source_test_performance_not_opened=True))
    missing=list(SCORE_FILES) if a.scores is None else [n for n in SCORE_FILES if not (a.scores/n).is_file()]
    readiness=dict(local_preflight_passed=True,score_artifacts_missing=missing,model_performance_computed=False,
        required_real_groups=50000,required_real_decisions=350000,budget_cap_cny=65,new_paid_inference_started=False)
    atomic_write_json(a.output/'readiness.json',readiness);print(readiness,flush=True)


def verify_preparation(a,c):
    reg=read_json(a.output/'registration.json')
    if reg['config']!=c:raise ValueError('Independent protocol changed')
    if str(a.root.resolve())!=reg['data_root']:raise ValueError('Registered data root changed; prepare a new output after a verified relocation')
    for p,h in reg['inputs'].items():
        if sha256_file(p)!=h:raise ValueError('Registered independent input/source changed')
    _,_,bundle,_=paths(a,c)
    if sha256_file(bundle/'manifest.json')!=c['candidate_manifest_sha256']:raise ValueError('Requested candidate differs from frozen manifest')
    candidate=Candidate(bundle);verify_upstream_runtime(candidate)
    return candidate


def evaluate(a,c):
    # This guard precedes model loading, materialization and any test prediction.
    require_scores(a.scores)
    candidate=verify_preparation(a,c);development,cohort,bundle,old_config=paths(a,c)
    common=a.output/'common';common.mkdir(exist_ok=True)
    old_args=SimpleNamespace(root=a.root,development=development,data=cohort,scores=a.scores,config=old_config,output=common)
    old_c=read_json(old_config);scores=audit_scores(old_args,old_c)
    d=materialize(old_args,old_c,scores);del scores
    # Legacy global encoded features are deliberately not an inference input.
    with np.load(cohort/'raw_slates.npz') as z:raw=z['x'];items=z['items'];times=z['decision']
    inputs=dict(raw_features=raw,behavior_scores=d.arrays['behavior_scores'].numpy(),jev_scores=d.arrays['jev_scores'].numpy(),
        jev_entropy=d.arrays['jev_entropy'].numpy(),jev_confidence=d.arrays['jev_confidence'].numpy(),
        user_ids=d.users,candidate_ids=items,mask=d.arrays['mask'].numpy(),popularity=d.arrays['popularity'].numpy(),history_length=d.arrays['history_length'].numpy())
    predictions={};routes=None
    for seed in c['training']['seeds']:
        for arm in ('emer','emer_jev','anchor'):
            r=candidate.predict(inputs,arm=arm,seed=seed);predictions[f'{arm}/{seed}']=r['scores'];routes=r['fold_ids']
            print('Predicted frozen routed arm:',arm,seed,flush=True)
        for f in range(3):
            row=candidate.manifest['routes'][str(f)]['seeds'][str(seed)]
            if row['emer']['path']==row['emer_jev']['path'] and not np.array_equal(predictions[f'emer/{seed}'][routes==f],predictions[f'emer_jev/{seed}'][routes==f]):raise ValueError('Fallback outputs differ from matched control')
    mc=dict(c,evaluation=dict(c['evaluation'],click_maximum_absolute_ndcg_drop=.005,hate_maximum_absolute_risk_increase=.005))
    comparisons={'signal':('emer','emer_jev'),'original_reference':('anchor','emer_jev'),'training_control':('anchor','emer')}
    def compare(p,data):
        result={}
        for name,(base,full) in comparisons.items():
            pair={f'{arm}/{seed}':p[f'{source}/{seed}'] for arm,source in [('emer',base),('emer_jev',full)] for seed in c['training']['seeds']}
            r=paired_report(pair,data,mc);r.pop('gates');r.pop('all_effect_gates');result[name]=r
        return result
    pooled=compare(predictions,d);by_route={name:{} for name in comparisons}
    for f in range(3):
        use=routes==f;sub=SimpleNamespace(arrays={k:v[use] for k,v in d.arrays.items()},users=d.users[use],sample_weights=d.sample_weights[use])
        for name,r in compare({k:v[use] for k,v in predictions.items()},sub).items():by_route[name][str(f)]=r
    conditions={name:screen(pooled[name],by_route[name],mc) for name in c['evaluation']['required_comparisons']}
    labels=d.arrays['labels'].numpy();mask=d.arrays['mask'].numpy();w=d.sample_weights
    support=dict(slates=len(d),users=len(np.unique(d.users)),candidate_exposures=int(mask.sum()),
        positive_exposures=labels.sum((0,1)).astype(int).tolist(),positive_slates=(labels.sum(1)>0).sum(0).tolist(),
        weight_kish_effective_slates=float(w.sum()**2/(w*w).sum()))
    days=(times+8*3600000)//86400000;by_day={}
    for day in np.unique(days):
        use=days==day;points={}
        for arm in ('emer','emer_jev','anchor'):
            points[arm]=np.mean([comparison_objective_metrics(predictions[f'{arm}/{seed}'][use],labels[use],mask[use],c['directions'],top_k=3,sample_weights=w[use])[0] for seed in c['training']['seeds']],0).tolist()
        by_day[str(np.datetime64(int(day),'D'))]=dict(slates=int(use.sum()),users=len(np.unique(d.users[use])),points=points)
    result=dict(role=c['role'],pooled=pooled,by_route=by_route,conditions=conditions,screen_passed=all(all(v.values()) for v in conditions.values()),
        support=support,by_day=by_day,candidate_manifest_sha256=c['candidate_manifest_sha256'],config_sha256=sha256_file(a.config),
        response_audit_sha256=sha256_file(common/'response_audit.json'),limitations=c['limitations'],online_business_success=False)
    if a.stage=='replay':
        with np.load(a.output/'predictions.npz') as old:
            if set(old.files)!=set(predictions) or any(not np.array_equal(old[k],v) for k,v in predictions.items()):raise ValueError('Independent prediction replay failed')
        if read_json(a.output/'evaluation.json')!=result:raise ValueError('Independent metric replay failed')
        atomic_write_json(a.output/'replay.json',dict(exact_predictions_and_metrics=True,evaluation_sha256=sha256_file(a.output/'evaluation.json'),predictions_sha256=sha256_file(a.output/'predictions.npz'),all_raw_responses_audited_again=True))
    else:
        if (a.output/'evaluation.json').exists():raise ValueError('Use replay for existing independent evaluation')
        np.savez_compressed(a.output/'predictions.npz',**predictions);immutable_json(a.output/'evaluation.json',result)
    print('Independent routed comparison conditions:',conditions,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path);p.add_argument('--scores',type=Path);p.add_argument('--config',type=Path,default=REPO/'configs/emer_routed_independent_v1.json')
    p.add_argument('--stage',choices=['prepare','evaluate','replay'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=read_json(a.config)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=2):
        prepare(a,c) if a.stage=='prepare' else evaluate(a,c)
