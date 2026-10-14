#!/usr/bin/env python3
"""One frozen chronological EMER/Jev comparison, with exact-source score audit.

Refuses to produce even baseline test metrics until all real Jev scores exist.
No fitting, checkpoint selection, score substitution or adaptive sample extension.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deployment/autodl'))
from score_confirmation import canonical, validate_group
from paretojev.data.emer import encode_features
from paretojev.evaluation.comparison import comparison_objective_metrics
from paretojev.experiments.comparison import _output_lock,_source_identity
from paretojev.training.behavior_calibration import apply_calibration
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file
from run_emer_jev import dataset,load_model,predict,paired_report
from run_emer_scale import immutable_json


def confirmation_gates(report, config):
    goals=list(config['objectives']);ev=config['evaluation']
    if len(goals)!=7 or goals[-1]!='is_hate':raise ValueError('Unsupported objective order')
    values=np.array([report['objectives'][name]['absolute_gain'] for name in goals])
    ci=np.array(report['positive_mean']['confidence_interval'])
    if not np.isfinite(values).all() or not np.isfinite(ci).all():raise ValueError('Missing goal coverage')
    per_seed=[float(np.mean(row['emer_jev'][:6])-np.mean(row['emer'][:6])) for row in report['per_seed'].values()]
    if not per_seed or not np.isfinite(per_seed).all():raise ValueError('Missing seed coverage')
    guards={name:bool(values[i]>=-ev['maximum_positive_goal_absolute_point_drop']) for i,name in enumerate(goals[:6])}
    gates=dict(primary_mean_ci_positive=bool(ci[0]>0),primary_mean_gain_every_seed=all(v>0 for v in per_seed),
        all_positive_goal_point_guards=all(guards.values()),
        hate_risk_point_guard=bool(values[-1]>=-ev['maximum_hate_risk_absolute_point_increase']))
    return dict(gates=gates,positive_goal_point_guards=guards,all_conditions=all(gates.values()),
        scope='Overall offline aggregate superiority with point-estimate guardrails; not simultaneous population Pareto dominance or online causal uplift')


def audit_scores(a,c):
    complete=read_json(a.scores/'complete.json');reg=read_json(a.scores/'registration.json')
    dev_reg=a.development/'jev-native-v9-fp16/registration.json'
    expected_dev=c['candidate_selection']['source_registration_sha256']
    packet=read_json(a.data/'transfer/transfer.json');prep=read_json(a.data/'preparation.json')
    if not complete['complete'] or complete['public_cache_mixed'] or complete['valid_decisions']!=350000 or complete['http_groups']!=50000:
        raise ValueError('All350000 real decisions must be complete before any test performance evaluation')
    if complete['registration_sha256']!=sha256_file(a.scores/'registration.json') or complete['scores_sha256']!=sha256_file(a.scores/'scores.npz'):
        raise ValueError('Score artifacts changed')
    if sha256_file(dev_reg)!=expected_dev or reg['development_registration_sha256']!=expected_dev:
        raise ValueError('Development inference receipt differs')
    if reg['source']!=read_json(dev_reg)['source'] or not reg['test_opened']:
        raise ValueError('Confirmation producer differs from frozen development')
    if reg['slates']!=50000 or reg['goals']!=c['objectives'] or reg['confirmation_config_sha256']!=sha256_file(a.config):
        raise ValueError('Scoring registration differs')
    if reg['script_sha256']!=sha256_file(Path(__file__).resolve().parents[1]/'deployment/autodl/score_confirmation.py'):
        raise ValueError('Deployed collector differs from registered local implementation')
    if reg['transfer_manifest_sha256']!=sha256_file(a.data/'transfer/transfer.json'):
        raise ValueError('Transferred manifest differs')
    if packet['confirmation_config_sha256']!=sha256_file(a.config) or packet['development_registration_sha256']!=expected_dev:
        raise ValueError('Packet protocol differs')
    if sha256_file(a.data/'sampling_replay.json')!=packet['sampling_replay_sha256'] or not read_json(a.data/'sampling_replay.json')['exact_selection_replayed']:
        raise ValueError('Frozen sample replay is missing or differs')
    if not packet['independent_test_included'] or packet['labels_included'] or packet['slates']!=50000:
        raise ValueError('Unexpected transfer scope')
    input_path=a.data/'transfer/jev_input.jsonl.gz'
    if sha256_file(input_path)!=packet['compressed_sha256'] or reg['input_sha256']!=packet['compressed_sha256']:
        raise ValueError('Scoring input bytes differ')
    if sha256_file(a.data/'jev_input.jsonl')!=packet['source_input_sha256'] or prep['prompts']['input_sha256']!=packet['source_input_sha256']:
        raise ValueError('Source prompt file differs')
    with gzip.open(input_path,'rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=packet['source_input_sha256']:raise ValueError('Decoded prompts differ')
    with gzip.open(input_path,'rt') as f:rows=[json.loads(line) for line in f]
    if len(rows)!=50000 or any(r['slate_id']!=i or r['split']!='test' or r['questions']!=c['questions'] for i,r in enumerate(rows)):
        raise ValueError('Prompt rows changed')
    with np.load(a.scores/'scores.npz',allow_pickle=False) as z:scores={k:z[k] for k in z.files}
    p=scores['probabilities']
    if p.shape!=(50000,10,7) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,rtol=0,atol=1e-5):
        raise ValueError('Invalid categorical probabilities')
    if scores['objective_names'].tolist()!=c['objectives'] or not np.array_equal(scores['slate_ids'],np.arange(50000)):
        raise ValueError('Tensor goal/row order differs')
    if str(scores['model'])!=c['jev']['model'] or str(scores['revision'])!=c['jev']['revision']:
        raise ValueError('Tensor producer differs')
    entropy=np.clip(-(p*np.log(np.maximum(p,1e-300))).sum(1)/np.log(10),0,1)
    for name,expected in [('entropy',np.broadcast_to(entropy[:,None,:],p.shape)),('confidence',np.broadcast_to(1-entropy[:,None,:],p.shape))]:
        if scores[name].shape!=p.shape or not np.allclose(scores[name],expected,rtol=0,atol=1e-10):raise ValueError('Signal summaries differ')
    with sqlite3.connect('file:'+str((a.scores/'native.sqlite3').resolve())+'?mode=ro',uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('Score database integrity failed')
        children=iter(db.execute('SELECT row_id,goal,payload_hash,raw_json FROM results ORDER BY row_id,goal'))
        groups=0;decisions=0
        for i,key,text in db.execute('SELECT row_id,payload_hash,raw_json FROM groups ORDER BY row_id'):
            if i!=groups or not 0<=i<len(rows):raise ValueError('Group coverage/order differs')
            row=rows[i]
            if not np.array_equal(scores['candidate_ids'][i],row['candidate_ids']):raise ValueError('Candidate IDs differ')
            wire=dict(kind='choice',state=row['state'],questions=row['questions'],options=row['options'],thinking='off')
            if key!=hashlib.sha256(canonical(wire).encode()).hexdigest():raise ValueError('Original request hash differs')
            group=json.loads(text);values=validate_group(group,row)
            for j,value in enumerate(values):
                saved=next(children,None)
                if saved is None or saved[:2]!=(i,j):raise ValueError('Missing child response')
                payload=dict(kind='choice',state=row['state'],question=row['questions'][j],options=row['options'],thinking='off')
                if saved[2]!=hashlib.sha256(canonical(payload).encode()).hexdigest() or json.loads(saved[3])!=value:
                    raise ValueError('Stored child differs from original HTTP response')
                if not np.array_equal(value['probabilities'],p[i,:,j]):raise ValueError('Materialized response differs')
                decisions+=1
            groups+=1
        if groups!=50000 or decisions!=350000 or next(children,None) is not None:raise ValueError('Raw coverage differs')
    receipt=dict(groups=groups,decisions=decisions,exact_http_response_alignment=True,
        scores_sha256=complete['scores_sha256'],database_sha256=sha256_file(a.scores/'native.sqlite3'),
        producer_id=reg['source']['producer_id'])
    immutable_json(a.output/'response_audit.json',receipt)
    return scores


def materialize(a,c,scores):
    prep=read_json(a.data/'preparation.json');audit=read_json(a.data/'cohort_audit.json')
    if prep['config_hash']!=canonical_hash(c) or prep['raw_slates_sha256']!=sha256_file(a.data/'raw_slates.npz'):
        raise ValueError('Cohort changed')
    if audit['source_candidates']!=500000 or audit['histories_checked']!=50000 or not all(audit[k] for k in
        ('source_labels_features_and_histories_match','weights_verified','fresh_time_window_verified')):
        raise ValueError('Full source audit is required')
    if not audit.get('exact_prompt_alignment') or audit['raw_sha256']!=prep['raw_slates_sha256'] or audit['input_sha256']!=prep['prompts']['input_sha256']:
        raise ValueError('Prompt/source audit identity differs')
    spec_path=a.development/'training/feature_spec.json'
    if sha256_file(spec_path)!=c['candidate_selection']['common_feature_spec_sha256'] or sha256_file(a.data/'feature_spec.json')!=sha256_file(spec_path):
        raise ValueError('Shared feature specification changed')
    common=read_json(a.development/'training/common_manifest.json')
    base=read_json(a.development/'training/emer_selection.json')
    if base['registration']['common_manifest_sha256']!=sha256_file(a.development/'training/common_manifest.json'):
        raise ValueError('Common feature provenance differs')
    tree_path=a.root/'behavior-v2/run-v2/upload_metadata_assumed_static.joblib'
    if sha256_file(tree_path)!=common['identity']['behavior_sha256']:raise ValueError('Frozen behavior model changed')
    manifest=dict(raw_sha256=sha256_file(a.data/'raw_slates.npz'),feature_spec_sha256=sha256_file(spec_path),
        behavior_sha256=sha256_file(tree_path),scores_sha256=sha256_file(a.scores/'scores.npz'),
        script_sha256=sha256_file(Path(__file__)),source_sha256=_source_identity()['source_sha256'])
    path=a.output/'test.npz'
    if path.exists():
        old=read_json(a.output/'tensor_manifest.json')
        if old['inputs']!=manifest or old['test_sha256']!=sha256_file(path):raise ValueError('Test tensor changed')
        return dataset(path)
    with np.load(a.data/'raw_slates.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    if len(raw['user'])!=50000 or not np.array_equal(raw['items'],scores['candidate_ids']):raise ValueError('Cohort identities differ')
    prior_path=a.development/'prior-popularity.npy'
    packet=read_json(a.data/'transfer/transfer.json')
    if not packet['frozen_popularity_verified'] or sha256_file(prior_path)!=packet['popularity_sha256']:
        raise ValueError('Popularity prior changed')
    if not np.array_equal(raw['popularity'],np.load(prior_path,mmap_mode='r')[raw['items']]):
        raise ValueError('Candidate popularity changed')
    if not ((raw['decision']>=c['fresh_test_start_ms'])&(raw['decision']<c['fresh_test_end_ms'])).all():raise ValueError('Wrong chronological window')
    tree=joblib.load(tree_path);flat=raw['x'].reshape(-1,26)
    behavior=apply_calibration(np.column_stack([m.predict_proba(flat)[:,1] for m in tree['models']]),tree['calibration']).reshape(-1,10,7)
    features=encode_features(raw['x'],read_json(spec_path))
    arrays=dict(candidate_features=features,behavior_scores=behavior.astype('float32'),
        jev_scores=scores['probabilities'].astype('float32'),jev_entropy=scores['entropy'].astype('float32'),
        jev_confidence=scores['confidence'].astype('float32'),labels=raw['labels'].astype('float32'),
        mask=np.ones(raw['items'].shape,dtype=bool),popularity=raw['popularity'][...,None].astype('float32'),
        history_length=raw['history_count'][:,None].astype('float32'),slate_weights=raw['weight'],
        user_ids=raw['user'],candidate_ids=raw['items'],decision_times=raw['decision'])
    np.savez_compressed(path,**arrays)
    immutable_json(a.output/'tensor_manifest.json',dict(inputs=manifest,test_sha256=sha256_file(path)))
    return dataset(path)


def main(a):
    c=read_json(a.config)
    candidates=c['candidate_selection'];base_sel=a.development/'training/emer_selection.json';full_sel=a.development/'training-native/selection.json'
    for p,key in [(base_sel,'baseline_selection_sha256'),(full_sel,'jev_selection_sha256')]:
        if sha256_file(p)!=candidates[key]:raise ValueError('Frozen model selection changed')
    base=read_json(base_sel)
    if base['registration']['source_sha256']!=_source_identity()['source_sha256'] or base['registration']['fit_script_sha256']!=sha256_file(Path('scripts/run_emer_jev.py')):
        raise ValueError('Inference implementation changed')
    paths={}
    for arm,seeds in candidates['checkpoints'].items():
        folder='training' if arm=='emer' else 'training-native'
        for seed,row in seeds.items():
            path=a.development/folder/row['path']
            if sha256_file(path)!=row['checkpoint_sha256']:raise ValueError('Frozen weights changed')
            paths[f'{arm}/{seed}']=path
    registration=dict(config=c,config_sha256=sha256_file(a.config),evaluator_sha256=sha256_file(Path(__file__)),
        paired_metric_script_sha256=sha256_file(Path('scripts/run_emer_jev.py')),
        score_validator_sha256=sha256_file(Path(__file__).resolve().parents[1]/'deployment/autodl/score_confirmation.py'),
        checkpoints={k:sha256_file(p) for k,p in paths.items()},independent_period='Apr23-25; no test-based fitting or model selection')
    immutable_json(a.output/'registration.json',registration)
    scores=audit_scores(a,c);d=materialize(a,c,scores);del scores
    predictions={}
    for name,path in paths.items():
        model=load_model(path)
        if model.config.use_jev!=name.startswith('emer_jev/'):raise ValueError('Wrong inference arm')
        predictions[name]=predict(model,d,c['training']['batch_size'])
        print('Predicted',name,flush=True)
    metric_config=dict(c,evaluation=dict(c['evaluation'],click_maximum_absolute_ndcg_drop=.005,hate_maximum_absolute_risk_increase=.005))
    report=paired_report(predictions,d,metric_config)
    report.pop('gates');report.pop('all_effect_gates')
    report['confirmation']=confirmation_gates(report,c)
    labels=d.arrays['labels'].numpy();mask=d.arrays['mask'].numpy();w=d.sample_weights
    support=dict(slates=len(d),users=len(np.unique(d.users)),candidate_exposures=int(mask.sum()),
        positives=labels.sum((0,1)).astype(int).tolist(),positive_slates=(labels.sum(1)>0).sum(0).tolist(),
        weight_kish_effective_slates=float(w.sum()**2/np.square(w).sum()))
    with np.load(a.output/'test.npz') as z:days=(z['decision_times']+8*3600000)//86400000
    per_day={}
    for day in np.unique(days):
        use=days==day;points={}
        for arm in ('emer','emer_jev'):
            rows=[comparison_objective_metrics(predictions[f'{arm}/{seed}'][use],labels[use],mask[use],c['directions'],
                top_k=3,sample_weights=w[use])[0] for seed in c['training']['seeds']]
            points[arm]=np.mean(rows,0).tolist()
        per_day[str(np.datetime64(int(day),'D'))]=dict(slates=int(use.sum()),users=len(np.unique(d.users[use])),points=points)
    result=dict(results=report,support=support,per_day=per_day,
        config_sha256=sha256_file(a.config),test_sha256=sha256_file(a.output/'test.npz'),
        response_audit_sha256=sha256_file(a.output/'response_audit.json'),
        limitations=['Logged-exposure pseudo-slates, not original unexposed retrieval sets.',
            'Sampling weights correct stratified inclusion, not exposure propensity.',
            'Intervals condition on three fitted seeds; not a training-population confidence interval.',
            'Aggregate benefit and point guards do not imply all seven population effects are positive.',
            'No online causal uplift or complete production latency claim.'])
    if a.replay:
        with np.load(a.output/'predictions.npz') as saved:
            if set(saved.files)!=set(predictions) or any(not np.array_equal(saved[k],v) for k,v in predictions.items()):raise ValueError('Prediction replay failed')
        if read_json(a.output/'evaluation.json')!=result:raise ValueError('Metric replay failed')
        atomic_write_json(a.output/'replay.json',dict(exact_predictions_and_metrics=True,
            evaluation_sha256=sha256_file(a.output/'evaluation.json'),predictions_sha256=sha256_file(a.output/'predictions.npz'),
            all_raw_responses_audited_again=True))
    else:
        if (a.output/'predictions.npz').exists():raise ValueError('Confirmation exists; use --replay')
        np.savez_compressed(a.output/'predictions.npz',**predictions)
        immutable_json(a.output/'evaluation.json',result)
    print('Frozen confirmation:',report['confirmation'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','development','data','scores','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_multiobjective_confirmation.json'));p.add_argument('--replay',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=4):main(a)
