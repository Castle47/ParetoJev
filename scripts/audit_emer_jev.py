#!/usr/bin/env python3
"""Audit frozen inputs, selected checkpoints and raw Choice responses."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_core import common_time_features
from paretojev.jev.cache import DecisionCache
from paretojev.jev.public_choice import validate_choice
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file
from prepare_emer_jev import DAY_EVENT
from prepare_kuairand_core import QUESTIONS
from run_emer_jev import dataset,loader,load_model,validation
from paretojev.models.emer import EMERConfig,EMERReranker
from paretojev.utils.seed import seed_everything


def audit(a):
    out=a.data;c=read_json(a.config);prep=read_json(out/'preparation.json')
    if prep['config_hash']!=canonical_hash(c) or prep['raw_slates_sha256']!=sha256_file(out/'raw_slates.npz'):
        raise ValueError('Preparation identity mismatch')
    if prep['feature_spec_sha256']!=sha256_file(out/'feature_spec.json'):raise ValueError('Feature spec changed')
    day=np.array(np.memmap(out/'day-events.bin',mode='r',dtype=DAY_EVENT),copy=True)
    if sha256_file(out/'day-events.bin')!=prep['sources']['sha256']:raise ValueError('Source day changed')
    raw=np.load(out/'raw_slates.npz');records=raw['records'].ravel()
    ordered=np.argsort(day['record']);sorted_records=day['record'][ordered]
    locations=np.searchsorted(sorted_records,records)
    if not np.array_equal(sorted_records[locations],records):raise ValueError('Unknown source record')
    source=day[ordered[locations]].reshape(raw['items'].shape)
    for target,key in [('items','item'),('user','user'),('exposure_times','time'),('duration','duration')]:
        expected=raw[target] if target!='user' else np.repeat(raw['user'][:,None],c['slate_size'],1)
        if not np.array_equal(source[key],expected):raise ValueError('Source occurrence differs: '+target)
    if not np.array_equal((source['labels'][...,None]>>np.arange(7))&1,raw['labels']):raise ValueError('Real labels differ')
    if not ((raw['decision']>=c['fresh_test_start_ms'])&(raw['decision']<c['fresh_test_end_ms'])).all():
        raise ValueError('Fresh date differs')
    if not np.array_equal(raw['decision'],raw['exposure_times'].min(1)):raise ValueError('Common time differs')
    if (raw['exposure_times'].max(1)-raw['decision']>c['maximum_slate_span_ms']).any():raise ValueError('Span differs')
    if not (source['tab']==source['tab'][:,:1]).all():raise ValueError('Mixed scenes')
    if (np.diff(np.sort(raw['items'],axis=1),axis=1)==0).any() or len(np.unique(records))!=len(records):
        raise ValueError('Repeated candidate/source exposure')
    budgets=np.array(c['test_stratum_budgets']);pop=np.array(prep['populations'])
    np.testing.assert_array_equal(np.bincount(raw['stratum'],minlength=6),budgets)
    np.testing.assert_allclose(raw['weight'],(pop/budgets)[raw['stratum']],rtol=0,atol=0)
    offsets=np.load(a.root/'behavior-pilot-v1/offsets.npy');metadata=np.array(np.load(a.root/'behavior-v2/basic.npy',mmap_mode='r'),copy=True)
    by_user={}
    for i,user in enumerate(raw['user']):by_user.setdefault(int(user),[]).append(i)
    # One vectorized user index bounds late-history scans instead of quadratic I/O.
    day_order=np.argsort(day['user'],kind='stable');users=day['user'][day_order]
    with (a.root/'behavior-pilot-v1/events.bin').open('rb') as handle:
        for user,indices in by_user.items():
            handle.seek(int(offsets[user])*EVENT.itemsize)
            early=np.fromfile(handle,dtype=EVENT,count=int(offsets[user+1]-offsets[user]))
            left,right=np.searchsorted(users,[user,user+1]);late_source=day[day_order[left:right]]
            late=np.empty(len(late_source),dtype=EVENT)
            for key in EVENT.names:late[key]=late_source[key]
            e=np.r_[early[early['time']<c['fresh_test_start_ms']],late];e=e[np.argsort(e['time'],kind='stable')]
            for i in indices:
                target=np.empty(c['slate_size'],dtype=EVENT)
                for key in EVENT.names:target[key]=source[key][i]
                x,prior,t0,_=common_time_features(e,target,metadata,50)
                np.testing.assert_array_equal(x,raw['x'][i]);assert len(prior)==raw['history_count'][i]
                tail=prior[-c['jev']['history_length']:]
                np.testing.assert_array_equal(tail['item'],raw['jev_history'][i,:len(tail)])
                np.testing.assert_array_equal(tail['time'],raw['jev_history_time'][i,:len(tail)])
                if not (tail['time']<t0).all():raise ValueError('History leakage')
    if a.cohort_only:
        report=dict(source_candidates=len(records),histories_checked=len(raw['items']),
            source_labels_features_and_histories_match=True,weights_verified=True,
            fresh_time_window_verified=True)
        atomic_write_json(out/'cohort_audit.json',report);print(report,flush=True);return
    complete=read_json(out/'jev/complete.json')
    if complete['registration']['config_hash']!=canonical_hash(c) or complete['scores_sha256']!=sha256_file(out/'jev/scores.npz'):
        raise ValueError('Score registration mismatch')
    jev=np.load(out/'jev/scores.npz');cache=DecisionCache(out/'jev/responses.sqlite3')
    prompts=[json.loads(line) for line in (out/'jev_input.jsonl').read_text().splitlines()]
    texts=read_json(out/'selected_captions.json');times=[]
    for i,row in enumerate(prompts):
        np.testing.assert_array_equal(row['candidate_ids'],raw['items'][i])
        state=json.loads(row['state'])
        expected_history=[(texts[str(item)] or '[text unavailable]')[:c['jev']['history_characters']]
            for item in raw['jev_history'][i] if item>=0]
        if state['recent_exposures_oldest_to_newest']!=expected_history or row['questions']!=QUESTIONS:
            raise ValueError('History text or questions changed')
        for j,candidate in enumerate(state['candidates']):
            expected=(texts[str(int(raw['items'][i,j]))] or '[text unavailable]')[:c['jev']['candidate_characters']]
            if candidate['text']!=expected or candidate['duration_seconds']!=round(max(int(raw['duration'][i,j]),0)/1000,1):
                raise ValueError('Candidate text/duration changed')
        if state['context']!={'tab':int(source['tab'][i,0]),'history_semantics':'strictly earlier exposures; feedback is not supplied'}:
            raise ValueError('Unexpected context fields')
        for j,question in enumerate(row['questions']):
            payload=dict(kind='choice',state=row['state'],question=question,options=row['options'])
            response=cache.get('autotrust-public-choice',c['jev']['model']+'@'+c['jev']['revision'],payload)
            p=np.array(validate_choice(response,model=c['jev']['model'],revision=c['jev']['revision'],options=row['options']))
            np.testing.assert_allclose(p,jev['probabilities'][i,:,j],rtol=0,atol=0)
            clipped=np.clip(p,1e-12,1);clipped=clipped/clipped.sum()
            entropy=-np.sum(clipped*np.log(clipped))/np.log(len(p))
            np.testing.assert_allclose(jev['entropy'][i,:,j],entropy,atol=1e-14,rtol=0)
            np.testing.assert_allclose(jev['confidence'][i,:,j],1-entropy,atol=1e-14,rtol=0)
            times.append(response['elapsed_seconds'])
    selection=read_json(out/'selection.json');validation_data=dataset(out/'dev-validation.npz');checked=0
    expected_trials=2*len(c['training']['seeds'])*len(c['training']['learning_rates'])*len(c['training']['prior_coefficients'])
    if len(selection['trials'])!=expected_trials:raise ValueError('Unequal or incomplete search')
    initial={}
    for seed in c['training']['seeds']:
        for use_jev in (False,True):
            seed_everything(seed);model=EMERReranker(EMERConfig(**c['model'],use_jev=use_jev))
            initial[(seed,use_jev)]=validation(model,validation_data,c)[0]
    for trial in selection['trials']:
        if len(trial['history'])!=c['training']['epochs'] or trial['checkpoint_sha256']!=sha256_file(out/trial['path']):
            raise ValueError('Training receipt or checkpoint changed')
        candidates=[(initial[(trial['seed'],trial['use_jev'])],0)]+[(r['validation'],r['epoch']) for r in trial['history']]
        best_score,best_epoch=max(candidates,key=lambda x:x[0])
        if best_score!=trial['validation'] or best_epoch!=trial['best_epoch']:
            raise ValueError('Checkpoint was not earliest best validation epoch')
    for seed,arms in selection['selected'].items():
        for arm,spec in arms.items():
            eligible=[row for row in selection['trials'] if row['seed']==int(seed) and row['use_jev']==(arm=='emer_jev')]
            if max(eligible,key=lambda row:row['validation'])['path']!=spec['path']:
                raise ValueError('Selected configuration was not validation best')
    for arms in selection['selected'].values():
        for arm,spec in arms.items():
            path=out/spec['path']
            if sha256_file(path)!=spec['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            model=load_model(path);score,_=validation(model,validation_data,c)
            if score!=spec['validation']:raise ValueError('Selection validation replay differs')
            b=next(iter(loader(validation_data,32)))
            with torch.no_grad():
                expected=model(b)['score_mean']
                changed=replace(b,labels=1-b.labels)
                np.testing.assert_array_equal(model(changed)['score_mean'].numpy(),expected.numpy())
                if arm=='emer':
                    changed=replace(b,jev_scores=1-b.jev_scores,jev_confidence=1-b.jev_confidence,jev_entropy=1-b.jev_entropy)
                    np.testing.assert_array_equal(model(changed)['score_mean'].numpy(),expected.numpy())
            checked+=1
    http=[json.loads(line) for line in (out/'jev/http_latency.jsonl').read_text().splitlines()]
    elapsed_by_payload={}
    for row in http:
        key=row['payload_sha256'];elapsed_by_payload[key]=elapsed_by_payload.get(key,0)+row['wall_seconds']
    serial=np.array([sum(elapsed_by_payload[canonical_hash(dict(kind='choice',state=row['state'],question=q,options=row['options']))]
        for q in row['questions']) for row in prompts])
    report=dict(source_candidates=len(records),histories_checked=len(prompts),real_jev_decisions=len(times),
        selected_models_checked=checked,training_receipts_checked=expected_trials,earliest_validation_selection_verified=True,
        source_labels_features_and_histories_match=True,
        no_jev_checkpoint_invariant=True,inference_labels_invariant=True,
        service_inference_seconds={key:float(np.quantile(times,q)) for key,q in [('p50',.5),('p95',.95),('p99',.99)]},
        serial_seven_http_seconds_excluding_pacing={key:float(np.quantile(serial,q)) for key,q in [('p50',.5),('p95',.95),('p99',.99)]},
        http_attempts=len(http),complete_recommendation_rt_measured=False)
    atomic_write_json(out/'audit.json',report);print(report,flush=True)


def audit_metrics(a):
    """Scalar reference formula independent of evaluator sufficient statistics."""
    out=a.data;c=read_json(a.config);test=np.load(out/'test.npz');pred=np.load(out/'predictions.npz')
    expected=read_json(out/'evaluation.json')['results'];points={}
    k=c['evaluation']['top_k'];discount=1/np.log2(np.arange(2,k+2));labels=test['labels'];weights=test['slate_weights']
    for key in pred.files:
        numerator=np.zeros(7);denominator=np.zeros(7)
        for scores,row,weight in zip(pred[key],labels,weights):
            order=np.argsort(-scores,kind='stable')[:k]
            for j in range(6):
                ideal=np.dot(np.sort(row[:,j])[::-1][:k],discount)
                if ideal>0:
                    numerator[j]+=weight*np.dot(row[order,j],discount)/ideal;denominator[j]+=weight
            numerator[6]+=weight*np.sum(1-row[order,6]);denominator[6]+=weight*k
        points[key]=numerator/denominator
    for seed,row in expected['per_seed'].items():
        for arm in ('emer','emer_jev'):np.testing.assert_allclose(points[f'{arm}/{seed}'],row[arm],rtol=0,atol=1e-12)
    for j,name in enumerate(c['objectives']):
        base=np.mean([points[f'emer/{seed}'][j] for seed in c['training']['seeds']])
        full=np.mean([points[f'emer_jev/{seed}'][j] for seed in c['training']['seeds']])
        np.testing.assert_allclose([base,full,full-base],[expected['objectives'][name][k] for k in
            ('emer','emer_jev','absolute_gain')],rtol=0,atol=1e-12)
    atomic_write_json(out/'metric_audit.json',dict(independent_scalar_metrics_match=True,
        checked_models=len(points),maximum_absolute_tolerance=1e-12))
    print('Independent scalar metric audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev.json'));p.add_argument('--metrics-only',action='store_true')
    p.add_argument('--cohort-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    with torch.no_grad():audit_metrics(a) if a.metrics_only else audit(a)
