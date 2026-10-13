#!/usr/bin/env python3
"""Real paired development probe: clarify the long-view event, change no inputs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from paretojev.jev.cache import DecisionCache
from paretojev.jev.durable_budget import DurableBudgetSession
from paretojev.jev.public_choice import validate_choice
from paretojev.utils.io import read_json,atomic_write_json,append_jsonl_row
from paretojev.utils.provenance import canonical_hash,sha256_file
from paretojev.experiments.comparison import _output_lock
from run_emer_jev import dataset
from audit_jev_increment import pair_statistics,summarize
from paretojev.evaluation.comparison import _contributions

QUESTION=("Which candidate is this user most likely to achieve a long_view on? "
    "Here long_view means watching the entire video when its duration is at most 18 seconds, "
    "or watching at least 18 seconds when its duration exceeds 18 seconds. "
    "Compare the likelihood of meeting each video's own threshold, not the total seconds watched. "
    "A zero or unavailable duration is unknown, not a zero-second completion target.")


def probe(a):
    c=read_json('configs/emer_jev.json');spec=c['jev']
    rows=[json.loads(line) for line in (a.core/'jev_input.jsonl').read_text().splitlines()]
    rows=[row for row in rows if row['split']=='validation']
    d=dataset(a.data/'dev-validation.npz')
    with np.load(a.data/'dev-validation.npz') as z:
        np.testing.assert_array_equal(z['candidate_ids'],np.array([row['candidate_ids'] for row in rows]))
    registration=dict(role='development probe only; no fresh-test claim',changed_field='question only',
        question=QUESTION,model=spec['model'],revision=spec['revision'],rows=len(rows),
        inputs_sha256=sha256_file(a.core/'jev_input.jsonl'),validation_sha256=sha256_file(a.data/'dev-validation.npz'),
        max_http_attempts=440,minimum_interval_seconds=1,bootstrap_samples=2000,
        criterion='new-minus-old long-view NDCG lower paired-user 95% CI >0; report AUC too; do not replace observed labels',
        decision='collect training/new development scores only if this criterion passes')
    path=a.output/'registration.json'
    if path.exists() and read_json(path)!=registration:raise ValueError('Probe design changed')
    atomic_write_json(path,registration)
    cache=DecisionCache(a.output/'responses.sqlite3');session=DurableBudgetSession(a.output/'attempts.jsonl',440)
    p=np.zeros((len(rows),10));last=0;calls=0
    for i,row in enumerate(rows):
        payload=dict(kind='choice',state=row['state'],question=QUESTION,options=row['options'])
        response=cache.get('explicit-longview',spec['model']+'@'+spec['revision'],payload)
        if response is None:
            for attempt in range(4):
                wait=1-(time.monotonic()-last)
                if wait>0:time.sleep(wait)
                last=time.monotonic();start=last
                try:
                    http=session.post(spec['endpoint'],json=payload,timeout=(10,60));http.raise_for_status();response=http.json()
                    validate_choice(response,model=spec['model'],revision=spec['revision'],options=row['options'])
                    cache.put('explicit-longview',spec['model']+'@'+spec['revision'],payload,response);calls+=1
                    with (a.output/'latency.jsonl').open('a') as handle:
                        append_jsonl_row(handle,dict(row=i,payload_sha256=canonical_hash(payload),wall_seconds=time.monotonic()-start),flush=True,fsync=True)
                    break
                except (requests.RequestException,json.JSONDecodeError) as exc:
                    status=getattr(getattr(exc,'response',None),'status_code',None)
                    if status is not None and status not in (408,429,500,502,503,504):raise
                    if attempt==3:raise
                    time.sleep(min(30,5*2**attempt))
        p[i]=validate_choice(response,model=spec['model'],revision=spec['revision'],options=row['options'])
        if (i+1)%50==0:print(f'Explicit definition: {i+1}/{len(rows)} real rows validated',flush=True)
    np.savez_compressed(a.output/'scores.npz',probabilities=p,slate_ids=np.array([r['slate_id'] for r in rows]),
        candidate_ids=np.array([r['candidate_ids'] for r in rows]))
    y=d.arrays['labels'].numpy();mask=d.arrays['mask'].numpy();old=d.arrays['jev_scores'].numpy()[...,1]
    behavior=d.arrays['behavior_scores'].numpy()[...,1]
    stats=[];aucs=[]
    for s in (old,p,behavior):
        num,den,*_=_contributions(s,y,mask,np.array([1]*6+[-1]),3,np.zeros_like(mask));stats.append((num[:,1],den[:,1]))
        auc,eligible,*_=pair_statistics(s,y[...,1]);aucs.append((auc*eligible,eligible))
    ndcg=summarize(stats,d.sample_weights,d.users,2000);auc=summarize(aucs,d.sample_weights,d.users,2000)
    result=dict(role='development probe only',column_order=['old Jev long-view','explicit-definition Jev long-view','behavior long-view'],
        ndcg=ndcg,within_slate_auc=auc,criterion_passed=ndcg['difference_ci'][0]>0,
        real_decisions=len(rows),http_attempts=session.used,new_calls_this_process=calls,
        scores_sha256=sha256_file(a.output/'scores.npz'),labels_unchanged=True,inputs_unchanged=True)
    atomic_write_json(a.output/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with _output_lock(a.output):probe(a)
