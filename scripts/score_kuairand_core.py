#!/usr/bin/env python3
"""Durable bounded one-at-a-time real Choice scoring; no outcome evaluation."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import requests
import numpy as np
from paretojev.jev.cache import DecisionCache
from paretojev.jev.durable_budget import DurableBudgetSession
from paretojev.jev.public_choice import validate_choice
from paretojev.jev.scoring import _categorical_entropy
from paretojev.utils.io import read_json,atomic_write_json,append_jsonl_row
from paretojev.utils.provenance import canonical_hash,sha256_file,runtime_provenance


def score(a):
    c=read_json(a.config);spec=c['jev'];prep=read_json(a.data/'preparation.json')
    if prep['config_hash']!=canonical_hash(c) or sha256_file(a.data/'jev_input.jsonl')!=prep['prompts']['input_sha256']:
        raise ValueError('Frozen prompt protocol changed')
    rows=[json.loads(line) for line in (a.data/'jev_input.jsonl').read_text().splitlines()]
    registration=dict(config_hash=canonical_hash(c),input_sha256=prep['prompts']['input_sha256'],
        planned_calls=len(rows)*len(c['objectives']),service=spec,
        source_sha256=sha256_file(Path(__file__)),validator_sha256=sha256_file(Path('src/paretojev/jev/public_choice.py')))
    if (a.output/'registration.json').exists():
        if read_json(a.output/'registration.json')['registration']!=registration: raise ValueError('Scoring registration changed')
    else: atomic_write_json(a.output/'registration.json',dict(registration=registration,provenance=runtime_provenance()))
    cache=DecisionCache(a.output/'responses.sqlite3');session=DurableBudgetSession(a.output/'attempts.jsonl',spec['max_http_calls'])
    provider='autotrust-public-choice';model=spec['model']+'@'+spec['revision'];last=0.
    probabilities=np.zeros((len(rows),c['slate_size'],len(c['objectives'])),dtype='float64')
    entropy=np.zeros_like(probabilities);hits=0;successes=0;start=time.monotonic()
    for i,row in enumerate(rows):
        if row['slate_id']!=i or row['objective_names']!=c['objectives']:raise ValueError('Prompt row identity changed')
        for j,question in enumerate(row['questions']):
            payload=dict(kind='choice',state=row['state'],question=question,options=row['options'])
            result=cache.get(provider,model,payload)
            if result is not None:hits+=1
            else:
                for attempt in range(7):
                    delay=spec['minimum_interval_seconds']-(time.monotonic()-last)
                    if delay>0:time.sleep(delay)
                    last=time.monotonic()
                    try:
                        response=session.post(spec['endpoint'],json=payload,timeout=(10,60))
                        response.raise_for_status();result=response.json()
                        validate_choice(result,model=spec['model'],revision=spec['revision'],options=row['options'])
                        cache.put(provider,model,payload,result);break
                    except (requests.RequestException,json.JSONDecodeError) as exc:
                        status=getattr(getattr(exc,'response',None),'status_code',None)
                        with (a.output/'transport_errors.jsonl').open('a') as handle:
                            append_jsonl_row(handle,dict(slate=i,objective=j,attempt=session.used,status=status,
                                error=str(exc),payload_sha256=canonical_hash(payload)),flush=True,fsync=True)
                        if status is not None and status not in (408,429,500,502,503,504):raise
                        if attempt==6:raise
                        delay=min(60,5*2**attempt)
                        print(f'Transport retry slate={i} objective={j} status={status} backoff={delay}s',flush=True)
                        time.sleep(delay)
            column=validate_choice(result,model=spec['model'],revision=spec['revision'],options=row['options'])
            probabilities[i,:,j]=column;entropy[i,:,j]=_categorical_entropy(column);successes+=1
            if successes%100==0:
                progress=dict(validated_decisions=successes,planned_decisions=registration['planned_calls'],
                    lifetime_attempts=session.used,cache_hits_this_process=hits,elapsed_seconds=time.monotonic()-start,
                    complete=False,provenance='real JEV Choice; no labels loaded')
                atomic_write_json(a.output/'progress.json',progress);print(progress,flush=True)
    np.savez_compressed(a.output/'scores.npz',probabilities=probabilities,entropy=entropy,confidence=1-entropy,
        slate_ids=np.arange(len(rows)),candidate_ids=np.array([r['candidate_ids'] for r in rows]),
        objective_names=np.array(c['objectives']),model=np.array(spec['model']),revision=np.array(spec['revision']))
    atomic_write_json(a.output/'complete.json',dict(registration=registration,
        scores_sha256=sha256_file(a.output/'scores.npz'),valid_decisions=successes,lifetime_attempts=session.used,
        cached_payloads=cache.count(),complete=True,score_semantics='within-slate categorical choice, not pXTR'))
    print('All real Choice decisions complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/kuairand_core.json'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);score(a)
