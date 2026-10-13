#!/usr/bin/env python3
"""Collect corrected training/development signal only after the paired probe."""
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
from probe_jev_longview import QUESTION


def score(a):
    cfg=read_json('configs/emer_jev.json')['jev']
    if not read_json(a.phase/'explicit-longview-validation/result.json')['criterion_passed']:
        raise ValueError('No signal-quality evidence; refuse expanded acquisition')
    if a.split=='train':
        prompt=a.core/'jev_input.jsonl';split='train';data=a.data/'dev-train.npz'
    else:
        prompt=a.data/'jev_input.jsonl';split='test';data=a.data/'test.npz'
    rows=[json.loads(line) for line in prompt.read_text().splitlines()]
    rows=[r for r in rows if r['split']==split]
    identity_file=data if a.split=='train' else a.data/'raw_slates.npz'
    with np.load(identity_file) as z:
        expected=z['candidate_ids'] if a.split=='train' else z['items']
        np.testing.assert_array_equal(expected,np.array([r['candidate_ids'] for r in rows]))
    registration=dict(role='training' if a.split=='train' else 'April22 reused development, not fresh confirmation',
        question=QUESTION,model=cfg['model'],revision=cfg['revision'],rows=len(rows),
        inputs_sha256=sha256_file(prompt),data_sha256=sha256_file(data),
        maximum_attempts=int(len(rows)*1.1),minimum_interval_seconds=1,
        validation_probe_sha256=sha256_file(a.phase/'explicit-longview-validation/result.json'))
    if a.split!='train':registration['candidate_identity_sha256']=sha256_file(identity_file)
    reg=a.output/'registration.json'
    if reg.exists() and read_json(reg)!=registration:raise ValueError('Acquisition registration changed')
    atomic_write_json(reg,registration);cache=DecisionCache(a.output/'responses.sqlite3')
    session=DurableBudgetSession(a.output/'attempts.jsonl',registration['maximum_attempts']);last=0;p=np.zeros((len(rows),10))
    for i,row in enumerate(rows):
        payload=dict(kind='choice',state=row['state'],question=QUESTION,options=row['options'])
        result=cache.get('explicit-longview',cfg['model']+'@'+cfg['revision'],payload)
        if result is None:
            for attempt in range(4):
                delay=1-(time.monotonic()-last)
                if delay>0:time.sleep(delay)
                last=time.monotonic();start=last
                try:
                    response=session.post(cfg['endpoint'],json=payload,timeout=(10,60));response.raise_for_status();result=response.json()
                    validate_choice(result,model=cfg['model'],revision=cfg['revision'],options=row['options'])
                    cache.put('explicit-longview',cfg['model']+'@'+cfg['revision'],payload,result)
                    with (a.output/'latency.jsonl').open('a') as handle:
                        append_jsonl_row(handle,dict(row=i,payload_sha256=canonical_hash(payload),wall_seconds=time.monotonic()-start),flush=True,fsync=True)
                    break
                except (requests.RequestException,json.JSONDecodeError) as exc:
                    status=getattr(getattr(exc,'response',None),'status_code',None)
                    if status is not None and status not in (408,429,500,502,503,504):raise
                    if attempt==3:raise
                    time.sleep(min(30,5*2**attempt))
        p[i]=validate_choice(result,model=cfg['model'],revision=cfg['revision'],options=row['options'])
        if (i+1)%100==0:print(f'{a.split}: {i+1}/{len(rows)} corrected real decisions',flush=True)
    np.savez_compressed(a.output/'scores.npz',probabilities=p,slate_ids=np.array([r['slate_id'] for r in rows]),candidate_ids=np.array([r['candidate_ids'] for r in rows]))
    atomic_write_json(a.output/'complete.json',dict(complete=True,real_decisions=len(rows),http_attempts=session.used,
        scores_sha256=sha256_file(a.output/'scores.npz'),registration=registration))
    print('Corrected signal collection complete:',a.split,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--phase',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','april22'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with _output_lock(a.output):score(a)
