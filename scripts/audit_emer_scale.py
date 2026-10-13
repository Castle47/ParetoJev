#!/usr/bin/env python3
"""Audit expanded development sources and report each objective's real support."""
import argparse
import json
from pathlib import Path

import numpy as np

from paretojev.data.kuairand_pilot import EVENT
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file


def audit(a):
    c=read_json(a.config);prep=read_json(a.data/'preparation.json')
    if prep['config_hash']!=canonical_hash(c):raise ValueError('Protocol mismatch')
    for file,digest in [('raw_slates.npz',prep['features']['raw_slates_sha256']),
                        ('jev_input.jsonl',prep['prompts']['input_sha256'])]:
        if sha256_file(a.data/file)!=digest:raise ValueError('Input changed')
    raw=dict(np.load(a.data/'raw_slates.npz'));n=len(raw['user']);k=c['slate_size']
    if len(np.unique(raw['records']))!=n*k:raise ValueError('Duplicate source exposures')
    if (np.diff(np.sort(raw['items'],axis=1),axis=1)==0).any():raise ValueError('Duplicate candidates')
    if not np.array_equal(raw['decision'],raw['exposure_times'].min(1)):raise ValueError('Decision time mismatch')
    known=raw['jev_history']>=0
    if (raw['jev_history_time'][known]>=np.broadcast_to(raw['decision'][:,None],known.shape)[known]).any():
        raise ValueError('Future history')
    store=a.root/'behavior-pilot-v1';offsets=np.load(store/'offsets.npy')
    if sha256_file(store/'offsets.npy')!=read_json(store/'events.json')['offsets_sha256']:raise ValueError('Offsets changed')
    # Read once per user, comparing all 360,000 targets with the canonical store.
    with (store/'events.bin').open('rb') as handle:
        for user in np.unique(raw['user']):
            use=np.flatnonzero(raw['user']==user);records=raw['records'][use]-offsets[user]
            size=int(offsets[user+1]-offsets[user])
            if (records<0).any() or (records>=size).any():raise ValueError('Wrong source user')
            handle.seek(int(offsets[user])*EVENT.itemsize);e=np.fromfile(handle,dtype=EVENT,count=size)
            target=e[records]
            for key,field in [('item','items'),('time','exposure_times'),('duration','duration')]:
                if not np.array_equal(target[key],raw[field][use]):raise ValueError('Source identity mismatch')
            if not np.array_equal((target['labels'][...,None]>>np.arange(7))&1,raw['labels'][use]):raise ValueError('Source label mismatch')
            if not (target['tab']==target['tab'][:,:1]).all():raise ValueError('Mixed tab slate')
    with (a.data/'jev_input.jsonl').open() as handle:
        count=0
        for i,line in enumerate(handle):
            row=json.loads(line);state=json.loads(row['state'])
            if row['slate_id']!=i or row['candidate_ids']!=raw['items'][i].tolist() or row['questions']!=c['questions']:
                raise ValueError('Prompt identity mismatch')
            if row['split']!=c['split_names'][raw['split'][i]] or row['objective_names']!=c['objectives']:
                raise ValueError('Prompt split or goals changed')
            if set(state)!=set(['task','context','recent_exposures_oldest_to_newest','candidates']):
                raise ValueError('Unexpected state fields')
            if set(state['context'])!={'tab','history_semantics'}:raise ValueError('Unexpected context fields')
            if any(set(v)!={'option','text','duration_seconds'} for v in state['candidates']):
                raise ValueError('Unexpected candidate fields')
            count+=1
    if count!=n:raise ValueError('Missing prompt rows')
    report=dict(status='development only; no April23+ test observations loaded',slates=n,
        source_exposures_checked=n*k,unique_source_exposures=True,strict_history=True,
        prompts_checked=count,raw_sha256=sha256_file(a.data/'raw_slates.npz'),splits={})
    populations=np.array(prep['selection']['population_by_split_stratum'])
    for split,name in enumerate(c['split_names']):
        use=raw['split']==split;labels=raw['labels'][use];w=raw['weight'][use];u=raw['user'][use]
        if use.sum()!=sum(c['sample_budgets'][name]):raise ValueError('Insufficient sample')
        if not ((raw['exposure_times'][use]>=c['boundaries_ms'][split]) &
                (raw['exposure_times'][use]<c['boundaries_ms'][split+1])).all():raise ValueError('Split time leakage')
        for h in range(6):
            cell=raw['stratum'][use]==h
            if cell.sum()!=c['sample_budgets'][name][h] or not np.allclose(w[cell],populations[split,h]/cell.sum()):
                raise ValueError('Stratum weight mismatch')
        support={}
        for j,goal in enumerate(c['objectives']):
            has_positive=labels[...,j].sum(1)>0;eligible=has_positive if j<6 else np.ones(len(w),dtype=bool)
            ew=w[eligible]
            support[goal]=dict(positive_labels=int(labels[...,j].sum()),positive_slates=int(has_positive.sum()),
                positive_users=len(np.unique(u[has_positive])),metric_eligible_slates=int(eligible.sum()),
                metric_eligible_users=len(np.unique(u[eligible])),
                kish_effective_eligible_slates=float(ew.sum()**2/(ew@ew)))
        report['splits'][name]=dict(slates=int(use.sum()),users=len(np.unique(u)),candidate_exposures=int(use.sum())*k,
            eligible_population=int(populations[split].sum()),stratum_populations=populations[split].tolist(),
            stratum_samples=c['sample_budgets'][name],support=support)
    atomic_write_json(a.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev_scale.json'));audit(p.parse_args())
