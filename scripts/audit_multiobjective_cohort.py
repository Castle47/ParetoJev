#!/usr/bin/env python3
"""Bounded-memory audit of the frozen50K cohort, including every prompt.

Reads stored raw tensors once, uses chunked record lookup and a single user
index, and memory-maps static metadata. Never fits or evaluates a model.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_core import common_time_features
from paretojev.experiments.comparison import _output_lock
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from prepare_emer_jev import DAY_EVENT


def lookup_occurrences(events,records,chunk_size=1000000):
    """Exact source-row lookup without sorting/copying all source records."""
    output=np.empty(len(records),dtype=events.dtype);seen=np.zeros(len(records),bool);last=None
    for start in range(0,len(events),chunk_size):
        chunk=events[start:start+chunk_size];ids=np.asarray(chunk['record'])
        if not len(ids):continue
        if (np.diff(ids)<=0).any() or (last is not None and ids[0]<=last):
            raise ValueError('Source occurrence order is not strictly increasing')
        last=ids[-1];chosen=np.flatnonzero((records>=ids[0])&(records<=ids[-1]))
        positions=np.searchsorted(ids,records[chosen])
        if (positions>=len(ids)).any() or not np.array_equal(ids[positions],records[chosen]):
            raise ValueError('Unknown source occurrence')
        output[chosen]=chunk[positions];seen[chosen]=True
    if not seen.all():raise ValueError('Source occurrence outside stored range')
    return output


def partition_by_user(events,folder,source_hash,width=512):
    """Sequential source pass into small user buckets; avoids random USB faults."""
    manifest=folder/'manifest.json'
    if manifest.exists():
        saved=read_json(manifest)
        if saved['source_sha256']!=source_hash or saved['width']!=width:raise ValueError('Partition source differs')
        for name,h in saved['files'].items():
            if sha256_file(folder/name)!=h:raise ValueError('Partition bytes differ')
        return sorted(folder.glob('part-*.bin'))
    if folder.exists():
        folder.rename(folder.with_name(folder.name+f'-incomplete-{os.getpid()}'))
    folder.mkdir();handles={};rows=0
    try:
        for start in range(0,len(events),500000):
            chunk=np.array(events[start:start+500000],copy=True)
            bucket=chunk['user']//width
            if (bucket<0).any():raise ValueError('Invalid user ID')
            order=np.argsort(bucket,kind='stable');bucket=bucket[order];chunk=chunk[order]
            boundaries=np.r_[0,np.flatnonzero(np.diff(bucket))+1,len(chunk)]
            for left,right in zip(boundaries[:-1],boundaries[1:]):
                part=int(bucket[left])
                if part not in handles:handles[part]=(folder/f'part-{part:03d}.bin').open('wb')
                chunk[left:right].tofile(handles[part])
            rows+=len(chunk)
            if rows%10000000==0:print(f'Partitioned source rows={rows}',flush=True)
    finally:
        for handle in handles.values():handle.close()
    files=sorted(folder.glob('part-*.bin'))
    atomic_write_json(manifest,dict(source_sha256=source_hash,width=width,rows=rows,files={f.name:sha256_file(f) for f in files}))
    return files


def main(a):
    c=read_json(a.config);prep=read_json(a.data/'preparation.json')
    if prep['config_hash']!=canonical_hash(c) or prep['raw_slates_sha256']!=sha256_file(a.data/'raw_slates.npz'):
        raise ValueError('Cohort registration differs')
    if prep['feature_spec_sha256']!=sha256_file(a.data/'feature_spec.json'):
        raise ValueError('Feature specification changed')
    if sha256_file(a.data/'day-events.bin')!=prep['sources']['sha256']:
        raise ValueError('Source event bytes changed')
    with np.load(a.data/'raw_slates.npz') as z:raw={k:z[k] for k in z.files}
    n,k=raw['items'].shape
    if (n,k)!=(50000,10):raise ValueError('Wrong cohort dimensions')
    day=np.memmap(a.data/'day-events.bin',mode='r',dtype=DAY_EVENT)
    records=raw['records'].ravel();source=lookup_occurrences(day,records).reshape(n,k)
    for target,key in [('items','item'),('exposure_times','time'),('duration','duration')]:
        if not np.array_equal(raw[target],source[key]):raise ValueError('Candidate source differs:'+target)
    if not np.array_equal(source['user'],np.broadcast_to(raw['user'][:,None],(n,k))):raise ValueError('User differs')
    if not np.array_equal((source['labels'][...,None]>>np.arange(7))&1,raw['labels']):raise ValueError('Labels differ')
    if not np.array_equal(raw['decision'],raw['exposure_times'].min(1)):raise ValueError('Common time differs')
    if not ((raw['decision']>=c['fresh_test_start_ms'])&(raw['decision']<c['fresh_test_end_ms'])).all():raise ValueError('Wrong time window')
    if ((raw['exposure_times'].max(1)-raw['decision'])>c['maximum_slate_span_ms']).any():raise ValueError('Slate span differs')
    if not (source['tab']==source['tab'][:,:1]).all():raise ValueError('Mixed scenes')
    if (np.diff(np.sort(raw['items'],axis=1),axis=1)==0).any() or len(np.unique(records))!=len(records):raise ValueError('Duplicate candidates/exposures')
    budgets=np.array(c['test_stratum_budgets']);pop=np.array(prep['populations'])
    np.testing.assert_array_equal(np.bincount(raw['stratum'],minlength=6),budgets)
    np.testing.assert_array_equal(raw['weight'],(pop/budgets)[raw['stratum']])
    print('All500000 source candidates, labels, time windows and weights match',flush=True)
    offsets=np.load(a.root/'behavior-pilot-v1/offsets.npy')
    partitions=partition_by_user(day,a.data/'audit-partitions-v1',prep['sources']['sha256'])
    del day
    metadata=np.array(np.load(a.root/'behavior-v2/basic.npy',mmap_mode='r'),copy=True)
    by_user={}
    for i,u in enumerate(raw['user']):by_user.setdefault(int(u),[]).append(i)
    checked=0;checked_users=0
    with (a.root/'behavior-pilot-v1/events.bin').open('rb') as handle:
        for file in partitions:
            part=np.fromfile(file,dtype=DAY_EVENT)
            part=part[np.argsort(part['user'],kind='stable')]
            users=part['user']
            for user in np.unique(users):
                user=int(user)
                if user not in by_user:continue
                indices=by_user[user]
                handle.seek(int(offsets[user])*EVENT.itemsize)
                early=np.fromfile(handle,dtype=EVENT,count=int(offsets[user+1]-offsets[user]))
                left,right=np.searchsorted(users,[user,user+1]);late_source=part[left:right]
                late=np.empty(len(late_source),dtype=EVENT)
                for key in EVENT.names:late[key]=late_source[key]
                e=np.r_[early[early['time']<c['fresh_test_start_ms']],late];e=e[np.argsort(e['time'],kind='stable')]
                for i in indices:
                    target=np.empty(k,dtype=EVENT)
                    for key in EVENT.names:target[key]=source[key][i]
                    x,prior,t0,_=common_time_features(e,target,metadata,50)
                    np.testing.assert_array_equal(x,raw['x'][i])
                    if len(prior)!=raw['history_count'][i]:raise ValueError('History count differs')
                    tail=prior[-c['jev']['history_length']:]
                    np.testing.assert_array_equal(tail['item'],raw['jev_history'][i,:len(tail)])
                    np.testing.assert_array_equal(tail['time'],raw['jev_history_time'][i,:len(tail)])
                    if not (tail['time']<t0).all():raise ValueError('History leakage')
                    checked+=1
                checked_users+=1
                if checked_users%500==0:print(f'History audit users={checked_users}/{len(by_user)}, slates={checked}',flush=True)
    del metadata,part
    texts=read_json(a.data/'selected_captions.json');prompt_count=0
    with (a.data/'jev_input.jsonl').open() as f:
        for i,line in enumerate(f):
            row=json.loads(line)
            if row['slate_id']!=i or row['split']!='test' or row['questions']!=c['questions'] or row['objective_names']!=c['objectives']:
                raise ValueError('Prompt identity differs')
            np.testing.assert_array_equal(row['candidate_ids'],raw['items'][i])
            state=json.loads(row['state'])
            expected=[(texts[str(item)] or '[text unavailable]')[:c['jev']['history_characters']] for item in raw['jev_history'][i] if item>=0]
            if state['recent_exposures_oldest_to_newest']!=expected:raise ValueError('Prompt history differs')
            if state['context']!={'tab':int(source['tab'][i,0]),'history_semantics':'strictly earlier exposures; feedback is not supplied'}:
                raise ValueError('Context differs')
            if len(state['candidates'])!=k:raise ValueError('Candidate count differs')
            for j,item in enumerate(state['candidates']):
                expected=(texts[str(int(raw['items'][i,j]))] or '[text unavailable]')[:c['jev']['candidate_characters']]
                if item!={'option':chr(65+j),'text':expected,'duration_seconds':round(max(int(raw['duration'][i,j]),0)/1000,1)}:
                    raise ValueError('Candidate prompt differs')
            prompt_count+=1
    if checked!=n or prompt_count!=n:raise ValueError('Incomplete audit')
    if sha256_file(a.data/'jev_input.jsonl')!=prep['prompts']['input_sha256']:raise ValueError('Prompt file changed')
    report=dict(source_candidates=len(records),histories_checked=checked,prompts_checked=prompt_count,
        source_labels_features_and_histories_match=True,weights_verified=True,fresh_time_window_verified=True,
        exact_prompt_alignment=True,raw_sha256=prep['raw_slates_sha256'],input_sha256=prep['prompts']['input_sha256'],
        script_sha256=sha256_file(Path(__file__)),preparation_config_hash=prep['config_hash'])
    atomic_write_json(a.data/'cohort_audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_multiobjective_confirmation.json'))
    a=p.parse_args()
    with _output_lock(a.data/'audit'):main(a)
