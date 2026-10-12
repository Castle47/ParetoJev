#!/usr/bin/env python3
"""Independent occurrence/prompt and raw-response audits for the core experiment."""
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
from paretojev.data.kuairand_pilot import EVENT
from paretojev.jev.cache import DecisionCache
from paretojev.jev.public_choice import validate_choice
from paretojev.jev.scoring import _categorical_entropy
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file,runtime_provenance


def cohort(a,c):
    data=dict(np.load(a.data/'raw_slates.npz'));prep=read_json(a.data/'preparation.json')
    if prep['config_hash']!=canonical_hash(c) or sha256_file(a.data/'raw_slates.npz')!=prep['features']['raw_slates_sha256']:
        raise ValueError('Changed frozen inputs')
    source=read_json(a.store/'events.json')
    if sha256_file(a.store/'events.bin')!=source['store_sha256']:raise ValueError('Event bytes changed')
    offsets=np.load(a.store/'offsets.npy');users={}
    for i,user in enumerate(data['user']):users.setdefault(int(user),[]).append(i)
    checked=0;all_records=[]
    with (a.store/'events.bin').open('rb') as handle:
        for user,rows in sorted(users.items()):
            handle.seek(int(offsets[user])*EVENT.itemsize)
            e=np.fromfile(handle,dtype=EVENT,count=int(offsets[user+1]-offsets[user]))
            chronological=e[np.argsort(e['time'],kind='stable')]
            for i in rows:
                records=data['records'][i];all_records.extend(records.tolist())
                if not ((records>=offsets[user])&(records<offsets[user+1])).all():raise ValueError('Wrong user occurrence')
                targets=e[records-offsets[user]];t=int(data['decision'][i]);split=int(data['split'][i]);stratum=int(data['stratum'][i])
                if int(targets['time'].min())!=t or not c['boundaries_ms'][split]<=t<c['boundaries_ms'][split+1]:raise ValueError('Bad decision boundary')
                if targets['time'].max()>=c['boundaries_ms'][split+1] or targets['time'].max()-t>c['maximum_slate_span_ms']:raise ValueError('Bad target period/span')
                if len(np.unique(targets['item']))!=c['slate_size'] or len(np.unique(targets['tab']))!=1:raise ValueError('Bad candidate set')
                for key,value in [('items',targets['item']),('exposure_times',targets['time']),('duration',targets['duration'])]:
                    if not np.array_equal(data[key][i],value):raise ValueError('Occurrence values changed')
                labels=np.array([[(int(e['labels'])>>j)&1 for j in range(7)] for e in targets],dtype='uint8')
                if not np.array_equal(labels,data['labels'][i]):raise ValueError('Labels not source observations')
                present=int(np.bitwise_or.reduce(targets['labels']))
                expected=next((j for j,bit in enumerate(c['stratum_priority_bits']) if present&bit),5)
                if expected!=stratum:raise ValueError('Stratum changed')
                counts=prep['selection']['population_by_split_stratum'];sizes=prep['selection']['sampled_by_split_stratum']
                if data['weight'][i]!=counts[split][stratum]/sizes[split][stratum]:raise ValueError('Inclusion weight mismatch')
                prior=chronological[chronological['time']<t];tail=prior[-c['jev_history_length']:]
                valid=data['jev_history'][i]>=0
                if data['history_count'][i]!=len(prior) or not np.array_equal(data['jev_history'][i,valid],tail['item']) or not np.array_equal(data['jev_history_time'][i,valid],tail['time']):raise ValueError('Jev history boundary mismatch')
                if not np.allclose(data['x'][i,:,11],min(len(prior),50)/50):raise ValueError('Behavior history boundary mismatch')
                shared=[0,3,4,5,6,7,8,9,10,11,12,13,15,16,17,25]
                if not np.array_equal(data['x'][i][:,shared],np.repeat(data['x'][i,:1,shared].reshape(1,-1),c['slate_size'],axis=0)):
                    raise ValueError('Candidate-specific future request/history features')
                checked+=1
    if len(set(all_records))!=len(all_records):raise ValueError('An outcome occurs in multiple sampled slates')
    rows=[json.loads(line) for line in (a.data/'jev_input.jsonl').read_text().splitlines()]
    if sha256_file(a.data/'jev_input.jsonl')!=prep['prompts']['input_sha256']:raise ValueError('Prompts changed')
    for i,row in enumerate(rows):
        state=json.loads(row['state'])
        if set(state)!=set(['task','context','recent_exposures_oldest_to_newest','candidates']):raise ValueError('Unexpected fields sent to Jev')
        if set(state['context'])!=set(['tab','history_semantics']):raise ValueError('Unexpected Jev context')
        if not all(isinstance(x,str) for x in state['recent_exposures_oldest_to_newest']):raise ValueError('History contains non-text fields')
        if any(set(candidate)!=set(['option','text','duration_seconds']) for candidate in state['candidates']):raise ValueError('Unexpected candidate fields')
        if row['candidate_ids']!=data['items'][i].tolist():raise ValueError('Prompt candidate order changed')
        if [x['option'] for x in state['candidates']]!=row['options']:raise ValueError('Choice aliases permuted')
    atomic_write_json(a.output,dict(all_source_occurrences_labels_and_weights_match=True,
        all_candidate_and_history_boundaries_match=True,outcome_fields_sent=False,
        slates_checked=checked,distinct_source_records=len(all_records),users=len(users),provenance=runtime_provenance()))


def scores(a,c):
    complete=read_json(a.scores/'complete.json')
    if not complete['complete'] or sha256_file(a.scores/'scores.npz')!=complete['scores_sha256']:raise ValueError('Incomplete/changed scores')
    data=dict(np.load(a.scores/'scores.npz'));spec=c['jev'];model=spec['model']+'@'+spec['revision']
    rows=[json.loads(line) for line in (a.data/'jev_input.jsonl').read_text().splitlines()]
    prep=read_json(a.data/'preparation.json')
    if complete['registration']['input_sha256']!=prep['prompts']['input_sha256'] or sha256_file(a.data/'jev_input.jsonl')!=prep['prompts']['input_sha256']:raise ValueError('Prompt checksum mismatch')
    checked=0
    with sqlite3.connect(f'file:{(a.scores/"responses.sqlite3").resolve()}?mode=ro',uri=True) as conn:
        for i,row in enumerate(rows):
            for j,question in enumerate(row['questions']):
                payload=dict(kind='choice',state=row['state'],question=question,options=row['options'])
                key=DecisionCache.key_for('autotrust-public-choice',model,payload)
                saved=conn.execute('select request_json,response_json,provider,model from decisions where cache_key=?',(key,)).fetchone()
                if saved is None or json.loads(saved[0])!=payload or saved[2:]!=('autotrust-public-choice',model):raise ValueError('Missing/mismatched raw decision')
                response=json.loads(saved[1]);p=validate_choice(response,model=spec['model'],revision=spec['revision'],options=row['options'])
                h=_categorical_entropy(p)
                if not np.array_equal(data['probabilities'][i,:,j],p) or not np.all(data['entropy'][i,:,j]==h) or not np.all(data['confidence'][i,:,j]==1-h):raise ValueError('Tensor differs from raw decision')
                checked+=1
    attempts=[json.loads(line) for line in (a.scores/'attempts.jsonl').read_text().splitlines()]
    if [r['attempt'] for r in attempts]!=list(range(1,len(attempts)+1)) or len(attempts)>spec['max_http_calls']:raise ValueError('Lifetime ledger invalid')
    atomic_write_json(a.output,dict(all_scores_match_validated_raw_responses=True,decisions_checked=checked,
        lifetime_http_attempts=len(attempts),model=spec['model'],revision=spec['revision'],
        score_semantics='categorical relative Choice, not event probability',provenance=runtime_provenance()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/kuairand_core.json'));p.add_argument('--store',type=Path)
    p.add_argument('--scores',type=Path);p.add_argument('--mode',choices=['cohort','scores'],required=True);a=p.parse_args()
    (cohort if a.mode=='cohort' else scores)(a,read_json(a.config))
