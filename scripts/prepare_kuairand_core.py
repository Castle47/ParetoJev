#!/usr/bin/env python3
"""Select a registered real cohort and prepare outcome-free Jev Choice requests."""
import argparse
import hashlib
import heapq
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_27k import ITEM_COUNT, HashReader
from paretojev.data.kuairand_features_v2 import CORE_FIELDS, EXTRA_FIELDS
from paretojev.data.kuairand_core import eligible_blocks, strata_for, common_time_features
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file, canonical_hash, runtime_provenance

QUESTIONS = [
 'Which candidate is this user most likely to click or validly play in this feed?',
 'Which candidate is this user most likely to watch for a long view?',
 'Which candidate is this user most likely to explicitly like?',
 'Which candidate is this user most likely to follow the creator after viewing?',
 'Which candidate is this user most likely to comment on?',
 'Which candidate is this user most likely to forward or share?',
 'Which candidate is this user most likely to give explicit negative feedback such as dislike or not interested?'
]


def select(root, output, config):
    store=root/'behavior-pilot-v1'; source=read_json(store/'events.json')
    if sha256_file(store/'offsets.npy') != source['offsets_sha256']: raise ValueError('Offsets changed')
    splits=len(config['split_names'])
    if len(config['boundaries_ms']) != splits+1 or np.any(np.diff(config['boundaries_ms'])<=0):
        raise ValueError('Strict chronological split boundaries required')
    if any(len(config['sample_budgets'][name])!=6 or min(config['sample_budgets'][name])<=0 for name in config['split_names']):
        raise ValueError('Six positive stratum budgets required per split')
    offsets=np.load(store/'offsets.npy'); counts=np.zeros((splits,6),dtype='int64')
    heaps=[[[] for _ in range(6)] for _ in range(splits)]
    popularity=np.zeros(ITEM_COUNT,dtype='uint32'); digest=hashlib.sha256()
    with (store/'events.bin').open('rb') as handle:
        for user in range(len(offsets)-1):
            size=int(offsets[user+1]-offsets[user])*EVENT.itemsize; raw=handle.read(size)
            if len(raw)!=size: raise ValueError('Truncated source')
            digest.update(raw); events=np.frombuffer(raw,dtype=EVENT)
            if not len(events): continue
            order=np.argsort(events['time'],kind='stable'); events=events[order]
            np.add.at(popularity,events['item'][events['time']<config['boundaries_ms'][0]],1)
            for split,name in enumerate(config['split_names']):
                blocks=eligible_blocks(events,*config['boundaries_ms'][split:split+2],config['slate_size'],config['maximum_slate_span_ms'])
                strata=strata_for(events,blocks,config['stratum_priority_bits'])
                rng=np.random.default_rng(np.random.SeedSequence([config['seed'],user,split]))
                priorities=rng.random(len(blocks))
                for block,stratum,priority in zip(blocks,strata,priorities):
                    counts[split,stratum]+=1; heap=heaps[split][stratum]
                    limit=config['sample_budgets'][name][int(stratum)]
                    record=int(offsets[user]+order[block[0]])
                    if len(heap)<limit or priority>heap[0][0]:
                        payload=(user,split,int(stratum),int(events['time'][block[0]]),(offsets[user]+order[block]).tolist())
                        entry=(float(priority),record,payload)
                        if len(heap)<limit: heapq.heappush(heap,entry)
                        else: heapq.heapreplace(heap,entry)
            if user%5000==0: print(f'Population scan user {user}',flush=True)
        if handle.read(1): raise ValueError('Unaccounted event records')
    if digest.hexdigest()!=source['store_sha256']: raise ValueError('Full event hash mismatch')
    rows=[]; sampled=np.zeros_like(counts)
    for split in range(splits):
        for stratum in range(6):
            selected=heaps[split][stratum]; sampled[split,stratum]=len(selected)
            for _,_,payload in selected: rows.append((*payload,counts[split,stratum]/len(selected)))
    rows.sort(key=lambda r:(r[1],r[3],r[0],r[4][0]))
    arrays=dict(user=np.array([r[0] for r in rows],dtype='int32'),split=np.array([r[1] for r in rows],dtype='int8'),
        stratum=np.array([r[2] for r in rows],dtype='int8'),decision=np.array([r[3] for r in rows],dtype='int64'),
        records=np.array([r[4] for r in rows],dtype='int64'),weight=np.array([r[5] for r in rows],dtype='float64'))
    np.savez_compressed(output/'selection.npz',**arrays); np.save(output/'prior-popularity.npy',popularity)
    report=dict(config_hash=canonical_hash(config),event_store_sha256=digest.hexdigest(),
        selection_sha256=sha256_file(output/'selection.npz'),popularity_sha256=sha256_file(output/'prior-popularity.npy'),
        population_by_split_stratum=counts.tolist(),sampled_by_split_stratum=sampled.tolist(),
        selected_slates=len(rows),source_record_unit='zero_based_EVENT_records',
        source_manifest_sha256=sha256_file(store/'events.json'),provenance=runtime_provenance())
    atomic_write_json(output/'selection.json',report)
    print('Selected',len(rows),'slates; population cells',counts.tolist(),flush=True)


def features(root,output,config,metadata_path):
    registration=read_json(output/'selection.json')
    if registration['config_hash']!=canonical_hash(config) or sha256_file(output/'selection.npz')!=registration['selection_sha256']:
        raise ValueError('Selection changed')
    metadata_manifest=read_json(root/'behavior-v2/metadata.json')
    if sha256_file(metadata_path)!=metadata_manifest['index_sha256']: raise ValueError('Metadata changed')
    metadata=np.load(metadata_path,mmap_mode='r')
    if config.get('metadata_in_memory',False): metadata=np.array(metadata,copy=True)
    store=root/'behavior-pilot-v1'
    offsets=np.load(store/'offsets.npy'); selected=dict(np.load(output/'selection.npz'))
    n=len(selected['user']); k=config['slate_size']; h=config['jev_history_length']
    x=np.zeros((n,k,26),dtype='float32'); labels=np.zeros((n,k,7),dtype='uint8')
    items=np.zeros((n,k),dtype='int32'); times=np.zeros((n,k),dtype='int64')
    history=np.full((n,h),-1,dtype='int32'); history_time=np.full((n,h),-1,dtype='int64')
    history_count=np.zeros(n,dtype='int64'); duration=np.zeros((n,k),dtype='int64')
    by_user={}
    for i,user in enumerate(selected['user']): by_user.setdefault(int(user),[]).append(i)
    with (store/'events.bin').open('rb') as handle:
        for number,(user,indices) in enumerate(sorted(by_user.items())):
            handle.seek(int(offsets[user])*EVENT.itemsize)
            original=np.fromfile(handle,dtype=EVENT,count=int(offsets[user+1]-offsets[user]))
            events=original[np.argsort(original['time'],kind='stable')]
            for i in indices:
                targets=original[selected['records'][i]-offsets[user]]
                rng=np.random.default_rng(np.random.SeedSequence([config['seed'],user,int(selected['split'][i]),int(selected['records'][i,0])]))
                order=rng.permutation(k); targets=targets[order]; selected['records'][i]=selected['records'][i,order]
                x[i],prior,t0,_=common_time_features(events,targets,metadata,config['history_length'])
                if t0!=selected['decision'][i]: raise ValueError('Decision mismatch')
                labels[i]=(targets['labels'][:,None]>>np.arange(7))&1
                items[i]=targets['item']; times[i]=targets['time']; duration[i]=targets['duration']
                history_count[i]=len(prior); tail=prior[-h:]
                history[i,:len(tail)]=tail['item']; history_time[i,:len(tail)]=tail['time']
                if len(tail) and not (tail['time']<t0).all(): raise ValueError('History leakage')
            if number%250==0: print(f'Common-time features user group {number}/{len(by_user)}',flush=True)
    popularity=np.load(output/'prior-popularity.npy',mmap_mode='r')[items]
    arrays=dict(**selected,x=x,labels=labels,items=items,exposure_times=times,duration=duration,
        jev_history=history,jev_history_time=history_time,history_count=history_count,popularity=popularity)
    np.savez_compressed(output/'raw_slates.npz',**arrays)
    atomic_write_json(output/'features.json',dict(raw_slates_sha256=sha256_file(output/'raw_slates.npz'),
        fields=CORE_FIELDS+EXTRA_FIELDS,metadata_index_sha256=metadata_manifest['index_sha256'],
        all_candidates_share_decision_time=True,all_histories_strictly_prior=True,
        total_slates=n,selected_users=len(by_user),
        positives_by_split={name:labels[selected['split']==i].sum((0,1)).tolist() for i,name in enumerate(config['split_names'])},
        weighted_represented_slates_by_split={name:float(selected['weight'][selected['split']==i].sum()) for i,name in enumerate(config['split_names'])}))


def captions(root,output):
    raw=dict(np.load(output/'raw_slates.npz')); needed=np.zeros(ITEM_COUNT,dtype=bool)
    ids=np.unique(np.r_[raw['items'].ravel(),raw['jev_history'].ravel()]); ids=ids[ids>=0]; needed[ids]=True
    source=root/'downloads/kuairand_video_captions.csv'; expected=read_json(root/'download_manifest.json')['assets'][source.name]['sha256']
    texts={}; count=0
    with source.open('rb') as handle:
        tracker=HashReader(handle)
        with io.BufferedReader(tracker) as buffered:
            for frame in pd.read_csv(buffered,usecols=['final_video_id','caption','show_cover_text'],
                dtype={'final_video_id':'int64','caption':'string','show_cover_text':'string'},keep_default_na=False,chunksize=100000):
                keep=frame.loc[needed[frame.final_video_id.to_numpy()]]
                for row in keep.itertuples(index=False):
                    text=' '.join(str(v).strip() for v in (row.caption,row.show_cover_text) if str(v).strip())
                    texts[str(int(row.final_video_id))]=re.sub(r'\s+',' ',text).strip()
                count+=len(frame)
                if count%5000000==0:print(f'Caption join rows {count}, matched {len(texts)}/{len(ids)}',flush=True)
    if tracker.digest.hexdigest()!=expected: raise ValueError('Caption source changed')
    if len(texts)!=len(ids): raise ValueError('Missing catalog rows')
    atomic_write_json(output/'selected_captions.json',texts)
    atomic_write_json(output/'captions.json',dict(source_sha256=expected,selected_text_sha256=sha256_file(output/'selected_captions.json'),
        selected_distinct_items=len(ids),nonblank_texts=sum(bool(v) for v in texts.values()),rows_scanned=count))


def prompts(output,config):
    raw=dict(np.load(output/'raw_slates.npz')); texts=read_json(output/'selected_captions.json')
    aliases=[chr(65+j) for j in range(config['slate_size'])]
    max_state=0; missing=0; truncated=0
    def text(item,limit):
        nonlocal missing,truncated
        value=texts[str(int(item))]
        if not value: missing+=1; return '[text unavailable]'
        if len(value)>limit:truncated+=1
        return value[:limit]
    with (output/'jev_input.jsonl').open('w') as handle:
        for i in range(len(raw['user'])):
            history=[text(item,config['jev']['history_text_characters']) for item in raw['jev_history'][i] if item>=0]
            candidates=[dict(option=aliases[j],text=text(item,config['jev']['candidate_text_characters']),
                duration_seconds=round(max(int(raw['duration'][i,j]),0)/1000,1)) for j,item in enumerate(raw['items'][i])]
            state=dict(task='short-video recommendation',context={'tab':int(raw['x'][i,0,0]),
                'history_semantics':'strictly earlier exposures; feedback is not supplied'},
                recent_exposures_oldest_to_newest=history,candidates=candidates)
            state=json.dumps(state,ensure_ascii=False,separators=(',',':'));max_state=max(max_state,len(state))
            if len(state)>6000:raise ValueError('Registered prompt exceeds service limit; no sample substitution')
            row=dict(slate_id=i,split=config['split_names'][int(raw['split'][i])],candidate_ids=raw['items'][i].tolist(),
                state=state,options=aliases,questions=config.get('questions',QUESTIONS),objective_names=config['objectives'])
            handle.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    atomic_write_json(output/'prompts.json',dict(input_sha256=sha256_file(output/'jev_input.jsonl'),
        max_state_characters=max_state,missing_text_occurrences=missing,truncated_text_occurrences=truncated,
        outcome_fields_sent=False,score_semantics='per-objective relative Choice distribution, not event probability'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/kuairand_core.json'))
    a=p.parse_args(); c=read_json(a.config);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'prompts.json').exists(): raise ValueError('Completed cohort is immutable')
    if not (a.output/'selection.json').exists():select(a.root,a.output,c)
    if not (a.output/'features.json').exists():features(a.root,a.output,c,a.metadata)
    if not (a.output/'captions.json').exists():captions(a.root,a.output)
    prompts(a.output,c)
    atomic_write_json(a.output/'preparation.json',dict(config=c,config_hash=canonical_hash(c),
        selection=read_json(a.output/'selection.json'),features=read_json(a.output/'features.json'),
        captions=read_json(a.output/'captions.json'),prompts=read_json(a.output/'prompts.json'),
        source_sha256=sha256_file(Path(__file__)),provenance=runtime_provenance()))
    print('Core cohort and outcome-free Choice prompts frozen',flush=True)
