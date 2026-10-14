#!/usr/bin/env python3
"""Prepare the frozen multi-objective chronological cohort without model evaluation."""
import argparse
import hashlib
import heapq
import io
from pathlib import Path

import numpy as np
import pandas as pd

from paretojev.data.kuairand_27k import BINARY_LABELS, HashReader
from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_core import eligible_blocks, strata_for, common_time_features
from paretojev.data.kuairand_features_v2 import CORE_FIELDS, EXTRA_FIELDS
from paretojev.data.emer import fit_feature_spec, encode_features
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash, sha256_file
from paretojev.experiments.comparison import _output_lock
from prepare_kuairand_core import captions, prompts

DAY_EVENT = np.dtype([('user', '<i4'), ('record', '<i8'), *EVENT.descr])


def ingest_day(root, out, c):
    path = out / 'day-events.bin'
    manifest = out / 'day-events.json'
    if manifest.exists():
        info = read_json(manifest)
        if info['config_hash'] != canonical_hash(c) or sha256_file(path) != info['sha256']:
            raise ValueError('Day event store changed')
        return np.memmap(path, mode='r', dtype=DAY_EVENT)
    if path.exists():
        raise ValueError('Incomplete day ingestion; choose a fresh directory')
    expected = read_json(root / 'extracted/extraction_manifest.json')['files']
    sources = []
    with path.open('wb') as output:
        for shard in (1, 2):
            name = f'log_standard_4_22_to_5_08_27k_part{shard}.csv'
            rows = kept = 0
            with (root / 'extracted/KuaiRand-27K/data' / name).open('rb') as raw:
                tracker = HashReader(raw)
                with io.BufferedReader(tracker) as handle:
                    columns = ['user_id', 'video_id', 'time_ms', 'duration_ms', 'tab', *BINARY_LABELS]
                    for f in pd.read_csv(handle, usecols=columns, dtype='int64', chunksize=500000):
                        # Later confirmation days must retain intervening late
                        # exposures for histories, not just the target test day.
                        history_start = c.get('history_start_ms', c['fresh_test_start_ms'])
                        use = (f.time_ms >= history_start) & (f.time_ms < c['fresh_test_end_ms'])
                        selected = f.loc[use]
                        values = np.empty(len(selected), dtype=DAY_EVENT)
                        values['user'] = selected.user_id
                        values['record'] = (shard-1) * 2**48 + rows + np.flatnonzero(use.to_numpy())
                        for dest, source in [('time', 'time_ms'), ('item', 'video_id'), ('duration', 'duration_ms'), ('tab', 'tab')]:
                            values[dest] = selected[source]
                        labels = selected[list(BINARY_LABELS)].to_numpy()
                        if not np.isin(labels, [0, 1]).all() or not selected.tab.between(0, 14).all():
                            raise ValueError('Invalid source values')
                        values['labels'] = (labels * (1 << np.arange(7))).sum(1)
                        values.tofile(output); rows += len(f); kept += len(values)
                        if rows % 10000000 == 0:
                            print(f'Fresh-day scan shard={shard} rows={rows:,} retained={kept:,}', flush=True)
            digest = tracker.digest.hexdigest()
            if digest != expected[f'KuaiRand-27K/data/{name}']['sha256']:
                raise ValueError('Late source hash mismatch')
            sources.append({'file': name, 'sha256': digest, 'rows': rows, 'retained': kept})
    atomic_write_json(manifest, dict(config_hash=canonical_hash(c), sha256=sha256_file(path), sources=sources))
    return np.memmap(path, mode='r', dtype=DAY_EVENT)


def prepare(a):
    c = read_json(a.config); out = a.output; root = a.root
    registration = out / 'preparation_registration.json'
    if registration.exists() and read_json(registration)['config'] != c:
        raise ValueError('Registered design changed')
    atomic_write_json(registration, {'config': c, 'script_sha256': sha256_file(Path(__file__))})
    if (out / 'preparation.json').exists():
        print('Frozen fresh cohort already prepared', flush=True); return
    # Keep Apr22 histories as well as Apr23-25 targets; no model predictions
    # or selection changes occur during this source-only preparation.
    day = np.array(ingest_day(root, out, c), copy=True)
    order = np.argsort(day['user'], kind='stable')
    sorted_users = day['user'][order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_users)) + 1, len(order)]
    populations = np.zeros(6, dtype='int64'); heaps = [[] for _ in range(6)]
    user_slices = {}
    for left, right in zip(starts[:-1], starts[1:]):
        user = int(sorted_users[left]); ids = order[left:right]
        ids = ids[np.argsort(day['time'][ids], kind='stable')]; user_slices[user] = ids
        e = np.empty(len(ids), dtype=EVENT)
        for key in EVENT.names: e[key] = day[key][ids]
        blocks = eligible_blocks(e, c['fresh_test_start_ms'], c['fresh_test_end_ms'], c['slate_size'], c['maximum_slate_span_ms'])
        strata = strata_for(e, blocks, c['stratum_priority_bits'])
        priorities = np.random.default_rng(np.random.SeedSequence([c['sample_seed'], user])).random(len(blocks))
        for block, stratum, priority in zip(blocks, strata, priorities):
            populations[stratum] += 1; heap = heaps[stratum]; limit = c['test_stratum_budgets'][stratum]
            entry = (float(priority), int(day['record'][ids[block[0]]]), user, ids[block].tolist())
            if len(heap) < limit: heapq.heappush(heap, entry)
            elif priority > heap[0][0]: heapq.heapreplace(heap, entry)
        if user % 5000 == 0:
            print(f'Fresh eligible population user={user}', flush=True)
    selected = []
    for stratum, heap in enumerate(heaps):
        if len(heap) != c['test_stratum_budgets'][stratum]: raise ValueError('Insufficient registered stratum')
        selected.extend((user, stratum, records, populations[stratum]/len(heap)) for _, _, user, records in heap)
    selected.sort(key=lambda row: (int(day['time'][row[2]].min()), row[0]))
    n = len(selected); k = c['slate_size']; h = c['jev']['history_length']
    arrays = dict(x=np.zeros((n,k,26),dtype='float32'), labels=np.zeros((n,k,7),dtype='uint8'),
        user=np.array([r[0] for r in selected],dtype='int32'), split=np.zeros(n,dtype='int8'),
        stratum=np.array([r[1] for r in selected],dtype='int8'), weight=np.array([r[3] for r in selected]),
        items=np.zeros((n,k),dtype='int32'), records=np.zeros((n,k),dtype='int64'),
        decision=np.zeros(n,dtype='int64'), exposure_times=np.zeros((n,k),dtype='int64'),
        duration=np.zeros((n,k),dtype='int64'), history_count=np.zeros(n,dtype='int64'),
        jev_history=np.full((n,h),-1,dtype='int32'), jev_history_time=np.full((n,h),-1,dtype='int64'))
    offsets = np.load(root/'behavior-pilot-v1/offsets.npy')
    metadata = np.array(np.load(root/'behavior-v2/basic.npy',mmap_mode='r'),copy=True)
    by_user = {}
    for i, row in enumerate(selected): by_user.setdefault(row[0], []).append(i)
    with (root/'behavior-pilot-v1/events.bin').open('rb') as handle:
        for number,(user, indices) in enumerate(by_user.items()):
            handle.seek(int(offsets[user])*EVENT.itemsize)
            early = np.fromfile(handle,dtype=EVENT,count=int(offsets[user+1]-offsets[user]))
            ids = user_slices[user]; late = np.empty(len(ids),dtype=EVENT)
            for key in EVENT.names: late[key] = day[key][ids]
            e = np.r_[early[early['time']<c['fresh_test_start_ms']],late]
            e = e[np.argsort(e['time'],kind='stable')]
            for i in indices:
                ids = np.array(selected[i][2]); ids = ids[np.random.default_rng(
                    np.random.SeedSequence([c['sample_seed'],user,i])).permutation(k)]
                target = np.empty(k,dtype=EVENT)
                for key in EVENT.names: target[key] = day[key][ids]
                x, prior, t0, _ = common_time_features(e,target,metadata,50)
                arrays['x'][i] = x; arrays['decision'][i] = t0
                arrays['labels'][i] = (target['labels'][:,None]>>np.arange(7))&1
                arrays['items'][i] = target['item']; arrays['records'][i] = day['record'][ids]
                arrays['exposure_times'][i] = target['time']; arrays['duration'][i] = target['duration']
                arrays['history_count'][i] = len(prior); tail = prior[-h:]
                arrays['jev_history'][i,:len(tail)] = tail['item']; arrays['jev_history_time'][i,:len(tail)] = tail['time']
            if number % 100 == 0:
                print(f'Fresh common-time feature users={number}/{len(by_user)}',flush=True)
    prior_pop = np.load(root/'emer-jev-scale-v3/prior-popularity.npy',mmap_mode='r')
    arrays['popularity'] = prior_pop[arrays['items']]
    np.savez_compressed(out/'raw_slates.npz',**arrays)
    development = root/'emer-jev-scale-v3'
    spec_path=development/'training/feature_spec.json'
    if sha256_file(spec_path)!=c['candidate_selection']['common_feature_spec_sha256']:
        raise ValueError('Frozen development feature specification changed')
    spec = read_json(spec_path)
    atomic_write_json(out/'feature_spec.json',spec)
    atomic_write_json(out/'features.json',dict(fields=CORE_FIELDS+EXTRA_FIELDS,raw_slates_sha256=sha256_file(out/'raw_slates.npz')))
    captions(root,out)
    prompt_config = dict(slate_size=k,split_names=['test'],jev=dict(history_text_characters=c['jev']['history_characters'],
        candidate_text_characters=c['jev']['candidate_characters']),objectives=c['objectives'],questions=c['questions'])
    prompts(out,prompt_config)
    atomic_write_json(out/'preparation.json',dict(config_hash=canonical_hash(c),populations=populations.tolist(),
        slates=n,users=len(by_user),raw_slates_sha256=sha256_file(out/'raw_slates.npz'),
        prompts=read_json(out/'prompts.json'),sources=read_json(out/'day-events.json'),
        development_raw_sha256=sha256_file(development/'raw_slates.npz'),feature_spec_sha256=sha256_file(out/'feature_spec.json'),
        source_record_semantics='(shard-1)*2^48 + zero-based source CSV row; no outcomes sent to Jev',
        old_test_excluded=True))
    print(f'Fresh Apr23-25 cohort frozen: {n} slates',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/emer_multiobjective_confirmation.json'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with _output_lock(a.output):prepare(a)
