#!/usr/bin/env python3
"""Exact occurrence replay of v2 sampling, adding only sequence/identity arrays."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_aligned import selected_indices, sequence_rows
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file, canonical_hash


def prepare(store, data, output, config):
    if (output/'sequences.json').exists():
        raise ValueError('Existing completed sequence artifacts')
    output.mkdir(parents=True, exist_ok=True)
    v2 = read_json(config['v2_config']); manifest = read_json(data/'samples.json')
    if canonical_hash(v2) != manifest['config_hash']:
        raise ValueError('Data protocol mismatch')
    source = read_json(store/'events.json')
    if sha256_file(store/'events.json') != manifest['event_manifest_sha256']:
        raise ValueError('Source manifest changed')
    if sha256_file(store/'offsets.npy') != source['offsets_sha256']:
        raise ValueError('Offsets changed')
    offsets = np.load(store/'offsets.npy')
    frozen, arrays, cursor = {}, {}, {}
    for split, spec in manifest['splits'].items():
        n = spec['rows']; frozen[split] = {}
        for key in ('user', 'time', 'y', 'weight'):
            path = data/f'{split}-{key}.npy'
            if sha256_file(path) != spec['files'][key]: raise ValueError('Frozen rows changed')
            frozen[split][key] = np.load(path)[:n]
        arrays[split] = dict(history=np.zeros((n, config['history_length']), dtype='int32'),
            item=np.zeros(n, dtype='int32'), record=np.zeros(n, dtype='int64'),
            history_max_time=np.zeros(n, dtype='int64'))
        cursor[split] = 0
    digest = hashlib.sha256()
    with (store/'events.bin').open('rb') as handle:
        for user in range(len(offsets)-1):
            size = int(offsets[user+1]-offsets[user])*EVENT.itemsize
            raw = handle.read(size)
            if len(raw) != size: raise ValueError('Truncated source')
            digest.update(raw)
            block = np.frombuffer(raw, dtype=EVENT)
            if not len(block): continue
            order = np.argsort(block['time'], kind='stable'); block = block[order]
            for salt, split in enumerate(('train', 'calibration', 'evaluation')):
                idx, weight = selected_indices(block, user, v2, salt)
                if not len(idx): continue
                start = cursor[split]; stop = start+len(idx); f = frozen[split]
                labels = ((block['labels'][idx, None] >> np.arange(7)) & 1).astype('uint8')
                if not (np.all(f['user'][start:stop] == user)
                    and np.array_equal(f['time'][start:stop], block['time'][idx])
                    and np.array_equal(f['y'][start:stop], labels)
                    and np.all(f['weight'][start:stop] == np.float32(weight))):
                    raise ValueError(f'Sampled occurrence mismatch: {split}/{user}')
                h, item, maximum = sequence_rows(block, idx, config['history_length'], config['item_hash_buckets'])
                a = arrays[split]; a['history'][start:stop] = h; a['item'][start:stop] = item
                a['history_max_time'][start:stop] = maximum
                a['record'][start:stop] = offsets[user]+order[idx]
                cursor[split] = stop
            if user % 5000 == 0: print(f'reconstructed user {user}', flush=True)
        if handle.read(1): raise ValueError('Unaccounted source records')
    if digest.hexdigest() != source['store_sha256']: raise ValueError('Event store changed')
    result = dict(config_hash=canonical_hash(config), sample_manifest_sha256=sha256_file(data/'samples.json'),
        event_store_sha256=digest.hexdigest(), record_unit='zero_based_EVENT_records', splits={})
    for split, a in arrays.items():
        if cursor[split] != manifest['splits'][split]['rows']: raise ValueError('Row count mismatch')
        files = {}
        for key, value in a.items():
            path = output/f'{split}-{key}.npy'; np.save(path, value); files[key] = sha256_file(path)
        result['splits'][split] = dict(rows=cursor[split], files=files, all_histories_strictly_prior=True)
    atomic_write_json(output/'sequences.json', result)
    print('Exact v2 occurrence and history reconstruction complete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('store', 'data', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/kuairand_27k_aligned.json'))
    a = p.parse_args(); prepare(a.store, a.data, a.output, read_json(a.config))
