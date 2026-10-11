"""Bounded-memory, all-user KuaiRand-27K behavior pilot preparation.

The target budget is sampled, not a claim of training on every exposure. The
complete early logs supply histories. Only exposure IDs (never delayed outcomes)
enter histories. Released user profiles are an explicitly separate sensitivity.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .kuairand_27k import BINARY_LABELS, HashReader, USER_COUNT, USER_FILE, USER_FIELDS
from ..utils.io import atomic_write_json, read_json
from ..utils.provenance import canonical_hash, sha256_file

DEFAULT_TRAIN_END_MS = 1650211200000

EVENT = np.dtype([('time', '<i8'), ('item', '<i4'), ('duration', '<i8'),
                  ('tab', 'u1'), ('labels', 'u1')])
NUMERIC_NAMES = ['log_duration', 'duration_invalid', 'log_prior_exposure_count',
                 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']


def hash_items(raw, buckets):
    """Fixed multiplicative hash; raw ID 0 is a real item, encoded IDs start at 2."""
    return ((np.asarray(raw, dtype=np.uint64) * np.uint64(2654435761)) % buckets + 2).astype('int32')


def label_diagnostics(frame):
    out = {}
    for tab, f in frame.groupby('tab'):
        d, p = f.duration_ms.to_numpy(), f.play_time_ms.to_numpy()
        y = f.long_view.to_numpy().astype(bool)
        valid = (d > 0) & (p >= 0)
        derived = p >= np.minimum(d, 18000)
        out[str(tab)] = {
            'rows': len(f), 'duration_zero': int((d == 0).sum()),
            'duration_negative': int((d < 0).sum()), 'play_negative': int((p < 0).sum()),
            'invalid_union': int((~valid).sum()),
            'provided_positive_formula_negative': int((valid & y & ~derived).sum()),
            'provided_negative_formula_positive': int((valid & ~y & derived).sum()),
            'mismatch_within_100ms_of_threshold': int((valid & (y != derived) & (np.abs(p - np.minimum(d, 18000)) <= 100)).sum()),
        }
    return out


def ingest(data: Path, audit_path: Path, out: Path, chunksize=250000, train_end_ms=DEFAULT_TRAIN_END_MS):
    """Use audited per-user counts for a 3 GB disk array, rehash sources in flight."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'events.json'
    if manifest.exists():
        result = read_json(manifest)
        if result.get('diagnostics_train_end_ms', DEFAULT_TRAIN_END_MS) != train_end_ms:
            raise ValueError('Training cutoff changed; label diagnostics would be stale')
        if result['audit_sha256'] != sha256_file(audit_path):
            raise ValueError('Audit changed')
        if (out / 'events.bin').stat().st_size != result['rows'] * EVENT.itemsize:
            raise ValueError('Incomplete event store')
        return result
    if (out / 'events.bin').exists():
        raise ValueError('Partial event store exists; choose a new output directory')
    audit = read_json(audit_path)
    sources = audit['logs']['early_standard']
    if any(s['scope'] != 'full_file' for s in sources):
        raise ValueError('Full audits of both early shards required')
    counts = np.sum([s['user_event_counts'] for s in sources], axis=0, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    cursor = offsets[:-1].copy()
    events = np.memmap(out / 'events.bin', dtype=EVENT, mode='w+', shape=(int(offsets[-1]),))
    diag, hashes = {}, {}
    for source in sources:
        path = data / source['file']
        rows = 0
        with path.open('rb') as raw:
            tracker = HashReader(raw)
            with io.BufferedReader(tracker) as handle:
                cols = ['user_id', 'video_id', 'time_ms', 'tab', 'duration_ms', 'play_time_ms', *BINARY_LABELS]
                for f in pd.read_csv(handle, usecols=cols, dtype='int64', chunksize=chunksize):
                    # Group within each chunk; shard and input row order are not chronology.
                    f = f.sort_values('user_id', kind='stable')
                    users = f.user_id.to_numpy()
                    if users.min() < 0 or users.max() >= len(counts):
                        raise ValueError('User ID out of range')
                    starts = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
                    ends = np.r_[starts[1:], len(f)]
                    values = np.empty(len(f), dtype=EVENT)
                    for dest, src in [('time', 'time_ms'), ('item', 'video_id'), ('duration', 'duration_ms'), ('tab', 'tab')]:
                        values[dest] = f[src].to_numpy()
                    values['labels'] = (f[list(BINARY_LABELS)].to_numpy(dtype='uint8') * (1 << np.arange(7))).sum(axis=1)
                    for start, end in zip(starts, ends):
                        user = users[start]; stop = cursor[user] + end - start
                        if stop > offsets[user + 1]:
                            raise ValueError('Audited count mismatch')
                        events[cursor[user]:stop] = values[start:end]
                        cursor[user] = stop
                    # Diagnose only the predeclared training period.
                    for tab, metrics in label_diagnostics(f[f.time_ms < train_end_ms]).items():
                        dst = diag.setdefault(tab, {k: 0 for k in metrics})
                        for key, value in metrics.items(): dst[key] += value
                    rows += len(f)
                    if rows % 5000000 == 0: print(f'ingest {path.name}: {rows:,}', flush=True)
            digest = tracker.digest.hexdigest()
        if digest != source['source_sha256'] or rows != source['rows']:
            raise ValueError('Source hash/count differs from audited source')
        hashes[path.name] = digest
    if not np.array_equal(cursor, offsets[1:]): raise ValueError('Incomplete per-user store')
    events.flush()
    np.save(out / 'offsets.npy', offsets)
    result = dict(status='complete', rows=int(offsets[-1]), dtype=EVENT.descr,
                  diagnostics_train_end_ms=int(train_end_ms),
                  audit_sha256=sha256_file(audit_path), source_sha256=hashes,
                  training_label_diagnostics_by_tab=diag,
                  store_sha256=sha256_file(out / 'events.bin'),
                  offsets_sha256=sha256_file(out / 'offsets.npy'))
    atomic_write_json(manifest, result)
    return result


def sample_user(events, user, config):
    """Stable ties; every same-timestamp exposure excluded from target history."""
    e = events[np.argsort(events['time'], kind='stable')]
    t = e['time']
    bounds = (0, config['train_end_exclusive_ms'], config['validation_end_exclusive_ms'])
    result = {}
    for split, lo, hi, budget, salt in zip(
        ('train', 'validation'), bounds[:-1], bounds[1:],
        (config['train_targets_per_user'], config['validation_targets_per_user']), (0, 1)
    ):
        eligible = np.flatnonzero((t >= lo) & (t < hi))
        if not len(eligible): continue
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], user, salt]))
        selected = np.sort(rng.choice(eligible, min(budget, len(eligible)), replace=False))
        target = e[selected]
        ends = np.searchsorted(t, target['time'], side='left')
        history = np.zeros((len(selected), config['history_length']), dtype='int32')
        history_max_time = np.full(len(selected), -1, dtype='int64')
        for j, end in enumerate(ends):
            prior = e[max(0, end - config['history_length']):end]
            history[j, :len(prior)] = hash_items(prior['item'], config['item_hash_buckets'])
            if len(prior): history_max_time[j] = prior['time'][-1]
        # Asia/Shanghai local clock, fixed cyclic encoding, no fitted future scale.
        day = (target['time'] // 1000 + 8 * 3600) / 86400
        hour_angle = (day % 1) * (2 * np.pi)
        weekday_angle = ((np.floor(day) + 3) % 7) * (2 * np.pi / 7)
        numeric = np.stack([
            np.log1p(np.maximum(target['duration'], 0)) / 12,
            (target['duration'] <= 0).astype(float), np.log1p(ends) / 12,
            np.sin(hour_angle), np.cos(hour_angle), np.sin(weekday_angle), np.cos(weekday_angle)
        ], axis=1).astype('float32')
        result[split] = dict(
            history=history, item=hash_items(target['item'], config['item_hash_buckets']),
            raw_item=target['item'], user=np.full(len(selected), user, dtype='int32'),
            tab=target['tab'].astype('int32'), numeric=numeric,
            labels=((target['labels'][:, None] >> np.arange(7)) & 1).astype('float32'),
            weight=np.full(len(selected), len(eligible) / len(selected), dtype='float32'),
            time=target['time'], history_max_time=history_max_time,
        )
    return result


def prepare_samples(out: Path, config):
    manifest = out / 'samples.json'
    if manifest.exists():
        result = read_json(manifest)
        if result['config_hash'] != canonical_hash(config): raise ValueError('Configuration changed')
        return result
    ev = read_json(out / 'events.json')
    offsets = np.load(out / 'offsets.npy')
    if sha256_file(out / 'offsets.npy') != ev['offsets_sha256']: raise ValueError('Offsets changed')
    if sha256_file(out / 'events.bin') != ev['store_sha256']: raise ValueError('Event store changed')
    events = np.memmap(out / 'events.bin', dtype=EVENT, mode='r', shape=(ev['rows'],))
    arrays = {'train': [], 'validation': []}
    for user in range(len(offsets) - 1):
        if offsets[user] == offsets[user + 1]: continue
        for split, values in sample_user(events[offsets[user]:offsets[user + 1]], user, config).items():
            arrays[split].append(values)
        if user % 5000 == 0: print(f'sample user {user}', flush=True)
    result = {'config_hash': canonical_hash(config), 'source_manifest_sha256': sha256_file(out / 'events.json'), 'splits': {}}
    for split, groups in arrays.items():
        merged = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
        if not (merged['history_max_time'] < merged['time']).all(): raise ValueError('History leakage')
        path = out / f'{split}.npz'
        np.savez(path, **merged)
        result['splits'][split] = dict(rows=len(merged['user']), users=len(np.unique(merged['user'])),
            represented_exposures=float(merged['weight'].sum(dtype='float64')),
            min_time=int(merged['time'].min()), max_time=int(merged['time'].max()),
            positives=merged['labels'].sum(axis=0).astype(int).tolist(),
            history_length_median=float(np.median((merged['history'] != 0).sum(axis=1))),
            history_length_50_fraction=float(np.mean((merged['history'] != 0).sum(axis=1) == 50)),
            sha256=sha256_file(path))
    if result['splits']['train']['max_time'] >= result['splits']['validation']['min_time']:
        raise ValueError('Temporal split overlap')
    atomic_write_json(manifest, result)
    return result


def user_snapshot(data: Path, out: Path, train_users):
    """Fit vocab ONLY to training users; unused/missing profile values are OOV=0.

    This does not solve unknown snapshot timing; use only in sensitivity runs.
    """
    frame = pd.read_csv(data / USER_FILE, dtype=str, keep_default_na=False).set_index('user_id')
    if len(frame) != USER_COUNT or frame.index.duplicated().any(): raise ValueError('User table identity mismatch')
    frame = frame.reindex([str(i) for i in range(USER_COUNT)])
    encoded = np.zeros((USER_COUNT, len(USER_FIELDS)), dtype='int32')
    vocab, cardinalities = {}, []
    for j, field in enumerate(USER_FIELDS):
        values = frame[field].fillna('').str.strip()
        # All 30 fields use categorical embeddings; raw counts intentionally are
        # not normalized over validation and have OOV handling for unseen values.
        observed = sorted(set(values.iloc[np.unique(train_users)]) - {'', 'nan', '-124'})
        mapping = {value: i + 1 for i, value in enumerate(observed)}
        encoded[:, j] = values.map(mapping).fillna(0).to_numpy(dtype='int32')
        vocab[field] = mapping; cardinalities.append(len(mapping) + 1)
    np.save(out / 'user_snapshot.npy', encoded)
    spec = dict(regime='released_user_snapshot_NOT_temporally_verified', fields=list(USER_FIELDS),
                cardinalities=cardinalities, vocabulary=vocab, source_sha256=sha256_file(data / USER_FILE),
                encoded_sha256=sha256_file(out / 'user_snapshot.npy'))
    atomic_write_json(out / 'user_snapshot.json', spec)
    return {k: v for k, v in spec.items() if k != 'vocabulary'}
