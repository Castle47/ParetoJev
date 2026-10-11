"""Bounded-memory exposure statistics and upload-property joins for 27K v2."""
from __future__ import annotations
import io
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .kuairand_27k import HashReader, ITEM_COUNT, USER_COUNT
from .kuairand_pilot import EVENT
from ..utils.io import read_json, atomic_write_json
from ..utils.provenance import canonical_hash, sha256_file

DAY_MS = 86400000
META = np.dtype([('author', '<i8'), ('upload_day', '<i4'), ('duration', '<f4'), ('width', '<f4'), ('height', '<f4')])
CORE_FIELDS = ['tab', 'log_duration', 'duration_missing', 'hour_sin', 'hour_cos',
 'weekday_sin', 'weekday_cos', 'log_exposures_1h', 'log_exposures_1d',
 'log_exposures_7d', 'log_hours_since_last', 'history_length_fraction',
 'history_mean_log_duration', 'history_duration_known_fraction',
 'candidate_repeat_fraction', 'same_tab_fraction', 'distinct_history_fraction',
 'log_observed_history_days']
EXTRA_FIELDS = ['log_basic_duration', 'log_width', 'log_height', 'aspect_ratio',
 'log_upload_age_days', 'metadata_available', 'author_repeat_fraction', 'history_author_known_fraction']


def local_day(times):
    return (np.asarray(times, dtype='int64') + 8 * 3600000) // DAY_MS


def make_metadata(source, extraction, cache, output, *, item_count=ITEM_COUNT, chunksize=250000):
    """Create a dense ID index; only content properties, never monthly aggregates."""
    output = Path(output); cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / 'basic.npy'; manifest_path = output / 'metadata.json'
    if manifest_path.exists():
        m = read_json(manifest_path)
        if not index_path.exists(): shutil.copyfile(output / 'basic.npy', index_path)
        if sha256_file(index_path) != m['index_sha256']: raise ValueError('Metadata cache changed')
        return np.load(index_path, mmap_mode='r'), m
    if index_path.exists(): raise ValueError('Partial metadata cache: choose a fresh cache directory')
    expected = read_json(extraction)['files']['KuaiRand-27K/data/video_features_basic_27k.csv']['sha256']
    seen = np.zeros(item_count, dtype=bool)
    dest = np.lib.format.open_memmap(index_path, mode='w+', dtype=META, shape=(item_count,))
    dest['author'] = -1; dest['upload_day'] = -1
    for key in ['duration', 'width', 'height']: dest[key] = np.nan
    rows = 0; missing = dict(author=0, upload_day=0, duration=0, width=0, height=0)
    with Path(source).open('rb') as raw:
        tracker = HashReader(raw)
        with io.BufferedReader(tracker) as handle:
            cols = ['video_id', 'author_id', 'upload_dt', 'video_duration', 'server_width', 'server_height']
            for f in pd.read_csv(handle, usecols=cols, chunksize=chunksize):
                ids = f.video_id.to_numpy(dtype='int64')
                if (ids < 0).any() or (ids >= item_count).any() or seen[ids].any() or len(np.unique(ids)) != len(ids):
                    raise ValueError('Duplicate or invalid catalog ID')
                seen[ids] = True
                dates = pd.to_datetime(f.upload_dt, format='%Y-%m-%d', errors='coerce')
                days = dates.to_numpy(dtype='datetime64[D]').astype('int64')
                valid = dates.notna().to_numpy() & (days >= 0)
                dest['upload_day'][ids] = np.where(valid, days, -1)
                missing['upload_day'] += int((~valid).sum())
                author = pd.to_numeric(f.author_id, errors='coerce').to_numpy(dtype='float64')
                valid = np.isfinite(author) & (author >= 0) & (author == np.floor(author)) & (author < 2**53)
                dest['author'][ids] = np.where(valid, author, -1).astype('int64')
                missing['author'] += int((~valid).sum())
                for target, raw_name in [('duration','video_duration'), ('width','server_width'), ('height','server_height')]:
                    value = pd.to_numeric(f[raw_name], errors='coerce').to_numpy(dtype='float64')
                    valid = np.isfinite(value) & (value > 0) & (value <= np.finfo('float32').max)
                    dest[target][ids] = np.where(valid, value, np.nan).astype('float32')
                    missing[target] += int((~valid).sum())
                rows += len(f)
                if rows % 5000000 == 0: print(f'metadata rows {rows:,}', flush=True)
        digest = tracker.digest.hexdigest()
    if digest != expected or not seen.all(): raise ValueError('Metadata source hash or full ID coverage failed')
    dest.flush(); del dest
    digest_index = sha256_file(index_path)
    shutil.copyfile(index_path, output / 'basic.npy')
    if sha256_file(output / 'basic.npy') != digest_index: raise ValueError('External metadata copy differs')
    result = dict(rows=rows, source_sha256=digest, index_sha256=digest_index,
        missing=missing, fields=META.descr, assumption='properties fixed at upload; not timestamp-proven')
    atomic_write_json(manifest_path, result)
    return np.load(index_path, mmap_mode='r'), result


def feature_rows(events, selected, metadata, *, history_length=50):
    """Events must be chronologically sorted; labels never enter these features."""
    e = events; times = e['time']; target = e[selected]; t = target['time']
    if np.any(times[1:] < times[:-1]): raise ValueError('Chronological input required')
    ends = np.searchsorted(times, t, side='left')
    indexes = ends[:, None] - history_length + np.arange(history_length)[None, :]
    valid = indexes >= 0
    history = e[np.maximum(indexes, 0)]
    lengths = valid.sum(axis=1); denom = np.maximum(lengths, 1)
    duration = target['duration']; hd = history['duration']
    positive_hd = valid & (hd > 0)
    phase = ((t + 8 * 3600000) % DAY_MS) / DAY_MS * 2 * np.pi
    weekday = ((local_day(t) + 3) % 7) * (2 * np.pi / 7)
    gap = np.where(ends > 0, (t - times[np.maximum(ends - 1, 0)]) / 3600000, 0)
    distinct = np.array([len(np.unique(row[mask])) for row,mask in zip(history['item'], valid)], dtype=float)
    core = [target['tab'], np.log1p(np.maximum(duration, 0))/12, duration <= 0,
        np.sin(phase), np.cos(phase), np.sin(weekday), np.cos(weekday)]
    core.extend(np.log1p(ends - np.searchsorted(times, t - window, side='left'))
                for window in (3600000, DAY_MS, 7 * DAY_MS))
    core.extend([np.log1p(gap), lengths/history_length,
        (np.log1p(np.maximum(hd, 0))/12 * positive_hd).sum(axis=1)/np.maximum(positive_hd.sum(axis=1),1),
        positive_hd.sum(axis=1)/denom,
        ((history['item'] == target['item'][:, None]) & valid).sum(axis=1)/denom,
        ((history['tab'] == target['tab'][:, None]) & valid).sum(axis=1)/denom,
        distinct/denom, np.log1p(np.maximum(t-times[0],0)/DAY_MS)])
    m = metadata[target['item']]
    available = (m['upload_day'] >= 0) & (m['upload_day'] <= local_day(t))
    hm = metadata[history['item']]
    havailable = valid & (hm['upload_day'] >= 0) & (hm['upload_day'] <= local_day(history['time'])) & (hm['author'] >= 0)
    same_author = havailable & available[:, None] & (m['author'][:, None] >= 0) & (hm['author'] == m['author'][:, None])
    def numeric(field):
        return np.where(available & np.isfinite(m[field]) & (m[field] > 0), m[field], np.nan)
    width, height = numeric('width'), numeric('height')
    extra = [np.log1p(numeric('duration'))/12, np.log1p(width), np.log1p(height),
        np.clip(width/height, 0.1, 10),
        np.where(available, np.log1p(np.maximum(local_day(t)-m['upload_day'],0)), np.nan),
        available.astype(float), same_author.sum(axis=1)/denom, havailable.sum(axis=1)/denom]
    x = np.stack(core+extra, axis=1).astype('float32')
    if np.isinf(x).any() or np.isnan(x[:, :len(CORE_FIELDS)]).any(): raise ValueError('Invalid core features')
    diag = dict(rows=len(t), metadata_available=int(available.sum()),
        future_upload_masked=int((m['upload_day'] > local_day(t)).sum()),
        missing_upload_masked=int((m['upload_day'] < 0).sum()),
        history_50=int((lengths == history_length).sum()))
    return x, diag


def prepare(store, metadata, output, config):
    store, output = Path(store), Path(output)
    if (output / 'samples.json').exists(): raise ValueError('Existing samples: use frozen artifacts or a new directory')
    source = read_json(store / 'events.json')
    if sha256_file(store / 'events.bin') != source['store_sha256']: raise ValueError('Event source changed')
    if sha256_file(store / 'offsets.npy') != source['offsets_sha256']: raise ValueError('User offsets changed')
    offsets = np.load(store / 'offsets.npy')
    e = np.memmap(store / 'events.bin', dtype=EVENT, mode='r', shape=(source['rows'],))
    arrays = {}; count = {}; diagnostics = {}
    for split, budget in config['targets_per_user'].items():
        max_rows = (len(offsets)-1) * budget
        specs = {'x': ('float32', (max_rows, len(CORE_FIELDS)+len(EXTRA_FIELDS))),
                 'y': ('uint8', (max_rows, 7)), 'weight': ('float32',(max_rows,)),
                 'user': ('int32',(max_rows,)), 'time': ('int64',(max_rows,))}
        arrays[split] = {name: np.lib.format.open_memmap(output/f'{split}-{name}.npy', mode='w+',dtype=dtype,shape=shape)
                         for name,(dtype,shape) in specs.items()}
        count[split] = 0; diagnostics[split] = {}
    boundaries = [0, config['train_end_ms'], config['calibration_end_ms'], config['evaluation_end_ms']]
    for user in range(len(offsets)-1):
        block = e[offsets[user]:offsets[user+1]]
        if not len(block): continue
        block = block[np.argsort(block['time'],kind='stable')]
        for salt, split in enumerate(('train','calibration','evaluation')):
            eligible = np.flatnonzero((block['time'] >= boundaries[salt]) & (block['time'] < boundaries[salt+1]))
            if not len(eligible): continue
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'], user, salt]))
            idx = np.sort(rng.choice(eligible, min(config['targets_per_user'][split], len(eligible)), replace=False))
            x, diag = feature_rows(block, idx, metadata, history_length=config['history_length'])
            start = count[split]; stop = start + len(idx); a = arrays[split]
            a['x'][start:stop] = x; a['y'][start:stop] = (block['labels'][idx,None] >> np.arange(7)) & 1
            a['weight'][start:stop] = len(eligible)/len(idx)
            a['user'][start:stop] = user; a['time'][start:stop] = block['time'][idx]
            count[split] = stop
            for k,v in diag.items(): diagnostics[split][k] = diagnostics[split].get(k,0)+v
        if user % 5000 == 0: print(f'feature user {user}', flush=True)
    result = dict(config_hash=canonical_hash(config), event_manifest_sha256=sha256_file(store/'events.json'),
        metadata_manifest_sha256=sha256_file(output/'metadata.json'),
        feature_source_sha256=sha256_file(Path(__file__)), core_fields=CORE_FIELDS, extra_fields=EXTRA_FIELDS, splits={})
    for split,a in arrays.items():
        n=count[split]
        if n == 0: raise ValueError('Empty period')
        for array in a.values(): array.flush()
        result['splits'][split] = dict(rows=n, users=int(len(np.unique(a['user'][:n]))),
            min_time=int(a['time'][:n].min()), max_time=int(a['time'][:n].max()),
            represented_exposures=float(a['weight'][:n].sum(dtype='float64')),
            positives=a['y'][:n].sum(axis=0).tolist(), diagnostics=diagnostics[split],
            files={key:sha256_file(output/f'{split}-{key}.npy') for key in a})
    if not result['splits']['train']['max_time'] < result['splits']['calibration']['min_time'] <= result['splits']['calibration']['max_time'] < result['splits']['evaluation']['min_time']:
        raise ValueError('Temporal overlap')
    atomic_write_json(output/'samples.json',result)
    return result
