"""Recover exact v2 sampled occurrences and strictly prior sequence inputs."""
import numpy as np
from .kuairand_pilot import hash_items


def selected_indices(events, user, config, salt):
    boundaries = [0, config['train_end_ms'], config['calibration_end_ms'], config['evaluation_end_ms']]
    split = ('train', 'calibration', 'evaluation')[salt]
    eligible = np.flatnonzero((events['time'] >= boundaries[salt]) & (events['time'] < boundaries[salt+1]))
    if not len(eligible):
        return np.empty(0, dtype='int64'), 0.
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], user, salt]))
    idx = np.sort(rng.choice(eligible, min(config['targets_per_user'][split], len(eligible)), replace=False))
    return idx, len(eligible)/len(idx)


def sequence_rows(events, selected, length, buckets):
    times = events['time']
    if np.any(times[1:] < times[:-1]):
        raise ValueError('Chronological input required')
    end = np.searchsorted(times, times[selected], side='left')
    counts = np.minimum(end, length)
    indices = np.maximum(end[:, None]-length, 0)+np.arange(length)[None, :]
    valid = np.arange(length)[None, :] < counts[:, None]
    safe = np.minimum(indices, len(events)-1)
    history = np.where(valid, hash_items(events['item'][safe], buckets), 0).astype('int32')
    max_time = np.where(counts > 0, times[np.maximum(end-1, 0)], -1)
    if not np.all(max_time < times[selected]):
        raise ValueError('History leakage')
    return history, hash_items(events['item'][selected], buckets), max_time
