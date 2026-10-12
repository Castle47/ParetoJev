"""Registered pseudo-slates, exact stratum sampling, and common-time features."""
import numpy as np
from .kuairand_features_v2 import feature_rows


def eligible_blocks(events, low, high, width, maximum_span):
    idx = np.flatnonzero((events['time'] >= low) & (events['time'] < high))
    blocks = idx[:len(idx)//width*width].reshape(-1, width)
    e = events[blocks]
    valid = ((e['tab'] == e['tab'][:, :1]).all(1)
        & ((e['time'][:, -1]-e['time'][:, 0]) <= maximum_span)
        & (np.diff(np.sort(e['item'], axis=1), axis=1) != 0).all(1))
    return blocks[valid]


def strata_for(events, blocks, priority_bits):
    union = np.bitwise_or.reduce(events['labels'][blocks], axis=1)
    strata = np.full(len(blocks), len(priority_bits), dtype='int8')
    for index, bit in enumerate(priority_bits):
        strata[(strata == len(priority_bits)) & ((union & bit) != 0)] = index
    return strata


def common_time_features(events, targets, metadata, history_length=50):
    """All candidates share the first exposure boundary, even later targets."""
    decision = int(targets['time'].min())
    end = int(np.searchsorted(events['time'], decision, side='left'))
    candidates = targets.copy(); candidates['time'] = decision; candidates['labels'] = 0
    common = np.concatenate([events[:end], candidates])
    x, diag = feature_rows(common, np.arange(end, len(common)), metadata, history_length=history_length)
    return x, events[:end], decision, diag
