"""Train-only feature encoding for the EMER experiment."""
import numpy as np


def fit_feature_spec(x, weights):
    numeric = np.asarray(x[..., 1:], dtype='float64').reshape(-1, 25)
    weights = np.repeat(weights, x.shape[1])[:, None]
    valid = np.isfinite(numeric)
    total = (valid * weights).sum(0)
    mean = (np.where(valid, numeric, 0) * weights).sum(0) / np.maximum(total, 1)
    variance = (np.where(valid, numeric - mean, 0) ** 2 * weights).sum(0) / np.maximum(total, 1)
    scale = np.sqrt(variance)
    # One-day training makes weekdays constant. An epsilon standard deviation
    # would amplify the next day's perfectly valid values by a million.
    constant = scale < 1e-5
    scale[constant] = 1.0
    return {'mean': mean.tolist(), 'scale': scale.tolist(),
        'constant_columns': np.flatnonzero(constant).tolist(),
        'fit_scope': 'training candidates only, inverse-inclusion-weighted moments',
        'constant_scale': 1.0}


def encode_features(x, spec):
    numeric = np.asarray(x[..., 1:], dtype='float64')
    valid = np.isfinite(numeric)
    normalized = np.where(valid, (numeric - spec['mean']) / spec['scale'], 0)
    tabs = x[..., 0].astype('int64')
    if ((tabs < 0) | (tabs >= 15)).any():
        raise ValueError('Unknown tab')
    return np.concatenate((normalized, (~valid).astype('float32'), np.eye(15)[tabs]), -1).astype('float32')
