from __future__ import annotations

import numpy as np


def rank_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Map values to [0, 1] by within-slate rank; ties are broken stably."""
    values = np.asarray(values)
    order = np.argsort(np.argsort(values, axis=axis, kind="stable"), axis=axis, kind="stable")
    denom = max(values.shape[axis] - 1, 1)
    return order / denom


def linear_hybrid(
    behavior_scores: np.ndarray,
    jev_scores: np.ndarray,
    alpha: float = 0.5,
    *,
    normalize_by_rank: bool = True,
) -> np.ndarray:
    """Simple transparent baseline: alpha * Jev + (1-alpha) * behavior."""
    if behavior_scores.shape != jev_scores.shape:
        raise ValueError("score arrays must have the same shape")
    b = rank_normalize(behavior_scores) if normalize_by_rank else behavior_scores
    j = rank_normalize(jev_scores) if normalize_by_rank else jev_scores
    return alpha * j + (1.0 - alpha) * b
