"""Predeclared score-only policies and exact expected metrics for score ties.

No policy accepts outcome labels. Labels are used only by the evaluator. Remaining
ties are evaluated by integrating uniformly over their possible rank positions.
"""
from __future__ import annotations

import numpy as np


METHODS = ("itemcf", "title_only", "jev_only", "title_ties", "jev_ties", "title_05", "jev_05")


def tied_candidate_mask(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Tie selection needs one finite score per candidate")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    return counts[inverse] > 1


def twice_average_ranks(scores: np.ndarray) -> np.ndarray:
    """Integer-valued twice the zero-based average rank; exact fusion at 5%."""
    values = np.asarray(scores)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ranks need a finite score vector")
    smaller = (values[:, None] > values[None, :]).sum(axis=1)
    equal = (values[:, None] == values[None, :]).sum(axis=1)
    return 2 * smaller + equal - 1


def policy_keys(behavior: np.ndarray, title: np.ndarray, jev: np.ndarray) -> dict[str, np.ndarray]:
    """Return lexicographic maximize keys [K,D], without inspecting labels.

    `*_ties` compares CF first and only uses the secondary key inside exact CF
    ties. `*_05` is precisely 95% CF average-rank + 5% secondary average-rank:
    integer rank keys avoid floating-point artifacts when weighted scores tie.
    """
    behavior, title, jev = (np.asarray(values) for values in (behavior, title, jev))
    if behavior.ndim != 1 or len(behavior) < 2 or title.shape != behavior.shape or jev.shape != behavior.shape:
        raise ValueError("Policies need aligned score vectors with at least two candidates")
    if not all(np.isfinite(x).all() for x in (behavior, title, jev)):
        raise ValueError("Policy scores must be finite")
    b, t, j = (twice_average_ranks(x) for x in (behavior, title, jev))
    return {
        "itemcf": behavior[:, None], "title_only": title[:, None], "jev_only": jev[:, None],
        "title_ties": np.column_stack((behavior, title)), "jev_ties": np.column_stack((behavior, jev)),
        "title_05": (19 * b + t)[:, None], "jev_05": (19 * b + j)[:, None],
    }


def expected_metrics(keys: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Exact one-positive AUC, HR and NDCG expectations over unresolved ties."""
    keys, labels = np.asarray(keys), np.asarray(labels)
    if keys.ndim != 2 or keys.shape[0] < 2 or labels.shape != (keys.shape[0],):
        raise ValueError("Expected keys [K,D] and labels [K]")
    if not np.isfinite(keys).all() or not np.isin(labels, (0, 1)).all() or labels.sum() != 1:
        raise ValueError("Use finite keys and exactly one positive binary label")
    positive = keys[labels.argmax()]
    tied = np.ones(len(labels), dtype=bool)
    higher = np.zeros(len(labels), dtype=bool)
    for column in range(keys.shape[1]):
        higher |= tied & (keys[:, column] > positive[column])
        tied &= keys[:, column] == positive[column]
    n_higher, n_tied = int(higher.sum()), int(tied.sum())
    ranks = np.arange(n_higher + 1, n_higher + n_tied + 1, dtype=np.float64)
    result = {"auc": float((len(labels) - n_higher - n_tied + 0.5 * (n_tied - 1)) / (len(labels) - 1)),
              "mrr": float(np.mean(1 / ranks))}
    for k in (1, 5, 10):
        result[f"hr@{k}"] = float(np.mean(ranks <= k))
        result[f"ndcg@{k}"] = float(np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
    return result
