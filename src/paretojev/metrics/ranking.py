from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def positive_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return 1-indexed ranks for rows containing exactly one positive item."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim != 2 or labels.shape != scores.shape:
        raise ValueError("scores and labels must both have shape [N,K]")
    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError("each row must contain exactly one positive label")
    # Stable sort makes tied-score behavior deterministic and reproducible.
    order = np.argsort(-scores, axis=1, kind="stable")
    positive_index = labels.argmax(axis=1)
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    for row in range(scores.shape[0]):
        ranks[row] = int(np.flatnonzero(order[row] == positive_index[row])[0]) + 1
    return ranks


def per_user_auc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUC for one-positive candidate sets, with 0.5 credit for score ties."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("scores and labels must have shape [N,K]")
    aucs = []
    for row_scores, row_labels in zip(scores, labels, strict=True):
        positives = row_scores[row_labels == 1]
        negatives = row_scores[row_labels == 0]
        if len(positives) != 1 or len(negatives) == 0:
            raise ValueError("each row must have one positive and at least one negative")
        p = positives[0]
        aucs.append(float(((p > negatives).sum() + 0.5 * (p == negatives).sum()) / len(negatives)))
    return np.asarray(aucs, dtype=np.float64)


def single_positive_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    ranks = positive_ranks(scores, labels)
    result: dict[str, float] = {
        "auc": float(per_user_auc(scores, labels).mean()),
        "mrr": float((1.0 / ranks).mean()),
    }
    for k in ks:
        result[f"hr@{k}"] = float((ranks <= k).mean())
        gains = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
        result[f"ndcg@{k}"] = float(gains.mean())
    return result


def dcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    relevance = np.asarray(relevance, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")[:k]
    gains = np.power(2.0, relevance[order]) - 1.0
    discounts = np.log2(np.arange(2, len(order) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    actual = dcg_at_k(relevance, scores, k)
    ideal = dcg_at_k(relevance, relevance, k)
    return actual / ideal if ideal > 0 else 0.0


def paired_bootstrap_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired bootstrap inputs must be same-shaped vectors")
    difference = a - b
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))
    means = difference[samples].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(difference.mean()), float(low), float(high)
