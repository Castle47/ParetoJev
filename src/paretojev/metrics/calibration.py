from __future__ import annotations

import numpy as np


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")
    return float(np.mean((probabilities - labels) ** 2))


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> list[dict[str, float]]:
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must have the same number of elements")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, boundaries[1:-1], right=False), n_bins - 1)
    rows: list[dict[str, float]] = []
    for idx in range(n_bins):
        selected = bins == idx
        if not selected.any():
            continue
        rows.append(
            {
                "bin": float(idx),
                "count": float(selected.sum()),
                "mean_probability": float(p[selected].mean()),
                "empirical_rate": float(y[selected].mean()),
            }
        )
    return rows


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    rows = reliability_curve(probabilities, labels, n_bins=n_bins)
    total = sum(row["count"] for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            row["count"] / total * abs(row["mean_probability"] - row["empirical_rate"])
            for row in rows
        )
    )
