from __future__ import annotations

import numpy as np


def _apply_directions(points: np.ndarray, directions: np.ndarray | None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape [N,M]")
    if directions is None:
        return points
    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape != (points.shape[1],):
        raise ValueError("directions shape mismatch")
    if not np.all(np.isin(directions, [-1, 1])):
        raise ValueError("directions must contain only -1 and 1")
    return points * directions


def pareto_front_mask(points: np.ndarray, directions: np.ndarray | None = None) -> np.ndarray:
    """Non-dominated points for a maximization problem."""
    adjusted = _apply_directions(points, directions)
    n = len(adjusted)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominates_i = np.all(adjusted >= adjusted[i], axis=1) & np.any(adjusted > adjusted[i], axis=1)
        if dominates_i.any():
            efficient[i] = False
    return efficient


def hypervolume_2d(
    points: np.ndarray,
    reference: np.ndarray,
    directions: np.ndarray | None = None,
) -> float:
    adjusted = _apply_directions(points, directions)
    reference = np.asarray(reference, dtype=np.float64)
    if directions is not None:
        reference = reference * np.asarray(directions, dtype=np.float64)
    if adjusted.shape[1] != 2 or reference.shape != (2,):
        raise ValueError("hypervolume_2d requires 2D points and a 2D reference")
    eligible = adjusted[np.all(adjusted >= reference, axis=1)]
    if len(eligible) == 0:
        return 0.0
    frontier = eligible[pareto_front_mask(eligible)]
    frontier = np.unique(frontier, axis=0)
    frontier = frontier[np.argsort(frontier[:, 0], kind="stable")]
    area = 0.0
    previous_x = reference[0]
    for x, y in frontier:
        width = max(0.0, x - previous_x)
        area += width * max(0.0, y - reference[1])
        previous_x = max(previous_x, x)
    return float(area)


def hypervolume_monte_carlo(
    points: np.ndarray,
    reference: np.ndarray,
    directions: np.ndarray | None = None,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """Approximate M-dimensional dominated hypervolume for maximization."""
    adjusted = _apply_directions(points, directions)
    reference = np.asarray(reference, dtype=np.float64)
    if directions is not None:
        reference = reference * np.asarray(directions, dtype=np.float64)
    if reference.shape != (adjusted.shape[1],):
        raise ValueError("reference shape mismatch")
    eligible = adjusted[np.all(adjusted >= reference, axis=1)]
    if len(eligible) == 0:
        return 0.0
    upper = eligible.max(axis=0)
    span = upper - reference
    if np.any(span <= 0):
        return 0.0
    rng = np.random.default_rng(seed)
    samples = reference + rng.random((n_samples, adjusted.shape[1])) * span
    dominated = np.zeros(n_samples, dtype=bool)
    for point in eligible[pareto_front_mask(eligible)]:
        dominated |= np.all(samples <= point, axis=1)
    return float(dominated.mean() * np.prod(span))


def paired_dominance_rate(
    candidate: np.ndarray,
    baseline: np.ndarray,
    directions: np.ndarray | None = None,
    atol: float = 1e-12,
) -> float:
    candidate = _apply_directions(candidate, directions)
    baseline = _apply_directions(baseline, directions)
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline must have identical shapes")
    weak = np.all(candidate >= baseline - atol, axis=1)
    strict = np.any(candidate > baseline + atol, axis=1)
    return float(np.mean(weak & strict))


def worst_objective_regret(
    metrics: np.ndarray,
    ideal: np.ndarray,
    directions: np.ndarray | None = None,
) -> float:
    metrics = np.asarray(metrics, dtype=np.float64)
    ideal = np.asarray(ideal, dtype=np.float64)
    if metrics.shape[-1] != ideal.shape[-1]:
        raise ValueError("metrics and ideal must agree on objective dimension")
    d = np.ones_like(ideal) if directions is None else np.asarray(directions, dtype=np.float64)
    regret = (ideal * d) - (metrics * d)
    return float(np.maximum(regret, 0.0).max())


def controllability_spearman(weights: np.ndarray, achieved: np.ndarray) -> dict[str, float]:
    """Per-objective Spearman correlation between requested and achieved emphasis."""
    weights = np.asarray(weights, dtype=np.float64)
    achieved = np.asarray(achieved, dtype=np.float64)
    if weights.shape != achieved.shape or weights.ndim != 2:
        raise ValueError("weights and achieved must have identical [N,M] shapes")

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        out = np.empty_like(order, dtype=np.float64)
        out[order] = np.arange(len(x), dtype=np.float64)
        return out

    result: dict[str, float] = {}
    for objective in range(weights.shape[1]):
        rw = ranks(weights[:, objective])
        ra = ranks(achieved[:, objective])
        if np.std(rw) == 0 or np.std(ra) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(rw, ra)[0, 1])
        result[f"objective_{objective}"] = corr
    result["mean"] = float(np.mean(list(result.values())))
    return result
