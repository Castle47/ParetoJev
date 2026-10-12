"""Paired checkpoint comparison on one immutable evaluation protocol.

This module evaluates already selected checkpoints; it never selects a method,
checkpoint, cold cutoff, or preference using evaluation outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from paretojev.data.schema import SlateBatch
from paretojev.evaluation.pareto import ParetoEvaluationConfig, build_preference_set
from paretojev.metrics.pareto import (
    controllability_spearman,
    paired_dominance_rate,
    pareto_front_mask,
)


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _ratio_gain(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return float((value - baseline) / abs(baseline))


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64), where=denominator > 0,
    )


def _finite_mean(values: np.ndarray) -> float | None:
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def _contributions(
    scores: np.ndarray, labels: np.ndarray, mask: np.ndarray,
    directions: np.ndarray, top_k: int, cold_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-slate sufficient statistics preserving conditional NDCG denominators.

    Positive objectives average NDCG over slates with positive ideal DCG.
    Negative objectives pool selected candidate utilities, including unequal
    slate lengths. Cold NDCG evaluates the ordinary ranking, with relevance
    restricted to cold candidates, and is undefined for negative objectives.
    """
    n, width, m = labels.shape
    selected_k = min(top_k, width)
    order = np.argsort(-np.where(mask, scores, -np.inf), axis=1, kind="stable")[:, :selected_k]
    selected_mask = np.take_along_axis(mask, order, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, selected_k + 2, dtype=np.float64))
    safe_labels = np.where(mask[..., None], labels, 0.0)
    numerator = np.zeros((n, m), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    cold_numerator = np.zeros_like(numerator)
    cold_denominator = np.zeros_like(numerator)
    for objective, direction in enumerate(directions):
        relevance = safe_labels[..., objective]
        selected = np.take_along_axis(relevance, order, axis=1)
        if direction < 0:
            numerator[:, objective] = ((1.0 - selected) * selected_mask).sum(axis=1)
            denominator[:, objective] = selected_mask.sum(axis=1)
            continue
        for rel, num, den in (
            (relevance, numerator, denominator),
            (np.where(cold_mask, relevance, 0.0), cold_numerator, cold_denominator),
        ):
            ideal = np.sort(rel, axis=1)[:, ::-1][:, :selected_k]
            ideal_dcg = (ideal * discounts).sum(axis=1)
            actual = np.take_along_axis(rel, order, axis=1)
            actual_dcg = (actual * selected_mask * discounts).sum(axis=1)
            np.divide(actual_dcg, ideal_dcg, out=num[:, objective], where=ideal_dcg > 0)
            den[:, objective] = ideal_dcg > 0
    return numerator, denominator, cold_numerator, cold_denominator


def _slate_weights(values, n):
    if values is None:
        return None
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (n,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weights must be finite positive values, one per slate")
    return weights


def comparison_objective_metrics(
    scores: np.ndarray, labels: np.ndarray, mask: np.ndarray,
    directions: Iterable[int], *, top_k: int, sample_weights=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared checkpoint-selection and comparison metric definitions.

    Returns maximize-oriented objective utilities and fractions of eligible
    slates. Missing support is NaN, so validation can reject it explicitly.
    Minimized objectives pool selected candidates, rather than averaging
    per-slate risk means when slate lengths differ.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    directions = np.asarray(tuple(directions))
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if scores.ndim != 2 or mask.shape != scores.shape or not scores.shape[0]:
        raise ValueError("scores and mask must have nonempty shape [N,K]")
    if labels.ndim != 3 or labels.shape[:2] != scores.shape:
        raise ValueError("labels must have shape [N,K,M]")
    if directions.shape != (labels.shape[2],) or not np.isin(directions, (-1, 1)).all():
        raise ValueError("directions must contain one +1 or -1 per objective")
    if not np.isfinite(scores[mask]).all():
        raise ValueError("valid scores must be finite")
    if not np.isfinite(labels[mask]).all() or np.any((labels[mask] < 0) | (labels[mask] > 1)):
        raise ValueError("valid labels must be finite and in [0,1]")
    num, den, _, _ = _contributions(scores, labels, mask, directions, top_k, np.zeros_like(mask))
    weights = _slate_weights(sample_weights, len(scores))
    coverage = (den > 0).mean(axis=0) if weights is None else np.average(den > 0, axis=0, weights=weights)
    if weights is not None:
        num, den = num * weights[:, None], den * weights[:, None]
    return _divide(num.sum(axis=0), den.sum(axis=0)), coverage


def _bootstrap(
    statistics: dict[str, tuple[np.ndarray, ...]], group_ids: np.ndarray,
    objective_names: list[str], positive: np.ndarray, baseline: str, samples: int, seed: int,
) -> dict[str, object]:
    """Paired cluster bootstrap; preferences and candidates are never resampled."""
    groups, codes = np.unique(group_ids, return_inverse=True)
    result: dict[str, object] = {
        "unit": "user_group", "groups": len(groups), "samples": samples,
        "confidence_level": 0.95, "seed": seed,
        "note": "Paired percentile intervals, conditional on fixed fitted checkpoints and preference grid; no hypervolume data interval is estimated.",
        "comparisons": {},
    }
    if len(groups) < 2:
        result["status"] = "unavailable: at least two distinct user groups are required"
        result["comparisons"] = {
            method: {
                "baseline": baseline,
                "objective_mean_absolute_gain": {
                    name: {"interval": None, "valid_replicates": 0}
                    for name in objective_names
                },
                "cold_ndcg_absolute_gain": {"interval": None, "valid_replicates": 0},
            }
            for method in statistics if method != baseline
        }
        return result
    grouped: dict[str, list[np.ndarray]] = {}
    for method, parts in statistics.items():
        grouped[method] = []
        for part in parts:
            aggregate = np.zeros((len(groups), part.shape[0], part.shape[2]), dtype=np.float64)
            np.add.at(aggregate, codes, part.transpose(1, 0, 2))
            grouped[method].append(aggregate)
    draws: dict[str, list[np.ndarray]] = {name: [] for name in statistics if name != baseline}
    cold_draws: dict[str, list[float]] = {name: [] for name in draws}
    covered_cold = positive & (statistics[baseline][3].sum(axis=(0, 1)) > 0)
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        counts = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        estimates: dict[str, tuple[np.ndarray, float | None]] = {}
        for method, parts in grouped.items():
            weighted = [np.tensordot(counts, part, axes=(0, 0)) for part in parts]
            points = _divide(weighted[0], weighted[1])
            cold = _divide(weighted[2], weighted[3])
            # Keep the original estimand fixed: a resample missing a covered
            # cold objective must not silently average a different subset.
            cold_mean = _number(cold[:, covered_cold].mean()) if covered_cold.any() else None
            estimates[method] = (points.mean(axis=0), cold_mean)
        baseline_points, baseline_cold = estimates[baseline]
        for method in draws:
            points, cold = estimates[method]
            draws[method].append(points - baseline_points)
            cold_draws[method].append(
                float(cold - baseline_cold) if cold is not None and baseline_cold is not None else np.nan
            )

    def interval(values: np.ndarray) -> dict[str, object]:
        finite = values[np.isfinite(values)]
        # Dropping unavailable replicates can imply unjustified certainty for
        # rare objectives. Report their count but withhold the interval.
        bounds = np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) == samples else None
        return {"interval": bounds, "valid_replicates": len(finite)}

    result["status"] = "computed"
    result["comparisons"] = {
        method: {
            "baseline": baseline,
            "objective_mean_absolute_gain": {
                name: interval(np.asarray(draws[method])[:, index])
                for index, name in enumerate(objective_names)
            },
            "cold_ndcg_absolute_gain": interval(np.asarray(cold_draws[method])),
        }
        for method in draws
    }
    return result


def evaluate_comparison(
    models: dict[str, torch.nn.Module], loader: Iterable[SlateBatch], *,
    directions: Iterable[int], objective_names: Sequence[str],
    cold_popularity_threshold: float,
    config: ParetoEvaluationConfig = ParetoEvaluationConfig(),
    baseline: str = "emer_like", device: torch.device | str = "cpu",
    audit_path: Path | str | None = None,
    group_ids: Sequence[str | int] | None = None,
    bootstrap_samples: int = 0,
    sample_weights=None,
) -> dict[str, object]:
    """Compare separately trained checkpoints using identical observed labels.

    ``cold_popularity_threshold`` must be fitted on the training split by the
    caller. No test quantile is computed. ``group_ids`` must follow loader row
    order (use a nonshuffled loader when supplying external IDs). A single
    traversal ensures all methods see the same rows even for a one-shot loader.
    Models must return ``{"score_mean": tensor[B,K]}``; labels are used only for
    evaluation, never to choose checkpoint or hyperparameters here.

    Positive utility is conditional NDCG; minimize utility is one minus pooled
    selected risk. Uncovered objectives, undefined relative gains, and missing
    cold metrics are JSON null, never fabricated zeros. NPZ audit arrays use
    [method, preference, slate, objective/candidate] axes.
    """
    names = list(objective_names)
    direction_array = np.asarray(tuple(directions))
    if direction_array.ndim != 1 or len(direction_array) < 2 or not np.isin(direction_array, (-1, 1)).all():
        raise ValueError("directions must contain at least two values, each -1 or 1")
    if len(names) != len(direction_array) or len(set(names)) != len(names):
        raise ValueError("objective_names must be unique and match directions")
    if not models or baseline not in models:
        raise ValueError("models must include the named baseline")
    if any(not isinstance(name, str) or not name for name in models):
        raise ValueError("method names must be nonempty strings")
    if not np.isfinite(cold_popularity_threshold):
        raise ValueError("cold_popularity_threshold must be a finite training-fitted value")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    if bootstrap_samples and group_ids is None:
        raise ValueError("user group IDs are required for a grouped bootstrap")
    preferences, preference_names = build_preference_set(len(names), config)
    device = torch.device(device)
    positive = direction_array > 0
    statistics_parts: dict[str, list[tuple[np.ndarray, ...]]] = {name: [] for name in models}
    score_parts: dict[str, list[np.ndarray]] = {name: [] for name in models}
    labels_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    popularity_parts: list[np.ndarray] = []
    training_modes = {name: model.training for name, model in models.items()}
    width: int | None = None
    try:
        for model in models.values():
            model.to(device).eval()
        with torch.no_grad():
            for original in loader:
                original.validate()
                batch = original.to(device)
                labels = batch.labels.detach().cpu().numpy()
                mask = batch.mask.detach().cpu().numpy().astype(bool)
                popularity = batch.popularity.detach().cpu().numpy()[..., 0]
                if labels.shape[2] != len(names):
                    raise ValueError("batch objective count does not match directions")
                if width is not None and labels.shape[1] != width:
                    raise ValueError("evaluation batches must use a common slate width")
                width = labels.shape[1]
                if not np.isfinite(labels[mask]).all() or np.any((labels[mask] < 0) | (labels[mask] > 1)):
                    raise ValueError("valid observed labels must be finite and in [0,1]")
                if not np.isfinite(popularity[mask]).all():
                    raise ValueError("valid popularity must be finite")
                cold_mask = mask & (popularity <= cold_popularity_threshold)
                labels_parts.append(labels.copy())
                mask_parts.append(mask.copy())
                popularity_parts.append(popularity.copy())
                for method, model in models.items():
                    contributions: list[tuple[np.ndarray, ...]] = []
                    scores_for_batch: list[np.ndarray] = []
                    for preference in preferences:
                        weights = torch.as_tensor(preference, dtype=batch.preference_weights.dtype, device=device).expand(batch.batch_size, -1)
                        prediction = model(replace(batch, preference_weights=weights))
                        scores = prediction["score_mean"].detach().cpu().numpy()
                        if scores.shape != mask.shape or not np.isfinite(scores[mask]).all():
                            raise ValueError(f"{method} must return finite valid scores with shape [B,K]")
                        contributions.append(_contributions(scores, labels, mask, direction_array, config.top_k, cold_mask))
                        if audit_path is not None:
                            scores_for_batch.append(scores.copy())
                    statistics_parts[method].append(tuple(np.stack([part[index] for part in contributions]) for index in range(4)))
                    if audit_path is not None:
                        score_parts[method].append(np.stack(scores_for_batch))
    finally:
        for name, model in models.items():
            model.train(training_modes[name])
    if not labels_parts:
        raise ValueError("evaluation loader is empty")
    labels = np.concatenate(labels_parts)
    mask = np.concatenate(mask_parts)
    popularity = np.concatenate(popularity_parts)
    if not mask.any():
        raise ValueError("evaluation data contains no valid candidates")
    groups: np.ndarray | None = None
    if group_ids is not None:
        if len(group_ids) != len(labels) or any(value is None or (isinstance(value, float) and not np.isfinite(value)) for value in group_ids):
            raise ValueError("group_ids must contain one nonmissing user ID per evaluated slate")
        groups = np.asarray(group_ids, dtype=str)
        if groups.ndim != 1:
            raise ValueError("group_ids must be one-dimensional")
    statistics = {
        method: tuple(np.concatenate([part[index] for part in parts], axis=1) for index in range(4))
        for method, parts in statistics_parts.items()
    }
    sample_weights = _slate_weights(sample_weights, len(labels))
    if sample_weights is not None:
        statistics = {method: tuple(part * sample_weights[None, :, None] for part in parts)
                      for method, parts in statistics.items()}
    rng = np.random.default_rng(config.seed)
    hv_samples = rng.random((config.hypervolume_samples, len(names)))
    summaries: dict[str, dict[str, object]] = {}
    point_arrays: dict[str, np.ndarray] = {}
    for method, (num, den, cold_num, cold_den) in statistics.items():
        points = _divide(num.sum(axis=1), den.sum(axis=1))
        cold_points = _divide(cold_num.sum(axis=1), cold_den.sum(axis=1))
        point_arrays[method] = points
        complete = np.isfinite(points).all()
        dominated = np.zeros(config.hypervolume_samples, dtype=bool)
        if complete:
            for point in points[pareto_front_mask(points)]:
                dominated |= np.all(hv_samples <= point, axis=1)
        hv = float(dominated.mean()) if complete else None
        control: dict[str, float | None] = {}
        for index, name in enumerate(names):
            if np.isfinite(points[:, index]).all():
                control[name] = controllability_spearman(preferences[:, index:index + 1], points[:, index:index + 1])["mean"]
            else:
                control[name] = None
        control["mean"] = float(np.mean(list(control.values()))) if complete else None
        summaries[method] = {
            "hypervolume": hv,
            "hypervolume_mc_standard_error": float(np.sqrt(hv * (1 - hv) / config.hypervolume_samples)) if hv is not None else None,
            "pareto_front_size": int(pareto_front_mask(points).sum()) if complete else None,
            "objective_mean": {name: _number(points[:, index].mean()) for index, name in enumerate(names)},
            "objective_best": {name: _number(points[:, index].max()) for index, name in enumerate(names)},
            "controllability": control,
            "cold_positive_ndcg_mean": _finite_mean(cold_points[:, positive]),
            "cold_positive_objectives_covered": int(np.isfinite(cold_points[0, positive]).sum()),
            "points": [[_number(value) for value in row] for row in points],
            "cold_positive_points": [[_number(value) for value in row] for row in cold_points[:, positive]],
        }
    def compare_pair(method: str, comparator: str) -> dict[str, object]:
        points = point_arrays[method]
        baseline_points = point_arrays[comparator]
        complete = np.isfinite(points).all() and np.isfinite(baseline_points).all()
        regret = np.maximum(baseline_points.max(axis=0) - points.max(axis=0), 0.0)
        candidate_hv = summaries[method]["hypervolume"]
        baseline_hv = summaries[comparator]["hypervolume"]
        candidate_cold = summaries[method]["cold_positive_ndcg_mean"]
        baseline_cold = summaries[comparator]["cold_positive_ndcg_mean"]
        return {
            "baseline": comparator, "hypervolume": candidate_hv,
            "baseline_hypervolume": baseline_hv,
            "hypervolume_absolute_gain": candidate_hv - baseline_hv if complete else None,
            "hypervolume_relative_gain": _ratio_gain(candidate_hv, baseline_hv),
            "cold_positive_ndcg_mean": candidate_cold,
            "baseline_cold_positive_ndcg_mean": baseline_cold,
            "cold_item_ndcg_relative_gain": _ratio_gain(candidate_cold, baseline_cold),
            "worst_objective_regret": float(regret.max()) if complete else None,
            "per_objective_regret": {name: _number(regret[index]) for index, name in enumerate(names)},
            "controllability_spearman_mean": summaries[method]["controllability"]["mean"],
            "paired_dominance_rate": paired_dominance_rate(points, baseline_points) if complete else None,
        }
    comparisons = {
        method: compare_pair(method, baseline)
        for method in point_arrays if method != baseline
    }
    # Research gates use both comparator identities fixed before evaluation.
    # Reuse the same predictions; never select whichever baseline is easier to
    # beat, and never let the generic `baseline` argument relabel a target.
    research_comparators = ("emer_like", "cmr_like")
    baseline_pair_comparisons = {
        comparator: compare_pair("paretojev", comparator)
        for comparator in research_comparators
        if "paretojev" in point_arrays and comparator in point_arrays
    }
    factorial_pairs = [("paretojev", "emer_like_no_jev"), ("paretojev", "cmr_like_no_jev"),
        ("paretojev", "paretojev_no_jev"), ("paretojev", "emer_like"), ("paretojev", "cmr_like"),
        ("emer_like", "emer_like_no_jev"), ("cmr_like", "cmr_like_no_jev"),
        ("paretojev", "behavior_weighted_sum")]
    factorial = {f"{method}_vs_{comparator}": compare_pair(method, comparator)
                 for method, comparator in factorial_pairs if method in models and comparator in models}
    first_stats = statistics[baseline]
    coverage = (first_stats[1][0] > 0).mean(axis=0)
    cold_coverage = (first_stats[3][0] > 0).mean(axis=0)
    if sample_weights is not None:
        coverage = np.average(first_stats[1][0] > 0, axis=0, weights=sample_weights)
        cold_coverage = np.average(first_stats[3][0] > 0, axis=0, weights=sample_weights)
    report: dict[str, object] = {
        "schema_version": "checkpoint-comparison-v1", "config": asdict(config),
        "objective_names": names, "directions": direction_array.tolist(),
        "preference_names": preference_names, "preference_vectors": preferences.tolist(),
        "data": {"slates": len(labels), "slate_width": width, "valid_candidates": int(mask.sum()), "cold_candidates": int((mask & (popularity <= cold_popularity_threshold)).sum())},
        "cold_popularity_threshold": float(cold_popularity_threshold),
        "objective_metric_coverage": dict(zip(names, coverage.tolist(), strict=True)),
        "cold_positive_objective_coverage": {name: float(cold_coverage[index]) for index, name in enumerate(names) if positive[index]},
        "protocol": {
            "baseline": baseline, "baseline_selection": "named before evaluation; no best-on-test selection",
            "research_comparators": list(research_comparators),
            "research_comparator_rule": "predeclared EMER-like and CMR-like comparisons must each satisfy research gates; neither comparator is selected from test results",
            "cold_threshold_source": "caller-supplied training-fitted threshold; config.cold_quantile is not evaluated on test",
            "hypervolume": "common random samples in shared [0,1]^M maximize-utility cube; unavailable if an objective has no coverage",
            "hypervolume_uncertainty": "reported standard error describes Monte Carlo integration only, not sampling or training uncertainty",
            "regret": "absolute positive utility shortfall of candidate coordinate-wise best over the fixed preference grid versus named baseline coordinate-wise best; not oracle or per-preference regret",
            "cold_metric": "mean conditional cold-relevance NDCG across preferences and covered positive objectives; uncovered objectives are excluded and their coverage is reported",
            "controllability": "Spearman across the shared preference grid; constant achieved utility uses zero by convention",
            "audit_axes": ["method", "preference", "slate", "objective_or_candidate"],
        },
        "methods": summaries, "comparisons": comparisons,
        "baseline_pair_comparisons": baseline_pair_comparisons,
    }
    target_namespaces = {
        namespace: baseline_pair_comparisons[comparator]
        for comparator, namespace in (("emer_like", "paretojev"), ("cmr_like", "paretojev_vs_cmr"))
        if comparator in baseline_pair_comparisons
    }
    if target_namespaces:
        report["kuairand"] = target_namespaces
    if bootstrap_samples and groups is not None:
        report["paired_user_bootstrap"] = _bootstrap(statistics, groups, names, positive, baseline, bootstrap_samples, config.seed)
    else:
        report["paired_user_bootstrap"] = {"status": "not_requested", "samples": 0}
    if any(name.endswith("_no_jev") for name in models):
        report["factorial_comparisons"] = factorial
    if sample_weights is not None:
        report["population_weighting"] = {"kind": "inverse inclusion weights", "sum": float(sample_weights.sum()),
            "effective_slates_kish": float(sample_weights.sum()**2 / np.square(sample_weights).sum())}
    if audit_path is not None:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "method_names": np.asarray(list(models), dtype=str),
            "objective_names": np.asarray(names, dtype=str), "preferences": preferences,
            "labels": labels, "mask": mask, "popularity": popularity,
            "directions": direction_array, "cold_popularity_threshold": np.asarray(cold_popularity_threshold),
            "scores": np.stack([np.concatenate(score_parts[method], axis=1) for method in models]),
        }
        for index, name in enumerate(("utility_numerator", "utility_denominator", "cold_numerator", "cold_denominator")):
            arrays[name] = np.stack([statistics[method][index] for method in models])
        if groups is not None:
            arrays["group_ids"] = groups
        if sample_weights is not None:
            arrays["slate_weights"] = sample_weights
        # A file object prevents numpy silently appending an unexpected suffix.
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        report["audit"] = {"path": str(path), "format": "npz", "contains": list(arrays)}
    return report
