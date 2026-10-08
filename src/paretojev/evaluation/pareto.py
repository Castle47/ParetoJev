from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from paretojev.data.schema import SlateBatch
from paretojev.metrics.pareto import (
    controllability_spearman,
    hypervolume_monte_carlo,
    paired_dominance_rate,
    pareto_front_mask,
)
from paretojev.models.pareto_reranker import ParetoJevReranker


@dataclass(frozen=True)
class ParetoEvaluationConfig:
    """Configuration for a runtime-preference sweep.

    The evaluator always includes one uniform preference and every one-hot
    preference. Optional pairwise and Dirichlet preferences provide denser front
    coverage. Objective utilities are normalized to [0, 1], so every method is
    compared in the same unit hypercube for Monte Carlo hypervolume estimation.
    """

    top_k: int = 5
    random_preferences: int = 8
    include_pairwise_preferences: bool = False
    dirichlet_alpha: float = 0.7
    hypervolume_samples: int = 100_000
    cold_quantile: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.random_preferences < 0:
            raise ValueError("random_preferences cannot be negative")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if self.hypervolume_samples <= 0:
            raise ValueError("hypervolume_samples must be positive")
        if not 0.0 < self.cold_quantile < 1.0:
            raise ValueError("cold_quantile must lie strictly between 0 and 1")


def build_preference_set(
    n_objectives: int,
    config: ParetoEvaluationConfig = ParetoEvaluationConfig(),
) -> tuple[np.ndarray, list[str]]:
    """Build a deterministic set of normalized objective preferences."""

    if n_objectives < 2:
        raise ValueError("Pareto evaluation requires at least two objectives")
    vectors: list[np.ndarray] = []
    names: list[str] = []

    vectors.append(np.full(n_objectives, 1.0 / n_objectives, dtype=np.float64))
    names.append("uniform")
    for objective in range(n_objectives):
        vector = np.zeros(n_objectives, dtype=np.float64)
        vector[objective] = 1.0
        vectors.append(vector)
        names.append(f"objective_{objective}")

    if config.include_pairwise_preferences:
        for left in range(n_objectives):
            for right in range(left + 1, n_objectives):
                vector = np.zeros(n_objectives, dtype=np.float64)
                vector[left] = 0.5
                vector[right] = 0.5
                vectors.append(vector)
                names.append(f"pair_{left}_{right}")

    if config.random_preferences:
        rng = np.random.default_rng(config.seed)
        random = rng.dirichlet(
            np.full(n_objectives, config.dirichlet_alpha),
            size=config.random_preferences,
        )
        vectors.extend(random)
        names.extend(f"dirichlet_{index}" for index in range(config.random_preferences))

    result = np.stack(vectors)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("Internal preference construction did not normalize weights")
    return result, names


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))


def objective_utility_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    directions: Sequence[int],
    *,
    top_k: int,
    cold_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate every objective under one shared candidate ranking.

    Positive objectives use conditional NDCG@K: only slates containing at least
    one relevant candidate contribute. Minimize objectives use one minus the
    observed negative-feedback rate among selected candidates. This converts all
    reported coordinates to a maximize-oriented [0, 1] utility space.

    With ``cold_mask``, positive relevance is restricted to cold candidates.
    Minimize objectives return NaN in that view because a "cold no-negative"
    statistic would mostly measure how many non-cold candidates were selected,
    rather than cold-item ranking quality.
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    directions_array = np.asarray(tuple(directions), dtype=np.int64)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if scores.ndim != 2 or mask.shape != scores.shape:
        raise ValueError("scores and mask must have shape [N,K]")
    if labels.shape[:2] != scores.shape or labels.ndim != 3:
        raise ValueError("labels must have shape [N,K,M]")
    if directions_array.shape != (labels.shape[2],):
        raise ValueError("directions shape mismatch")
    if not np.all(np.isin(directions_array, (-1, 1))):
        raise ValueError("directions must contain only -1 and 1")
    if cold_mask is not None:
        cold_mask = np.asarray(cold_mask, dtype=bool)
        if cold_mask.shape != scores.shape:
            raise ValueError("cold_mask shape mismatch")
    if not np.isfinite(scores[mask]).all():
        raise ValueError("valid scores must be finite")
    valid_labels = labels[np.broadcast_to(mask[..., None], labels.shape)]
    if not np.isfinite(valid_labels).all():
        raise ValueError("valid labels must be finite")
    if np.any((valid_labels < -1e-7) | (valid_labels > 1.0 + 1e-7)):
        raise ValueError("labels must lie in [0,1]")

    n, _, m = labels.shape
    values = np.full(m, np.nan, dtype=np.float64)
    coverage = np.zeros(m, dtype=np.float64)
    selected_orders: list[np.ndarray] = []
    for row in range(n):
        valid = np.flatnonzero(mask[row])
        if len(valid) == 0:
            selected_orders.append(np.empty(0, dtype=np.int64))
            continue
        order = valid[np.argsort(-scores[row, valid], kind="stable")]
        selected_orders.append(order[: min(top_k, len(order))])

    discounts = _discounts(top_k)
    for objective in range(m):
        if directions_array[objective] > 0:
            ndcgs: list[float] = []
            for row, selected in enumerate(selected_orders):
                if len(selected) == 0:
                    continue
                relevance = labels[row, :, objective].copy()
                relevance[~mask[row]] = 0.0
                if cold_mask is not None:
                    relevance[~cold_mask[row]] = 0.0
                ideal = np.sort(relevance[mask[row]])[::-1][:top_k]
                ideal_dcg = float((ideal * discounts[: len(ideal)]).sum())
                if ideal_dcg <= 0:
                    continue
                actual = relevance[selected]
                actual_dcg = float((actual * discounts[: len(actual)]).sum())
                ndcgs.append(actual_dcg / ideal_dcg)
            values[objective] = float(np.mean(ndcgs)) if ndcgs else 0.0
            coverage[objective] = len(ndcgs) / max(n, 1)
        elif cold_mask is None:
            selected_risk: list[float] = []
            for row, selected in enumerate(selected_orders):
                if len(selected):
                    selected_risk.extend(labels[row, selected, objective].tolist())
            values[objective] = (
                1.0 - float(np.mean(selected_risk)) if selected_risk else 0.0
            )
            coverage[objective] = 1.0 if selected_risk else 0.0
    return values, coverage


def _static_arrays(loader: DataLoader[SlateBatch]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {
        "behavior": [],
        "jev": [],
        "labels": [],
        "mask": [],
        "popularity": [],
    }
    for batch in loader:
        values["behavior"].append(batch.behavior_scores.detach().cpu().numpy())
        values["jev"].append(batch.jev_scores.detach().cpu().numpy())
        values["labels"].append(batch.labels.detach().cpu().numpy())
        values["mask"].append(batch.mask.detach().cpu().numpy())
        values["popularity"].append(batch.popularity.detach().cpu().numpy()[..., 0])
    if not values["labels"]:
        raise ValueError("Evaluation loader is empty")
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in values.items()}
    valid = arrays["mask"].astype(bool)
    if not valid.any():
        raise ValueError("Evaluation data contains no valid candidates")
    for name in ("behavior", "jev", "labels"):
        selected = arrays[name][np.broadcast_to(valid[..., None], arrays[name].shape)]
        if not np.isfinite(selected).all():
            raise ValueError(f"{name} contains non-finite values on valid candidates")
    valid_popularity = arrays["popularity"][valid]
    if not np.isfinite(valid_popularity).all():
        raise ValueError("popularity contains non-finite values on valid candidates")
    return arrays


def _model_scores(
    model: ParetoJevReranker,
    loader: DataLoader[SlateBatch],
    preference: np.ndarray,
    device: torch.device,
    *,
    collect_fused: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    scores: list[np.ndarray] = []
    fused: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            weights = torch.as_tensor(
                preference,
                dtype=batch.preference_weights.dtype,
                device=device,
            ).expand(batch.batch_size, -1)
            batch.preference_weights = weights
            output = model(batch)
            scores.append(output["score_mean"].detach().cpu().numpy())
            if collect_fused:
                fused.append(output["fused_scores"].detach().cpu().numpy())
    if not scores:
        raise ValueError("Evaluation loader is empty")
    return np.concatenate(scores), (np.concatenate(fused) if fused else None)


def _weighted_probability_scores(
    probabilities: np.ndarray,
    preference: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    utility = np.where(directions[None, None, :] > 0, probabilities, 1.0 - probabilities)
    return np.einsum("nkm,m->nk", utility, preference)


def _relative_gain(candidate: float, baseline: float) -> float:
    # Keep artifacts strict-JSON-compatible even for a degenerate zero baseline.
    return float((candidate - baseline) / max(abs(baseline), 1e-12))


def _named_controllability(
    preferences: np.ndarray,
    points: np.ndarray,
    objective_names: Sequence[str],
) -> dict[str, float]:
    raw = controllability_spearman(preferences, points)
    result = {
        name: float(raw[f"objective_{index}"])
        for index, name in enumerate(objective_names)
    }
    result["mean"] = float(raw["mean"])
    return result


def evaluate_pareto_model(
    model: ParetoJevReranker,
    loader: DataLoader[SlateBatch],
    *,
    directions: Iterable[int],
    config: ParetoEvaluationConfig = ParetoEvaluationConfig(),
    objective_names: Sequence[str] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Sweep runtime preferences and compare ParetoJev with signal baselines.

    Baselines are deliberately inference-only and use exactly the same slates:
    weighted behavior probabilities, weighted Jev probabilities, the learned
    confidence-gated fusion probabilities, and a label oracle. A separately
    trained EMER-like or CMR-like checkpoint should be evaluated by the same
    routine and compared outside this report; the static baselines here are not
    presented as substitutes for those trainable baselines.
    """

    device = torch.device(device)
    directions_array = np.asarray(tuple(int(value) for value in directions), dtype=np.int64)
    if len(directions_array) != model.config.n_objectives:
        raise ValueError("directions count must equal model objective count")
    if not np.all(np.isin(directions_array, (-1, 1))):
        raise ValueError("directions must contain only -1 and 1")
    if objective_names is None:
        objective_names = [f"objective_{index}" for index in range(len(directions_array))]
    objective_names = list(objective_names)
    if len(objective_names) != len(directions_array):
        raise ValueError("objective_names count must equal objective count")
    if len(set(objective_names)) != len(objective_names):
        raise ValueError("objective_names must be unique")

    preferences, preference_names = build_preference_set(len(directions_array), config)
    static = _static_arrays(loader)
    valid_mask = static["mask"].astype(bool)
    valid_popularity = static["popularity"][valid_mask]
    cold_cut = float(np.quantile(valid_popularity, config.cold_quantile))
    cold_mask = (static["popularity"] <= cold_cut) & valid_mask

    method_points: dict[str, list[np.ndarray]] = {
        "paretojev": [],
        "behavior": [],
        "jev": [],
        "fused": [],
        "oracle": [],
    }
    method_cold: dict[str, list[float]] = {key: [] for key in method_points}
    coverage_reference: np.ndarray | None = None
    cold_coverage_reference: np.ndarray | None = None
    fused_probabilities: np.ndarray | None = None

    for preference_index, preference in enumerate(preferences):
        model_scores, maybe_fused = _model_scores(
            model,
            loader,
            preference,
            device,
            collect_fused=preference_index == 0,
        )
        if maybe_fused is not None:
            fused_probabilities = maybe_fused
        if fused_probabilities is None:
            raise AssertionError("Fused probabilities were not collected")
        score_map = {
            "paretojev": model_scores,
            "behavior": _weighted_probability_scores(
                static["behavior"], preference, directions_array
            ),
            "jev": _weighted_probability_scores(
                static["jev"], preference, directions_array
            ),
            "fused": _weighted_probability_scores(
                fused_probabilities, preference, directions_array
            ),
            "oracle": _weighted_probability_scores(
                static["labels"], preference, directions_array
            ),
        }
        for method, scores in score_map.items():
            points, coverage = objective_utility_metrics(
                scores,
                static["labels"],
                valid_mask,
                directions_array,
                top_k=config.top_k,
            )
            cold_points, cold_coverage = objective_utility_metrics(
                scores,
                static["labels"],
                valid_mask,
                directions_array,
                top_k=config.top_k,
                cold_mask=cold_mask,
            )
            method_points[method].append(points)
            positive_cold = cold_points[directions_array > 0]
            method_cold[method].append(
                float(np.nanmean(positive_cold))
                if np.isfinite(positive_cold).any()
                else 0.0
            )
            if coverage_reference is None and method == "paretojev":
                coverage_reference = coverage
                cold_coverage_reference = cold_coverage

    summaries: dict[str, dict[str, object]] = {}
    point_arrays: dict[str, np.ndarray] = {}
    reference = np.zeros(len(directions_array), dtype=np.float64)
    upper = np.ones(len(directions_array), dtype=np.float64)
    unit_volume = float(np.prod(upper - reference))
    for method, rows in method_points.items():
        points = np.stack(rows)
        if np.any((points < -1e-7) | (points > 1.0 + 1e-7)):
            raise ValueError(f"{method} objective utilities escaped [0,1]")
        points = np.clip(points, 0.0, 1.0)
        point_arrays[method] = points
        hypervolume = hypervolume_monte_carlo(
            points,
            reference,
            n_samples=config.hypervolume_samples,
            seed=config.seed,
            upper=upper,
        )
        dominated_probability = hypervolume / unit_volume
        hypervolume_se = unit_volume * float(
            np.sqrt(
                max(dominated_probability * (1.0 - dominated_probability), 0.0)
                / config.hypervolume_samples
            )
        )
        controllability = _named_controllability(
            preferences, points, objective_names
        )
        summaries[method] = {
            "hypervolume": hypervolume,
            "hypervolume_mc_standard_error": hypervolume_se,
            "pareto_front_size": int(pareto_front_mask(points).sum()),
            "objective_mean": {
                name: float(value)
                for name, value in zip(objective_names, points.mean(axis=0), strict=True)
            },
            "objective_best": {
                name: float(value)
                for name, value in zip(objective_names, points.max(axis=0), strict=True)
            },
            "controllability": controllability,
            "cold_positive_ndcg_mean": float(np.mean(method_cold[method])),
            "points": points.tolist(),
        }

    baselines = ("behavior", "jev", "fused")
    best_hv_baseline = max(
        baselines, key=lambda name: float(summaries[name]["hypervolume"])
    )
    best_cold_baseline = max(
        baselines,
        key=lambda name: float(summaries[name]["cold_positive_ndcg_mean"]),
    )
    baseline_hv = float(summaries[best_hv_baseline]["hypervolume"])
    model_hv = float(summaries["paretojev"]["hypervolume"])
    baseline_cold = float(summaries[best_cold_baseline]["cold_positive_ndcg_mean"])
    model_cold = float(summaries["paretojev"]["cold_positive_ndcg_mean"])
    oracle_hv = float(summaries["oracle"]["hypervolume"])

    ideal = np.max(
        np.concatenate([point_arrays[name] for name in (*baselines, "paretojev")], axis=0),
        axis=0,
    )
    model_best = point_arrays["paretojev"].max(axis=0)
    per_objective_regret = np.maximum(ideal - model_best, 0.0)
    worst_regret = float(per_objective_regret.max())
    control_mean = float(summaries["paretojev"]["controllability"]["mean"])
    dominance = {
        baseline: paired_dominance_rate(
            point_arrays["paretojev"], point_arrays[baseline]
        )
        for baseline in baselines
    }

    target_summary = {
        "hypervolume": model_hv,
        "best_available_static_baseline": best_hv_baseline,
        "best_available_static_baseline_hypervolume": baseline_hv,
        "hypervolume_absolute_gain": model_hv - baseline_hv,
        "hypervolume_relative_gain": _relative_gain(model_hv, baseline_hv),
        "oracle_hypervolume": oracle_hv,
        "oracle_hypervolume_ratio": model_hv / max(oracle_hv, 1e-12),
        "cold_positive_ndcg_mean": model_cold,
        "best_cold_static_baseline": best_cold_baseline,
        "best_cold_static_baseline_ndcg": baseline_cold,
        "cold_item_ndcg_relative_gain": _relative_gain(model_cold, baseline_cold),
        "worst_objective_regret": worst_regret,
        "per_objective_regret": {
            name: float(value)
            for name, value in zip(objective_names, per_objective_regret, strict=True)
        },
        "controllability_spearman_mean": control_mean,
        "paired_dominance_rate": dominance,
        "paired_dominance_rate_vs_best_baseline": dominance[best_hv_baseline],
    }

    coverage = coverage_reference if coverage_reference is not None else np.zeros(len(objective_names))
    cold_coverage = (
        cold_coverage_reference
        if cold_coverage_reference is not None
        else np.zeros(len(objective_names))
    )
    return {
        "schema_version": "pareto-evaluation-v1",
        "config": asdict(config),
        "data": {
            "slates": int(static["labels"].shape[0]),
            "slate_width": int(static["labels"].shape[1]),
            "valid_candidates": int(valid_mask.sum()),
            "cold_candidates": int(cold_mask.sum()),
        },
        "objective_names": objective_names,
        "directions": directions_array.tolist(),
        "preference_names": preference_names,
        "preference_vectors": preferences.tolist(),
        "objective_metric_coverage": {
            name: float(value)
            for name, value in zip(objective_names, coverage, strict=True)
        },
        "cold_positive_objective_coverage": {
            objective_names[index]: float(cold_coverage[index])
            for index in range(len(objective_names))
            if directions_array[index] > 0
        },
        "cold_popularity_threshold": cold_cut,
        "baseline_scope": {
            "included": list(baselines),
            "note": (
                "These are static behavior/Jev/fusion scalarizations, not separately "
                "trained EMER-like or CMR-like baseline checkpoints."
            ),
        },
        "methods": summaries,
        # Namespace consumed by configs/targets_research.yaml.
        "kuairand": {"paretojev": target_summary},
    }
