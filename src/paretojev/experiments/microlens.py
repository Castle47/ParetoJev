from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paretojev.data.microlens import grouped_score_arrays, per_user_metric_matrix
from paretojev.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from paretojev.metrics.ranking import (
    TiePolicy,
    break_score_ties,
    paired_bootstrap_difference,
    positive_ranks,
    single_positive_metrics,
)

METRIC_NAMES = ("auc", "hr@1", "hr@5", "ndcg@10", "mrr")


def _validate_probability_values(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0,1]")
    return array


def _average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Ascending average percentile rank in [0, 1], preserving exact ties."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    if len(values) <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return ranks / (len(values) - 1)


def _slate_feature_frame(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Create leakage-safe per-candidate stacking features.

    The stacker receives three views of every base score:

    * the raw value, so absolute Jev probability/confidence can matter;
    * its within-slate percentile rank, which is robust to incompatible scales;
    * its within-slate z-score, which captures relative margin.

    Every transformation is candidate/slate local and is therefore safe to
    compute before the user-level train/validation split.
    """

    missing = set(score_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing stacking columns: {sorted(missing)}")
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    users: list[np.ndarray] = []
    row_positions: list[np.ndarray] = []
    if not frame.index.is_unique:
        raise ValueError("Candidate frame index must be unique")
    position_lookup = pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
    feature_names = [f"raw::{column}" for column in score_columns]
    feature_names += [f"rank::{column}" for column in score_columns]
    feature_names += [f"z::{column}" for column in score_columns]
    pairs = list(itertools.combinations(range(len(score_columns)), 2))
    feature_names += [
        f"rank_absdiff::{score_columns[left]}::{score_columns[right]}"
        for left, right in pairs
    ]
    feature_names += [
        f"rank_product::{score_columns[left]}::{score_columns[right]}"
        for left, right in pairs
    ]
    feature_names += [
        f"raw_product::{score_columns[left]}::{score_columns[right]}"
        for left, right in pairs
    ]
    feature_names += [
        f"z_product::{score_columns[left]}::{score_columns[right]}"
        for left, right in pairs
    ]

    for user, group in frame.groupby("user", sort=False):
        raw = np.stack(
            [group[column].to_numpy(dtype=np.float64) for column in score_columns],
            axis=1,
        )
        if not np.isfinite(raw).all():
            raise ValueError(f"Non-finite stacking score for user {user}")
        ranks = np.stack(
            [_average_percentile_rank(raw[:, index]) for index in range(raw.shape[1])],
            axis=1,
        )
        mean = raw.mean(axis=0, keepdims=True)
        scale = raw.std(axis=0, keepdims=True)
        scale[scale < 1e-12] = 1.0
        zscore = (raw - mean) / scale
        extras: list[np.ndarray] = []
        for left, right in pairs:
            extras.append(np.abs(ranks[:, left] - ranks[:, right])[:, None])
        for left, right in pairs:
            extras.append((ranks[:, left] * ranks[:, right])[:, None])
        for left, right in pairs:
            extras.append((raw[:, left] * raw[:, right])[:, None])
        for left, right in pairs:
            extras.append((zscore[:, left] * zscore[:, right])[:, None])
        base_features = [raw, ranks, zscore]
        features = np.concatenate([*base_features, *extras], axis=1)
        feature_rows.append(features)
        labels.append(group["y"].to_numpy(dtype=np.int64))
        users.append(np.full(len(group), int(user), dtype=np.int64))
        row_positions.append(position_lookup.loc[group.index].to_numpy(dtype=np.int64))
    return (
        np.concatenate(feature_rows, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(users, axis=0),
        np.concatenate(row_positions, axis=0),
        feature_names,
    )


def _user_folds(users: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(int(value) for value in users)), dtype=np.int64)
    if folds < 2 or folds > len(unique):
        raise ValueError("folds must be between 2 and the number of users")
    rng = np.random.default_rng(seed)
    return [part for part in np.array_split(unique[rng.permutation(len(unique))], folds) if len(part)]


def _choice_temperature_matrix(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = _validate_probability_values(scores, name="choice probabilities")
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every choice slate must have positive probability mass")
    probabilities = probabilities / totals
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _choice_nll(scores: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("Choice scores and labels must have shape [N,K]")
    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError("Every Choice slate must contain exactly one positive")
    calibrated = _choice_temperature_matrix(scores, temperature)
    positive = calibrated[labels.astype(bool)]
    return float(-np.log(np.clip(positive, 1e-12, 1.0)).mean())


def cross_validated_choice_temperature(
    frame: pd.DataFrame,
    score_column: str,
    *,
    folds: int = 5,
    temperatures: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
    seed: int = 0,
    output_column: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Calibrate Choice distributions with user-level out-of-fold temperature.

    Temperature is selected only on the outer training users by multiclass NLL
    and then applied to the held-out users. The transform is monotonic within a
    slate, so it changes calibration without manufacturing a ranking gain.
    """

    if score_column not in frame:
        raise ValueError(f"Missing Choice probability column: {score_column}")
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must contain positive values")
    output_column = output_column or f"{score_column}_temp_cv"
    if output_column in frame:
        raise ValueError(f"Output column already exists: {output_column}")

    users, scores, labels = grouped_score_arrays(frame, score_column)
    fold_users = _user_folds(users, folds, seed)
    result = frame.copy()
    result[output_column] = np.nan
    reports: list[dict[str, object]] = []

    for fold_index, held_out_array in enumerate(fold_users):
        held_out = set(int(value) for value in held_out_array)
        held_mask = np.isin(users, list(held_out))
        train_mask = ~held_mask
        candidates: list[dict[str, float]] = []
        best_temperature: float | None = None
        best_nll = float("inf")
        for value in temperatures:
            nll = _choice_nll(scores[train_mask], labels[train_mask], float(value))
            candidates.append({"temperature": float(value), "train_nll": nll})
            # Resolve numerical ties toward no calibration (T=1), then lower T.
            if nll < best_nll - 1e-12 or (
                np.isclose(nll, best_nll)
                and (
                    best_temperature is None
                    or (abs(np.log(value)), value)
                    < (abs(np.log(best_temperature)), best_temperature)
                )
            ):
                best_nll = nll
                best_temperature = float(value)

        assert best_temperature is not None
        transformed = _choice_temperature_matrix(
            scores[held_mask], best_temperature
        )
        transformed_by_user = {
            int(user): row
            for user, row in zip(users[held_mask], transformed, strict=True)
        }
        for user in held_out:
            index = result.index[result["user"] == user]
            result.loc[index, output_column] = transformed_by_user[user]
        reports.append(
            {
                "fold": fold_index,
                "held_out_users": sorted(held_out),
                "selected_temperature": best_temperature,
                "train_nll": best_nll,
                "temperature_search": candidates,
            }
        )

    if result[output_column].isna().any():
        raise RuntimeError("Some out-of-fold temperature scores were not assigned")
    return result, reports


def cross_validated_platt_calibration(
    frame: pd.DataFrame,
    score_column: str,
    *,
    folds: int = 5,
    seed: int = 0,
    output_column: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Group-CV Platt calibration for pointwise Noul/binary probabilities."""

    if score_column not in frame:
        raise ValueError(f"Missing probability column: {score_column}")
    output_column = output_column or f"{score_column}_platt_cv"
    if output_column in frame:
        raise ValueError(f"Output column already exists: {output_column}")
    raw = _validate_probability_values(
        frame[score_column].to_numpy(dtype=np.float64), name=score_column
    )
    logit = np.log(np.clip(raw, 1e-8, 1 - 1e-8)) - np.log1p(
        -np.clip(raw, 1e-8, 1 - 1e-8)
    )
    users = frame["user"].to_numpy(dtype=np.int64)
    unique_users = np.asarray(sorted(frame["user"].unique()), dtype=np.int64)
    fold_users = _user_folds(unique_users, folds, seed)
    result = frame.copy()
    result[output_column] = np.nan
    reports: list[dict[str, object]] = []

    for fold_index, held_out_array in enumerate(fold_users):
        held_out = set(int(value) for value in held_out_array)
        held_mask = np.isin(users, list(held_out))
        train_mask = ~held_mask
        labels = frame.loc[train_mask, "y"].to_numpy(dtype=np.int64)
        if len(np.unique(labels)) != 2:
            raise ValueError("Platt training labels must contain both classes")
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        model.fit(logit[train_mask, None], labels)
        output_position = result.columns.get_loc(output_column)
        held_positions = np.flatnonzero(held_mask)
        result.iloc[held_positions, output_position] = model.predict_proba(
            logit[held_mask, None]
        )[:, 1]
        reports.append(
            {
                "fold": fold_index,
                "held_out_users": sorted(held_out),
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
            }
        )

    if result[output_column].isna().any():
        raise RuntimeError("Some out-of-fold Platt scores were not assigned")
    return result, reports


def probability_diagnostics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    n_bins: int = 15,
    normalize_as_choice: bool = False,
) -> dict[str, object]:
    """Report candidate-level and slate-level probability calibration."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    _, scores, labels = grouped_score_arrays(frame, score_column)
    scores = _validate_probability_values(scores, name=score_column)
    if normalize_as_choice:
        totals = scores.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("Every choice slate must have positive probability mass")
        scores = scores / totals
    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    binary_nll = -np.mean(
        labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)
    )
    positive_nll = -np.log(clipped[labels.astype(bool)]).mean()
    sums = scores.sum(axis=1)
    return {
        "brier": brier_score(scores, labels),
        "ece": expected_calibration_error(scores, labels, n_bins=n_bins),
        "candidate_binary_nll": float(binary_nll),
        "positive_item_nll": float(positive_nll),
        "mean_slate_probability_sum": float(sums.mean()),
        "std_slate_probability_sum": float(sums.std()),
        "min_probability": float(scores.min()),
        "max_probability": float(scores.max()),
        "reliability_curve": reliability_curve(scores, labels, n_bins=n_bins),
    }


def _objective_value(metrics: dict[str, float], objective: str) -> float:
    if objective == "auc":
        return metrics["auc"]
    if objective == "ndcg@10":
        return metrics["ndcg@10"]
    if objective == "balanced":
        return (metrics["auc"] + metrics["hr@5"] + metrics["ndcg@10"]) / 3.0
    raise ValueError(f"Unknown objective: {objective}")


def _fit_stacker(features: np.ndarray, labels: np.ndarray, c_value: float) -> Pipeline:
    if len(np.unique(labels)) != 2:
        raise ValueError("Stacker training labels must contain both classes")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


def _candidate_predictions_to_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    selected_users: set[int],
    *,
    tie_policy: TiePolicy,
    tie_seed: int,
) -> dict[str, float]:
    temporary = frame.copy()
    temporary["__prediction"] = predictions
    selected = temporary[temporary["user"].isin(selected_users)]
    groups = list(selected.groupby("user", sort=False))
    scores = np.stack([group["__prediction"].to_numpy() for _, group in groups])
    labels = np.stack([group["y"].to_numpy() for _, group in groups])
    return single_positive_metrics(
        scores,
        labels,
        ks=(1, 5, 10),
        tie_policy=tie_policy,
        tie_seed=tie_seed,
    )


def cross_validated_logistic_stacking(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    folds: int = 5,
    inner_folds: int = 4,
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    objective: str = "balanced",
    seed: int = 0,
    output_column: str = "stack_cv",
    tie_policy: TiePolicy = "upstream_jitter",
    tie_seed: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Nested group-CV logistic stacker over fixed base-model scores.

    Outer held-out users receive predictions from a model that never saw their
    candidate labels. Hyperparameter selection is performed only inside the
    outer training users, so the final 200-user MicroLens report remains honest.
    """

    if len(score_columns) < 2:
        raise ValueError("At least two score columns are required for stacking")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_values must contain positive values")
    features, labels, candidate_users, row_positions, feature_names = _slate_feature_frame(
        frame, score_columns
    )
    outer = _user_folds(candidate_users, folds, seed)
    result = frame.copy()
    result[output_column] = np.nan
    reports: list[dict[str, object]] = []

    for outer_index, held_out_users_array in enumerate(outer):
        held_out_users = set(int(value) for value in held_out_users_array)
        train_users = np.asarray(
            sorted(set(int(value) for value in candidate_users) - held_out_users),
            dtype=np.int64,
        )
        inner_count = min(inner_folds, len(train_users))
        if inner_count < 2:
            raise ValueError("Not enough outer-training users for inner CV")
        inner = _user_folds(train_users, inner_count, seed + 10_000 + outer_index)
        c_reports: list[dict[str, object]] = []
        best_c: float | None = None
        best_value = -float("inf")

        for c_value in c_values:
            inner_values: list[float] = []
            inner_metrics: list[dict[str, float]] = []
            for inner_held_array in inner:
                inner_held = set(int(value) for value in inner_held_array)
                train_mask = np.isin(candidate_users, train_users) & ~np.isin(
                    candidate_users, list(inner_held)
                )
                validation_mask = np.isin(candidate_users, list(inner_held))
                model = _fit_stacker(features[train_mask], labels[train_mask], c_value)
                predictions = np.full(len(frame), np.nan, dtype=np.float64)
                predictions[row_positions[validation_mask]] = model.predict_proba(
                    features[validation_mask]
                )[:, 1]
                metrics = _candidate_predictions_to_metrics(
                    frame,
                    predictions,
                    inner_held,
                    tie_policy=tie_policy,
                    tie_seed=tie_seed,
                )
                inner_metrics.append(metrics)
                inner_values.append(_objective_value(metrics, objective))
            mean_value = float(np.mean(inner_values))
            c_reports.append(
                {
                    "c": float(c_value),
                    "objective_mean": mean_value,
                    "fold_metrics": inner_metrics,
                }
            )
            if mean_value > best_value + 1e-12 or (
                np.isclose(mean_value, best_value) and (best_c is None or c_value < best_c)
            ):
                best_value = mean_value
                best_c = float(c_value)

        assert best_c is not None
        outer_train_mask = np.isin(candidate_users, train_users)
        outer_test_mask = np.isin(candidate_users, list(held_out_users))
        model = _fit_stacker(features[outer_train_mask], labels[outer_train_mask], best_c)
        held_out_predictions = model.predict_proba(features[outer_test_mask])[:, 1]
        output_position = result.columns.get_loc(output_column)
        result.iloc[row_positions[outer_test_mask], output_position] = held_out_predictions
        logistic = model.named_steps["logistic"]
        reports.append(
            {
                "fold": outer_index,
                "held_out_users": sorted(held_out_users),
                "selected_c": best_c,
                "inner_objective": best_value,
                "c_search": c_reports,
                "feature_names": feature_names,
                "coefficients": {
                    name: float(value)
                    for name, value in zip(feature_names, logistic.coef_[0], strict=True)
                },
                "intercept": float(logistic.intercept_[0]),
            }
        )

    if result[output_column].isna().any():
        raise RuntimeError("Some out-of-fold stacker scores were not assigned")
    return result, reports


def add_reciprocal_rank_fusion(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    rank_constant: float = 60.0,
    output_column: str = "rrf",
) -> pd.DataFrame:
    if len(score_columns) < 2:
        raise ValueError("At least two score columns are required for RRF")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    missing = set(score_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing RRF columns: {sorted(missing)}")
    result = frame.copy()
    result[output_column] = np.nan
    for _, group in result.groupby("user", sort=False):
        fused = np.zeros(len(group), dtype=np.float64)
        for column in score_columns:
            percentile = _average_percentile_rank(group[column].to_numpy(dtype=np.float64))
            descending_rank = len(group) - percentile * max(len(group) - 1, 1)
            fused += 1.0 / (rank_constant + descending_rank)
        result.loc[group.index, output_column] = fused
    return result


def _validate_bootstrap_args(n_bootstrap: int, confidence: float) -> None:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    score_column: str,
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
    tie_policy: TiePolicy = "upstream_jitter",
    tie_seed: int = 1,
) -> dict[str, dict[str, float]]:
    _validate_bootstrap_args(n_bootstrap, confidence)
    matrix = per_user_metric_matrix(
        frame, score_column, tie_policy=tie_policy, tie_seed=tie_seed
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(matrix), size=(n_bootstrap, len(matrix)))
    boot = matrix[samples].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    output: dict[str, dict[str, float]] = {}
    for index, metric in enumerate(METRIC_NAMES):
        low, high = np.quantile(boot[:, index], [alpha, 1.0 - alpha])
        output[metric] = {
            "mean": float(matrix[:, index].mean()),
            "low": float(low),
            "high": float(high),
            "confidence": float(confidence),
        }
    return output


def pairwise_metric_differences(
    frame: pd.DataFrame,
    candidate_column: str,
    reference_column: str,
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
    tie_policy: TiePolicy = "upstream_jitter",
    tie_seed: int = 1,
) -> dict[str, dict[str, float | bool]]:
    _validate_bootstrap_args(n_bootstrap, confidence)
    candidate = per_user_metric_matrix(
        frame, candidate_column, tie_policy=tie_policy, tie_seed=tie_seed
    )
    reference = per_user_metric_matrix(
        frame, reference_column, tie_policy=tie_policy, tie_seed=tie_seed
    )
    output: dict[str, dict[str, float | bool]] = {}
    for index, metric in enumerate(METRIC_NAMES):
        mean, low, high = paired_bootstrap_difference(
            candidate[:, index],
            reference[:, index],
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + index,
        )
        output[metric] = {
            "difference": mean,
            "low": low,
            "high": high,
            "positive_interval": bool(low > 0),
        }
    return output


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_percentile_rank(left)
    right_rank = _average_percentile_rank(right)
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def complementarity_report(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    tie_policy: TiePolicy = "upstream_jitter",
    tie_seed: int = 1,
) -> dict[str, dict[str, float | dict[str, float]]]:
    """Quantify whether model errors leave room for a hybrid to win."""

    if len(score_columns) < 2:
        raise ValueError("At least two score columns are required")
    missing = set(score_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing complementarity columns: {sorted(missing)}")

    report: dict[str, dict[str, float | dict[str, float]]] = {}
    grouped = list(frame.groupby("user", sort=False))
    for left, right in itertools.combinations(score_columns, 2):
        _, left_scores, labels = grouped_score_arrays(frame, left)
        _, right_scores, _ = grouped_score_arrays(frame, right)
        left_ranks = positive_ranks(
            left_scores, labels, tie_policy=tie_policy, tie_seed=tie_seed
        )
        right_ranks = positive_ranks(
            right_scores, labels, tie_policy=tie_policy, tie_seed=tie_seed
        )
        correlations = [
            _rank_correlation(
                group[left].to_numpy(dtype=np.float64),
                group[right].to_numpy(dtype=np.float64),
            )
            for _, group in grouped
        ]
        evaluated_left = break_score_ties(
            left_scores, policy=tie_policy, seed=tie_seed
        )
        evaluated_right = break_score_ties(
            right_scores, policy=tie_policy, seed=tie_seed
        )
        top1_left = np.argmax(evaluated_left, axis=1)
        top1_right = np.argmax(evaluated_right, axis=1)
        top5_jaccard: list[float] = []
        for left_row, right_row in zip(evaluated_left, evaluated_right, strict=True):
            left_top = set(np.argsort(-left_row, kind="stable")[:5].tolist())
            right_top = set(np.argsort(-right_row, kind="stable")[:5].tolist())
            top5_jaccard.append(len(left_top & right_top) / len(left_top | right_top))
        oracle_ranks = np.minimum(left_ranks, right_ranks)
        oracle_metrics = {
            "hr@1": float((oracle_ranks <= 1).mean()),
            "hr@5": float((oracle_ranks <= 5).mean()),
            "ndcg@10": float(
                np.where(
                    oracle_ranks <= 10,
                    1.0 / np.log2(oracle_ranks + 1.0),
                    0.0,
                ).mean()
            ),
            "mrr": float((1.0 / oracle_ranks).mean()),
        }
        left_miss5 = left_ranks > 5
        right_miss5 = right_ranks > 5
        report[f"{left}__vs__{right}"] = {
            "mean_spearman_rank_correlation": float(np.mean(correlations)),
            "top1_agreement": float((top1_left == top1_right).mean()),
            "mean_top5_jaccard": float(np.mean(top5_jaccard)),
            "left_positive_rank_win_rate": float((left_ranks < right_ranks).mean()),
            "right_positive_rank_win_rate": float((right_ranks < left_ranks).mean()),
            "positive_rank_tie_rate": float((right_ranks == left_ranks).mean()),
            "union_hr@5": float((~(left_miss5 & right_miss5)).mean()),
            "right_rescues_left_miss@5": float(
                ((left_miss5) & (~right_miss5)).sum() / max(left_miss5.sum(), 1)
            ),
            "left_rescues_right_miss@5": float(
                ((right_miss5) & (~left_miss5)).sum() / max(right_miss5.sum(), 1)
            ),
            "oracle": oracle_metrics,
        }
    return report
