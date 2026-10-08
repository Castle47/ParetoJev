from __future__ import annotations

import numpy as np
import pandas as pd

from paretojev.data.microlens import evaluate_score_column
from paretojev.experiments.microlens import (
    add_reciprocal_rank_fusion,
    bootstrap_metric_intervals,
    complementarity_report,
    cross_validated_choice_temperature,
    cross_validated_logistic_stacking,
    cross_validated_platt_calibration,
    pairwise_metric_differences,
    probability_diagnostics,
)


def _candidate_frame(n_users: int = 24, slate_size: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    for user in range(n_users):
        positive = user % slate_size
        for position in range(slate_size):
            y = int(position == positive)
            # A and B have deliberately non-identical errors.
            score_a = rng.normal(0.0, 0.4) + (1.5 * y if user % 2 == 0 else 0.15 * y)
            score_b = rng.normal(0.0, 0.4) + (1.5 * y if user % 2 == 1 else 0.15 * y)
            rows.append(
                {
                    "user": user,
                    "item": user * 10 + position,
                    "position": position,
                    "y": y,
                    "score_a": score_a,
                    "score_b": score_b,
                }
            )
    return pd.DataFrame(rows)


def test_rrf_stacking_bootstrap_and_complementarity() -> None:
    frame = _candidate_frame().sample(frac=1.0, random_state=9)
    rrf = add_reciprocal_rank_fusion(frame, ["score_a", "score_b"])
    assert rrf["rrf"].notna().all()

    stacked, reports = cross_validated_logistic_stacking(
        rrf,
        ["score_a", "score_b"],
        folds=4,
        inner_folds=3,
        c_values=(0.1, 1.0),
        seed=3,
    )
    assert stacked["stack_cv"].notna().all()
    assert len(reports) == 4
    metrics = evaluate_score_column(stacked, "stack_cv")
    assert 0.0 <= metrics["auc"] <= 1.0

    intervals = bootstrap_metric_intervals(
        stacked, "stack_cv", n_bootstrap=200, seed=4
    )
    assert intervals["auc"]["low"] <= intervals["auc"]["mean"] <= intervals["auc"]["high"]

    differences = pairwise_metric_differences(
        stacked,
        "stack_cv",
        "score_a",
        n_bootstrap=200,
        seed=5,
    )
    assert set(differences) == {"auc", "hr@1", "hr@5", "ndcg@10", "mrr"}

    complementarity = complementarity_report(stacked, ["score_a", "score_b"])
    pair = complementarity["score_a__vs__score_b"]
    left_hr5 = evaluate_score_column(stacked, "score_a")["hr@5"]
    right_hr5 = evaluate_score_column(stacked, "score_b")["hr@5"]
    assert pair["union_hr@5"] >= max(left_hr5, right_hr5)
    assert 0.0 <= pair["top1_agreement"] <= 1.0


def test_group_cv_probability_calibration() -> None:
    frame = _candidate_frame(n_users=30, slate_size=5)
    frame["choice"] = frame.groupby("user", group_keys=False)["score_a"].transform(
        lambda values: np.exp(values - values.max())
        / np.exp(values - values.max()).sum()
    )
    frame["noul"] = 1.0 / (1.0 + np.exp(-frame["score_b"]))

    calibrated_choice, choice_reports = cross_validated_choice_temperature(
        frame,
        "choice",
        folds=5,
        temperatures=(0.5, 1.0, 2.0),
        seed=7,
    )
    assert len(choice_reports) == 5
    sums = calibrated_choice.groupby("user")["choice_temp_cv"].sum().to_numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-10)
    before = evaluate_score_column(calibrated_choice, "choice")
    after = evaluate_score_column(calibrated_choice, "choice_temp_cv")
    assert before == after

    calibrated_noul, platt_reports = cross_validated_platt_calibration(
        calibrated_choice,
        "noul",
        folds=5,
        seed=8,
    )
    assert len(platt_reports) == 5
    assert calibrated_noul["noul_platt_cv"].between(0.0, 1.0).all()

    choice_diagnostics = probability_diagnostics(
        calibrated_noul,
        "choice_temp_cv",
        normalize_as_choice=True,
        n_bins=5,
    )
    noul_diagnostics = probability_diagnostics(
        calibrated_noul,
        "noul_platt_cv",
        n_bins=5,
    )
    assert np.isclose(choice_diagnostics["mean_slate_probability_sum"], 1.0)
    assert 0.0 <= choice_diagnostics["ece"] <= 1.0
    assert 0.0 <= noul_diagnostics["brier"] <= 1.0


def test_bootstrap_argument_validation() -> None:
    frame = _candidate_frame()
    try:
        bootstrap_metric_intervals(frame, "score_a", n_bootstrap=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid bootstrap count to fail")
