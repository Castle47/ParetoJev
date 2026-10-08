from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paretojev.cli import build_parser
from paretojev.data.behavior_prior import (
    HistoricalBayesConfig,
    historical_bayes_behavior_scores,
)
from paretojev.data.kuairand import KUAI_OBJECTIVES, PreparedKuaiRandSlates
from paretojev.data.materialize import SlateScoreTensor


def _prepared(
    *,
    user_ids: list[int],
    item_ids: list[list[int]],
    timestamps: list[list[int]],
    click_labels: list[list[float]],
) -> PreparedKuaiRandSlates:
    item_array = np.asarray(item_ids, dtype=np.int64)
    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    click_array = np.asarray(click_labels, dtype=np.float32)
    if item_array.shape != timestamp_array.shape or item_array.shape != click_array.shape:
        raise ValueError("test fixture shapes differ")
    n, k = item_array.shape
    mask = item_array >= 0
    labels = np.zeros((n, k, len(KUAI_OBJECTIVES)), dtype=np.float32)
    labels[..., 0] = click_array
    labels[~mask] = 0.0
    return PreparedKuaiRandSlates(
        user_ids=np.asarray(user_ids, dtype=np.int64),
        item_ids=item_array,
        timestamps=timestamp_array,
        labels=labels,
        mask=mask,
        duration_ms=np.where(mask, 1_000.0, 0.0).astype(np.float32),
        tab=np.where(mask, 1, -1).astype(np.int16),
        is_random=np.zeros((n, k), dtype=np.int8),
        watch_ratio=np.zeros((n, k), dtype=np.float32),
        objective_names=KUAI_OBJECTIVES,
        item_prior_count=np.zeros((n, k), dtype=np.int64),
        user_history_count=np.zeros(n, dtype=np.int64),
    )


def test_historical_prior_freezes_all_candidates_to_pre_slate_history() -> None:
    slates = _prepared(
        user_ids=[1, 1],
        item_ids=[[10, 11], [10, 12]],
        timestamps=[[100, 101], [200, 201]],
        click_labels=[[1.0, 0.0], [0.0, 0.0]],
    )
    scores, report = historical_bayes_behavior_scores(
        slates,
        config=HistoricalBayesConfig(
            alpha=1.0,
            beta=1.0,
            user_strength=1.0,
            item_strength=1.0,
        ),
    )

    # Both candidates in the first pseudo-slate see exactly the same empty history,
    # even though the first candidate later receives a positive click label.
    assert np.allclose(scores.probabilities[0, 0], scores.probabilities[0, 1])
    assert np.isclose(scores.probabilities[0, 0, 0], 0.5)

    # At the next slate, item 10 benefits from the strictly earlier positive event.
    assert scores.probabilities[1, 0, 0] > scores.probabilities[1, 1, 0]
    assert report["applied_current_events_during_scoring"] == 2
    assert report["current_unique_events"] == 4


def test_historical_prior_prefix_initializes_current_period_without_overlap() -> None:
    prefix = _prepared(
        user_ids=[7],
        item_ids=[[42, 99]],
        timestamps=[[10, 11]],
        click_labels=[[1.0, 0.0]],
    )
    current = _prepared(
        user_ids=[7],
        item_ids=[[42, 100]],
        timestamps=[[100, 101]],
        click_labels=[[0.0, 0.0]],
    )
    config = HistoricalBayesConfig(
        alpha=1.0,
        beta=1.0,
        user_strength=1.0,
        item_strength=1.0,
    )
    without_prefix, _ = historical_bayes_behavior_scores(current, config=config)
    with_prefix, report = historical_bayes_behavior_scores(
        current,
        prefix_slates=prefix,
        config=config,
    )

    assert with_prefix.probabilities[0, 0, 0] > without_prefix.probabilities[0, 0, 0]
    assert with_prefix.probabilities[0, 0, 0] > with_prefix.probabilities[0, 1, 0]
    assert report["prefix_unique_events"] == 2
    assert report["global_history_events_at_end_of_scoring"] == 2

    overlapping = _prepared(
        user_ids=[7],
        item_ids=[[42, 99]],
        timestamps=[[100, 101]],
        click_labels=[[1.0, 0.0]],
    )
    with pytest.raises(ValueError, match="strictly before"):
        historical_bayes_behavior_scores(current, prefix_slates=overlapping)


def test_historical_prior_deduplicates_events_from_overlapping_pseudo_slates() -> None:
    slates = _prepared(
        user_ids=[3, 3, 3],
        item_ids=[[1, 2], [2, 3], [4, 5]],
        timestamps=[[100, 110], [110, 120], [200, 210]],
        click_labels=[[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
    )
    _, report = historical_bayes_behavior_scores(slates)

    # Six candidate slots contain only five unique logged events because
    # (user=3, item=2, timestamp=110) is shared by two overlapping windows.
    assert report["valid_candidates"] == 6
    assert report["current_unique_events"] == 5
    assert report["applied_current_events_during_scoring"] == 3


def test_score_kuairand_behavior_prior_cli_writes_auditable_tensor(tmp_path: Path) -> None:
    slates = _prepared(
        user_ids=[1, 1],
        item_ids=[[10, 11], [10, 12]],
        timestamps=[[100, 101], [200, 201]],
        click_labels=[[1.0, 0.0], [0.0, 0.0]],
    )
    slates_path = tmp_path / "slates.npz"
    output_path = tmp_path / "behavior.npz"
    report_path = tmp_path / "behavior.report.json"
    slates.save(slates_path)

    parser = build_parser()
    args = parser.parse_args(
        [
            "score-kuairand-behavior-prior",
            str(slates_path),
            str(output_path),
            "--user-strength",
            "2",
            "--item-strength",
            "2",
            "--report",
            str(report_path),
        ]
    )
    assert args.func(args) == 0

    tensor = SlateScoreTensor.load(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert tensor.probabilities.shape == (2, 2, len(KUAI_OBJECTIVES))
    assert tensor.objective_names == KUAI_OBJECTIVES
    assert report["schema_version"] == "historical-bayes-behavior-v1"
    assert report["config"]["user_strength"] == 2.0
    assert report["provenance"]["inputs"]
