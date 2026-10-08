from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paretojev.cli import build_parser
from paretojev.data.kuairand import KUAI_OBJECTIVES, PreparedKuaiRandSlates
from paretojev.data.materialize import (
    SlateScoreTensor,
    align_decision_score_rows,
    fit_kuairand_feature_spec,
    materialize_kuairand_precomputed,
)
from paretojev.data.precomputed import PrecomputedSlateDataset
from paretojev.utils.io import write_jsonl


JEV_OBJECTIVES = (
    "click",
    "long_view",
    "like",
    "follow",
    "deep_engagement",
    "negative_feedback",
)


def _slates() -> PreparedKuaiRandSlates:
    n, k, m = 3, 3, len(KUAI_OBJECTIVES)
    mask = np.array(
        [[True, True, True], [True, True, False], [True, True, True]], dtype=bool
    )
    labels = np.zeros((n, k, m), dtype=np.float32)
    labels[0, 0, 0] = 1.0
    labels[2, 1, 2] = 1.0
    labels[2, 2, 5] = 1.0
    return PreparedKuaiRandSlates(
        user_ids=np.array([10, 11, 12], dtype=np.int64),
        item_ids=np.array([[1, 2, 3], [4, 5, -1], [6, 7, 8]], dtype=np.int64),
        timestamps=np.array(
            [[100, 101, 102], [200, 201, 0], [300, 301, 302]], dtype=np.int64
        ),
        labels=labels,
        mask=mask,
        duration_ms=np.array(
            [[1000, 2000, 3000], [1500, 2500, 0], [1200, 2200, 3200]],
            dtype=np.float32,
        ),
        tab=np.array([[1, 1, 2], [1, 3, -1], [2, 3, 4]], dtype=np.int16),
        is_random=np.zeros((n, k), dtype=np.int8),
        watch_ratio=np.zeros((n, k), dtype=np.float32),
        objective_names=KUAI_OBJECTIVES,
        item_prior_count=np.array([[0, 2, 4], [1, 3, 0], [5, 10, 20]], dtype=np.int64),
        user_history_count=np.array([2, 8, 20], dtype=np.int64),
    )


def _score_rows(slates: PreparedKuaiRandSlates, slate_ids: list[int]) -> list[dict]:
    rows = []
    for slate_id in slate_ids:
        valid = slates.mask[slate_id]
        k = int(valid.sum())
        base = np.linspace(0.1, 0.9, k * len(JEV_OBJECTIVES), dtype=np.float32).reshape(
            k, len(JEV_OBJECTIVES)
        )
        rows.append(
            {
                "schema_version": 2,
                "provider": "fake",
                "mode": "noul",
                "slate_id": slate_id,
                "candidate_ids": [str(value) for value in slates.item_ids[slate_id][valid]],
                "objective_names": list(JEV_OBJECTIVES),
                "probabilities": base.tolist(),
                "confidence": np.full_like(base, 0.8).tolist(),
                "entropy": np.full_like(base, 0.2).tolist(),
                "resolved_model": "fake-jev",
            }
        )
    return rows


def _behavior(slates: PreparedKuaiRandSlates) -> SlateScoreTensor:
    rng = np.random.default_rng(4)
    values = rng.uniform(
        0.05,
        0.95,
        size=(len(slates.user_ids), slates.item_ids.shape[1], len(JEV_OBJECTIVES)),
    ).astype(np.float32)
    values[~slates.mask] = 0.0
    return SlateScoreTensor(
        slate_ids=np.arange(len(slates.user_ids)),
        candidate_ids=slates.item_ids.copy(),
        probabilities=values,
        mask=slates.mask.copy(),
        objective_names=JEV_OBJECTIVES,
    )


def test_materialize_selected_kuairand_slates_with_objective_aliases(tmp_path: Path) -> None:
    slates = _slates()
    selected = [2, 0]
    spec = fit_kuairand_feature_spec(slates, selected, max_tab_values=3)
    output = tmp_path / "features.npz"
    report = materialize_kuairand_precomputed(
        slates,
        selected,
        behavior_scores=_behavior(slates),
        decision_rows=_score_rows(slates, selected),
        feature_spec=spec,
        output_path=output,
    )

    dataset = PrecomputedSlateDataset(output)
    assert len(dataset) == 2
    assert dataset.n_objectives == len(KUAI_OBJECTIVES)
    assert dataset.feature_dim == len(spec.feature_names)
    assert report["slate_id_min"] == 0
    assert report["slate_id_max"] == 2
    with np.load(output, allow_pickle=False) as data:
        assert data["slate_ids"].tolist() == selected
        # negative_feedback in the Jev rows must align to is_hate in KuaiRand.
        expected = _score_rows(slates, selected)[0]["probabilities"][0][-1]
        assert np.isclose(data["jev_scores"][0, 0, -1], expected)
        assert data["feature_names"].tolist() == list(spec.feature_names)


def test_decision_alignment_rejects_candidate_order_mismatch() -> None:
    slates = _slates()
    rows = _score_rows(slates, [0])
    rows[0]["candidate_ids"] = list(reversed(rows[0]["candidate_ids"]))
    with pytest.raises(ValueError, match="candidate order mismatch"):
        align_decision_score_rows(slates, [0], rows)


def test_materialize_kuairand_cli_fits_and_reuses_feature_contract(tmp_path: Path) -> None:
    slates = _slates()
    slates_path = tmp_path / "slates.npz"
    slates.save(slates_path)
    selected = [2, 0]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"sample_order": [0, 1], "slate_id": selected}
    ).to_csv(manifest, index=False)
    decision_path = tmp_path / "decision.jsonl"
    write_jsonl(decision_path, _score_rows(slates, selected))
    behavior_path = tmp_path / "behavior.npz"
    _behavior(slates).save(behavior_path)
    output = tmp_path / "precomputed.npz"
    feature_spec = tmp_path / "feature_spec.json"
    report_path = tmp_path / "report.json"

    parser = build_parser()
    args = parser.parse_args(
        [
            "materialize-kuairand",
            str(slates_path),
            str(decision_path),
            str(behavior_path),
            str(output),
            "--manifest",
            str(manifest),
            "--feature-spec",
            str(feature_spec),
            "--fit-feature-spec",
            "--max-tab-values",
            "3",
            "--report",
            str(report_path),
        ]
    )
    assert args.func(args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["contract_validation"]["slates"] == 2
    assert report["feature_spec"]["fit_on_this_run"] is True
    assert feature_spec.exists()


def test_feature_spec_is_permutation_safe_by_default_and_replays_v1(tmp_path: Path) -> None:
    slates = _slates()
    default_spec = fit_kuairand_feature_spec(slates, [0, 1, 2], max_tab_values=3)
    assert default_spec.include_sequence_position is False
    assert "sequence_position_fraction" not in default_spec.feature_names

    legacy_path = tmp_path / "legacy_feature_spec.json"
    legacy_path.write_text(
        json.dumps(
            {
                "log_prior_mean": 0.0,
                "log_prior_std": 1.0,
                "log_duration_mean": 0.0,
                "log_duration_std": 1.0,
                "tab_values": [1, 2],
                "max_tab_values": 3,
                "schema_version": "kuairand-candidate-features-v1",
            }
        ),
        encoding="utf-8",
    )
    legacy = type(default_spec).load(legacy_path)
    assert legacy.include_sequence_position is True
    assert "sequence_position_fraction" in legacy.feature_names
