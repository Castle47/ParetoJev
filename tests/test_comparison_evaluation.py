from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from paretojev.data.schema import SlateBatch
from paretojev.evaluation.comparison import comparison_objective_metrics, evaluate_comparison
from paretojev.evaluation.pareto import ParetoEvaluationConfig, objective_utility_metrics


class FixedRanker(torch.nn.Module):
    def __init__(self, feature: int) -> None:
        super().__init__()
        self.feature = feature
        self.calls = 0

    def forward(self, batch: SlateBatch) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {"score_mean": batch.candidate_features[..., self.feature]}


def batch(n: int = 4) -> SlateBatch:
    labels = torch.tensor([[[0.25, 0.5], [1.0, 0.0], [0.0, 1.0]]]).repeat(n, 1, 1)
    features = torch.tensor([[[3.0, 0.0], [2.0, 3.0], [1.0, 2.0]]]).repeat(n, 1, 1)
    return SlateBatch(
        candidate_features=features, behavior_scores=labels.clone(), jev_scores=labels.clone(),
        labels=labels, preference_weights=torch.full((n, 2), 0.5),
        mask=torch.ones((n, 3), dtype=torch.bool), jev_confidence=torch.ones_like(labels),
        jev_entropy=torch.zeros_like(labels), popularity=torch.tensor([[[0.1], [0.2], [0.9]]]).repeat(n, 1, 1),
        history_length=torch.ones((n, 1)),
    )


def compare(models=None, batches=None, **kwargs):
    return evaluate_comparison(
        models or {"emer_like": FixedRanker(0), "cmr_like": FixedRanker(1), "paretojev": FixedRanker(0)},
        batches if batches is not None else [batch()],
        directions=[1, -1], objective_names=["click", "risk"],
        config=ParetoEvaluationConfig(top_k=1, random_preferences=1, hypervolume_samples=2_000, seed=13),
        cold_popularity_threshold=kwargs.pop("cold_popularity_threshold", 0.3), **kwargs,
    )


def test_comparison_uses_named_checkpoint_and_shared_protocol() -> None:
    models = {"emer_like": FixedRanker(0), "cmr_like": FixedRanker(1), "paretojev": FixedRanker(0)}
    original = batch()
    preferences = original.preference_weights.clone()
    report = compare(models, iter([original]))
    target = report["kuairand"]["paretojev"]
    assert report["schema_version"] == "checkpoint-comparison-v1"
    assert target["baseline"] == "emer_like"
    assert target["hypervolume_relative_gain"] == 0.0
    assert report["methods"]["cmr_like"]["hypervolume"] > target["hypervolume"]
    assert target["worst_objective_regret"] == 0.0
    assert target["paired_dominance_rate"] == 0.0
    assert report["methods"]["emer_like"]["points"] == report["methods"]["paretojev"]["points"]
    assert all(model.calls == 4 for model in models.values())
    cmr_target = report["kuairand"]["paretojev_vs_cmr"]
    assert cmr_target["baseline"] == "cmr_like"
    assert cmr_target["hypervolume_relative_gain"] < 0
    assert cmr_target["worst_objective_regret"] == 0.75
    assert all(model.training for model in models.values())
    assert torch.equal(original.preference_weights, preferences)
    json.dumps(report, allow_nan=False)


def test_missing_cold_and_uncovered_objective_are_null() -> None:
    report = compare(cold_popularity_threshold=-1)
    assert report["data"]["cold_candidates"] == 0
    assert report["cold_positive_objective_coverage"] == {"click": 0.0}
    for method in report["methods"].values():
        assert method["cold_positive_ndcg_mean"] is None
    assert report["kuairand"]["paretojev"]["cold_item_ndcg_relative_gain"] is None
    uncovered = batch()
    uncovered.labels[..., 0] = 0
    report = compare(batches=[uncovered])
    assert report["objective_metric_coverage"]["click"] == 0
    for method in report["methods"].values():
        assert method["points"][0][0] is None
        assert method["hypervolume"] is None
        assert method["controllability"]["mean"] is None
    assert report["kuairand"]["paretojev"]["worst_objective_regret"] is None
    json.dumps(report, allow_nan=False)


def test_zero_baseline_has_no_relative_gain_and_regret_uses_baseline_frontier() -> None:
    data = batch()
    data.labels[:, 0, :] = torch.tensor([0.0, 1.0])
    models = {"emer_like": FixedRanker(0), "cmr_like": FixedRanker(0), "paretojev": FixedRanker(1)}
    target = compare(models, [data])["kuairand"]["paretojev"]
    assert target["baseline_hypervolume"] == 0.0
    assert target["hypervolume_absolute_gain"] == 1.0
    assert target["hypervolume_relative_gain"] is None
    assert target["cold_item_ndcg_relative_gain"] is None
    assert target["worst_objective_regret"] == 0.0
    reversed_models = {"emer_like": FixedRanker(1), "paretojev": FixedRanker(0)}
    target = compare(reversed_models)["kuairand"]["paretojev"]
    assert target["per_objective_regret"] == {"click": 0.75, "risk": 0.5}
    assert target["worst_objective_regret"] == 0.75


def test_audit_reconstructs_metrics_with_masks_and_unequal_slate_lengths(tmp_path: Path) -> None:
    data = batch()
    data.mask[0, 1:] = False
    data.labels[0, 1:] = float("nan")
    data.popularity[0, 1:] = float("nan")
    data.mask[1, :] = False
    path = tmp_path / "audit.npz"
    report = compare(batches=[data], audit_path=path, group_ids=["a", "a", "b", "c"])
    with np.load(path, allow_pickle=False) as audit:
        assert audit["scores"].shape == (3, 4, 4, 3)
        assert audit["utility_numerator"].shape == (3, 4, 4, 2)
        assert audit["group_ids"].tolist() == ["a", "a", "b", "c"]
        for index, method in enumerate(audit["method_names"]):
            actual = audit["utility_numerator"][index].sum(axis=1) / audit["utility_denominator"][index].sum(axis=1)
            np.testing.assert_allclose(actual, report["methods"][str(method)]["points"])
            reference, coverage = objective_utility_metrics(
                audit["scores"][index, 0], audit["labels"], audit["mask"], [1, -1], top_k=1,
            )
            np.testing.assert_allclose(reference, actual[0])
        assert report["objective_metric_coverage"] == {"click": 0.75, "risk": 0.75}


def test_group_bootstrap_is_paired_deterministic_and_requires_multiple_groups() -> None:
    first = compare(group_ids=["a", "a", "b", "c"], bootstrap_samples=50)
    second = compare(group_ids=["a", "a", "b", "c"], bootstrap_samples=50)
    bootstrap = first["paired_user_bootstrap"]
    assert bootstrap == second["paired_user_bootstrap"]
    assert bootstrap["groups"] == 3
    comparison = bootstrap["comparisons"]["paretojev"]
    assert comparison["objective_mean_absolute_gain"]["click"]["interval"] == [0.0, 0.0]
    assert comparison["cold_ndcg_absolute_gain"]["interval"] == [0.0, 0.0]
    one_group = compare(group_ids=["a"] * 4, bootstrap_samples=50)["paired_user_bootstrap"]
    assert one_group["status"].startswith("unavailable")
    assert one_group["comparisons"]["paretojev"]["cold_ndcg_absolute_gain"]["interval"] is None
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("kwargs,match", [
    ({"bootstrap_samples": 10}, "group IDs"),
    ({"bootstrap_samples": -1}, "negative"),
    ({"cold_popularity_threshold": float("nan")}, "finite"),
    ({"group_ids": ["a"]}, "per evaluated slate"),
])
def test_invalid_protocol_inputs_are_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        compare(**kwargs)


def test_invalid_predictions_restore_model_mode() -> None:
    data = batch()
    data.candidate_features[0, 0, 0] = float("nan")
    model = FixedRanker(0)
    model.eval()
    with pytest.raises(ValueError, match="finite valid scores"):
        compare({"emer_like": model, "paretojev": FixedRanker(1)}, [data])
    assert not model.training


def test_observed_labels_do_not_change_checkpoint_predictions(tmp_path: Path) -> None:
    first_path, second_path = tmp_path / "first.npz", tmp_path / "second.npz"
    data = batch()
    compare(batches=[data], audit_path=first_path)
    data.labels = 1.0 - data.labels
    compare(batches=[data], audit_path=second_path)
    with np.load(first_path) as first, np.load(second_path) as second:
        np.testing.assert_array_equal(first["scores"], second["scores"])


def test_bootstrap_withholds_interval_when_rare_objective_disappears() -> None:
    data = batch()
    data.labels[1:, :, 0] = 0.0
    result = compare(batches=[data], group_ids=["a", "b", "c", "d"], bootstrap_samples=100)
    comparison = result["paired_user_bootstrap"]["comparisons"]["paretojev"]
    for value in (comparison["objective_mean_absolute_gain"]["click"], comparison["cold_ndcg_absolute_gain"]):
        assert value["interval"] is None
        assert 0 < value["valid_replicates"] < 100


def test_pooled_negative_utility_matches_reference_with_variable_lengths(tmp_path: Path) -> None:
    data = batch()
    data.mask[0, 1:] = False
    data.mask[1, 2:] = False
    path = tmp_path / "pooled.npz"
    report = evaluate_comparison(
        {"emer_like": FixedRanker(0), "paretojev": FixedRanker(1)}, [data],
        directions=[1, -1], objective_names=["click", "risk"], cold_popularity_threshold=0.3,
        config=ParetoEvaluationConfig(top_k=3, random_preferences=0, hypervolume_samples=100), audit_path=path,
    )
    with np.load(path) as audit:
        for index, method in enumerate(audit["method_names"]):
            reference, _ = objective_utility_metrics(
                audit["scores"][index, 0], audit["labels"], audit["mask"], [1, -1], top_k=3,
            )
            np.testing.assert_allclose(reference, report["methods"][str(method)]["points"][0])
            shared, coverage = comparison_objective_metrics(
                audit["scores"][index, 0], audit["labels"], audit["mask"], [1, -1], top_k=3,
            )
            np.testing.assert_allclose(shared, report["methods"][str(method)]["points"][0])
            assert coverage.tolist() == [1.0, 1.0]


def test_emer_win_does_not_hide_cmr_loss_or_repeat_inference() -> None:
    data = batch()
    data.labels[:, 2, :] = torch.tensor([0.5, 0.25])
    third_scores = torch.tensor([[[0.0], [1.0], [3.0]]]).repeat(4, 1, 1)
    data.candidate_features = torch.cat([data.candidate_features, third_scores], dim=-1)
    models = {"emer_like": FixedRanker(0), "cmr_like": FixedRanker(1), "paretojev": FixedRanker(2)}
    report = compare(models, [data], cold_popularity_threshold=1.0)
    emer = report["kuairand"]["paretojev"]
    cmr = report["kuairand"]["paretojev_vs_cmr"]
    assert emer["hypervolume_relative_gain"] > 0
    assert emer["cold_item_ndcg_relative_gain"] == 1.0
    assert emer["worst_objective_regret"] == 0.0
    assert cmr["hypervolume_relative_gain"] < 0
    assert cmr["cold_item_ndcg_relative_gain"] == -0.5
    assert cmr["worst_objective_regret"] == 0.5
    assert report["baseline_pair_comparisons"] == {"emer_like": emer, "cmr_like": cmr}
    assert report["protocol"]["research_comparators"] == ["emer_like", "cmr_like"]
    assert all(model.calls == 4 for model in models.values())
    json.dumps(report, allow_nan=False)


def test_custom_generic_baseline_does_not_relabel_fixed_research_namespaces() -> None:
    report = compare(baseline="cmr_like")
    assert report["comparisons"]["paretojev"]["baseline"] == "cmr_like"
    assert report["kuairand"]["paretojev"]["baseline"] == "emer_like"
    assert report["kuairand"]["paretojev_vs_cmr"]["baseline"] == "cmr_like"
