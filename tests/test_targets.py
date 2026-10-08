import json
from pathlib import Path

import pytest

from paretojev.config import load_yaml
from paretojev.training.targets import (
    TargetResult,
    check_target_files,
    evaluate_targets,
    target_report,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _gate(**overrides):
    return {"targets": {"score": {"metric": "score", "op": ">=", "threshold": 0.5, **overrides}}}


def test_target_checker() -> None:
    metrics = {"training": {"loss_reduction_fraction": 0.1}, "fusion": {"brier": 0.02}}
    config = {
        "targets": {
            "loss": {
                "metric": "training.loss_reduction_fraction",
                "op": ">=",
                "threshold": 0.05,
            },
            "brier": {
                "metric": "fusion.brier",
                "op": "<=",
                "threshold": 0.03,
            },
        }
    }
    report = target_report(evaluate_targets(metrics, config))
    assert report["passed"]
    assert report["status"] == "passed"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), "NaN", "undefined", [], True])
@pytest.mark.parametrize("op", [">=", "<=", "=="])
def test_undefined_or_nonfinite_metrics_are_not_evaluable(value, op) -> None:
    result = evaluate_targets({"score": value}, _gate(op=op))[0]
    assert not result.passed
    assert result.status == "not_evaluable"
    assert result.actual is None
    assert "undefined, nonnumeric, or nonfinite" in result.reason
    # Target report JSON must remain standards-compliant for nonfinite inputs.
    json.dumps(target_report([result]), allow_nan=False)


def test_missing_metric_is_distinct_from_finite_failure() -> None:
    missing = evaluate_targets({}, _gate())[0]
    failed = evaluate_targets({"score": 0.49}, _gate())[0]
    assert missing.status == "not_evaluable"
    assert "missing metric score" in missing.reason
    assert failed.status == "failed"
    assert failed.actual == 0.49
    assert target_report([missing])["status"] == "not_evaluable"
    assert target_report([failed])["status"] == "failed"


@pytest.mark.parametrize("threshold", [None, float("nan"), float("inf"), float("-inf"), "NaN", "wrong", {}, True])
def test_invalid_threshold_rejected_even_with_missing_metric(threshold) -> None:
    with pytest.raises(ValueError, match="finite numeric threshold"):
        evaluate_targets({}, _gate(threshold=threshold))


@pytest.mark.parametrize("op, value, expected", [(">=", 0.5, True), (">", 0.5, False), ("<=", 0.5, True), ("<", 0.5, False), ("==", 0.5, True)])
def test_numeric_boundary_comparisons(op, value, expected) -> None:
    assert evaluate_targets({"score": value}, _gate(op=op))[0].passed is expected


def test_empty_or_optional_only_gates_never_pass_vacuously() -> None:
    for config in ({}, {"targets": {}}, _gate(required=False)):
        report = target_report(evaluate_targets({"score": 1}, config))
        assert not report["passed"]
        assert report["status"] == "not_evaluable"
        assert report["required_total"] == 0


def test_optional_undefined_does_not_block_required_success() -> None:
    config = _gate()
    config["targets"]["optional"] = {"threshold": 1, "required": False}
    report = target_report(evaluate_targets({"score": 1}, config))
    assert report["passed"]
    assert report["targets"][1]["status"] == "not_evaluable"


def _comparison_metrics():
    return {
        "evidence": {
            "real_data": True,
            "data_kind": "real",
            "dataset": "kuairand",
            "protocol_version": "baseline-harness-v1",
            "comparison_baseline": "emer_like",
        },
        "methods": {name: {"hypervolume": 0.5} for name in ("emer_like", "cmr_like", "paretojev")},
        "kuairand": {"paretojev_vs_cmr": {
            "hypervolume_relative_gain": 0.02,
            "cold_item_ndcg_relative_gain": 0.03,
            "worst_objective_regret": 0.005,
        }, "paretojev": {
            "hypervolume_relative_gain": 0.02,
            "cold_item_ndcg_relative_gain": 0.03,
            "worst_objective_regret": 0.005,
            "controllability_spearman_mean": 0.6,
        }},
    }


def test_complete_real_comparison_evidence_allows_evaluating_gates() -> None:
    report = target_report(evaluate_targets(_comparison_metrics(), load_yaml(CONFIGS / "targets_comparison.yaml")))
    assert report["passed"]
    assert report["required_total"] == 7


@pytest.mark.parametrize("field, value", [
    ("real_data", False), ("real_data", 1), ("real_data", "true"),
    ("data_kind", "synthetic"), ("dataset", "microlens"),
    ("protocol_version", "pareto-evaluation-v1"), ("comparison_baseline", "fused"),
])
def test_synthetic_wrong_dataset_or_static_evidence_cannot_pass_comparison(field, value) -> None:
    metrics = _comparison_metrics()
    metrics["evidence"][field] = value
    report = target_report(evaluate_targets(metrics, load_yaml(CONFIGS / "targets_comparison.yaml")))
    assert not report["passed"]
    assert report["status"] == "not_evaluable"
    assert all(f"evidence.{field}" in target["reason"] for target in report["targets"])


@pytest.mark.parametrize("method_report", [None, {}, [], "trained"])
def test_all_registered_methods_require_evaluated_report_mappings(method_report) -> None:
    metrics = _comparison_metrics()
    metrics["methods"]["cmr_like"] = method_report
    report = target_report(evaluate_targets(metrics, load_yaml(CONFIGS / "targets_comparison.yaml")))
    assert report["status"] == "not_evaluable"
    assert "missing evaluated method cmr_like" in report["targets"][0]["reason"]


def test_research_gates_require_baseline_evidence_not_just_matching_numbers() -> None:
    config = load_yaml(CONFIGS / "targets_research.yaml")
    metrics = _comparison_metrics()
    metrics["microlens"] = {"hybrid_cv": {"auc": 0.8, "hr@5": 0.7, "ndcg@10": 0.6}}
    metrics["serving"] = {"student": {"uplift_retention": 1.0, "p95_ms_k50": 10.0}}
    assert target_report(evaluate_targets(metrics, config))["passed"]
    del metrics["evidence"]
    report = target_report(evaluate_targets(metrics, config))
    assert report["status"] == "not_evaluable"
    assert not report["passed"]
    assert "missing required evidence evidence.real_data" in report["targets"][0]["reason"]


def test_microlens_requires_explicit_real_dataset_origin() -> None:
    config = load_yaml(CONFIGS / "targets_microlens.yaml")
    metrics = {
        "microlens": {"hybrid_cv": {"auc": 0.8, "hr@5": 0.7, "ndcg@10": 0.6}},
        "pairwise_vs_reference": {"hybrid_cv": {"auc": {"low": 0.01}}},
    }
    assert target_report(evaluate_targets(metrics, config))["status"] == "not_evaluable"
    metrics["evidence"] = {"real_data": True, "data_kind": "real", "dataset": "microlens"}
    assert target_report(evaluate_targets(metrics, config))["passed"]


def test_file_checker_persists_actionable_missing_evidence(tmp_path) -> None:
    metrics = _comparison_metrics()
    metrics["evidence"]["real_data"] = False
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    output_path = tmp_path / "targets.json"
    report = check_target_files(metrics_path, CONFIGS / "targets_comparison.yaml", output_path)
    assert json.loads(output_path.read_text()) == report
    assert report["status"] == "not_evaluable"


def test_legacy_direct_target_mapping_and_result_constructor_remain_supported() -> None:
    assert evaluate_targets({"score": "0.6"}, _gate()["targets"])[0].passed
    legacy = TargetResult("score", "score", ">=", 0.5, 0.6, True, True)
    assert legacy.status == "passed"
    assert target_report([legacy])["passed"]


@pytest.mark.parametrize("configuration, match", [
    ({"targets": [], "evidence_requirements": {}}, "targets.*mapping"),
    ({"targets": {}, "evidence_requirements": []}, "evidence_requirements"),
    ({"targets": {}, "evidence_requirements": {"real": None}}, "scalar"),
    ({"targets": {}, "evidence_requirements": {"real": float("inf")}}, "finite"),
    ({"targets": {}, "required_methods": "emer_like"}, "required_methods"),
    (_gate(required="false"), "required must be a boolean"),
    (_gate(metric=""), "nonempty metric path"),
    (_gate(op="!="), "Unsupported operator"),
])
def test_invalid_configuration_is_rejected(configuration, match) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_targets({}, configuration)


@pytest.mark.parametrize("metric,value", [
    ("hypervolume_relative_gain", 0.019),
    ("cold_item_ndcg_relative_gain", 0.029),
    ("worst_objective_regret", 0.006),
])
def test_paretojev_must_clear_cmr_gates_even_when_emer_gates_pass(metric, value) -> None:
    metrics = _comparison_metrics()
    metrics["kuairand"]["paretojev_vs_cmr"][metric] = value
    for name in ("targets_comparison.yaml", "targets_research.yaml"):
        config = load_yaml(CONFIGS / name)
        if name == "targets_research.yaml":
            metrics["microlens"] = {"hybrid_cv": {"auc": 0.8, "hr@5": 0.7, "ndcg@10": 0.6}}
            metrics["serving"] = {"student": {"uplift_retention": 1.0, "p95_ms_k50": 10.0}}
        report = target_report(evaluate_targets(metrics, config))
        assert not report["passed"]
        assert report["status"] == "failed"
        assert report["required_passed"] == report["required_total"] - 1
