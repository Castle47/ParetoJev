"""Evaluate the complete, frozen real MicroLens public-demo pilot locally."""
from __future__ import annotations

import json
from pathlib import Path

from paretojev.config import load_yaml
from paretojev.data.microlens import (
    _within_slate_rank_normalize, cross_validated_hybrid,
    evaluate_score_column, load_prepared_candidates,
)
from paretojev.experiments.merge import merge_decision_scores
from paretojev.experiments.microlens import (
    bootstrap_metric_intervals, complementarity_report, pairwise_metric_differences,
)
from paretojev.training.targets import evaluate_targets, target_report
from paretojev.utils.io import atomic_write_json, read_json, read_jsonl
from paretojev.utils.provenance import runtime_provenance, sha256_file


def main() -> None:
    output = Path("reports/microlens-real-2026-10-08")
    raw = Path("artifacts/microlens-real-public")
    candidates = Path("data/processed/microlens-real-pilot/candidates.csv")
    registration = read_json(output / "registration.json")
    scoring = read_json(raw / "scoring_report.json")
    if scoring["status"] != "completed" or scoring["completed_rows_total"] != 20:
        raise ValueError("Evaluate only a complete 20-user pilot, not successful-row subsets")
    if sha256_file(candidates) != registration["pilot"]["candidate_file_sha256"]:
        raise ValueError("Candidate input changed after registration")
    if scoring["evidence"]["registration_sha256"] != sha256_file(output / "registration.json"):
        raise ValueError("Scoring registration mismatch")
    scores = read_jsonl(raw / "scores.jsonl")
    if len(scores) != 20 or len({row["slate_id"] for row in scores}) != 20:
        raise ValueError("Need exactly 20 unique scored users")
    expected_revision = registration["decision_service"]["expected_revision"]
    for row in scores:
        if row["resolved_model_revision"] != expected_revision:
            raise ValueError("Mismatched real Jev revision")
        if row["objective_names"] != ["click"] or len(set(row["candidate_ids"])) != 20:
            raise ValueError("Incorrect objective or candidate dimensions")
    frame = load_prepared_candidates(candidates)
    if set(frame.user) != set(registration["pilot"]["users"]):
        raise ValueError("Frozen user selection changed")
    frame = merge_decision_scores(frame, scores, objective="click", score_column="jev_click")
    frame["hybrid_fixed_05"] = 0.0
    for _, group in frame.groupby("user", sort=False):
        b = _within_slate_rank_normalize(group["itemcf"].to_numpy()[None, :])[0]
        j = _within_slate_rank_normalize(group["jev_click"].to_numpy()[None, :])[0]
        frame.loc[group.index, "hybrid_fixed_05"] = 0.5 * b + 0.5 * j
    options = {"tie_policy": "upstream_jitter", "tie_seed": 1}
    frame, folds = cross_validated_hybrid(frame, ["itemcf", "jev_click"], folds=5,
        weight_step=0.05, objective="balanced", seed=0, output_column="hybrid_cv", **options)
    columns = ["pop", "title_sim", "itemcf", "jev_click", "hybrid_fixed_05", "hybrid_cv"]
    metrics = {column: evaluate_score_column(frame, column, **options) for column in columns}
    bootstrap = {"n_bootstrap": 5000, "confidence": 0.95, "seed": 0, **options}
    report = {
        "schema_version": "microlens-real-pilot-evaluation-v1",
        "evidence": {"data_kind": "real", "real_data": True, "dataset": "microlens",
                     "scope": "20_user_pilot", "users": 20, "candidates": 400,
                     "model": "autotrust/JEV-27B", "revision": expected_revision},
        "metrics": metrics, "microlens": metrics,
        "confidence_intervals": {column: bootstrap_metric_intervals(frame, column, **bootstrap) for column in columns},
        "paired_vs_itemcf": {column: pairwise_metric_differences(frame, column, "itemcf", **bootstrap)
                             for column in columns if column != "itemcf"},
        "complementarity": complementarity_report(frame, ["itemcf", "jev_click"], **options),
        "fusion": {"fixed_weights": {"itemcf": 0.5, "jev_click": 0.5}, "cv_folds": folds,
                   "normalization": "within-slate average ranks for ties"},
        "score_distribution": {"min": float(frame.jev_click.min()), "max": float(frame.jev_click.max()),
                               "mean": float(frame.jev_click.mean()), "std": float(frame.jev_click.std()),
                               "distinct_scores": int(frame.jev_click.nunique())},
        "limitations": [
            "20 users chosen before Jev scoring; a feasibility pilot, not the 200-user target protocol.",
            "AUC/HR/NDCG measure ranking among sampled alternatives, not calibrated real CTR.",
            "Other-user item-CF histories span the retrospective dataset; no global time-causal claim.",
            "Titles only; this is not a reproduction of cover-image JEV-27B-VL results.",
            "Bootstrap is conditional on the recorded out-of-fold scores; it does not refit fusion weights in each resample.",
            "One candidate title is missing in the source; no users or candidates were dropped after observing scores.",
            "No multi-objective ParetoJev training or serving performance claim is made.",
        ],
        "provenance": runtime_provenance(input_paths=[candidates, raw / "scores.jsonl", output / "registration.json"]),
    }
    targets = load_yaml("configs/targets_microlens.yaml")
    # Phase A was defined on 200 users; a favorable pilot must not satisfy it.
    targets.setdefault("evidence_requirements", {})["evidence.scope"] = "full_200_user_protocol"
    report["target_configuration"] = targets
    report["target_report"] = target_report(evaluate_targets(report, targets))
    json.dumps(report, allow_nan=False)
    atomic_write_json(output / "pilot_evaluation.json", report)
    frame.to_csv(raw / "evaluated_candidates.csv", index=False)
    print(json.dumps({"metrics": metrics, "target_status": report["target_report"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
