"""Evaluate the complete, frozen real MicroLens public-demo pilot locally."""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/microlens-real-2026-10-08")
    parser.add_argument("--score-dir", default="artifacts/microlens-real-public")
    parser.add_argument("--candidates", default="data/processed/microlens-real-pilot/candidates.csv")
    args = parser.parse_args()
    output, raw, candidates = Path(args.report_dir), Path(args.score_dir), Path(args.candidates)
    registration = read_json(output / "registration.json")
    cohort = registration.get("cohort", registration.get("pilot"))
    n_users = cohort["n_users"]
    scope = registration.get("evaluation_scope", "20_user_pilot")
    scoring = read_json(raw / "scoring_report.json")
    if scoring["status"] != "completed" or scoring["completed_rows_total"] != n_users:
        raise ValueError("Evaluate only the complete registered cohort, not successful-row subsets")
    if sha256_file(candidates) != cohort["candidate_file_sha256"]:
        raise ValueError("Candidate input changed after registration")
    if scoring["evidence"]["registration_sha256"] != sha256_file(output / "registration.json"):
        raise ValueError("Scoring registration mismatch")
    scores = read_jsonl(raw / "scores.jsonl")
    if len(scores) != n_users or len({row["slate_id"] for row in scores}) != n_users:
        raise ValueError("Need all registered users scored exactly once")
    expected_revision = registration["decision_service"]["expected_revision"]
    for row in scores:
        if row["resolved_model_revision"] != expected_revision:
            raise ValueError("Mismatched real Jev revision")
        if row["objective_names"] != ["click"] or len(set(row["candidate_ids"])) != 20:
            raise ValueError("Incorrect objective or candidate dimensions")
    frame = load_prepared_candidates(candidates)
    if set(frame.user) != set(cohort["users"]):
        raise ValueError("Frozen user selection changed")
    if {int(row["slate_id"]) for row in scores} != set(cohort["users"]):
        raise ValueError("Scored users differ from the registered cohort")
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
        "schema_version": "microlens-real-evaluation-v2",
        "evidence": {"data_kind": "real", "real_data": True, "dataset": "microlens",
                     "scope": scope, "users": n_users, "candidates": len(frame),
                     "model": "autotrust/JEV-27B", "revision": expected_revision},
        "metrics": metrics, "microlens": metrics,
        "confidence_intervals": {column: bootstrap_metric_intervals(frame, column, **bootstrap) for column in columns},
        "pairwise_vs_reference": {column: pairwise_metric_differences(frame, column, "itemcf", **bootstrap)
                             for column in columns if column != "itemcf"},
        "complementarity": complementarity_report(frame, ["itemcf", "jev_click"], **options),
        "fusion": {"fixed_weights": {"itemcf": 0.5, "jev_click": 0.5}, "cv_folds": folds,
                   "normalization": "within-slate average ranks for ties"},
        "score_distribution": {"min": float(frame.jev_click.min()), "max": float(frame.jev_click.max()),
                               "mean": float(frame.jev_click.mean()), "std": float(frame.jev_click.std()),
                               "distinct_scores": int(frame.jev_click.nunique())},
        "limitations": [
            ("The full preselected 200 users include 20 previously observed pilot users; the additional 180 users are reported separately."
             if scope == "full_200_user_protocol" else "20-user feasibility pilot, not the 200-user target protocol."),
            "AUC/HR/NDCG measure ranking among sampled alternatives, not calibrated real CTR.",
            "Other-user item-CF histories span the retrospective dataset; no global time-causal claim.",
            "Titles only; this is not a reproduction of cover-image JEV-27B-VL results.",
            "Bootstrap is conditional on the recorded out-of-fold scores; it does not refit fusion weights in each resample.",
            f"{int(frame.title.fillna('').str.strip().eq('').sum())} candidate titles are missing in the source; no users or candidates were dropped after observing scores.",
            "No multi-objective ParetoJev training or serving performance claim is made.",
        ],
        "provenance": runtime_provenance(input_paths=[candidates, raw / "scores.jsonl", output / "registration.json"]),
    }
    if scope == "full_200_user_protocol":
        if n_users != 200 or len(frame) != 4000:
            raise ValueError("Full protocol must contain 200 users and 4000 candidates")
        fresh = frame[~frame.user.isin(registration["reuse"]["users"])]
        if fresh.user.nunique() != 180:
            raise ValueError("Expected exactly 180 users not seen in the pilot")
        report["fresh_180_users"] = {
            "users": 180,
            "note": "Same full-cohort out-of-fold predictions; subgroup metrics are diagnostic and not used to refit weights.",
            "metrics": {column: evaluate_score_column(fresh, column, **options) for column in columns},
            "confidence_intervals": {column: bootstrap_metric_intervals(fresh, column, **bootstrap) for column in columns},
            "pairwise_vs_reference": {column: pairwise_metric_differences(fresh, column, "itemcf", **bootstrap)
                                       for column in columns if column != "itemcf"},
        }
    targets = load_yaml("configs/targets_microlens.yaml")
    # Phase A was defined on 200 users; a favorable pilot must not satisfy it.
    targets.setdefault("evidence_requirements", {})["evidence.scope"] = "full_200_user_protocol"
    report["target_configuration"] = targets
    report["target_report"] = target_report(evaluate_targets(report, targets))
    json.dumps(report, allow_nan=False)
    filename = "evaluation.json" if scope == "full_200_user_protocol" else "pilot_evaluation.json"
    atomic_write_json(output / filename, report)
    serializable = frame.copy()
    for key in ("hist_items", "hist_titles"):
        if key in serializable:
            serializable[key] = serializable[key].map(json.dumps)
    serializable.to_csv(raw / "evaluated_candidates.csv", index=False)
    print(json.dumps({"metrics": metrics, "target_status": report["target_report"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
