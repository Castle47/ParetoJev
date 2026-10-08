"""Evaluate the frozen new-user experiment; no parameter fitting or selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paretojev.config import load_yaml
from paretojev.data.microlens import load_prepared_candidates
from paretojev.experiments.merge import merge_decision_scores
from paretojev.experiments.tie_validation import METHODS, policy_keys, expected_metrics, tied_candidate_mask
from paretojev.metrics.ranking import paired_bootstrap_difference
from paretojev.training.targets import evaluate_targets, target_report
from paretojev.utils.io import read_json, read_jsonl, atomic_write_json
from paretojev.utils.provenance import sha256_file, runtime_provenance


def main() -> None:
    report_dir = Path("reports/microlens-independent-2026-10-08")
    source = Path("data/processed/microlens-independent-200")
    raw = Path("artifacts/microlens-real-independent-200")
    registration = read_json(report_dir / "registration.json")
    scoring = read_json(raw / "scoring_report.json")
    initial_path = report_dir / "scoring_initial_report.json"
    collection = read_json(initial_path) if initial_path.exists() else scoring
    audit = read_json(report_dir / "score_audit.json")
    if scoring["status"] != "completed" or scoring["completed_rows_total"] != 200:
        raise ValueError("Wait for the complete registered cohort")
    if sha256_file(source / "candidates.csv") != registration["cohort"]["candidate_file_sha256"]:
        raise ValueError("Frozen candidates changed")
    if scoring["evidence"]["registration_sha256"] != sha256_file(report_dir / "registration.json"):
        raise ValueError("Scoring registration changed")
    if collection["evidence"]["registration_sha256"] != sha256_file(report_dir / "registration.json"):
        raise ValueError("Original collection cost belongs to a different registration")
    if audit["score_sha256"] != sha256_file(raw / "scores.jsonl") or not audit["all_scores_equal_raw_responses"]:
        raise ValueError("Exact raw-response audit is required")
    frame = load_prepared_candidates(source / "candidates.csv")
    users = set(int(x) for x in frame.user.unique())
    excluded = set(registration["excluded_previous_evaluation_users"])
    if users != set(registration["cohort"]["users"]) or len(users) != 200 or users & excluded:
        raise ValueError("Independent user membership changed")
    scores = read_jsonl(raw / "scores.jsonl")
    if len(scores) != 200 or {int(row["slate_id"]) for row in scores} != users:
        raise ValueError("Score coverage mismatch")
    frame = merge_decision_scores(frame, scores, objective="click", score_column="jev_click")
    request_plan = pd.read_csv(source / "tie_request_plan.csv")
    if sha256_file(source / "tie_request_plan.csv") != registration["cost_plan"]["request_mask_sha256"]:
        raise ValueError("Pre-outcome call selection mask changed")
    plan = request_plan.set_index(["user", "item"])["tie_only_requires_jev"]
    measurements = {method: [] for method in METHODS}
    per_user = []
    actual_tie_calls = 0
    for user, group in frame.groupby("user", sort=False):
        if len(group) != 20 or group.y.sum() != 1:
            raise ValueError("Every user must have 20 candidates and one next item")
        b, t, j = (group[column].to_numpy() for column in ("itemcf", "title_sim", "jev_click"))
        tie_mask = tied_candidate_mask(b)
        registered_mask = np.array([bool(plan.loc[(int(user), int(item))]) for item in group.item])
        if not np.array_equal(tie_mask, registered_mask):
            raise ValueError("Call selection changed after scoring")
        actual_tie_calls += int(tie_mask.sum())
        keys = policy_keys(b, t, j)
        row = {"user": int(user)}
        for method in METHODS:
            metric = expected_metrics(keys[method], group.y.to_numpy())
            measurements[method].append(metric)
            row[method] = metric
        per_user.append(row)
    metric_names = list(measurements["itemcf"][0])
    arrays = {method: {metric: np.array([row[metric] for row in rows]) for metric in metric_names}
              for method, rows in measurements.items()}
    summaries = {method: {metric: float(values.mean()) for metric, values in rows.items()} for method, rows in arrays.items()}
    config = registration["evaluation"]

    def compare(left, right, confidence):
        result = {}
        for metric in metric_names:
            difference, low, high = paired_bootstrap_difference(arrays[left][metric], arrays[right][metric],
                n_bootstrap=config["bootstrap_samples"], confidence=confidence, seed=config["bootstrap_seed"])
            result[metric] = {"difference": difference, "low": low, "high": high,
                              "confidence": confidence, "interval_above_zero": low > 0}
        return result

    primary = {f"{left}_vs_{right}": compare(left, right, config["primary_ci_confidence"])
               for left, right in config["primary_comparisons"]}
    secondary_pairs = [(method, "itemcf") for method in METHODS if method != "itemcf"]
    secondary_pairs += [("jev_05", "title_05"), ("jev_05", "jev_ties")]
    secondary = {f"{left}_vs_{right}": compare(left, right, config["secondary_ci_confidence"])
                 for left, right in secondary_pairs}
    result = {
        "schema_version": "microlens-independent-evaluation-v1",
        "evidence": {"real_data": True, "dataset": "microlens", "scope": "independent_200_user_protocol",
                     "users": 200, "overlap_previous_users": 0, "weights_tuned_on_new_users": False,
                     "model": registration["decision_service"]["model"],
                     "revision": registration["decision_service"]["expected_revision"]},
        "metrics": summaries, "methods": {method: {"definition": registration["policies"][method]} for method in METHODS},
        "metric_definition": "AUC half-credit ties; HR/NDCG/MRR integrate uniformly over remaining tied rank positions",
        "primary_metric": "auc", "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "cost": {"actual_http_calls_to_collect_all_methods": collection["client_stats"]["http_calls"],
                 "observed_decisions": 4000,
                 "policy_candidate_requests": {method: (actual_tie_calls if method == "jev_ties" else 4000 if method in ("jev_only", "jev_05") else 0) for method in METHODS},
                 "tie_only_saved_fraction": 1 - actual_tie_calls / 4000,
                 "actual_scoring_wall_seconds": collection["elapsed_seconds"],
                 "note": "Policy request counts are the frozen label-free plan. Actual measurement collects all scores once; no separate tie-only latency, price or savings was measured."},
        "limitations": registration["limitations"] + [
            "Only the two predeclared AUC comparisons are confirmatory at 97.5% two-sided intervals; other metrics and method comparisons are secondary.",
            "Fixed 5% weights were chosen from the earlier development experiment, then frozen; no CV or fitting uses these users.",
            "A positive AUC result alone does not prove every ranking metric improves or justify serving cost.",
            "Item-CF's exact ties do not automatically generalize to a neural behavior model with continuous probabilities.",
        ],
        "provenance": runtime_provenance(input_paths=[report_dir / "registration.json", source / "candidates.csv", raw / "scores.jsonl"]),
    }
    targets = load_yaml("configs/targets_independent_ties.yaml")
    result["target_configuration"] = targets
    result["target_report"] = target_report(evaluate_targets(result, targets))
    json.dumps(result, allow_nan=False)
    atomic_write_json(report_dir / "evaluation.json", result)
    atomic_write_json(raw / "per_user_metrics.json", per_user)
    print(json.dumps({"metrics": summaries, "primary_auc": {k:v['auc'] for k,v in primary.items()},
                      "target_status": result['target_report']['status'], "cost": result['cost']}, indent=2))


if __name__ == "__main__":
    main()
