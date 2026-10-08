"""Pre-register a user-disjoint, no-tuning validation of Jev tie-breaking."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from paretojev.data.microlens import MicroLensProtocol, prepare_microlens_candidates, save_microlens_protocol
from paretojev.experiments.tie_validation import METHODS, tied_candidate_mask
from paretojev.jev.prompts import build_state, build_open_jev_noul_requests, select_objectives
from paretojev.utils.io import read_json, read_jsonl, atomic_write_json
from paretojev.utils.provenance import sha256_file, runtime_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/processed/microlens-independent-200")
    parser.add_argument("--replay-data", action="store_true", help="Rebuild into an empty output directory and verify the existing registration")
    args = parser.parse_args()
    previous_path = Path("reports/microlens-real-200-2026-10-08/registration.json")
    previous = read_json(previous_path)
    excluded = sorted(previous["cohort"]["users"])
    root = Path(args.output_dir)
    report = Path("reports/microlens-independent-2026-10-08")
    frozen = read_json(report / "registration.json") if args.replay_data else None
    if frozen and sha256_file(previous_path) != frozen["development_registration_sha256"]:
        raise ValueError("Development registration changed")
    for path in (root,) if args.replay_data else (root, report):
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Refuse to overwrite a registered cohort: {path}")
    for entry in previous["data_files"]:
        if sha256_file(Path("data/raw/microlens") / entry["file"]) != entry["sha256"]:
            raise ValueError("Raw source file changed")
    protocol = MicroLensProtocol(n_users=200, n_negatives=19, history_length=5,
        time_window_days=3, seed=20261009, negative_sampling="upstream_occurrence")
    if frozen and asdict(protocol) != frozen["protocol"]:
        raise ValueError("Preparation protocol changed")
    frame = prepare_microlens_candidates("data/raw/microlens", protocol,
        excluded_users=excluded, require_strict_history=True)
    users = [int(x) for x in frame.user.unique()]
    assert len(users) == 200 and not set(users).intersection(excluded)
    parts = []
    call_rows = []
    for user, group in frame.groupby("user", sort=False):
        order = list(range(len(group))); random.Random(37 + int(user)).shuffle(order)
        group = group.iloc[order].copy(); group["position"] = range(len(group))
        needs_jev_for_ties = tied_candidate_mask(group.itemcf.to_numpy())
        for row, selected in zip(group.itertuples(), needs_jev_for_ties, strict=True):
            call_rows.append({"user": int(user), "item": int(row.item), "tie_only_requires_jev": bool(selected)})
        parts.append(group)
    frame = pd.concat(parts, ignore_index=True)
    save_microlens_protocol(frame, root)
    # This request mask is derived solely from CF scores, never from y/labels.
    pd.DataFrame(call_rows).to_csv(root / "tie_request_plan.csv", index=False)
    max_state = max_question = 0
    for row in read_jsonl(root / "jev_input.jsonl"):
        state = build_state(row["history"], context=row["context"], include_candidates=False)
        for _, state, question, _ in build_open_jev_noul_requests(state, row["candidates"], select_objectives(["click"])):
            max_state = max(max_state, len(json.dumps(state, ensure_ascii=False, separators=(",", ":"))))
            max_question = max(max_question, len(question))
    if max_state > 6000 or max_question > 1000:
        raise ValueError("Frozen request exceeds public endpoint limits; no truncation performed")
    tie_calls = sum(int(row["tie_only_requires_jev"]) for row in call_rows)
    if frozen:
        expected = {"candidates.csv": frozen["cohort"]["candidate_file_sha256"],
                    "jev_input.jsonl": frozen["cohort"]["jev_input_sha256"],
                    "tie_request_plan.csv": frozen["cost_plan"]["request_mask_sha256"]}
        if users != frozen["cohort"]["users"] or any(sha256_file(root / name) != digest for name, digest in expected.items()):
            raise ValueError("Recreated independent input hashes differ")
        print(json.dumps({"status": "verified", "users": 200, "decisions": 4000, "all_three_input_hashes_identical": True}, indent=2))
        return
    registration = {
        "schema_version": "microlens-independent-v1", "registered_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens", "data_kind": "real", "evaluation_scope": "independent_200_user_protocol",
        "data_files": previous["data_files"], "development_registration_sha256": sha256_file(previous_path),
        "protocol": asdict(protocol), "candidate_order_seed": 37,
        "excluded_previous_evaluation_users": excluded,
        "itemcf_training_excluded_users": sorted(set(users) | set(excluded)),
        "strict_history_boundary_required": True,
        "cohort": {"users": users, "n_users": 200, "planned_model_requests": 4000,
                   "max_new_http_calls": 4000, "max_concurrency": 1,
                   "minimum_request_interval_seconds": 1.0,
                   "objectives": [x.__dict__ for x in select_objectives(["click"])],
                   "candidate_file_sha256": sha256_file(root / "candidates.csv"),
                   "jev_input_sha256": sha256_file(root / "jev_input.jsonl")},
        "decision_service": previous["decision_service"],
        "methods": list(METHODS),
        "policies": {"itemcf": "raw cooccurrence score", "title_only": "max historical TF-IDF title similarity",
                     "jev_only": "frozen real Noul p(true)",
                     "title_ties": "lexicographic (CF, title), remaining ties averaged",
                     "jev_ties": "lexicographic (CF, Jev), remaining ties averaged",
                     "title_05": "95% CF average rank + 5% title average rank; exact integer key",
                     "jev_05": "95% CF average rank + 5% Jev average rank; exact integer key"},
        "evaluation": {"no_tuning_on_new_users": True, "auc": "half credit for exact key ties",
                       "ranking_metrics": "exact expectation over uniform permutations inside remaining tied key groups",
                       "primary_comparisons": [["jev_ties", "itemcf"], ["jev_ties", "title_ties"]],
                       "primary_metric": "auc", "primary_ci_confidence": 0.975,
                       "primary_ci_note": "two predeclared comparisons; conservative Bonferroni 97.5% two-sided intervals",
                       "secondary_ci_confidence": 0.95, "bootstrap_samples": 10000, "bootstrap_seed": 20261009,
                       "winner_rule": "both primary AUC interval lower bounds > 0; secondary metrics and cost also reported"},
        "cost_plan": {"actual_full_comparison_decisions": 4000,
                      "tie_only_policy_required_decisions": tie_calls,
                      "tie_only_policy_saved_fraction": 1 - tie_calls / 4000,
                      "request_mask_sha256": sha256_file(root / "tie_request_plan.csv"),
                      "note": "All 4000 decisions are collected for the matched 5% fusion comparison. Tie-only count is a label-free deployable call plan, not measured separate-run latency or money."},
        "preflight": {"overlap_previous_users": 0, "training_excluded_user_count": 400,
                      "max_state_characters": max_state, "max_question_characters": max_question},
        "limitations": ["User-disjoint evaluation on the same retrospective public dataset, not a new platform or global time holdout.",
            "These users were not evaluated previously, but may have contributed to earlier aggregate CF training; this run refits CF excluding all 400 evaluation users.",
            "Sampled alternatives are not observed nonclick exposures. Only next-item ranking is evaluated.",
            "Metrics integrate ties and therefore are not directly interchangeable with earlier jitter-based HR/NDCG.",
            "No new user labels tune weights, prompts, eligibility rules or method selection."],
        "provenance": runtime_provenance(input_paths=[previous_path]),
    }
    report.mkdir(parents=True)
    atomic_write_json(report / "registration.json", registration)
    print(json.dumps({"users": 200, "overlap": 0, "decisions": 4000, "tie_only_required_decisions": tie_calls}, indent=2))


if __name__ == "__main__":
    main()
