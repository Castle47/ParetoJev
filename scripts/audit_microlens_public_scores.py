"""Verify every final score against its exact request and cached raw response.

This read-only audit is run only after the registered cohort is complete. It
does not call the model or use outcome labels to select/filter any predictions.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from paretojev.jev.cache import DecisionCache
from paretojev.jev.prompts import build_state, build_open_jev_noul_requests, select_objectives
from paretojev.jev.public_demo import PUBLIC_DEMO_MODEL, validate_public_response
from paretojev.jev.scoring import _binary_entropy
from paretojev.utils.io import read_json, read_jsonl, atomic_write_json
from paretojev.utils.provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/microlens-real-200-2026-10-08")
    parser.add_argument("--score-dir", default="artifacts/microlens-real-public-200")
    parser.add_argument("--input", default="data/processed/microlens-real-200/jev_input.jsonl")
    args = parser.parse_args()
    report, output = Path(args.report_dir), Path(args.score_dir)
    registration = read_json(report / "registration.json")
    cohort = registration.get("cohort", registration.get("pilot"))
    if sha256_file(args.input) != cohort["jev_input_sha256"]:
        raise ValueError("Frozen input hash mismatch")
    inputs, scores = read_jsonl(args.input), read_jsonl(output / "scores.jsonl")
    expected_users = set(cohort["users"])
    if len(scores) != cohort["n_users"] or {row["slate_id"] for row in scores} != expected_users:
        raise ValueError("Audit requires exact complete cohort coverage")
    by_user = {row["slate_id"]: row for row in scores}
    if len(by_user) != len(scores):
        raise ValueError("Duplicate scored users")
    revision = registration["decision_service"]["expected_revision"]
    model = f"{PUBLIC_DEMO_MODEL}@{revision}"
    cache_path = output / "requests.sqlite3"
    with sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True) as conn:
        cached = {key: (json.loads(request), json.loads(response)) for key, request, response
                  in conn.execute("SELECT cache_key, request_json, response_json FROM decisions")}
    request_keys, inference_seconds = set(), []
    count = 0
    for index, row in enumerate(inputs):
        scored = by_user[row["slate_id"]]
        ids = [str(candidate["candidate_id"]) for candidate in row["candidates"]]
        if scored["candidate_ids"] != ids or scored["input_index"] != index:
            raise ValueError("Candidate order or input index differs")
        if scored["resolved_model"] != PUBLIC_DEMO_MODEL or scored["resolved_model_revision"] != revision:
            raise ValueError("Unexpected model identity")
        if scored["objective_names"] != ["click"]:
            raise ValueError("Unexpected objective")
        state = build_state(row["history"], context=row["context"], include_candidates=False)
        values = []
        for kind, state, question, _ in build_open_jev_noul_requests(state, row["candidates"], select_objectives(["click"])):
            payload = {"kind": kind, "state": json.dumps(state, ensure_ascii=False, separators=(",", ":")), "question": question}
            key = DecisionCache.key_for("autotrust-public-demo", model, payload)
            if key not in cached or cached[key][0] != payload:
                raise ValueError("Missing exact raw request/response")
            response = cached[key][1]
            validate_public_response(response, revision)
            values.append([response["probabilities"][1]])
            if key not in request_keys:
                inference_seconds.append(response["elapsed_seconds"])
            request_keys.add(key)
        observed = np.asarray(scored["probabilities"])
        if not np.array_equal(observed, np.asarray(values)):
            raise ValueError("Final score does not equal the cached real model response")
        entropy = _binary_entropy(observed)
        if not np.allclose(entropy, scored["entropy"], atol=1e-12, rtol=0):
            raise ValueError("Entropy differs from the model probabilities")
        if not np.allclose(1 - entropy, scored["confidence"], atol=1e-12, rtol=0):
            raise ValueError("Confidence proxy differs")
        count += len(ids)
    if count != cohort["planned_model_requests"]:
        raise ValueError("Decision coverage mismatch")
    audit = {"users": len(scores), "decisions": count, "unique_request_response_pairs": len(request_keys),
             "cached_responses": len(cached), "all_scores_equal_raw_responses": True,
             "all_candidate_ids_and_order_verified": True, "revision": revision,
             "model": PUBLIC_DEMO_MODEL, "score_sha256": sha256_file(output / "scores.jsonl"),
             "cache_sha256": sha256_file(cache_path), "input_sha256": sha256_file(args.input),
             "backend_reported_inference_seconds_sum": sum(inference_seconds), "new_api_calls": 0}
    atomic_write_json(report / "score_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
