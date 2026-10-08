"""Score a pre-registered small real-data pilot on AutoTrust's public demo.

Run from the repository root with the project Python environment. Raw requests,
responses and cache remain local; no data files are redistributed with the code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paretojev.jev.cache import DecisionCache
from paretojev.jev.prompts import select_objectives
from paretojev.jev.public_demo import PublicDemoJevClient, score_public_demo_noul, validate_public_response
from paretojev.jev.runner import ScoreJsonlConfig, run_score_jsonl
from paretojev.utils.io import atomic_write_json, read_json, read_jsonl
from paretojev.utils.provenance import canonical_hash, runtime_provenance, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", default="reports/microlens-real-2026-10-08/registration.json")
    parser.add_argument("--input", default="data/processed/microlens-real-pilot/jev_input.jsonl")
    parser.add_argument("--output-dir", default="artifacts/microlens-real-public")
    args = parser.parse_args()
    registration = read_json(args.registration)
    if registration.get("data_kind") != "real" or registration.get("dataset") != "microlens":
        raise ValueError("This runner requires the registered real MicroLens probe")
    pilot = registration.get("cohort", registration.get("pilot"))
    scope = registration.get("evaluation_scope", "20_user_pilot")
    if sha256_file(args.input) != pilot["jev_input_sha256"]:
        raise ValueError("Frozen pilot input hash mismatch")
    rows = read_jsonl(args.input)
    expected = {str(row["slate_id"]): [str(item["candidate_id"]) for item in row["candidates"]] for row in rows}
    if len(expected) != len(rows) or set(expected) != set(map(str, pilot["users"])):
        raise ValueError("Pilot user selection differs from registration")
    decisions = sum(len(row["candidates"]) for row in rows)
    full_scopes = {"full_200_user_protocol", "independent_200_user_protocol"}
    cap = 4000 if scope in full_scopes else 400
    if scope in full_scopes and len(rows) != 200:
        raise ValueError("Full registered protocol requires exactly 200 users")
    if decisions != pilot["planned_model_requests"] or decisions > cap:
        raise ValueError(f"This registered run is limited to {cap} pointwise decisions")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    service = registration["decision_service"]
    cache = DecisionCache(out / "requests.sqlite3")
    client = PublicDemoJevClient(revision=service["expected_revision"], cache=cache,
                               max_http_calls=pilot.get("max_new_http_calls", decisions), minimum_interval=1.0)
    if service["endpoint"] != client.endpoint:
        raise ValueError("Registered endpoint differs from public client")
    # Reuse the one recorded connectivity canary, after the same strict checks.
    canary_request, canary_response = out / "first-request.json", out / "first-response.json"
    if canary_request.exists() and canary_response.exists():
        payload, observed = read_json(canary_request), read_json(canary_response)
        if canonical_hash(payload) != observed["request_sha256"]:
            raise ValueError("Canary request hash mismatch")
        validate_public_response(observed["response"], client.revision)
        cached = cache.get(client.provider, client.model, payload)
        if cached is None:
            cache.put(client.provider, client.model, payload, observed["response"])
        elif cached != observed["response"]:
            raise ValueError("Existing cached canary differs from the recorded response")
    registration_hash = sha256_file(args.registration)
    run_path = out / "run_registration.json"
    identity = {"registration_sha256": registration_hash, "input_sha256": sha256_file(args.input),
                "model": client.model, "endpoint": client.endpoint}
    if run_path.exists():
        if read_json(run_path)["identity"] != identity:
            raise ValueError("Existing scoring run uses a different registration")
    else:
        atomic_write_json(run_path, {"identity": identity, "provenance": runtime_provenance(input_paths=[args.input, args.registration])})
    config = ScoreJsonlConfig(provider=client.provider, mode="noul", objective_names=("click",),
                             noul_scope="pointwise", durable_every=1)
    result = run_score_jsonl(args.input, out / "scores.jsonl", client=client,
        score_function=score_public_demo_noul, objectives=select_objectives(["click"]), config=config,
        report_path=out / "scoring_report.json", error_path=out / "errors.jsonl", cache_count=cache.count)
    scores = read_jsonl(out / "scores.jsonl")
    if len(scores) != len(expected):
        raise ValueError("Real pilot has incomplete slate scores")
    for row in scores:
        if row["candidate_ids"] != expected[str(row["slate_id"])]:
            raise ValueError("Candidate identity or order mismatch in scored slate")
        if row["resolved_model_revision"] != client.revision:
            raise ValueError("Scored slate revision mismatch")
    result["evidence"] = {"real_data": True, "dataset": "microlens", "data_kind": "real",
                          "scope": scope, "users": len(rows), "endpoint": client.endpoint,
                          "resolved_model": service["model"], "resolved_revision": client.revision,
                          "registration_sha256": registration_hash,
                          "raw_response_cache_sha256": sha256_file(cache.path)}
    atomic_write_json(out / "scoring_report.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
