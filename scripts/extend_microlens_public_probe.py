"""Freeze all 200 original users and reuse the validated 20-user pilot verbatim.

Only population size changes: prompts, model revision, candidate membership,
per-user candidate permutation, rate limit, fusion rules and seeds stay fixed.
"""
from __future__ import annotations

import copy
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from paretojev.data.microlens import load_prepared_candidates, save_microlens_protocol
from paretojev.utils.io import read_json, read_jsonl, atomic_write_json
from paretojev.utils.provenance import sha256_file, runtime_provenance


def main() -> None:
    original = Path("reports/microlens-real-2026-10-08")
    report = Path("reports/microlens-real-200-2026-10-08")
    root = Path("data/processed/microlens-real-200")
    out = Path("artifacts/microlens-real-public-200")
    for directory in (report, root, out):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refuse overwriting an existing registration: {directory}")
    previous = read_json(original / "registration.json")
    for entry in previous["data_files"]:
        if sha256_file(Path("data/raw/microlens") / entry["file"]) != entry["sha256"]:
            raise ValueError("Raw data hash changed")
    source = Path("data/processed/microlens-real/candidates.csv")
    baseline = read_json(original / "baseline_200_users.json")
    if sha256_file(source) != baseline["provenance"]["inputs"][0]["sha256"]:
        raise ValueError("Original 200-user candidate file changed")
    frame = load_prepared_candidates(source)
    if frame.user.nunique() != 200 or len(frame) != 4000:
        raise ValueError("Expected original 200-user, 4000-candidate population")
    save_microlens_protocol(frame, root)
    rows = read_jsonl(root / "jev_input.jsonl")
    for row in rows:
        random.Random(37 + int(row["slate_id"])).shuffle(row["candidates"])
    with (root / "jev_input.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    frame = load_prepared_candidates(root / "candidates.csv")
    parts = []
    for row in rows:
        group = frame[frame.user == row["slate_id"]].set_index("item")
        group = group.loc[[int(x["candidate_id"]) for x in row["candidates"]]].reset_index()
        group["position"] = range(len(group))
        parts.append(group)
    frame = pd.concat(parts, ignore_index=True)
    for key in ("hist_items", "hist_titles"):
        frame[key] = frame[key].map(json.dumps)
    frame.to_csv(root / "candidates.csv", index=False)
    # Prove exact semantic inputs on all 20 reused users before copying scores.
    pilot_rows = read_jsonl("data/processed/microlens-real-pilot/jev_input.jsonl")
    by_id = {row["slate_id"]: row for row in rows}
    if any(by_id[row["slate_id"]] != row for row in pilot_rows):
        raise ValueError("Expanded inputs differ from the fixed pilot")
    old_scores = read_jsonl("artifacts/microlens-real-public/scores.jsonl")
    if {row["slate_id"] for row in old_scores} != {row["slate_id"] for row in pilot_rows}:
        raise ValueError("Pilot score coverage differs")
    out.mkdir(parents=True)
    old_cache = Path("artifacts/microlens-real-public/requests.sqlite3")
    with sqlite3.connect(old_cache) as src, sqlite3.connect(out / "requests.sqlite3") as dst:
        src.backup(dst)
    indices = {row["slate_id"]: index for index, row in enumerate(rows)}
    old_score_hash = sha256_file("artifacts/microlens-real-public/scores.jsonl")
    with (out / "scores.jsonl").open("w") as handle:
        for row in old_scores:
            if row["resolved_model_revision"] != previous["decision_service"]["expected_revision"]:
                raise ValueError("Pilot deployment revision differs")
            if row["candidate_ids"] != [str(c["candidate_id"]) for c in by_id[row["slate_id"]]["candidates"]]:
                raise ValueError("Pilot candidate order differs")
            row["reused_from_score_sha256"] = old_score_hash
            row["input_index"] = indices[row["slate_id"]]
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    registration = copy.deepcopy(previous)
    registration.pop("pilot")
    registration.update(schema_version="microlens-real-200-v1",
        registered_at=datetime.now(timezone.utc).isoformat(), evaluation_scope="full_200_user_protocol",
        source_pilot_registration_sha256=sha256_file(original / "registration.json"),
        provenance=runtime_provenance(input_paths=[source, original / "registration.json"]),
        cohort={"users": [int(x) for x in frame.user.unique()], "n_users": 200,
                "mode": "noul", "objective": "click", "planned_model_requests": 4000,
                "max_new_http_calls": 3600, "max_concurrency": 1,
                "minimum_request_interval_seconds": 1.0,
                "candidate_order": previous["pilot"]["candidate_order"],
                "objectives": previous["pilot"]["objectives"],
                "candidate_file_sha256": sha256_file(root / "candidates.csv"),
                "jev_input_sha256": sha256_file(root / "jev_input.jsonl")},
        reuse={"users": previous["pilot"]["users"], "decisions": 400,
               "score_sha256": old_score_hash, "original_cache_sha256": sha256_file(old_cache)},
        secondary_analysis={"population": "180 users not seen in the pilot",
            "predictions": "same full-cohort out-of-fold predictions; no retuning for the subgroup"},
        protocol_notes=[
            "Extension from the preselected pilot to the original fixed 200 users, without changing any model/prompt/fusion setting.",
            "All users use the same frozen per-user permutation; baseline tie metrics can differ from the positive-first upstream-reproduction table.",
            "The primary 200-user result includes the previously observed 20-user pilot; the fresh 180-user subgroup is reported separately.",
            "One request at a time, at least one second between starts; stop and retain partial data on service error.",
        ])
    report.mkdir(parents=True)
    atomic_write_json(report / "registration.json", registration)
    print(json.dumps({"users": 200, "decisions": 4000, "reused": 400, "remaining": 3600}, indent=2))


if __name__ == "__main__":
    main()
