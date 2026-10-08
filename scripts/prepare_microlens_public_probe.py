"""Rebuild the registered real pilot from the two official raw CSV files.

The recorded raw/input hashes are checked; existing output files are not replaced.
The registration is immutable and is never rewritten by this replay utility.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from paretojev.data.microlens import (
    MicroLensProtocol, load_prepared_candidates, prepare_microlens_candidates,
    save_microlens_protocol,
)
from paretojev.utils.io import read_json, read_jsonl
from paretojev.utils.provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", default="reports/microlens-real-2026-10-08/registration.json")
    parser.add_argument("--raw-dir", default="data/raw/microlens")
    parser.add_argument("--full-dir", default="data/processed/microlens-real")
    parser.add_argument("--pilot-dir", default="data/processed/microlens-real-pilot")
    args = parser.parse_args()
    registration = read_json(args.registration)
    for entry in registration["data_files"]:
        if sha256_file(Path(args.raw_dir) / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Official source hash differs: {entry['file']}")
    for directory in (Path(args.full_dir), Path(args.pilot_dir)):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Use an empty output directory: {directory}")
    cfg = registration["protocol"]
    protocol = MicroLensProtocol(n_users=cfg["users"], n_negatives=cfg["negatives"],
        history_length=cfg["history"], time_window_days=cfg["window_days"],
        seed=cfg["seed"], negative_sampling=cfg["negative_sampling"])
    save_microlens_protocol(prepare_microlens_candidates(args.raw_dir, protocol), args.full_dir)
    frame = load_prepared_candidates(Path(args.full_dir) / "candidates.csv")
    pilot = registration["pilot"]
    users = random.Random(pilot["user_selection_seed"]).sample(sorted(frame.user.unique().tolist()), pilot["n_users"])
    if users != pilot["users"]:
        raise ValueError("Recreated pilot user selection differs")
    root = Path(args.pilot_dir)
    save_microlens_protocol(frame[frame.user.isin(users)].copy(), root)
    rows = read_jsonl(root / "jev_input.jsonl")
    for row in rows:
        random.Random(cfg["candidate_order_seed"] + int(row["slate_id"])).shuffle(row["candidates"])
    with (root / "jev_input.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    frame = load_prepared_candidates(root / "candidates.csv")
    parts = []
    for row in rows:
        group = frame[frame.user == row["slate_id"]].set_index("item")
        group = group.loc[[int(item["candidate_id"]) for item in row["candidates"]]].reset_index()
        group["position"] = range(len(group))
        parts.append(group)
    result = pd.concat(parts, ignore_index=True)
    for key in ("hist_items", "hist_titles"):
        result[key] = result[key].map(json.dumps)
    result.to_csv(root / "candidates.csv", index=False)
    for filename, key in (("candidates.csv", "candidate_file_sha256"), ("jev_input.jsonl", "jev_input_sha256")):
        if sha256_file(root / filename) != pilot[key]:
            raise ValueError(f"Recreated {filename} hash differs; inspect versions/protocol before scoring")
    print(json.dumps({"status": "verified", "users": len(users), "rows": len(result),
                      "input_sha256": sha256_file(root / "jev_input.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
