#!/usr/bin/env python3
"""Freeze a label-free 10K pilot and preserve the untouched 40K remainder."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash, sha256_file


def priority(namespace, seed, slate_id):
    value = f"{namespace}|{seed}|{int(slate_id)}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def select(strata, config):
    chosen = []
    selection = config["selection"]
    for stratum, quota in enumerate(selection["stratum_quotas"]):
        ids = np.flatnonzero(strata == stratum)
        ranked = sorted(ids.tolist(), key=lambda row: (priority(selection["namespace"], selection["seed"], row), row))
        if len(ranked) < quota:
            raise ValueError("Pilot quota exceeds the frozen source stratum")
        chosen.extend(ranked[:quota])
    chosen = np.array(sorted(chosen), dtype="int64")
    if len(chosen) != selection["slates"] or len(np.unique(chosen)) != len(chosen):
        raise ValueError("Pilot selection coverage differs")
    return chosen


def main(args):
    config = read_json(args.config)
    parent_config = read_json(args.parent_config)
    if config["parent_confirmation_config"] != str(args.parent_config):
        raise ValueError("Parent confirmation path differs")
    args.output.mkdir(parents=True, exist_ok=True)
    registration = args.output / "registration.json"
    identity = {
        "config": config,
        "config_hash": canonical_hash(config),
        "config_sha256": sha256_file(args.config),
        "parent_config_sha256": sha256_file(args.parent_config),
        "parent_preparation_sha256": sha256_file(args.source / "preparation.json"),
        "parent_cohort_audit_sha256": sha256_file(args.source / "cohort_audit.json"),
        "parent_sampling_replay_sha256": sha256_file(args.source / "sampling_replay.json"),
        "parent_raw_sha256": sha256_file(args.source / "raw_slates.npz"),
        "parent_prompt_sha256": sha256_file(args.source / "jev_input.jsonl"),
        "script_sha256": sha256_file(Path(__file__)),
        "labels_or_model_outputs_used_for_selection": False
    }
    if registration.exists() and read_json(registration) != identity:
        raise ValueError("Pilot registration changed")
    atomic_write_json(registration, identity)
    if (args.output / "complete.json").exists():
        print("Frozen 10K pilot already prepared", flush=True)
        return
    with np.load(args.source / "raw_slates.npz", allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}
    if len(raw["user"]) != 50000 or raw["stratum"].shape != (50000,):
        raise ValueError("Expected the frozen 50K source cohort")
    chosen = select(raw["stratum"], config)
    heldout = np.setdiff1d(np.arange(50000), chosen, assume_unique=True)
    if len(heldout) != config["selection"]["heldout_remaining_slates"]:
        raise ValueError("Heldout remainder differs")
    np.savez_compressed(args.output / "raw_slates.npz", **{key: value[chosen] for key, value in raw.items()})
    np.savez_compressed(args.output / "selection.npz", pilot_original_slate_ids=chosen, heldout_original_slate_ids=heldout)
    selected_set = set(chosen.tolist())
    source_rows = 0
    pilot_rows = []
    with (args.source / "jev_input.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["slate_id"] != source_rows:
                raise ValueError("Parent prompt order differs")
            if source_rows in selected_set:
                copy = dict(row)
                copy["slate_id"] = len(pilot_rows)
                pilot_rows.append(copy)
            source_rows += 1
    if source_rows != 50000 or len(pilot_rows) != 10000:
        raise ValueError("Prompt selection coverage differs")
    prompt = args.output / "jev_input.jsonl"
    with prompt.open("w") as handle:
        for row in pilot_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    compressed = args.output / "jev_input.jsonl.gz"
    with prompt.open("rb") as source, compressed.open("wb") as target:
        with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as stream:
            for block in iter(lambda: source.read(1 << 20), b""):
                stream.write(block)
    parent_dev = parent_config["candidate_selection"]["source_registration_sha256"]
    manifest = {
        "slates": 10000,
        "planned_decisions": 70000,
        "independent_test_included": True,
        "pilot_subset": True,
        "labels_included": False,
        "source_input_sha256": sha256_file(prompt),
        "compressed_sha256": sha256_file(compressed),
        "confirmation_config_sha256": sha256_file(args.config),
        "parent_confirmation_config_sha256": sha256_file(args.parent_config),
        "development_registration_sha256": parent_dev,
        "selection_sha256": sha256_file(args.output / "selection.npz"),
        "parent_raw_sha256": sha256_file(args.source / "raw_slates.npz"),
        "pilot_raw_sha256": sha256_file(args.output / "raw_slates.npz"),
        "note": "Label-free fixed-hash pilot packet; 40K remainder is not scored by this run."
    }
    atomic_write_json(args.output / "transfer.json", manifest)
    counts = np.bincount(raw["stratum"][chosen], minlength=6).tolist()
    receipt = {
        "complete": True,
        "pilot_slates": len(chosen),
        "heldout_slates": len(heldout),
        "stratum_counts": counts,
        "users": int(len(np.unique(raw["user"][chosen]))),
        "selection_min": int(chosen.min()),
        "selection_max": int(chosen.max()),
        "pilot_dates_opened_for_performance": False,
        "labels_read_for_selection": False,
        "raw_sha256": manifest["pilot_raw_sha256"],
        "prompt_sha256": manifest["source_input_sha256"],
        "transfer_sha256": sha256_file(args.output / "transfer.json")
    }
    atomic_write_json(args.output / "complete.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/emer_10k_pilot_v1.json"))
    parser.add_argument("--parent-config", type=Path, default=Path("configs/emer_multiobjective_confirmation.json"))
    main(parser.parse_args())
