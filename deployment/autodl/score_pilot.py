#!/usr/bin/env python3
"""Collect the registered 10K pilot with the exact development producer."""
import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time

import score_confirmation as scorer


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads(args.manifest.read_text())
        if scorer.digest(args.input) != manifest["compressed_sha256"]:
            raise ValueError("Uploaded compressed pilot differs")
        with gzip.open(args.input, "rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["source_input_sha256"]:
                raise ValueError("Decoded pilot prompts differ")
        end = dt.datetime.fromisoformat(args.stop_at)
        if end.tzinfo is None or time.time() >= end.timestamp():
            raise ValueError("Missing timezone or expired pilot deadline")
        rows = scorer.load_rows(args.input)
        if (len(rows) != 10000 or manifest.get("slates") != 10000 or
                manifest.get("planned_decisions") != 70000 or not manifest.get("pilot_subset") or
                not manifest.get("independent_test_included") or manifest.get("labels_included")):
            raise ValueError("Frozen 10K pilot scope changed")
        development = json.loads(args.development_registration.read_text())
        identity = scorer.producer(args.model_dir, args.endpoint)
        scorer.require_same_producer(identity, development)
        if manifest["development_registration_sha256"] != scorer.digest(args.development_registration):
            raise ValueError("Development producer receipt differs")
        registration = {
            "input_sha256": scorer.digest(args.input),
            "source": identity,
            "slates": len(rows),
            "goals": rows[0]["objective_names"],
            "script_sha256": scorer.digest(__file__),
            "collector_library_sha256": scorer.digest(scorer.__file__),
            "max_attempts": args.max_attempts,
            "stop_at": args.stop_at,
            "workers": args.workers,
            "separate_probe_http_requests": 144,
            "separate_probe_decisions": 336,
            "transfer_manifest_sha256": scorer.digest(args.manifest),
            "transport": "One real HTTP group per pilot slate; exact wire envelopes retained.",
            "collection_order": "Ascending reindexed pilot IDs; fixed original IDs are in selection.npz.",
            "kind": "frozen exploratory new-date 10K pilot; no imported public responses",
            "test_opened": True,
            "development_registration_sha256": scorer.digest(args.development_registration),
            "confirmation_config_sha256": manifest["confirmation_config_sha256"]
        }
        path = args.output / "registration.json"
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError("Pilot scoring registration changed")
        scorer.save(path, registration)
        if not (args.output / "probe.json").exists():
            scorer.probe(rows, args.endpoint, args.output)
        if not json.loads((args.output / "probe.json").read_text())["passed"]:
            raise ValueError("Producer probe has not passed")
        if args.stage == "score":
            scorer.collect(rows, args, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-registration", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/root/autodl-tmp/JEV-27B"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/decide")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--max-attempts", type=int, default=11000)
    parser.add_argument("--stage", choices=["probe", "score"], default="probe")
    main(parser.parse_args())
