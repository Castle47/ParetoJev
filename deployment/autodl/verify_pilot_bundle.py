#!/usr/bin/env python3
"""CPU-only validation for the fixed 10K packet and exact model environment."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main(args):
    manifest = json.loads((args.input_dir / "transfer.json").read_text())
    development = args.input_dir / "development-registration.json"
    if (sha(development) != manifest["development_registration_sha256"] or
            manifest.get("labels_included") or not manifest.get("pilot_subset")):
        raise ValueError("Pilot upload scope or producer identity differs")
    if manifest.get("slates") != 10000 or manifest.get("planned_decisions") != 70000:
        raise ValueError("Expected exactly the frozen 10K pilot packet")
    if sha(args.input_dir / "pilot_config.json") != manifest["confirmation_config_sha256"]:
        raise ValueError("Pilot protocol differs")
    payload = args.input_dir / "jev_input.jsonl.gz"
    if sha(payload) != manifest["compressed_sha256"]:
        raise ValueError("Uploaded input hash differs")
    with gzip.open(payload, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["source_input_sha256"]:
            raise ValueError("Decoded pilot input differs")
    source = json.loads(development.read_text())["source"]
    verified = 0
    for name, expected in source["files"].items():
        path = Path("/root/torch_head_server.py") if name == "custom/torch_head_server.py" else args.model_dir / name
        if sha(path) != expected:
            raise ValueError("Producer file differs: " + name)
        verified += 1
    for name, expected in source["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            raise ValueError("Producer package differs: " + name)
    result = {
        "input_and_model_hashes_verified": True,
        "producer_files": verified,
        "compressed_sha256": manifest["compressed_sha256"],
        "development_registration_sha256": sha(development),
        "packages_match": True,
        "gpu_inference_started": False
    }
    (args.input_dir / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/root/autodl-tmp/JEV-27B"))
    main(parser.parse_args())
