#!/usr/bin/env python3
"""Create a deterministic upload bundle for the frozen 10K AutoDL pilot."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file


FILES = {
    "jev_input.jsonl.gz": "jev_input.jsonl.gz",
    "transfer.json": "transfer.json",
    "pilot_config.json": "config",
    "development-registration.json": "development_registration",
    "score_confirmation.py": "score_confirmation",
    "score_pilot.py": "score_pilot",
    "verify_pilot_bundle.py": "verify_pilot_bundle",
    "export_confirmation.py": "export_confirmation",
    "setup_pilot.sh": "setup_pilot",
    "start_pilot.sh": "start_pilot",
    "torch_head_server.py": "torch_head_server"
}


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Use a fresh immutable pilot bundle directory")
    sources = {
        "jev_input.jsonl.gz": args.pilot / "jev_input.jsonl.gz",
        "transfer.json": args.pilot / "transfer.json",
        "pilot_config.json": args.config,
        "development-registration.json": args.development / "jev-native-v9-fp16/registration.json",
        "score_confirmation.py": args.repo / "deployment/autodl/score_confirmation.py",
        "score_pilot.py": args.repo / "deployment/autodl/score_pilot.py",
        "verify_pilot_bundle.py": args.repo / "deployment/autodl/verify_pilot_bundle.py",
        "export_confirmation.py": args.repo / "deployment/autodl/export_confirmation.py",
        "setup_pilot.sh": args.repo / "deployment/autodl/setup_pilot.sh",
        "start_pilot.sh": args.repo / "deployment/autodl/start_pilot.sh",
        "torch_head_server.py": args.repo / "deployment/autodl/torch_head_server.py"
    }
    for name, source in sources.items():
        (args.output / name).write_bytes(source.read_bytes())
    transfer = read_json(args.output / "transfer.json")
    if (sha256_file(args.output / "jev_input.jsonl.gz") != transfer["compressed_sha256"] or
            sha256_file(args.output / "pilot_config.json") != transfer["confirmation_config_sha256"] or
            sha256_file(args.output / "development-registration.json") != transfer["development_registration_sha256"]):
        raise ValueError("Pilot upload identity differs")
    with gzip.open(args.output / "jev_input.jsonl.gz", "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != transfer["source_input_sha256"]:
            raise ValueError("Decoded pilot upload differs")
    files = {name: sha256_file(args.output / name) for name in sorted(sources)}
    manifest = {
        "pilot_slates": 10000,
        "planned_decisions": 70000,
        "files": files,
        "transfer_sha256": sha256_file(args.output / "transfer.json"),
        "labels_included": False,
        "model_weights_included": False,
        "paid_inference_started": False
    }
    atomic_write_json(args.output / "bundle.json", manifest)
    archive = args.output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(args.output.iterdir()):
            tar.add(path, arcname=args.output.name + "/" + path.name, recursive=False)
    print(json.dumps({
        "bundle": str(args.output),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "input_sha256": transfer["compressed_sha256"]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/emer_10k_pilot_v1.json"))
    main(parser.parse_args())
