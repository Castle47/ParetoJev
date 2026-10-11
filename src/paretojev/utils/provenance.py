from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(start: str | Path | None = None) -> str | None:
    directory = Path(start or Path.cwd())
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def runtime_provenance(
    *,
    input_paths: Iterable[str | Path] = (),
    config: Any | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Capture compact, machine-readable provenance for an experiment artifact."""

    packages: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "requests", "PyYAML", "scipy", "joblib", "threadpoolctl"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue

    inputs: list[dict[str, Any]] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        entry: dict[str, Any] = {"path": str(path)}
        if path.exists() and path.is_file():
            stat = path.stat()
            entry.update(
                {
                    "type": "file",
                    "bytes": int(stat.st_size),
                    "sha256": sha256_file(path),
                }
            )
        elif path.exists() and path.is_dir():
            entry["type"] = "directory"
        else:
            entry["missing"] = True
        inputs.append(entry)

    result: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git_commit(repository),
        "inputs": inputs,
    }
    if config is not None:
        result["config_hash"] = canonical_hash(config)
    return result
