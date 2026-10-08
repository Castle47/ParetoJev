"""Strict interchange protocol for separately trained reranker comparisons."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from paretojev.data.precomputed import PrecomputedSlateDataset
from paretojev.data.synthetic import SyntheticConfig, SyntheticSlateDataset
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import canonical_hash, sha256_file

PROTOCOL_VERSION = "baseline-harness-v1"
SPLITS = ("train", "validation", "test")


class ComparisonSlateDataset(PrecomputedSlateDataset):
    """Feature-complete NPZ plus auditable identity and temporal metadata.

    Unlike the legacy training loader, this rejects incomplete identity, invalid
    probabilities and all-padding rows. Metadata never enters model features.
    Optional simulator latent targets are forbidden: all compared trainers use
    the same observed outcomes. Padding is zeroed before any model consumes it.
    """

    def __init__(self, path: str | Path, *, seed: int = 0) -> None:
        with np.load(path, allow_pickle=False) as data:
            metadata_keys = {
                "slate_ids", "candidate_ids", "user_ids", "timestamps",
                "objective_names", "directions", "feature_names", "feature_spec_hash",
                "data_kind",
            }
            missing = metadata_keys - set(data.files)
            if missing:
                raise ValueError(f"Comparison metadata missing: {sorted(missing)}")
            if "target_probabilities" in data.files:
                raise ValueError("Comparison protocol forbids privileged target_probabilities")
            self.metadata = {key: np.asarray(data[key]).copy() for key in metadata_keys}
            if data["mask"].dtype != np.bool_:
                raise ValueError("mask must be stored as boolean")
        super().__init__(path, seed=seed)
        n, k, m = len(self), self.slate_size, self.n_objectives
        if not n or self.feature_dim < 1 or m < 2:
            raise ValueError("Comparison requires nonempty slates/features and at least two objectives")
        meta = self.metadata
        for key in ("slate_ids", "user_ids", "candidate_ids", "objective_names", "feature_names", "feature_spec_hash", "data_kind"):
            if meta[key].dtype.kind == "S":
                meta[key] = np.char.decode(meta[key], "utf-8")
        for key in ("objective_names", "feature_names", "feature_spec_hash", "data_kind"):
            if meta[key].dtype.kind != "U":
                raise ValueError(f"{key} must contain text, not numeric identifiers")
        for key in ("slate_ids", "user_ids"):
            if meta[key].shape != (n,):
                raise ValueError(f"{key} must have shape [N]")
            if meta[key].dtype.kind not in "iuUS" or any(not str(x).strip() for x in meta[key]):
                raise ValueError(f"{key} must contain nonempty string or integer IDs")
        if len(set(meta["slate_ids"].astype(str))) != n:
            raise ValueError("slate_ids must be unique within a split")
        for key in ("candidate_ids", "timestamps"):
            if meta[key].shape != (n, k):
                raise ValueError(f"{key} must have shape [N,K]")
        if meta["candidate_ids"].dtype.kind not in "iuUS":
            raise ValueError("candidate_ids must contain strings or integers")
        if meta["timestamps"].dtype.kind not in "iu":
            raise ValueError("timestamps must be integer event times")
        names = meta["objective_names"].tolist()
        if meta["objective_names"].shape != (m,) or len(set(names)) != m or any(not str(x).strip() for x in names):
            raise ValueError("objective_names must be unique nonempty names in score order")
        if meta["directions"].shape != (m,) or not np.isin(meta["directions"], (-1, 1)).all():
            raise ValueError("directions must be +1/-1 in objective order")
        if meta["feature_names"].shape != (self.feature_dim,) or len(set(meta["feature_names"].tolist())) != self.feature_dim or any(not x.strip() for x in meta["feature_names"]):
            raise ValueError("feature_names must match feature columns and be unique")
        for key in ("feature_spec_hash", "data_kind"):
            if meta[key].shape != ():
                raise ValueError(f"{key} must be a scalar string")
        digest = str(meta["feature_spec_hash"].item())
        if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
            raise ValueError("feature_spec_hash must be a SHA-256 digest")
        mask = self.arrays["mask"].numpy()
        if (mask.sum(axis=1) < 2).any():
            raise ValueError("Every comparison slate must contain at least two valid candidates")
        if (meta["timestamps"][mask] < 0).any():
            raise ValueError("valid timestamps must be nonnegative")
        if any(not str(x).strip() for x in meta["candidate_ids"][mask]):
            raise ValueError("valid candidate IDs cannot be empty")
        for key, tensor in self.arrays.items():
            if key == "mask":
                continue
            values = tensor.numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"{key} must be finite, including padding")
            if key in ("behavior_scores", "jev_scores", "labels", "jev_confidence", "jev_entropy"):
                if ((values[mask] < 0) | (values[mask] > 1)).any():
                    raise ValueError(f"{key} must lie in [0,1] on valid candidates")
            if key in ("popularity", "history_length", "preference_weights") and (values < 0).any():
                raise ValueError(f"{key} cannot be negative")
            if values.ndim == 3:
                tensor[~self.arrays["mask"]] = 0
        weights = self.arrays["preference_weights"].numpy()
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("preference_weights must sum to 1")


def load_comparison_data(manifest_path: str | Path) -> tuple[dict[str, ComparisonSlateDataset], dict[str, Any]]:
    """Verify fixed chronological splits and a shared upstream feature transform."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported comparison protocol_version")
    kind = manifest.get("data_kind")
    if kind not in ("synthetic", "real"):
        raise ValueError("data_kind must explicitly be synthetic or real")
    if not isinstance(manifest.get("dataset"), str) or not manifest["dataset"]:
        raise ValueError("dataset name is required")
    if kind == "real" and manifest["dataset"] != "kuairand":
        raise ValueError("Real comparison harness currently supports kuairand only")
    datasets: dict[str, ComparisonSlateDataset] = {}
    fingerprints = {}
    for split in SPLITS:
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError(f"{split} needs a path and a registered sha256")
        path = (manifest_path.parent / entry["path"]).resolve()
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{split} file hash mismatch")
        dataset = ComparisonSlateDataset(path)
        if str(dataset.metadata["data_kind"].item()) != kind:
            raise ValueError(f"{split} data_kind conflicts with manifest")
        datasets[split] = dataset
        fingerprints[split] = {"path": str(path), "sha256": digest, "slates": len(dataset)}
    train = datasets["train"]
    common = ("objective_names", "directions", "feature_names", "feature_spec_hash")
    for split, dataset in datasets.items():
        if (dataset.feature_dim, dataset.n_objectives, dataset.slate_size) != (train.feature_dim, train.n_objectives, train.slate_size):
            raise ValueError(f"{split} dimensions differ from train")
        for key in common:
            if not np.array_equal(dataset.metadata[key], train.metadata[key]):
                raise ValueError(f"{split} {key} order/value differs from train")
    for left, right in zip(SPLITS, SPLITS[1:]):
        a, b = datasets[left], datasets[right]
        a_times = a.metadata["timestamps"][a.arrays["mask"].numpy()]
        b_times = b.metadata["timestamps"][b.arrays["mask"].numpy()]
        if a_times.max() >= b_times.min():
            raise ValueError(f"Strict chronological split violated: {left} overlaps {right}")
    if kind == "real":
        # This is an explicit input provenance contract, not automatic proof of
        # honest upstream generation. Hash and preserve each evidence document.
        required = ("feature_spec", "behavior_provenance", "decision_provenance", "evaluation_protocol")
        for key in required:
            entry = manifest.get("upstream", {}).get(key, {})
            if not entry.get("path") or not entry.get("sha256"):
                raise ValueError(f"Real comparison requires upstream {key} evidence")
            path = (manifest_path.parent / entry["path"]).resolve()
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"upstream {key} hash mismatch")
            if key == "feature_spec" and canonical_hash(read_json(path)) != train.metadata["feature_spec_hash"].item():
                raise ValueError("upstream feature_spec does not match the NPZ feature transform")
    return datasets, {"manifest": manifest, "manifest_sha256": sha256_file(manifest_path), "splits": fingerprints}


def write_synthetic_comparison_data(directory: str | Path, *, seed: int = 123,
                                    sizes: tuple[int, int, int] = (96, 48, 48),
                                    slate_size: int = 8, n_objectives: int = 3,
                                    feature_dim: int = 6) -> Path:
    """Write a deterministic protocol fixture; never label it a real benchmark."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Synthetic data output directory must be empty")
    if len(sizes) != 3 or any(size < 2 for size in sizes):
        raise ValueError("Provide three split sizes of at least two slates")
    if slate_size < 2 or n_objectives < 2 or feature_dim < 1:
        raise ValueError("Require slate_size>=2, n_objectives>=2 and feature_dim>=1")
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION, "dataset": "synthetic",
        "data_kind": "synthetic", "splits": {},
        "generator": {"seed": seed, "sizes": list(sizes), "slate_size": slate_size,
                      "n_objectives": n_objectives, "feature_dim": feature_dim},
    }
    names = [f"objective_{i}" for i in range(n_objectives)]
    spec = {"generator": "SyntheticSlateDataset", "feature_dim": feature_dim}
    time_offset = 0
    for index, (split, size) in enumerate(zip(SPLITS, sizes, strict=True)):
        config = SyntheticConfig(size, slate_size, n_objectives, feature_dim, seed + index * 10000, n_objectives - 1)
        data = SyntheticSlateDataset(config)
        arrays = {key: value.numpy().copy() for key, value in data.arrays.items()
                  if key != "target_probabilities"}
        # Padded candidates exercise the mask without changing the data story.
        if slate_size > 2:
            arrays["mask"][::3, -1] = False
        arrays.update(
            slate_ids=np.asarray([f"{split}-{i}" for i in range(size)]),
            candidate_ids=np.arange(size * slate_size).reshape(size, slate_size),
            user_ids=np.arange(size) // 2,
            timestamps=time_offset + np.arange(size * slate_size).reshape(size, slate_size),
            objective_names=np.asarray(names), directions=np.asarray([1] * (n_objectives - 1) + [-1]),
            feature_names=np.asarray([f"feature_{i}" for i in range(feature_dim)]),
            feature_spec_hash=np.asarray(canonical_hash(spec)), data_kind=np.asarray("synthetic"),
        )
        time_offset += size * slate_size + 1
        path = directory / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        manifest["splits"][split] = {"path": path.name, "sha256": sha256_file(path), "generator": asdict(config)}
    path = directory / "manifest.json"
    atomic_write_json(path, manifest)
    return path
