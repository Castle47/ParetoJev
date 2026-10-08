from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_REQUIRED_KEYS = {
    "candidate_features",
    "behavior_scores",
    "jev_scores",
    "labels",
    "mask",
    "jev_confidence",
    "jev_entropy",
    "popularity",
    "history_length",
}


class PrecomputedSlateDataset(Dataset[dict[str, torch.Tensor]]):
    """Load feature-complete slates for ParetoJev training.

    The NPZ contract deliberately separates feature generation from reranker
    training. This lets behavior models and Jev scoring run on different hardware
    while keeping the final experiment deterministic and auditable.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seed: int = 0,
        dirichlet_alpha: float = 1.0,
    ) -> None:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            missing = _REQUIRED_KEYS - set(data.files)
            if missing:
                raise ValueError(f"Precomputed NPZ is missing keys: {sorted(missing)}")
            arrays = {key: np.asarray(data[key]) for key in data.files}

        n, k, feature_dim = arrays["candidate_features"].shape
        _, k_scores, objectives = arrays["behavior_scores"].shape
        if k_scores != k:
            raise ValueError("candidate and behavior slate sizes differ")
        expected_bkm = (n, k, objectives)
        for key in ("behavior_scores", "jev_scores", "labels", "jev_confidence", "jev_entropy"):
            if arrays[key].shape != expected_bkm:
                raise ValueError(f"{key} shape {arrays[key].shape}, expected {expected_bkm}")
        if arrays["mask"].shape != (n, k):
            raise ValueError("mask shape mismatch")
        if arrays["popularity"].shape == (n, k):
            arrays["popularity"] = arrays["popularity"][..., None]
        if arrays["popularity"].shape != (n, k, 1):
            raise ValueError("popularity must have shape [N,K] or [N,K,1]")
        if arrays["history_length"].shape == (n,):
            arrays["history_length"] = arrays["history_length"][:, None]
        if arrays["history_length"].shape != (n, 1):
            raise ValueError("history_length must have shape [N] or [N,1]")
        if "target_probabilities" in arrays and arrays["target_probabilities"].shape != expected_bkm:
            raise ValueError("target_probabilities shape mismatch")

        if "preference_weights" not in arrays:
            rng = np.random.default_rng(seed)
            arrays["preference_weights"] = rng.dirichlet(
                np.full(objectives, dirichlet_alpha), size=n
            ).astype(np.float32)
        if arrays["preference_weights"].shape != (n, objectives):
            raise ValueError("preference_weights shape mismatch")

        # Normalize storage dtypes once, rather than in every __getitem__ call.
        float_keys = {
            "candidate_features",
            "behavior_scores",
            "jev_scores",
            "labels",
            "preference_weights",
            "jev_confidence",
            "jev_entropy",
            "popularity",
            "history_length",
            "target_probabilities",
        }
        self.arrays: dict[str, torch.Tensor] = {}
        for key, value in arrays.items():
            if key not in _REQUIRED_KEYS | {"preference_weights", "target_probabilities"}:
                continue
            if key == "mask":
                tensor = torch.from_numpy(value.astype(bool, copy=False))
            elif key in float_keys:
                tensor = torch.from_numpy(value.astype(np.float32, copy=False))
            else:
                tensor = torch.from_numpy(value)
            self.arrays[key] = tensor
        self.feature_dim = feature_dim
        self.n_objectives = objectives
        self.slate_size = k

    def __len__(self) -> int:
        return int(self.arrays["candidate_features"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.arrays.items()}
