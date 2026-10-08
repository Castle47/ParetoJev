from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import SlateBatch


@dataclass(frozen=True)
class SyntheticConfig:
    n_slates: int = 512
    slate_size: int = 20
    n_objectives: int = 6
    feature_dim: int = 12
    seed: int = 7
    hate_objective: int | None = 5


class SyntheticSlateDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic task where behavior and Jev signals are deliberately complementary.

    The behavior signal is strongest on popular/warm items; the Jev-like signal is
    strongest on cold items. This makes it useful for exercising the confidence gate.
    """

    def __init__(self, config: SyntheticConfig = SyntheticConfig()) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        n, k, m, f = (
            config.n_slates,
            config.slate_size,
            config.n_objectives,
            config.feature_dim,
        )

        popularity = rng.beta(1.5, 2.0, size=(n, k, 1)).astype(np.float32)
        semantic = rng.normal(size=(n, k, m)).astype(np.float32)
        collaborative = rng.normal(size=(n, k, m)).astype(np.float32)
        user_taste = rng.normal(scale=0.5, size=(n, 1, m)).astype(np.float32)

        raw = 0.62 * semantic + 0.38 * collaborative + user_taste
        if config.hate_objective is not None and config.hate_objective < m:
            # Hate risk is positively represented as a probability; the optimization
            # direction is supplied separately as -1.
            raw[..., config.hate_objective] = (
                -0.35 * semantic[..., config.hate_objective]
                + 0.45 * collaborative[..., config.hate_objective]
                + 0.6 * rng.normal(size=(n, k)).astype(np.float32)
            )
        true_p = 1.0 / (1.0 + np.exp(-raw))

        # Jev-like semantic signal: low noise for cold items, higher for warm items.
        jev_std = 0.05 + 0.18 * popularity
        jev_noise = rng.normal(size=(n, k, m)).astype(np.float32) * jev_std
        jev = np.clip(true_p + jev_noise, 1e-4, 1 - 1e-4)

        # Behavioral signal: high noise for cold items, lower for warm items.
        behavior_std = 0.04 + 0.28 * (1.0 - popularity)
        behavior_noise = rng.normal(size=(n, k, m)).astype(np.float32) * behavior_std
        behavior = np.clip(true_p + behavior_noise, 1e-4, 1 - 1e-4)

        jev_conf = np.clip(1.0 - np.abs(jev_noise) / 0.8, 0.0, 1.0).astype(np.float32)
        jev_entropy = (-(jev * np.log(jev) + (1 - jev) * np.log(1 - jev)) / math.log(2.0)).astype(np.float32)

        labels = rng.binomial(1, true_p).astype(np.float32)
        weights = rng.dirichlet(np.ones(m), size=n).astype(np.float32)
        history_length = rng.integers(3, 100, size=(n, 1)).astype(np.float32)

        base_features = rng.normal(size=(n, k, max(0, f - 3))).astype(np.float32)
        candidate_features = np.concatenate(
            [
                semantic.mean(axis=-1, keepdims=True),
                collaborative.mean(axis=-1, keepdims=True),
                popularity,
                base_features,
            ],
            axis=-1,
        )[..., :f]

        self.arrays = {
            "candidate_features": torch.from_numpy(candidate_features),
            "behavior_scores": torch.from_numpy(behavior.astype(np.float32)),
            "jev_scores": torch.from_numpy(jev.astype(np.float32)),
            "labels": torch.from_numpy(labels),
            "preference_weights": torch.from_numpy(weights),
            "mask": torch.ones((n, k), dtype=torch.bool),
            "jev_confidence": torch.from_numpy(jev_conf),
            "jev_entropy": torch.from_numpy(jev_entropy),
            "popularity": torch.from_numpy(popularity),
            "history_length": torch.from_numpy(history_length),
            "target_probabilities": torch.from_numpy(true_p.astype(np.float32)),
        }

    def __len__(self) -> int:
        return self.config.n_slates

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.arrays.items()}


def collate_slates(rows: list[dict[str, torch.Tensor]]) -> SlateBatch:
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    stacked = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    batch = SlateBatch(**stacked)
    batch.validate()
    return batch
