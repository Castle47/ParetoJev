from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass
class SlateBatch:
    """A fixed-width recommendation slate.

    Shapes:
      candidate_features: [B, K, F]
      behavior_scores:    [B, K, M]
      jev_scores:         [B, K, M]
      labels:             [B, K, M]
      preference_weights: [B, M]
      mask:               [B, K] (True for valid candidates)
      jev_confidence:     [B, K, M]
      jev_entropy:        [B, K, M]
      popularity:         [B, K, 1]
      history_length:     [B, 1]
    """

    candidate_features: torch.Tensor
    behavior_scores: torch.Tensor
    jev_scores: torch.Tensor
    labels: torch.Tensor
    preference_weights: torch.Tensor
    mask: torch.Tensor
    jev_confidence: torch.Tensor
    jev_entropy: torch.Tensor
    popularity: torch.Tensor
    history_length: torch.Tensor
    target_probabilities: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    @property
    def batch_size(self) -> int:
        return int(self.candidate_features.shape[0])

    @property
    def slate_size(self) -> int:
        return int(self.candidate_features.shape[1])

    @property
    def n_objectives(self) -> int:
        return int(self.behavior_scores.shape[-1])

    def to(self, device: torch.device | str) -> "SlateBatch":
        kwargs: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            kwargs[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return SlateBatch(**kwargs)

    def validate(self) -> None:
        b, k, _ = self.candidate_features.shape
        expected_bkm = (b, k, self.n_objectives)
        for name in ("behavior_scores", "jev_scores", "labels", "jev_confidence", "jev_entropy"):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != expected_bkm:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected_bkm}")
        if tuple(self.preference_weights.shape) != (b, self.n_objectives):
            raise ValueError("preference_weights shape mismatch")
        if tuple(self.mask.shape) != (b, k):
            raise ValueError("mask shape mismatch")
        if tuple(self.popularity.shape) != (b, k, 1):
            raise ValueError("popularity shape mismatch")
        if tuple(self.history_length.shape) != (b, 1):
            raise ValueError("history_length shape mismatch")
        if self.target_probabilities is not None and tuple(self.target_probabilities.shape) != expected_bkm:
            raise ValueError("target_probabilities shape mismatch")
        weight_sums = self.preference_weights.sum(dim=-1)
        if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4):
            raise ValueError("preference_weights must sum to 1 per row")
