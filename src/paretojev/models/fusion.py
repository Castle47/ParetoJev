from __future__ import annotations

import torch
from torch import nn


class ConfidenceAwareFusion(nn.Module):
    """Fuse behavioral and decision-model probabilities with a learned gate.

    The same small gate network is shared across objectives, which keeps the
    parameter count low and encourages a reusable reliability rule.
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        # behavior, jev, absolute disagreement, entropy, confidence,
        # log-popularity, normalized history length
        self.gate = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        behavior_scores: torch.Tensor,
        jev_scores: torch.Tensor,
        jev_entropy: torch.Tensor,
        jev_confidence: torch.Tensor,
        popularity: torch.Tensor,
        history_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if behavior_scores.shape != jev_scores.shape:
            raise ValueError("behavior_scores and jev_scores must have identical shapes")
        b, k, m = behavior_scores.shape
        if popularity.shape != (b, k, 1):
            raise ValueError(f"popularity must have shape {(b, k, 1)}")
        if history_length.shape != (b, 1):
            raise ValueError(f"history_length must have shape {(b, 1)}")

        pop = torch.log1p(torch.clamp(popularity, min=0.0)).expand(-1, -1, m)
        hist = torch.log1p(torch.clamp(history_length, min=0.0))
        hist = hist[:, None, :].expand(-1, k, m)
        gate_features = torch.stack(
            [
                behavior_scores,
                jev_scores,
                (behavior_scores - jev_scores).abs(),
                jev_entropy,
                jev_confidence,
                pop,
                hist,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_features).squeeze(-1))
        fused = gate * jev_scores + (1.0 - gate) * behavior_scores
        return fused, gate
