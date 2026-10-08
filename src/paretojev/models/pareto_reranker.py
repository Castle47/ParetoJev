from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from paretojev.data.schema import SlateBatch
from .fusion import ConfidenceAwareFusion


@dataclass(frozen=True)
class ParetoJevConfig:
    candidate_feature_dim: int
    n_objectives: int
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    fusion_hidden_dim: int = 32
    predict_uncertainty: bool = True


class PreferenceFiLM(nn.Module):
    def __init__(self, n_objectives: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_objectives, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(weights).chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma[:, None, :] * x + beta[:, None, :]


class ParetoJevReranker(nn.Module):
    """Set-aware, preference-conditioned reranker.

    Inputs combine candidate features, behavioral pXTR-like signals, Jev-like
    decision probabilities, learned confidence gates, and a runtime objective
    preference vector. A Transformer encoder models comparisons inside the slate.
    """

    def __init__(self, config: ParetoJevConfig) -> None:
        super().__init__()
        self.config = config
        m = config.n_objectives
        self.fusion = ConfidenceAwareFusion(config.fusion_hidden_dim, config.dropout)

        # candidate features + behavior + jev + fused + gates + reliability summary
        input_dim = config.candidate_feature_dim + 4 * m + 3
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.preference_film = PreferenceFiLM(m, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.score_head = nn.Linear(config.hidden_dim, 1)
        self.objective_head = nn.Linear(config.hidden_dim, m)
        self.logvar_head = nn.Linear(config.hidden_dim, 1) if config.predict_uncertainty else None

    def forward(self, batch: SlateBatch) -> dict[str, torch.Tensor]:
        batch.validate()
        fused, gate = self.fusion(
            batch.behavior_scores,
            batch.jev_scores,
            batch.jev_entropy,
            batch.jev_confidence,
            batch.popularity,
            batch.history_length,
        )
        reliability = torch.cat(
            [
                batch.jev_confidence.mean(dim=-1, keepdim=True),
                batch.jev_entropy.mean(dim=-1, keepdim=True),
                batch.popularity,
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                batch.candidate_features,
                batch.behavior_scores,
                batch.jev_scores,
                fused,
                gate,
                reliability,
            ],
            dim=-1,
        )
        hidden = self.input_projection(features)
        hidden = self.preference_film(hidden, batch.preference_weights)
        encoded = self.set_encoder(hidden, src_key_padding_mask=~batch.mask)

        mean = self.score_head(encoded).squeeze(-1)
        objective_logits = self.objective_head(encoded)
        mean = mean.masked_fill(~batch.mask, torch.finfo(mean.dtype).min)
        objective_logits = objective_logits.masked_fill(~batch.mask.unsqueeze(-1), 0.0)

        output: dict[str, torch.Tensor] = {
            "score_mean": mean,
            "objective_logits": objective_logits,
            "fused_scores": fused,
            "fusion_gate": gate,
        }
        if self.logvar_head is not None:
            logvar = self.logvar_head(encoded).squeeze(-1).clamp(-8.0, 5.0)
            logvar = logvar.masked_fill(~batch.mask, 0.0)
            output["score_logvar"] = logvar
        return output
