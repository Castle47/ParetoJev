from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SASRecMMoEConfig:
    num_items: int
    n_objectives: int
    max_history_length: int = 100
    embedding_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    n_experts: int = 4
    expert_dim: int = 128
    context_dim: int = 0
    dropout: float = 0.1
    padding_idx: int = 0

    def __post_init__(self) -> None:
        if self.num_items <= 1:
            raise ValueError("num_items must be greater than 1")
        if self.n_objectives < 1:
            raise ValueError("n_objectives must be positive")
        if self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")


class SASRecEncoder(nn.Module):
    """Compact SASRec-style user encoder for the behavior branch."""

    def __init__(self, config: SASRecMMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(
            config.num_items,
            config.embedding_dim,
            padding_idx=config.padding_idx,
        )
        self.position_embedding = nn.Embedding(config.max_history_length, config.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, history_item_ids: torch.Tensor) -> torch.Tensor:
        if history_item_ids.ndim != 2:
            raise ValueError("history_item_ids must have shape [B,L]")
        batch, length = history_item_ids.shape
        if length > self.config.max_history_length:
            history_item_ids = history_item_ids[:, -self.config.max_history_length :]
            length = history_item_ids.shape[1]
        padding_mask = history_item_ids.eq(self.config.padding_idx)
        positions = torch.arange(length, device=history_item_ids.device)[None, :].expand(batch, -1)
        hidden = self.item_embedding(history_item_ids) + self.position_embedding(positions)
        hidden = self.dropout(self.norm(hidden))
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=history_item_ids.device),
            diagonal=1,
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        valid_positions = torch.where(
            ~padding_mask,
            positions,
            torch.full_like(positions, -1),
        )
        last_index = valid_positions.max(dim=1).values.clamp_min(0)
        user = encoded[torch.arange(batch, device=encoded.device), last_index]
        all_padding = padding_mask.all(dim=1)
        return user.masked_fill(all_padding[:, None], 0.0)


class SASRecMMoERanker(nn.Module):
    """SASRec user state plus MMoE task heads for pXTR-like behavior scores."""

    def __init__(self, config: SASRecMMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.user_encoder = SASRecEncoder(config)
        # Share the exact item embedding table with the user encoder.
        self.item_embedding = self.user_encoder.item_embedding
        feature_dim = 3 * config.embedding_dim + config.context_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, config.expert_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.expert_dim, config.expert_dim),
                    nn.SiLU(),
                )
                for _ in range(config.n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(feature_dim, config.n_experts) for _ in range(config.n_objectives)]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(config.expert_dim, 1) for _ in range(config.n_objectives)]
        )

    def forward(
        self,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        candidate_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return objective logits with shape [B,K,M]."""

        if candidate_item_ids.ndim != 2:
            raise ValueError("candidate_item_ids must have shape [B,K]")
        batch, candidates = candidate_item_ids.shape
        user = self.user_encoder(history_item_ids)
        item = self.item_embedding(candidate_item_ids)
        user_expanded = user[:, None, :].expand(-1, candidates, -1)
        features = [user_expanded, item, user_expanded * item]
        if self.config.context_dim:
            if candidate_context is None:
                raise ValueError("candidate_context is required when context_dim > 0")
            if candidate_context.shape != (batch, candidates, self.config.context_dim):
                raise ValueError("candidate_context shape mismatch")
            features.append(candidate_context)
        elif candidate_context is not None and candidate_context.shape[-1] != 0:
            raise ValueError("Set context_dim in config before supplying candidate_context")
        joint = torch.cat(features, dim=-1)
        expert_outputs = torch.stack([expert(joint) for expert in self.experts], dim=-2)
        task_logits: list[torch.Tensor] = []
        for gate, head in zip(self.gates, self.heads, strict=True):
            mixture = torch.softmax(gate(joint), dim=-1)
            task_hidden = (expert_outputs * mixture[..., None]).sum(dim=-2)
            task_logits.append(head(task_hidden).squeeze(-1))
        return torch.stack(task_logits, dim=-1)

    def probabilities(
        self,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        candidate_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self(history_item_ids, candidate_item_ids, candidate_context))
