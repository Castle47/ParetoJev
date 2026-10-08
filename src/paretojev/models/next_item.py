"""Compact SASRec-style next-item model for implicit interaction sequences.

This is a modern pre-norm PyTorch implementation, not an exact reproduction of
Kang & McAuley's TensorFlow graph. PAD=0, OOV=1, and both have fixed zero vectors.
Scores are uncalibrated next-item logits, not click/long-view probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NextItemConfig:
    num_items: int
    max_length: int = 50
    hidden: int = 64
    heads: int = 2
    layers: int = 2
    dropout: float = 0.2


class SASRecNextItem(nn.Module):
    def __init__(self, config: NextItemConfig):
        super().__init__()
        if config.num_items < 3 or config.max_length < 1:
            raise ValueError("Need a nonempty training catalog and positive sequence length")
        self.config = config
        self.items = nn.Embedding(config.num_items, config.hidden, padding_idx=0)
        self.positions = nn.Embedding(config.max_length, config.hidden)
        block = nn.TransformerEncoderLayer(config.hidden, config.heads, config.hidden * 4,
            config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, config.layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden)
        nn.init.normal_(self.items.weight, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)
        self.reset_reserved()

    @torch.no_grad()
    def reset_reserved(self):
        self.items.weight[:2].zero_()

    def states(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 2 or not 1 <= history.shape[1] <= self.config.max_length:
            raise ValueError("History must be [B,L] with 1 <= L <= max_length")
        padding = history == 0
        # Right padding only. Empty rows get one harmless unmasked key to avoid
        # all-masked attention softmax; their final output is explicitly zero.
        if ((history[:, 1:] != 0) & padding[:, :-1]).any():
            raise ValueError("History must use right padding")
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), 0] = False
        x = self.items(history) * self.config.hidden ** 0.5
        x = self.dropout(x + self.positions(torch.arange(history.shape[1], device=history.device)))
        x = x.masked_fill(padding[..., None], 0)
        causal = torch.ones(history.shape[1], history.shape[1], dtype=torch.bool, device=history.device).triu(1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=safe_padding)
        return self.norm(x).masked_fill(padding[..., None], 0)

    def forward(self, history: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        states = self.states(history)
        last = (history.ne(0).sum(dim=1) - 1).clamp_min(0)
        user = states[torch.arange(len(history), device=history.device), last]
        return torch.einsum("bd,bkd->bk", user, self.items(candidates))

    def sampled_loss(self, history, positives, negatives, valid):
        states = self.states(history)
        pos = (states * self.items(positives)).sum(dim=-1)
        neg = (states * self.items(negatives)).sum(dim=-1)
        if not valid.any():
            raise ValueError("Batch has no chronological next-item targets")
        return (F.softplus(-pos)[valid] + F.softplus(neg)[valid]).mean()

    def specification(self):
        return asdict(self.config)


def training_arrays(sequences, timestamps, max_length):
    """One last-window example per user; only strictly later targets contribute."""
    x = np.zeros((len(sequences), max_length), dtype=np.int64)
    y = np.zeros_like(x)
    valid = np.zeros_like(x, dtype=bool)
    for index, (items, times) in enumerate(zip(sequences, timestamps, strict=True)):
        if len(items) != len(times) or len(items) < 2 or np.any(np.diff(times) < 0):
            raise ValueError("Need aligned chronological sequences with at least two events")
        items, times = items[-max_length-1:], times[-max_length-1:]
        length = len(items) - 1
        x[index, :length], y[index, :length] = items[:-1], items[1:]
        valid[index, :length] = np.diff(times) > 0
    return x, y, valid


def sample_negatives(seen, lengths, num_items, rng):
    """Uniform training-catalog negatives, excluding the user's complete history."""
    out = np.zeros((len(seen), max(lengths)), dtype=np.int64)
    for row, (history, length) in enumerate(zip(seen, lengths, strict=True)):
        if len(set(history)) >= num_items - 2:
            raise ValueError("User has no unseen training-catalog item")
        values = rng.integers(2, num_items, size=length)
        for _ in range(32):
            bad = np.isin(values, history)
            if not bad.any():
                break
            values[bad] = rng.integers(2, num_items, size=int(bad.sum()))
        else:
            available = np.setdiff1d(np.arange(2, num_items), history)
            values = rng.choice(available, size=length)
        out[row, :length] = values
    return out


def bucket_batches(lengths, batch_size, rng):
    """Shuffle length buckets to reduce padding without using outcome labels."""
    order = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(order), batch_size * 20):
        pool = order[start:start + batch_size * 20]
        pool = pool[np.argsort(lengths[pool], kind="stable")]
        batches.extend(pool[i:i+batch_size] for i in range(0, len(pool), batch_size))
    rng.shuffle(batches)
    return batches
