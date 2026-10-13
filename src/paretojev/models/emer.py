"""Paper-aligned EMER components for the disclosed KuaiRand adaptation.

Reference: https://arxiv.org/html/2508.05093v2 (CC BY 4.0).
The public dataset lacks original unexposed request candidates and p_watchtime.
This is an independent implementation, not the authors' industrial system.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from paretojev.data.schema import SlateBatch


@dataclass(frozen=True)
class EMERConfig:
    feature_dim: int = 65
    objectives: int = 7
    hidden_dim: int = 48
    layers: int = 2
    heads: int = 4
    ff_dim: int = 96
    dropout: float = 0.05
    use_jev: bool = False


def utility(values: torch.Tensor) -> torch.Tensor:
    """Six positive goals followed by explicit negative feedback."""
    return torch.cat((values[..., :-1], 1 - values[..., -1:]), -1)


def normalized_ranks(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Descending average ranks / valid count; ties preserve equivariance."""
    valid_pairs = mask[:, :, None, None] & mask[:, None, :, None]
    better = (values[:, None, :, :] > values[:, :, None, :]) & valid_pairs
    equal = (values[:, None, :, :] == values[:, :, None, :]) & valid_pairs
    ranks = 1 + better.sum(2) + (equal.sum(2) - 1).clamp_min(0) * 0.5
    return (ranks / mask.sum(1).clamp_min(1)[:, None, None]).masked_fill(~mask[..., None], 0)


class EMERReranker(nn.Module):
    """Shared score from candidate comparisons and normalized signal ranks.

    Both arms have identical dimensions and initialization. The baseline zeroes
    the entire Jev block before ranks or any other computation. Neither arm
    consumes feedback labels or runtime preference vectors at inference.
    """
    def __init__(self, config: EMERConfig):
        super().__init__()
        if config.objectives != 7:
            raise ValueError('This registered adaptation requires seven ordered goals')
        self.config = config
        width = config.feature_dim + 6 * config.objectives + 2
        self.projection = nn.Sequential(nn.Linear(width, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.SiLU())
        layer = nn.TransformerEncoderLayer(config.hidden_dim, config.heads,
            config.ff_dim, config.dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.score = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: SlateBatch):
        batch.validate()
        if not batch.mask.any(1).all():
            raise ValueError('Empty slate')
        behavior = batch.behavior_scores
        block = torch.cat((batch.jev_scores, normalized_ranks(utility(batch.jev_scores), batch.mask),
            batch.jev_entropy, batch.jev_confidence), -1) if self.config.use_jev else behavior.new_zeros(
                (*behavior.shape[:2], 4 * self.config.objectives))
        features = torch.cat((batch.candidate_features, behavior,
            normalized_ranks(utility(behavior), batch.mask), block,
            torch.log1p(batch.popularity.clamp_min(0)),
            torch.log1p(batch.history_length.clamp_min(0))[:, None, :].expand(-1, batch.slate_size, -1)), -1)
        features = features.masked_fill(~batch.mask[..., None], 0)
        hidden = self.encoder(self.projection(features), src_key_padding_mask=~batch.mask)
        scores = self.score(hidden).squeeze(-1)
        return {'score_mean': scores.masked_fill(~batch.mask, torch.finfo(scores.dtype).min)}


def pairwise_losses(scores, targets, mask):
    """Per-slate, per-goal mean logistic losses; equal targets have no ordering."""
    scores = scores.masked_fill(~mask, 0)
    difference = scores[:, :, None] - scores[:, None, :]
    pairs = (targets[:, :, None, :] > targets[:, None, :, :]) & (
        mask[:, :, None, None] & mask[:, None, :, None])
    losses = F.softplus(-difference)[..., None].expand_as(pairs).masked_fill(~pairs, 0)
    return losses.sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)


def prior_mean_at_k(scores, prior, mask, k=3):
    order = scores.masked_fill(~mask, -torch.inf).argsort(dim=1, descending=True, stable=True)[:, :k]
    selected = torch.gather(prior, 1, order[..., None].expand(-1, -1, prior.shape[-1]))
    valid = mask.gather(1, order)
    return (selected * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]


def advantage_weights(current_scores, previous_scores, prior, mask, k=3):
    """Equation 9: previous / current prior Mean@K, independently per slate.

    Frozen behavior priors only; Jev and labels never change this evaluator.
    Epoch snapshots replace the unavailable online serving-model snapshot.
    """
    with torch.no_grad():
        old = prior_mean_at_k(previous_scores, prior, mask, k)
        new = prior_mean_at_k(current_scores, prior, mask, k)
        return (old + 1e-6) / (new + 1e-6)


def emer_loss(scores, batch, weights, prior_coefficient=1.0, row_weights=None):
    # Count six positive feedback bits; subtract the one negative bit. The
    # long-view bit is the public-data proxy for implicit satisfaction.
    satisfaction = batch.labels[..., :-1].sum(-1) - batch.labels[..., -1]
    posterior = pairwise_losses(scores, satisfaction[..., None], batch.mask).squeeze(-1)
    prior = pairwise_losses(scores, utility(batch.behavior_scores), batch.mask)
    per_row = posterior + prior_coefficient * (prior * weights).mean(-1)
    if row_weights is not None:
        per_row = per_row * row_weights
    return {'loss': per_row.mean(),
        'posterior': posterior.mean(), 'prior': prior.mean()}
