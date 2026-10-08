"""Controlled, independently trainable project baselines, not paper reproductions.

Both ``-like`` models see the same pre-exposure raw inputs available to ParetoJev:
candidate features, behavior/Jev probabilities, per-objective entropy/confidence,
popularity, history length and runtime preferences. Their shared set encoder has
the configured Transformer capacity, but lacks ParetoJev's learned fusion gate.
Observed labels are used only by the training loss. Optional synthetic oracle
probabilities are ignored by this comparison track, including during training.

CMR-like uses preference FiLM and objective utility heads, with a preference-
weighted sum of those logits as the ranking score. EMER-like adds an independent
shared score head and training-only objective-loss feedback: current detached
mean objective losses / a preceding EMA, clipped to [0.5, 2], reweight each
request's preference. This is a surrogate advantage evaluator, not EMER's
published self-evolving evaluator or RAS reproduction. Neither model implements
the full original paper pipeline.

Every method trains its served ranking score on pairwise differences of the
request's weighted target utility. EMER-like additionally minimizes feedback-
weighted per-objective shared-score losses; CMR-like minimizes preference-weighted
objective-head losses. Full ParetoJev uses its existing architecture, dominance,
uncertainty, Jev distillation and fusion calibration, and applies augmented
Tchebycheff to *per-request shared-score* losses. Thus incomparable candidates
also supervise the full model's score head. Fixed loss weights are published as
``COMPARISON_LOSS_WEIGHTS``; this harness does not claim parameter-count equality.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from paretojev.data.schema import SlateBatch
from paretojev.losses.pareto import (
    augmented_tchebycheff,
    brier_loss,
    candidate_distribution_kl,
    gaussian_pareto_pairwise_loss,
    pareto_pairwise_loss,
)
from paretojev.models.pareto_reranker import ParetoJevConfig, ParetoJevReranker, PreferenceFiLM


COMPARISON_METHODS = ("emer_like", "cmr_like", "paretojev")
COMPARISON_LOSS_WEIGHTS = {
    "preference_ranking": 1.0,
    "calibration": 0.25,
    "emer_advantage_ranking": 1.0,
    "cmr_objective_ranking": 1.0,
    "pareto_pairwise": 1.0,
    "tchebycheff": 1.0,
    "uncertainty": 0.25,
    "jev_distillation": 0.35,
    "fusion_calibration": 0.5,
}


class _FeatureSetEncoder(nn.Module):
    """A set encoder with raw signals and no label/target access."""

    def __init__(self, config: ParetoJevConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.candidate_feature_dim + 4 * config.n_objectives + 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.preference_film = PreferenceFiLM(config.n_objectives, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.objective_head = nn.Linear(config.hidden_dim, config.n_objectives)

    def encode(self, batch: SlateBatch) -> torch.Tensor:
        batch.validate()
        if not batch.mask.any(dim=1).all():
            raise ValueError("each comparison slate must contain a valid candidate")
        history = torch.log1p(batch.history_length.clamp_min(0))[:, None, :]
        features = torch.cat(
            [
                batch.candidate_features,
                batch.behavior_scores,
                batch.jev_scores,
                batch.jev_entropy,
                batch.jev_confidence,
                torch.log1p(batch.popularity.clamp_min(0)),
                history.expand(-1, batch.slate_size, -1),
            ],
            dim=-1,
        )
        hidden = self.preference_film(self.input_projection(features), batch.preference_weights)
        return self.set_encoder(hidden, src_key_padding_mask=~batch.mask)

    @staticmethod
    def masked_output(
        score: torch.Tensor, objective_logits: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "score_mean": score.masked_fill(~mask, torch.finfo(score.dtype).min),
            "objective_logits": objective_logits.masked_fill(~mask.unsqueeze(-1), 0),
        }


class CMRLikeReranker(_FeatureSetEncoder):
    """Project controllable baseline: preference FiLM and weighted utility logits."""

    def forward(self, batch: SlateBatch) -> dict[str, torch.Tensor]:
        objective_logits = self.objective_head(self.encode(batch))
        score = (objective_logits * batch.preference_weights[:, None, :]).sum(dim=-1)
        return self.masked_output(score, objective_logits, batch.mask)


class EMERLikeReranker(_FeatureSetEncoder):
    """Project shared-score baseline with a training-only EMA loss evaluator.

    EMA state is checkpointed, updated only during gradient-enabled training, and
    never consulted during inference. Batch t uses the reference from batches
    before t; the first batch uses unmodified preference weights. This evaluator
    requires neither evaluation labels nor externally measured validation scores.
    """

    def __init__(self, config: ParetoJevConfig) -> None:
        super().__init__(config)
        self.score_head = nn.Linear(config.hidden_dim, 1)
        self.register_buffer("advantage_ema", torch.zeros(config.n_objectives))
        self.register_buffer("advantage_updates", torch.tensor(0, dtype=torch.long))

    def forward(self, batch: SlateBatch) -> dict[str, torch.Tensor]:
        hidden = self.encode(batch)
        return self.masked_output(
            self.score_head(hidden).squeeze(-1), self.objective_head(hidden), batch.mask
        )

    def advantage_weights(
        self, preferences: torch.Tensor, per_request_losses: torch.Tensor
    ) -> torch.Tensor:
        current = per_request_losses.detach().mean(dim=0)
        if self.advantage_updates.item() == 0:
            multiplier = torch.ones_like(current)
        else:
            multiplier = (current / self.advantage_ema.clamp_min(1e-6)).clamp(0.5, 2.0)
        adjusted = preferences * multiplier
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                if self.advantage_updates.item() == 0:
                    self.advantage_ema.copy_(current)
                else:
                    self.advantage_ema.lerp_(current, 0.1)
                self.advantage_updates.add_(1)
        return adjusted


def build_comparison_model(method: str, config: ParetoJevConfig) -> nn.Module:
    """Build a fresh model; canonical method names are intentionally strict."""
    classes = {
        "emer_like": EMERLikeReranker,
        "cmr_like": CMRLikeReranker,
        "paretojev": ParetoJevReranker,
    }
    if method not in classes:
        raise ValueError(f"unknown comparison method {method!r}; expected {COMPARISON_METHODS}")
    return classes[method](config)


def _utility_targets(targets: torch.Tensor, directions: Iterable[int]) -> torch.Tensor:
    direction = torch.as_tensor(tuple(directions), device=targets.device, dtype=targets.dtype)
    if direction.shape != (targets.shape[-1],) or not torch.all((direction == 1) | (direction == -1)):
        raise ValueError("directions must contain one +1 or -1 per objective")
    return torch.where(direction > 0, targets, 1.0 - targets)


def _per_request_objective_losses(
    scores: torch.Tensor, utility_targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return [B,M], preserving request preferences instead of averaging them."""
    if scores.ndim == 2:
        scores = scores.unsqueeze(-1).expand_as(utility_targets)
    # Mask *before* subtracting: ranking output uses the minimum finite float at
    # padded positions; summing/subtracting that sentinel can otherwise overflow.
    scores = scores.masked_fill(~mask.unsqueeze(-1), 0)
    difference = scores[:, :, None, :] - scores[:, None, :, :]
    target_difference = utility_targets[:, :, None, :] - utility_targets[:, None, :, :]
    valid = mask[:, :, None, None] & mask[:, None, :, None]
    better = (target_difference > 1e-8) & valid
    penalty = F.softplus(-difference).masked_fill(~better, 0)
    return penalty.sum(dim=(1, 2)) / better.sum(dim=(1, 2)).clamp_min(1)


def preference_ranking_loss(
    scores: torch.Tensor,
    utility_targets: torch.Tensor,
    preferences: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic supervision of the served score at each preference.

    Each request is equally weighted. Strictly better weighted target utilities
    define pairs; ties and padded candidates contribute zero. This is explicit
    preference supervision, including trade-off pairs with no Pareto dominance.
    """
    scalar_utility = (utility_targets * preferences[:, None, :]).sum(dim=-1, keepdim=True)
    return _per_request_objective_losses(scores.unsqueeze(-1), scalar_utility, mask).mean()


def comparison_training_loss(
    method: str,
    model: nn.Module,
    batch: SlateBatch,
    directions: Iterable[int],
) -> dict[str, torch.Tensor]:
    """Compute fixed, inspectable losses under a common target/input protocol.

    Only observed ``labels`` train every method. Optional synthetic oracle
    ``target_probabilities`` are ignored, even when present in ``SlateBatch``.
    Objective logits always model *utility*: a negative target becomes ``1-p``.
    """
    expected = {
        "emer_like": EMERLikeReranker,
        "cmr_like": CMRLikeReranker,
        "paretojev": ParetoJevReranker,
    }
    if method not in expected or not isinstance(model, expected[method]):
        raise ValueError("comparison method and model type must agree")
    batch.validate()
    if not batch.mask.any(dim=1).all():
        raise ValueError("each comparison slate must contain a valid candidate")
    if not torch.isfinite(batch.preference_weights).all() or (batch.preference_weights < 0).any():
        raise ValueError("preferences must be finite and nonnegative")
    directions = tuple(directions)
    targets = batch.labels
    utility = _utility_targets(targets, directions)
    output = model(batch)
    scores = output["score_mean"].masked_fill(~batch.mask, 0)
    components = {
        "preference_ranking": preference_ranking_loss(scores, utility, batch.preference_weights, batch.mask),
        "calibration": brier_loss(torch.sigmoid(output["objective_logits"]), utility, batch.mask),
    }
    if method == "emer_like":
        shared_losses = _per_request_objective_losses(scores, utility, batch.mask)
        weights = model.advantage_weights(batch.preference_weights, shared_losses)
        components["emer_advantage_ranking"] = (shared_losses * weights).sum(dim=-1).mean()
    elif method == "cmr_like":
        objective_losses = _per_request_objective_losses(output["objective_logits"], utility, batch.mask)
        components["cmr_objective_ranking"] = (objective_losses * batch.preference_weights).sum(dim=-1).mean()
    else:
        shared_losses = _per_request_objective_losses(scores, utility, batch.mask)
        components["pareto_pairwise"] = pareto_pairwise_loss(scores, batch.labels, batch.mask, directions)
        components["tchebycheff"] = augmented_tchebycheff(
            shared_losses, batch.preference_weights, rho=0.05
        ).mean()
        components["uncertainty"] = (
            gaussian_pareto_pairwise_loss(scores, output["score_logvar"], batch.labels, batch.mask, directions)
            if "score_logvar" in output else scores.sum() * 0
        )
        components["jev_distillation"] = candidate_distribution_kl(
            output["objective_logits"], _utility_targets(batch.jev_scores, directions), batch.mask
        )
        components["fusion_calibration"] = brier_loss(output["fused_scores"], targets, batch.mask)
    total = sum(COMPARISON_LOSS_WEIGHTS[name] * loss for name, loss in components.items())
    return {"total_loss": total, **components, **output}
