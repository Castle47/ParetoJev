from __future__ import annotations

import torch
import torch.nn.functional as F


def _directions_tensor(
    directions: torch.Tensor | list[int] | tuple[int, ...] | None,
    n_objectives: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if directions is None:
        return torch.ones(n_objectives, device=device, dtype=dtype)
    out = torch.as_tensor(directions, device=device, dtype=dtype)
    if out.shape != (n_objectives,):
        raise ValueError(f"directions must have shape {(n_objectives,)}, got {tuple(out.shape)}")
    if not torch.all((out == 1) | (out == -1)):
        raise ValueError("directions entries must be +1 or -1")
    return out


def pareto_dominance_mask(
    labels: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return [B,K,K] where (i,j) means candidate i Pareto-dominates j."""
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B,K,M]")
    b, k, m = labels.shape
    if mask.shape != (b, k):
        raise ValueError("mask must have shape [B,K]")
    d = _directions_tensor(directions, m, labels.device, labels.dtype)
    adjusted = labels * d
    lhs = adjusted[:, :, None, :]
    rhs = adjusted[:, None, :, :]
    weakly_better = (lhs >= rhs - eps).all(dim=-1)
    strictly_better = (lhs > rhs + eps).any(dim=-1)
    valid = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(k, dtype=torch.bool, device=labels.device)[None, :, :]
    return weakly_better & strictly_better & valid & ~eye


def pareto_pairwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Logistic pairwise loss on unambiguous Pareto-dominance relations."""
    dominance = pareto_dominance_mask(labels, mask, directions)
    if not dominance.any():
        return scores.sum() * 0.0
    diff = scores[:, :, None] - scores[:, None, :]
    return F.softplus(-diff[dominance]).mean()


def objective_pairwise_losses(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """One differentiable pairwise ranking loss per objective.

    `scores` may be [B,K] (one shared ranking score, EMER-style) or [B,K,M]
    (one auxiliary score per objective).
    """
    if targets.ndim != 3:
        raise ValueError("targets must have shape [B,K,M]")
    b, k, m = targets.shape
    if scores.ndim == 2:
        score_by_objective = scores[:, :, None].expand(-1, -1, m)
    elif scores.shape == targets.shape:
        score_by_objective = scores
    else:
        raise ValueError("scores must have shape [B,K] or [B,K,M]")
    d = _directions_tensor(directions, m, targets.device, targets.dtype)
    adjusted = targets * d
    valid = mask[:, :, None] & mask[:, None, :]
    losses: list[torch.Tensor] = []
    for objective in range(m):
        target_diff = adjusted[:, :, None, objective] - adjusted[:, None, :, objective]
        better = (target_diff > eps) & valid
        if better.any():
            score_diff = (
                score_by_objective[:, :, None, objective]
                - score_by_objective[:, None, :, objective]
            )
            losses.append(F.softplus(-score_diff[better]).mean())
        else:
            losses.append(score_by_objective[..., objective].sum() * 0.0)
    return torch.stack(losses)


def augmented_tchebycheff(
    objective_losses: torch.Tensor,
    weights: torch.Tensor,
    ideal: torch.Tensor | None = None,
    rho: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Augmented Tchebycheff scalarization, including non-convex front coverage.

    objective_losses and weights can be [M] or [...,M].
    """
    if objective_losses.shape[-1] != weights.shape[-1]:
        raise ValueError("objective_losses and weights must agree on the objective axis")
    weights = torch.clamp(weights, min=eps)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ideal = torch.zeros_like(objective_losses) if ideal is None else ideal
    gap = torch.clamp(objective_losses - ideal, min=0.0)
    weighted = weights * gap
    return weighted.max(dim=-1).values + rho * weighted.sum(dim=-1)


def gaussian_pareto_pairwise_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    directions: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Uncertainty-aware pairwise loss using P(score_i > score_j)."""
    dominance = pareto_dominance_mask(labels, mask, directions)
    if not dominance.any():
        return mean.sum() * 0.0
    var = logvar.exp()
    delta = mean[:, :, None] - mean[:, None, :]
    denom = torch.sqrt(var[:, :, None] + var[:, None, :] + eps)
    probability = torch.special.ndtr(delta / denom).clamp_min(eps)
    return -torch.log(probability[dominance]).mean()


def candidate_distribution_kl(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Distill candidate distributions independently for each objective."""
    if student_logits.shape != teacher_probabilities.shape:
        raise ValueError("student_logits and teacher_probabilities must have the same shape")
    if student_logits.ndim != 3:
        raise ValueError("expected [B,K,M]")
    masked_logits = student_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
    teacher = teacher_probabilities.clamp_min(0.0) * mask.unsqueeze(-1)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(eps)
    log_student = F.log_softmax(masked_logits / temperature, dim=1)
    per_item = F.kl_div(log_student, teacher, reduction="none")
    valid = mask.unsqueeze(-1).expand_as(per_item)
    return per_item[valid].mean() * (temperature**2)


def brier_loss(probabilities: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    squared = (probabilities - labels) ** 2
    if mask is None:
        return squared.mean()
    while mask.ndim < squared.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(squared)
    return squared[mask].mean()
