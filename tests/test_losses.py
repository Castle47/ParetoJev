import torch

from paretojev.losses.pareto import (
    augmented_tchebycheff,
    candidate_distribution_kl,
    pareto_dominance_mask,
    pareto_pairwise_loss,
)


def test_pareto_dominance_respects_minimize_direction() -> None:
    # Item 0 is better: more positive feedback and less hate.
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    dominance = pareto_dominance_mask(labels, mask, directions=[1, -1])
    assert dominance[0, 0, 1]
    assert dominance[0, 0, 2]
    assert not dominance[0, 2, 0]


def test_pairwise_loss_prefers_correct_order() -> None:
    labels = torch.tensor([[[1.0], [0.0]]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    good = pareto_pairwise_loss(torch.tensor([[2.0, -1.0]]), labels, mask)
    bad = pareto_pairwise_loss(torch.tensor([[-1.0, 2.0]]), labels, mask)
    assert good < bad


def test_tchebycheff_and_distillation_are_finite() -> None:
    losses = torch.tensor([0.2, 0.4, 0.1])
    weights = torch.tensor([0.2, 0.7, 0.1])
    value = augmented_tchebycheff(losses, weights)
    assert torch.isfinite(value)

    logits = torch.tensor([[[2.0], [0.0], [-1.0]]])
    teacher = torch.tensor([[[0.7], [0.2], [0.1]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    kl = candidate_distribution_kl(logits, teacher, mask)
    assert torch.isfinite(kl)
    assert kl >= 0
