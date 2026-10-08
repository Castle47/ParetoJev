from dataclasses import replace

import pytest
import torch

from paretojev.data.synthetic import SyntheticConfig, SyntheticSlateDataset, collate_slates
from paretojev.models.comparison import (
    COMPARISON_METHODS,
    EMERLikeReranker,
    build_comparison_model,
    comparison_training_loss,
    preference_ranking_loss,
)
from paretojev.models.pareto_reranker import ParetoJevConfig, ParetoJevReranker


def _config(**kwargs):
    return ParetoJevConfig(
        candidate_feature_dim=4, n_objectives=2, hidden_dim=8, n_heads=2,
        n_layers=1, ff_dim=16, dropout=0, fusion_hidden_dim=4, **kwargs,
    )


def _batch():
    dataset = SyntheticSlateDataset(
        SyntheticConfig(n_slates=2, slate_size=4, n_objectives=2, feature_dim=4, seed=19)
    )
    return collate_slates([dataset[0], dataset[1]])


def _tradeoff_batch():
    """Both requests see identical candidates; opposing preferences reverse utility."""
    batch = _batch()
    batch.mask[:, 2:] = False
    batch.labels[:] = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    batch.preference_weights[:] = torch.eye(2)
    # Candidate 0 wins the positive target; candidate 1 wins minimizing target 1.
    for field in ("candidate_features", "behavior_scores", "jev_scores", "jev_confidence", "jev_entropy", "popularity"):
        tensor = getattr(batch, field)
        tensor[1] = tensor[0]
    batch.history_length[1] = batch.history_length[0]
    return batch


@pytest.mark.parametrize("method", COMPARISON_METHODS)
def test_comparison_inference_cannot_read_labels_or_oracle_targets(method):
    torch.manual_seed(3)
    model = build_comparison_model(method, _config()).eval()
    batch = _batch()
    changed = replace(batch, labels=1 - batch.labels, target_probabilities=torch.zeros_like(batch.labels))
    with torch.no_grad():
        actual, other = model(batch), model(changed)
    assert actual["score_mean"].shape == (2, 4)
    assert actual["objective_logits"].shape == (2, 4, 2)
    torch.testing.assert_close(actual["score_mean"], other["score_mean"])
    torch.testing.assert_close(actual["objective_logits"], other["objective_logits"])
    with torch.no_grad():
        original_loss = comparison_training_loss(method, model, batch, [1, -1])["total_loss"]
        oracle_changed_loss = comparison_training_loss(
            method, model, replace(batch, target_probabilities=torch.zeros_like(batch.labels)), [1, -1]
        )["total_loss"]
    torch.testing.assert_close(original_loss, oracle_changed_loss)


@pytest.mark.parametrize("method", COMPARISON_METHODS)
def test_padding_does_not_change_valid_scores_or_losses(method):
    torch.manual_seed(9)
    model = build_comparison_model(method, _config()).eval()
    batch = _batch()
    batch.mask[:, 2:] = False
    altered = {}
    for name in ("candidate_features", "behavior_scores", "jev_scores", "jev_confidence", "jev_entropy", "popularity", "labels"):
        value = getattr(batch, name).clone()
        value[:, 2:] = 17
        altered[name] = value
    changed = replace(batch, **altered)
    with torch.no_grad():
        output = comparison_training_loss(method, model, batch, [1, -1])
        other = comparison_training_loss(method, model, changed, [1, -1])
    assert torch.isfinite(output["total_loss"])
    torch.testing.assert_close(output["score_mean"][:, :2], other["score_mean"][:, :2])
    torch.testing.assert_close(output["objective_logits"][:, :2], other["objective_logits"][:, :2])
    torch.testing.assert_close(output["total_loss"], other["total_loss"])
    assert (output["score_mean"][:, 2:] == torch.finfo(torch.float32).min).all()


@pytest.mark.parametrize("method", COMPARISON_METHODS)
def test_tied_single_candidate_has_finite_loss_and_gradients(method):
    model = build_comparison_model(method, _config())
    batch = _batch()
    batch.mask[:, 1:] = False
    batch.labels.zero_()
    output = comparison_training_loss(method, model, batch, [1, -1])
    assert output["preference_ranking"] == 0
    assert torch.isfinite(output["total_loss"])
    output["total_loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_preference_supervision_reverses_gradient_on_incomparable_candidates():
    utility = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    scores = torch.zeros((1, 2), requires_grad=True)
    favor_first = preference_ranking_loss(scores, utility, torch.tensor([[1.0, 0.0]]), mask)
    first_gradient = torch.autograd.grad(favor_first, scores)[0]
    favor_second = preference_ranking_loss(scores, utility, torch.tensor([[0.0, 1.0]]), mask)
    second_gradient = torch.autograd.grad(favor_second, scores)[0]
    assert first_gradient[0, 0] < 0 < first_gradient[0, 1]
    torch.testing.assert_close(first_gradient, -second_gradient)


@pytest.mark.parametrize("method", COMPARISON_METHODS)
def test_served_ranking_head_receives_gradient_without_pareto_dominance(method):
    torch.manual_seed(11)
    model = build_comparison_model(method, _config(predict_uncertainty=False))
    batch = _tradeoff_batch()
    output = comparison_training_loss(method, model, batch, [1, -1])
    if method == "paretojev":
        assert isinstance(model, ParetoJevReranker)
        assert output["pareto_pairwise"] == 0
    output["total_loss"].backward()
    head = model.objective_head if method == "cmr_like" else model.score_head
    assert head.weight.grad is not None
    assert head.weight.grad.abs().sum() > 0
    assert torch.isfinite(head.weight.grad).all()
    assert model.preference_film.net[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("method", COMPARISON_METHODS)
def test_training_learns_opposing_preferences_on_same_candidates(method):
    torch.manual_seed(7)
    model = build_comparison_model(method, _config(predict_uncertainty=False))
    batch = _tradeoff_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    for _ in range(60):
        optimizer.zero_grad()
        losses = comparison_training_loss(method, model, batch, [1, -1])
        losses["total_loss"].backward()
        optimizer.step()
    with torch.no_grad():
        scores = model.eval()(batch)["score_mean"]
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_emer_advantage_state_updates_only_during_training_and_is_checkpointed():
    model = build_comparison_model("emer_like", _config())
    assert isinstance(model, EMERLikeReranker)
    preferences = torch.tensor([[0.5, 0.5]])
    model.advantage_weights(preferences, torch.tensor([[0.5, 0.5]]))
    adjusted = model.advantage_weights(preferences, torch.tensor([[1.0, 0.5]]))
    assert adjusted[0, 0] > adjusted[0, 1]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.eval()
    comparison_training_loss("emer_like", model, _batch(), [1, -1])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    restored = build_comparison_model("emer_like", _config())
    restored.load_state_dict(before)
    torch.testing.assert_close(restored.advantage_ema, model.advantage_ema)
    assert restored.advantage_updates == 2
    restored.train()
    with torch.no_grad():
        comparison_training_loss("emer_like", restored, _batch(), [1, -1])
    assert restored.advantage_updates == 2


def test_comparison_rejects_method_model_mismatch_and_invalid_directions():
    with pytest.raises(ValueError, match="unknown comparison method"):
        build_comparison_model("paper_reproduction", _config())
    model = build_comparison_model("cmr_like", _config())
    with pytest.raises(ValueError, match="must agree"):
        comparison_training_loss("emer_like", model, _batch(), [1, -1])
    with pytest.raises(ValueError, match="directions"):
        comparison_training_loss("cmr_like", model, _batch(), [1, 0])
