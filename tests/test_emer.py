from dataclasses import replace

import numpy as np
import torch

from paretojev.data.emer import encode_features, fit_feature_spec
from paretojev.data.schema import SlateBatch
from paretojev.models.emer import (EMERConfig, EMERReranker, normalized_ranks,
    advantage_weights, pairwise_losses, emer_loss, utility)


def batch():
    torch.manual_seed(1)
    p = torch.rand(2, 4, 7)
    return SlateBatch(torch.randn(2, 4, 65), p, torch.rand_like(p),
        (p > .5).float(), torch.ones(2, 7) / 7, torch.ones(2, 4, dtype=torch.bool),
        torch.rand_like(p), torch.rand_like(p), torch.ones(2, 4, 1), torch.ones(2, 1))


def test_rank_ties_and_padding():
    values = torch.tensor([[[.8], [.8], [.2], [1.]]])
    mask = torch.tensor([[True, True, True, False]])
    torch.testing.assert_close(normalized_ranks(values, mask), torch.tensor([[[.5], [.5], [1.], [0.]]]))


def test_inference_does_not_consume_labels_preferences_or_absent_jev():
    b = batch()
    m = EMERReranker(EMERConfig(dropout=0)).eval()
    expected = m(b)['score_mean']
    changed = replace(b, labels=1-b.labels, preference_weights=b.preference_weights.flip(-1),
        jev_scores=torch.full_like(b.jev_scores, float('nan')),
        jev_entropy=torch.full_like(b.jev_entropy, float('nan')),
        jev_confidence=torch.full_like(b.jev_confidence, float('nan')))
    torch.testing.assert_close(m(changed)['score_mean'], expected, rtol=0, atol=0)


def test_full_is_permutation_equivariant_and_padding_is_inert():
    b = batch(); m = EMERReranker(EMERConfig(use_jev=True, dropout=0)).eval()
    order = torch.tensor([2, 0, 3, 1])
    swapped = replace(b, **{name: getattr(b, name)[:, order] for name in
        ('candidate_features', 'behavior_scores', 'jev_scores', 'labels', 'mask',
         'jev_confidence', 'jev_entropy', 'popularity')})
    torch.testing.assert_close(m(swapped)['score_mean'], m(b)['score_mean'][:, order], atol=1e-6, rtol=1e-5)
    b.mask[:, -1] = False
    expected = m(b)['score_mean'][:, :-1]
    changed = replace(b, candidate_features=b.candidate_features.clone(), behavior_scores=b.behavior_scores.clone())
    changed.candidate_features[:, -1] = 1e20; changed.behavior_scores[:, -1] = .99
    torch.testing.assert_close(m(changed)['score_mean'][:, :-1], expected)


def test_prior_advantage_ratio_has_expected_direction_and_no_gradient():
    p = torch.tensor([[[.9], [.3]]]); mask = torch.ones(1, 2, dtype=torch.bool)
    new = torch.tensor([[0., 1.]], requires_grad=True); old = new.flip(1)
    weights = advantage_weights(new, old, p, mask, k=1)
    assert not weights.requires_grad
    torch.testing.assert_close(weights, torch.tensor([[3.]]), atol=1e-5, rtol=1e-5)


def test_loss_feedback_ordering_and_prior_targets_are_independent_of_jev():
    b = batch(); scores = torch.randn(2, 4, requires_grad=True)
    result = emer_loss(scores, b, torch.ones(2, 7))
    changed = replace(b, jev_scores=1-b.jev_scores)
    torch.testing.assert_close(result['loss'], emer_loss(scores, changed, torch.ones(2, 7))['loss'])
    result['loss'].backward(); assert torch.isfinite(scores.grad).all()
    positive = torch.tensor([[2., 1., 0.]])
    target = torch.tensor([[[2.], [1.], [0.]]]); mask = torch.ones(1, 3, dtype=torch.bool)
    assert pairwise_losses(positive, target, mask) < pairwise_losses(-positive, target, mask)
    assert utility(torch.tensor([[[0., 0., 0., 0., 0., 0., 1.]]]))[0, 0, -1] == 0


def test_constant_training_feature_remains_bounded_out_of_period():
    x = np.zeros((2, 3, 26), dtype='float32')
    spec = fit_feature_spec(x, np.ones(2))
    later = x.copy(); later[..., 1] = 1
    assert np.max(np.abs(encode_features(later, spec))) == 1
    assert spec['mean'] == [0.] * 25
