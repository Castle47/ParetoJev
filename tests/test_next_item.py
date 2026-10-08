import numpy as np
import pytest
import torch

from paretojev.models.next_item import (
    NextItemConfig, SASRecNextItem, training_arrays, sample_negatives, bucket_batches,
)


def model():
    torch.manual_seed(0)
    return SASRecNextItem(NextItemConfig(20, max_length=6, hidden=8, heads=2, layers=1, dropout=0))


def test_causal_states_and_padding_invariance():
    m = model().eval()
    a, b = torch.tensor([[2, 3, 4, 5]]), torch.tensor([[2, 3, 8, 9]])
    torch.testing.assert_close(m.states(a)[:, :2], m.states(b)[:, :2])
    candidates = torch.tensor([[2, 6, 1]])
    torch.testing.assert_close(m(a, candidates), m(torch.tensor([[2, 3, 4, 5, 0, 0]]), candidates))
    assert m(a, candidates)[0, -1] == 0
    assert m(torch.zeros((1, 3), dtype=torch.long), candidates).eq(0).all()
    with pytest.raises(ValueError, match="right padding"):
        m.states(torch.tensor([[0, 2, 3]]))


def test_chronology_and_negative_exclusion():
    x, y, mask = training_arrays([[2, 3, 4, 5]], [[1, 2, 2, 4]], 5)
    assert mask.tolist() == [[True, False, True, False, False]]
    assert x.tolist() == [[2, 3, 4, 0, 0]]
    assert y.tolist() == [[3, 4, 5, 0, 0]]
    negative = sample_negatives([[2, 3, 4, 5]], [1000], 20, np.random.default_rng(7))
    assert not np.isin(negative, [0, 1, 2, 3, 4, 5]).any()
    with pytest.raises(ValueError, match="unseen"):
        sample_negatives([[2, 3]], [1], 4, np.random.default_rng(7))
    lengths = np.arange(1, 94) % 20 + 1
    batches = bucket_batches(lengths, 8, np.random.default_rng(7))
    assert sorted(np.concatenate(batches)) == list(range(93))


def test_model_learns_next_item_and_has_finite_gradients():
    m = model()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.02)
    x = torch.tensor([[2, 3, 0], [4, 5, 0]])
    y = torch.tensor([[3, 6, 0], [5, 7, 0]])
    n = torch.tensor([[8, 9, 0], [10, 11, 0]])
    valid = x.ne(0)
    initial = m.sampled_loss(x, y, n, valid).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = m.sampled_loss(x, y, n, valid)
        loss.backward()
        assert all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
        optimizer.step(); m.reset_reserved()
    assert m.sampled_loss(x, y, n, valid).item() < initial / 2
    assert m.items.weight[:2].eq(0).all()
