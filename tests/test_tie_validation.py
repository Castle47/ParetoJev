import itertools

import numpy as np
import pandas as pd
import pytest

from paretojev.data.microlens import MicroLensProtocol, prepare_microlens_candidates
from paretojev.experiments.tie_validation import expected_metrics, policy_keys, tied_candidate_mask, twice_average_ranks


def test_expected_tie_metrics_match_all_possible_orders():
    keys = np.array([[1., 3.], [0., 1.], [1., 3.], [0., 1.]])
    labels = np.array([1, 0, 0, 0])
    actual = expected_metrics(keys, labels)
    possible_ranks = []
    for order in itertools.permutations(range(4)):
        sequence = [tuple(keys[i]) for i in order]
        if sequence == sorted(sequence, reverse=True):
            possible_ranks.append(order.index(0) + 1)
    ranks = np.array(possible_ranks)
    assert actual['auc'] == pytest.approx(np.mean((4 - ranks) / 3))
    assert actual['mrr'] == pytest.approx(np.mean(1 / ranks))
    assert actual['hr@1'] == 0.5
    assert actual['ndcg@5'] == pytest.approx(np.mean(1 / np.log2(ranks + 1)))


def test_tie_policy_preserves_strict_behavior_order_and_ignores_singleton_secondary():
    b = np.array([3, 2, 2, 1]); t = np.array([0., 0.2, 0.8, 1.]); j = np.array([0., 0.9, 0.1, 1.])
    keys = policy_keys(b, t, j)
    order = np.lexsort((-keys['jev_ties'][:, 1], -keys['jev_ties'][:, 0]))
    assert order.tolist() == [0, 1, 2, 3]
    assert tied_candidate_mask(b).tolist() == [False, True, True, False]
    altered = j.copy(); altered[[0, 3]] = [1, 0]
    order2 = np.lexsort((-policy_keys(b, t, altered)['jev_ties'][:, 1], -b))
    assert np.array_equal(order, order2)


def test_metrics_and_fixed_fusion_are_permutation_invariant():
    b = np.array([0., 0., 5., 5., 8., 8.]); t = np.arange(6.); j = t[::-1]
    labels = np.array([1, 0, 0, 0, 0, 0]); permutation = [5, 2, 0, 3, 1, 4]
    first = policy_keys(b, t, j); second = policy_keys(b[permutation], t[permutation], j[permutation])
    assert np.issubdtype(first['jev_05'].dtype, np.integer)
    for method in first:
        assert expected_metrics(first[method], labels) == expected_metrics(second[method], labels[permutation])
    assert np.array_equal(twice_average_ranks(np.zeros(4)), np.full(4, 3))


def test_all_ties_use_expectation_instead_of_positive_position():
    metrics = expected_metrics(np.zeros((20, 1)), np.eye(20, dtype=int)[0])
    assert metrics['auc'] == 0.5 and metrics['hr@5'] == 0.25
    assert metrics['hr@10'] == 0.5


def test_sampler_excludes_development_and_current_test_users_from_cf(tmp_path):
    rows = []
    for user in range(12):
        for index in range(6):
            rows.append({'user': user, 'item': (user + index) % 18, 'timestamp': user * 10 + index})
    pd.DataFrame(rows).to_csv(tmp_path/'MicroLens-100k_pairs.csv', index=False)
    pd.DataFrame([[i, f'topic video {i}'] for i in range(18)]).to_csv(tmp_path/'MicroLens-100k_title_en.csv', index=False, header=False)
    excluded = [0, 1, 2]
    frame = prepare_microlens_candidates(tmp_path, MicroLensProtocol(n_users=2, n_negatives=2, history_length=2, seed=12), excluded_users=excluded, require_strict_history=True)
    assert not set(frame.user).intersection(excluded)
    training = pd.DataFrame(rows)
    training = training[~training.user.isin(set(frame.user) | set(excluded))]
    for row in frame.itertuples():
        expected = 0
        for _, group in training.groupby('user'):
            items = group.item.tolist()[-20:]
            expected += sum(1 for h in row.hist_items for a in items for b in items if a == h and b == row.item and a != b)
        assert row.itemcf == expected
