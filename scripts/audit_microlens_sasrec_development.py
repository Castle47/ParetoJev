"""Audit real split invariants and replay evaluation with network requests blocked."""
from __future__ import annotations

import contextlib
import io
import json
from unittest.mock import patch

import numpy as np

import run_microlens_sasrec_development as run
from paretojev.config import load_yaml
from paretojev.data.microlens import load_pairs, load_prepared_candidates
from paretojev.models.next_item import training_arrays
from paretojev.training.targets import evaluate_targets, target_report
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import sha256_file


if __name__ == '__main__':
    registration = run.frozen()
    pack = np.load(run.DATA / 'train.npz')
    train = set(int(u) for u in pack['users'])
    val = set(registration['cohorts']['checkpoint_validation_users'])
    dev = set(registration['cohorts']['development_users'])
    assert not (train & val or train & dev or val & dev)
    pairs = load_pairs(run.RAW)
    grouped = pairs.groupby('user', sort=True).agg(list)
    eligible = set(int(u) for u in grouped[grouped.item.str.len() >= 6].index)
    assert train == eligible - val - dev
    training = grouped.loc[list(pack['users'])]
    catalog = sorted({int(i) for sequence in training.item for i in sequence})
    assert catalog == pack['catalog'].tolist()
    vocab = {item: i + 2 for i, item in enumerate(catalog)}
    seqs = [[vocab[int(i)] for i in sequence] for sequence in training.item]
    rebuilt = training_arrays(seqs, training.timestamp.tolist(), registration['model']['max_length'])
    assert all(np.array_equal(pack[name], array) for name, array in zip(('x', 'y', 'valid'), rebuilt, strict=True))
    flat, offsets = pack['seen'], pack['offsets']
    assert all(set(sequence) == set(flat[a:b]) for sequence, a, b in zip(seqs, offsets[:-1], offsets[1:], strict=True))
    time_index = {int(item): np.sort(g.timestamp.to_numpy()) for item, g in pairs.groupby('item')}
    audited_slates = 0
    for name, users in [('validation', val), ('development', dev)]:
        frame = load_prepared_candidates(run.DATA / name / 'candidates.csv')
        assert set(int(u) for u in frame.user.unique()) == users
        for user, slate in frame.groupby('user', sort=False):
            source = grouped.loc[user]
            assert len(slate) == 20 and slate.item.nunique() == 20 and slate.y.sum() == 1
            assert int(slate.loc[slate.y == 1, 'item'].iloc[0]) == source['item'][-1]
            assert all(h == source['item'][-6:-1] for h in slate.hist_items)
            assert max(source['timestamp'][-6:-1]) < source['timestamp'][-1]
            negatives = slate.loc[slate.y == 0, 'item']
            assert not set(negatives) & set(source['item'])
            delta = 3 * 86_400_000
            for item in negatives:
                times = time_index[int(item)]
                lo = np.searchsorted(times, source['timestamp'][-1] - delta, side='left')
                hi = np.searchsorted(times, source['timestamp'][-1] + delta, side='left')
                assert lo < hi
            audited_slates += 1
    before = read_json(run.REPORT / 'evaluation.json')
    prediction_hash = sha256_file(run.ROOT / 'predictions.csv')
    metric_hash = sha256_file(run.ROOT / 'per_user_metrics.json')
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden during replay')):
        with contextlib.redirect_stdout(io.StringIO()):
            run.evaluate()
    after = read_json(run.REPORT / 'evaluation.json')
    before.pop('provenance'); after.pop('provenance')
    assert before == after
    assert sha256_file(run.ROOT / 'predictions.csv') == prediction_hash
    assert sha256_file(run.ROOT / 'per_user_metrics.json') == metric_hash
    training_report = read_json(run.REPORT / 'training.json')
    payload = dict(after)
    payload['validation_ndcg10_gain'] = (training_report['selected']['validation']['ndcg@10']
                                       - training_report['validation_itemcf']['ndcg@10'])
    target_config = load_yaml('configs/targets_sasrec_development.yaml')
    targets = target_report(evaluate_targets(payload, target_config))
    atomic_write_json(run.REPORT / 'target_report.json', {
        'scope': 'development_only', 'configuration': target_config, 'report': targets,
        'original_project_targets_achieved': False})
    audit = {'status': 'passed', 'train_users': len(train), 'validation_users': len(val),
        'development_users': len(dev), 'user_overlap': 0, 'raw_slates_audited': audited_slates,
        'all_histories_strictly_precede_targets': True, 'train_arrays_rebuilt_exactly': True,
        'train_only_vocabulary_and_negative_exclusion_sets_verified': True,
        'all_negative_items_unseen_and_within_time_window': True,
        'network_blocked_during_replay': True, 'all_metrics_and_intervals_replayed_exactly': True,
        'predictions_sha256': prediction_hash, 'per_user_metrics_sha256': metric_hash,
        'registration_sha256': sha256_file(run.REPORT / 'registration.json')}
    atomic_write_json(run.REPORT / 'audit.json', audit)
    print(json.dumps({'audit': audit, 'targets': targets}, indent=2))
