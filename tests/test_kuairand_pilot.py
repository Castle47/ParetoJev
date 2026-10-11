import json
from pathlib import Path
import numpy as np
import pandas as pd
from paretojev.data.kuairand_pilot import EVENT, hash_items, label_diagnostics, sample_user, ingest, prepare_samples
from paretojev.utils.provenance import sha256_file


def config():
    return dict(seed=11, train_end_exclusive_ms=100, validation_end_exclusive_ms=200,
        train_targets_per_user=20, validation_targets_per_user=20, history_length=3, item_hash_buckets=16)


def test_unsorted_ties_and_empty_history():
    e = np.zeros(5, dtype=EVENT)
    e['time'] = [150, 20, 10, 20, 100]
    e['item'] = [5, 2, 1, 3, 4]
    e['labels'] = [127, 0, 1, 2, 4]
    r = sample_user(e, 0, config())
    assert r['train']['time'].tolist() == [10, 20, 20]
    assert r['validation']['time'].tolist() == [100, 150]
    assert not r['train']['history'][0].any()
    assert np.array_equal(r['train']['history'][1], r['train']['history'][2])
    assert (r['validation']['history_max_time'] < r['validation']['time']).all()
    assert r['validation']['labels'][-1].tolist() == [1] * 7
    assert r['validation']['history'][-1].tolist() == hash_items([2, 3, 4], 16).tolist()
    changed = e.copy(); changed['labels'] = 0
    assert np.array_equal(sample_user(changed, 0, config())['train']['history'], r['train']['history'])


def test_sampling_weights_and_future_invariance():
    e = np.zeros(300, dtype=EVENT); e['time'] = np.arange(300); e['item'] = np.arange(300)
    c = config(); c['train_targets_per_user'] = 7
    a = sample_user(e, 10, c); b = sample_user(e[:100], 10, c)
    assert np.array_equal(a['train']['time'], b['train']['time'])
    assert np.isclose(a['train']['weight'].sum(), 100)
    assert len(np.unique(a['train']['time'])) == 7
    assert (hash_items([0, 1, 32], 16) >= 2).all()


def test_label_diagnostics_preserves_invalid_and_raw_labels():
    f = pd.DataFrame(dict(tab=[0]*5, duration_ms=[0, -1, 1000, 1000, 1000],
        play_time_ms=[10, 0, -1, 999, 1001], long_view=[1, 0, 0, 1, 0]))
    before = f.copy(deep=True)
    d = label_diagnostics(f)['0']
    assert d['invalid_union'] == 3
    assert d['duration_zero'] == d['duration_negative'] == d['play_negative'] == 1
    assert d['provided_positive_formula_negative'] == d['provided_negative_formula_positive'] == 1
    assert d['mismatch_within_100ms_of_threshold'] == 2
    pd.testing.assert_frame_equal(f, before)


def test_disk_ingestion_shards_and_hash_guard(tmp_path):
    from paretojev.data.kuairand_27k import BINARY_LABELS
    sources = []
    for i, rows in enumerate([[(1, 110), (0, 20)], [(0, 10), (1, 30)]]):
        f = pd.DataFrame(rows, columns=['user_id', 'time_ms'])
        f['video_id'] = [i*2, i*2+1]; f['duration_ms'] = 1000
        f['play_time_ms'] = 0; f['tab'] = 0
        for label in BINARY_LABELS: f[label] = 0
        path = tmp_path / f'{i}.csv'; f.to_csv(path, index=False)
        sources.append(dict(file=path.name, scope='full_file', rows=2,
            user_event_counts=[1, 1], source_sha256=sha256_file(path)))
    audit = tmp_path / 'audit.json'; audit.write_text(json.dumps({'logs': {'early_standard': sources}}))
    out = tmp_path / 'out'
    result = ingest(tmp_path, audit, out, chunksize=1)
    assert result['rows'] == 4
    m = prepare_samples(out, config())
    assert m['splits']['train']['rows'] == 3
    assert m['splits']['validation']['rows'] == 1
    with np.load(out / 'validation.npz') as data:
        assert data['history_max_time'].tolist() == [30]


def test_model_empty_history_gradients_and_regime_guard():
    import pytest
    import torch
    from paretojev.models.kuairand_pilot import KuaiRandPilotModel
    c = dict(item_hash_buckets=16, history_length=3, hidden_dim=8, heads=2, layers=1, experts=2)
    torch.manual_seed(0)
    m = KuaiRandPilotModel(c)
    fields = dict(history=torch.tensor([[0, 0, 0], [2, 3, 0]]), item=torch.tensor([2, 4]),
        user=torch.tensor([0, 1]), tab=torch.tensor([0, 1]), numeric=torch.zeros(2, 7))
    m.set_training_users([0])
    assert m.known_users[0] and not m.known_users[1]
    assert not m.users.weight[0].any()
    logits = m(**fields)
    assert logits.shape == (2, 7) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match='forbidden'):
        m(**fields, profile=torch.zeros(2, 1, dtype=torch.long))
    enriched = KuaiRandPilotModel(c, profile_cardinalities=[3, 2])
    assert torch.isfinite(enriched(**fields, profile=torch.tensor([[0, 0], [2, 1]]))).all()


def test_profile_vocab_does_not_fit_validation_users(tmp_path, monkeypatch):
    import paretojev.data.kuairand_pilot as module
    from paretojev.data.kuairand_27k import USER_FIELDS, USER_FILE
    monkeypatch.setattr(module, 'USER_COUNT', 2)
    frame = pd.DataFrame({'user_id': [0, 1], **{name: ['train_value', 'future_only_value'] for name in USER_FIELDS}})
    frame.to_csv(tmp_path / USER_FILE, index=False)
    module.user_snapshot(tmp_path, tmp_path, np.array([0]))
    encoded = np.load(tmp_path / 'user_snapshot.npy')
    assert (encoded[0] == 1).all()
    assert (encoded[1] == 0).all()
    spec = json.loads((tmp_path / 'user_snapshot.json').read_text())
    assert all('future_only_value' not in vocab for vocab in spec['vocabulary'].values())
