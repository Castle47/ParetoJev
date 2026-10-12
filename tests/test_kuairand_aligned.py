import numpy as np
import pytest
import torch
from paretojev.data.kuairand_pilot import EVENT, hash_items
from paretojev.data.kuairand_aligned import selected_indices, sequence_rows
from paretojev.models.kuairand_aligned import AlignedBehaviorModel, fit_normalization


def test_history_excludes_all_ties_and_right_pads():
    e = np.zeros(6, dtype=EVENT)
    e['time'] = [10, 20, 20, 30, 40, 50]; e['item'] = np.arange(6)
    h, item, maximum = sequence_rows(e, np.arange(6), 3, 16)
    assert not h[0].any()
    assert h[1].tolist() == [int(hash_items([0], 16)[0]), 0, 0]
    assert np.array_equal(h[1], h[2])
    assert h[-1].tolist() == hash_items([2, 3, 4], 16).tolist()
    assert (maximum < e['time']).all()
    e['labels'] = 127
    assert np.array_equal(sequence_rows(e, np.arange(6), 3, 16)[0], h)
    changed = np.append(e, np.zeros(1, dtype=EVENT)); changed['time'][-1] = 100
    assert np.array_equal(sequence_rows(changed, np.arange(6), 3, 16)[0], h)


def test_sample_identity_and_weight_reproduce_v2():
    e = np.zeros(300, dtype=EVENT); e['time'] = np.arange(300)
    c = dict(train_end_ms=100, calibration_end_ms=200, evaluation_end_ms=300,
        targets_per_user=dict(train=64, calibration=8, evaluation=8), seed=20261012)
    for salt, split in enumerate(('train', 'calibration', 'evaluation')):
        idx, w = selected_indices(e, 7, c, salt)
        rng = np.random.default_rng(np.random.SeedSequence([c['seed'], 7, salt]))
        expected = np.sort(rng.choice(np.arange(100*salt, 100*(salt+1)), c['targets_per_user'][split], replace=False))
        assert np.array_equal(idx, expected)
        assert w*len(idx) == 100


def test_weighted_normalization_and_model_missing_values():
    mean, scale = fit_normalization(np.array([[1., np.nan], [3., np.nan]]), [3, 1])
    np.testing.assert_allclose(mean, [1.5, 0])
    np.testing.assert_allclose(scale, [np.sqrt(.75), 1])
    config = dict(columns=3, hidden_dim=8, heads=2, layers=1, experts=2,
        item_hash_buckets=16, history_length=3)
    x = torch.tensor([[0., 1., float('nan')], [1., 3., 4.]])
    for regime in ('tabular_mmoe', 'sasrec_mmoe'):
        model = AlignedBehaviorModel(config, regime)
        model.mean.copy_(torch.tensor(mean)); model.scale.copy_(torch.tensor(scale))
        fields = {} if regime == 'tabular_mmoe' else dict(history=torch.tensor([[0,0,0],[2,3,0]]),
            item=torch.tensor([2,4]), user=torch.tensor([0,1]))
        pred = model(x, **fields)
        assert pred.shape == (2,7) and torch.isfinite(pred).all()
        pred.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if regime == 'tabular_mmoe':
            with pytest.raises(ValueError, match='forbidden'): model(x, user=torch.tensor([0,1]))
        else:
            assert not model.users.weight[0].any()
            assert not model.known_users.any()


def test_exact_occurrence_preparation_with_duplicate_timestamps(tmp_path):
    import importlib.util
    from pathlib import Path
    from paretojev.utils.io import atomic_write_json, read_json
    from paretojev.utils.provenance import sha256_file, canonical_hash
    spec = importlib.util.spec_from_file_location('prepare_aligned', Path('scripts/prepare_kuairand_aligned.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    store, data, out = [tmp_path/name for name in ('store', 'data', 'out')]
    store.mkdir(); data.mkdir()
    e = np.zeros(12, dtype=EVENT)
    e['time'] = [20, 10, 20, 110, 120, 210, 220, 30, 110, 230, 130, 240]
    e['item'] = np.arange(12); e['labels'] = np.arange(12)
    e.tofile(store/'events.bin'); np.save(store/'offsets.npy', [0, 7, 12])
    atomic_write_json(store/'events.json', dict(rows=12, store_sha256=sha256_file(store/'events.bin'),
        offsets_sha256=sha256_file(store/'offsets.npy')))
    c = dict(train_end_ms=100, calibration_end_ms=200, evaluation_end_ms=300,
        targets_per_user=dict(train=64, calibration=8, evaluation=8), seed=20261012)
    atomic_write_json(tmp_path/'v2.json', c)
    manifest = dict(config_hash=canonical_hash(c), event_manifest_sha256=sha256_file(store/'events.json'), splits={})
    for salt, split in enumerate(('train', 'calibration', 'evaluation')):
        rows = {k: [] for k in ('user', 'time', 'y', 'weight')}
        for user, block in enumerate((e[:7], e[7:])):
            block = block[np.argsort(block['time'], kind='stable')]
            idx, w = selected_indices(block, user, c, salt)
            rows['user'].extend([user]*len(idx)); rows['time'].extend(block['time'][idx])
            rows['y'].extend((block['labels'][idx, None] >> np.arange(7)) & 1)
            rows['weight'].extend([w]*len(idx))
        files = {}
        for key, values in rows.items():
            path = data/f'{split}-{key}.npy'; np.save(path, np.array(values))
            files[key] = sha256_file(path)
        manifest['splits'][split] = dict(rows=len(rows['user']), files=files)
    atomic_write_json(data/'samples.json', manifest)
    module.prepare(store, data, out, dict(v2_config=str(tmp_path/'v2.json'), history_length=3, item_hash_buckets=16))
    assert read_json(out/'sequences.json')['splits']['train']['rows'] == 4
    records = np.load(out/'train-record.npy')
    assert records.tolist() == [1, 0, 2, 7]
    assert np.array_equal(e['time'][records], np.load(data/'train-time.npy'))
    with pytest.raises(ValueError, match='Existing'): module.prepare(store, data, out, {})
