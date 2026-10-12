#!/usr/bin/env python3
"""Fit aligned neural controls and compare with a verified frozen v2 tree."""
import argparse
import fcntl
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from run_kuairand_behavior_v2 import probabilities
from paretojev.data.kuairand_27k import BINARY_LABELS
from paretojev.data.aligned_batches import stage_epochs, sequential_batches
from paretojev.models.kuairand_aligned import AlignedBehaviorModel, fit_normalization
from paretojev.evaluation.behavior_metrics import metrics
from paretojev.training.behavior_calibration import fit_calibration, apply_calibration, cluster_logloss_difference
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash, sha256_file, runtime_provenance


def cached_array(path, digest, cache):
    """Verified local SSD file pages may be reclaimed without anonymous swap."""
    if sha256_file(path) != digest: raise ValueError(f'Changed artifact: {path}')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f'{digest}.npy'
    if not target.exists():
        temporary = target.with_suffix('.partial')
        shutil.copyfile(path, temporary)
        if sha256_file(temporary) != digest: raise ValueError('Cache copy differs')
        temporary.replace(target)
    elif sha256_file(target) != digest:
        raise ValueError('Local cache changed')
    return np.load(target, mmap_mode='r')


def load_split(root, manifest, split, cache):
    spec = manifest['splits'][split]
    result = {key: cached_array(root/f'{split}-{key}.npy', digest, cache)[:spec['rows']]
              for key, digest in spec['files'].items()}
    if result['time'].min() != spec['min_time'] or result['time'].max() != spec['max_time']:
        raise ValueError('Recorded period mismatch')
    return result


def load_sequences(root, manifest, split, cache):
    result = {key: cached_array(root/f'{split}-{key}.npy', manifest['splits'][split]['files'][key], cache)
              for key in ('history', 'item')}
    if len(result['item']) != manifest['splits'][split]['rows']: raise ValueError('Sequence row mismatch')
    return result


def batch(data, idx, regime):
    fields = dict(x=torch.from_numpy(np.array(data['x'][idx], copy=True)))
    if regime == 'sasrec_mmoe':
        for key in ('history', 'item', 'user'):
            fields[key] = torch.as_tensor(np.array(data[key][idx], copy=True), dtype=torch.long)
    return fields


def predict(model, data, size):
    model.eval(); parts = []
    with torch.inference_mode():
        for start in range(0, len(data['x']), size):
            parts.append(model(**batch(data, slice(start, start+size), model.regime)).sigmoid().numpy())
    return np.concatenate(parts)


def validate(a, config):
    manifest = read_json(a.data/'samples.json'); sequence = read_json(a.sequences/'sequences.json')
    v2 = read_json(config['v2_config'])
    if config['research_target_mode'] or config['jev_requests'] or config['columns'] != 26:
        raise ValueError('Unsupported protocol')
    if canonical_hash(v2) != manifest['config_hash']: raise ValueError('V2 protocol changed')
    if sequence['config_hash'] != canonical_hash(config) or sequence['sample_manifest_sha256'] != sha256_file(a.data/'samples.json'):
        raise ValueError('Sequence protocol changed')
    tree = read_json(a.tree/'results.json')
    payload = joblib.load(a.tree/'upload_metadata_assumed_static.joblib')
    if tree['samples'] != manifest or tree['config'] != v2 or payload['sample_manifest_sha256'] != sha256_file(a.data/'samples.json'):
        raise ValueError('Tree not fitted on the exact shared rows/protocol')
    spec = tree['models']['upload_metadata_assumed_static']
    if sha256_file(a.tree/'upload_metadata_assumed_static.joblib') != spec['model_sha256']:
        raise ValueError('Frozen tree changed')
    for mode in ('raw', 'calibrated'):
        if sha256_file(a.tree/f'upload_metadata_assumed_static-{mode}.npy') != spec[f'{mode}_sha256']:
            raise ValueError('Frozen tree predictions changed')
    return manifest, sequence, v2, tree


def fit(a, config, manifest, sequence, v2):
    if (list(a.output.glob('*.pt')) and not a.resume_frozen) or (a.output/'results.json').exists():
        raise ValueError('Existing run: use --replay or a fresh output')
    train = load_split(a.data, manifest, 'train', a.cache)
    train.update(load_sequences(a.sequences, sequence, 'train', a.cache))
    mean, scale = fit_normalization(train['x'][:, 1:], train['weight'])
    prior = np.average(train['y'], axis=0, weights=train['weight'].astype('float64')).clip(1e-7, 1-1e-7)
    weight = train['weight']/train['weight'].mean(dtype='float64')
    frozen = read_json(a.output/'frozen_models.json') if a.resume_frozen else {}
    n = len(train['x'])
    staged = stage_epochs(train, weight, a.cache/('epochs-'+canonical_hash(config)),
        seed=config['seed'], epochs=config['epochs'],
        identity=canonical_hash([manifest, sequence, config]))
    training_users = np.unique(train['user'])
    del train, weight
    start = time.monotonic()
    registration = dict(config=config,
        sample_manifest_sha256=sha256_file(a.data/'samples.json'),
        sequence_manifest_sha256=sha256_file(a.sequences/'sequences.json'),
        provenance=runtime_provenance(input_paths=[a.config, Path(__file__),
            Path('src/paretojev/models/kuairand_aligned.py'),
            Path('src/paretojev/data/kuairand_aligned.py'),
            Path('src/paretojev/data/aligned_batches.py'),
            Path('src/paretojev/training/behavior_calibration.py'),
            Path('src/paretojev/evaluation/behavior_metrics.py')], config=config))
    if a.resume_frozen:
        old = read_json(a.output/'registration.json')
        for key in ('config', 'sample_manifest_sha256', 'sequence_manifest_sha256'):
            if registration[key] != old[key]: raise ValueError('Resume protocol mismatch')
        atomic_write_json(a.output/'resume_provenance.json', registration)
    else:
        atomic_write_json(a.output/'registration.json', registration)
    for regime in config['regimes']:
        if regime in frozen:
            if sha256_file(a.output/f'{regime}.pt') != frozen[regime]['checkpoint_sha256']:
                raise ValueError('Completed arm changed before resume')
            print(f'Reusing already frozen {regime}', flush=True)
            continue
        torch.manual_seed(config['seed'])
        model = AlignedBehaviorModel(config, regime)
        with torch.no_grad():
            model.mean.copy_(torch.from_numpy(mean)); model.scale.copy_(torch.from_numpy(scale))
            if regime == 'sasrec_mmoe': model.known_users[training_users] = True
            for j, head in enumerate(model.heads): head.bias.fill_(float(np.log(prior[j]/(1-prior[j]))))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        records = []; regime_start = time.monotonic()
        for epoch in range(config['epochs']):
            model.train(); loss_sum = 0.; epoch_start = time.monotonic()
            for step, rows in enumerate(sequential_batches(staged[epoch], config['batch_size'])):
                y = torch.as_tensor(rows['y'], dtype=torch.float32)
                w = torch.from_numpy(rows['weight'])
                optimizer.zero_grad(set_to_none=True)
                logits = model(**batch(rows, slice(None), regime))
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none').mean(1)*w).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                optimizer.step(); model.reset_reserved()
                loss_sum += float(loss.detach())*len(rows['x'])
                if step % 200 == 0:
                    print(f'{regime} epoch={epoch+1} step={step} loss={float(loss.detach()):.6f} epoch_seconds={time.monotonic()-epoch_start:.1f}', flush=True)
            records.append(dict(epoch=epoch+1, training_logloss=loss_sum/n, elapsed_seconds=time.monotonic()-regime_start))
            atomic_write_json(a.output/f'{regime}-epochs.json', records)
            print(f'{regime} epoch={epoch+1} complete mean_loss={loss_sum/n:.6f}', flush=True)
        cal = load_split(a.data, manifest, 'calibration', a.cache)
        if regime == 'sasrec_mmoe': cal.update(load_sequences(a.sequences, sequence, 'calibration', a.cache))
        cal_raw = predict(model, cal, config['batch_size'])
        parameters = fit_calibration(cal['y'], cal_raw, cal['weight'], v2['calibration'])
        checkpoint = a.output/f'{regime}.pt'
        torch.save(dict(state_dict=model.state_dict(), config=config, regime=regime,
            calibration=parameters, sample_manifest_sha256=sha256_file(a.data/'samples.json'),
            sequence_manifest_sha256=sha256_file(a.sequences/'sequences.json')), checkpoint)
        np.save(a.output/f'{regime}-calibration-raw.npy', cal_raw)
        frozen[regime] = dict(checkpoint_sha256=sha256_file(checkpoint), calibration=parameters,
            parameters=sum(p.numel() for p in model.parameters()), epochs=records,
            fit_and_calibration_seconds=time.monotonic()-regime_start)
        atomic_write_json(a.output/'frozen_models.json', frozen)
        print(f'frozen {regime}', flush=True)
        del model, optimizer, cal, cal_raw
    print(f'All neural models/calibrators frozen seconds={time.monotonic()-start:.1f}; evaluation starts now', flush=True)


def evaluate(a, config, manifest, sequence, v2, tree, replay=False):
    frozen = read_json(a.output/'frozen_models.json')
    if set(frozen) != set(config['regimes']): raise ValueError('All arms must be frozen before evaluation')
    data = load_split(a.data, manifest, 'evaluation', a.cache)
    data.update(load_sequences(a.sequences, sequence, 'evaluation', a.cache))
    y, w = data['y'], data['weight']; predictions = {}; models = {}; exact = {}
    for regime in ['tree']+config['regimes']:
        if regime == 'tree':
            payload = joblib.load(a.tree/'upload_metadata_assumed_static.joblib')
            raw = probabilities(payload['models'], data['x'], payload['columns'])
            cal = apply_calibration(raw, payload['calibration'])
            for mode, p in [('raw', raw), ('calibrated', cal)]:
                if not np.array_equal(p, np.load(a.tree/f'upload_metadata_assumed_static-{mode}.npy')):
                    raise ValueError('Existing tree prediction replay failed')
        else:
            path = a.output/f'{regime}.pt'
            if sha256_file(path) != frozen[regime]['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed')
            saved = torch.load(path, weights_only=True, map_location='cpu')
            if saved['config'] != config or saved['sample_manifest_sha256'] != sha256_file(a.data/'samples.json') or saved['sequence_manifest_sha256'] != sha256_file(a.sequences/'sequences.json'):
                raise ValueError('Checkpoint protocol mismatch')
            model = AlignedBehaviorModel(config, regime); model.load_state_dict(saved['state_dict'])
            raw = predict(model, data, config['batch_size']); cal = apply_calibration(raw, saved['calibration'])
            del model
        for mode, p in [('raw', raw), ('calibrated', cal)]:
            path = a.output/f'{regime}-{mode}.npy'
            if replay:
                if not np.array_equal(p, np.load(path)): raise ValueError(f'Prediction replay mismatch: {regime}/{mode}')
            else: np.save(path, p)
        score = metrics(y, cal, w); gates = {}
        for name, value in score.items():
            baseline = tree['baselines']['tab'][name]
            gates[name] = value['sampled_positives'] >= 20 and value['auc'] is not None and value['auc'] > .5 and value['logloss'] < baseline['logloss'] and value['brier'] < baseline['brier']
        by_tab = {}
        for tab in np.unique(data['x'][:, 0]):
            mask = data['x'][:, 0] == tab
            by_tab[str(int(tab))] = dict(rows=int(mask.sum()), raw=metrics(y[mask], raw[mask], w[mask]),
                calibrated=metrics(y[mask], cal[mask], w[mask]))
        models[regime] = dict(raw=metrics(y, raw, w), calibrated=score, gates=gates, by_tab=by_tab,
            all_gates_pass=all(gates.values()),
            raw_sha256=sha256_file(a.output/f'{regime}-raw.npy'),
            calibrated_sha256=sha256_file(a.output/f'{regime}-calibrated.npy'))
        if regime == 'tree':
            old = tree['models']['upload_metadata_assumed_static']
            if any(models[regime][k] != old[k] for k in ('raw', 'calibrated', 'gates', 'by_tab')):
                raise ValueError('Tree metric parity failed')
        predictions[regime] = cal; exact[regime] = True
        print(f'{regime} gates={sum(gates.values())}/7', flush=True)
    comparisons = {}
    for first, second in [('tabular_mmoe', 'tree'), ('sasrec_mmoe', 'tree'), ('sasrec_mmoe', 'tabular_mmoe')]:
        comparisons[f'{first}_minus_{second}'] = cluster_logloss_difference(y,
            predictions[first], predictions[second], w, data['user'],
            replicates=v2['bootstrap']['replicates'], seed=config['seed'])
    result = dict(config=config, sample_manifest_sha256=sha256_file(a.data/'samples.json'),
        frozen_models=frozen, baselines=tree['baselines'], models=models,
        calibrated_logloss_comparisons=comparisons, research_targets_achieved=False,
        tree_results_sha256=sha256_file(a.tree/'results.json'))
    if replay:
        if result != read_json(a.output/'results.json'): raise ValueError('Metrics/bootstrap replay mismatch')
        atomic_write_json(a.output/'replay.json', dict(exact_predictions_metrics_and_bootstrap=True,
            models=exact, results_sha256=sha256_file(a.output/'results.json'), provenance=runtime_provenance()))
    else: atomic_write_json(a.output/'results.json', result)
    print('Replay complete' if replay else 'Aligned comparison complete', flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('data', 'sequences', 'tree', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/kuairand_27k_aligned.json'))
    p.add_argument('--replay', action='store_true')
    p.add_argument('--resume-frozen', action='store_true')
    p.add_argument('--cache', type=Path, default=Path('data/processed/kuairand-aligned-local-cache'))
    a = p.parse_args()
    config = read_json(a.config); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    with (a.output/'.lock').open('a') as lock, threadpool_limits(limits=config['threads']):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, sequence, v2, tree = validate(a, config)
        if not a.replay: fit(a, config, manifest, sequence, v2)
        evaluate(a, config, manifest, sequence, v2, tree, replay=a.replay)


if __name__ == '__main__': main()
