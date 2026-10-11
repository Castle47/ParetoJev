#!/usr/bin/env python3
"""Train registered 27K development pilots; never enables research target mode."""
import argparse
import fcntl
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from paretojev.data.kuairand_27k import BINARY_LABELS
from paretojev.data.kuairand_pilot import NUMERIC_NAMES
from paretojev.models.kuairand_pilot import KuaiRandPilotModel
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash, sha256_file, runtime_provenance


def metrics(y, p, w):
    result = {}
    for j, name in enumerate(BINARY_LABELS):
        truth = y[:, j]; prob = np.clip(p[:, j].astype('float64'), 1e-7, 1-1e-7)
        ece = 0.0
        for b in range(10):
            mask = (prob >= b/10) & (prob < (b+1)/10)
            if mask.any():
                ece += w[mask].sum()/w.sum() * abs(np.average(truth[mask]-prob[mask], weights=w[mask]))
        both = 0 < truth.sum() < len(truth)
        result[name] = dict(sampled_positives=int(truth.sum()),
            prevalence=float(np.average(truth, weights=w)), mean_prediction=float(np.average(prob, weights=w)),
            auc=float(roc_auc_score(truth, prob, sample_weight=w)) if both else None,
            average_precision=float(average_precision_score(truth, prob, sample_weight=w)) if both else None,
            logloss=float(np.average(-truth*np.log(prob)-(1-truth)*np.log1p(-prob), weights=w)),
            brier=float(np.average((truth-prob)**2, weights=w)), ece10=float(ece))
    return result


def batch(data, idx, profiles, device):
    fields = {}
    for key in ['history', 'item', 'user', 'tab', 'numeric']:
        dtype = torch.float32 if key == 'numeric' else torch.long
        fields[key] = torch.as_tensor(data[key][idx], dtype=dtype, device=device)
    if profiles is not None:
        fields['profile'] = torch.as_tensor(profiles[data['user'][idx]], dtype=torch.long, device=device)
    return fields


def predict(model, data, profiles, device, batch_size):
    model.eval(); parts = []
    with torch.inference_mode():
        for start in range(0, len(data['user']), batch_size):
            parts.append(model(**batch(data, slice(start, start + batch_size), profiles, device)).sigmoid().cpu().numpy())
    return np.concatenate(parts)


def run(args):
    if (args.output / 'results.json').exists() or list(args.output.glob('*.pt')):
        raise ValueError('Existing run output; choose a fresh directory')
    config = read_json(args.config); manifest = read_json(args.data / 'samples.json')
    if manifest['config_hash'] != canonical_hash(config): raise ValueError('Registration mismatch')
    if config['research_target_mode'] or config['jev_requests'] != 0: raise ValueError('Pilot only')
    train, val = {}, {}
    for name, dest in [('train', train), ('validation', val)]:
        path = args.data / f'{name}.npz'
        if sha256_file(path) != manifest['splits'][name]['sha256']: raise ValueError('Sample checksum mismatch')
        with np.load(path) as z: dest.update({k: z[k] for k in z.files})
    prior = np.average(train['labels'], axis=0, weights=train['weight']).clip(1e-7, 1-1e-7)
    tab_prior = np.tile(prior, (15, 1)).astype('float64')
    for tab in range(15):
        mask = train['tab'] == tab; w = train['weight'][mask].astype('float64')
        tab_prior[tab] = ((train['labels'][mask]*w[:, None]).sum(axis=0)+100*prior)/(w.sum()+100)
    baselines = {'global_prevalence': metrics(val['labels'], np.broadcast_to(prior, val['labels'].shape), val['weight']),
                 'tab_prevalence': metrics(val['labels'], tab_prior[val['tab']], val['weight'])}
    result = dict(scope=config['scope'], dataset='KuaiRand-27K', config=config,
        sample_manifest=manifest, baselines=baselines, models={},
        probability_status='sigmoid estimates; calibration measured, not assumed',
        input_fields={'event_only': ['history_video_ids', 'candidate_video_id', 'user_id', 'tab',
            'duration_ms', 'time_ms', 'strictly_prior_exposure_count'], 'derived_numeric_columns': NUMERIC_NAMES},
        provenance=runtime_provenance(input_paths=[args.config, args.data / 'preparation.json',
            Path(__file__), Path('src/paretojev/models/kuairand_pilot.py')]))
    atomic_write_json(args.output / 'results.json', result)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for regime in config['feature_regimes']:
        checkpoint = args.output / f'{regime}.pt'
        if checkpoint.exists(): raise ValueError('Existing checkpoint; choose a fresh run directory')
        torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
        profiles = None; cardinalities = []
        if regime == 'released_user_snapshot':
            spec = read_json(args.data / 'user_snapshot.json')
            if sha256_file(args.data / 'user_snapshot.npy') != spec['encoded_sha256']: raise ValueError('Profile checksum mismatch')
            profiles = np.load(args.data / 'user_snapshot.npy'); cardinalities = spec['cardinalities']
        model = KuaiRandPilotModel(config, profile_cardinalities=cardinalities).to(args.device)
        with torch.no_grad():
            for j, head in enumerate(model.heads):
                head.bias.fill_(float(np.log(prior[j]/(1-prior[j]))))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        records = []; best = float('inf'); start_time = time.monotonic()
        mean_weight = float(train['weight'].mean()); n = len(train['user'])
        for epoch in range(config['epochs']):
            model.train(); order = rng.permutation(n); loss_sum = 0
            for step, start in enumerate(range(0, n, config['batch_size'])):
                idx = order[start:start + config['batch_size']]
                y = torch.as_tensor(train['labels'][idx], device=args.device)
                weights = torch.as_tensor(train['weight'][idx] / mean_weight, device=args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(**batch(train, idx, profiles, args.device))
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none').mean(dim=1) * weights).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); model.sequence.reset_reserved()
                loss_sum += float(loss.detach()) * len(idx)
                if step % 200 == 0: print(f'{regime} epoch={epoch+1} step={step} loss={float(loss.detach()):.6f}', flush=True)
            pred = predict(model, val, profiles, args.device, config['batch_size'])
            score = metrics(val['labels'], pred, val['weight'])
            mean_loss = np.mean([v['logloss'] for v in score.values()])
            record = dict(epoch=epoch+1, train_logloss=loss_sum/n, validation=score, elapsed_seconds=time.monotonic()-start_time)
            records.append(record)
            if mean_loss < best:
                best = mean_loss
                torch.save(dict(state_dict=model.state_dict(), config=config, regime=regime,
                    profile_cardinalities=cardinalities, epoch=epoch+1, config_hash=canonical_hash(config),
                    sample_manifest_sha256=sha256_file(args.data / 'samples.json')), checkpoint)
                np.save(args.output / f'{regime}-validation-predictions.npy', pred)
                best_score = score; best_epoch = epoch+1
            atomic_write_json(args.output / f'{regime}-epochs.json', records)
            print(f'{regime} epoch={epoch+1} validation_mean_logloss={mean_loss:.6f}', flush=True)
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(saved['state_dict'])
        replay = predict(model, val, profiles, args.device, config['batch_size'])
        stored = np.load(args.output / f'{regime}-validation-predictions.npy')
        if not np.array_equal(replay, stored): raise ValueError('Checkpoint replay mismatch')
        gates = {name: (v['sampled_positives'] >= 20 and v['auc'] is not None and v['auc'] > .5
            and v['logloss'] < baselines['global_prevalence'][name]['logloss']
            and v['brier'] < baselines['global_prevalence'][name]['brier']) for name,v in best_score.items()}
        by_tab = {}
        for tab in np.unique(val['tab']):
            mask = val['tab'] == tab
            by_tab[str(tab)] = dict(rows=int(mask.sum()), metrics=metrics(val['labels'][mask], stored[mask], val['weight'][mask]))
        result['models'][regime] = dict(parameters=sum(p.numel() for p in model.parameters()),
            best_epoch=best_epoch, validation=best_score, by_tab=by_tab, readiness_gates=gates,
            all_gates_pass=all(gates.values()), exact_checkpoint_replay=True,
            checkpoint_sha256=sha256_file(checkpoint), epochs=records,
            temporal_evidence_eligible=regime=='event_only')
        atomic_write_json(args.output / 'results.json', result)
        del model, optimizer
    print('Training complete', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/kuairand_27k_behavior_pilot.json'))
    p.add_argument('--device', choices=['cpu'], default='cpu')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__': main()
