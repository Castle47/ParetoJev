"""Registered real next-item training and cached-score DEVELOPMENT comparisons.

Run from the repository: prepare -> train -> semantic -> evaluate.
No stage sends Jev requests. The 400 previously examined users are development
users; only the separate 1,000-user cohort selects neural checkpoints.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paretojev.data.microlens import (MicroLensProtocol, load_pairs, load_prepared_candidates,
    load_titles, prepare_microlens_candidates, save_microlens_protocol)
from paretojev.experiments.merge import merge_decision_scores
from paretojev.experiments.tie_validation import expected_metrics, twice_average_ranks, tied_candidate_mask
from paretojev.metrics.ranking import paired_bootstrap_difference
from paretojev.models.next_item import (NextItemConfig, SASRecNextItem, training_arrays,
    sample_negatives, bucket_batches)
from paretojev.utils.io import atomic_write_json, read_json, read_jsonl
from paretojev.utils.provenance import sha256_file, runtime_provenance
from paretojev.utils.seed import seed_everything

ROOT = Path('artifacts/microlens-real-sasrec-dev')
DATA = Path('data/processed/microlens-sasrec-dev')
REPORT = Path('reports/microlens-sasrec-development-2026-10-08')
RAW = Path('data/raw/microlens')
SEED = 20261010
MINILM_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
INPUTS = [
    RAW / 'MicroLens-100k_pairs.csv', RAW / 'MicroLens-100k_title_en.csv',
    Path('data/processed/microlens-real-200/candidates.csv'),
    Path('data/processed/microlens-independent-200/candidates.csv'),
    Path('artifacts/microlens-real-public-200/scores.jsonl'),
    Path('artifacts/microlens-real-independent-200/scores.jsonl'),
]
SOURCES = [Path(__file__).relative_to(Path.cwd()), Path('src/paretojev/models/next_item.py'),
    Path('src/paretojev/data/microlens.py'), Path('src/paretojev/experiments/tie_validation.py'),
    Path('src/paretojev/experiments/merge.py'), Path('src/paretojev/metrics/ranking.py')]


def hashes(paths):
    return {str(path): sha256_file(path) for path in paths}


def frozen():
    registration = read_json(REPORT / 'registration.json')
    for name, digest in {**registration['input_hashes'], **registration['source_hashes'],
                         **registration['prepared_hashes']}.items():
        if sha256_file(name) != digest:
            raise ValueError(f'Registered file changed: {name}')
    return registration


def prepare():
    if (REPORT / 'registration.json').exists():
        raise ValueError('Registration already exists; refusing to replace it')
    REPORT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    development = pd.concat([load_prepared_candidates(p) for p in INPUTS[2:4]], ignore_index=True)
    dev_users = sorted(int(u) for u in development.user.unique())
    if len(dev_users) != 400 or len(development) != 8000:
        raise ValueError('Expected exactly 400 cached development users')
    print('Preparing 1,000 separate checkpoint-selection users...', flush=True)
    protocol = MicroLensProtocol(n_users=1000, seed=SEED)
    validation = prepare_microlens_candidates(RAW, protocol, excluded_users=dev_users, require_strict_history=True)
    val_users = sorted(int(u) for u in validation.user.unique())
    excluded = set(dev_users + val_users)
    pairs = load_pairs(RAW)
    grouped = pairs.groupby('user', sort=True).agg(list)
    training = grouped.loc[(grouped.item.str.len() >= 6) & ~grouped.index.isin(excluded)]
    catalog = sorted({int(i) for seq in training.item for i in seq})
    vocab = {item: index + 2 for index, item in enumerate(catalog)}
    sequences = [[vocab[int(i)] for i in seq] for seq in training.item]
    x, y, valid = training_arrays(sequences, training.timestamp.tolist(), 50)
    if not valid.any(axis=1).all():
        raise ValueError('Found all-simultaneous training user; explicitly revise protocol before fitting')
    seen = [sorted(set(seq)) for seq in sequences]
    offsets = np.r_[0, np.cumsum([len(s) for s in seen])]
    np.savez_compressed(DATA / 'train.npz', x=x, y=y, valid=valid,
        seen=np.concatenate(seen), offsets=offsets, users=training.index.to_numpy(), catalog=np.array(catalog))
    print('Refitting development CF on the identical training-user population...', flush=True)
    cooc = collections.defaultdict(collections.Counter)
    for sequence in training.item:
        for source in sequence[-20:]:
            for target in sequence[-20:]:
                if source != target:
                    cooc[int(source)][int(target)] += 1
    development['itemcf'] = [float(sum(cooc[h][int(row.item)] for h in row.hist_items))
                             for row in development.itertuples()]
    for row in validation.itertuples():
        if row.itemcf != float(sum(cooc[h][int(row.item)] for h in row.hist_items)):
            raise ValueError('Training-user parity with validation CF failed')
    save_microlens_protocol(validation, DATA / 'validation')
    save_microlens_protocol(development, DATA / 'development')
    configuration = NextItemConfig(len(catalog) + 2)
    config = {
        'schema': 'sasrec-cached-development-v1', 'scope': 'development_only',
        'seed': SEED, 'model': asdict(configuration),
        'training': {'learning_rates': [0.001, 0.0003], 'max_epochs': 30, 'patience': 5,
            'batch_size': 256, 'optimizer': 'Adam', 'betas': [0.9, 0.98],
            'gradient_clip': 5.0, 'threads': 4, 'device': 'cpu',
            'selection': 'highest validation expected NDCG@10; first epoch/recipe wins exact ties',
            'loss': 'one uniform unseen negative per valid next-item target; unweighted positive+negative BCE',
            'timestamp_rule': 'input event time < target event time; equal-time targets masked',
            'training_users_min_events': 6},
        'cohorts': {'development_users': dev_users, 'checkpoint_validation_users': val_users,
            'training_user_count': len(training), 'training_user_hash': sha256_file(DATA / 'train.npz'),
            'user_overlap': 0, 'train_targets': int(valid.sum()),
            'equal_timestamp_targets_masked': int(((x != 0) & ~valid).sum()),
            'training_interactions': int(training.item.str.len().sum()),
            'train_catalog_size': len(catalog), 'history_length_at_evaluation': 5},
        'validation_protocol': asdict(protocol),
        'semantic': {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'revision': MINILM_REVISION,
            'pooling': 'attention-mask mean then L2 normalization', 'max_wordpieces': 256,
            'similarity': 'maximum cosine against the same five history titles',
            'fine_tuning': False, 'missing_title': 'empty or nan titles become zero embedding'},
        'evaluation': {'new_jev_requests': 0, 'fusion_secondary_weights': [0.05, 0.20],
            'tie_policy': 'exact half-credit AUC; expected ranking metrics over unresolved ties',
            'bootstrap_samples': 10000, 'confidence': 0.95, 'bootstrap_seed': SEED,
            'interval_scope': 'descriptive pointwise intervals; no confirmatory significance claim',
            'development_gates': ['sasrec_validation_ndcg10 > itemcf_validation_ndcg10',
                'sasrec_jev05_auc_gain_over_sasrec_lower95 > 0',
                'sasrec_jev05_auc_gain_over_sasrec_minilm05_lower95 > 0'],
            'original_project_targets_achieved': False},
        'limitations': ['Previously examined 400 users, so all downstream Jev comparisons are developmental.',
            'User holdout within one retrospective dataset, not a global chronological split.',
            'Training users can contain interactions later than evaluation events.',
            'Sampled alternatives are unobserved items, not verified nonclick exposures.',
            'One training seed and two learning rates; PyTorch SASRec-style variant, not paper reproduction.',
            'SASRec scores are uncalibrated next-item logits; only one observed objective.',
            'OOV/padding embeddings are zero; OOV rates and tied predictions must be reported.',
            'MiniLM is frozen external pretraining; its training corpus independence from MicroLens is unverified.'],
        'input_hashes': hashes(INPUTS), 'source_hashes': hashes(SOURCES),
        'prepared_hashes': hashes([DATA / 'train.npz', DATA / 'validation/candidates.csv', DATA / 'development/candidates.csv']),
        'provenance': runtime_provenance(input_paths=INPUTS),
    }
    atomic_write_json(REPORT / 'registration.json', config)
    print(json.dumps({k:v for k,v in config['cohorts'].items() if not k.endswith('_users')}, indent=2), flush=True)


def frame_tensors(frame, vocab):
    groups = list(frame.groupby('user', sort=False))
    history = np.array([[vocab.get(int(i), 1) for i in g.hist_items.iloc[0]] for _, g in groups])
    candidates = np.array([[vocab.get(int(i), 1) for i in g.item] for _, g in groups])
    labels = np.array([g.y.to_numpy() for _, g in groups])
    return history, candidates, labels


def predict(model, histories, candidates):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(histories), 256):
            values.append(model(torch.as_tensor(histories[start:start+256]),
                torch.as_tensor(candidates[start:start+256])).numpy())
    return np.concatenate(values)


def metrics(scores, labels):
    rows = [expected_metrics(s[:, None], y) for s, y in zip(scores, labels, strict=True)]
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.pt.part')
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
    os.replace(temporary, path)


def train():
    reg = frozen(); config = reg['training']
    if (REPORT / 'training.json').exists():
        raise ValueError('Completed training exists; use evaluate for a read-only checkpoint replay')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    pack = np.load(DATA / 'train.npz')
    x, y, valid = (pack[k] for k in ('x', 'y', 'valid'))
    lengths = (x != 0).sum(axis=1)
    seen_flat, offsets = pack['seen'], pack['offsets']
    seen = [seen_flat[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    vocab = {int(i): n+2 for n, i in enumerate(pack['catalog'])}
    frame = load_prepared_candidates(DATA / 'validation/candidates.csv')
    histories, candidates, labels = frame_tensors(frame, vocab)
    baseline = metrics(np.array([g.itemcf.to_numpy() for _, g in frame.groupby('user', sort=False)]), labels)
    print('Validation item-CF:', json.dumps(baseline), flush=True)
    recipes = []; overall = None; started = time.perf_counter()
    for recipe_id, lr in enumerate(config['learning_rates']):
        seed_everything(reg['seed']); torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(reg['seed'])
        model = SASRecNextItem(NextItemConfig(**reg['model']))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(config['betas']))
        curve = []; best = -float('inf'); stale = 0
        path = ROOT / f'sasrec_lr{lr}.pt'
        for epoch in range(1, config['max_epochs'] + 1):
            epoch_start = time.perf_counter(); model.train(); total_loss = count = 0
            for indices in bucket_batches(lengths, config['batch_size'], rng):
                width = int(lengths[indices].max())
                negatives = sample_negatives([seen[i] for i in indices], lengths[indices], reg['model']['num_items'], rng)
                tensors = [torch.as_tensor(a[indices, :width]) for a in (x, y, valid)]
                optimizer.zero_grad(set_to_none=True)
                loss = model.sampled_loss(tensors[0], tensors[1], torch.as_tensor(negatives), tensors[2])
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); model.reset_reserved()
                n = int(tensors[2].sum()); total_loss += loss.item() * n; count += n
            scores = predict(model, histories, candidates)
            result = metrics(scores, labels)
            row = {'epoch': epoch, 'loss': total_loss / count, 'validation': result,
                   'elapsed_seconds': time.perf_counter() - epoch_start}
            curve.append(row); print(json.dumps({'learning_rate': lr, **row}), flush=True)
            if result['ndcg@10'] > best:
                best = result['ndcg@10']; stale = 0; best_epoch = epoch
                save_checkpoint(path, {'model': model.state_dict(), 'config': reg['model'],
                    'registration_sha256': sha256_file(REPORT / 'registration.json'),
                    'epoch': epoch, 'learning_rate': lr, 'validation': result})
                np.save(ROOT / f'validation_lr{lr}.npy', scores)
            else:
                stale += 1
            atomic_write_json(REPORT / f'curve_lr{lr}.json', curve)
            if stale >= config['patience']:
                break
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        reloaded = predict(model, histories, candidates)
        if not np.array_equal(reloaded, np.load(ROOT / f'validation_lr{lr}.npy')):
            raise ValueError('Checkpoint reload changed validation predictions')
        entry = {'learning_rate': lr, 'best_epoch': best_epoch, 'epochs_run': len(curve),
            'checkpoint': str(path), 'checkpoint_sha256': sha256_file(path),
            'validation': checkpoint['validation'], 'exact_reload': True}
        recipes.append(entry)
        if overall is None or best > overall['validation']['ndcg@10']:
            overall = entry
    result = {'recipes': recipes, 'selected': overall, 'validation_itemcf': baseline,
        'elapsed_seconds': time.perf_counter() - started,
        'registration_sha256': sha256_file(REPORT / 'registration.json'),
        'validation_oov': oov_summary(histories, candidates, labels),
        'provenance': runtime_provenance(input_paths=[REPORT / 'registration.json'])}
    atomic_write_json(REPORT / 'training.json', result)
    print(json.dumps(result, indent=2), flush=True)


def oov_summary(history, candidates, labels):
    return {'history_rate': float(np.mean(history == 1)), 'candidate_rate': float(np.mean(candidates == 1)),
            'positive_rate': float(np.mean(candidates[labels == 1] == 1))}


def semantic():
    from transformers import AutoModel, AutoTokenizer
    reg = frozen(); started = time.perf_counter()
    torch.set_num_threads(reg['training']['threads'])
    seed_everything(reg['seed']); torch.use_deterministic_algorithms(True)
    local = ROOT / 'minilm'
    if read_json(local / 'revision.json')['revision'] != reg['semantic']['revision']:
        raise ValueError('MiniLM revision mismatch')
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(local, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
    titles = load_titles(RAW)
    items = sorted(titles)
    cleaned = [titles[i].strip() if titles[i].strip().lower() != 'nan' else '' for i in items]
    # Sorting only by text length reduces padding; original item order is restored.
    order = sorted(range(len(items)), key=lambda i: len(cleaned[i]))
    embeddings = np.zeros((len(items), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), 64):
        indices = order[start:start+64]
        tokens = tokenizer([cleaned[i] for i in indices], padding=True, truncation=True,
            max_length=reg['semantic']['max_wordpieces'], return_tensors='pt')
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
        for offset, index in enumerate(indices):
            if cleaned[index]: embeddings[index] = vectors[offset]
        if start % 2048 == 0:
            print(f'Embedded {min(start+64,len(order))}/{len(order)} titles', flush=True)
    np.savez_compressed(ROOT / 'title_embeddings.npz', items=np.array(items), embeddings=embeddings)
    result = {'model': reg['semantic'], 'titles': len(items), 'missing_titles': cleaned.count(''),
        'dimensions': embeddings.shape[1], 'elapsed_seconds': time.perf_counter() - started,
        'weights_and_tokenizer_sha256': hashes(sorted(p for p in local.iterdir() if p.is_file())),
        'embedding_sha256': sha256_file(ROOT / 'title_embeddings.npz'),
        'packages': {p: version(p) for p in ('torch', 'transformers', 'tokenizers', 'safetensors')},
        'registration_sha256': sha256_file(REPORT / 'registration.json')}
    atomic_write_json(REPORT / 'semantic.json', result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate():
    reg = frozen(); training = read_json(REPORT / 'training.json'); sem = read_json(REPORT / 'semantic.json')
    digest = sha256_file(REPORT / 'registration.json')
    if training['registration_sha256'] != digest or sem['registration_sha256'] != digest:
        raise ValueError('Model artifacts belong to another registration')
    selected = training['selected']; path = Path(selected['checkpoint'])
    if sha256_file(path) != selected['checkpoint_sha256'] or sha256_file(ROOT / 'title_embeddings.npz') != sem['embedding_sha256']:
        raise ValueError('Checkpoint or embeddings changed')
    torch.set_num_threads(reg['training']['threads']); torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['registration_sha256'] != digest:
        raise ValueError('Checkpoint registration changed')
    model = SASRecNextItem(NextItemConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    vocab = {int(i): n+2 for n, i in enumerate(np.load(DATA / 'train.npz')['catalog'])}
    validation = load_prepared_candidates(DATA / 'validation/candidates.csv')
    vh, vc, vy = frame_tensors(validation, vocab)
    if not np.array_equal(predict(model, vh, vc), np.load(ROOT / f"validation_lr{selected['learning_rate']}.npy")):
        raise ValueError('Selected checkpoint prediction replay failed')
    frame = load_prepared_candidates(DATA / 'development/candidates.csv')
    scores = read_jsonl(INPUTS[4]) + read_jsonl(INPUTS[5])
    frame = merge_decision_scores(frame, scores, objective='click', score_column='jev')
    history, candidates, labels = frame_tensors(frame, vocab)
    predictions = predict(model, history, candidates)
    emb = np.load(ROOT / 'title_embeddings.npz')
    lookup = {int(item): vector for item, vector in zip(emb['items'], emb['embeddings'], strict=True)}
    zero = np.zeros(emb['embeddings'].shape[1], dtype=np.float32)
    measurements = collections.defaultdict(list); rows = []; ties = collections.Counter()
    for i, (user, group) in enumerate(frame.groupby('user', sort=False)):
        history_vectors = np.stack([lookup.get(int(item), zero) for item in group.hist_items.iloc[0]])
        candidate_vectors = np.stack([lookup.get(int(item), zero) for item in group.item])
        semantic_scores = (candidate_vectors @ history_vectors.T).max(axis=1)
        b, j, t = predictions[i], group.jev.to_numpy(), group.title_sim.to_numpy()
        cf = group.itemcf.to_numpy()
        keys = {name: value[:, None] for name, value in {'sasrec': b, 'itemcf': cf, 'jev': j,
                 'minilm': semantic_scores, 'tfidf': t}.items()}
        for name, secondary in [('jev', j), ('minilm', semantic_scores)]:
            keys[f'cf_{name}_ties'] = np.column_stack((cf, secondary))
            for weight, multiplier in [(5, 19), (20, 4)]:
                keys[f'sasrec_{name}{weight:02}'] = (multiplier * twice_average_ranks(b) + twice_average_ranks(secondary))[:, None]
        row = {'user': int(user), 'methods': {}}
        for name, key in keys.items():
            measured = expected_metrics(key, labels[i]); measurements[name].append(measured); row['methods'][name] = measured
        rows.append(row)
        ties['sasrec_tied_candidates'] += int(tied_candidate_mask(b).sum())
        ties['itemcf_tied_candidates'] += int(tied_candidate_mask(cf).sum())
        if len(np.unique(b)) < len(b): ties['sasrec_users_with_ties'] += 1
        for index, candidate in enumerate(group.item):
            frame.loc[group.index[index], 'sasrec'] = float(b[index])
            frame.loc[group.index[index], 'minilm'] = float(semantic_scores[index])
    names = list(measurements['sasrec'][0])
    arrays = {method: {name: np.array([r[name] for r in values]) for name in names} for method, values in measurements.items()}
    summary = {method: {name: float(value.mean()) for name, value in values.items()} for method, values in arrays.items()}
    pairs = [('sasrec', 'itemcf'), ('jev', 'minilm'), ('sasrec_jev05', 'sasrec'),
        ('sasrec_minilm05', 'sasrec'), ('sasrec_jev05', 'sasrec_minilm05'),
        ('sasrec_jev20', 'sasrec'), ('sasrec_jev20', 'sasrec_minilm20'), ('cf_jev_ties', 'cf_minilm_ties')]
    comparisons = {}
    for left, right in pairs:
        comparisons[f'{left}_vs_{right}'] = {}
        for metric in ('auc', 'hr@5', 'ndcg@10'):
            diff, low, high = paired_bootstrap_difference(arrays[left][metric], arrays[right][metric],
                n_bootstrap=reg['evaluation']['bootstrap_samples'], confidence=0.95, seed=SEED)
            comparisons[f'{left}_vs_{right}'][metric] = {'difference': diff, 'low': low, 'high': high}
    gates = {
        'behavior_validation_ndcg10_improves': selected['validation']['ndcg@10'] > training['validation_itemcf']['ndcg@10'],
        'jev05_over_sasrec_auc_lower95_positive': comparisons['sasrec_jev05_vs_sasrec']['auc']['low'] > 0,
        'jev05_over_minilm05_auc_lower95_positive': comparisons['sasrec_jev05_vs_sasrec_minilm05']['auc']['low'] > 0}
    result = {'scope': 'development_only', 'users': len(rows), 'new_jev_requests': 0,
        'metrics': summary, 'descriptive_comparisons': comparisons, 'development_gates': gates,
        'original_project_targets_achieved': False, 'oov': oov_summary(history, candidates, labels),
        'ties': dict(ties), 'selected_checkpoint': selected, 'limitations': reg['limitations'],
        'provenance': runtime_provenance(input_paths=[REPORT / 'registration.json', path, ROOT / 'title_embeddings.npz'])}
    atomic_write_json(REPORT / 'evaluation.json', result)
    atomic_write_json(ROOT / 'per_user_metrics.json', rows)
    frame.drop(columns=['hist_items', 'hist_titles']).to_csv(ROOT / 'predictions.csv', index=False)
    print(json.dumps({'metrics': summary, 'gates': gates, 'oov': result['oov'], 'ties': result['ties']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'semantic', 'evaluate'])
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == '__main__':
    main()
