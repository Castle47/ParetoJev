"""Fixed 700-user confirmation: prepare, train, score, audit, evaluate, replay.

No outcome inspection or method selection is allowed until complete scoring and
raw-response audit. Reuse the development implementation without modifying it.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
from datetime import datetime, timezone
import fcntl
import io
import json
import random
import time
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_microlens_sasrec_development as base
from paretojev.config import load_yaml
from paretojev.data.microlens import MicroLensProtocol, load_pairs, load_prepared_candidates, prepare_microlens_candidates, save_microlens_protocol
from paretojev.experiments.merge import merge_decision_scores
from paretojev.experiments.tie_validation import expected_metrics, twice_average_ranks
from paretojev.jev.cache import DecisionCache
from paretojev.jev.durable_budget import DurableBudgetSession
from paretojev.jev.prompts import build_state, build_open_jev_noul_requests, select_objectives
from paretojev.jev.public_demo import PublicDemoJevClient, score_public_demo_noul, validate_public_response
from paretojev.jev.runner import run_score_jsonl, ScoreJsonlConfig
from paretojev.jev.scoring import _binary_entropy
from paretojev.metrics.ranking import paired_bootstrap_difference
from paretojev.models.next_item import SASRecNextItem, NextItemConfig, training_arrays, sample_negatives, bucket_batches
from paretojev.training.targets import evaluate_targets, target_report
from paretojev.utils.io import atomic_write_json, read_json, read_jsonl
from paretojev.utils.provenance import sha256_file, runtime_provenance, canonical_hash
from paretojev.utils.seed import seed_everything

REPORT = Path('reports/microlens-sasrec-confirmatory-2026-10-09')
DATA = Path('data/processed/microlens-sasrec-confirmatory')
ROOT = Path('artifacts/microlens-real-sasrec-confirmatory')
N_USERS = 700
TEST_SEED = 20261011
METHODS = ('itemcf', 'sasrec', 'minilm', 'jev', 'sasrec_minilm20', 'sasrec_jev20')
TARGETS = Path('configs/targets_sasrec_confirmatory.yaml')
SOURCES = base.SOURCES + [Path(__file__).relative_to(Path.cwd()),
    Path('src/paretojev/jev/durable_budget.py'), Path('src/paretojev/jev/public_demo.py'),
    Path('src/paretojev/jev/runner.py'), Path('src/paretojev/jev/prompts.py'),
    Path('src/paretojev/jev/cache.py'), Path('src/paretojev/models/next_item.py'), TARGETS]


def verify():
    reg = read_json(REPORT / 'registration.json')
    for name, digest in {**reg['input_hashes'], **reg['source_hashes'], **reg['prepared_hashes']}.items():
        if sha256_file(name) != digest:
            raise ValueError(f'Frozen experiment file changed: {name}')
    return reg


def prepare():
    if (REPORT / 'registration.json').exists():
        raise ValueError('Refuse to replace an existing registration')
    previous = base.frozen()
    training_previous = read_json(base.REPORT / 'training.json')
    if training_previous['selected']['learning_rate'] != 0.001 or training_previous['selected']['best_epoch'] != 15:
        raise ValueError('Development recipe changed')
    excluded = sorted(previous['cohorts']['development_users'] + previous['cohorts']['checkpoint_validation_users'])
    protocol = MicroLensProtocol(n_users=N_USERS, seed=TEST_SEED)
    print('Preparing frozen unseen evaluation cohort...', flush=True)
    test = prepare_microlens_candidates(base.RAW, protocol, excluded_users=excluded, require_strict_history=True)
    parts = []
    for user, group in test.groupby('user', sort=False):
        order = list(range(20)); random.Random(37 + int(user)).shuffle(order)
        group = group.iloc[order].copy(); group['position'] = range(20); parts.append(group)
    test = pd.concat(parts, ignore_index=True)
    users = [int(x) for x in test.user.unique()]
    all_excluded = set(excluded + users)
    grouped = load_pairs(base.RAW).groupby('user', sort=True).agg(list)
    training = grouped.loc[(grouped.item.str.len() >= 6) & ~grouped.index.isin(all_excluded)]
    catalog = sorted({int(i) for seq in training.item for i in seq})
    vocab = {item: i + 2 for i, item in enumerate(catalog)}
    sequences = [[vocab[int(i)] for i in seq] for seq in training.item]
    x, y, valid = training_arrays(sequences, training.timestamp.tolist(), previous['model']['max_length'])
    if not valid.any(axis=1).all():
        raise ValueError('A training user lacks strict-time targets')
    if len(users) != N_USERS or set(users) & set(excluded) or set(training.index) & all_excluded:
        raise ValueError('User isolation failed')
    for path in (DATA, ROOT, REPORT): path.mkdir(parents=True, exist_ok=True)
    seen = [sorted(set(seq)) for seq in sequences]
    offsets = np.r_[0, np.cumsum([len(seq) for seq in seen])]
    np.savez_compressed(DATA / 'train.npz', x=x, y=y, valid=valid, users=training.index.to_numpy(),
        seen=np.concatenate(seen), offsets=offsets, catalog=np.array(catalog))
    validation = load_prepared_candidates(base.DATA / 'validation/candidates.csv')
    cooc = collections.defaultdict(collections.Counter)
    for sequence in training.item:
        for source in sequence[-20:]:
            for target in sequence[-20:]:
                if source != target: cooc[int(source)][int(target)] += 1
    validation['itemcf'] = [float(sum(cooc[h][int(row.item)] for h in row.hist_items)) for row in validation.itertuples()]
    for row in test.itertuples():
        if row.itemcf != float(sum(cooc[h][int(row.item)] for h in row.hist_items)):
            raise ValueError('CF training population differs from SASRec')
    save_microlens_protocol(test, DATA / 'test')
    save_microlens_protocol(validation, DATA / 'validation')
    max_state = max_question = 0
    for row in read_jsonl(DATA / 'test/jev_input.jsonl'):
        state = build_state(row['history'], context=row['context'], include_candidates=False)
        for _, state, question, _ in build_open_jev_noul_requests(state, row['candidates'], select_objectives(['click'])):
            max_state = max(max_state, len(json.dumps(state, ensure_ascii=False, separators=(',', ':'))))
            max_question = max(max_question, len(question))
    if max_state > 6000 or max_question > 1000:
        raise ValueError('Frozen public-demo prompts exceed limits; no automatic user replacement')
    power = {}
    dev_rows = read_json(base.ROOT / 'per_user_metrics.json')
    z = NormalDist().inv_cdf(0.9875)
    for comparator in ('sasrec', 'sasrec_minilm20'):
        diffs = np.array([r['methods']['sasrec_jev20']['auc'] - r['methods'][comparator]['auc'] for r in dev_rows])
        mean, sd = float(diffs.mean()), float(diffs.std(ddof=1))
        power[comparator] = {'development_mean_difference': mean, 'development_sd': sd,
            'normal_approximate_power_at_700': NormalDist().cdf(mean * N_USERS ** .5 / sd - z),
            'note': 'Planning approximation using reused development effects; not a guarantee or joint-test power.'}
    model = dict(previous['model']); model['num_items'] = len(catalog) + 2
    source_registration = Path('reports/microlens-independent-2026-10-08/registration.json')
    config = {'schema': 'sasrec-new-user-confirmation-v1', 'registered_at': datetime.now(timezone.utc).isoformat(),
        'dataset': 'microlens', 'data_kind': 'real', 'evaluation_scope': 'sasrec_new_700_user_confirmation',
        'cohort': {'users': users, 'n_users': N_USERS, 'planned_model_requests': N_USERS * 20,
            'max_new_http_calls_lifetime': N_USERS * 20, 'minimum_request_interval_seconds': 1.0,
            'max_concurrency': 1, 'jev_input_sha256': sha256_file(DATA / 'test/jev_input.jsonl'),
            'candidate_file_sha256': sha256_file(DATA / 'test/candidates.csv')},
        'excluded_prior_users': excluded, 'training_user_count': len(training),
        'training_interactions': int(training.item.str.len().sum()), 'training_targets': int(valid.sum()),
        'training_seed': previous['seed'], 'test_seed': TEST_SEED, 'model': model,
        'training': {**previous['training'], 'learning_rates': [0.001], 'epochs': 15,
            'selection': 'fixed final epoch 15; validation is diagnostic only', 'patience': None,
            'max_epochs': 15, 'warm_start': False},
        'protocol': {'n_negatives': 19, 'history_length': 5, 'time_window_days': 3,
            'negative_sampling': 'upstream_occurrence', 'strict_history': True, 'candidate_order_seed': 37},
        'decision_service': read_json(source_registration)['decision_service'],
        'semantic': previous['semantic'], 'methods': list(METHODS), 'secondary_weight': 0.20,
        'evaluation': {'primary_comparisons': [['sasrec_jev20', 'sasrec'], ['sasrec_jev20', 'sasrec_minilm20']],
            'primary_metric': 'auc', 'primary_confidence': 0.975,
            'correction': 'two predeclared comparisons, Bonferroni two-sided 97.5% intervals',
            'secondary_confidence': 0.95, 'bootstrap_samples': 10000, 'bootstrap_seed': TEST_SEED,
            'winner_rule': 'both primary AUC lower bounds > 0; ranking metrics also reported',
            'no_interim_outcome_evaluation': True, 'no_optional_stopping_or_expansion': True,
            'original_project_targets_achieved': False}, 'sample_size_planning': power,
        'prompt_limits': {'state': max_state, 'question': max_question},
        'limitations': ['Same retrospective dataset, not a global chronological or cross-platform holdout.',
            'New evaluation users contributed to previous development model fits; current models are refit from scratch excluding them.',
            'Fixed weight and neural recipe were selected using previous development evidence, not this cohort.',
            'Sampled alternatives are not observed nonclick exposures; one next-item objective only.',
            'One seed and one model checkpoint, with no claim about training-seed variability.',
            'Title-only frozen MiniLM; its external pretraining corpus independence is unverified.',
            'Statistical intervals condition on this checkpoint, candidate sample and public dataset.',
            'A successful fixed fusion result does not establish neural ParetoJev multi-objective superiority.'],
        'input_hashes': base.hashes(base.INPUTS[:2] + [base.REPORT / 'registration.json', base.REPORT / 'training.json',
            base.REPORT / 'semantic.json', base.ROOT / 'title_embeddings.npz', base.ROOT / 'per_user_metrics.json', source_registration]),
        'source_hashes': base.hashes(SOURCES),
        'prepared_hashes': base.hashes([DATA / 'train.npz', DATA / 'test/candidates.csv',
            DATA / 'test/jev_input.jsonl', DATA / 'validation/candidates.csv']),
        'provenance': runtime_provenance()}
    atomic_write_json(REPORT / 'registration.json', config)
    print(json.dumps({'users': len(users), 'train_users': len(training), 'training_targets': int(valid.sum()),
        'max_state': max_state, 'max_question': max_question, 'power': power}, indent=2), flush=True)


def load_model():
    reg = verify(); info = read_json(REPORT / 'training.json')
    if info['registration_sha256'] != sha256_file(REPORT / 'registration.json') or sha256_file(ROOT / 'sasrec.pt') != info['checkpoint_sha256']:
        raise ValueError('Registered behavior checkpoint changed')
    checkpoint = torch.load(ROOT / 'sasrec.pt', weights_only=True, map_location='cpu')
    if checkpoint['registration_sha256'] != info['registration_sha256']:
        raise ValueError('Checkpoint belongs to another registration')
    torch.set_num_threads(reg['training']['threads']); torch.use_deterministic_algorithms(True)
    model = SASRecNextItem(NextItemConfig(**reg['model'])); model.load_state_dict(checkpoint['model'])
    return model, {int(item): i+2 for i, item in enumerate(np.load(DATA / 'train.npz')['catalog'])}


def train():
    reg = verify(); cfg = reg['training']
    if (REPORT / 'training.json').exists(): raise ValueError('Completed fixed training already exists')
    seed_everything(reg['training_seed']); torch.use_deterministic_algorithms(True); torch.set_num_threads(cfg['threads'])
    pack = np.load(DATA / 'train.npz'); x, y, valid = (pack[k] for k in ('x','y','valid'))
    lengths = (x != 0).sum(axis=1); flat, offsets = pack['seen'], pack['offsets']
    seen = [flat[a:b] for a,b in zip(offsets[:-1], offsets[1:], strict=True)]
    vocab = {int(item): i+2 for i,item in enumerate(pack['catalog'])}
    validation = load_prepared_candidates(DATA / 'validation/candidates.csv')
    vh, vc, vy = base.frame_tensors(validation, vocab)
    cf = base.metrics(np.array([g.itemcf.to_numpy() for _,g in validation.groupby('user',sort=False)]), vy)
    model = SASRecNextItem(NextItemConfig(**reg['model']))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=tuple(cfg['betas']))
    rng = np.random.default_rng(reg['training_seed']); curve = []; started = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); total = count = 0; epoch_start = time.perf_counter()
        for indices in bucket_batches(lengths, cfg['batch_size'], rng):
            width = int(lengths[indices].max())
            negatives = sample_negatives([seen[i] for i in indices], lengths[indices], reg['model']['num_items'], rng)
            tx, ty, tv = [torch.as_tensor(a[indices,:width]) for a in (x,y,valid)]
            optimizer.zero_grad(set_to_none=True); loss = model.sampled_loss(tx,ty,torch.as_tensor(negatives),tv)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); model.reset_reserved(); n=int(tv.sum()); total+=loss.item()*n; count+=n
        val_scores=base.predict(model,vh,vc); result=base.metrics(val_scores,vy)
        row={'epoch':epoch,'loss':total/count,'validation':result,'elapsed_seconds':time.perf_counter()-epoch_start}
        curve.append(row); atomic_write_json(REPORT/'training_curve.json',curve); print(json.dumps(row),flush=True)
    checkpoint={'model':model.state_dict(),'config':reg['model'],'epoch':cfg['epochs'],
        'registration_sha256':sha256_file(REPORT/'registration.json')}
    base.save_checkpoint(ROOT/'sasrec.pt',checkpoint)
    reloaded=SASRecNextItem(NextItemConfig(**reg['model']));reloaded.load_state_dict(torch.load(ROOT/'sasrec.pt',weights_only=True)['model'])
    if not np.array_equal(val_scores,base.predict(reloaded,vh,vc)): raise ValueError('Checkpoint replay changed validation scores')
    np.save(ROOT/'validation_predictions.npy',val_scores)
    atomic_write_json(REPORT/'training.json',{'checkpoint_sha256':sha256_file(ROOT/'sasrec.pt'),
        'registration_sha256':checkpoint['registration_sha256'],'fixed_epoch':cfg['epochs'],
        'validation':result,'validation_cf':cf,'exact_reload':True,'test_metrics_inspected':False,
        'elapsed_seconds':time.perf_counter()-started,'provenance':runtime_provenance()})
    # Score label-free methods now and freeze their outputs before any Jev call.
    test=load_prepared_candidates(DATA/'test/candidates.csv')
    groups=list(test.groupby('user',sort=False))
    histories=np.array([[vocab.get(int(i),1) for i in g.hist_items.iloc[0]] for _,g in groups])
    candidates=np.array([[vocab.get(int(i),1) for i in g.item] for _,g in groups])
    sasrec=base.predict(model,histories,candidates)
    embeddings=np.load(base.ROOT/'title_embeddings.npz')
    lookup={int(i):v for i,v in zip(embeddings['items'],embeddings['embeddings'],strict=True)}
    zero=np.zeros(embeddings['embeddings'].shape[1],dtype=np.float32)
    semantic=[]
    for _,group in groups:
        h=np.stack([lookup.get(int(i),zero) for i in group.hist_items.iloc[0]])
        c=np.stack([lookup.get(int(i),zero) for i in group.item])
        semantic.append((c@h.T).max(axis=1))
    np.savez_compressed(ROOT/'frozen_predictions.npz',users=np.array([int(u) for u,_ in groups]),
        items=np.stack([g.item.to_numpy() for _,g in groups]),sasrec=sasrec,minilm=np.array(semantic),
        itemcf=np.stack([g.itemcf.to_numpy() for _,g in groups]),history_ids=histories,candidate_ids=candidates)
    atomic_write_json(REPORT/'prediction_freeze.json',{'registered_at':datetime.now(timezone.utc).isoformat(),
        'registration_sha256':checkpoint['registration_sha256'],'checkpoint_sha256':sha256_file(ROOT/'sasrec.pt'),
        'prediction_sha256':sha256_file(ROOT/'frozen_predictions.npz'),'outcome_metrics_computed':False})


def validate_frozen_predictions():
    frozen=read_json(REPORT/'prediction_freeze.json')
    if frozen['registration_sha256']!=sha256_file(REPORT/'registration.json') or frozen['checkpoint_sha256']!=sha256_file(ROOT/'sasrec.pt') or frozen['prediction_sha256']!=sha256_file(ROOT/'frozen_predictions.npz'):
        raise ValueError('Behavior/semantic prediction freeze changed')


def score():
    reg=verify(); validate_frozen_predictions()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'scoring.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        session=DurableBudgetSession(ROOT/'http_attempts.jsonl',reg['cohort']['max_new_http_calls_lifetime'])
        cache=DecisionCache(ROOT/'requests.sqlite3')
        client=PublicDemoJevClient(revision=reg['decision_service']['expected_revision'],cache=cache,
            max_http_calls=max(1,reg['cohort']['max_new_http_calls_lifetime']-session.used),minimum_interval=1.0)
        client.session.close(); client.session=session
        if client.endpoint!=reg['decision_service']['endpoint']: raise ValueError('Endpoint differs')
        reports=ROOT/'collection_runs';reports.mkdir(exist_ok=True)
        number=len(list(reports.glob('*.json')))+1
        path=reports/f'{number:04d}.json'
        identity={'registration_sha256':sha256_file(REPORT/'registration.json'),
            'prediction_freeze_sha256':sha256_file(REPORT/'prediction_freeze.json')}
        identity_path=ROOT/'collection_identity.json'
        if identity_path.exists() and read_json(identity_path)!=identity: raise ValueError('Collection identity changed')
        atomic_write_json(identity_path,identity)
        try:
            run_score_jsonl(DATA/'test/jev_input.jsonl',ROOT/'scores.jsonl',client=client,
                score_function=score_public_demo_noul,objectives=select_objectives(['click']),
                config=ScoreJsonlConfig(provider=client.provider,mode='noul',objective_names=('click',),durable_every=1),
                report_path=path,error_path=ROOT/'errors.jsonl',cache_count=cache.count)
        finally:
            client.session.close()
            if path.exists():
                result=read_json(path);result['identity']=identity;result['lifetime_attempts']=session.used
                atomic_write_json(path,result);atomic_write_json(ROOT/'scoring_report.json',result)
                print(json.dumps(result,indent=2),flush=True)


def require_complete():
    reg=verify();validate_frozen_predictions()
    rows=read_jsonl(ROOT/'scores.jsonl')
    if len(rows)!=N_USERS or {int(row['slate_id']) for row in rows}!=set(reg['cohort']['users']):
        raise ValueError('No partial-cohort evaluation: wait for all 700 users')
    return reg,rows


def audit():
    reg,scores=require_complete();cache=DecisionCache(ROOT/'requests.sqlite3')
    by_user={int(r['slate_id']):r for r in scores};total=0;model=f"autotrust/JEV-27B@{reg['decision_service']['expected_revision']}"
    for index,row in enumerate(read_jsonl(DATA/'test/jev_input.jsonl')):
        scored=by_user[int(row['slate_id'])]
        if scored['input_index']!=index or scored['candidate_ids']!=[str(c['candidate_id']) for c in row['candidates']]:
            raise ValueError('Score identity/order changed')
        if scored['resolved_model']!='autotrust/JEV-27B' or scored['resolved_model_revision']!=reg['decision_service']['expected_revision'] or scored['objective_names']!=['click']:
            raise ValueError('Score model or objective changed')
        state=build_state(row['history'],context=row['context'],include_candidates=False);observed=[]
        for kind,state,question,_ in build_open_jev_noul_requests(state,row['candidates'],select_objectives(['click'])):
            payload={'kind':kind,'state':json.dumps(state,ensure_ascii=False,separators=(',',':')),'question':question}
            raw=cache.get('autotrust-public-demo',model,payload)
            validate_public_response(raw,reg['decision_service']['expected_revision']);observed.append([raw['probabilities'][1]])
        if not np.array_equal(scored['probabilities'],observed):raise ValueError('Final scores differ from real raw responses')
        entropy=_binary_entropy(np.array(observed))
        if not np.allclose(entropy,scored['entropy'],atol=1e-12,rtol=0) or not np.allclose(1-entropy,scored['confidence'],atol=1e-12,rtol=0):
            raise ValueError('Confidence/entropy mismatch')
        total+=len(observed)
    pack=np.load(DATA/'train.npz');train_users=set(map(int,pack['users']))
    if train_users & (set(reg['cohort']['users'])|set(reg['excluded_prior_users'])): raise ValueError('Train/test user overlap')
    # Independently rebuild training sequences and check each final held-out target.
    grouped=load_pairs(base.RAW).groupby('user',sort=True).agg(list)
    training=grouped.loc[list(pack['users'])];catalog=sorted({int(i) for seq in training.item for i in seq})
    if catalog!=pack['catalog'].tolist(): raise ValueError('Non-training item vocabulary')
    vocab={item:i+2 for i,item in enumerate(catalog)}
    rebuilt=training_arrays([[vocab[int(i)] for i in seq] for seq in training.item],training.timestamp.tolist(),reg['model']['max_length'])
    if any(not np.array_equal(pack[k],a) for k,a in zip(('x','y','valid'),rebuilt,strict=True)):raise ValueError('Training raw-log rebuild differs')
    for filename in ('test','validation'):
        frame=load_prepared_candidates(DATA/filename/'candidates.csv')
        for user,g in frame.groupby('user',sort=False):
            source=grouped.loc[user];hist=source['item'][-6:-1]
            if len(g)!=20 or g.item.nunique()!=20 or g.y.sum()!=1 or int(g.loc[g.y==1,'item'].iloc[0])!=source['item'][-1]:raise ValueError('Slate target malformed')
            if not all(h==hist for h in g.hist_items) or max(source['timestamp'][-6:-1])>=source['timestamp'][-1]:raise ValueError('History includes target/future')
            if set(g.loc[g.y==0,'item']) & set(source['item']):raise ValueError('Negative is a previously seen item')
    atomic_write_json(REPORT/'audit.json',{'status':'passed','users':N_USERS,'decisions':total,
        'all_scores_equal_raw_responses':True,'score_sha256':sha256_file(ROOT/'scores.jsonl'),
        'raw_cache_sha256':sha256_file(ROOT/'requests.sqlite3'),'training_raw_rebuild_exact':True,
        'train_test_overlap':0,'previous_evaluation_overlap':0,'history_strict':True,
        'registration_sha256':sha256_file(REPORT/'registration.json')})
    print('Raw-response and split audit passed',flush=True)


def evaluate():
    reg,scores=require_complete();a=read_json(REPORT/'audit.json')
    if a['status']!='passed' or a['score_sha256']!=sha256_file(ROOT/'scores.jsonl') or a['registration_sha256']!=sha256_file(REPORT/'registration.json'):
        raise ValueError('Complete unchanged raw score audit required')
    pack=np.load(ROOT/'frozen_predictions.npz');frame=merge_decision_scores(load_prepared_candidates(DATA/'test/candidates.csv'),scores,objective='click',score_column='jev')
    values={method:[] for method in METHODS};per_user=[]
    for index,(user,g) in enumerate(frame.groupby('user',sort=False)):
        if int(pack['users'][index])!=int(user) or not np.array_equal(pack['items'][index],g.item):raise ValueError('Frozen baseline candidate identity changed')
        b,t,j=pack['sasrec'][index],pack['minilm'][index],g.jev.to_numpy()
        keys={'itemcf':pack['itemcf'][index],'sasrec':b,'minilm':t,'jev':j,
              'sasrec_minilm20':4*twice_average_ranks(b)+twice_average_ranks(t),
              'sasrec_jev20':4*twice_average_ranks(b)+twice_average_ranks(j)}
        row={'user':int(user),'methods':{}}
        for method,k in keys.items():
            m=expected_metrics(k[:,None],g.y.to_numpy());values[method].append(m);row['methods'][method]=m
        per_user.append(row)
    names=list(values['sasrec'][0]);arrays={m:{key:np.array([r[key] for r in rs]) for key in names} for m,rs in values.items()}
    metrics={m:{key:float(arr.mean()) for key,arr in rs.items()} for m,rs in arrays.items()}
    def comparison(left,right,confidence):
        result={}
        for metric in names:
            diff,low,high=paired_bootstrap_difference(arrays[left][metric],arrays[right][metric],n_bootstrap=reg['evaluation']['bootstrap_samples'],confidence=confidence,seed=TEST_SEED)
            result[metric]={'difference':diff,'low':low,'high':high,'confidence':confidence}
        return result
    primary={f'{left}_vs_{right}':{'auc':comparison(left,right,.975)['auc']} for left,right in reg['evaluation']['primary_comparisons']}
    secondary={f'{left}_vs_{right}':comparison(left,right,.95) for left,right in reg['evaluation']['primary_comparisons']+[['sasrec','itemcf']]}
    runs=[read_json(p) for p in sorted((ROOT/'collection_runs').glob('*.json'))]
    result={'evidence':{'real_data':True,'scope':reg['evaluation_scope'],'users':N_USERS,
        'previous_evaluation_overlap':0,'training_user_overlap':0,'weight_tuned_on_test':False,'complete_raw_score_audit':True},
        'metrics':metrics,'methods':{m:{'evaluated':True} for m in METHODS},'primary_comparisons':primary,
        'secondary_comparisons':secondary,'original_project_targets_achieved':False,'limitations':reg['limitations'],
        'cost':{'total_http_attempts':len(read_jsonl(ROOT/'http_attempts.jsonl')),
            'collection_wall_seconds':sum(r['elapsed_seconds'] for r in runs if r['client_stats']['http_calls']>0),
            'scored_candidate_decisions':N_USERS*20,'note':'No online serving latency or price claim.'},
        'oov':{'candidate_rate':float(np.mean(pack['candidate_ids']==1)),'history_rate':float(np.mean(pack['history_ids']==1))},
        'provenance':runtime_provenance(input_paths=[REPORT/'registration.json',REPORT/'prediction_freeze.json',ROOT/'scores.jsonl'])}
    result['target_report']=target_report(evaluate_targets(result,load_yaml(TARGETS)))
    atomic_write_json(REPORT/'evaluation.json',result);atomic_write_json(ROOT/'per_user_metrics.json',per_user)
    print(json.dumps({'metrics':metrics,'primary_comparisons':primary,'targets':result['target_report'],'cost':result['cost']},indent=2),flush=True)


def replay():
    before=read_json(REPORT/'evaluation.json');before.pop('provenance');metric_hash=sha256_file(ROOT/'per_user_metrics.json')
    model,vocab=load_model();val=load_prepared_candidates(DATA/'validation/candidates.csv');h,c,_=base.frame_tensors(val,vocab)
    if not np.array_equal(base.predict(model,h,c),np.load(ROOT/'validation_predictions.npy')):raise ValueError('Validation checkpoint replay differs')
    pack=np.load(ROOT/'frozen_predictions.npz')
    if not np.array_equal(base.predict(model,pack['history_ids'],pack['candidate_ids']),pack['sasrec']):raise ValueError('Frozen test predictions differ')
    with patch('requests.sessions.Session.request',side_effect=AssertionError('Network prohibited during replay')):
        with contextlib.redirect_stdout(io.StringIO()): evaluate()
    after=read_json(REPORT/'evaluation.json');after.pop('provenance')
    if before!=after or metric_hash!=sha256_file(ROOT/'per_user_metrics.json'):raise ValueError('Metric or cost replay changed')
    atomic_write_json(REPORT/'replay.json',{'status':'passed','checkpoint_predictions_exact':True,
        'all_metrics_intervals_and_cost_exact':True,'http_requests':0,'per_user_metrics_sha256':metric_hash})
    print('Exact offline checkpoint and evaluation replay passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','score','audit','evaluate','replay'])
    globals()[p.parse_args().stage]()
