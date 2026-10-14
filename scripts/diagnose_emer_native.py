#!/usr/bin/env python3
"""Exhaustive existing-checkpoint diagnostics on development validation only."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from paretojev.evaluation.comparison import comparison_objective_metrics
from paretojev.experiments.comparison import _output_lock, _source_identity
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file
from run_emer_jev import dataset, load_model, predict, paired_report
from run_emer_scale import immutable_json
from audit_jev_increment import audit_split


def subgroup(data, mask):
    return SimpleNamespace(arrays={k: v[mask] for k, v in data.arrays.items()},
        users=data.users[mask], sample_weights=data.sample_weights[mask])


def diagnose(a):
    base = read_json(a.root/'training/emer_selection.json')
    full = read_json(a.root/'training-native/selection.json')
    completed = read_json(a.root/'evaluation-native/evaluation.json')
    replay = read_json(a.root/'evaluation-native/replay.json')
    for name, key in [('evaluation.json','evaluation_sha256'), ('predictions.npz','predictions_sha256')]:
        if sha256_file(a.root/'evaluation-native'/name) != replay[key]:
            raise ValueError('Original main evaluation has changed')
    if not replay['exact_predictions_and_metrics']:
        raise ValueError('Main evaluation must be replayed before diagnostics')
    c = read_json('configs/emer_jev_scale.json')
    target = read_json('configs/emer_jev_scale_target.json')
    c['evaluation'].update(target['effect_gates'])
    if full['registration']['package_sha256'] != _source_identity()['source_sha256']:
        raise ValueError('Model implementation changed')
    for file, key in [('configs/emer_jev_scale.json','dataset_config_sha256'),
                      ('scripts/run_emer_jev.py','fit_script_sha256')]:
        if sha256_file(file) != full['registration'][key]:
            raise ValueError('Frozen training protocol changed')
    common = read_json(a.root/'training/common_manifest.json')
    for name in ('train.npz','validation.npz'):
        if sha256_file(a.root/'training'/name) != common['files'][name]:
            raise ValueError('Common data changed')
    native = a.root/'jev-native-v9-fp16'
    receipt = read_json(native/'complete.json')
    if not receipt['complete'] or receipt['public_cache_mixed'] or sha256_file(native/'scores.npz') != receipt['scores_sha256']:
        raise ValueError('Native scores changed')
    d = dataset(a.root/'training/validation.npz')
    if not d.arrays['mask'].all():
        raise ValueError('Pair diagnostic requires the registered complete ten-candidate slates')
    with np.load(a.root/'training/validation.npz') as z, np.load(native/'scores.npz') as n:
        ids = z['slate_ids']
        if not np.array_equal(z['candidate_ids'], n['candidate_ids'][ids]):
            raise ValueError('Candidate identities differ')
        for key, src in [('jev_scores','probabilities'), ('jev_entropy','entropy'), ('jev_confidence','confidence')]:
            d.arrays[key] = torch.from_numpy(n[src][ids].astype('float32'))
    with np.load(a.root/'training/train.npz') as z:
        history_threshold = float(np.median(z['history_length']))
    history = d.arrays['history_length'].numpy().ravel()
    prior = d.arrays['popularity'].numpy().squeeze(-1)
    behavior = d.arrays['behavior_scores'].numpy()
    jev = d.arrays['jev_scores'].numpy()
    b_top = np.argsort(-behavior[...,1],axis=1,kind='stable')[:,:3]
    j_top = np.argsort(-jev[...,1],axis=1,kind='stable')[:,:3]
    overlap = (b_top[:,:,None] == j_top[:,None,:]).sum((1,2))
    masks = dict(history_below_train_median=history < history_threshold,
        history_at_or_above_train_median=history >= history_threshold,
        majority_zero_prior_count=(prior == 0).sum(1) >= 5,
        minority_zero_prior_count=(prior == 0).sum(1) < 5,
        longview_top3_overlap_0_or_1=overlap <= 1,
        longview_top3_overlap_2_or_3=overlap >= 2)
    inputs = {str(p):sha256_file(p) for p in [a.root/'training/emer_selection.json',
        a.root/'training-native/selection.json', a.root/'evaluation-native/evaluation.json',
        native/'scores.npz', a.root/'training/train.npz', a.root/'training/validation.npz']}
    checkpoints = {}
    for arm, selection, folder in [('emer',base,a.root/'training'), ('emer_jev',full,a.root/'training-native')]:
        if len(selection['trials']) != 12:
            raise ValueError('Expected all twelve trials per arm')
        for r in selection['trials']:
            p = folder/r['path']
            if sha256_file(p) != r['checkpoint_sha256']:
                raise ValueError('Trial checkpoint changed')
            checkpoints[str(p)] = r['checkpoint_sha256']
    protocol = dict(role='post-selection exhaustive development diagnostic, not confirmation',
        no_retraining=True, independent_test_opened=False, inputs=inputs, checkpoints=checkpoints,
        scripts={str(p):sha256_file(p) for p in [Path(__file__), Path('scripts/audit_jev_increment.py'),Path('scripts/run_emer_jev.py')]},
        history_threshold_from_unweighted_training_median=history_threshold,
        groups='Six fixed complementary masks; previous-exposure count, at least five zero-prior candidates, and long-view top3 overlap. No outcome-selected thresholds.',
        model_bootstrap=2000, signal_bootstrap=1000,
        multiplicity='Intervals are descriptive, unadjusted for model selection or multiple diagnostic comparisons.',
        matched_settings='Same learning rate and prior coefficient; each checkpoint keeps its original validation-selected epoch. No new checkpoint selection.')
    immutable_json(a.output/'registration.json',protocol)
    cache = a.output/'all_predictions.npz'
    if a.replay:
        with np.load(cache) as z: predictions={k:z[k] for k in z.files}
        old = read_json(a.output/'diagnostics.json')
        if sha256_file(cache) != old['predictions_sha256']:
            raise ValueError('Diagnostic predictions changed')
    else:
        if cache.exists():raise ValueError('Existing prediction cache; use --replay')
        predictions = {}
        for arm, selection, folder in [('emer',base,a.root/'training'), ('emer_jev',full,a.root/'training-native')]:
            for i,r in enumerate(selection['trials']):
                trial = i % 4
                model = load_model(folder/r['path'])
                if model.config.use_jev != (arm=='emer_jev'):raise ValueError('Wrong arm')
                scores = predict(model,d,32)
                points = comparison_objective_metrics(scores,d.arrays['labels'].numpy(),d.arrays['mask'].numpy(),c['directions'],top_k=3,sample_weights=d.sample_weights)[0]
                if not np.allclose(points,r['validation_objectives'],rtol=0,atol=1e-10):
                    raise ValueError('Per-trial validation does not replay')
                predictions[f'{arm}/{r["seed"]}/{trial}'] = scores
                print('prediction',arm,r['seed'],trial,flush=True)
        np.savez_compressed(cache,**predictions)
    matched = []
    for i,(lr,coef) in enumerate((lr,coef) for lr in c['training']['learning_rates'] for coef in c['training']['prior_coefficients']):
        for selection in (base,full):
            for seed in c['training']['seeds']:
                rows=[r for r in selection['trials'] if r['seed']==seed]
                if (rows[i]['lr'],rows[i]['prior_coefficient']) != (lr,coef):raise ValueError('Grid alignment differs')
        pred = {f'{arm}/{seed}':predictions[f'{arm}/{seed}/{i}'] for arm in ('emer','emer_jev') for seed in c['training']['seeds']}
        matched.append(dict(trial=i,learning_rate=lr,prior_coefficient=coef,results=paired_report(pred,d,c)))
        print('matched',i,flush=True)
    with np.load(a.root/'evaluation-native/predictions.npz') as z:selected={k:z[k] for k in z.files}
    signal = audit_split(d,[selected[f'emer/{seed}'] for seed in c['training']['seeds']],c['objectives'],1000)
    groups = {}
    for name,mask in masks.items():
        sd=subgroup(d,mask)
        pred={k:v[mask] for k,v in selected.items()}
        groups[name]=dict(slates=int(mask.sum()),users=len(np.unique(sd.users)),
            results=paired_report(pred,sd,c),signal=audit_split(sd,[pred[f'emer/{seed}'] for seed in c['training']['seeds']],c['objectives'],1000))
        print('group',name,int(mask.sum()),flush=True)
    epochs = []
    for i in range(4):
        for epoch in (1,5,10,20,30):
            rows=[]
            for seed in c['training']['seeds']:
                points={}
                for arm,sel in [('emer',base),('emer_jev',full)]:
                    r=[r for r in sel['trials'] if r['seed']==seed][i]
                    h=r['history'][epoch-1]
                    if h['epoch']!=epoch:raise ValueError('Epoch accounting differs')
                    points[arm]=h['objectives']
                rows.append(dict(seed=seed,**points))
            epochs.append(dict(trial=i,epoch=epoch,per_seed=rows,
                mean_delta=np.mean([np.array(r['emer_jev'])-r['emer'] for r in rows],0).tolist()))
    result=dict(protocol=protocol,predictions_sha256=sha256_file(cache),matched=matched,signal=signal,
        groups=groups,epoch_receipts=epochs,original_main_results=completed['results'],
        interpretation='Strict rescue/harm pairs are hindsight diagnostics, not an implementable oracle or proof of causal incremental value. No favorable config or subgroup is promoted to a new main result.')
    if a.replay:
        if old!=result:raise ValueError('Diagnostic statistics replay differs')
        atomic_write_json(a.output/'replay.json',dict(exact_cached_prediction_statistics=True,
            diagnostic_sha256=sha256_file(a.output/'diagnostics.json'),predictions_sha256=sha256_file(cache),
            model_predictions_checked_against_all_24_original_training_receipts=True,
            note='Recomputes all diagnostics from hashed cached predictions; does not claim a second model inference pass.'))
    else:atomic_write_json(a.output/'diagnostics.json',result)
    print('diagnostics complete; independent test unopened',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=4):diagnose(a)
