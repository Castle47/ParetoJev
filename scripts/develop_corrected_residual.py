#!/usr/bin/env python3
"""One corrected long-view signal, one bounded coefficient, frozen EMER."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file
from paretojev.evaluation.comparison import comparison_objective_metrics
from run_emer_jev import dataset,load_model,predict,paired_report
from jev_residual import rank_basis,corrected_scores,fit_correction
from probe_jev_longview import QUESTION


def main(a):
    c=read_json('configs/emer_jev.json');selection=read_json(a.data/'selection.json')
    splits={'train':('dev-train.npz','explicit-longview-train'),
        'validation':('dev-validation.npz','explicit-longview-validation'),
        'april22_development':('test.npz','explicit-longview-april22')}
    ds={};xs={};hashes={}
    for name,(file,directory) in splits.items():
        ds[name]=dataset(a.data/file);score_file=a.phase/directory/'scores.npz'
        if name=='validation':
            completion=read_json(a.phase/directory/'result.json')
            if not completion['criterion_passed']:raise ValueError('Definition probe failed')
        else:completion=read_json(a.phase/directory/'complete.json')
        if completion['scores_sha256']!=sha256_file(score_file):raise ValueError('Signal bytes changed')
        with np.load(score_file) as z,np.load(a.data/file) as original:
            expected=original['candidate_ids'] if name!='april22_development' else np.load(a.data/'raw_slates.npz')['items']
            np.testing.assert_array_equal(z['candidate_ids'],expected)
            xs[name]=rank_basis(z['probabilities'],ds[name].arrays['mask'].numpy())
        hashes[name]=sha256_file(score_file)
    registration=dict(role='development only; no fresh-test claim',question=QUESTION,
        model='frozen v1 no-Jev EMER + one corrected LongView rank coefficient',
        training='April20 only; weighted long-view pair logistic loss',
        selection='April21 long-view maximum subject to click/hate absolute decrease <=.005 and six-positive mean >= baseline; zero fallback',
        cap=.25,ridge_grid=[.001,.01,.1],additional_parameters=1,signal_sha256=hashes,
        confirmation_gate='same five effect conditions on reused April22 development; new April23 only if all pass')
    a.output.mkdir(parents=True,exist_ok=True);reg=a.output/'registration.json'
    if reg.exists() and read_json(reg)!=registration:raise ValueError('Development registration changed')
    atomic_write_json(reg,registration);scores={};chosen_models={}
    def metrics(s,d):
        return comparison_objective_metrics(s,d.arrays['labels'].numpy(),d.arrays['mask'].numpy(),c['directions'],top_k=3,sample_weights=d.sample_weights)[0]
    for seed in c['training']['seeds']:
        spec=selection['selected'][str(seed)]['emer'];path=a.data/spec['path'];before=sha256_file(path)
        if before!=spec['checkpoint_sha256']:raise ValueError('Frozen model changed')
        model=load_model(path);pred={name:predict(model,d,32) for name,d in ds.items()}
        baseline=metrics(pred['validation'],ds['validation']);beta=np.zeros(1);best=baseline[1];chosen_ridge=None
        candidates=[dict(beta=[0.],ridge=None,validation=baseline.tolist(),feasible=True)]
        for ridge in registration['ridge_grid']:
            train=ds['train'];fit_beta,receipt=fit_correction(pred['train'],xs['train'],train.arrays['labels'].numpy()[...,1],
                train.arrays['mask'].numpy(),train.sample_weights,ridge)
            val=metrics(corrected_scores(pred['validation'],xs['validation'],fit_beta,ds['validation'].arrays['mask'].numpy()),ds['validation'])
            feasible=val[0]>=baseline[0]-.005 and val[6]>=baseline[6]-.005 and val[:6].mean()>=baseline[:6].mean()
            candidates.append(dict(beta=fit_beta.tolist(),ridge=ridge,validation=val.tolist(),feasible=bool(feasible),fit=receipt))
            if feasible and val[1]>best:beta=fit_beta;best=val[1];chosen_ridge=ridge
        chosen_models[str(seed)]=dict(beta=beta.tolist(),ridge=chosen_ridge,base_path=spec['path'],base_sha256=before,candidates=candidates)
        d=ds['april22_development'];scores[f'emer/{seed}']=pred['april22_development']
        scores[f'emer_jev/{seed}']=corrected_scores(pred['april22_development'],xs['april22_development'],beta,d.arrays['mask'].numpy())
        if sha256_file(path)!=before:raise ValueError('Base mutated')
    result=paired_report(scores,ds['april22_development'],c)
    np.savez_compressed(a.output/'predictions.npz',**scores)
    atomic_write_json(a.output/'selection.json',chosen_models)
    atomic_write_json(a.output/'result.json',dict(role='reused April22 development; not fresh confirmation',
        results=result,confirmation_allowed=result['all_effect_gates'],baseline_weights_unchanged=True,
        baseline_parameters=43345,additional_parameters=1,signal=QUESTION))
    print('Corrected one-signal development:',result['objectives']['long_view'],result['gates'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--phase',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);main(a)
