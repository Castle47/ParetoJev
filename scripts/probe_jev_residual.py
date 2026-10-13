#!/usr/bin/env python3
"""Development-only bounded correction; fresh confirmation is conditional."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file
from paretojev.evaluation.comparison import comparison_objective_metrics
from run_emer_jev import dataset,load_model,predict,paired_report
from jev_residual import rank_basis,corrected_scores,fit_correction


def probe(a):
    c=read_json('configs/emer_jev.json');selection=read_json(a.data/'selection.json')
    datasets={name:dataset(a.data/file) for name,file in [('train','dev-train.npz'),('validation','dev-validation.npz'),('april22_development','test.npz')]}
    basis={}
    for name,d in datasets.items():
        p=d.arrays['jev_scores'].numpy().copy();p[...,6]=1-p[...,6]
        basis[name]=rank_basis(p,d.arrays['mask'].numpy())
    registration=dict(role='exploratory reused data; no new confirmation',training='April20 only; binary long-view pairs with inverse inclusion weights',
        validation='April21 only; maximize long-view subject to click/hate absolute drop <=.005 and six-positive mean not below baseline',
        frozen_base='v1 selected no-Jev EMER checkpoints; weights never changed',
        correction='seven Jev rank coefficients; initialized zero; tanh bounded to .25 baseline within-slate standard deviation',
        ridge_grid=[.001,.01,.1],fallback='zero correction always eligible',
        new_confirmation_gate='April22 development primary lower CI >0, positive in all seeds, six-positive mean lower CI >=0, click/hate guardrails pass',
        source_inputs={name:sha256_file(a.data/file) for name,file in [('train','dev-train.npz'),('validation','dev-validation.npz'),('april22_development','test.npz')]})
    a.output.mkdir(parents=True,exist_ok=True);reg=a.output/'registration.json'
    if reg.exists() and read_json(reg)!=registration:raise ValueError('Development design changed')
    atomic_write_json(reg,registration);scores={};selections={};validation_data=datasets['validation']
    def metrics(s,d):
        return comparison_objective_metrics(s,d.arrays['labels'].numpy(),d.arrays['mask'].numpy(),c['directions'],top_k=3,sample_weights=d.sample_weights)[0]
    for seed in c['training']['seeds']:
        spec=selection['selected'][str(seed)]['emer'];path=a.data/spec['path'];before=sha256_file(path)
        if before!=spec['checkpoint_sha256']:raise ValueError('Frozen base changed')
        base=load_model(path);preds={name:predict(base,d,32) for name,d in datasets.items()}
        baseline=metrics(preds['validation'],validation_data)
        chosen=dict(beta=np.zeros(7),ridge=None,validation=baseline.tolist(),feasible=True)
        candidates=[dict(beta=[0.]*7,ridge=None,validation=baseline.tolist(),feasible=True)]
        for ridge in registration['ridge_grid']:
            train=datasets['train'];beta,fit=fit_correction(preds['train'],basis['train'],train.arrays['labels'].numpy()[...,1],
                train.arrays['mask'].numpy(),train.sample_weights,ridge)
            val=metrics(corrected_scores(preds['validation'],basis['validation'],beta,validation_data.arrays['mask'].numpy()),validation_data)
            feasible=val[0]>=baseline[0]-.005 and val[6]>=baseline[6]-.005 and val[:6].mean()>=baseline[:6].mean()
            candidates.append(dict(beta=beta.tolist(),ridge=ridge,validation=val.tolist(),feasible=bool(feasible),fit=fit))
            if feasible and val[1]>chosen['validation'][1]:chosen=dict(beta=beta,ridge=ridge,validation=val.tolist(),feasible=True)
        d=datasets['april22_development'];scores[f'emer/{seed}']=preds['april22_development']
        scores[f'emer_jev/{seed}']=corrected_scores(preds['april22_development'],basis['april22_development'],chosen['beta'],d.arrays['mask'].numpy())
        selections[str(seed)]=dict(beta=chosen['beta'].tolist(),ridge=chosen['ridge'],candidates=candidates,base_sha256=before)
        if sha256_file(path)!=before:raise ValueError('Base mutated during correction fit')
    result=paired_report(scores,datasets['april22_development'],c)
    atomic_write_json(a.output/'selection.json',selections)
    atomic_write_json(a.output/'result.json',dict(role='development only; April22 already seen',results=result,
        fresh_confirmation_allowed=result['all_effect_gates'],base_weights_unchanged=True,baseline_parameters=43345,additional_parameters=7))
    np.savez_compressed(a.output/'predictions.npz',**scores)
    print('Bounded correction development:',result['objectives']['long_view'],result['gates'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);probe(a)
