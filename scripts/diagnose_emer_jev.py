#!/usr/bin/env python3
"""Report every existing matched setting after confirmation; never reselect."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.evaluation.comparison import comparison_objective_metrics
from run_emer_jev import dataset,load_model,predict


def diagnose(a):
    p=a.data;c=read_json(a.config);selection=read_json(p/'selection.json');d=dataset(p/'test.npz')
    if not (p/'evaluation.json').exists():raise ValueError('Primary confirmation must finish first')
    rows=[]
    count=len(c['training']['learning_rates'])*len(c['training']['prior_coefficients'])
    for trial in range(count):
        pairs=[]
        for seed in c['training']['seeds']:
            points={}
            for arm in ('emer','emer_jev'):
                scores=predict(load_model(p/f'{arm}-seed{seed}-trial{trial}.pt'),d,32)
                points[arm]=comparison_objective_metrics(scores,d.arrays['labels'].numpy(),d.arrays['mask'].numpy(),
                    c['directions'],top_k=c['evaluation']['top_k'],sample_weights=d.sample_weights)[0]
            pairs.append(dict(seed=seed,emer=points['emer'].tolist(),emer_jev=points['emer_jev'].tolist(),
                delta=(points['emer_jev']-points['emer']).tolist()))
        first=c['training']['seeds'][0]
        spec=next(r for r in selection['trials'] if r['path']==f'emer-seed{first}-trial{trial}.pt')
        delta=np.mean([r['delta'] for r in pairs],0)
        rows.append(dict(trial=trial,learning_rate=spec['lr'],prior_coefficient=spec['prior_coefficient'],
            per_seed=pairs,mean_delta=delta.tolist(),positive_mean_delta=float(delta[:6].mean())))
    result=dict(status='exploratory after confirmation; all four pre-existing settings reported; not a new confirmation or selection rule',rows=rows)
    path=p/'matched_config_diagnostic.json'
    if path.exists() and read_json(path)!=result:raise ValueError('Existing diagnostic replay differs')
    atomic_write_json(path,result);print('All matched-setting diagnostics reproduced',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev.json'));a=p.parse_args()
    torch.set_num_threads(4);diagnose(a)
