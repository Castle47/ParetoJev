#!/usr/bin/env python3
"""Replay saved correction coefficients, predictions and paired intervals."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file
from run_emer_jev import dataset,load_model,predict,paired_report
from jev_residual import rank_basis,corrected_scores


def replay(a):
    c=read_json('configs/emer_jev.json');base_selection=read_json(a.data/'selection.json');d=dataset(a.data/'test.npz')
    all_results={}
    for mode in ('bounded-residual','corrected-residual'):
        selected=read_json(a.phase/mode/'selection.json');saved=np.load(a.phase/mode/'predictions.npz');scores={}
        if mode=='bounded-residual':
            probabilities=d.arrays['jev_scores'].numpy().copy();probabilities[...,6]=1-probabilities[...,6]
        else:probabilities=np.load(a.phase/'explicit-longview-april22/scores.npz')['probabilities']
        x=rank_basis(probabilities,d.arrays['mask'].numpy())
        for seed in c['training']['seeds']:
            b=base_selection['selected'][str(seed)]['emer'];path=a.data/b['path']
            if sha256_file(path)!=b['checkpoint_sha256']:raise ValueError('Base checkpoint changed')
            base=predict(load_model(path),d,32);beta=np.array(selected[str(seed)]['beta'])
            scores[f'emer/{seed}']=base;scores[f'emer_jev/{seed}']=corrected_scores(base,x,beta,d.arrays['mask'].numpy())
        if set(saved.files)!=set(scores) or any(not np.array_equal(saved[k],v) for k,v in scores.items()):
            raise ValueError('Prediction replay mismatch: '+mode)
        result=paired_report(scores,d,c)
        if result!=read_json(a.phase/mode/'result.json')['results']:raise ValueError('Metric/interval replay mismatch: '+mode)
        all_results[mode]=dict(exact_predictions=True,exact_metrics_and_intervals=True)
    atomic_write_json(a.phase/'replay.json',dict(status='exact',models=all_results,baseline_weights_unchanged=True))
    print('Both residual variants replay exactly',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--phase',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);replay(a)
