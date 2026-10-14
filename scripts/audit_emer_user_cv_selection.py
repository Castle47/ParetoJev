#!/usr/bin/env python3
"""Verify saved CV checkpoints and inner-only model selection; no outer metrics."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paretojev.models.emer import EMERConfig,EMERReranker
from paretojev.utils.seed import seed_everything
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from paretojev.experiments.comparison import _source_identity
from run_emer_jev import dataset,validation,load_model


def audit(a):
    c=read_json(a.config);reg=read_json(a.run/'registration.json')
    if reg['config_hash']!=canonical_hash(c) or reg['model_source_sha256']!=_source_identity()['source_sha256']:
        raise ValueError('CV protocol or model implementation differs')
    for path,h in reg['files'].items():
        if sha256_file(path)!=h:raise ValueError('Registered source differs')
    grid=[(lr,coef) for lr in c['training']['learning_rates'] for coef in c['training']['prior_coefficients']]
    checked=[]
    for fold in a.folds:
        folder=a.run/f'fold-{fold}';selected=read_json(folder/'selection.json');manifest=read_json(folder/'data_manifest.json')
        if sha256_file(folder/'validation.npz')!=manifest['files']['validation.npz']:raise ValueError('Inner data changed')
        inner=dataset(folder/'validation.npz');models=[];trials=0
        for seed in c['training']['seeds']:
            for arm,use_jev in [('emer',False),('emer_jev',True)]:
                seed_everything(seed)
                initial=EMERReranker(EMERConfig(**c['model'],use_jev=use_jev))
                initial_score,_=validation(initial,inner,c);options=[]
                for i,(lr,coef) in enumerate(grid):
                    path=folder/f'{arm}-seed{seed}-trial{i}.pt';row=read_json(path.with_suffix('.json'))
                    if row['seed']!=seed or row['use_jev']!=use_jev or row['lr']!=lr or row['prior_coefficient']!=coef:
                        raise ValueError('Arm, seed or grid mismatch')
                    if row['parameters']!=43345 or len(row['history'])!=30 or [h['epoch'] for h in row['history']]!=list(range(1,31)):
                        raise ValueError('Training budget differs')
                    if row['checkpoint_sha256']!=sha256_file(path):raise ValueError('Trial checkpoint changed')
                    scores=[initial_score]+[h['validation'] for h in row['history']]
                    best=int(np.argmax(scores))
                    if row['best_epoch']!=best or not np.isclose(row['validation'],scores[best],rtol=0,atol=1e-12):
                        raise ValueError('Earliest best epoch selection does not replay')
                    options.append(dict(row,path=path.name));trials+=1
                best=max(options,key=lambda r:r['validation']);spec=selected[str(seed)][arm]
                if any(spec[key]!=best[key] for key in spec):raise ValueError('Best configuration differs')
                model=load_model(folder/spec['path']);point,goals=validation(model,inner,c)
                if not np.isclose(point,spec['validation'],rtol=0,atol=1e-12):raise ValueError('Selected validation mean does not replay')
                if not np.allclose(goals,best['validation_objectives'],rtol=0,atol=1e-12):raise ValueError('Selected goal metrics do not replay')
                models.append(dict(seed=seed,arm=arm,checkpoint_sha256=spec['checkpoint_sha256'],inner_validation_replayed=True))
        if trials!=24 or len(models)!=6:raise ValueError('Incomplete fold budget')
        result=dict(fold=fold,complete_fits=trials,selected_models=models,all_trial_hashes_match=True,
            initial_epoch_and_first_best_selection_replayed=True,config_selection_replayed=True,
            outer_performance_computed=False,validation_sha256=sha256_file(folder/'validation.npz'),
            selection_sha256=sha256_file(folder/'selection.json'),script_sha256=sha256_file(Path(__file__)))
        atomic_write_json(folder/'selection_audit.json',result);checked.append(fold)
        print('Verified all24 fits and6 selected checkpoints for fold',fold,flush=True)
    print('Inner-selection audit complete; no outer metrics computed:',checked,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--folds',type=int,nargs='+',required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_user_cv_v1.json'));a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):audit(a)
