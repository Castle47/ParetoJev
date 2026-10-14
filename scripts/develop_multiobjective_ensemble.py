#!/usr/bin/env python3
"""Equal-budget ensembles of existing EMER arms under the clarified objective."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_emer_jev import dataset, paired_report
from run_emer_scale import immutable_json
from jev_residual import rank_basis
from paretojev.evaluation.comparison import comparison_objective_metrics
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def ensemble(values,mask,recipe):
    if values.ndim!=3 or values.shape[1:]!=mask.shape or len(values)!=3:
        raise ValueError('Expected three models on the same candidates')
    if not np.isfinite(values[:,mask]).all():raise ValueError('Invalid scores')
    outputs=[]
    for s in values:
        if recipe=='equal_weight_three_seed_zscore':
            mean=np.where(mask,s,0).sum(1)/mask.sum(1)
            centered=np.where(mask,s-mean[:,None],0)
            std=np.sqrt(np.square(centered).sum(1)/mask.sum(1)).clip(1e-6)
            outputs.append(centered/std[:,None])
        elif recipe=='equal_weight_three_seed_rank':outputs.append(rank_basis(s,mask)[...,0])
        else:raise ValueError('Unregistered aggregation rule')
    return np.mean(outputs,axis=0)


def main(a):
    design=read_json(a.config);c=read_json('configs/emer_jev_scale.json')
    c['training']['seeds']=[0]  # one deployed ensemble; constituents are not bootstrap observations
    c['evaluation'].update(bootstrap_samples=design['bootstrap_samples'],bootstrap_seed=design['bootstrap_seed'],
        click_maximum_absolute_ndcg_drop=.005,hate_maximum_absolute_risk_increase=.005)
    previous=a.root/'evaluation-native';replay=read_json(previous/'replay.json')
    for name,key in [('predictions.npz','predictions_sha256'),('evaluation.json','evaluation_sha256')]:
        if sha256_file(previous/name)!=replay[key]:raise ValueError('Original evaluation changed')
    original=read_json(previous/'evaluation.json')
    path=a.root/'training/validation.npz'
    if sha256_file(path)!=original['protocol']['validation_sha256']:raise ValueError('Validation bytes differ')
    selections={arm:read_json(a.root/folder/file) for arm,folder,file in
        [('emer','training','emer_selection.json'),('emer_jev','training-native','selection.json')]}
    checkpoints={}
    for arm,s in selections.items():
        folder='training' if arm=='emer' else 'training-native'
        for row in s['selected'].values():
            file=a.root/folder/row['path']
            if sha256_file(file)!=row['checkpoint_sha256']:raise ValueError('Constituent checkpoint changed')
            checkpoints[str(file)]=row['checkpoint_sha256']
    registration=dict(design=design,inputs={str(p):sha256_file(p) for p in
        [a.config,Path(__file__),Path('scripts/jev_residual.py'),Path('scripts/run_emer_jev.py'),
         previous/'predictions.npz',previous/'evaluation.json',path]},checkpoints=checkpoints,
        parameters_per_arm=3*43345,deployed_models_per_arm=3,new_learned_parameters=0,
        independent_test_opened=False,role='development selection only')
    immutable_json(a.output/'registration.json',registration)
    d=dataset(path);mask=d.arrays['mask'].numpy();labels=d.arrays['labels'].numpy()
    with np.load(previous/'predictions.npz') as z:raw={arm:np.stack([z[f'{arm}/{seed}'] for seed in (7,17,29)]).astype('float64') for arm in ('emer','emer_jev')}
    predictions={};trials={};selected={}
    for arm in raw:
        options=[]
        for recipe in design['candidate_recipes']:
            scores=ensemble(raw[arm],mask,recipe);predictions[f'{arm}/{recipe}']=scores
            points=comparison_objective_metrics(scores,labels,mask,c['directions'],top_k=3,sample_weights=d.sample_weights)[0]
            options.append(dict(recipe=recipe,positive_mean=float(points[:6].mean()),objectives=points.tolist()))
        trials[arm]=options;selected[arm]=max(options,key=lambda r:r['positive_mean'])
    comparisons={}
    for recipe in design['candidate_recipes']+['selected_by_same_rule']:
        pair={f'{arm}/0':predictions[f'{arm}/{selected[arm]["recipe"] if recipe=="selected_by_same_rule" else recipe}'] for arm in raw}
        report=paired_report(pair,d,c)
        # Preserve the old metric implementation but do not relabel its long-view gates as the new objective.
        report['legacy_longview_gates_not_used']=report.pop('gates');report.pop('all_effect_gates')
        tol=design['development_gate']['numerical_tolerance']
        report['multiobjective_gates']=dict(mean_gain_lower_95_ci_positive=report['positive_mean']['confidence_interval'][0]>0,
            all_seven_point_gains_nonnegative=all(r['absolute_gain']>=-tol for r in report['objectives'].values()))
        report['development_gate_passed']=all(report['multiobjective_gates'].values())
        comparisons[recipe]=report
    result=dict(role='development only; does not satisfy independent confirmation requirement',trials=trials,selected=selected,
        comparisons=comparisons,independent_test_opened=False,online_business_success=False)
    if a.replay:
        if read_json(a.output/'evaluation.json')!=result:raise ValueError('Metric replay differs')
        with np.load(a.output/'predictions.npz') as old:
            if set(old.files)!=set(predictions) or any(not np.array_equal(old[k],v) for k,v in predictions.items()):raise ValueError('Prediction replay differs')
        atomic_write_json(a.output/'replay.json',dict(exact_aggregation_and_metrics=True,
            evaluation_sha256=sha256_file(a.output/'evaluation.json'),predictions_sha256=sha256_file(a.output/'predictions.npz')))
    else:
        if (a.output/'predictions.npz').exists():raise ValueError('Use --replay')
        np.savez_compressed(a.output/'predictions.npz',**predictions);atomic_write_json(a.output/'evaluation.json',result)
    print(result['selected'],flush=True)
    print(comparisons['selected_by_same_rule']['multiobjective_gates'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_multiobjective_v1.json'));p.add_argument('--replay',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=4):main(a)
