#!/usr/bin/env python3
"""Multi-objective bounded correction; versioned separately from long-view run."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paretojev.evaluation.comparison import comparison_objective_metrics
from paretojev.experiments.comparison import _source_identity, _output_lock
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file
from run_emer_scale import immutable_json
from run_emer_jev import dataset, load_model, predict, paired_report
from jev_residual import rank_basis, corrected_scores, score_scale
from multiobjective_residual import fit_correction
from audit_jev_increment import oriented


def feasible(points, baseline, guards):
    if not np.isfinite(points).all() or not np.isfinite(baseline).all():
        return False
    eps=guards['numerical_tolerance']
    return bool(points[1]>=baseline[1]-guards['long_view_max_absolute_drop']-eps and
        points[:6].mean()>=baseline[:6].mean()-guards['positive_mean_max_absolute_drop']-eps and
        points[0]>=baseline[0]-guards['click_max_absolute_drop']-eps and
        points[6]>=baseline[6]-guards['no_hate_max_absolute_drop']-eps)


def choose(candidates, guards):
    baseline=np.array(candidates[0]['validation']);best=0
    for i,row in enumerate(candidates):
        points=np.array(row['validation']);row['feasible']=feasible(points,baseline,guards)
        if row['feasible'] and points[:6].mean()>np.mean(candidates[best]['validation'][:6])+guards['numerical_tolerance']:
            best=i
    return best


def main(a):
    design=read_json(a.config);c=read_json('configs/emer_jev_scale.json');target=read_json('configs/emer_jev_scale_target.json')
    c['evaluation'].update(target['effect_gates'])
    base=read_json(a.root/'training/emer_selection.json');manifest=read_json(a.root/'training/common_manifest.json')
    native=a.root/'jev-native-v9-fp16'
    complete=read_json(native/'complete.json');audit=read_json(a.root/'training-native/native_response_audit.json')
    if not complete['complete'] or complete['public_cache_mixed'] or not audit['all_raw_responses_match'] or audit['decisions']!=252000:
        raise ValueError('Native collection was not fully audited')
    if sha256_file(native/'scores.npz')!=complete['scores_sha256']:
        raise ValueError('Score bytes differ')
    if sha256_file(native/'registration.json')!=complete['registration_sha256']:
        raise ValueError('Producer registration differs')
    if base['registration']['source_sha256']!=_source_identity()['source_sha256']:
        raise ValueError('Baseline implementation changed')
    if sha256_file(Path('scripts/run_emer_jev.py'))!=base['registration']['fit_script_sha256']:
        raise ValueError('Original fit implementation changed')
    if design['seeds']!=c['training']['seeds'] or design['coefficient_bounds']!=[-4.,4.]:
        raise ValueError('Unsupported fitting protocol')
    if base['registration']['common_manifest_sha256']!=sha256_file(a.root/'training/common_manifest.json'):
        raise ValueError('Baseline was trained on different common inputs')
    if (design['bootstrap_samples'],design['bootstrap_seed'])!=(c['evaluation']['bootstrap_samples'],c['evaluation']['bootstrap_seed']):
        raise ValueError('Bootstrap protocol differs')
    files=[a.config,Path('configs/emer_jev_scale.json'),Path('configs/emer_jev_scale_target.json'),
        a.root/'training/emer_selection.json',a.root/'training/common_manifest.json',
        a.root/'evaluation-native/evaluation.json',a.root/'evaluation-native/predictions.npz',
        a.root/'training-native/native_response_audit.json',native/'scores.npz',native/'registration.json',
        Path('scripts/jev_residual.py'),Path('scripts/multiobjective_residual.py'),Path('scripts/run_emer_jev.py'),Path(__file__)]
    for name in c['split_names']:
        path=a.root/'training'/f'{name}.npz'
        if sha256_file(path)!=manifest['files'][path.name]:raise ValueError('Common input changed')
        files.append(path)
    for spec in base['selected'].values():
        path=a.root/'training'/spec['path']
        if sha256_file(path)!=spec['checkpoint_sha256']:raise ValueError('Frozen baseline changed')
        files.append(path)
    registration=dict(design=design,input_sha256={str(p):sha256_file(p) for p in files},
        model_package_sha256=_source_identity()['source_sha256'],baseline_parameters=43345,
        additional_parameters=7,independent_test_opened=False)
    immutable_json(a.output/'registration.json',registration)
    ds={};basis={}
    with np.load(native/'scores.npz') as scores:
        for name in c['split_names']:
            ds[name]=dataset(a.root/'training'/f'{name}.npz')
            with np.load(a.root/'training'/f'{name}.npz') as z:
                ids=z['slate_ids']
                if not np.array_equal(z['candidate_ids'],scores['candidate_ids'][ids]):raise ValueError('Candidate alignment changed')
                probs=scores['probabilities'][ids]
            mask=ds[name].arrays['mask'].numpy()
            # Use the same float32 signal representation as the trained EMER arms.
            basis[name]={'jev_correction':rank_basis(oriented(probs.astype('float32')),mask),
                'behavior_correction':rank_basis(oriented(ds[name].arrays['behavior_scores'].numpy()),mask)}
    selection={};predictions={};cap=design['cap_standard_deviations'];changes={}
    def metrics(scores,d):
        return comparison_objective_metrics(scores,d.arrays['labels'].numpy(),d.arrays['mask'].numpy(),
            c['directions'],top_k=3,sample_weights=d.sample_weights)[0]
    previous=read_json(a.output/'selection.json') if a.replay else None
    with np.load(a.root/'evaluation-native/predictions.npz') as saved:
        original={k:saved[k] for k in saved.files}
    for seed in design['seeds']:
        spec=base['selected'][str(seed)];model=load_model(a.root/'training'/spec['path'])
        pred={name:predict(model,data,32) for name,data in ds.items()}
        if not np.array_equal(pred['validation'],original[f'emer/{seed}']):raise ValueError('Baseline prediction replay differs')
        val=ds['validation'];mask=val.arrays['mask'].numpy();baseline=metrics(pred['validation'],val)
        predictions[f'frozen_emer/{seed}']=pred['validation'];selection[str(seed)]={}
        for arm in ('behavior_correction','jev_correction'):
            if a.replay:
                record=previous[str(seed)][arm]
            else:
                candidates=[dict(beta=[0.]*7,ridge=None,fit=None,validation=baseline.tolist())]
                for ridge in design['ridge_grid']:
                    train=ds['train'];tm=train.arrays['mask'].numpy()
                    beta,receipt=fit_correction(pred['train'],basis['train'][arm],train.arrays['labels'].numpy(),
                        tm,train.sample_weights,ridge,cap)
                    points=metrics(corrected_scores(pred['validation'],basis['validation'][arm],beta,mask,cap),val)
                    candidates.append(dict(beta=beta.tolist(),ridge=ridge,fit=receipt,validation=points.tolist()))
                index=choose(candidates,design['guards'])
                record=dict(selected_index=index,beta=candidates[index]['beta'],ridge=candidates[index]['ridge'],candidates=candidates)
            # Every candidate's validation and selection must replay, including rejected candidates.
            for candidate in record['candidates']:
                output=corrected_scores(pred['validation'],basis['validation'][arm],np.array(candidate['beta']),mask,cap)
                if not np.array_equal(metrics(output,val),candidate['validation']):raise ValueError('Candidate validation differs')
            if choose(record['candidates'],design['guards'])!=record['selected_index']:
                raise ValueError('Guarded selection differs')
            chosen=record['candidates'][record['selected_index']]
            if record['beta']!=chosen['beta'] or record['ridge']!=chosen['ridge']:
                raise ValueError('Selected coefficients differ from chosen candidate')
            output=corrected_scores(pred['validation'],basis['validation'][arm],np.array(record['beta']),mask,cap)
            bound=cap*score_scale(pred['validation'],mask)[:,None]
            if np.any(np.abs(output-pred['validation'])[mask]>(np.broadcast_to(bound,mask.shape)+1e-12)[mask]):
                raise ValueError('Correction exceeded cap')
            selection[str(seed)][arm]=record;predictions[f'{arm}/{seed}']=output
            before=np.argsort(-pred['validation'],axis=1,kind='stable')[:,:3]
            after=np.argsort(-output,axis=1,kind='stable')[:,:3]
            changed=(np.sort(before,axis=1)!=np.sort(after,axis=1)).any(1)
            changes[f'{arm}/{seed}']=dict(top3_membership_changed_slates=int(changed.sum()),
                weighted_fraction=float(np.average(changed,weights=val.sample_weights)),zero_fallback=record['selected_index']==0)
            print(seed,arm,'ridge',record['ridge'],'long_gain',record['candidates'][record['selected_index']]['validation'][1]-baseline[1],flush=True)
    comparisons={}
    for enhanced,control in [('jev_correction','frozen_emer'),('behavior_correction','frozen_emer'),('jev_correction','behavior_correction')]:
        mapped={f'{tag}/{seed}':predictions[f'{arm}/{seed}'] for tag,arm in [('emer',control),('emer_jev',enhanced)] for seed in design['seeds']}
        comparisons[enhanced+'_vs_'+control]=paired_report(mapped,ds['validation'],c)
    for comparison in comparisons.values():
        comparison['legacy_longview_gates_not_used']=comparison.pop('gates');comparison.pop('all_effect_gates')
        comparison['multiobjective_gates']=dict(mean_gain_lower_95_ci_positive=comparison['positive_mean']['confidence_interval'][0]>0,
            all_seven_point_gains_nonnegative=all(v['absolute_gain']>=-1e-12 for v in comparison['objectives'].values()))
        comparison['development_gate_passed']=all(comparison['multiobjective_gates'].values())
    advance=all(comparisons[key]['development_gate_passed'] for key in ('jev_correction_vs_frozen_emer','jev_correction_vs_behavior_correction'))
    result=dict(role='selected-on-validation development only; not confirmation',comparisons=comparisons,
        changes=changes,test_planning_gate_passed=advance,independent_test_opened=False,online_business_success=False,
        interpretation='Equal-capacity behavior correction controls for generic score adjustment; observed guards are validation-only, not a deployment guarantee.')
    if a.replay:
        if previous!=selection or read_json(a.output/'evaluation.json')!=result:raise ValueError('Selection or evaluation replay differs')
        with np.load(a.output/'predictions.npz') as old:
            if set(old.files)!=set(predictions) or any(not np.array_equal(old[k],v) for k,v in predictions.items()):
                raise ValueError('Predictions differ')
        atomic_write_json(a.output/'replay.json',dict(exact_model_predictions=True,exact_coefficients_and_selection=True,
            exact_metrics_and_intervals=True,optimizer_refit=False,
            selection_sha256=sha256_file(a.output/'selection.json'),predictions_sha256=sha256_file(a.output/'predictions.npz'),
            evaluation_sha256=sha256_file(a.output/'evaluation.json')))
    else:
        immutable_json(a.output/'selection.json',selection)
        np.savez_compressed(a.output/'predictions.npz',**predictions)
        atomic_write_json(a.output/'evaluation.json',result)
    print('Development test-planning gate:',advance,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_multiobjective_bounded.json'));p.add_argument('--replay',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=4):main(a)
