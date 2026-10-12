#!/usr/bin/env python3
"""Independent arithmetic and artifact audit; frozen reports are never rewritten."""
import argparse
from dataclasses import replace
from itertools import combinations
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from paretojev.data.comparison import load_comparison_data
from paretojev.data.synthetic import collate_slates
from paretojev.experiments.comparison import load_comparison_checkpoint,_source_identity
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file,runtime_provenance


def inclusion_exclusion(points):
    total=0.
    for size in range(1,len(points)+1):
        for ids in combinations(range(len(points)),size):
            total+=(-1)**(size+1)*np.min(points[list(ids)],axis=0).prod()
    return float(total)


def reference_statistics(scores,labels,popularity,weights,preferences,top_k,cutoff):
    pcount,n,_=scores.shape;m=labels.shape[2]
    outputs=[np.zeros((pcount,n,m),dtype='float64') for _ in range(4)]
    discounts=1/np.log2(np.arange(2,top_k+2,dtype='float64'))
    for pref in range(pcount):
        selected=np.argsort(-scores[pref],axis=1,kind='stable')[:,:top_k]
        for objective in range(m):
            relevance=labels[:,:,objective].astype('float64')
            if objective==m-1:
                risk=np.take_along_axis(relevance,selected,axis=1)
                outputs[0][pref,:,objective]=(top_k-risk.sum(1))*weights
                outputs[1][pref,:,objective]=top_k*weights
                continue
            for offset,rel in [(0,relevance),(2,np.where(popularity<=cutoff,relevance,0))]:
                ideal=(np.sort(rel,axis=1)[:,-top_k:][:,::-1]*discounts).sum(1)
                dcg=(np.take_along_axis(rel,selected,axis=1)*discounts).sum(1)
                eligible=ideal>0
                outputs[offset][pref,:,objective]=np.divide(dcg,ideal,out=np.zeros(n),where=eligible)*weights
                outputs[offset+1][pref,:,objective]=eligible*weights
    def average(num,den):
        nsum,dsum=num.sum(1),den.sum(1)
        return np.divide(nsum,dsum,out=np.full_like(nsum,np.nan),where=dsum>0)
    points=average(outputs[0],outputs[1]);cold=average(outputs[2],outputs[3])[:,:-1]
    correlations=[]
    for j in range(m):
        a,b=rankdata(preferences[:,j]),rankdata(points[:,j])
        correlations.append(0. if np.ptp(a)==0 or np.ptp(b)==0 else float(np.corrcoef(a,b)[0,1]))
    return outputs,points,cold,float(np.mean(correlations))


def verify(a):
    sys.path.append(str(a.hv_library_dir.resolve()))
    import moocore
    c=read_json(a.config);root=a.data;run=root/'comparison';registry=read_json(run/'registry.json')
    summary=read_json(run/'summary.json');replay=read_json(run/'replay.json')
    assert registry['status']=='complete' and len(registry['completed'])==18
    assert replay['exact_predictions_metrics_and_intervals'] and replay['summary_sha256']==sha256_file(run/'summary.json')
    assert registry['registration']['config']==c
    assert registry['registration']['source_sha256']==_source_identity()['source_sha256']
    assert registry['registration']['script_sha256']==sha256_file('scripts/run_kuairand_core.py')
    base,_=load_comparison_data(root/'base/manifest.json');full,_=load_comparison_data(root/'full/manifest.json')
    scored=dict(np.load(root/'jev/scores.npz'));complete=read_json(root/'jev/complete.json')
    assert complete['valid_decisions']==complete['lifetime_attempts']==complete['cached_payloads']==14000
    assert sha256_file(root/'jev/scores.npz')==complete['scores_sha256']
    assert read_json(root/'full/manifest.json')['real_jev'] and not read_json(root/'base/manifest.json')['real_jev']
    for split in c['split_names']:
        with np.load(root/'base'/f'{split}.npz') as b,np.load(root/'full'/f'{split}.npz') as f:
            assert set(b.files)==set(f.files)
            for key in b.files:
                if not key.startswith('jev_'):np.testing.assert_array_equal(b[key],f[key])
            np.testing.assert_array_equal(f['jev_scores'],scored['probabilities'][f['slate_ids']].astype('float32'))
            np.testing.assert_array_equal(f['jev_confidence'],scored['confidence'][f['slate_ids']].astype('float32'))
            np.testing.assert_array_equal(f['jev_entropy'],scored['entropy'][f['slate_ids']].astype('float32'))
            assert f['mask'].all() and f['directions'].tolist()==c['directions']
            assert ((f['labels']==0)|(f['labels']==1)).all()
            assert b['jev_scores'].min()==b['jev_scores'].max()==.5
            assert not b['jev_confidence'].any() and (b['jev_entropy']==1).all()
    # Small 2D/3D/7D fixtures have independently enumerable exact union volumes.
    rng=np.random.default_rng(417)
    fixture_results=[]
    for dim in (2,3,7):
        points=rng.uniform(.05,.95,(6,dim));expected=inclusion_exclusion(points)
        actual=float(moocore.hypervolume(points,ref=0,maximise=True))
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-14)
        fixture_results.append(dict(dimensions=dim,points=6,exact=actual,inclusion_exclusion=expected))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    probe=collate_slates([full['test'][i] for i in range(3)])
    perturbed=replace(probe,jev_scores=1-probe.jev_scores,jev_confidence=1-probe.jev_confidence,jev_entropy=1-probe.jev_entropy)
    epochs={};raw=full['test'];results={};primary_per_seed=[]
    metadata=raw.metadata
    for seed in c['training']['seeds']:
        directory=run/f'seed-{seed}';report=read_json(directory/'evaluation.json')
        assert sha256_file(directory/'evaluation.json')==registry['evaluations'][str(seed)]['report_sha256']
        assert sha256_file(directory/'predictions.npz')==registry['evaluations'][str(seed)]['predictions_sha256']
        for method in c['methods']:
            entry=registry['completed'][f'{seed}/{method}'];checkpoint=directory/f'{method}.pt';receipt=directory/f'{method}.training.json'
            assert sha256_file(checkpoint)==entry['checkpoint_sha256'] and sha256_file(receipt)==entry['record_sha256']
            tr=read_json(receipt);history=tr['history']
            assert tr['epochs_executed']==20 and len(history)==20
            values=[tr['initial_validation_score']]+[h['validation_selection_score'] for h in history]
            assert tr['best_epoch']==int(np.argmax(values)) and tr['best_validation_score']==max(values)
            model,payload=load_comparison_checkpoint(checkpoint)
            assert payload['best_epoch']==tr['best_epoch'] and payload['seed']==seed and payload['method']==method
            assert payload['run_id']==registry['run_id'] and payload['directions']==c['directions']
            assert model.config.use_jev==entry['real_jev']==(not method.endswith('_no_jev'))
            if method.endswith('_no_jev'):
                with torch.inference_mode():
                    torch.testing.assert_close(model(probe)['score_mean'],model(perturbed)['score_mean'],rtol=0,atol=0)
            epochs[f'{seed}/{method}']=tr['best_epoch'];del model
        audit=dict(np.load(directory/'predictions.npz'));names=audit['method_names'].tolist()
        assert names==c['methods']+['behavior_weighted_sum']
        np.testing.assert_array_equal(audit['labels'],raw.arrays['labels'].numpy())
        np.testing.assert_array_equal(audit['mask'],raw.arrays['mask'].numpy())
        np.testing.assert_array_equal(audit['group_ids'],metadata['user_ids'].astype(str))
        with np.load(root/'full/test.npz') as f:np.testing.assert_array_equal(audit['slate_weights'],f['slate_weights'])
        per_method={};all_points={}
        for index,method in enumerate(names):
            parts,points,cold,control=reference_statistics(audit['scores'][index],audit['labels'],audit['popularity'],
                audit['slate_weights'],audit['preferences'],c['evaluation']['top_k'],float(audit['cold_popularity_threshold']))
            for key,part in zip(('utility_numerator','utility_denominator','cold_numerator','cold_denominator'),parts):
                np.testing.assert_allclose(part,audit[key][index],rtol=1e-12,atol=1e-10)
            registered=report['methods'][method]
            np.testing.assert_allclose(points,registered['points'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(cold,np.asarray(registered['cold_positive_points'],dtype=float),rtol=1e-12,atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(np.nanmean(cold),registered['cold_positive_ndcg_mean'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(control,registered['controllability']['mean'],rtol=1e-12,atol=1e-12)
            hv=float(moocore.hypervolume(points,ref=0,maximise=True));all_points[method]=points
            per_method[method]=dict(exact_hypervolume=hv,registered_mc_hypervolume=registered['hypervolume'],
                registered_mc_hits=round(registered['hypervolume']*c['evaluation']['hypervolume_samples']),
                registered_mc_standard_error=registered['hypervolume_mc_standard_error'],cold_ndcg=float(np.nanmean(cold)),controllability=control)
        comparisons={}
        for pair,original in report['factorial_comparisons'].items():
            first,second=pair.split('_vs_')
            np.testing.assert_allclose(np.maximum(all_points[second].max(0)-all_points[first].max(0),0).max(),original['worst_objective_regret'],rtol=1e-12,atol=1e-12)
            gain=per_method[first]['exact_hypervolume']/per_method[second]['exact_hypervolume']-1
            comparisons[pair]=dict(exact_hypervolume_relative_gain=gain,registered_mc_hypervolume_relative_gain=original['hypervolume_relative_gain'])
        results[str(seed)]=dict(methods=per_method,comparisons=comparisons)
        print('Verified every checkpoint/input/statistic; exact hypervolume seed',seed,flush=True)
    comparison_summary={}
    for pair,original in summary['comparisons'].items():
        first,second=pair.split('_vs_')
        exact_gains=[results[str(seed)]['comparisons'][pair]['exact_hypervolume_relative_gain'] for seed in c['training']['seeds']]
        mean_first=np.mean([results[str(seed)]['methods'][first]['exact_hypervolume'] for seed in c['training']['seeds']])
        mean_second=np.mean([results[str(seed)]['methods'][second]['exact_hypervolume'] for seed in c['training']['seeds']])
        gates={**original['gates'],'hypervolume':float(np.mean(exact_gains))>=c['evaluation']['required_hypervolume_relative_gain']}
        comparison_summary[pair]=dict(exact_mean_seed_relative_gain=float(np.mean(exact_gains)),exact_per_seed=exact_gains,
            ratio_of_mean_exact_hypervolumes=float(mean_first/mean_second-1),
            note='Registered aggregation is mean of per-seed relative gains; ratio of means is supplementary, not a replacement.',
            other_metrics=original['mean'],gates_with_exact_hv=gates,all_point_gates_with_exact_hv=all(gates.values()))
    out=dict(all_18_checkpoints_and_validation_selection_verified=True,all_real_no_jev_checkpoints_invariant=True,
        base_and_full_non_jev_inputs_identical=True,full_jev_tensors_match_raw_scored_artifact=True,
        independently_recomputed_weighted_objectives_cold_metrics_controllability_regret=True,
        refinement='Exact box-union volume of the SAME frozen maximize-utility points and origin; no changed models/preferences/reference/labels/thresholds.',
        moocore_version=moocore.__version__,numpy_version=np.__version__,library_fixture_checks=fixture_results,
        best_epochs=epochs,per_seed=results,comparisons=comparison_summary,
        registry_sha256=sha256_file(run/'registry.json'),registered_summary_sha256=sha256_file(run/'summary.json'),
        replay_sha256=sha256_file(run/'replay.json'),provenance=runtime_provenance())
    atomic_write_json(a.output,out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','output','hv-library-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/kuairand_core.json'));verify(p.parse_args())
