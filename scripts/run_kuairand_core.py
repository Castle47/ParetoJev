#!/usr/bin/env python3
"""Fit no-Jev arms while scoring runs; later compare all six frozen arms."""
import argparse
import hashlib
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from paretojev.config import load_yaml
from paretojev.data.comparison import load_comparison_data
from paretojev.data.synthetic import collate_slates
from paretojev.evaluation.pareto import ParetoEvaluationConfig
from paretojev.evaluation.comparison import evaluate_comparison, _bootstrap
from paretojev.experiments.comparison import (_output_lock,_train_one,_source_identity,
    _assert_source_unchanged,load_comparison_checkpoint)
from paretojev.models.pareto_reranker import ParetoJevConfig
from paretojev.training.engine import PrecomputedTrainingConfig
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file,runtime_provenance


class WeightedBehaviorScore(torch.nn.Module):
    def __init__(self,directions):
        super().__init__();self.register_buffer('direction',torch.tensor(directions))
    def forward(self,batch):
        utility=torch.where(self.direction>0,batch.behavior_scores,1-batch.behavior_scores)
        score=(utility*batch.preference_weights[:,None,:]).sum(-1)
        return dict(score_mean=score.masked_fill(~batch.mask,torch.finfo(score.dtype).min))


def common_identity(datasets):
    hashes={}
    for split,d in datasets.items():
        digest=hashlib.sha256()
        for key in sorted(d.arrays):
            if key.startswith('jev_'):continue
            value=d.arrays[key].numpy()
            digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        for key,value in sorted(d.metadata.items()):
            digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        for key,value in sorted(d.source_arrays.items()):
            digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(value.tobytes())
        digest.update(d.sample_weights.tobytes());hashes[split]=digest.hexdigest()
    return hashes


def weighted_quantile(values,weights,q):
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order]
    return float(v[np.searchsorted(np.cumsum(w),q*w.sum(),side='left')])


def evaluate_seed(models,datasets,c,evaluation,cold_cut,directory,seed,audit_name="predictions.npz"):
    models=dict(models);models['behavior_weighted_sum']=WeightedBehaviorScore(c['directions'])
    loader=DataLoader(datasets['test'],batch_size=c['training']['batch_size'],shuffle=False,collate_fn=collate_slates)
    report=evaluate_comparison(models,loader,directions=c['directions'],objective_names=c['objectives'],
        cold_popularity_threshold=cold_cut,config=evaluation,baseline='emer_like_no_jev',
        group_ids=datasets['test'].metadata['user_ids'],sample_weights=datasets['test'].sample_weights,
        bootstrap_samples=c['evaluation']['bootstrap_samples'],audit_path=directory/audit_name)
    # Extra paired intervals reuse the same saved predictions and bootstrap seed.
    with np.load(directory/audit_name) as saved:
        statistics={name:tuple(saved[key][i] for key in ('utility_numerator','utility_denominator','cold_numerator','cold_denominator'))
            for i,name in enumerate(saved['method_names'])}
    report['paired_user_bootstrap_by_baseline']={}
    for baseline in ('cmr_like_no_jev','paretojev_no_jev','emer_like','cmr_like'):
        report['paired_user_bootstrap_by_baseline'][baseline]=_bootstrap(statistics,
            datasets['test'].metadata['user_ids'].astype(str),c['objectives'],np.array(c['directions'])>0,
            baseline,c['evaluation']['bootstrap_samples'],c['evaluation']['seed'])
    report['seed']=seed
    return report


def run(a):
    c=read_json(a.config);datasets,info=load_comparison_data(a.data/'manifest.json')
    manifest=info['manifest']
    if manifest['core_config_hash']!=canonical_hash(c):raise ValueError('Core config mismatch')
    if a.stage=='no-jev' and manifest['real_jev']:raise ValueError('Use registered absent-channel data for initial no-Jev fit')
    if a.stage in ('full','replay') and manifest['data_kind']=='real' and not manifest['real_jev']:
        raise ValueError('Complete real Jev scores required; no fallback')
    for split,dataset in datasets.items():
        with np.load(a.data/f'{split}.npz') as z:
            dataset.sample_weights=z['slate_weights'].astype('float64')
            dataset.source_arrays={key:z[key].copy() for key in ('source_records','source_exposure_times') if key in z}
        if dataset.metadata['objective_names'].tolist()!=c['objectives']:raise ValueError('Objective order mismatch')
        if not np.isfinite(dataset.sample_weights).all() or (dataset.sample_weights<=0).any():raise ValueError('Bad inclusion weights')
    common=common_identity(datasets);train=datasets['train'];t=c['training']
    training=PrecomputedTrainingConfig(**{key:t[key] for key in ('batch_size','epochs','learning_rate','weight_decay',
        'hidden_dim','n_heads','n_layers','ff_dim','dropout','fusion_hidden_dim','device')})
    evaluation=ParetoEvaluationConfig(**{key:c['evaluation'][key] for key in ('top_k','random_preferences',
        'include_pairwise_preferences','dirichlet_alpha','hypervolume_samples','cold_quantile','seed')})
    mc=ParetoJevConfig(candidate_feature_dim=train.feature_dim,n_objectives=train.n_objectives,
        hidden_dim=t['hidden_dim'],n_heads=t['n_heads'],n_layers=t['n_layers'],ff_dim=t['ff_dim'],
        dropout=t['dropout'],fusion_hidden_dim=t['fusion_hidden_dim'],log_popularity_reliability=True)
    cold=weighted_quantile(train.arrays['popularity'].numpy().ravel(),np.repeat(train.sample_weights,train.slate_size),c['evaluation']['cold_quantile'])
    source=_source_identity();source_script=sha256_file(Path(__file__))
    registration=dict(config=c,common_inputs=common,source_sha256=source['source_sha256'],script_sha256=source_script,
        model_config=asdict(mc),cold_cutoff=cold,data_kind=manifest['data_kind'],
        training_preference_distribution='fixed per-row Dirichlet(alpha=1,seed=0); shared across arms',
        provenance=runtime_provenance())
    identity={k:v for k,v in registration.items() if k!='provenance'};run_id=canonical_hash(identity)
    path=a.output/'registry.json'
    if path.exists():
        registry=read_json(path)
        if registry['run_id']!=run_id:raise ValueError('Registered inputs/source/config changed; refuse resume')
    else:
        if a.stage=='replay':raise ValueError('Missing run to replay')
        registry=dict(run_id=run_id,registration=registration,completed={},status='registered')
        atomic_write_json(path,registry)
    torch.set_num_threads(t['cpu_threads']);torch.use_deterministic_algorithms(True)
    reports=[];selected=[m for m in c['methods'] if a.stage!='no-jev' or m.endswith('_no_jev')]
    for seed in t['seeds']:
        directory=a.output/f'seed-{seed}';directory.mkdir(exist_ok=True);models={}
        for method in selected:
            key=f'{seed}/{method}';checkpoint=directory/f'{method}.pt';record=directory/f'{method}.training.json'
            if key in registry['completed']:
                saved=registry['completed'][key]
                if sha256_file(checkpoint)!=saved['checkpoint_sha256'] or sha256_file(record)!=saved['record_sha256']:
                    raise ValueError('Checkpoint or training receipt changed')
                model,payload=load_comparison_checkpoint(checkpoint)
                if payload['run_id']!=run_id or payload['method']!=method or payload['seed']!=seed:raise ValueError('Checkpoint identity mismatch')
            else:
                if a.stage=='replay':raise ValueError('Missing frozen method')
                print(f'Fit seed={seed} method={method}',flush=True)
                model,trained=_train_one(method,datasets,seed=seed,training=training,evaluation=evaluation,
                    model_config=mc,directory=directory,run_id=run_id,directions=c['directions'])
                _assert_source_unchanged(source['source_sha256'])
                registry['completed'][key]=dict(checkpoint_sha256=sha256_file(checkpoint),record_sha256=sha256_file(record),
                    source_manifest_sha256=sha256_file(a.data/'manifest.json'),real_jev=manifest['real_jev'])
                registry['status']='training';atomic_write_json(path,registry)
                print(f'Frozen {key} best_epoch={trained["best_epoch"]}',flush=True)
            models[method]=model
        if a.stage!='no-jev':
            audit_name='replay-predictions.npz' if a.stage=='replay' else 'predictions.npz'
            if a.stage=='replay':
                if sha256_file(directory/'predictions.npz')!=registry['evaluations'][str(seed)]['predictions_sha256']:
                    raise ValueError('Frozen audit arrays changed')
            report=evaluate_seed(models,datasets,c,evaluation,cold,directory,seed,audit_name)
            report['evidence']=dict(data_kind=manifest['data_kind'],real_jev=manifest['real_jev'],
                retrospective_pseudo_slates=True,paper_baseline_reproduction=False)
            if a.stage=='replay':
                previous=read_json(directory/'evaluation.json')
                previous['audit']['path']=report['audit']['path']
                if report!=previous:raise ValueError('Full metric/interval replay differs')
                with np.load(directory/'predictions.npz') as old,np.load(directory/audit_name) as new:
                    if set(old.files)!=set(new.files) or any(not np.array_equal(old[key],new[key]) for key in old.files):
                        raise ValueError('Prediction/statistic replay differs')
            else:
                atomic_write_json(directory/'evaluation.json',report)
                registry.setdefault('evaluations',{})[str(seed)]=dict(predictions_sha256=sha256_file(directory/'predictions.npz'),
                    report_sha256=sha256_file(directory/'evaluation.json'))
                atomic_write_json(path,registry)
            reports.append(report);print(f'Evaluated all arms seed={seed}',flush=True)
    if a.stage=='no-jev':
        registry['status']='no_jev_arms_frozen_waiting_for_real_scores';atomic_write_json(path,registry)
        print('No-Jev arms frozen; test comparison has not run',flush=True);return
    _assert_source_unchanged(source['source_sha256'])
    if sha256_file(Path(__file__))!=source_script:raise ValueError('Runner changed during execution')
    summary=dict(run_id=run_id,seeds=t['seeds'],data_kind=manifest['data_kind'],real_jev=manifest['real_jev'],
        comparisons={},research_targets_achieved=False,
        limitations=['Weighted retrospective exposed-session pseudo-slates, not causal online uplift.',
            'Project EMER-like/CMR-like variants, not original-paper reproductions.',
            'Choice distributions are not calibrated event probabilities.',
            'Bootstrap conditions on fitted checkpoints and sampled cohorts; no HV population interval.'])
    for pair in reports[0]['factorial_comparisons']:
        keys=('hypervolume_relative_gain','cold_item_ndcg_relative_gain','worst_objective_regret','controllability_spearman_mean')
        per_seed=[r['factorial_comparisons'][pair] for r in reports]
        means={key:float(np.mean([r[key] for r in per_seed])) if all(r[key] is not None for r in per_seed) else None for key in keys}
        threshold=c['evaluation']
        gates=dict(hypervolume=means[keys[0]] is not None and means[keys[0]]>=threshold['required_hypervolume_relative_gain'],
            cold_ndcg=means[keys[1]] is not None and means[keys[1]]>=threshold['required_cold_ndcg_relative_gain'],
            regret=means[keys[2]] is not None and means[keys[2]]<=threshold['maximum_worst_objective_regret'],
            controllability=means[keys[3]] is not None and means[keys[3]]>=threshold['minimum_controllability'])
        summary['comparisons'][pair]=dict(mean=means,gates=gates,all_point_gates=all(gates.values()),per_seed=per_seed)
    if a.stage=='replay':
        if summary!=read_json(a.output/'summary.json'):raise ValueError('Summary replay mismatch')
        atomic_write_json(a.output/'replay.json',dict(exact_predictions_metrics_and_intervals=True,summary_sha256=sha256_file(a.output/'summary.json'),provenance=runtime_provenance()))
    else:
        atomic_write_json(a.output/'summary.json',summary);registry['status']='complete'
        registry['summary_sha256']=sha256_file(a.output/'summary.json');atomic_write_json(path,registry)
    print('Core comparison complete; inspect failures and uncertainty before claiming a win',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/kuairand_core.json'))
    p.add_argument('--stage',choices=['no-jev','full','replay'],required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with _output_lock(a.output):run(a)
