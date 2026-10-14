#!/usr/bin/env python3
"""Freeze inner-only comparator-aware routing, then evaluate both existing recipes."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from paretojev.experiments.comparison import _output_lock,_source_identity
from run_emer_scale import immutable_json
from run_emer_jev import dataset,load_model,validation
from run_emer_balanced_transfer import evaluate
from report_emer_balanced_transfer import report


def use_jev(candidate,matched,anchor,limits):
    candidate=np.asarray(candidate);matched=np.asarray(matched);anchor=np.asarray(anchor)
    if not all(x.shape==(7,) and np.isfinite(x).all() for x in (candidate,matched,anchor)):raise ValueError('Seven finite inner metrics required')
    bound=np.array([limits['maximum_positive_goal_absolute_point_drop']]*6+[limits['maximum_hate_risk_absolute_point_increase']])
    return bool(candidate[:6].mean()>matched[:6].mean() and (candidate-matched>=-bound).all() and (candidate-anchor>=-bound).all())


def prepare(a,c):
    hashes={};parents={}
    for recipe,directory in c['recipes'].items():
        source=a.root/directory;reg=read_json(source/'registration.json');parents[recipe]=reg
        if reg['model_source_sha256']!=_source_identity()['source_sha256']:raise ValueError('Frozen model source differs')
        for path,h in reg['inputs'].items():
            if path in hashes and hashes[path]!=h:raise ValueError('Conflicting input ancestry')
            hashes[path]=h
        replay=read_json(source/'replay.json');audit=read_json(source/'audit.json')
        if not replay['exact_predictions_and_metrics'] or not audit['all18_verified']:raise ValueError('Completed source audit/replay required')
        for n,k in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
            if sha256_file(source/n)!=replay[k]:raise ValueError('Parent evaluation changed')
        if sha256_file(source/'selection.json')!=audit['selection_sha256']:raise ValueError('Parent selection changed')
        for n in ('registration.json','selection.json','audit.json','evaluation.json','predictions.npz','replay.json'):
            hashes[str(source/n)]=sha256_file(source/n)
    for path,h in hashes.items():
        if sha256_file(path)!=h:raise ValueError('Source ancestry changed')
    original=read_json(a.root/'user-cv-v1/registration.json')['config']
    for recipe,directory in c['recipes'].items():
        source=a.root/directory;out=a.output/recipe;out.mkdir(parents=True,exist_ok=True)
        parent_selection=read_json(source/'selection.json');chosen={};decisions=[]
        config=copy.deepcopy(parents[recipe]['config']);config.update(role=c['role'],reuse_models_only=True,
            matched_fallback_policy=True,objectives=original['objectives'],directions=original['directions'])
        config['source_experiment_id']=config['experiment_id'];config['experiment_id']=f'{c["experiment_id"]}-{recipe}'
        for key in ('maximum_positive_goal_absolute_point_drop','maximum_hate_risk_absolute_point_increase'):
            if config['evaluation'][key]!=c[key]:raise ValueError('Original guard must not change')
        config['evaluation'].update(bootstrap_samples=c['bootstrap_samples'],bootstrap_seed=c['bootstrap_seed'],
            click_maximum_absolute_ndcg_drop=.005,hate_maximum_absolute_risk_increase=.005)
        config['limitations']+=c['limitations']
        for f,rows in parent_selection['folds'].items():
            chosen[f]={}
            for seed,arms in rows.items():
                records={};paths={}
                for arm,spec in arms.items():
                    path=source/spec['path'];receipt=path.with_suffix('.json');row=read_json(receipt)
                    if sha256_file(path)!=spec['checkpoint_sha256'] or row['checkpoint_sha256']!=spec['checkpoint_sha256']:raise ValueError('Parent checkpoint differs')
                    hashes[str(path)]=sha256_file(path);hashes[str(receipt)]=sha256_file(receipt);records[arm]=row;paths[arm]=path
                matched=records['emer']['validation_objectives'];candidate=records['emer_jev']['validation_objectives'];anchor=records['emer']['anchor_objectives']
                keep=use_jev(candidate,matched,anchor,c);source_arm='emer_jev' if keep else 'emer'
                chosen[f][seed]={}
                for arm,take in [('emer','emer'),('emer_jev',source_arm)]:
                    row=records[take];chosen[f][seed][arm]=dict(path=str(paths[take]),checkpoint_sha256=row['checkpoint_sha256'],best_epoch=row['best_epoch'])
                decisions.append(dict(fold=int(f),seed=int(seed),use_jev=keep,source_arm=source_arm,
                    candidate_inner=candidate,matched_inner=matched,anchor_inner=anchor,
                    inner_mean_gain=float(np.mean(candidate[:6])-np.mean(matched[:6]))))
        immutable_json(out/'registration.json',dict(config=config,parent_registration_sha256=sha256_file(source/'registration.json'),new_training=False))
        immutable_json(out/'selection.json',dict(folds=chosen,completed_fits=0,reused_model_slots=18,config_hash=canonical_hash(config)))
        immutable_json(out/'policy.json',dict(decisions=decisions,all_decisions_inner_only=True,outer_metrics_computed=False))
        for n in ('registration.json','selection.json','policy.json'):hashes[str(out/n)]=sha256_file(out/n)
        print(recipe,'retained Jev',sum(r['use_jev'] for r in decisions),'of9; all choices frozen',flush=True)
    hashes[str(a.config)]=sha256_file(a.config);hashes[str(Path(__file__))]=sha256_file(Path(__file__))
    immutable_json(a.output/'registration.json',dict(config=c,inputs=hashes,new_training=False,all18_policy_choices_frozen=True,previous_outer_results_inspected=True))


def verify(a):
    reg=read_json(a.output/'registration.json')
    if reg['config']!=read_json(a.config):raise ValueError('Policy configuration changed')
    for p,h in reg['inputs'].items():
        if sha256_file(p)!=h:raise ValueError('Policy/source hash differs')
    return reg['config']


def audit(a,c):
    for recipe,directory in c['recipes'].items():
        out=a.output/recipe;source=a.root/directory;config=read_json(out/'registration.json')['config']
        policy=read_json(out/'policy.json');selected=read_json(out/'selection.json');models=[];cache={}
        for f in range(3):
            inner=dataset(a.root/'user-cv-v1'/f'fold-{f}'/'validation.npz')
            for decision in [r for r in policy['decisions'] if r['fold']==f]:
                seed=decision['seed'];parent=read_json(source/'selection.json')['folds'][str(f)][str(seed)];points={}
                for arm,spec in parent.items():
                    p=source/spec['path'];model=load_model(p);_,goals=validation(model,inner,config)
                    if goals!=decision['candidate_inner' if arm=='emer_jev' else 'matched_inner']:raise ValueError('Inner metrics do not replay')
                    points[arm]=goals;cache[str(p)]=(sum(t.numel() for t in model.parameters()),model.config.use_jev)
                old=read_json(a.root/'user-cv-v1'/'selection.json')['folds'][str(f)][str(seed)]['emer']
                _,anchor=validation(load_model(a.root/'user-cv-v1'/f'fold-{f}'/old['path']),inner,config)
                if anchor!=decision['anchor_inner'] or use_jev(points['emer_jev'],points['emer'],anchor,c)!=decision['use_jev']:raise ValueError('Policy decision does not replay')
                for arm in ('emer','emer_jev'):
                    s=selected['folds'][str(f)][str(seed)][arm];path=Path(s['path']);parameters,flag=cache[str(path)]
                    take='emer' if arm=='emer' or not decision['use_jev'] else 'emer_jev'
                    if path!=source/parent[take]['path'] or sha256_file(path)!=s['checkpoint_sha256'] or parameters!=43345 or flag!=(take=='emer_jev'):raise ValueError('Routed model differs')
                    models.append(dict(path=str(path),checkpoint_sha256=s['checkpoint_sha256'],best_epoch=s['best_epoch'],validation_replayed=True))
        if len(models)!=18:raise ValueError('Missing selected model slot')
        atomic_write_json(out/'audit.json',dict(models=models,selection_sha256=sha256_file(out/'selection.json'),all18_verified=True,new_training=False,policy_replayed=True))
        print(recipe,'all9 inner decisions and18 model slots audited',flush=True)


def publish(a,c):
    for recipe in c['recipes']:report(a.output/recipe,a.reports/recipe)
    lines=['# 匹配基线回退：完整对照','',
        '两种原配方均使用相同的内层策略：Jev候选必须优于当前匹配无Jev模型并满足七目标守护，否则使用该匹配无Jev模型。所有18项选择在任何新外层计算前冻结。',
        '这是一项独立记录的新策略评估，没有新增训练、Jev请求或付费GPU。原实验结果不改写。此前外层结果已被检查，因此不能称为独立确认。','',
        '|原配方|保留Jev槽位|同配方平均增量|主比较条件|原EMER参考条件|报告|','|---|---:|---:|---|---|---|']
    for recipe in c['recipes']:
        r=read_json(a.output/recipe/'evaluation.json');p=read_json(a.output/recipe/'policy.json');m=r['pooled']['signal']['positive_mean']
        lines.append(f'|{recipe}|{sum(d["use_jev"] for d in p["decisions"])}/9|{100*m["absolute_gain"]/m["emer"]:+.2f}%|{sum(r["conditions"]["signal"].values())}/5|{sum(r["conditions"]["original_reference"].values())}/5|[完整七目标]({recipe}/README.md)|')
    lines+=['','所有种子、折、守护和零增益均完整报告；条件区间不覆盖全部重训练不确定性。Apr23–25独立队列仍未打分。']
    a.reports.mkdir(parents=True,exist_ok=True);(a.reports/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reports',type=Path,default=Path('reports/emer-matched-fallback-v1'));p.add_argument('--config',type=Path,default=Path('configs/emer_matched_fallback_v1.json'))
    p.add_argument('--stage',choices=['prepare','audit','evaluate','replay','report'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=2):
        if a.stage=='prepare':prepare(a,read_json(a.config))
        else:
            c=verify(a)
            if a.stage=='audit':audit(a,c)
            elif a.stage=='report':publish(a,c)
            else:
                for recipe in c['recipes']:
                    out=a.output/recipe;args=SimpleNamespace(run=a.root/'user-cv-v1',output=out,stage=a.stage)
                    evaluate(args,read_json(out/'registration.json')['config'])
