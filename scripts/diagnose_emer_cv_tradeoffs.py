#!/usr/bin/env python3
"""Post-run tradeoff diagnosis. Never selects or evaluates new outer models."""
import argparse
from pathlib import Path
import numpy as np
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file
from run_emer_scale import immutable_json


def support_statistics(labels, mask, weights, users):
    valid=mask.astype(bool); positive=(labels>0)&valid[...,None]
    n=len(labels); population=float(np.sum(weights[:,None]*valid))
    groups,codes=np.unique(users,return_inverse=True); result=[]
    for j in range(labels.shape[-1]):
        has=positive[...,j].any(1); eligible=has & ((~positive[...,j])&valid).any(1)
        masses=np.bincount(codes,weights=weights*has,minlength=len(groups))
        result.append(dict(positive_exposures=int(positive[...,j].sum()),
            positive_slates=int(has.sum()),positive_users=int((masses>0).sum()),
            weighted_prevalence=float(np.sum(weights[:,None]*positive[...,j])/population),
            weighted_positive_slate_fraction=float(np.sum(weights*has)/weights.sum()),
            weighted_pair_supervision_fraction=float(np.sum(weights*eligible)/weights.sum()),
            positive_support_user_weight_ess=float(masses.sum()**2/np.sum(masses**2)) if masses.any() else 0.0))
    return result


def weighted_pair_auc(scores, labels, mask, weights):
    pairs=(labels[:,:,None]>labels[:,None,:]) & mask[:,:,None] & mask[:,None,:]
    count=pairs.sum((1,2)); delta=scores[:,:,None]-scores[:,None,:]
    wins=(((delta>0)+.5*(delta==0))*pairs).sum((1,2))
    auc=np.divide(wins,count,out=np.zeros(len(scores)),where=count>0)
    w=weights*(count>0)
    return float(np.sum(w*auc)/w.sum()) if w.sum() else None


def diagnose(a):
    reg=read_json(a.run/'registration.json');c=reg['config'];selection=read_json(a.run/'selection.json')
    main=read_json(a.run/'evaluation.json'); replay=read_json(a.run/'replay.json')
    if not replay['exact_predictions_and_metrics'] or replay['evaluation_sha256']!=sha256_file(a.run/'evaluation.json'):
        raise ValueError('Completed exact evaluation replay required')
    paths=[a.run/n for n in ('registration.json','selection.json','evaluation.json','replay.json')]
    inner=[];support={};candidates=[]
    for f in range(c['outer_folds']):
        folder=a.run/f'fold-{f}';manifest=read_json(folder/'data_manifest.json')
        for role in ('train','validation','test'):
            path=folder/f'{role}.npz';paths.append(path)
            if sha256_file(path)!=manifest['files'][path.name]:raise ValueError('Fold data changed')
            with np.load(path) as z:
                y=z['labels'];mask=z['mask'];w=z['slate_weights'];users=z['user_ids']
                stats=support_statistics(y,mask,w,users)
                follow=dict(behavior=weighted_pair_auc(z['behavior_scores'][...,3],y[...,3],mask,w),
                    jev=weighted_pair_auc(z['jev_scores'][...,3],y[...,3],mask,w))
            support[f'{f}/{role}']=dict(slates=len(y),users=len(np.unique(users)),
                objectives=dict(zip(c['objectives'],stats)),follow_within_slate_auc=follow)
        for seed,arms in selection['folds'][str(f)].items():
            selected={}
            for arm,spec in arms.items():
                path=(folder/spec['path']).with_suffix('.json');paths.append(path);selected[arm]=read_json(path)
                if selected[arm]['checkpoint_sha256']!=spec['checkpoint_sha256']:raise ValueError('Selected receipt differs')
            base=np.array(selected['emer']['validation_objectives'])
            delta=np.array(selected['emer_jev']['validation_objectives'])-base
            inner.append(dict(fold=f,seed=int(seed),selected_jev=arms['emer_jev']['path'],
                mean_gain=float(delta[:6].mean()),delta=delta.tolist()))
            for trial in range(4):
                path=folder/f'emer_jev-seed{seed}-trial{trial}.json';paths.append(path);row=read_json(path)
                d=np.array(row['validation_objectives'])-base
                fails=[name for j,name in enumerate(c['objectives']) if d[j] < -c['evaluation'][
                    'maximum_hate_risk_absolute_point_increase' if j==6 else 'maximum_positive_goal_absolute_point_drop']]
                candidates.append(dict(fold=f,seed=int(seed),trial=trial,mean_gain=float(d[:6].mean()),
                    delta=d.tolist(),failed_inner_point_guards=fails))
    result=dict(role='Exploratory diagnosis after outer results were opened; no new confirmation',
        new_model_predictions=False,new_training=False,new_jev_calls=0,
        inputs={str(p):sha256_file(p) for p in sorted(set(paths))},script_sha256=sha256_file(Path(__file__)),
        selected_inner=inner,all_existing_jev_candidates=candidates,support=support,
        selected_inner_follow_guard_failures=sum(r['delta'][3]<-.005 for r in inner),
        existing_candidates_passing_all_inner_guards=sum(not r['failed_inner_point_guards'] for r in candidates),
        original_outer_follow={f:r['objectives']['is_follow'] for f,r in main['by_fold'].items()},
        notes=['Pair AUC averages per-slate positive-negative comparisons with original weights and half credit for ties.',
            'Support ESS summarizes concentration of positive-slate weights per user, not a formal power calculation.',
            'Positive-label frequency and pair eligibility do not measure actual gradient attribution or prove causality.',
            'Training satisfaction sums raw feedback bits; selection maximizes six conditional NDCG means. These are different objectives.',
            'No candidate is selected using outer outcomes; original results and guards remain unchanged.'])
    a.output.mkdir(parents=True,exist_ok=True);immutable_json(a.output/'diagnosis.json',result)
    lines=['# 用户留出实验：目标退化诊断','',
        '这是打开外层结果后的探索诊断，没有新训练、Jev请求或新模型外层预测；不能作为独立确认。','',
        f'9组选中模型中，{result["selected_inner_follow_guard_failures"]}组在内层验证已超过关注绝对下降0.005上限。'
        f'36个既有Jev候选中，{result["existing_candidates_passing_all_inner_guards"]}个满足全部内层单项守护。'
        '因此不能通过简单重选一个既有候选解决全部约束。','',
        '|折|种子|内层平均变化|内层关注变化|内层评论变化|','|---|---|---:|---:|---:|']
    for row in inner:lines.append(f'|{row["fold"]}|{row["seed"]}|{row["mean_gain"]:+.6f}|{row["delta"][3]:+.6f}|{row["delta"][4]:+.6f}|')
    lines+=['','|外层折|关注相对变化|关注正例列表|关注正例用户|用户权重ESS|行为/JeV关注配对AUC|','|---|---:|---:|---:|---:|---|']
    for f,row in result['original_outer_follow'].items():
        s=support[f'{f}/test'];q=s['objectives']['is_follow'];auc=s['follow_within_slate_auc']
        lines.append(f'|{f}|{100*row["relative_gain"]:+.2f}%|{q["positive_slates"]}|{q["positive_users"]}|{q["positive_support_user_weight_ess"]:.1f}|{auc["behavior"]:.4f} / {auc["jev"]:.4f}|')
    lines+=['','|训练折|点击正例率|长看正例率|关注正例率|关注配对监督列表权重占比|','|---|---:|---:|---:|---:|']
    for f in range(3):
        s=support[f'{f}/train']['objectives']
        lines.append(f'|{f}|{100*s["is_click"]["weighted_prevalence"]:.3f}%|{100*s["long_view"]["weighted_prevalence"]:.3f}%|{100*s["is_follow"]["weighted_prevalence"]:.3f}%|{100*s["is_follow"]["weighted_pair_supervision_fraction"]:.3f}%|')
    lines+=['','训练后验损失把六个正向反馈相加，而评价分别计算有正例列表上的NDCG再取六目标均值。'
        '稀疏任务在两者中的相对贡献可能不一致；这是待验证假设，不能把频率直接当作梯度贡献或已证明的原因。',
        '模型选择也只最大化均值，没有把单项守护放进选模。下一步应检验任务平衡的训练目标，并在相同训练策略下对比有无Jev，避免把损失修改的收益误归因于Jev。',
        '完整七目标支持数、全部36候选内层差异和来源哈希见diagnosis.json；本报告不改变原4/5结论。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    print('Tradeoff diagnosis saved:',a.output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    diagnose(p.parse_args())
