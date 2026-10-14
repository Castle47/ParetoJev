#!/usr/bin/env python3
"""Publish all controlled comparisons only after exact replay."""
import argparse
from pathlib import Path
import shutil
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def report(source,output):
    result=read_json(source/'evaluation.json');replay=read_json(source/'replay.json')
    config=read_json(source/'registration.json')['config']
    recipe='混合后验' if 'posterior_mix' in config['training'] else '任务平衡'
    if not replay['exact_predictions_and_metrics']:raise ValueError('Exact replay required')
    for file,key in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(source/file)!=replay[key]:raise ValueError('Replay hashes differ')
    if result['selection_sha256']!=sha256_file(source/'selection.json'):raise ValueError('Selection changed')
    labels=dict(is_click='点击',long_view='长看',is_like='点赞',is_follow='关注',is_comment='评论',is_forward='转发',is_hate='无负反馈比例')
    titles=dict(signal=f'主比较：{recipe}EMER＋Jev vs {recipe}EMER',original_reference=f'参考比较：{recipe}EMER＋Jev vs 原EMER',training_control=f'训练方法对照：{recipe}EMER vs 原EMER')
    lines=[f'# 相同{recipe}训练下的Jev增量实验','',
        f'使用既有{result["slates"]:,}组、{result["users"]:,}名用户，两组共18次训练。没有新增Jev请求或付费GPU。',
        '**本轮在原外层结果已被检查后设计，只能视为探索性改进，不能称为新的独立确认。**',
        '两组共享训练损失和预算。主比较用于识别Jev增量；原EMER参考和训练方法对照用于区分共享训练修改的收益。',
        '前六项是按原样本权重计算的NDCG@3，最后一项为top3无负反馈比例。表中为三种子均值；区间是固定预测的条件用户bootstrap95%区间。']
    if 'posterior_mix' in config['training']:
        mix=config['training']['posterior_mix']
        lines+=['',f'两组共同使用{100*(1-mix):g}%原EMER后验＋{100*mix:g}%条件任务平衡后验；该系数固定，没有进行系数搜索。']
    for name,title in titles.items():
        r=result['pooled'][name];m=r['positive_mean'];lo,hi=m['confidence_interval']
        lines+=['',f'## {title}','',f'六目标均值：{m["emer"]:.6f} → {m["emer_jev"]:.6f}，相对{100*m["absolute_gain"]/m["emer"]:+.2f}%；绝对变化区间[{lo:+.6f}, {hi:+.6f}]。','',
            '|目标|对照|实验|绝对变化|相对变化|条件95%区间|','|---|---:|---:|---:|---:|---|']
        for goal,row in r['objectives'].items():
            low,high=row['confidence_interval'];relative='—' if goal=='is_hate' else f'{100*row["relative_gain"]:+.2f}%'
            lines.append(f'|{labels[goal]}|{row["emer"]:.6f}|{row["emer_jev"]:.6f}|{row["absolute_gain"]:+.6f}|{relative}|[{low:+.6f}, {high:+.6f}]|')
        lines+=['','|外层折|均值绝对变化|','|---|---:|']
        for f,row in result['by_fold'][name].items():lines.append(f'|{f}|{row["positive_mean"]["absolute_gain"]:+.6f}|')
        lines+=['','|种子|均值绝对变化|','|---|---:|']
        for seed,row in r['per_seed'].items():lines.append(f'|{seed}|{(sum(row["emer_jev"][:6])-sum(row["emer"][:6]))/6:+.6f}|')
        if name in result['conditions']:
            lines+=['','|预设条件|通过|','|---|---|']
            for key,value in result['conditions'][name].items():lines.append(f'|{key}|{"是" if value else "否"}|')
            conditions=result['conditions'][name]
            unchanged=[seed for seed,row in r['per_seed'].items() if row['emer']==row['emer_jev']]
            lines+=['',f'预设条件通过{sum(conditions.values())}/{len(conditions)}项。']
            if unchanged:lines.append(f'种子{"、".join(unchanged)}的两组指标完全相同，未达到“每个种子严格提升”；没有把零收益改写为正收益。')
    lines+=['','## 选模与证据边界','',f'两项要求的比较同时通过：{"是" if result["screen_passed"] else "否"}。这不自动等价于独立确认或线上收益。',
        f'18个模型中有{len(result["fallback_models"])}个保留epoch0原EMER。全部回退记录和逐目标结果见evaluation.json。',
        '单项守护允许最多0.005绝对下降；通过守护并不代表每个目标都提升。应结合上表查看长看等目标的实际退化。','',
        '|折|种子|模型|选中轮次|','|---|---|---|---:|']
    for fold,rows in read_json(source/'selection.json')['folds'].items():
        for seed,arms in rows.items():
            for arm,spec in arms.items():lines.append(f'|{fold}|{seed}|{arm}|{spec["best_epoch"]}|')
    lines+=['']+result['limitations']+['','全部检查点和内层选模通过审计，外层预测、指标和区间在独立进程精确重放。原用户留出结果与Apr23–25未打分独立队列保持原样。']
    output.mkdir(parents=True,exist_ok=True)
    for name in ('registration.json','selection.json','audit.json','evaluation.json','replay.json'):shutil.copy2(source/name,output/name)
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.source,a.output)
