#!/usr/bin/env python3
"""Report all retrospective out-of-fold evidence after exact prediction replay."""
import argparse
from pathlib import Path
import shutil
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def report(a):
    result=read_json(a.source/'evaluation.json');replay=read_json(a.source/'replay.json');prep=read_json(a.source/'preparation.json')
    for file,key in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(a.source/file)!=replay[key]:raise ValueError('Replay evidence differs')
    if not replay['exact_predictions_and_metrics']:raise ValueError('Exact replay required')
    r=result['pooled'];m=r['positive_mean'];lo,hi=m['confidence_interval']
    names=dict(is_click='点击',long_view='长看',is_like='点赞',is_follow='关注',is_comment='评论',is_forward='转发',is_hate='无负反馈比例')
    lines=['# EMER＋Jev：三折用户留出的回溯验证','',
        f'使用既有{result["slates"]:,}组真实分数、{result["users"]:,}名用户，未新增Jev请求或付费GPU。'
        '每组预测只来自该用户未参与训练和内层选模的模型；两臂共72次从头拟合。', '',
        '**本报告是已探索数据上的回溯性、按用户分组的样本外评价，不是新日期独立测试。**'
        '用户此前可能参与过Apr19之前的冻结行为模型；用户隔离约束针对本轮重排器训练与选模。', '',
        f'六正向目标均值：{m["emer"]:.6f} → {m["emer_jev"]:.6f}，'
        f'相对变化{100*m["absolute_gain"]/m["emer"]:+.2f}%；固定折外预测的条件用户bootstrap区间[{lo:+.6f}, {hi:+.6f}]。','',
        '|目标|EMER|EMER＋Jev|绝对变化|相对变化|条件区间|',
        '|---|---:|---:|---:|---:|---|']
    for name,row in r['objectives'].items():
        low,high=row['confidence_interval'];relative='—' if name=='is_hate' else f'{100*row["relative_gain"]:+.2f}%'
        lines.append(f'|{names[name]}|{row["emer"]:.6f}|{row["emer_jev"]:.6f}|{row["absolute_gain"]:+.6f}|{relative}|[{low:+.6f}, {high:+.6f}]|')
    lines+=['','|外层折|训练/验证/测试列表|EMER均值|EMER＋Jev均值|相对变化|','|---|---|---:|---:|---:|']
    for fold,row in result['by_fold'].items():
        sizes=prep['folds'][int(fold)]['sizes'];size='/'.join(str(sizes[n]['slates']) for n in ('train','validation','test'))
        mean=row['positive_mean'];relative=100*mean['absolute_gain']/mean['emer']
        lines.append(f'|{fold}|{size}|{mean["emer"]:.6f}|{mean["emer_jev"]:.6f}|{relative:+.2f}%|')
    lines+=['','|固定种子|EMER均值|EMER＋Jev均值|相对变化|','|---|---:|---:|---:|']
    for seed,row in r['per_seed'].items():
        base=sum(row['emer'][:6])/6;full=sum(row['emer_jev'][:6])/6
        lines.append(f'|{seed}|{base:.6f}|{full:.6f}|{100*(full/base-1):+.2f}%|')
    lines+=['','|预登记回溯证据条件|满足|','|---|---|']
    for key,value in result['conditions'].items():lines.append(f'|{key}|{"是" if value else "否"}|')
    lines+=['','上述条件仅为回溯证据筛查，不自动等价于完整项目成功。'
        '各折训练集相互重叠，不能把3折或9个种子-折组合当成独立实验重复。'
        '条件区间没有包含重新训练整套算法的全部不确定性，也未消除既往探索该数据的影响。', '',
        '方法依据：[Cawley与Talbot：选模过拟合与评价偏差](https://www.jmlr.org/beta/papers/v11/cawley10a.html)、'
        '[Bengio与Grandvalet：K折交叉验证方差的限制](https://www.jmlr.org/papers/v5/grandvalet04a.html)。', '',
        '全部用户划分由固定哈希决定，不依据标签或结果重分；标准化只拟合内层训练候选。'
        '每个外层折、每个种子和每个方法都有相同四组配置、每组30轮；只用内层验证集选模型。'
        '全部72次拟合和选择冻结后才计算折外指标。', '',
        '原Apr23-25的5万组确认数据未参与本轮，原模型和此前结果未改写。'
        '模型来源仍为固定原生JEV-27B，七目标树行为预测在Apr19之前拟合并冻结。', '',
        '全部预测、指标和区间在独立进程重放。证据：evaluation.json、selection.json、preparation.json、registration.json和replay.json。']
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('evaluation.json','selection.json','preparation.json','registration.json','replay.json'):
        shutil.copy2(a.source/name,a.output/name)
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    report(p.parse_args())
