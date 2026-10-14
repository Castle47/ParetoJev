#!/usr/bin/env python3
"""Render the frozen independent-period outcome, only after exact replay."""
import argparse
from pathlib import Path
import shutil
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def report(a):
    result=read_json(a.source/'evaluation.json');replay=read_json(a.source/'replay.json')
    for file,key in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(a.source/file)!=replay[key]:raise ValueError('Replay identity differs')
    if not replay['exact_predictions_and_metrics'] or not replay['all_raw_responses_audited_again']:
        raise ValueError('Complete response and prediction replay required')
    r=result['results'];passed=r['confirmation']['all_conditions'];support=result['support']
    names=dict(is_click='点击',long_view='长看',is_like='点赞',is_follow='关注',is_comment='评论',is_forward='转发',is_hate='无负反馈比例')
    m=r['positive_mean'];lo,hi=m['confidence_interval']
    outcome='达到预登记的整体多目标条件' if passed else '未达到预登记的整体多目标条件'
    lines=['# EMER＋Jev 独立时期验证：'+outcome,'',
        f'KuaiRand-27K，2022-04-23至25；{support["slates"]:,}组列表、{support["users"]:,}名用户、{support["candidate_exposures"]:,}次候选曝光。'
        '模型、抽样、提示词和评价规则均在本次模型效果计算前冻结。', '',
        f'主指标六正向NDCG@3均值：**{m["emer"]:.6f} → {m["emer_jev"]:.6f}**，'
        f'相对变化 **{100*m["absolute_gain"]/m["emer"]:+.2f}%**；绝对差95%区间[{lo:+.6f}, {hi:+.6f}]。','',
        '|目标|EMER|EMER＋Jev|绝对变化|相对变化|绝对差95%区间|',
        '|---|---:|---:|---:|---:|---|']
    for name,row in r['objectives'].items():
        low,high=row['confidence_interval'];rel='—' if name=='is_hate' else f'{row["relative_gain"]*100:+.2f}%'
        lines.append(f'|{names[name]}|{row["emer"]:.6f}|{row["emer_jev"]:.6f}|{row["absolute_gain"]:+.6f}|{rel}|[{low:+.6f}, {high:+.6f}]|')
    lines+=['','前六项为NDCG@3，负反馈为top3无负反馈比例。每项完整报告，不隐藏退化。'
        '点估计为三组固定训练种子均值；2,000次配对用户聚类bootstrap条件于这六个已训练模型。', '',
        '|确认条件|结果|','|---|---|']
    labels=dict(primary_mean_ci_positive='六正向均值差区间下限>0',primary_mean_gain_every_seed='每个固定种子六正向均值均提升',
        all_positive_goal_point_guards='各正向目标点估计绝对下降均不超过0.005',hate_risk_point_guard='负反馈风险点估计绝对增加不超过0.005')
    for key,value in r['confirmation']['gates'].items():lines.append(f'|{labels[key]}|{"通过" if value else "未通过"}|')
    lines+=['','这些点估计约束不等价于总体非劣性证明，也不意味着每项目标都显著提升。'
        '本报告不会将整体收益称为严格Pareto占优。','',
        '|种子|EMER六目标均值|EMER＋Jev六目标均值|变化|','|---|---:|---:|---:|']
    for seed,row in r['per_seed'].items():
        b=sum(row['emer'][:6])/6;f=sum(row['emer_jev'][:6])/6
        lines.append(f'|{seed}|{b:.6f}|{f:.6f}|{f-b:+.6f}|')
    lines+=['','|日期|列表数|用户数|EMER均值|EMER＋Jev均值|','|---|---:|---:|---:|---:|']
    for day,row in result['per_day'].items():
        b=sum(row['points']['emer'][:6])/6;f=sum(row['points']['emer_jev'][:6])/6
        lines.append(f'|{day}|{row["slates"]}|{row["users"]}|{b:.6f}|{f:.6f}|')
    lines+=['','日分项用于描述稳定性，不用于事后剔除日期。','',
        '两臂共同输入、行为预测、43,345参数网络和开发训练预算相同；Jev臂额外使用七项真实JEV-27B信号。'
        '本次没有在测试数据上训练、重新选模型、调阈值或补样追求显著。', '',
        '全部35万项原始响应逐条匹配候选、目标和最终分数。共同特征只使用原开发期的编码和行为模型。'
        '所有预测、指标和区间在独立进程精确重放。', '',
        '结论范围：公开日志候选的离线整体多目标排序。没有在线A/B、业务因果提升或完整生产RT证据。'
        '各目标正例支持、权重有效样本量和完整限制见evaluation.json。', '',
        '证据：[完整结果](evaluation.json)、[冻结注册](registration.json)、[响应审计](response_audit.json)、[重放](replay.json)。']
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('evaluation.json','registration.json','response_audit.json','replay.json','tensor_manifest.json'):
        shutil.copy2(a.source/name,a.output/name)
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    report(p.parse_args())
