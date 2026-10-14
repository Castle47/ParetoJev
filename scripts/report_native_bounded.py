#!/usr/bin/env python3
"""Render the bounded development experiment with its equal-capacity control."""
import argparse
from pathlib import Path
import shutil
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def report(a):
    result=read_json(a.source/'evaluation.json');selection=read_json(a.source/'selection.json')
    reg=read_json(a.source/'registration.json');replay=read_json(a.source/'replay.json')
    for name,key in [('evaluation.json','evaluation_sha256'),('selection.json','selection_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(a.source/name)!=replay[key]:raise ValueError('Replay evidence differs')
    comparisons=result['comparisons'];j=comparisons['jev_correction_vs_frozen_emer'];b=comparisons['behavior_correction_vs_frozen_emer']
    direct=comparisons['jev_correction_vs_behavior_correction']
    names=dict(is_click='点击',long_view='长看',is_like='点赞',is_follow='关注',is_comment='评论',is_forward='转发',is_hate='无负反馈比例')
    lines=['# 有基线保底的有限修正：Jev与行为信号同容量对照','',
        '**2026-10-14；开发实验，未开放独立测试。** 固定原EMER模型，分别用行为信号或真实Jev信号训练有限修正。'
        '两组各7个系数、相同3个正则强度、同样的训练/验证规则；每候选修正幅度≤原列表分数标准差的25%。','',
        '## 数据、训练和选模','',
        '- 30,000组训练、6,000组验证（60,000验证曝光，4,253验证用户），全为已准备的真实开发数据。',
        '- 原43,345参数的EMER完全冻结，两修正组各新增7个系数；3种子×3正则×2组，共18次小模型拟合。',
        '- 输入是七项信号的列表内中心化排名，负反馈方向取反；训练只优化训练集加权长看成对逻辑损失。',
        '- 验证选模以长看最大化为准；长看和六正向均值不能低于原EMER，点击/无负反馈绝对下降最多0.005。',
        '- 零修正先进入候选且并列优先，不满足约束时回退原EMER；系数范围[−4,4]，幅度上限和搜索空间在训练前冻结。',
        '- 这些约束只保证被选候选在本验证集的点估计，不是总体、置信区间或线上无损保证。','',
        '协议提交：`ce73612`；完整输入校验及守护测试提交：`cbe4406`，均先于本轮训练结果。','',
        '## 七项目标','',
        '前三列均为三个固定种子均值。前六项为NDCG@3，最后为top3无负反馈比例，均越高越好。','',
        '|目标|冻结EMER|行为信号修正|Jev修正|Jev相对原EMER|',
        '|---|---:|---:|---:|---:|']
    for name in names:
        row=j['objectives'][name];control=b['objectives'][name]
        change=f'{row["relative_gain"]*100:+.2f}%' if name!='is_hate' else f'{row["absolute_gain"]:+.8f}（绝对）'
        lines.append(f'|{names[name]}|{row["emer"]:.6f}|{control["emer_jev"]:.6f}|{row["emer_jev"]:.6f}|{change}|')
    lines+=['','### 长看比较','',
        '|比较|相对变化|绝对变化95%区间|五项开发条件满足数|',
        '|---|---:|---|---:|']
    for label,row in [('Jev修正 vs 原EMER',j),('行为修正 vs 原EMER',b),('Jev修正 vs 行为修正',direct)]:
        g=row['objectives']['long_view'];lo,hi=g['confidence_interval']
        lines.append(f'|{label}|{g["relative_gain"]*100:+.2f}%|[{lo:+.6f}, {hi:+.6f}]|{sum(row["gates"].values())}/5|')
    lines+=['','区间为2,000次成对用户聚类bootstrap，条件于已拟合并在本验证集选出的模型，'
        '未校正选模偏差或多重比较。不能将其视为独立测试显著性或业务因果收益。','',
        '### 各种子的长看与回退','',
        '|种子|原EMER|行为修正|Jev修正|Jev是否零修正回退|',
        '|---|---:|---:|---:|---|']
    for seed,row in j['per_seed'].items():
        yes=selection[seed]['jev_correction']['selected_index']==0
        lines.append(f'|{seed}|{row["emer"][1]:.6f}|{b["per_seed"][seed]["emer_jev"][1]:.6f}|{row["emer_jev"][1]:.6f}|{"是" if yes else "否"}|')
    lines+=['','## 所有候选结果（含被拒绝者）','',
        '|种子|修正信号|正则|长看绝对变化|六目标均值绝对变化|满足守护|最终选中|',
        '|---|---|---:|---:|---:|---|---|']
    for seed,arms in selection.items():
        for arm,record in arms.items():
            baseline=record['candidates'][0]['validation']
            for index,row in enumerate(record['candidates']):
                points=row['validation'];ridge='零修正' if row['ridge'] is None else str(row['ridge'])
                lines.append(f'|{seed}|{"Jev" if arm=="jev_correction" else "行为"}|{ridge}|{points[1]-baseline[1]:+.6f}|'
                    f'{(sum(points[:6])-sum(baseline[:6]))/6:+.6f}|{"是" if row["feasible"] else "否"}|{"是" if index==record["selected_index"] else "否"}|')
    lines+=['','候选的七项目标、系数和拟合损失均保存在selection.json；未隐藏失败的正则强度或种子。','',
        '## 结论与下一步','']
    if not result['test_planning_gate_passed']:
        lines+=['本轮没有达到进入新测试规划的开发条件。Jev有限修正未同时优于原EMER和同容量行为修正，'
            '因此不能把微小的长看点估计上涨归因为有价值的Jev增量。', '',
            '不启动5万组独立测试，不新增GPU预算，不再扩大本轮搜索空间。'
            '若继续，应先核查现有Jev输入的信息充分性和目标相关性，再提出信号改进实验；'
            '不以不断微调融合系数寻找验证集正结果。']
    else:lines+=['开发规划条件满足，但仍不是正式成功；独立测试须先冻结最终协议并取得资源预算。']
    lines+=['','这一结论仅限本数据、当前文本Jev信号和已登记的有限修正框架；不能推论所有Jev应用都无效。','',
        '## 完整性和复现','',
        '- 全部原EMER权重、共同数据和25.2万项真实分数保持不变。',
        '- 独立进程重新推理冻结模型、恢复已保存的修正系数，精确重放所有候选指标、选模决定、最终预测和bootstrap区间。',
        '- 重放不重新拟合优化器，不声称跨硬件训练位级可复现。',
        '- 9项针对性测试通过：选择守护、零修正、并列优先、非有限输入拒绝、梯度、幅度上限和加权统计。', '',
        '训练：`scripts/develop_native_bounded.py --root <scale-dir> --output <run-dir>`；重放加`--replay`。', '',
        '证据：[协议与哈希](registration.json)、[全部候选](selection.json)、[完整比较](evaluation.json)、[重放](replay.json)。']
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('registration.json','selection.json','evaluation.json','replay.json'):shutil.copy2(a.source/name,a.output/name)
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    report(p.parse_args())
