#!/usr/bin/env python3
"""Summarize the complete registered experiment, including negative conclusions."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def fmt(value,percent=False):
    if value is None:return '不可评估'
    return f'{100*value:+.2f}%' if percent else f'{value:.6f}'


def report(a):
    summary=read_json(a.run/'summary.json');replay=read_json(a.run/'replay.json')
    if not replay['exact_predictions_metrics_and_intervals'] or replay['summary_sha256']!=sha256_file(a.run/'summary.json'):
        raise ValueError('Exact real replay required')
    if summary['data_kind']!='real' or not summary['real_jev']:raise ValueError('No real conclusion from synthetic/absent Jev')
    scores=read_json(a.data/'jev/complete.json');audit=read_json(a.output/'score_audit.json')
    cohort=read_json(a.output/'cohort_audit.json');prep=read_json(a.data/'preparation.json')
    if not audit['all_scores_match_validated_raw_responses'] or audit['decisions_checked']!=scores['valid_decisions']:
        raise ValueError('Raw response audit incomplete')
    if not cohort['all_source_occurrences_labels_and_weights_match']:raise ValueError('Cohort audit missing')
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','registry.json','replay.json'):shutil.copyfile(a.run/name,a.output/name)
    for name in ('preparation.json','selection.json','features.json','prompts.json','captions.json'):
        shutil.copyfile(a.data/name,a.output/name)
    shutil.copyfile(a.data/'jev/complete.json',a.output/'decision_provenance.json')
    reports=[]
    for seed in summary['seeds']:
        source=a.run/f'seed-{seed}';dest=a.output/f'seed-{seed}';dest.mkdir(exist_ok=True)
        for name in ['evaluation.json']+[p.name for p in source.glob('*.training.json')]:shutil.copyfile(source/name,dest/name)
        reports.append(read_json(source/'evaluation.json'))
    primary=summary['comparisons']['paretojev_vs_emer_like_no_jev']
    title='点指标达到全部预设门槛，需结合区间判断证据强度' if primary['all_point_gates'] else '未达到全部预设门槛'
    lines=['# ParetoJev 核心验证：真实 Jev 与无 Jev 基线', '',
        f'主比较（ParetoJev + Jev vs EMER-like 无 Jev）：**{title}**。', '',
        '这是已登记的真实 KuaiRand-27K 离线伪候选组实验；结论仅适用于本协议。',
        'EMER-like/CMR-like 是项目实现，并非原论文复现；不能据此宣布超过已发表 EMER 系统或产生在线因果收益。', '',
        '## 比较了什么', '',
        '- 同一冻结行为树模型、相同候选特征、标签、抽样权重和偏好网格。',
        '- EMER-like、CMR-like、ParetoJev 各自有／无 Jev，另报告未经训练的行为分数加权和。',
        '- 1,000 组训练、400 组验证、600 组测试，每组 10 个候选；三次随机种子，固定 20 轮预算。',
        '- 验证集选择 checkpoint；测试结果不用于调参或挑种子。7 个原始目标均保留，负反馈方向取最小。',
        f'- {scores["valid_decisions"]:,} 个真实 JEV-27B Choice 决策，{scores["lifetime_attempts"]:,} 次 HTTP 尝试；所有输出与原始响应逐一核对。',
        '- Choice 是候选间选择分布，不能当作独立点击率/转化率；没有合成 Jev 分数计入真实结果。', '',
        '## 全部预登记比较（三个种子的均值）', '',
        '|比较|HV 相对变化|冷门 NDCG 相对变化|最差目标 regret|可控性|4 项点门槛通过数|',
        '|---|---:|---:|---:|---:|---:|']
    for pair,value in summary['comparisons'].items():
        m=value['mean'];lines.append(f'|{pair}|{fmt(m["hypervolume_relative_gain"],True)}|{fmt(m["cold_item_ndcg_relative_gain"],True)}|{fmt(m["worst_objective_regret"])}|{fmt(m["controllability_spearman_mean"])}|{sum(value["gates"].values())}/4|')
    lines += ['', '原门槛保持 HV ≥2%、冷门 NDCG ≥3%、regret ≤0.005、可控性 ≥0.60。',
        'JeV 信息增量看 EMER-like/CMR-like 的有无 Jev 对照；框架增量看所有方法都有 Jev 的比较。',
        '不能只选取最容易获胜的基线或目标。点门槛通过不等于统计确认。', '',
        '## 主比较各目标的配对用户区间', '',
        '以下为跨偏好平均效用绝对差值的 95% 用户聚类 bootstrap 区间。正值有利于 ParetoJev。',
        '正向目标用条件 NDCG；负反馈用 1−选择风险。未做多重比较校正。', '',
        '|种子|目标|95% 区间|有效重采样次数|', '|---|---|---|---:|']
    for r in reports:
        intervals=r['paired_user_bootstrap']['comparisons']['paretojev']['objective_mean_absolute_gain']
        for objective,value in intervals.items():
            bounds=value['interval'];text='不可评估' if bounds is None else '['+', '.join(f'{v:+.6f}' for v in bounds)+']'
            lines.append(f'|{r["seed"]}|{objective}|{text}|{value["valid_replicates"]}|')
        v=r['paired_user_bootstrap']['comparisons']['paretojev']['cold_ndcg_absolute_gain'];bounds=v['interval']
        text='不可评估' if bounds is None else '['+', '.join(f'{x:+.6f}' for x in bounds)+']'
        lines.append(f'|{r["seed"]}|冷门 NDCG|{text}|{v["valid_replicates"]}|')
    lines += ['', '其他有／无 Jev、等 Jev 对照的区间与每个种子的完整指标见对应 `seed-*/evaluation.json`。', '',
        '## 数据和解释边界', '',
        '- 抽样覆盖稀疏目标，指标使用精确的逆包含概率权重。样本的阳性比例不是自然曝光发生率。',
        '- 全体候选的历史和请求时点固定到组内首次曝光；不读取后续候选各自时点的额外历史。',
        '- 原始日志中的后续曝光仍受原系统策略与会话反馈影响，伪候选组不是真实召回集合。',
        '- 文本和物理属性快照时间、外部预训练与公开数据是否重叠未获独立证明。',
        '- 超体积误差是 Monte Carlo 积分误差，不是用户总体置信区间；种子均值不是显著性证明。',
        '- Bootstrap 条件于拟合模型、当前分层样本和固定权重；保留有效样本量及未定义区间。',
        '- 检查点、预测、所有指标和区间已独立重载复算。原始数据、缓存与大文件留在外接硬盘。', '',
        '本报告不自动宣布项目全部研究/部署目标达成，仍需逐项解释主比较与等信号对照。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_write_json(a.output/'verification.json',dict(real_response_audit=True,real_cohort_audit=True,
        exact_predictions_metrics_intervals=True,primary_all_point_gates=primary['all_point_gates'],
        full_project_research_targets_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','run','output'):p.add_argument('--'+key,type=Path,required=True)
    report(p.parse_args())
