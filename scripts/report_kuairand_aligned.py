#!/usr/bin/env python3
"""Render complete, nonselective aligned development evidence."""
import argparse
import shutil
from pathlib import Path
from paretojev.data.kuairand_27k import BINARY_LABELS
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def report(a):
    r = read_json(a.run/'results.json'); replay = read_json(a.run/'replay.json')
    if not replay['exact_predictions_metrics_and_bootstrap'] or replay['results_sha256'] != sha256_file(a.run/'results.json'):
        raise ValueError('Verified frozen replay required')
    if r['config']['scope'] != 'previously_seen_development_period_not_confirmatory':
        raise ValueError('Real report requires registered development scope')
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ('results.json', 'replay.json', 'registration.json', 'frozen_models.json',
                 'tabular_mmoe-epochs.json', 'sasrec_mmoe-epochs.json'):
        shutil.copyfile(a.run/name, a.output/name)
    s = read_json(a.data/'samples.json')
    shutil.copyfile(a.sequences/'sequences.json', a.output/'sequences.json')
    lines = ['# KuaiRand-27K：对齐样本后的行为模型对照', '',
        '本轮优先补公平对照，未优化负反馈采样。所有结果仍是开发结果，不能证明 Jev 或 ParetoJev 的增量。', '',
        '## 对齐了什么', '',
        '- 复用冻结 v2 样本：训练 1,699,506 条、4 月 18 日校准 188,642 条、4 月 19 日评估 187,419 条。',
        '- 相同七个原始标签、样本顺序、逆包含概率权重、概率校准方法、评价器和通过标准。',
        '- 逐条恢复原采样 occurrence，校验用户/时间/标签/权重；保存源记录编号，排除同时间戳历史。',
        '- 树模型复用冻结 checkpoint，并重新预测、核对全部指标；无需重复训练相同模型。',
        '- 两种神经模型训练固定 3 轮，最终轮 checkpoint；不按校准/评估结果选轮次。', '',
        '|模型|输入信息|训练预算|', '|---|---|---|',
        '|树模型（七个独立头）|26 个请求/历史统计/视频属性特征|每头固定 64 次 boosting|',
        '|纯 MMoE|相同 26 个特征；训练集拟合标准化、缺失指示、场景 embedding|固定 3 轮|',
        '|SASRec-style + MMoE|相同 26 个特征 + 用户 ID、候选 ID、50 条严格更早曝光视频 ID|固定 3 轮|', '',
        '**严格相同信息的架构对照是树模型 vs 纯 MMoE。SASRec 版本多了序列和 ID 信息，',
        '其比较检验整个增强方案，不能单独归因于架构。三个 epoch 与 64 次 boosting 不是相同计算预算。**', '',
        '## 校准后的完整结果', '',
        '|目标|树 AUC|MMoE AUC|SASRec+MMoE AUC|树 NLL|MMoE NLL|SASRec+MMoE NLL|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in BINARY_LABELS:
        scores = [r['models'][model]['calibrated'][name] for model in ('tree', 'tabular_mmoe', 'sasrec_mmoe')]
        lines.append('|'+name+'|'+'|'.join(f'{x["auc"]:.6f}' for x in scores)+'|'+'|'.join(f'{x["logloss"]:.8f}' for x in scores)+'|')
    lines += ['', '|目标|场景基线 Brier|树 Brier|MMoE Brier|SASRec+MMoE Brier|', '|---|---:|---:|---:|---:|']
    for name in BINARY_LABELS:
        values = [r['baselines']['tab'][name]['brier']]+[r['models'][m]['calibrated'][name]['brier'] for m in ('tree','tabular_mmoe','sasrec_mmoe')]
        lines.append('|'+name+'|'+'|'.join(f'{v:.9f}' for v in values)+'|')
    lines += ['', '通过标准不变：至少 20 个评估正例、AUC>0.5，校准后 NLL 与 Brier 均低于训练集场景发生率基线。', '']
    for model in ('tree','tabular_mmoe','sasrec_mmoe'):
        gates=r['models'][model]['gates']; failed=[n for n in BINARY_LABELS if not gates[n]]
        lines.append(f'- {model}：{sum(gates.values())}/7；未通过：'+(', '.join(failed) if failed else '无')+'。')
    lines += ['', '## 配对不确定性', '',
        '校准 NLL 差值（前者减后者）；负值有利于前者。用户聚类 bootstrap 1,000 次，',
        '95% 区间仅为探索性结果，条件于已拟合模型，未做多重比较校正或跨种子训练验证。', '',
        '|比较|目标|差值|区间下限|区间上限|', '|---|---|---:|---:|---:|']
    for comparison, value in r['calibrated_logloss_comparisons'].items():
        for i,name in enumerate(BINARY_LABELS):
            lines.append(f'|{comparison}|{name}|{value["point"][i]:.8f}|{value["lower"][i]:.8f}|{value["upper"][i]:.8f}|')
    lines += ['', '## 执行与限制', '',
        '- 同一 CPU 上重新加载后，三个模型的原始/校准预测、按目标/场景指标及 bootstrap 结果完全一致。',
        '- 所有原始指标、ECE、平均预测概率、校准参数和训练损失见 `results.json`，未根据效果挑选 raw/calibrated。',
        '- 只有一个预注册 seed 和固定训练预算，未证明各模型已充分优化或收敛。',
        '- 4 月 19 日此前已用于开发，本轮不是新测试集确认；4 月 20 日以后未参与。',
        '- item hash 仍有碰撞；历史是曝光序列；物理视频属性仍依赖 static-at-upload 假设。',
        '- 相同评估样本仍只有 85 个 is_hate 正例，未做特殊重采样、改标签或降低门槛。',
        '- 真实 Jev/ParetoJev 融合尚未训练/评估；本轮 Jev 请求为零，research target mode=false。', '',
        '## 复现', '', '```bash',
        '.venv/bin/python scripts/run_kuairand_aligned.py \\',
        '  --data data/raw/kuairand-27k/behavior-v2 \\',
        '  --sequences data/raw/kuairand-27k/behavior-aligned-v3 \\',
        '  --tree data/raw/kuairand-27k/behavior-v2/run-v2 \\',
        '  --output data/raw/kuairand-27k/behavior-aligned-v3/run-v1 --replay',
        '```', '', '重新训练时去掉 `--replay` 并使用新的 output 目录。完整源数据保存在外接硬盘项目目录。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('run','data','sequences','output'):p.add_argument('--'+name,type=Path,required=True)
    report(p.parse_args())
