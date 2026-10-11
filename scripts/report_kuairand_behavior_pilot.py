#!/usr/bin/env python3
"""Publish compact evidence from a completed real 27K pilot and replay."""
import argparse
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from paretojev.utils.io import read_json, atomic_write_json

LABELS = [('is_click', '点击/有效播放'), ('long_view', '长看'), ('is_like', '点赞'),
          ('is_follow', '关注'), ('is_comment', '评论'), ('is_forward', '转发'), ('is_hate', '负反馈')]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); r = read_json(a.run / 'results.json'); prep = read_json(a.data / 'preparation.json')
    replay = read_json(a.run / 'replay.json')
    if replay['status'] != 'passed' or r['config']['scope'] != 'development_pilot_not_confirmatory_benchmark':
        raise ValueError('Completed real pilot replay required')
    a.output.mkdir(parents=True, exist_ok=True)
    for source in [a.run / 'results.json', a.run / 'replay.json', a.data / 'preparation.json']:
        shutil.copyfile(source, a.output / source.name)
    s = prep['samples']['splits']; primary = r['models']['event_only']; snapshot = r['models']['released_user_snapshot']
    gates = sum(primary['readiness_gates'].values())
    d = prep['events']['training_label_diagnostics_by_tab']; totals = {k: sum(v[k] for v in d.values()) for k in next(iter(d.values()))}
    detail = {}
    with np.load(a.data / 'train.npz') as train, np.load(a.data / 'validation.npz') as val:
        nominal_start = int(datetime(2022, 4, 8, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp() * 1000)
        detail['sampled_train_targets_before_nominal_4_08'] = int((train['time'] < nominal_start).sum())
        detail['sample_time_ranges_local'] = {name: [datetime.fromtimestamp(s[name][key] / 1000, ZoneInfo('Asia/Shanghai')).isoformat() for key in ('min_time', 'max_time')] for name in ('train', 'validation')}
        detail['validation_users_not_in_training'] = int(len(np.setdiff1d(val['user'], train['user'])))
        for name,z in [('train', train), ('validation', val)]:
            detail[name] = dict(unique_candidate_ids=int(len(np.unique(z['raw_item']))),
                occupied_candidate_hash_buckets=int(len(np.unique(z['item']))))
    atomic_write_json(a.output / 'summary.json', dict(status='completed_development_pilot',
        research_targets_achieved=False, jev_calls=0, primary_readiness_gates_passed=gates,
        primary_readiness_gates_total=7,
        primary_beats_tab_logloss_count=sum(primary['validation'][k]['logloss'] < r['baselines']['tab_prevalence'][k]['logloss'] for k,_ in LABELS), splits=s, label_diagnostics_training_only=totals,
        encoding=detail, source_code_commit=r['provenance']['git_commit'],
        external_artifact_directory=str(a.data.resolve())))
    lines = ['# KuaiRand-27K 首轮真实多目标行为试验', '',
        f'**已完成真实训练；event_only 开发筛查通过 {gates}/7 个任务。尚未证明 ParetoJev 的增益。**', '',
        '预先注册口径：[REGISTRATION.md](REGISTRATION.md)。本轮没有 Jev 调用，没有训练融合/重排器，',
        '没有查看后期 standard 或 random 测试指标。画像组仅为时间来源不明的敏感性对照。', '',
        '## 实际数据', '',
        f"完整读取并重验两个早期分片的 {prep['events']['rows']:,} 条曝光。来源是完整 27K，",
        '但目标采用固定的每用户采样预算，**不是在全部曝光上训练**。历史为最近 50 次严格更早曝光，',
        '不使用历史反馈，避免把尚未发生完的观看结果当作即时可用特征。', '',
        '| 期间 | 目标样本 | 用户 | 权重代表的曝光数 | 历史达到 50 的比例 |',
        '|---|---:|---:|---:|---:|']
    for name,label in [('train','早期分片→4/18 零时训练'),('validation','4/18–4/19 验证')]:
        v=s[name]; lines.append(f"| {label} | {v['rows']:,} | {v['users']:,} | {v['represented_exposures']:,.0f} | {v['history_length_50_fraction']:.2%} |")
    lines += ['', f"实际训练样本时间范围：{' 至 '.join(detail['sample_time_ranges_local']['train'])}。",
        f"其中 {detail['sampled_train_targets_before_nominal_4_08']} 条目标早于文件名标注的 4/8；分片 date 与 time_ms 并非完全对齐。",
        '实际切分始终按预注册 time_ms 阈值，未因观察结果改变边界或剔除这些记录；只能保证按记录时间隔离，字段来源差异仍待核实。', '',f"验证期有 {detail['validation_users_not_in_training']} 名训练未见用户，采用统一 OOV 用户表示。",
        '按逆抽样概率加权训练与评估，避免把均匀按用户抽样误当作曝光分布；未做正负类别重加权。发生率基线仅从同一批训练目标及其权重估计。', '',
        '## 加权 ROC-AUC', '', '| 任务 | 验证阳性数 | 按 tab 发生率 | event_only | 加发布版用户画像 |',
        '|---|---:|---:|---:|---:|']
    for key,label in LABELS:
        v=primary['validation'][key]; b=r['baselines']['tab_prevalence'][key]; q=snapshot['validation'][key]
        lines.append(f"| {label} | {v['sampled_positives']:,} | {b['auc']:.4f} | {v['auc']:.4f} | {q['auc']:.4f} |")
    lines += ['', '全局发生率基线的 AUC 为 0.5。点击标签混合了不同界面的点击/有效播放含义，不能统称纯 CTR；',
        '尚未确认 numeric tab 对应具体 UI。各 tab 分层指标完整保存在 results.json。', '',
        '## 概率误差与开发门槛', '', '| 任务 | 全局 log loss | tab log loss | event log loss | 全局 Brier | event Brier | event 筛查 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for key,label in LABELS:
        v=primary['validation'][key]; b=r['baselines']['global_prevalence'][key]; t=r['baselines']['tab_prevalence'][key]
        lines.append(f"| {label} | {b['logloss']:.6f} | {t['logloss']:.6f} | {v['logloss']:.6f} | {b['brier']:.6f} | {v['brier']:.6f} | {'通过' if primary['readiness_gates'][key] else '未通过'} |")
    lines += ['', f"主模型点击/有效播放平均预测 {primary['validation']['is_click']['mean_prediction']:.2%}，验证发生率 {primary['validation']['is_click']['prevalence']:.2%}；长看平均预测 {primary['validation']['long_view']['mean_prediction']:.2%}，实际 {primary['validation']['long_view']['prevalence']:.2%}。概率偏低，不能直接声称已校准。"]
    lines += ['', '## 训练曲线', '', '| 特征口径 | epoch | 训练 log loss | 验证平均 log loss |', '|---|---:|---:|---:|']
    for regime in ('event_only', 'released_user_snapshot'):
        for epoch in r['models'][regime]['epochs']:
            loss = sum(v['logloss'] for v in epoch['validation'].values()) / 7
            lines.append(f"| {regime} | {epoch['epoch']} | {epoch['train_logloss']:.6f} | {loss:.6f} |")
    lines += ['', '模型按最低验证误差选择；后续 epoch 如果训练下降而验证上升，即显示本预算下的过拟合迹象。']
    failed=[label for key,label in LABELS if not primary['readiness_gates'][key]]
    lines += ['', '未通过任务：'+('、'.join(failed) if failed else '无')+'。',
        '门槛为预先固定的开发点估计筛查，无独立测试或置信区间，不能等同于研究目标达成。',
        'PR-AUC、10-bin ECE、发生率/平均预测值、全部 epoch 和画像组概率误差见 results.json。',
        '低发生率任务的 ECE/Brier 本来就可能很小，不能单看绝对数值宣称校准良好。', '',
        '## 数据质量与限制', '',
        f"仅训练期间的时长/播放有效性并集异常为 {totals['invalid_union']:,}/{totals['rows']:,}。",
        f"时长为零 {totals['duration_zero']:,}，负时长 {totals['duration_negative']:,}，负播放时长 {totals['play_negative']:,}；这些集合可能重叠。",
        f"在有效时长/播放记录中，原 long_view 阳性而公式阴性 {totals['provided_positive_formula_negative']:,}，反方向 {totals['provided_negative_formula_positive']:,}。",
        f"其中距公式阈值 100ms 内的差异 {totals['mismatch_within_100ms_of_threshold']:,}；不足以据此推断定义或重写标签。",
        '所有原始二值标签均保留，不因为时长异常就删掉点击/点赞等反馈。', '',
        f"event_only 参数量 {primary['parameters']:,}，最佳 epoch {primary['best_epoch']}；画像组参数量 {snapshot['parameters']:,}，最佳 epoch {snapshot['best_epoch']}。",
        'event_only 是 7 个字段及其派生编码；画像组增加 30 个发布版字段，其中有常量，快照时间未知。',
        '它们不是已经充分验证的工业级富特征精排模型。候选时长假定曝光时可用，未做线上可用性审计。',
        '视频嵌入仅 65,536 桶，存在明显容量与碰撞限制；summary.json 记录候选 ID 与桶数量。',
        '尚未加入 item basic 内容属性；整月视频统计始终排除。原项目研究门槛保持未达成。', '',
        '## 复现与下一步', '',
        '来源、样本、checkpoint、预测均留存哈希；最佳 checkpoint 的预测与指标已在独立进程逐项精确复现。',
        '这不等于重新训练一次或跨硬件复现。完整数组和 checkpoint 位于外接盘专属项目目录，路径见 summary.json。',
        '先针对弱任务检查稀疏性、校准和场景差异，再注册更多独立训练曝光、视频侧可用特征和正则化对照；不能把增加 epoch 当作过拟合的解决办法。',
        '行为模型冻结后才能为更晚的重排训练期生成 pXTR，再开展真实 Jev 与融合方式的公平比较。', '']
    (a.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__': main()
