#!/usr/bin/env python3
"""Report exploratory signal repair separately from model superiority."""
import argparse
import shutil
from pathlib import Path

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def report(a):
    audit=read_json(a.phase/'signal_audit.json');probe=read_json(a.phase/'explicit-longview-validation/result.json')
    old=read_json(a.phase/'bounded-residual/result.json');new=read_json(a.phase/'corrected-residual/result.json')
    native=read_json(a.phase/'corrected_signal_audit.json');a.output.mkdir(parents=True,exist_ok=True)
    allowed=new['confirmation_allowed']
    lines=['# Jev 信号诊断与任务定义修复','',
        '**日期：2026-10-13。** 本阶段全部属于开发研究。4 月 22 日已被查看，不能把它重新称为独立测试。',
        '', '**已证实的开发发现：明确长看事件定义，能改善 Jev 的长看信号。** 这与证明“EMER＋Jev 有稳定业务收益”是两个问题。',
        '', '## 原信号质量','',
        '以下为同一候选组正负候选对的加权 AUC，平分同分对；它不是跨用户总体 AUC，也不是 CTR。',
        '', '|目标|验证：行为|验证：原 Jev|4月22日开发：行为|4月22日开发：原 Jev|','|---|---:|---:|---:|---:|']
    for name,v in audit['splits']['validation']['metrics'].items():
        b,j,*_=v['within_slate_auc']['values'];d=audit['splits']['april22_development']['metrics'][name]['within_slate_auc']['values']
        lines.append(f'|{name}|{b:.6f}|{j:.6f}|{d[0]:.6f}|{d[1]:.6f}|')
    lines+=['','原长看 Jev 分数在验证／开发两期均约 0.50，明显弱于目标行为预测。存在纠错对，但也存在更多把行为正确对排错的情况，不能用事后 oracle 挑选正确对来证明可部署增益。',
        '', '## 问题定义的真实对照','',
        '原问题只说 long view，没有说明 [官方数据集定义](https://github.com/chongminggao/KuaiRand#data-descriptions)：短于或等于 18 秒的视频需看完，否则至少观看 18 秒。零／未知时长不解释成零秒完成。',
        '', '同一 400 组验证候选、同一上下文和模型 revision，只改问题文本，保留真实标签。新问题不新增点击历史，也不发送反馈标签。',
        '', '|指标|旧问题 Jev|明确事件定义 Jev|目标行为预测|新−旧绝对差|差值 95% 用户区间|','|---|---:|---:|---:|---:|---|']
    for key,label in [('within_slate_auc','候选内 AUC'),('ndcg','NDCG@3')]:
        row=probe[key];v=row['values'];lo,hi=row['difference_ci']
        lines.append(f'|{label}|{v[0]:.6f}|{v[1]:.6f}|{v[2]:.6f}|{row["difference"]:+.6f}|[{lo:+.6f}, {hi:+.6f}]|')
    periods=read_json(a.phase/'corrected_signal_periods.json')
    lines+=['','其他已使用时期的复核也完整报告：','',
        '|时期|旧 Jev AUC|修正 Jev AUC|行为 AUC|旧 Jev NDCG|修正 Jev NDCG|行为 NDCG|','|---|---:|---:|---:|---:|---:|---:|']
    for name,row in periods.items():
        v=row['auc']['values'];n=row['ndcg']['values']
        lines.append(f'|{name}|{v[0]:.6f}|{v[1]:.6f}|{v[2]:.6f}|{n[0]:.6f}|{n[1]:.6f}|{n[2]:.6f}|')
    lines+=['','三个时期新旧信号 AUC 与 NDCG 差值区间下界均为正，但修正后的信号仍弱于目标行为预测的点估计。这支持任务定义修复，不证明额外信息已被融合模型有效利用。']
    lines+=['','这是开发探针，区间不代表独立新日期确认。原标签中的少量时长公式不一致始终保留，没有重写为有利标签。',
        '', '## 冻结 EMER 的小幅修正','',
        'EMER 三个原始 checkpoint 的权重不变。修正从零开始，限定每个候选最多变化四分之一组内基线标准差；没有信号、均匀信号或零系数时返回原基线。',
        '', '|开发方案|新增系数|长看绝对差|长看相对变化|绝对差 95% 用户区间|','|---|---:|---:|---:|---|']
    for result,label,count in [(old,'原七信号',7),(new,'修正后的单一长看信号',1)]:
        row=result['results']['objectives']['long_view'];lo,hi=row['confidence_interval']
        lines.append(f'|{label}|{count}|{row["absolute_gain"]:+.6f}|{row["relative_gain"]*100:+.2f}%|[{lo:+.6f}, {hi:+.6f}]|')
    lines+=['','修正后单信号方案的全部开发指标：','',
        '|目标|冻结 EMER|EMER＋修正信号|绝对变化|绝对变化 95% 区间|','|---|---:|---:|---:|---|']
    for name,row in new['results']['objectives'].items():
        lo,hi=row['confidence_interval']
        lines.append(f'|{name}|{row["emer"]:.6f}|{row["emer_jev"]:.6f}|{row["absolute_gain"]:+.6f}|[{lo:+.6f}, {hi:+.6f}]|')
    lines+=['','is_hate 仍表示无负反馈比例，其余为 NDCG@3。系数只由 4 月 20 日训练、4 月 21 日选择；未按 4 月 22 日重新选择模型或扩大修正上限。',
        '', '## 是否进入新日期确认','',
        ('开发条件全部通过，可以按已登记协议进入 4 月 23 日确认；当前报告仍不是确认结果。' if allowed else
         '**开发条件没有全部通过，未准备或打分 4 月 23 日。** 保留该日期，不消耗预算强行验证缺少开发证据的方案。'),
        '', '|预登记开发条件|结果|','|---|---|']
    lines += [f'|{key}|{"通过" if value else "未通过"}|' for key,value in new['results']['gates'].items()]
    lines+=['','## 审计与边界','',
        f'本阶段新增 {native["total_real_decisions"]} 次真实、明确事件定义的判断：1,000 训练＋400 验证＋600 开发。原生响应、候选映射、数据输入与实际标签已核对。',
        '', '此前 4,200 次新日期主实验及其负结果完整保留。新探针和新修正不能覆盖、替换此前结论。',
        '', '曝光历史不等于正向兴趣历史；日志没有反馈可用时间戳，未伪造已完成反馈。缺少真实未曝光请求候选池和 p_watchtime，仍是公开数据 EMER 适配。',
        '', '仍未验证生产 RT、吞吐、货币成本或线上业务收益。单一信号只需每组一个 Jev 问题，比七问题减少请求量，但不等于零在线延迟。',
        '', '协议：[信号诊断设计](../../docs/JEV_SIGNAL_DIAGNOSIS.md)。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    for source,target in [('signal_audit.json','signal_audit.json'),('corrected_signal_audit.json','corrected_signal_audit.json'),
        ('explicit-longview-validation/result.json','definition_probe.json'),('explicit-longview-validation/registration.json','definition_registration.json'),
        ('bounded-residual/result.json','old_residual_result.json'),('bounded-residual/selection.json','old_residual_selection.json'),
        ('corrected-residual/result.json','corrected_residual_result.json'),('corrected-residual/selection.json','corrected_residual_selection.json'),
        ('corrected-residual/registration.json','corrected_residual_registration.json'),('prospective_confirmation.json','prospective_confirmation.json'),
        ('corrected_signal_periods.json','corrected_signal_periods.json')]:
        shutil.copyfile(a.phase/source,a.output/target)
    atomic_write_json(a.output/'evidence.json',dict(report_sha256=sha256_file(a.output/'README.md'),
        signal_audit_sha256=sha256_file(a.phase/'signal_audit.json'),corrected_response_audit_sha256=sha256_file(a.phase/'corrected_signal_audit.json'),
        confirmation_allowed=allowed,fresh_confirmation_done=False))
    print('Signal-repair report written; fresh-confirmation gate:',allowed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    report(p.parse_args())
