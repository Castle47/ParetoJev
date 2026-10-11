#!/usr/bin/env python3
"""Publish completed real v2 controls without converting development gates into research wins."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from paretojev.data.kuairand_27k import BINARY_LABELS
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file

def auc_text(value):
    return 'NA' if value is None else f'{value:.4f}'


NAMES=['点击/有效播放','长看','点赞','关注','评论','转发','负反馈']


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=read_json(a.run/'results.json');replay=read_json(a.run/'replay.json')
    if r['scope']!='previously_seen_development_period_not_confirmatory' or replay['status']!='passed':
        raise ValueError('Completed real run and successful replay required')
    if read_json(a.data/'samples.json')!=r['samples'] or sha256_file(a.data/'metadata.json')!=r['samples']['metadata_manifest_sha256']:
        raise ValueError('Report input provenance changed')
    a.output.mkdir(parents=True,exist_ok=True)
    for path in [a.run/'results.json',a.run/'replay.json',a.run/'frozen_models.json',a.data/'samples.json',a.data/'metadata.json']:
        shutil.copyfile(path,a.output/path.name)
    metadata=read_json(a.data/'metadata.json');s=r['samples']['splits'];models=r['models']
    context=models['exposure_context'];content=models['upload_metadata_assumed_static']
    missing={}
    for split,spec in s.items():
        x=np.load(a.data/f'{split}-x.npy',mmap_mode='r');counts=np.zeros(x.shape[1],dtype='int64')
        for start in range(0,spec['rows'],50000): counts+=np.isnan(x[start:min(start+50000,spec['rows'])]).sum(axis=0)
        missing[split]=dict(zip(r['feature_fields'],counts.tolist()))
    summary=dict(scope=r['scope'],research_targets_achieved=False,jev_requests=0,
        source_commit=r['provenance']['git_commit'],splits=s,
        gates={name:sum(v['gates'].values()) for name,v in models.items()},
        missing_sample_counts=missing,artifact_directory=str(a.data.resolve()),
        metadata_assumption=r['config']['metadata_assumption'])
    atomic_write_json(a.output/'summary.json',summary)
    lines=['# KuaiRand-27K 第二轮：视频属性与独立校准时段','',
        f"**已完成真实开发实验：请求/历史统计组通过 {sum(context['gates'].values())}/7，加入视频属性组通过 {sum(content['gates'].values())}/7 项更严格的开发门槛。**",'',
        '本轮新增的是七个独立树模型头，不是重新训练的 SASRec，也不是 ParetoJev 融合器。',
        '4/19 已在上一轮开发中看过，不能称为独立未见测试。原项目研究目标仍未完成，Jev 调用为零。',
        '方案先于结果登记，见 [REGISTRATION.md](REGISTRATION.md)。','',
        '## 样本和隔离','',
        '| 用途 | 时间边界（Asia/Shanghai） | 目标样本 | 用户 | 权重代表曝光 |',
        '|---|---|---:|---:|---:|']
    for split,label,period in [('train','训练','早期日志，time_ms < 4/18 零时'),('calibration','校准','4/18'),('evaluation','开发评估','4/19')]:
        v=s[split];lines.append(f"| {label} | {period} | {v['rows']:,} | {v['users']:,} | {v['represented_exposures']:,.0f} |")
    lines += ['', '使用完整早期曝光历史，每用户最多 64 条训练、8 条校准和 8 条评估目标；使用逆抽样概率权重。',
        '这是目标采样训练，不是完整 9,955 万条训练期曝光全部拟合。4/20 以后的重排数据未用于本轮指标或选择。',
        '只按记录 time_ms 隔离；已知 date/time_ms 不完全对齐。所有模型及校准器冻结后才加载本轮评估标签。', '',
        '## 特征与公平比较','',
        '两组使用完全相同的目标、标签、权重和训练预算。基础组包含 18 项请求/曝光历史衍生特征；',
        '视频组额外增加 8 项视频属性/作者匹配衍生特征。这是 18/26 维模型输入，不等于同等数量的独立原始信息源。',
        '历史只取严格更早的曝光；不输入过去或当前反馈、用户快照或整月视频统计。原始 ID 不作树模型的有序数值输入。',
        '每目标固定 64 轮、最多 15 叶/深度 4，未用评估或校准标签选树数；概率校准只拟合 4/18，七任务都保持正斜率。',
        '与首轮 SASRec 相比，架构、特征和数据预算同时改变，不能将差异归因于架构优越。本轮两组之间才能检验新增视频属性的同预算增量。','',
        '## 4/19 加权结果','',
        '| 任务 | 阳性样本 | tab 基线 log loss | 统计组 raw | 统计组校准后 | 视频组 raw | 视频组校准后 | 视频组门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for key,label in zip(BINARY_LABELS,NAMES):
        b=r['baselines']['tab'][key];c=context['calibrated'][key];v=content['calibrated'][key]
        lines.append(f"| {label} | {v['sampled_positives']} | {b['logloss']:.6f} | {context['raw'][key]['logloss']:.6f} | {c['logloss']:.6f} | {content['raw'][key]['logloss']:.6f} | {v['logloss']:.6f} | {'通过' if content['gates'][key] else '未通过'} |")
    lines += ['', '| 任务 | tab AUC | 统计组 AUC | 视频组 AUC | tab Brier | 视频组校准 Brier | 实际发生率 | 视频组平均预测 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,label in zip(BINARY_LABELS,NAMES):
        b=r['baselines']['tab'][key];c=context['calibrated'][key];v=content['calibrated'][key]
        lines.append(f"| {label} | {auc_text(b['auc'])} | {auc_text(c['auc'])} | {auc_text(v['auc'])} | {b['brier']:.7f} | {v['brier']:.7f} | {v['prevalence']:.3%} | {v['mean_prediction']:.3%} |")
    lines += ['', '开发门槛要求：每任务至少 20 个阳性、AUC>0.5，校准后 log loss 和 Brier 均优于按 tab 发生率。',
        '基线仅从同一批训练目标估计，tab 平滑使用 100 个全局先验伪曝光。所有 PR-AUC、ECE、原始/校准均值、参数与各 tab 分层结果见 results.json。',
        '正斜率校准不增加理论排序信息；浮点饱和/评估裁剪可能造成极小的并列变化。校准对单项指标有害时也保留结果，未按评估结果选择是否校准。','',
        '## 视频属性增量的探索性区间','',
        '以下为「视频组 − 统计组」校准后 log loss；负值有利于视频组。按用户簇做 1,000 次 bootstrap。',
        '这些区间条件于当前已拟合模型、未做多重检验校正，而且属于已经看过时段的开发分析，不能当作确认性显著性。','',
        '| 任务 | 差值 | 95% 区间 |','|---|---:|---|']
    b=r['metadata_minus_context_logloss_bootstrap']
    for j,label in enumerate(NAMES): lines.append(f"| {label} | {b['point'][j]:.7f} | [{b['lower'][j]:.7f}, {b['upper'][j]:.7f}] |")
    lines += ['', '## 属性质量与结论边界','',
        f"完整视频表 {metadata['rows']:,} 个唯一 ID，源文件与索引 SHA-256 已验证。目录级时长缺失/无效 {metadata['missing']['duration']:,} 条、上传日期缺失/无效 {metadata['missing']['upload_day']:,} 条。",
        '目录缺失率不是曝光加权缺失率；具体目标样本每列缺失计数保存在 summary.json。未因属性缺失删除反馈行。','',
        '| 时段 | 元数据可用目标 | 未来上传屏蔽 | 上传日期缺失屏蔽 |','|---|---:|---:|---:|']
    for split,v in s.items():
        d=v['diagnostics'];lines.append(f"| {split} | {d['metadata_available']:,}/{v['rows']:,} | {d['future_upload_masked']:,} | {d['missing_upload_masked']:,} |")
    lines += ['', '视频组依赖作者、上传日期、物理时长和宽高在上传后不变的明确假设。发布表无快照时间，',
        '屏蔽未来上传并不能证明属性从未更新。因此保留不依赖该表的统计组，并限制视频组结论的适用条件。',
        '本轮是行为预测工程与开发证据，没有 Jev 增量、ParetoJev 优势或线上推荐收益证据。','',
        '## 复现和后续','',
        '两个模型组的原始/校准预测及全部指标在独立进程精确复现。未另做完整重训或跨硬件复现。',
        '源文件、数组、模型、预测均留存哈希；重型文件在外接盘，路径见 summary.json。',
        '后续需按未通过项决定是否继续完善行为基线；通过开发筛查也不能跳过后续时段验证。',
        '向重排器导出特征时，必须把同一 slate 的历史和请求时间冻结在共同决策时点，不能直接混用各曝光自己的未来历史。',
        '七原始目标与原六目标协议的映射仍须明确登记，再开展真实 Jev 和统一融合比较。','']
    (a.output/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
