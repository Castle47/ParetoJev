#!/usr/bin/env python3
"""Write an evidence-based report after confirmation and independent audit."""
import argparse
import shutil
from pathlib import Path

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def report(a):
    evaluation=read_json(a.data/'evaluation.json');audit=read_json(a.data/'audit.json')
    replay=read_json(a.data/'replay.json');selection=read_json(a.data/'selection.json')
    if not replay['exact_predictions_and_metrics']:raise ValueError('Replay incomplete')
    results=evaluation['results'];a.output.mkdir(parents=True,exist_ok=True)
    outcome='本轮离线效果条件全部通过' if results['all_effect_gates'] else '本轮未达到预登记的完整效果条件'
    lines=['# EMER＋Jev 增量实验：'+outcome,'',
        '**日期：2026-10-13。** 两组仅改变 Jev 输入是否可用；同一行为模型、候选、监督、网络尺寸与调参预算。',
        '这是独立实现的 EMER 公开数据适配，不能称为快手工业系统完整复现。',
        '','## 新日期确认结果','',
        '主指标 long_view NDCG@3。以下为三个固定训练种子的指标均值；区间为 2,000 次配对用户 bootstrap，条件于已训练模型。',
        '','|目标|EMER|EMER＋Jev|绝对变化|相对变化|绝对变化 95% 区间|',
        '|---|---:|---:|---:|---:|---|']
    for name,row in results['objectives'].items():
        lo,hi=row['confidence_interval'];relative=f'{row["relative_gain"]*100:+.2f}%' if row['relative_gain'] is not None else '未定义'
        lines.append(f'|{name}|{row["emer"]:.6f}|{row["emer_jev"]:.6f}|{row["absolute_gain"]:+.6f}|{relative}|[{lo:+.6f}, {hi:+.6f}]|')
    lines+=['','is_hate 列是“无负反馈比例”，越高越好，其余为 NDCG@3。','',
        '|验收条件|结果|','|---|---|']
    lines += [f'|{key}|{"通过" if value else "未通过"}|' for key,value in results['gates'].items()]
    lines+=['','六正向目标均值：'+str(results['positive_mean']),'',
        '|随机种子|EMER 长看|EMER＋Jev 长看|差值|','|---|---:|---:|---:|']
    for seed,row in results['per_seed'].items():
        b,f=row['emer'][1],row['emer_jev'][1];lines.append(f'|{seed}|{b:.6f}|{f:.6f}|{f-b:+.6f}|')
    lines+=['','## 模型与数据','',
        '独立实现候选内 NormalizedRanks、候选比较 Transformer、正反馈数量减负反馈的后验排序损失、七行为先验排序损失，以及上一 epoch/当前模型 Mean@3 比值权重。Jev 仅作为输入，未增加 Jev 监督标签或教师损失。',
        '','网络输入 109 维：81 维共同输入＋28 维 Jev 分数/排名/熵/置信度。基线的 Jev 块全部置零；没有运行时偏好 FiLM。',
        '','训练 1,000 组（4 月 20 日），验证 400 组（4 月 21 日上午），新测试 600 组（4 月 22 日）；每组十个候选。旧核心实验测试不参与本轮开发。',
        '','两组各三个种子、四配置、每配置 30 轮，共 24 次训练。仅验证集选择 checkpoint 和配置，初始化 checkpoint 也可选。',
        '','修复一天训练中恒定星期字段的标准化尺度；旧验证最大数值约 678,448，新数值约 13.6。这是两组共同修复，不归因于 Jev。',
        '','|种子|模型|参数量|选中 epoch|学习率|先验系数|','|---|---|---:|---:|---:|---:|']
    for seed,arms in selection['selected'].items():
        for arm,row in arms.items():lines.append(f'|{seed}|{arm}|{row["parameters"]}|{row["best_epoch"]}|{row["lr"]}|{row["prior_coefficient"]}|')
    lines+=['','## 计算开销','',
        '真实 Jev 判断数量：'+str(audit['real_jev_decisions'])+'；HTTP 尝试：'+str(audit['http_attempts'])+'。模型为固定 revision 的 autotrust/JEV-27B，开发分数复用旧真实缓存，新测试重新采集。',
        '','|本地缓存分数就绪的排序模块|p50 ms|p95 ms|p99 ms|','|---|---:|---:|---:|']
    for arm,row in evaluation['latency']['milliseconds'].items():
        lines.append(f'|{arm}|{row["p50"]:.3f}|{row["p95"]:.3f}|{row["p99"]:.3f}|')
    lines+=['','本地 CPU 四线程、10 候选、batch 1、50 次预热、500 次交替测量。从准备好的数值张量到分数/排序，不包含特征获取和 Jev 生产。',
        '','七次串行 HTTP 累计耗时（含重试尝试网络耗时，排除请求间限速/退避等待）：'+str(audit['serial_seven_http_seconds_excluding_pacing'])+' 秒。',
        '','服务报告的单次推理耗时分位数：'+str(audit['service_inference_seconds'])+' 秒。两者不能当作目标硬件的完整推荐 RT。',
        '','**当前没有证明生产低延迟或线上业务收益。** 未指定部署硬件/并发/新增 RT 预算，未完成学生蒸馏，也没有线上 A/B。',
        '','## 审计与边界','',
        f'全部 {audit["source_candidates"]} 条来源候选、标签、共同时间和历史重算一致；{audit["real_jev_decisions"]} 次原始 Jev 响应与张量一致；六个选中 checkpoint 通过验证指标重算。',
        '','基线 checkpoint 对全部 Jev 通道变动不变；两组推理均不使用标签。预测和 2,000 次 bootstrap 指标在独立进程精确重放。',
        '','已曝光伪候选组不包含原始未曝光召回项；逆包含权重只修正本次分层抽样。没有 p_watchtime，未实现 IPUT，也不拿视频总时长替代观看时长。内容属性与文本快照时点仍有局限。',
        '','结论限定于当前采样、冻结行为模型、文本 Choice 信号和训练预算。公开数据离线增益不能等同于业务因果收益。',
        '','协议见 [实验设计](../../docs/EMER_JEV_EXPERIMENT.md)；配置见 [注册配置](../../configs/emer_jev.json)。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    for name in ('evaluation.json','audit.json','replay.json','preparation.json','selection.json','development_registration.json','preparation_registration.json','feature_spec.json'):
        shutil.copyfile(a.data/name,a.output/name)
    atomic_write_json(a.output/'evidence.json',dict(files={name:sha256_file(a.data/name) for name in
        ('predictions.npz','test.npz','raw_slates.npz','selection.json','evaluation.json','audit.json','replay.json')},
        report_sha256=sha256_file(a.output/'README.md')))
    print(outcome,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    report(p.parse_args())
