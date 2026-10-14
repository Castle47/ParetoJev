#!/usr/bin/env python3
"""Render all registered development diagnostics without selecting a new winner."""
import argparse
import shutil
from pathlib import Path
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def report(source, output):
    d=read_json(source/'diagnostics.json');replay=read_json(source/'replay.json')
    if sha256_file(source/'diagnostics.json')!=replay['diagnostic_sha256']:
        raise ValueError('Replayed evidence differs')
    names=dict(is_click='点击',long_view='长看',is_like='点赞',is_follow='关注',is_comment='评论',is_forward='转发',is_hate='无负反馈')
    lines=['# EMER＋Jev 开发集诊断：尚无稳定长看增益证据','',
        '2026-10-14。全部24个既有checkpoint重新推理，并逐项重算对应训练收据中的验证指标。'
        '无新增训练、Jev请求或付费GPU；仅用6000组开发验证数据。原先的主比较与选模结果不改写。','',
        '## 主要判断','',
        '1. 原主比较长看+2.00%对配置选择敏感。四组同超参数比较中，三组长看点估计为负；'
        '唯一正向组的差值区间跨零。没有发现同配置下稳定可信的长看提升。',
        '2. 当前选模最大化六正向目标的平均NDCG，而项目主指标是长看。'
        '这两者可能发生权衡；本次原规则得到的结果是真实的，但不能被解读为Jev在各种配置下都改善长看。',
        '3. Jev单独按长看排序弱于行为预测，不能因为存在纠错案例就推断它能带来净收益。'
        '负反馈识别出现更积极的开发迹象，仍不等于融合后或线上收益。',
        '4. 固定分组均未解决跨种子不稳定问题。暂不启动新的5万组独立测试打分。','',
        '所有区间均未校正选模及多重诊断比较，仅作探索描述；不能拿本报告重新挑一个有利配置作为新的确认结果。','',
        '## 四组同配置比较','',
        '同学习率、同先验系数；各模型保留原规则选择的epoch，所以不是同epoch对照。'
        '四组全部展示，三种子不筛选。','',
        '|学习率|先验系数|长看相对变化|长看绝对差95%区间|长看提升种子数|六正向均值绝对变化|均值差95%区间|',
        '|---|---:|---:|---|---:|---:|---|']
    def ci(row):
        low,high=row['confidence_interval'];return f'[{low:+.6f}, {high:+.6f}]'
    for m in d['matched']:
        r=m['results'];g=r['objectives']['long_view'];mean=r['positive_mean']
        positive=sum(x['emer_jev'][1]>x['emer'][1] for x in r['per_seed'].values())
        lines.append(f'|{m["learning_rate"]}|{m["prior_coefficient"]}|{g["relative_gain"]*100:+.2f}%|{ci(g)}|{positive}/3|{mean["absolute_gain"]:+.6f}|{ci(mean)}|')
    lines+=['','### 为什么原主比较平均长看能升2%？','',
        '原选模下，seed7的EMER选择学习率0.001、先验系数1.0，长看为0.394230；'
        'EMER＋Jev选择0.0003、0.25，长看为0.422263。'
        '然而，既有EMER在同一个0.0003、0.25配置下已经达到0.422067，'
        '对应Jev差值仅+0.000196。seed17/29原主比较长看均下降。', '',
        '这解释了结果对选模的敏感性，不否认原来的等预算选择流程，也不是把事后选出的新基线替换原主比较。'
        'seed7基线被选中的六目标均值只比其0.0003/0.25配置高约0.000984，长看却低约0.027837。','',
        '### 四组七项目标完整变化','',
        '|配置序号|点击%|长看%|点赞%|关注%|评论%|转发%|无负反馈绝对变化|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in d['matched']:
        goals=m['results']['objectives'];values=[f'{goals[n]["relative_gain"]*100:+.2f}' for n in list(names)[:6]]
        lines.append('|'+str(m['trial'])+'|'+'|'.join(values)+f'|{goals["is_hate"]["absolute_gain"]:+.8f}|')
    lines+=['','0–3分别对应上表四行。完整每种子、每目标区间见diagnostics.json。','',
        '## Jev信号本身是否补足行为预测','',
        '以下为每组列表内正负候选对AUC，再按抽样权重求平均；平分计0.5，'
        '仅在该目标同时有正负候选的列表上计算。负反馈转为无负反馈方向。'
        'Jev是列表内Choice分布，不是校准后的点击率或转化率。','',
        '|目标|行为预测AUC|Jev AUC|Jev−行为 AUC 95%区间|行为错/Jev对比例|行为对/Jev错比例|有效列表|',
        '|---|---:|---:|---|---:|---:|---:|']
    for name,r in d['signal'].items():
        auc=r['within_slate_auc'];co=r['complementarity'];low,high=auc['difference_ci']
        lines.append(f'|{names[name]}|{auc["values"][0]:.4f}|{auc["values"][1]:.4f}|[{low:+.4f}, {high:+.4f}]|'
            f'{co["behavior_wrong_jev_right"]*100:.2f}%|{co["behavior_right_jev_wrong"]*100:.2f}%|{r["eligible_pair_slates"]}|')
    lines+=['','纠错/损坏两列是全部有效候选对中的严格排序分歧比例，先列表内平均，再按列表抽样权重汇总；'
        '不是“行为错例中被修好多少”的条件概率，且平分不计入这两列。'
        '真实反馈只用于事后诊断，不作为在线选择哪个模型正确的依据。', '',
        '长看有19.84%的候选对是行为错而Jev对，但25.73%恰好相反；'
        'Jev长看AUC约0.5354，行为预测约0.5928。'
        '弱信号可能含有可用的互补信息，但这些统计没有证明模型能够可靠识别并利用它。', '',
        '负反馈AUC约0.4959→0.5532，差值区间为正；评论/转发AUC点估计上升，但差值区间跨零。'
        '负反馈样本稀疏且本报告包含多重诊断，不能据此直接改写主目标或宣称上线成功。','',
        '## 固定用户与候选分组','',
        '历史阈值7218来自训练样本历史曝光总数中位数，并非模型输入序列长度；'
        '一个用户可能在不同时间跨组。零历史计数仅表示当前数据范围内没有更早曝光，不等同于视频首次发布。'
        '第三个维度是行为与Jev长看各自top3的交集数量。阈值均在运行诊断前记录。','',
        '|分组|列表数|用户数|原选中模型长看相对变化|绝对差95%区间|长看提升种子数|',
        '|---|---:|---:|---:|---|---:|']
    group_names=['历史曝光<7218','历史曝光≥7218','至少5个候选零历史计数','少于5个候选零历史计数','长看top3交集0或1','长看top3交集2或3']
    for label,r in zip(group_names,d['groups'].values()):
        goal=r['results']['objectives']['long_view'];positive=sum(x['emer_jev'][1]>x['emer'][1] for x in r['results']['per_seed'].values())
        lines.append(f'|{label}|{r["slates"]}|{r["users"]}|{goal["relative_gain"]*100:+.2f}%|{ci(goal)}|{positive}/3|')
    lines+=['','这六组是三个不同划分维度，不能相加当成额外样本。'
        '所有分组中seed29长看均下降；没有找到可直接用于上线门控的、跨种子稳定的获益人群。'
        '少曝光候选较多的组虽有较好的均值结果，但不能仅凭某组区间为正、另一组跨零就断言组间差异成立。','',
        '## 决策与下一轮边界','',
        '当前停止在开发诊断，不花费额外预算打开独立测试。下一轮应先把长看守护条件与选模规则协调好，'
        '再在训练/验证范围内检验有行为基线保底的有限Jev修正。'
        '这是待验证的开发假设，尚未实现或训练，也不能通过事后改门槛把本轮宣布为成功。', '',
        '已有相同epoch的1/5/10/20/30轮验证曲线差值也完整保留在epoch_receipts中；'
        '只比较已有训练收据，不重新选择epoch，不据此断言导致不稳定的唯一原因。', '',
        '## 验证与复现','',
        '- 全部24个模型的七项验证指标与原始训练收据重算一致。',
        '- 全部4组对比、7个信号审计及6个分组统计在独立进程从哈希锁定的预测缓存精确重放。',
        '- 原main evaluation及其预测哈希已核对；主报告、模型和注册配置保持原值。',
        '- 六项已有信号/统计协议测试通过；未引入新训练或测试数据。', '',
        '运行：`scripts/diagnose_emer_native.py --root <scale-dir> --output <diagnostics-dir>`；'
        '重放同命令加`--replay`；渲染本报告用`scripts/report_emer_native_diagnostic.py`。', '',
        '证据：[完整统计](diagnostics.json)、[诊断注册](registration.json)、[重放收据](replay.json)。']
    output.mkdir(parents=True,exist_ok=True)
    for name in ('diagnostics.json','registration.json','replay.json'):
        shutil.copy2(source/name,output/name)
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.source,a.output)
