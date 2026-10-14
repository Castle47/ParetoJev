#!/usr/bin/env python3
"""Report the frozen new-date evaluation only after exact audited replay."""
import argparse
from pathlib import Path
import shutil
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def report(source,output):
    r=read_json(source/'evaluation.json');replay=read_json(source/'replay.json');reg=read_json(source/'registration.json')
    if not replay['exact_predictions_and_metrics'] or not replay['all_raw_responses_audited_again']:raise ValueError('Independent exact replay and raw-response audit required')
    for name,key in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(source/name)!=replay[key]:raise ValueError('Replay evidence differs')
    if r['candidate_manifest_sha256']!=reg['candidate_manifest_sha256'] or r['response_audit_sha256']!=sha256_file(source/'common/response_audit.json'):raise ValueError('Candidate/score provenance differs')
    names=dict(is_click='点击',long_view='长看',is_like='点赞',is_follow='关注',is_comment='评论',is_forward='转发',is_hate='无负反馈比例')
    titles=dict(signal='同配方Jev增量',original_reference='相对原EMER参考',training_control='无Jev训练方法对照')
    support=r['support'];lines=['# 固定候选的新日期独立评价','',
        f'2022-04-23至25：{support["slates"]:,}组、{support["candidate_exposures"]:,}条候选曝光、{support["users"]:,}名用户。',
        '全部350,000项真实Jev判断先完成请求/原始响应/来源审计，再使用预先冻结的模型包计算指标。模型、用户路由和每折编码器未用本测试选模。',
        '六正向目标采用按原抽样权重计算的NDCG@3，无负反馈采用top3比例；表中为三个固定种子的均值。区间为固定模型预测的配对用户bootstrap95%区间。']
    for name,title in titles.items():
        p=r['pooled'][name];m=p['positive_mean'];lo,hi=m['confidence_interval']
        lines+=['',f'## {title}','',f'六目标均值{m["emer"]:.6f} → {m["emer_jev"]:.6f}，相对变化{100*m["absolute_gain"]/m["emer"]:+.2f}%；绝对变化区间[{lo:+.6f}, {hi:+.6f}]。','',
            '|目标|对照|实验|绝对变化|相对变化|95%条件区间|','|---|---:|---:|---:|---:|---|']
        for goal,row in p['objectives'].items():
            low,high=row['confidence_interval'];relative='—' if goal=='is_hate' else f'{100*row["relative_gain"]:+.2f}%'
            lines.append(f'|{names[goal]}|{row["emer"]:.6f}|{row["emer_jev"]:.6f}|{row["absolute_gain"]:+.6f}|{relative}|[{low:+.6f}, {high:+.6f}]|')
        lines+=['','|种子|均值绝对变化|','|---|---:|']
        for seed,row in p['per_seed'].items():lines.append(f'|{seed}|{(sum(row["emer_jev"][:6])-sum(row["emer"][:6]))/6:+.6f}|')
        lines+=['','|用户路由组|均值绝对变化|','|---|---:|']
        for route,row in r['by_route'][name].items():lines.append(f'|{route}|{row["positive_mean"]["absolute_gain"]:+.6f}|')
        if name in r['conditions']:
            lines+=['','|冻结条件|满足|','|---|---|']
            for key,value in r['conditions'][name].items():lines.append(f'|{key}|{"是" if value else "否"}|')
    lines+=['','## 支持数与解释边界','',f'两项要求的比较同时通过：{"是" if r["screen_passed"] else "否"}。不据结果扩样或调阈值。','',
        '|目标|正例曝光数|包含正例的列表数|','|---|---:|---:|']
    for i,name in enumerate(names):lines.append(f'|{names[name]}|{support["positive_exposures"][i]}|{support["positive_slates"][i]}|')
    lines+=['','逐日七项目标点估计和各路由组完整区间见evaluation.json。','']+r['limitations']
    lines+=['','模型清单SHA256：`'+r['candidate_manifest_sha256']+'`。原始评分审计、固定配置、输入预检及独立重放记录随本报告保存。']
    output.mkdir(parents=True,exist_ok=True)
    for name in ('evaluation.json','registration.json','input_preflight.json','replay.json'):shutil.copy2(source/name,output/name)
    shutil.copy2(source/'common/response_audit.json',output/'response_audit.json')
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.source,a.output)
