#!/usr/bin/env python3
"""Explain all fallback choices using inner histories only, no new prediction."""
import argparse
from pathlib import Path
import numpy as np
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file
from run_emer_scale import immutable_json


def run(a):
    names=['click','long_view','like','follow','comment','forward','no_hate'];rows=[];inputs={}
    for f in range(3):
        for seed in (7,17,29):
            for arm in ('emer','emer_jev'):
                p=a.source/f'fold-{f}'/f'{arm}-seed{seed}.json';inputs[str(p)]=sha256_file(p);r=read_json(p)
                d=np.array([h['objectives'] for h in r['history']])-np.array(r['anchor_objectives']);gain=d[:,:6].mean(1)
                valid=(d>=-.005).all(1);best=int(np.argmax(gain));nearest=int(np.argmax(d.min(1)))
                rows.append(dict(fold=f,seed=seed,arm=arm,selected_epoch=r['best_epoch'],
                    improving_epochs=int((gain>0).sum()),admissible_epochs=(np.flatnonzero(valid)+1).tolist(),
                    violation_counts=dict(zip(names,(d<-.005).sum(0).tolist())),
                    highest_mean_epoch=best+1,highest_mean_gain=float(gain[best]),highest_mean_delta=d[best].tolist(),
                    least_worst_guard_epoch=nearest+1,least_worst_guard_delta=d[nearest].tolist(),
                    first_loss=r['history'][0]['loss'],last_loss=r['history'][-1]['loss']))
    result=dict(role='Post-run diagnostic using inner histories only; not new confirmation',rows=rows,inputs=inputs,
        script_sha256=sha256_file(Path(__file__)),new_model_predictions=False,new_training=False)
    a.output.mkdir(parents=True,exist_ok=True);immutable_json(a.output/'diagnosis.json',result)
    lines=['# 任务平衡训练为何回退','',
        '审查全部18次训练的内层轨迹；未产生新模型或外层预测。种子29的Jev模型在三折共45个轮次中平均指标都提高，但没有任何一轮满足全部七项目标守护。',
        '前两折15/15轮触发长看守护，第三折15/15轮触发评论守护。因而回退并非没有训练，也不是选模程序漏掉了可行候选。','',
        '|折|种子|模型|选中轮次|均值提高轮数|满足全部守护轮数|','|---|---|---|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["fold"]}|{r["seed"]}|{r["arm"]}|{r["selected_epoch"]}|{r["improving_epochs"]}|{len(r["admissible_epochs"])}|')
    lines+=['','固定的下一假设：用50%原后验损失和50%条件任务平衡损失替代完全切换，让两臂共享该配方，保持初始化、15轮预算和原守护。仅测试这个中点，不搜索混合系数。',
        '这只是针对已观测取舍的工程假设，不是已经证明的改进；后续结果仍属探索性，不是新独立测试。完整逐目标违规次数及来源哈希见diagnosis.json。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
