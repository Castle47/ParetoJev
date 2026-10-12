# KuaiRand-27K：对齐样本后的行为模型对照

本轮优先补公平对照，未优化负反馈采样。所有结果仍是开发结果，不能证明 Jev 或 ParetoJev 的增量。

## 对齐了什么

- 复用冻结 v2 样本：训练 1,699,506 条、4 月 18 日校准 188,642 条、4 月 19 日评估 187,419 条。
- 相同七个原始标签、样本顺序、逆包含概率权重、概率校准方法、评价器和通过标准。
- 逐条恢复原采样 occurrence，校验用户/时间/标签/权重；保存源记录编号，排除同时间戳历史。
- 树模型复用冻结 checkpoint，并重新预测、核对全部指标；无需重复训练相同模型。
- 两种神经模型训练固定 3 轮，最终轮 checkpoint；不按校准/评估结果选轮次。

|模型|输入信息|训练预算|
|---|---|---|
|树模型（七个独立头）|26 个请求/历史统计/视频属性特征|每头固定 64 次 boosting|
|纯 MMoE|相同 26 个特征；训练集拟合标准化、缺失指示、场景 embedding|固定 3 轮|
|SASRec-style + MMoE|相同 26 个特征 + 用户 ID、候选 ID、50 条严格更早曝光视频 ID|固定 3 轮|

**严格相同信息的架构对照是树模型 vs 纯 MMoE。SASRec 版本多了序列和 ID 信息，
其比较检验整个增强方案，不能单独归因于架构。三个 epoch 与 64 次 boosting 不是相同计算预算。**

## 校准后的完整结果

|目标|树 AUC|MMoE AUC|SASRec+MMoE AUC|树 NLL|MMoE NLL|SASRec+MMoE NLL|
|---|---:|---:|---:|---:|---:|---:|
|is_click|0.756743|0.755097|0.714946|0.56286795|0.56489427|0.59929624|
|long_view|0.777706|0.776753|0.736444|0.47845801|0.47936879|0.51132003|
|is_like|0.768106|0.765202|0.781552|0.06925482|0.06952707|0.06658859|
|is_follow|0.744593|0.744276|0.766109|0.00684621|0.00683929|0.00665377|
|is_comment|0.718115|0.717880|0.704870|0.01657557|0.01655460|0.01669102|
|is_forward|0.677093|0.700545|0.735875|0.00672351|0.00667528|0.00658349|
|is_hate|0.574459|0.578838|0.567248|0.00381758|0.00379412|0.00379236|

|目标|场景基线 Brier|树 Brier|MMoE Brier|SASRec+MMoE Brier|
|---|---:|---:|---:|---:|
|is_click|0.215708220|0.192121584|0.192866351|0.206346383|
|long_view|0.180829692|0.159722032|0.160029707|0.170116354|
|is_like|0.014596427|0.014233531|0.014258877|0.013773539|
|is_follow|0.000897683|0.000897146|0.000897033|0.000893210|
|is_comment|0.002466790|0.002462913|0.002463055|0.002464089|
|is_forward|0.000852173|0.000851939|0.000851887|0.000851635|
|is_hate|0.000431378|0.000431779|0.000431436|0.000432308|

通过标准不变：至少 20 个评估正例、AUC>0.5，校准后 NLL 与 Brier 均低于训练集场景发生率基线。

- tree：6/7；未通过：is_hate。
- tabular_mmoe：6/7；未通过：is_hate。
- sasrec_mmoe：6/7；未通过：is_hate。

## 配对不确定性

校准 NLL 差值（前者减后者）；负值有利于前者。用户聚类 bootstrap 1,000 次，
95% 区间仅为探索性结果，条件于已拟合模型，未做多重比较校正或跨种子训练验证。

|比较|目标|差值|区间下限|区间上限|
|---|---|---:|---:|---:|
|tabular_mmoe_minus_tree|is_click|0.00202632|0.00115974|0.00290200|
|tabular_mmoe_minus_tree|long_view|0.00091078|0.00022731|0.00158967|
|tabular_mmoe_minus_tree|is_like|0.00027224|0.00002675|0.00048053|
|tabular_mmoe_minus_tree|is_follow|-0.00000692|-0.00007636|0.00005910|
|tabular_mmoe_minus_tree|is_comment|-0.00002097|-0.00013573|0.00008676|
|tabular_mmoe_minus_tree|is_forward|-0.00004823|-0.00015436|0.00003239|
|tabular_mmoe_minus_tree|is_hate|-0.00002346|-0.00011493|0.00005371|
|sasrec_mmoe_minus_tree|is_click|0.03642829|0.03387637|0.03906552|
|sasrec_mmoe_minus_tree|long_view|0.03286203|0.03058961|0.03528054|
|sasrec_mmoe_minus_tree|is_like|-0.00266623|-0.00371172|-0.00168071|
|sasrec_mmoe_minus_tree|is_follow|-0.00019245|-0.00040510|-0.00000589|
|sasrec_mmoe_minus_tree|is_comment|0.00011545|-0.00010268|0.00034440|
|sasrec_mmoe_minus_tree|is_forward|-0.00014002|-0.00030648|-0.00000347|
|sasrec_mmoe_minus_tree|is_hate|-0.00002521|-0.00012855|0.00008157|
|sasrec_mmoe_minus_tabular_mmoe|is_click|0.03440197|0.03209717|0.03687482|
|sasrec_mmoe_minus_tabular_mmoe|long_view|0.03195124|0.02985504|0.03413721|
|sasrec_mmoe_minus_tabular_mmoe|is_like|-0.00293848|-0.00396828|-0.00196940|
|sasrec_mmoe_minus_tabular_mmoe|is_follow|-0.00018553|-0.00037542|-0.00002963|
|sasrec_mmoe_minus_tabular_mmoe|is_comment|0.00013642|-0.00008448|0.00037750|
|sasrec_mmoe_minus_tabular_mmoe|is_forward|-0.00009179|-0.00018937|-0.00000656|
|sasrec_mmoe_minus_tabular_mmoe|is_hate|-0.00000175|-0.00010411|0.00008829|

## 执行与限制

- 同一 CPU 上重新加载后，三个模型的原始/校准预测、按目标/场景指标及 bootstrap 结果完全一致。
- 所有原始指标、ECE、平均预测概率、校准参数和训练损失见 `results.json`，未根据效果挑选 raw/calibrated。
- 只有一个预注册 seed 和固定训练预算，未证明各模型已充分优化或收敛。
- 4 月 19 日此前已用于开发，本轮不是新测试集确认；4 月 20 日以后未参与。
- item hash 仍有碰撞；历史是曝光序列；物理视频属性仍依赖 static-at-upload 假设。
- 相同评估样本仍只有 85 个 is_hate 正例，未做特殊重采样、改标签或降低门槛。
- 真实 Jev/ParetoJev 融合尚未训练/评估；本轮 Jev 请求为零，research target mode=false。

## 复现

```bash
.venv/bin/python scripts/run_kuairand_aligned.py \
  --data data/raw/kuairand-27k/behavior-v2 \
  --sequences data/raw/kuairand-27k/behavior-aligned-v3 \
  --tree data/raw/kuairand-27k/behavior-v2/run-v2 \
  --output data/raw/kuairand-27k/behavior-aligned-v3/run-v1 --replay
```

重新训练时去掉 `--replay` 并使用新的 output 目录。完整源数据保存在外接硬盘项目目录。

结论、限制和后续决策见 [FINDINGS.md](FINDINGS.md)。
