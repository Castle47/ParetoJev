# 实验执行计划

## Stage 0：已完成的 bootstrap

- [x] package/CLI/config 结构；
- [x] synthetic complementary signals；
- [x] behavior branch、fusion、reranker；
- [x] Pareto/uncertainty/distillation losses；
- [x] tests；
- [x] smoke target。

## Stage 1：MicroLens 可比较 Demo

当前代码状态：协议、决策打分 runner、校准、融合、统计检验和 target gate
均已实现；尚缺真实数据与 Jev 推理结果。

### 1.1 数据与复现

1. 下载 pairs/title；
2. 构造 200 用户、5 history、1 positive + 19 time-local negatives；默认严格使用
   occurrence-weighted upstream sampler，同时另存 uniform-unique 研究变体；
3. 验证每个用户恰好一个 positive；
4. 运行 popularity、TF-IDF title similarity、item-CF；
5. 对比公开 Demo 指标差异。

### 1.2 Decision score

至少生成：

- official Jev Choice(click)；
- official Jev Noul(click per candidate)；
- open JEV-27B title Noul；
- 有 GPU 时导入 JEV-27B-VL cover/image scores。

记录：

- resolved model version；
- input tokens / API cost；
- total/p50/p95 latency；
- cache hit；
- per-user entropy/confidence。

### 1.3 Hybrid

顺序：

1. RRF（不使用标签）；
2. rank-normalized 5-fold user-CV simplex；
3. Choice temperature / Noul Platt 的外层 user-CV 概率校准；
4. nested user-CV logistic stacker；
5. confidence-aware neural gate；
6. oracle per-user upper bound。

输出：AUC、HR@1、HR@5、NDCG@10、MRR、Brier、ECE、NLL、配对
bootstrap、top-k error complementarity 和每次运行的 provenance。

### 1.4 Phase A target

必须同时满足：

- hybrid AUC ≥ 0.735；
- HR@5 ≥ 0.600；
- NDCG@10 ≥ 0.510；
- 相比本仓库复现 item-CF 的 AUC paired-bootstrap 95% CI 下界 > 0。

若只达到公开数字但未超过本地复现 baseline，不能标记为 win；若 point estimate
提升但 CI 跨 0，标记为 inconclusive，而不是成功。

## Stage 2：KuaiRand 主实验

### 2.1 数据切分

4/08–4/21 standard 先按完整时间戳组切成三个互斥区间：

1. `behavior_train`：训练 SASRec+MMoE；
2. `behavior_validation`：early stopping 与 behavior diagnostics；
3. `reranker_train`：生成下游 reranker 的 out-of-period pXTR、Jev annotation 与 labels。

4/22–5/08 standard 再按时间切成 reranker validation/test；4/22–5/08 random 作为独立 random-exposure test。metadata 使用 captions/categories。禁止：

- 把同一时间戳拆到两个 split；
- 用 behavior model 对其自身训练事件的 in-sample 输出训练 reranker；
- 使用跨测试期聚合统计作为训练特征；
- 用 reranker validation/test label 拟合 behavior calibrator。

### 2.2 三类 protocol

1. **Hard candidate next-item**：便于与 Jev reranking 论文形式比较；
2. **Exposed-session pseudo-slate**：用于多目标 posterior/Pareto；
3. **Random-exposure track**：降低 exposure-selection 的单一偏差，独立报告。

### 2.3 Jev 采样策略

当前状态：采样器已经实现。`prepare-kuairand` 会写入严格早于当前曝光的 item
prior count 和 user history count；`sample-kuairand-jev` 负责生成可审计 manifest
与仅包含选中 slate 的 Jev JSONL。

不要一开始打满 100 万+样本。首轮每个 bucket 采样相同数量：

- cold / mid / warm item；
- history 1–3 / 4–10 / >10；
- standard / random exposure；
- positive-rich / sparse-positive / no-positive slate；
- Chinese original / translated English。

首轮 5k–20k slates 足以评估互补性和成本；只有 gate/分桶有明确增益才扩容。
默认 first probe 为 5k，joint cell 上限 250，单用户最多 2 个 slate；如果某些
cell 库存不足，报告 shortfall，而不是从高频 cell 静默补齐并破坏平衡。

### 2.4 物料化与 leakage gate

当前状态：已实现并测试。

1. `split-kuairand-log --write-prefixes` 产生 strict chronological CSV 和 cumulative prefix；
2. `train-kuairand-behavior` 在 `behavior_train` 上训练、在后续 `behavior_validation` 上 early-stop；
3. `score-kuairand-behavior` 对未参与 behavior fitting 的 reranker slates 生成正式 `SlateScoreTensor`；
4. `score-jsonl` 只对抽样 manifest 中的 slate 生成 Jev 输出；
5. `score-kuairand-behavior-prior` 仅提供严格 pre-slate 的 CPU bootstrap signal；
6. `materialize-kuairand` 对 slate/candidate/objective 做强一致性校验；
7. candidate feature transform 只在训练 split 拟合，后续 split 必须复用；
8. 默认排除 pseudo-slate sequence position，避免直接编码 logged production order；
9. validation bootstrap 可通过更早的 prepared slates 初始化，但 prefix 必须严格早于当前 split。

首个真实物料 gate：

- Jev JSONL 行数与 manifest 一致；
- 0 个 candidate-order/objective-order mismatch；
- final NPZ 可被 `PrecomputedSlateDataset` 重载；
- 训练与验证 feature spec hash 一致；
- 全部 feature 都是 exposure 前可用或静态字段。
- behavior checkpoint 与 scoring report 记录 item/history OOV、Brier、macro AUC/AP、throughput 和版本 provenance。

### 2.5 Baselines

- SASRec；
- SASRec + MMoE；
- PLE（第二轮）；
- fixed weighted sum；
- EMER-like RAS + Set Transformer + Advantage Evaluator；
- CMR-like preference-conditioned reranker；
- Jev Single Choice；
- Jev MultiChoice；
- Jev MultiNoul；
- behavior + Jev linear；
- full ParetoJev。

### 2.6 Ablations

- no Jev；
- no behavior；
- no confidence/entropy；
- fixed gate；
- no P-RAS；
- weighted sum vs Tchebycheff；
- no uncertainty；
- no Advantage Evaluator；
- Chinese vs translated English；
- Choice vs Noul；
- teacher vs student。

## Stage 3：Student

候选结构：

- multilingual encoder 100M–400M；
- item text embedding offline；
- incremental history state；
- multi-objective heads；
- KL + hard-label + calibration mixed training；
- INT8 first, INT4 only after quality check。

评估：

- uplift retention；
- all/cold/warm buckets；
- p50/p95/p99；
- QPS and memory；
- calibration drift。

## Stage 4：论文式报告

必须包含：

- protocol limitations；
- upstream self-report vs reproduced values；
- cost/latency；
- confidence intervals；
- failures and negative results；
- prompt examples and hashes；
- reproducibility checklist。
