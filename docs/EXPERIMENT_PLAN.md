# 实验执行计划

## Stage 0：已完成的 bootstrap

- [x] package/CLI/config 结构；
- [x] synthetic complementary signals；
- [x] behavior branch、fusion、reranker；
- [x] Pareto/uncertainty/distillation losses；
- [x] tests；
- [x] smoke target。

## Stage 1：MicroLens 可比较 Demo

### 1.1 数据与复现

1. 下载 pairs/title；
2. 构造 200 用户、5 history、1 positive + 19 time-local negatives；
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

1. rank-normalized linear blend；
2. 5-fold user CV alpha/simplex；
3. logistic/calibrated blend；
4. confidence gate；
5. oracle per-user upper bound。

输出：AUC、HR@1、HR@5、NDCG@10、MRR 和配对 bootstrap。

## Stage 2：KuaiRand 主实验

### 2.1 数据切分

- train：4/08–4/21 standard；
- validation/test：4/22–5/08 standard 按时间再切；
- random test：4/22–5/08 random；
- metadata：captions/categories；
- 禁止使用跨测试期聚合统计作为训练特征。

### 2.2 三类 protocol

1. **Hard candidate next-item**：便于与 Jev reranking 论文形式比较；
2. **Exposed-session pseudo-slate**：用于多目标 posterior/Pareto；
3. **Random-exposure track**：降低 exposure-selection 的单一偏差，独立报告。

### 2.3 Jev 采样策略

不要一开始打满 100 万+样本。首轮每个 bucket 采样相同数量：

- cold / mid / warm item；
- history 1–3 / 4–10 / >10；
- standard / random exposure；
- positive-rich / no-positive slate；
- Chinese original / translated English。

首轮 5k–20k slates 足以评估互补性和成本；只有 gate/分桶有明确增益才扩容。

### 2.4 Baselines

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

### 2.5 Ablations

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
