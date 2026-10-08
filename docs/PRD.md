# ParetoJev 需求文档（PRD）

**版本：** 0.1  
**状态：** 已进入执行；Phase 0 工程底座完成  
**更新时间：** 2026-10-08

## 1. 项目目标

构建并验证一套“Jev-style Decision Model × 推荐重排”的方法：在传统用户行为模型基础上，引入内容语义决策概率、置信度和不确定性，通过显式 Pareto 优化完成可控的多目标重排，并将最终能力蒸馏到可线上部署的小模型。

项目不是要证明“大模型全面替代推荐模型”，而是回答四个可检验问题：

1. 决策模型在冷启动、短历史和语义依赖更强的候选上，是否能补充行为模型？
2. 多个原子决策问题是否优于一个模糊的综合 Choice？
3. 偏好条件化的 Pareto 重排是否优于固定加权和、EMER-like 和 CMR-like 基线？
4. 决策模型增益能否以可接受的损失蒸馏到低延迟学生模型？

## 2. 成功定义

项目采用分阶段成功定义，不允许仅凭单个平均指标宣称成功。

### Phase A：可比较 Demo

在 MicroLens-100K 的公开 200-user / 20-candidate 协议上：

- 公平复现候选构造和 item-CF；
- 使用 user-level cross-validation 选择融合权重；
- 目标：AUC ≥ 0.735、HR@5 ≥ 0.600、NDCG@10 ≥ 0.510；
- 与 item-CF、Jev-only、标题相似度分别做配对 bootstrap。

### Phase B：主研究结果

在 KuaiRand-Pure 上：

- 对标准曝光、随机曝光、冷/暖 item、短/长历史分别报告；
- Pareto hypervolume 相对 strongest baseline 提升 ≥ 2%；
- cold-item NDCG 相对提升 ≥ 3%；
- worst-objective regret ≤ 0.005；
- 控制权重与实现指标的 Spearman 平均相关 ≥ 0.60；
- 至少一个主要增益的 95% user-level bootstrap CI 不跨 0。

### Phase C：工程可用性

- Student 保留 ≥ 90% 的 teacher uplift；
- 在声明硬件、K=50、固定 batch 条件下 p95 ≤ 30 ms；
- 所有 Jev 请求可恢复、去重、追踪版本与成本；
- 数据划分、负样本、校准和调参均无测试泄漏。

## 3. 用户与使用场景

### 研究人员

需要快速复现现有 Jev reranking 结果，并进一步测试多目标、冷启动和蒸馏假设。

### 推荐系统工程师

需要看到可插入现有级联链路的模块，而不是一个无法满足吞吐量的端到端 LLM 排序方案。

### 论文/开源评审者

需要清晰区分：

- 公开上游结果 vs 本项目实际结果；
- true request slate vs exposed-session pseudo-slate；
- Choice 相对概率 vs 绝对 pXTR；
- 离线 teacher 成本 vs 在线 student 延迟。

## 4. 功能范围

### 必须实现

- MicroLens 官方文件下载、候选构造、item-CF/title baseline、CV hybrid；
- KuaiRand 日志校验、session 划分、固定宽度 pseudo-slate、文本 metadata 合并；
- official TypeSafe Jev 与 open JEV HTTP client；
- Choice 和 Noul 两种打分；
- SQLite request cache；
- SASRec + MMoE behavior branch；
- confidence-aware fusion gate；
- preference-conditioned set reranker；
- Pareto dominance loss、Tchebycheff、uncertainty、distillation；
- AUC/HR/NDCG/MRR、calibration、hypervolume、regret、controllability；
- target checker、测试、README、执行日志。

### 第二阶段实现

- Amazon Reviews 2023 hard-candidate replication；
- 完整 CMR/EMER/UAME baseline reproduction；
- multilingual lightweight student；
- GPU serving benchmark；
- demo dashboard。

### 非目标

- 直接让 Jev 在百万 item 上召回；
- 把 KuaiRand pseudo-slate 描述成生产原始 request slate；
- 在没有 propensity 验证的情况下宣称完全无偏 OPE；
- 用测试集选择融合权重或 prompt；
- 将“schema 内输出”宣传为“模型不会判断错”。

## 5. 数据需求

### MicroLens-100K

必需：

- `MicroLens-100k_pairs.csv`
- `MicroLens-100k_title_en.csv`

多模态轨道增加：

- `MicroLens-100k_covers.zip`

### KuaiRand-Pure

必需：

- 三份 Pure 日志；
- caption 与 category supplementary files。

原则：原始数据不提交 Git；只提交下载说明、hash/manifest 和派生小型指标文件。

## 6. 模型需求

### Behavior branch

输入：用户历史 item IDs、候选 IDs、可选上下文。  
输出：每个候选各目标 logits/probabilities。

### Decision branch

输入：最近历史文本、反馈、当前上下文、候选文本。  
输出：candidate × objective 概率、confidence、entropy。

### Fusion

Gate 特征至少包括：

- behavior p；
- Jev p；
- disagreement；
- entropy；
- confidence；
- item popularity；
- history length。

### Pareto reranker

- slate-aware；
- 接收运行时偏好向量；
- 输出 shared ranking score、objective utility logits、uncertainty；
- 支持固定 weighted sum baseline 与 augmented Tchebycheff 主方法。

## 7. 实验约束

- Jev 最终表必须 pin versioned model ID；
- Choice 概率先作为 slate 内相对分布，不直接叫 pCTR；
- Noul confidence 若由 entropy 推导，必须标为 proxy；
- 所有概率在下游数据上重新校准；
- prompt、翻译、校准器和融合权重只能在 train/validation 上开发；
- 关键结果至少 3 个 seed 或 user bootstrap；
- 负目标（hate）必须在数据、utility transformation 和指标中保持一致方向。

## 8. 交付物

- 可安装 Python package；
- CLI 与配置文件；
- 数据预处理与 Jev scoring pipeline；
- 单元测试和 CI；
- target-mode 报告；
- 每阶段可复现 metrics JSON；
- 论文式实验报告；
- 最终 student checkpoint 与 serving benchmark。

## 9. 当前已完成

- 工程结构与 package；
- MicroLens/KuaiRand 数据处理代码；
- Jev client、prompt、parser、cache；
- behavior/fusion/reranker/loss/metrics；
- 14 项测试全部通过；
- synthetic smoke target 通过。

尚未完成的核心工作是获取真实数据、配置 decision endpoint、跑 Phase A/B 实验并基于失败模式迭代。
