# 实验执行计划

## 当前执行优先级（2026-10-12 用户最新指示）

暂停行为模型调优和负反馈专项优化，直接验证核心假设。固定 v2 行为树模型，
对照 EMER-like、CMR-like、ParetoJev 的有／无 Jev 六组，另有行为加权和控制。
真实 2,000 组候选已冻结；7 个原始目标、不修改门槛、分层抽样后加权评估。
三种子共 9 个无 Jev 检查点已训练，14,000 次真实 Choice 打分正在进行；
完成后由自动驱动继续其余训练、统一评估和精确复算。效果结论尚未产生。
详见 `reports/kuairand-core-2026-10-12/STATUS.md` 与 `REGISTRATION.md`。
下面的历史计划保留作记录；不得把进一步 SASRec 调优重新设为核心验证的阻塞项。

## Stage 0：已完成的 bootstrap

- [x] package/CLI/config 结构；
- [x] synthetic complementary signals；
- [x] behavior branch、fusion、reranker；
- [x] Pareto/uncertainty/distillation losses；
- [x] tests；
- [x] smoke target。

## Stage 1：MicroLens 可比较 Demo

当前状态：官方 MicroLens 数据和两批各 200 名用户的真实 AutoTrust JEV-27B
打分已完成。原始 Phase A 总目标尚未全部达标；新评价用户上的 Jev tie-breaking
通过两项预注册 AUC 检验，但仅支持有限的增量信号结论。接下来的 SASRec 与
MiniLM 对照将这 400 名已看过结果的用户视为开发集，独立保留 1,000 名用户
选择行为模型 checkpoint。该开发阶段现已完成；随后固定 700 名新评价用户的
20% Jev 融合确认通过 1/2 项 AUC 要求：优于 SASRec，但未确认优于 MiniLM
融合。HR/NDCG 增益仍不确定。详见各阶段报告，不再把真实数据列为当前缺失输入。

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

当前主数据集已由用户明确指定为 **KuaiRand-27K**。Pure 不作默认替代。
获取与字段审计入口见 `scripts/prepare_kuairand_27k.py`。2026-10-11 已在外接盘
完成全量获取和 323,464,444 条日志审计，存储阻塞已解除。首轮真实 27K 多目标行为模型已训练：全早期日志提供历史，抽取 426,832 条训练
目标、197,078 条验证目标，预测七个原始反馈；50 历史在验证样本中覆盖 99.63%。
不加快照画像的主模型开发筛查通过 6/7，转发未通过，且仅 3/7 任务的 log loss
优于按 tab 发生率基线。加入 30 个用户画像字段的敏感性组通过 7/7 最低筛查，
但画像快照时点未知，不能替代主证据。两组最佳 epoch 均为 1，继续训练均过拟合。
详见 `reports/kuairand-27k-behavior-pilot-2026-10-11/`；原研究指标仍未达成。

第二轮 v2 已完成：1,699,506 个训练目标；4/18 的 188,642 个目标单独校准，
4/19 的 187,419 个目标评估。新增七个独立树模型头，18 项请求/历史统计特征，
另有增加 8 项视频属性衍生特征的对照；本轮未重训 SASRec。两组均在 6/7 目标
上同时胜过按 tab 发生率的 log loss/Brier，负反馈仍未通过。视频属性组的点击/
有效播放 AUC 0.7567、长看 0.7777；新增属性的七项 log loss 点估计均更低，
但探索性用户 bootstrap 区间仅点击、点赞、负反馈三项不跨零。
4/19 是已看过的开发期，不能算独立研究胜利；静态视频属性的时点假设仍须保留。
详见 `reports/kuairand-27k-behavior-v2-2026-10-11/`。

2026-10-12 按用户纠正调整优先级：先完成同样本、同权重、同校准与同评估的
行为模型对照，暂停稀疏负反馈专项优化。沿用 v2 的 170 万训练目标和冻结树模型，
新增相同 26 项信息的 MMoE，以及额外使用用户/候选/50 历史视频 ID 的 SASRec + MMoE。
前者才是严格相同信息的架构对照；后者评价序列/ID 增强后的整体方案。
固定 3 轮训练，不按校准或评估结果选择 checkpoint。详见
`reports/kuairand-27k-aligned-2026-10-12/REGISTRATION.md`。

本次对齐比较已完成：树模型与纯 MMoE 接近；固定三轮的 SASRec + MMoE
在点击/长看更差、点赞/关注/转发更好，三个模型均 6/7 项通过。不能宣称
SASRec 全面更优，也不能把较差结果直接归因于序列编码器。若继续改进神经
行为模型，先在训练窗口内部登记时间验证和 ID/序列分支对照，保留独立校准期，
再决定用于 Jev 增量实验的冻结行为模型。详见该轮 `FINDINGS.md`。

负反馈仍保留原有 856 个训练、85 个评估阳性，不增加采样或修改通过条件。
全量训练 48,688 个、校准/评估 5,597/4,471 个阳性索引留待后续独立登记；
本轮不追求七项全部过线，也不依据评估结果改标签或拟合校准器。
行为对照没有外部输入阻塞，Jev 调用保持零。
七原始目标与原六目标重排协议的映射需另行注册，不将两个边缘概率取最大
冒充评论/转发联合概率。导出 pXTR 时必须把一整个 slate 的历史/请求时点固定
到共同决策时间，不能直接复用逐曝光各自时点的特征。

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

正式 behavior checkpoint 可用后，优先采用 behavior-informed ordering：先保持上述
结构 cell，再在 cell 内根据三类信号混合排序：

- 多目标 Bernoulli entropy；
- 每个目标 top-1/top-2 margin 的不确定性；
- 不同目标 argmax candidate 的 winner disagreement。

默认 `behavior_priority_strength=0.5`，即 active priority 与随机顺序各保留一部分；
`1.0` 只用于 active-learning ablation。该 manifest 是 teacher annotation 的训练子集，
不能直接作为总体评估集。由于 feedback-density cell 使用了 outcome label，最终指标
必须在完整固定 validation/test protocol 上计算，或显式使用抽样权重。

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
- EMER-like project variant：raw-signal Set Transformer + FiLM + shared score + training-only loss/EMA feedback；
- CMR-like project variant：raw-signal Set Transformer + FiLM + preference-weighted objective heads；
- Jev Single Choice；
- Jev MultiChoice；
- Jev MultiNoul；
- behavior + Jev linear；
- full ParetoJev。

### 2.5.1 已实现的统一 baseline harness

`compare-baselines` 在 `baseline-harness-v1` 上独立训练 `emer_like`、
`cmr_like`、`paretojev`。这两个 `-like` 名称代表项目变体，不能写成原论文复现。
精确定义、loss 系数、manifest 与命令见 [BASELINE_HARNESS.md](BASELINE_HARNESS.md)。

注册先于训练：固定三个 chronological NPZ 的 hash、上游 evidence、方法、seeds、
模型/优化器预算、preference grid、target 配置、source hash 与环境。每种方法使用
相同原始信号、数据顺序、epochs、batch 和优化器设置；参数量及耗时分别报告，
不宣称参数量或 FLOPs 相同。所有方法只用 observed labels，不用合成 oracle。

checkpoint 由 validation 的固定 preference grid 平均加权 utility 选择，允许
选择 epoch zero，相等时取最早者；配置 epochs 仍全部执行。test 不参与选择。
冷 item cutoff 只从 train popularity 拟合。两个比较对象预先固定为 EMER-like 和
CMR-like，主 namespace 保留 EMER-like；不得用 test 选择“最佳 baseline”或修改阈值。

执行顺序：

1. 合成 fixture 验证数据协议、训练、重载、审计、resume 和 target 拒绝真实声明；
2. 取得真实三份 feature-complete NPZ 与四份上游 evidence 后，先检查 provenance；
3. 在验证集确定正式预算、目标顺序、preference grid 和 seeds，注册新目录；
4. 每 seed 保存三种 checkpoint、training history、test 预测和 identity；
5. 逐 seed 报告，另报所有预注册 seed 的算术平均；任一 seed 未定义则该均值为 null；
6. 同时检查对两种固定 baseline 的 hypervolume ≥2%、cold NDCG ≥3%、regret ≤0.005，
   并检查 controllability ≥0.60；有限数值未过阈值是 failed，输入不足是 not_evaluable。

均值不是显著性证据。现有 paired user bootstrap 对固定 checkpoint 的 objective mean
和 cold NDCG 差值估区间，不估 hypervolume 的数据置信区间；其 Monte Carlo standard
error 只描述积分误差。缺失 objective coverage、零 baseline 相对增益等保留 null。
接近阈值的增益必须提高积分精度并补充分层、配对不确定性分析，不能靠增加 seed
直到“过线”。Phase B 的随机曝光一致性和显著性要求仍需独立验证。

没有真实数据或 score/API 时，继续工程和合成验证；不修改 evidence 让真实 gate 通过。
最小交接为 train/validation/test NPZ、hash manifest，以及 feature_spec、
behavior_provenance、decision_provenance、evaluation_protocol 四份注册文档。
如果只有原始数据，则需要日志、caption/category metadata，以及已对齐 Jev 分数
或可用决策 endpoint；行为 checkpoint 可在本地训练，已有 checkpoint 可缩短流程。
已提供分数时不需要 live API。

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
