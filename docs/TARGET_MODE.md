# Target Mode 执行规范

Target Mode 的核心不是“设一个很高的最终指标”，而是将研究拆成有退出条件的可证伪阶段。

## 1. Target 层级

### T0：工程正确性

- 单元测试通过；
- smoke loss 有下降；
- fusion 能在已知互补的合成信号上优于平均 source；
- cold item 的 Jev gate 高于 warm item。

未通过时不得进入真实 API 消耗阶段。

### T1：协议复现

- MicroLens candidate IDs 与公开 Demo 的 deterministic protocol 一致；
- item-CF 指标在合理容差内接近上游报告；
- 同一 score 文件重复评估完全一致；
- Jev request cache 命中率和 resolved model ID 可审计。

未通过时先排查数据版本、候选顺序、tie-breaking、test-user leakage。

### T1.5：Behavior baseline 可审计性

在 KuaiRand 消耗 Jev annotation 预算前，必须完成：

- behavior-train、behavior-validation、reranker-train 严格按时间互斥；
- 等时间戳事件不互相进入 history 或 popularity；
- validation/reranker item OOV 比例、macro AUC/AP、Brier 与 prevalence 可审计；
- slate scorer 的 candidate order、objective order 和 checkpoint vocabulary 一致；
- downstream reranker 不使用 behavior model 对自身训练事件的 in-sample 预测。

若正式 behavior branch 尚未跑通，只允许把 historical-Bayes signal 标为 bootstrap，不得作为主 baseline。

Jev annotation manifest 还必须记录 behavior priority 的 inventory mean、selected mean、
priority bucket 分布与随机混合强度。高 priority lift 只说明选样器生效，不是推荐指标。

### T2：低风险胜点

目标：MicroLens CV hybrid 同时改善至少两项核心指标，且没有显著伤害第三项。

优先顺序：

1. rank-normalized item-CF + JEV-VL/官方 Jev；
2. 加入 popularity/title similarity；
3. confidence/entropy 条件 gate；
4. 再考虑复杂学生模型。

若融合没有增益：先分析 per-user disagreement 与 oracle upper bound，而不是直接增加模型复杂度。

### T3：主研究胜点

目标：真实 KuaiRand 上 ParetoJev 优于预注册的 EMER-like/CMR-like 项目变体。
这不是对两篇原论文的复现声明。

必须同时满足：

- 对两种固定 baseline 各自 hypervolume 相对改善 ≥2%；
- 对两种 baseline 各自 cold-item NDCG 相对改善 ≥3%；
- 对两种 baseline 各自 worst-objective regret ≤0.005；
- ParetoJev preference controllability Spearman mean ≥0.60；
- 随机曝光轨道方向一致或能解释差异。

### T4：蒸馏/服务

只有 teacher 的真实增益稳定后才训练 student。否则蒸馏只会压缩一个无增益系统。

## 2. 默认研究目标

阈值是 go/no-go 条件，不是结果陈述。按阶段选择配置：

| 配置 | 含义 |
|---|---|
| `targets_smoke.yaml` | 合成工程 gate |
| `targets_kuairand_interim.yaml` | 静态 scalarization 的数值诊断；不证明真实研究胜点 |
| `targets_comparison.yaml` | 真实 KuaiRand 统一 harness 的双固定 baseline gate |
| `targets_microlens.yaml` | 真实 MicroLens 的 CV 与配对区间 gate |
| `targets_research.yaml` | 合并的 MicroLens、KuaiRand、serving 计划；未做阶段保持不可评估 |

比较 harness 保留主 namespace `kuairand.paretojev`（EMER-like），第二个固定比较为
`kuairand.paretojev_vs_cmr`。不得从 test 选择最有利的 baseline、seed 或 preference。
所有 seed 预注册；均值只作描述统计，不能替代研究要求的配对显著性和随机曝光分析。


## 3. 失败后的固定诊断树

### Jev-only 不如行为模型

依次检查：

1. prompt 中是否有足够区分候选的信息；
2. 中文 vs 英文翻译；
3. Choice 是否因 K 增大过于平坦；
4. Noul 是否更适合绝对行为预测；
5. 按 item popularity、history length、category 分桶；
6. 只在高 confidence/低 entropy bucket 使用 Jev。

### Hybrid 不如最佳单模型

检查：

1. 是否在测试集上调 alpha；
2. score 尺度是否不兼容；
3. rank normalization / calibration；
4. 是否存在 per-user oracle 增益；
5. gate 是否只学到固定权重；
6. source disagreement 是否与正确性相关。

### Pareto 指标提升但单目标退化过大

检查：

1. reference point 是否固定；
2. objective normalization；
3. negative objective 方向；
4. Tchebycheff ideal point；
5. Advantage Evaluator 是否过度补偿；
6. 增加 regret constraint 或 ε-constraint。

### Student retention 低

检查：

1. teacher soft distribution temperature；
2. item text embedding 是否离线预计算一致；
3. student capacity；
4. cold/warm 分层蒸馏；
5. 只蒸馏高置信 teacher 样本；
6. behavior/Jev disagreement sample reweighting。

## 4. 状态与证据规则

`evaluate_targets` 与 `check-targets` 输出每个 target 的 `status`、`actual`、
`threshold`、`passed`、`required` 和可操作的 `reason`，同时保留原有布尔接口：

- `passed`：证据满足，有限数值达到阈值；
- `failed`：证据满足，有限数值未达到阈值；
- `not_evaluable`：指标缺失、null、非数值、NaN/Infinity，或必须的 evidence 不满足。

配置中的 threshold 必须有限；布尔值不算数值指标。空 target 或没有 required
目标时不能因空集合判断而通过。可选目标不可评估不阻止 required 目标通过。
任何 required 目标不可评估时，总状态是 not_evaluable；否则存在 required 失败时为
failed，全部通过时才是 passed。`check-targets` 的退出码为通过 `0`，其余 `3`。
`compare-baselines` 的退出码 `0` 只表示实验完成，应另读 target report。

配置支持 `evidence_requirements`（report dotted path → 严格匹配值）和
`required_methods`（必须存在的非空 method report）。真实 KuaiRand gates 要求：

```yaml
evidence_requirements:
  evidence.real_data: true
  evidence.data_kind: real
  evidence.dataset: kuairand
  evidence.protocol_version: baseline-harness-v1
  evidence.comparison_baseline: emer_like
required_methods: [emer_like, cmr_like, paretojev]
```

`real_data` 必须是真正的 boolean，`1` 或字符串 `"true"` 不算。
MicroLens 单独 gate 要求真实来源和 `dataset: microlens`；现有 evaluator 不会仅凭
输入文件名自动建立这种来源证据。reviewed evidence 缺失时应保留 not_evaluable，
不能为了通过而给合成输入改标签。完整研究文件含多个阶段，单个阶段产物无需强行
填充其他阶段指标。

合成 harness 固定输出 `real_data: false`、`data_kind: synthetic`；即使所有数值很高，
也不能通过真实 gate。旧静态 evaluator 缺少统一方法/协议 evidence，不能替代主实验。
Manifest 的 hash 验证保证登记文件未变，不能独立证明文件来自真实平台或不存在
上游泄漏；这些仍需审阅 feature/behavior/decision/evaluation evidence。

## 5. 注册、结果和不确定性

每次运行保存注册信息、request/config/hash、checkpoint、training history、逐 seed
metrics 和预测审计；不要只保留最终表格。准确的产物结构见
[BASELINE_HARNESS.md](BASELINE_HARNESS.md)。恢复运行只能沿用相同 source、输入、
环境和配置；新研究假设需要新注册目录。

所有预注册 seed 的算术平均要求每个值都定义；缺一个则 aggregate 为 null。
Objective coverage 缺失、baseline 为零的相对增益等不能用零填补。
Hypervolume Monte Carlo standard error 只反映积分误差；现有 user-cluster bootstrap
对固定 checkpoint 的 objective/cold 差值估区间，不包含重新训练的不确定性，
也未实现 hypervolume 的数据置信区间。数值过线不自动满足显著性要求。

缺少真实数据/API 时继续协议、模型和合成验证。真实执行的最小输入是三份已对齐
NPZ、hash manifest 和四份上游证据文档；已提供分数无需 API。缺失项列入执行日志，
保留 not_evaluable，不伪造 benchmark。
