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

### T2：低风险胜点

目标：MicroLens CV hybrid 同时改善至少两项核心指标，且没有显著伤害第三项。

优先顺序：

1. rank-normalized item-CF + JEV-VL/官方 Jev；
2. 加入 popularity/title similarity；
3. confidence/entropy 条件 gate；
4. 再考虑复杂学生模型。

若融合没有增益：先分析 per-user disagreement 与 oracle upper bound，而不是直接增加模型复杂度。

### T3：主研究胜点

目标：KuaiRand 上 ParetoJev 优于 EMER-like/CMR-like。

必须同时满足：

- Pareto hypervolume 改善；
- cold bucket 改善；
- worst-objective regret 受控；
- preference controllability 成立；
- 随机曝光轨道方向一致或能解释差异。

### T4：蒸馏/服务

只有 teacher 的真实增益稳定后才训练 student。否则蒸馏只会压缩一个无增益系统。

## 2. 默认研究目标

见 `configs/targets_research.yaml`。阈值是 go/no-go 条件，不是结果陈述。

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

## 4. Target 报告格式

每个阶段输出一个 JSON：

```json
{
  "dataset": "microlens-100k",
  "protocol_version": "v1",
  "model_versions": {"jev": "jev-1.13.0"},
  "metrics": {},
  "confidence_intervals": {},
  "latency": {},
  "cost": {},
  "target_report": {},
  "known_limitations": []
}
```

不允许只保留最终表格而丢失 request/config/hash。
