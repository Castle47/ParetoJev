# KuaiRand-27K：真实多目标行为模型开发试验

注册时间：2026-10-11，首次模型训练与验证结果之前。配置见
`configs/kuairand_27k_behavior_pilot.json`。这不是最终 ParetoJev benchmark。

## 目的与口径

首先验证真实反馈的多任务预测是否超过发生率基线。尚不使用 Jev，尚不训练
ParetoJev 重排器；不沿用 MicroLens 下一物品 AUC 来替代多目标预测证据。
七个任务直接保留原始点击、长看、点赞、关注、评论、转发、负反馈标签；
不把评论/转发合并成 deep_engagement，以便暴露各自稀疏性。后续与六目标
重排协议对接须显式登记映射，不能将两个概率直接取最大当作联合概率。

## 数据与隔离

使用完整 KuaiRand-27K 两个早期 standard 分片共 136,296,576 次曝光构建
外存索引。每次读取重算 SHA-256，与既有全量审计核对；按用户归档后逐用户
稳定排序，不能依赖 CSV/分片顺序。用户数以各时间段实际出现者为准，
晚期才出现的用户不可能出现在这轮训练中，不伪称全 27,285 人均已训练。

- 训练：time_ms < 2022-04-18 00:00:00 Asia/Shanghai。
- 验证：4 月 18 日零时至 4 月 20 日零时（右开）。
- 4 月 20–21 日留给下游 out-of-period 特征与重排训练；后期 standard 和
  random 日志不参与本次模型训练或指标计算。
- 每用户每期间均匀、不放回抽取最多 16/8 个训练/验证目标，随机种子固定；
  不按结果标签抽样。**这是全数据集历史上的目标采样试验，不是全曝光训练。**
- 每条目标的输入历史最多 50 次严格更早的曝光，不使用历史反馈标签；
  同毫秒曝光全部排除。允许验证期已发生的曝光 ID 滚动进入后续历史，
  验证标签不更新模型。
- 每条样本权重 N_user_period / n_sampled；训练 BCE 与指标均使用该权重，
  估计按曝光加权的分布。无需人为 class weighting 或标签平衡采样。
- 未来窗口记录不进入训练历史、候选选择、训练用户编码词表或尺度估计。

## 输入与模型

采用 SASRec-style 1 层 32 维、2 attention heads 的序列编码器和 4 个 MMoE
experts，输出七个 logits；sigmoid 只是概率估计，是否校准要由结果检验。
这属于既有架构的工程实现，不声称 SASRec/MMoE 原创，也不是作者原实现复现。

两个独立训练、同预算的特征口径：

1. **event_only**：7 个原始/派生输入字段：历史视频 ID 序列、候选视频 ID、
   用户 ID（训练未见用户统一 OOV）、tab、候选时长、请求时间、严格过去曝光次数。时间派生小时/星期
   正余弦，时长包含缺失标志；最终数值向量 7 维。这些不是额外独立信息源。
2. **released_user_snapshot**：在上面基础上加发布的 30 个非 ID 用户画像字段，
   共 37 个字段，按类别嵌入，训练用户上建立词表，缺失/未见值归 OOV。
   其中存在常量字段；不把“37 个字段”说成 37 个有用特征。
   **画像无快照时间，本组仅作敏感性分析，不能作为无时序泄漏的主证据。**

候选时长是曝光日志的内容属性，假设曝光时可用；这不是在线可用性审计。
不输入当前播放/停留时长、任何反馈标签、整月视频统计或文本信号。
视频 ID 使用固定 65,536 桶哈希（PAD=0/OOV=1，真实 ID 从 2 开始），避免
32M 全目录 dense embedding 超过 8GB 主机内存。哈希有碰撞，会损害容量；
本轮不是充分调优的强精排系统。真实 item basic 属性和严格历史统计仍待纳入。

## 固定预算、对照和判断

CPU、3 epochs、batch 256、AdamW lr=0.001、梯度裁剪 5、随机种子固定。
按验证七任务平均加权 log loss 最低选择 checkpoint，不按研究终点选模型。
以训练期全局发生率、按 tab 发生率（100 个全局先验伪曝光平滑）为两条基线。
报告每任务 ROC-AUC、PR-AUC、Brier、log loss、10-bin ECE、真实发生率和
平均预测值，按 tab 分层；is_click 的跨界面混合语义不能称为统一 CTR。
不推断 numeric tab 对应具体 UI，尚无可靠映射。

开发就绪门槛：七任务各有 >=20 个验证样本阳性，AUC>0.5，Brier 和 log loss
均低于训练全局发生率基线。还要展示场景基线，不能仅凭击败常量就称为强模型。
这是预先固定的点估计筛查，未作独立测试/置信区间，不代表研究指标达成。
原项目 targets 不变；research_target_mode=false，无 Jev 调用。
保存样本、来源哈希、checkpoint、预测和训练曲线，重新加载最佳 checkpoint
必须逐元素复现验证预测。无自动扩预算或验证失败后改门槛。

## 运行

```bash
.venv/bin/python -u scripts/prepare_kuairand_behavior_pilot.py \
  --data-root '/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k' \
  --output '/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/behavior-pilot-v1' \
  --audit reports/kuairand-27k-acquisition-2026-10-11/full_audit.json
.venv/bin/python -u scripts/train_kuairand_behavior_pilot.py \
  --data '/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/behavior-pilot-v1' \
  --output '/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/behavior-pilot-v1/run-v1'
```

外存索引/样本在专属项目目录，无需将大文件提交 Git。准备阶段可复用完整且
已提交 manifest 的产物；残缺索引拒绝使用，须另选新目录。训练不承诺自动续跑；
已存在 run 输出时拒绝覆盖，新的重试必须用新目录并记录原因。

## 实施核对（不改变注册边界）

采样完成后核对发现：早期分片虽然按 date 标注 4/8 起始，但部分 time_ms
落在 4/7 晚间；训练样本有 341 条此类目标，最早为 4/7 23:11:25.486 +08:00。
本协议原定义为早期日志中 time_ms < 4/18 零时，并未另设 4/8 下界；因此
保留这些记录，没有看到结果后更改切分规则。报告单列实际时间范围。
