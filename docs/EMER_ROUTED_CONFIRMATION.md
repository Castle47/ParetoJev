# 固定导出候选的新日期确认

入口与最小输入见[就绪报告](../reports/emer-routed-confirmation-ready-v1/README.md)。模型包见[导出报告](../reports/emer-candidate-export-v1/README.md)。

- `configs/emer_routed_independent_v1.json`固定模型清单SHA、旧评分协议SHA、50K数据/请求/抽样审计哈希，以及独立评价条件。
- `evaluate_emer_routed_confirmation.py --stage prepare`只验证来源、路由和原始特征编码；不加载标签用于选模或计算模型效果。
- `--stage evaluate`先要求全部真实评分存在，再逐项审计；随后从已有数据读取原始26维特征，交由模型包按用户路由应用每折编码器。旧公共张量的全局编码不输入模型。
- `--stage replay`重新审计全部真实响应、重新计算所有模型预测/指标/区间，必须逐项一致。
- `finish_emer_routed_confirmation.sh`组合上述步骤及完整报告生成。它不启动GPU、采集API或训练模型。

主比较是固定策略与匹配EMER，原EMER参考比较同时受约束，训练方法对照另行报告。每个种子、路由组、目标和日期均保留，不按结果重选。该确认尚未执行，因为真实评分仍缺失；不能用36K探索性5/5替代独立结果。

当前全量319项测试及真实输入预检通过。候选清单SHA256为`60a4b4b9fd9224c672e5b9bfc228ddce2e13afc4a250a6e3de1e793d9932de71`。65元预算不变，不重复提出已被拒绝的追加预算请求。
