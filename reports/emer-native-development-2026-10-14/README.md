# EMER＋Jev 扩大开发实验执行记录

2026-10-14启动：25.2万项真实Jev响应逐条审计通过，候选/目标/请求哈希/输出张量一致。
EMER基线12次训练已经完成；Jev臂固定3种子×4配置×30轮，本地执行中。
此文为启动快照，不是最终效果结论；最新完成数见外置盘training-native/progress.json。

训练输出：`emer-jev-scale-v3/training-native`。训练完成后，
`finish_emer_native.sh`自动运行相同evaluator，再用独立进程精确重放预测、指标和区间。
结果将写入`emer-jev-scale-v3/evaluation-native/README.md`及`evaluation.json`、`replay.json`。
验证集参与选模，所有结果明确标为开发集诊断。独立测试仍未开放；不新增付费GPU。

相关12项模型及native协议测试通过，评估脚本语法校验通过；真实评估尚待全量训练完成。
