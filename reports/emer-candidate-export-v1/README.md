# 已冻结的EMER＋Jev推理交付包

模型包：[emer-matched-blended-v1.zip](/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/emer-jev-scale-v3/exports/emer-matched-blended-v1.zip)，3.77 MiB。解压目录：[emer-matched-blended-v1](/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/emer-jev-scale-v3/exports/emer-matched-blended-v1)。

包含23个去重检查点、三折训练专属编码器、固定用户路由、冻结Python源码、输入契约、依赖版本和六条真实无标签示例。保留三个种子及匹配EMER、Jev策略、原EMER参考三个分支；没有挑选或合并种子。

**验证完成：9组分支×种子输出，每组36,000条，预测分数、排序和用户路由均与已审计结果完全一致。独立命令行在仓库目录之外运行通过。压缩包全部文件与导出目录逐字节一致。**

此包仅执行重排：上游26维原始数值特征、七目标行为概率和真实Jev选择分数需事先提供。包内不含27B模型或行为树模型，也不会调用API。推理入口拒绝标签和预编码的65维特征，按用户哈希使用对应折的编码器；详见[input_schema.json](input_schema.json)。

```sh
cd '/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/emer-jev-scale-v3/exports/emer-matched-blended-v1'
python infer.py --input example_input.npz --output /tmp/emer-example-output.npz --arm emer_jev --seed 7
```

要求安装requirements-lock.txt中的版本；参考环境Python3.12.11、arm64 CPU。精确重放在该环境核验，不承诺跨硬件逐位一致。实际业务端到端RT及Jev生成耗时不包含在此推理程序的范围内。

模型清单SHA256：`60a4b4b9fd9224c672e5b9bfc228ddce2e13afc4a250a6e3de1e793d9932de71`。
压缩包SHA256：`152194673ce3c86ca2c1ce253a6d4825c386bab4e88e69094b4103603939d177`。

原探索性比较平均NDCG@3+1.97%、5/5条件通过，但长看−0.66%，且该数据已被探索。此包不把结果升级为独立确认。Apr23–25的50K队列仍缺真实Jev分数，新的独立验证入口正在适配；不启动额外付费GPU。

证据：[manifest.json](manifest.json)、[verification.json](verification.json)、[delivery.json](delivery.json)。
