# KuaiRand-27K：全量数据入口与真实特征审计准备

已按用户要求把当前工作数据源明确改为 **KuaiRand-27K**，不自动退回 1K、Pure
或抽取少数用户冒充全量实验。当前已经完成 27K 的获取/审计入口、真实用户特征
小文件核查及工程测试；**全量数据获取、富特征多目标行为模型训练、ParetoJev
真实多目标验证均尚未完成。**

## 实际取得与核查的内容

- 官方 27K 压缩包的 HTTP Range 请求成功，报告总长度 **9,892,191,178 字节**，
  约 9.89 GB。官方文档给出的解压后核心文件规模约 46 GB。
- 仅取文件开头的 64 KiB 作目录检查，再取 2 MiB 前缀，完整恢复其中较小的
  `user_features_27k.csv`，大小 **3,519,028 字节**。这不是下载了完整 27K。
- 用户特征表实际含 **27,285 行、27,285 个不同用户 ID、30 个非 ID 原始字段**。
- `is_lowactive_period` 在这张表中为常量，不能当作有效区分信息。
- `onehot_feat4` 缺失 874 条；`onehot_feat12` 至 `onehot_feat17` 各缺失 714 条。
  后续需要训练集拟合的缺失处理/未知类别，而非悄悄用标签或未来统计填补。
- 表里没有快照时间列。30 个字段的预测时点可用性尚未得到证明；不能声称
  已有 30 个可直接投入严格时间实验的安全特征。
- 前缀源来自官方完整 27K 包，已记录实际文件 SHA-256；**无法仅凭前缀验证整包
  发布 MD5**。正式获取必须继续完成整包校验。

证据：[前缀目录探测](remote_prefix_probe.json)、[完整用户表核查](user_feature_probe.json)、
[数据和资源计划](acquisition_plan.json)、[磁盘阻塞测试](storage_guard_check.json)。

## 已完成的工程

1. 默认 `configs/kuairand.yaml` 改为 27K 的四个 standard 分片及一个 random 文件。
   原 Pure 配置/脚本明确保留为 `*_pure_legacy`，不作自动回退。
2. 新入口 `scripts/prepare_kuairand_27k.py` 提供 plan / download / extract / audit。
   下载固定官方文件名、URL 和发布 MD5，同时记录 SHA-256；支持严格 HTTP Range
   续传，拒绝将不支持 Range 的响应追加到已有文件。
3. 保留下载中断文件，不改写已有完整文件；错误停止，不跳过 TLS 校验。下载前
   检查工作空间预算，写入时持续保留至少 2 GiB，不删除用户文件或旧研究成果。
4. 解压只接受正确 27K 根目录下的普通文件；拒绝路径越界、链接、重复文件和
   已有文件内容变化。逐文件原子提交并保存校验清单，支持完成文件复用。
5. 数据审计逐 CSV 分块读取，不拼成一个大 DataFrame。检查字段、二元标签、
   用户/物品 ID 范围、standard/random 区分、日期范围、每目标正例量、日期/场景
   分布及长看标签一致性；记录完整文件 SHA-256。
6. 分片与原始行次序不被当作全局时间次序；后续必须单独建立外存时间/历史索引。
   显式前缀审计始终标记为 `prefix_only`，不能冒充全量统计或训练结果。
7. 整月汇总的 `video_features_statistic_27k_part*.csv` 默认不作严格时间模型的
   输入；需要的统计量从当前事件之前的日志重新计算。
8. 暂停新 Jev 打分。须先完成行为模型、目标定义、语言/内容覆盖与预算登记。

工程测试：**226 项通过**。其中新增 17 项覆盖分块结果/哈希一致、前缀范围标记、
非法标签/ID/混合轨道拒绝处理、完整 27K 分片识别、磁盘不足、下载校验与续传、
解压越界/符号链接及已有文件篡改等情况。测试样例是合成工程用例，不是推荐 benchmark。

## 当前阻塞与最小输入

首次检查系统盘约剩 4.3 GiB，后续系统释放空间后约 10 GiB，仍不足以存放完整核心
数据及处理文件。目前只看到系统卷和小型 Docker 安装卷，没有可用的大容量数据盘。

需要一个可写的大容量目录，例如外接盘上的项目目录。建议为此次工作保留约
**100 GiB**：核心压缩包约 9.89 GB，解压核心约 46 GB，caption/category 原始补充
文件约 6.9 GB，还需处理文件和运行余量。100 GiB 是保守工作空间预算，不是发布方
声称的数据集大小；续传时会计入本任务已经占用的下载/解压空间。

不需要购买数据或申请 API Key；当前缺的是存储目录。已经验证磁盘不足时下载
入口会在访问大文件之前停止，不会把系统盘写满。没有删除任何旧结果。

## 给出目录后的操作

将 `/path/to/large-disk/kuairand27k` 替换为实际目录：

```bash
.venv/bin/python scripts/prepare_kuairand_27k.py plan --root /path/to/large-disk/kuairand27k
.venv/bin/python scripts/prepare_kuairand_27k.py download --root /path/to/large-disk/kuairand27k
.venv/bin/python scripts/prepare_kuairand_27k.py extract --root /path/to/large-disk/kuairand27k
.venv/bin/python scripts/prepare_kuairand_27k.py audit --root /path/to/large-disk/kuairand27k \
  --report reports/kuairand-27k-readiness-2026-10-09/full_audit.json
```

这些步骤完成后仍不自动声称“训练已就绪”。还需要：

- 将 `tab` 与界面场景对齐，区分 `is_click` 的点击和有效播放含义；
- 审计用户/物品特征的时点可用性，核对内容文本覆盖、语言与 ID 对齐；
- 按时间固定行为训练/验证、重排训练/验证/测试，历史严格早于预测事件；
- 实现外存读取、历史索引和有界词表/Embedding，接入真正的丰富特征；
- 训练并校准多目标 pXTR，再开展 Jev 融合与 ParetoJev 的公平比较。

现有 `train-kuairand-behavior` 仍是内存读取和少量上下文特征的旧训练入口，
**不能直接当作已经适配 27K 的富特征训练器**。以全部约 3,204 万视频、64 维
FP32 Embedding 估算，仅物品表就约 8.2 GB，尚未计入梯度和优化器；后续必须
采用明确登记的可扩展表示与读取方案，不能在本机直接创建全目录稠密表。

## 官方来源

- [KuaiRand 官方文档与校验值](https://github.com/chongminggao/KuaiRand)
- [27K 主数据](https://zenodo.org/records/10439422)
- [caption/category 补充文件](https://zenodo.org/records/18159199)
