# Real SASRec and frozen semantic control — development experiment

This experiment trains a SASRec-style next-item recommender on real MicroLens
interactions and compares cached genuine AutoTrust Jev scores with a cheap frozen
MiniLM title encoder. The previously examined 400 users are **development users**.
They are never used for neural checkpoint selection. This is not a fresh test of
Jev, not multi-objective ParetoJev training, and not an exact original-paper SASRec
reproduction.

## 结果与结论

**已完成真实 SASRec 训练。更强行为模型成立，但原有 5% Jev 融合规则几乎失效；
预登记的 20% Jev 融合有开发阶段增益，尚不能宣布新的独立验证成功。**

在 1,000 名独立保留的 checkpoint 验证用户上，item-CF 的 AUC / NDCG@10 为
0.697447 / 0.485438；最终 SASRec 为 0.791421 / 0.548039。按 NDCG@10 选中
学习率 0.001 的第 15 轮，随后连续五轮未改善而在第 20 轮停止。另一个学习率
0.0003 跑满 30 轮，最佳为第 28 轮、NDCG@10 0.547204，未被选中。
这两个 checkpoint 的保存/重载预测均逐值相同。

以下为 **400 名已看过结果的开发用户**，每人 20 个候选。item-CF 已排除新的
1,000 名 checkpoint 验证用户重新拟合，因此不能直接替换以前报告中的 CF 数字。

| 方法 | AUC | HR@5 | NDCG@10 |
|---|---:|---:|---:|
| 同训练用户的 item-CF | 0.681447 | 0.538054 | 0.467663 |
| 真实训练 SASRec | 0.762237 | 0.590000 | 0.507418 |
| Jev 单独排序 | 0.647434 | 0.464583 | 0.400727 |
| TF-IDF 单独排序 | 0.577632 | 0.372950 | 0.338346 |
| MiniLM 单独排序 | 0.584342 | 0.375000 | 0.343234 |
| CF + Jev 消除平局 | 0.724013 | 0.576250 | 0.504900 |
| CF + MiniLM 消除平局 | 0.689737 | 0.545000 | 0.479876 |
| SASRec + 5% Jev | 0.762237 | 0.590000 | 0.507425 |
| SASRec + 5% MiniLM | 0.762171 | 0.590000 | 0.507403 |
| SASRec + 20% Jev | 0.770132 | 0.621250 | 0.513021 |
| SASRec + 20% MiniLM | 0.757763 | 0.598750 | 0.487814 |

描述性配对 bootstrap（10,000 次，pointwise 95%，未做多重比较校正）：

- SASRec − CF：AUC +0.080789，区间 [0.053026, 0.108355]；NDCG@10 +0.039755，
  区间 [0.002831, 0.076056]。
- 20% Jev 融合 − SASRec：AUC +0.007895，区间 [0.001316, 0.014276]；
  HR@5 +0.031250，区间 [0.003750, 0.060000]；NDCG@10 +0.005603，
  区间 [-0.010441, 0.021086]，这一项尚不确定。
- 20% Jev 融合 − 20% MiniLM 融合：AUC +0.012368，区间 [0.005592, 0.019145]。
- CF 的 Jev 消平局 − MiniLM 消平局：AUC +0.034276，区间 [0.014934, 0.054474]。

**为什么 5% 几乎没效果？** SASRec 的 8,000 个候选分数在各自用户内没有精确平局。
对 20 个严格有序候选，主分数的名次差至少为 1，第二信号最大名次差为 19。
95% 主名次 + 5% 第二名次的最小差为 `0.95 - 0.05 × 19 = 0`：它不能严格逆转
主排序，极端情况下只能制造新平局。因此旧 CF 上的 5% 经验不能直接迁移到
连续分数的 SASRec。20% 权重是训练前同时登记的对照，并非看到结果后追加，
但不能替换原先的 5% gate。

预登记开发诊断为 **1/3 通过**：SASRec 验证提升通过；两项 5% Jev AUC 增量
下界要求均未通过。20% 结果的点估计虽然超过旧 Phase A 的三个绝对阈值，
但评价范围不同、且这些用户已参与开发，**不能据此把原项目标为达标**。
这轮也没有训练 Jev、MMoE 六目标行为头或 full ParetoJev 融合重排模型。

## 成本与核验

- M1 / CPU、4 线程；两个 SASRec 训练合计 418.16 秒。一个随机种子，未重跑
  多种子稳定性实验；保证了 checkpoint 重放，未宣称完整训练已独立复跑。
- MiniLM 权重约 90.9 MB（十进制），19,738 个标题生成 384 维向量共 24.67 秒
  （包含加载和编码，不含下载）。72 个缺失标题编码为零向量。这不是线上延迟基准，
  也不能直接与带请求限速的 Jev API 采集耗时比较。
- 开发候选 OOV 比例 0.0625%，历史 OOV 比例 0.25%，positive OOV 为 0。
- 同批开发候选中 CF 有 7,564/8,000 个落入平局组，SASRec 为 0。
- 203 项测试通过。原始日志审计验证 1,400 个验证/开发 slate 的候选、时间窗口、
  历史边界、正样本和训练用户隔离；训练数组由原始日志重建完全一致。
- 禁止 HTTP 后重放评估；全部指标、区间、逐用户指标与预测文件完全一致。

机器可读证据：[训练](training.json)、[语义编码](semantic.json)、
[评估](evaluation.json)、[核验](audit.json)、[开发目标](target_report.json)。

## 下一步与最小输入

下一轮若验证 20% 规则，需要先冻结该权重、选择从未评价的新用户，并把这些新用户
从行为模型和 CF 的训练中排除后重训；不能直接用当前训练用户冒充新测试集，
也不能把已经用于选 checkpoint 的 1,000 名用户转成独立测试集。
再固定 MiniLM 对照、候选协议和检验方式，收集新 cohort 的真实 Jev 分数。

现有公开数据、代码和本地算力足以准备与重训，不需要用户补充材料。外部依赖仅是
新 cohort 对齐的真实 Jev 分数：可由仍可用的同版本端点生成，也可提供缓存文件。
如果公共端点不可用，最小替代输入是新 cohort 的逐候选分数、candidate/user IDs、
模型版本与请求协议溯源。多目标 ParetoJev 主实验另需真实多目标反馈及对应信号；
当前 MicroLens 下一条观看数据不能制造点赞、长看等标签。

## Registered protocol

- 57,845 training users, 503,713 interactions, 442,815 last-window next-item targets.
- 1,000 separate users select checkpoints by expected NDCG@10. Both the 400
  development users and these validation users are excluded entirely from training.
- Item-CF is refitted on exactly the same eligible training-user population.
- Two learning rates (0.001 and 0.0003), one fixed seed, at most 30 epochs each,
  patience 5. All other architecture/training choices are frozen before training.
- Two causal attention layers, width 64, two heads, maximum training length 50;
  sampled positive/negative BCE with unseen training-catalog negatives. This is a
  pre-norm PyTorch SASRec-style variant with tied item embeddings and no item bias.
- All compared methods see exactly the same five history items at evaluation.
  The item vocabulary comes from training users only; PAD/OOV vectors are zero.
- Frozen `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; masked mean pooling, L2 normalization,
  maximum cosine against the five history titles. No recommendation fine-tuning.
- Reuse all 8,000 cached real Jev decisions; **zero new Jev HTTP calls**.
- Fixed 5% and 20% average-rank fusion, and CF tie-breaking controls. No downstream
  fusion weight is fitted or selected from these results.
- Exact half-credit AUC and expected HR/NDCG/MRR over unresolved ties. Bootstrap
  intervals are descriptive pointwise 95% intervals, not multiplicity-controlled
  confirmatory evidence.

The [registration](registration.json) was committed before training. It records
raw/prepared input hashes, relevant source hashes, user membership, architecture,
selection rules, diagnostic gates and limitations. The [target configuration](../../configs/targets_sasrec_development.yaml)
implements the same three development gates and cannot satisfy original research gates.

## Reproduction

From the repository root, install the project and locked dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
python scripts/download_microlens_semantic_model.py
python scripts/run_microlens_sasrec_development.py train
python scripts/run_microlens_sasrec_development.py semantic
python scripts/run_microlens_sasrec_development.py evaluate
python scripts/audit_microlens_sasrec_development.py
```

Use `.venv/bin/python` (or activate the environment) for those Python commands.
Raw MicroLens files, the two frozen candidate files and both genuine score caches
must already be present at the paths recorded in the registration. These are
ignored artifacts, not included in a source-only Git bundle. A completed training
report is protected from accidental overwrite. Evaluation reuses checkpoints and
embeddings without training or Jev access; replay checks block HTTP explicitly.
The `prepare` stage creates a **new** registration and refuses to replace an
existing one. The audit rebuilds training arrays exactly from raw logs and checks candidate
identities, time windows and history boundaries; preserve the registered prepared
data for straightforward evaluation replay.

This is a retrospective user holdout, not a global time holdout: interactions of
other training users may occur after an evaluation event. Sampled alternatives
are not observed nonclick impressions. The English semantic encoder's pretraining
corpus independence from MicroLens is unverified. These restrictions apply even
if absolute metrics are high.

## Sources

- [Original SASRec author implementation](https://github.com/kang205/SASRec)
- [MiniLM model card and pooling recipe](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Prior real Jev collection and new-user validation](../microlens-independent-2026-10-08/README.md)
