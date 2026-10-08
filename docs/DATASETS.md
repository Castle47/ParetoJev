# 数据集与目录约定

## MicroLens-100K

推荐目录：

```text
data/raw/microlens/
├── MicroLens-100k_pairs.csv
├── MicroLens-100k_title_en.csv
└── MicroLens-100k_covers.zip        # optional
```

`paretojev prepare-microlens` 输出：

```text
data/processed/microlens/
├── candidates.csv
├── candidates.json
├── jev_input.jsonl
└── prepare_report.json
```

不提交原始数据或 cover archive。

## KuaiRand-Pure

```text
data/raw/kuairand/
├── log_standard_4_08_to_4_21_pure.csv
├── log_standard_4_22_to_5_08_pure.csv
├── log_random_4_22_to_5_08_pure.csv
├── user_features_pure.csv
├── video_features_basic_pure.csv
├── video_features_statistic_pure.csv
├── kuairand_video_captions.csv
└── kuairand_video_categories.csv
```

主要 labels：

| Objective | Source | Direction |
|---|---|---:|
| valid click/play | `is_click` | maximize |
| long view | `long_view` | maximize |
| like | `is_like` | maximize |
| follow | `is_follow` | maximize |
| deep engagement | `max(is_comment, is_forward)` | maximize |
| negative feedback | `is_hate` | minimize |

`play_time_ms / duration_ms` 用于连续观察指标，默认 clip 到 5；不把 outcome 本身泄漏进候选 pre-ranking feature。

`prepare-kuairand` 额外保存：

- `item_prior_count[N,K]`：每次候选曝光之前，该 item 在全局时间序列中已经出现的次数；
- `user_history_count[N]`：每个 pseudo-slate 首次曝光之前，该用户已经发生的曝光次数。

二者用于 cold/warm 与 short/long history 分桶，不使用当前或未来曝光。旧版 NPZ
仍可加载，但运行 `sample-kuairand-jev` 前需要重新构建，避免用全周期 popularity
替代严格 prior count。

对 validation/test 日志构建 slate 时，应通过 `--prefix-log` 传入更早的训练日志，
从而让第一条 validation 曝光的 prior count 继承训练期历史。工具要求 prefix 的
最大时间戳严格小于当前日志的最小时间戳；若两段时间重叠会直接失败。

推荐先把 4/08–4/21 日志按完整时间戳组切成：

```text
data/processed/kuairand/splits/
├── behavior_train.csv
├── behavior_validation.csv
├── reranker_train.csv
├── prefix_before_behavior_validation.csv
├── prefix_before_reranker_train.csv
└── temporal_split_report.json
```

`prefix_before_*` 只包含目标 split 之前的完整时间段，用于严格初始化 history
和 popularity。边界要求前一段最大时间戳严格小于后一段最小时间戳。正式 behavior
checkpoint 只在 `behavior_train` 拟合，并在 `behavior_validation` early-stop；用于
下游 reranker 的 pXTR 必须来自更晚的 `reranker_train` 或后续 split。

首轮 decision annotation 与训练物料推荐输出：

```text
data/processed/kuairand/
├── train_slates.npz
├── jev_sample_manifest.csv
├── jev_sample_report.json
├── train_jev_sample.jsonl
├── kuairand_behavior.pt
├── train_behavior_scores.npz
├── train_behavior_prior.npz        # bootstrap-only fallback
├── candidate_feature_spec.json
├── train_features.npz
└── train_features.materialize.report.json
```

其中：

- `train_behavior_scores.npz` 是正式 SASRec+MMoE 输出的 `SlateScoreTensor`；
- `jev_sample_manifest.csv` 可在结构 cell 内使用 behavior entropy/margin/objective disagreement 做软优先级；
- `train_behavior_prior.npz` 使用同一协议，但仅是 bootstrap-only fallback；
- `candidate_feature_spec.json` 只在训练 split 拟合一次，validation/test 必须复用；
- `train_features.npz` 是 `train-precomputed` 的完整输入；
- materialization 会严格核验 slate ID、candidate ID/顺序、mask 和 objective alias；
- 默认不加入 pseudo-slate 内的 sequence position，避免将日志中的原排序当成内容特征。


`jev_sample_manifest.csv` 是成本受控的 teacher-training annotation subset。其
feedback-density strata 使用了离线 outcome，因此不得直接在该子集上报告总体模型
指标；validation/test 使用完整固定 protocol 或声明 sampling weights。

历史 Bayes scorer 只看当前 slate 首次曝光之前、且出现在已准备 pseudo-slate 中的
去重事件。它用于先跑通真实 Jev 物料链路，不应替代最终 SASRec+MMoE baseline。

## ID 和时间规则

- content metadata 用 `final_video_id` 与 KuaiRand re-indexed `video_id` 对齐；
- history 必须严格早于 slate 首次曝光；
- train/validation/test 首先按时间切，再拟合 calibration/blend；
- test users 或 test interactions 不得参与 item-CF co-occurrence training。

## 数据许可

数据许可和下载条款以各官方项目为准。本仓库不代理分发。
