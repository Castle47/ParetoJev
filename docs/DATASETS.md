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

## ID 和时间规则

- content metadata 用 `final_video_id` 与 KuaiRand re-indexed `video_id` 对齐；
- history 必须严格早于 slate 首次曝光；
- train/validation/test 首先按时间切，再拟合 calibration/blend；
- test users 或 test interactions 不得参与 item-CF co-occurrence training。

## 数据许可

数据许可和下载条款以各官方项目为准。本仓库不代理分发。
