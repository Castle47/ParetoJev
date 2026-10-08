# 架构与算法说明

## 1. 信号定义

对用户 `u`、候选 `i`、目标 `m`：

- `p^B_uim`：behavior model probability；
- `p^J_uim`：decision-model probability；
- `g_uim`：Jev reliability gate；
- `r_uim = g p^J + (1-g) p^B`。

Gate 输入：

```text
[p_behavior, p_jev, |difference|, entropy, confidence,
 log1p(popularity), log1p(history_length)]
```

## 2. Choice 与 Noul

### MultiChoice

每个目标一个 Choice，输出候选间分布：

```text
sum_i q_i,m = 1
```

它适合 slate 内相对比较，不能未经校准直接称为绝对 CTR。

### MultiNoul

每个 candidate × objective 一个二元问题，输出近似绝对概率。官方 Noul 不单独返回 Choice-style confidence；代码将 `1 - binary_entropy` 标为 concentration proxy。

## 3. Preference conditioning

运行时权重 `w` 通过 FiLM 调节 candidate hidden state：

```text
h' = gamma(w) * h + beta(w)
```

同一个 checkpoint 可在 long-view、engagement、follow、negative-feedback control 等偏好间切换。

## 4. P-RAS

若 item `i` 在所有目标上不差于 `j`，且至少一个目标严格更好，则：

```text
i Pareto-dominates j
```

只对这种无歧义 pair 施加 shared-score pairwise loss。互不支配 pair 交给 preference-conditioned objective loss，而不是强行按“正反馈数量”排序。

## 5. Augmented Tchebycheff

```text
L = max_m w_m (L_m - z*_m) + rho * sum_m w_m (L_m - z*_m)
```

相比单纯 weighted sum，它更适合探索非凸 trade-off front。

## 6. Uncertainty

Reranker 输出 `mean` 与 `log variance`。候选 pair 的胜出概率由高斯差分计算，用于处理多目标反馈冲突与 Jev/behavior disagreement。

## 7. Distillation

Teacher 概率先按优化方向变成 utility probability；再将每个目标的候选分布蒸馏到 student objective head。负反馈目标使用 `1 - p(hate)`，避免方向冲突。

## Preference-sweep evaluation layer

A saved reranker is evaluated at a deterministic set of runtime preference vectors:
uniform, each one-hot objective, optional pairwise trade-offs, and seeded Dirichlet
samples. Every preference produces one objective-utility point. Positive objectives
are measured with conditional NDCG@K; negative feedback is converted to
`1 - top-K risk`, so the complete point set is maximize-oriented.

Behavior-only, Jev-only and confidence-fused probability scalarizations use the same
preference vectors and candidate slates. Hypervolume is estimated inside a common
`[0,1]^M` box with identical random samples for every method. This makes the
intermediate comparison reproducible, while final claims still require separately
trained EMER-like and CMR-like baselines evaluated by the same contract.

## Pre-slate behavior bootstrap and materialization

Real Jev annotations can now be materialized before the final neural behavior branch
is available. The bootstrap scorer maintains global, user and item empirical-Bayes
states and scores every candidate from history strictly earlier than the pseudo-slate
start. Outcomes inside one pseudo-slate are queued and applied only before a later
slate, so an early candidate label cannot leak into a later candidate in the same
constructed set. Overlapping windows are de-duplicated by `(user, item, timestamp)`.

This path exists to unblock pipeline and complementarity probes:

```text
Prepared slates
  ├─ strict historical Bayes scores ─┐
  ├─ Jev score JSONL ────────────────┼─ strict ID/objective alignment
  └─ train-fitted candidate features ┘
                                      ↓
                         feature-complete training NPZ
```

The bootstrap signal is deliberately reported separately from SASRec+MMoE. Its Brier
value is a prequential diagnostic on represented pseudo-slate events, not an
independent held-out result.

Candidate feature transforms are train-fitted. The default contract excludes
pseudo-slate sequence position, preserving set semantics and avoiding direct use of
the logged production order. Position can be enabled only for a named ablation.
