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

P-RAS 只对这种无歧义 pair 施加 dominance loss。统一 comparison harness 另外用
每个请求的偏好加权 utility 对 shared score 施加 pairwise supervision，因此互不支配
pair 也参与学习；这不是按“正反馈数量”固定排序。

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


## Formal behavior branch and temporal contract

The production-comparable behavior path is a SASRec user encoder with shared candidate
embeddings and MMoE objective heads. Its input contract is deliberately stricter than
a random event split:

```text
behavior_train ── train weights/vocabulary/context transform
       ↓ strictly later
behavior_validation ── early stopping and diagnostics
       ↓ strictly later
reranker_train ── out-of-period pXTR features for ParetoJev
```

Histories are left padded so the most recent item remains at the final sequence
position. Every history lookup uses a strict `< current_timestamp` boundary; events
sharing a timestamp cannot see one another. Item popularity and user-history counts
use the same rule. The item vocabulary is fit only on behavior-train, reserving index
0 for padding and index 1 for unseen items. Validation and downstream candidate/history
OOV fractions are reported rather than silently expanding the embedding table.

The scorer emits a versioned `SlateScoreTensor` and validates objective order against
the prepared slate file. Unweighted BCE is used for the first formal baseline so its
sigmoid is interpretable as a probability estimate; no post-hoc calibrator is fit on
the early-stopping labels. Class reweighting/focal variants require a separate,
chronologically later calibration split.

## Behavior-informed annotation policy

Jev annotation is expensive, so the main pipeline does not send every pseudo-slate.
It first preserves the existing structural cells over coldness, user-history length,
random/standard exposure and feedback density. Inside each cell, a behavior checkpoint
can provide an active-learning priority:

```text
priority = 0.5 * mean Bernoulli entropy
         + 0.3 * (1 - mean top1/top2 utility margin)
         + 0.2 * cross-objective winner disagreement
```

For `is_hate`, utility is `1 - p(hate)` before winner/margin calculation. The configured
`behavior_priority_strength` interpolates percentile priority with a seeded random rank.
A value of `0` reproduces random within-cell order; the recommended first probe uses
`0.5`; `1` is a deterministic high-uncertainty ablation. This is an annotation-training
policy, not an evaluation sampling policy.

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

## Unified trainable-baseline layer

`data/comparison.py` enforces `baseline-harness-v1`: three hashed chronological
splits, common objective/feature ordering, train-fitted feature identity, user/item
identities and declared origin. Optional simulator oracle targets are forbidden.
`experiments/comparison.py` registers source/data/config/environment before training;
`models/comparison.py` supplies independently trainable variants;
`evaluation/comparison.py` evaluates the selected checkpoints on the same test rows.

The raw input signals are common to all methods. CMR-like uses FiLM-conditioned
set encoding and a preference-weighted sum of objective logits. EMER-like adds a
shared score head and training-only detached current-loss/preceding-EMA feedback,
with clipped, renormalized preference multipliers. These are explicit project
variants, not reproductions of the named papers.

Full ParetoJev retains confidence fusion, FiLM, its set encoder and uncertainty
head. In this harness all methods train their served score using request-weighted
observed-utility pairs, including incomparable trade-off pairs. The full model also
uses dominance, **per-request shared-score** Tchebycheff, uncertainty, Jev utility
KL and fusion calibration. This comparison loss differs from the legacy smoke and
`train-precomputed` training path; their training numbers are not interchangeable.
Exact loss coefficients and variant definitions are in
[BASELINE_HARNESS.md](BASELINE_HARNESS.md).

All methods have equal data/order, epochs and optimizer settings, with parameter
counts reported separately. Validation chooses maximum fixed-grid weighted utility;
epoch zero is eligible and ties retain the earliest checkpoint. Test outcomes do
not select checkpoints or settings. The cold cutoff is frozen from training.
Both EMER-like and CMR-like comparison identities are fixed before evaluation;
EMER-like remains the primary report namespace, and no test-selected winner changes
the denominator. Undefined metrics remain null and coverage is explicit. Registered
seed means are descriptive; user-bootstrap intervals condition on fitted models,
and hypervolume integration error is not a data confidence interval.
