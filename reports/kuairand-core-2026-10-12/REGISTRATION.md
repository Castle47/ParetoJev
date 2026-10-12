# Core ParetoJev validation — registered before sampling, scoring or fitting

The question is whether full ParetoJev with real Jev signals outperforms a
behavior-only EMER-like reranker, and whether gains survive equal-signal controls.
This experiment supersedes further upstream behavior-model tuning.

## Comparisons and claim boundary

Six independently fitted methods, shared data/features/behavior predictions and
preference grid: EMER-like, CMR-like and ParetoJev, each with and without Jev.
The primary comparator is EMER-like without Jev. Equal-signal EMER-like/CMR-like
and ParetoJev without Jev separate total-package, signal and fusion contributions.
The baselines are explicitly project variants, not original-paper reproductions.
No result here alone proves superiority over the published EMER system.

Use seven raw observed labels rather than silently approximating a comment-or-
forward union probability. All seven targets are evaluated, with hate minimized.
Keep the existing 2% hypervolume / 3% cold NDCG / .005 regret / .60 controllability
thresholds; report both no-Jev and equal-Jev comparisons, all seeds and failures.
Do not choose the easiest comparator, drop an objective or tune on test results.

## Data and timing

Use complete early KuaiRand-27K logs, already hash-verified. Behavior tree fit ends
before April 18 and its probability calibration uses April 18. April 19 was prior
behavior development. Reranker train is April 20; validation April 21 00:00–12:00;
test April 21 12:00–24:00 (all UTC+08). These windows were not used for prior behavior
fitting/calibration/development evaluation. No new behavior training is required.

Within each user/period, take chronological nonoverlapping groups of ten exposures;
retain groups with one tab, ten distinct items, and span <=5 minutes. Freeze every
candidate's request time and history at the first exposure time. Item duration
is available content metadata; source log duration/static-upload assumptions remain
explicit. Last-50 exposure histories and rolling counts are strictly prior; Jev
gets last-ten exposure captions with no feedback labels. Candidate order is randomly
permuted deterministically, shared by every model. Original event record indices,
exposure times and common decision time are retained for audit.

These are exposed-session pseudo-slates, not true retrieval candidate sets.
Later exposures/labels can depend on the logging policy and prior session actions;
this offline test cannot establish causal online uplift. The common decision-time
boundary prevents target-specific histories and later context from leaking in.
Published text/physical metadata has no immutable snapshot proof; retain that limit.

## Coverage, sampling and weights

Before examining any model result, partition eligible slates by the first present
positive bit in this fixed order: hate, follow, forward, comment, like, otherwise.
Uniform reservoir sampling without replacement per stratum uses the registered
budgets (1,000 train / 400 validation / 600 test; no silent redistribution if a
stratum is smaller). Save population sizes and exact inclusion probabilities.
Validation/test statistics use N_h/n_h weights, including conditional NDCG,
negative-risk denominators, cold metrics and user-cluster bootstrap. The estimand
is the full eligible pseudo-slate population, not the enriched sample prevalence.
Training intentionally uses the same enriched rows unweighted for rare-objective
coverage; no output of the reranker is claimed to be a calibrated event probability.
Feature normalization is train-only. Item popularity counts are frozen before
April 20; cold cutoff is fitted only on weighted training candidate counts.

## Real Jev and budgets

The author's public JEV-27B endpoint has passed health and native Choice probes;
probe output is engineering evidence, not a benchmark observation. Pin the actual
weight revision to 7ce153d6f7933769527939bcb92ec3f3d8292884. Seven Choice questions
per slate, one per raw outcome, with A–J mapped to the shared shuffled candidates.
Choice scores are relative within-slate selection distributions, not pCTR/pLike;
retain raw responses, entropy and an explicit score-semantics field. No simulated
scores, objective broadcasts, or substituted smaller model may count as real Jev.

Native Chinese captions/cover text are preserved, with deterministic documented
character limits (160 history / 220 candidate). No machine translation or image
claims. Missing text is explicitly marked; no outcome-based text filtering.
One request at a time, >=1 second interval, 14,000 successful requests at full
sample capacity, lifetime attempt ceiling 15,400 including retries. Persist each
attempt before sending and each response before continuing. On transport failure,
back off and retry the identical payload; never substitute an easier sample.

## Learning, evaluation and audit

Three seeds (7/17/29), 20 epochs, fixed common optimizer/set-encoder settings.
Select checkpoints only on weighted validation utility; test is opened for model
comparison only after all six arms are fitted for a seed. Non-Jev methods must be
invariant to all Jev values/confidence/entropy; no-Jev Pareto disables Jev teacher
loss and bypasses its Jev fusion contribution. Parameter/compute differences remain
reported. Include a transparent behavior-weighted-sum score as an untrained sanity
control. Report preference controllability, objective coverage, hypervolume,
cold NDCG, regret and 1,000 paired user-bootstrap intervals. MC hypervolume error is
not a population uncertainty interval; seed means are not significance tests.

The outcome can be improvement, failure or insufficient evidence. Do not relabel
failures as success, broaden a narrow result into a universal claim, or imply that
engineering tests prove the recommendation hypothesis. Preserve checkpoints and
verify exact prediction/metric replay from saved data before completing the goal.
