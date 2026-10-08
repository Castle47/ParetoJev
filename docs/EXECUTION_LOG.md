# Execution Log

## 2026-10-08 — New-user confirmation of Jev tie-breaking

- Pre-registered `94c5744`: 200 new evaluation users, seed 20261009, zero overlap
  with the earlier cohort; CF refitted excluding all 400 evaluation users. Strict
  per-user history boundaries were required before selection. The same retrospective
  dataset is used; this is not a new-platform or global time-holdout experiment.
- Froze seven policies, with no weight fitting on new users: CF, title-only,
  Jev-only, CF ties resolved by title/Jev, and fixed 95/5 rank fusions. Added exact
  half-credit AUC and expected tie-aware HR/NDCG/MRR; integer fusion keys prevent
  floating-point artifacts. No candidate-position jitter is used in this study.
- Collected 4,000 new real responses in 4,034.10 seconds, one request at a time,
  minimum one-second start spacing, zero failed slates. Model/revision stayed fixed.
- AUC: CF 0.665000; title ties 0.673816; Jev ties 0.717632. Predeclared paired
  97.5% AUC intervals: Jev-CF [+0.020789,+0.084211], Jev-title [+0.002105,+0.085263].
  Both mechanism gates pass. This does not mark the original project gates complete.
- Expected HR@5 / NDCG@10: CF 0.516100 / 0.431302; title ties 0.554000 / 0.446698;
  Jev ties 0.577500 / 0.481703. Secondary HR gain versus title still has an interval
  crossing zero; no blanket superiority across all measures is claimed.
- Post-evaluation relation audit: ties and 5% fusion produce identical full weak
  orders on all 200 users, for both title and Jev. The observed gain does not require
  the more complex neural ParetoJev framework.
- Cost plan was frozen from CF scores alone before Jev results: ties need 3,782/4,000
  candidate decisions, only 5.45% fewer. All 4,000 were actually collected to evaluate
  the matched 5% policy. No separate deployment latency or monetary savings claimed.
- All scores match exact raw responses. Scoring replay performs zero HTTP requests
  and preserves score/cache bytes. Offline metrics, intervals, targets and original
  collection cost are unchanged on replay. Rebuilt input, candidate and request-plan
  files have identical SHA-256. Original collection cost is preserved separately from
  replay timing, preventing a zero-call replay from erasing the measured cost.
- **200 tests passed in 6.30s**; compilation, shell syntax and whitespace checks passed.
- Full report: [`reports/microlens-independent-2026-10-08`](../reports/microlens-independent-2026-10-08/README.md).

## 2026-10-08 — Complete 200-user real JEV evaluation

- Extended the frozen pilot to the original 200 users, keeping model revision,
  prompt, candidate membership/permutation, fusion rules, grids and seeds unchanged.
  Verified exact request equality for all 20 reused users before scoring.
- Registered `0e57183`, 4,000 decisions total, 400 reused responses, 3,600 new calls,
  single concurrency and at least one second between request starts.
- Completed all 180 new users in 3,619.43 seconds, with 0 failed slates. All 4,000
  final scores match exact input requests and validated raw cached responses.
- Primary CV fusion: AUC 0.727368, HR@5 0.570, NDCG@10 0.518763 versus item-CF
  0.687632 / 0.540 / 0.493369 under the frozen shuffled-candidate protocol.
  Paired AUC gain: +0.039737, 95% CI [+0.006053,+0.074217].
- Fresh 180-user subgroup: CV AUC gain +0.038304, CI [+0.002339,+0.075146].
  These are the same OOF predictions, not separately retuned subgroup models.
- Required targets: **2/4 passed**, overall `failed`. AUC and HR@5 miss their
  thresholds; NDCG@10 and the paired AUC lower-bound gate pass. HR/NDCG gain
  intervals still cross zero. No real neural ParetoJev training is claimed.
- Post-hoc tie audit: original versus shuffled item-CF scores/labels are identical.
  Conventional half-credit AUC is 0.695526 in either order; CV is 0.727237, gain
  +0.031711 with CI [+0.006842,+0.057237]. The earlier positive-first upstream
  reproduction AUC 0.727632 is affected by fixed position-wise tie jitter.
  Primary registered metrics/targets were not replaced with this diagnostic.
- Gains concentrate in the 131 users whose positive item-CF score is tied; four
  folds select 5% Jev weight and one selects 20%. This motivates independent
  validation of conservative tie-breaking, not tuning further on these users.
- Fixed evaluated-CSV list serialization so histories can be reloaded. All primary
  metrics, intervals, weights and fresh-subgroup values remained exactly unchanged.
- **195 tests passed in 5.89s**; compilation, shell syntax and whitespace checks
  passed. A full replay made zero HTTP calls and preserved score/cache bytes.
- Evidence: [`reports/microlens-real-200-2026-10-08`](../reports/microlens-real-200-2026-10-08/README.md).

## 2026-10-08 — First real MicroLens and public JEV-27B pilot

- Downloaded official MicroLens pairs (18,940,682 bytes) and English titles
  (1,500,488 bytes) through the official browser links after direct HTTP returned 403.
  Registered their SHA-256 hashes; raw data is not redistributed in Git.
- Audited 719,405 interactions, 100,000 users and 19,738 items. Reproduced the
  200-user item-CF baseline: AUC 0.727632, HR@5 0.490, NDCG@10 0.503160.
- No configured Jev API key/endpoint was available. Checked local resources (8GB
  Apple M1, no CUDA) and found the author's public AutoTrust demo rather than
  installing an infeasible 27B model or substituting a different model.
- Registered a 20-user subset and 400 pointwise Noul requests before real scoring;
  pinned model and deployment revision. Used 1 canary + 399 new calls, with the
  canary cached, single concurrency, ≥1s request-start interval and no hidden retries.
- All 400 real responses passed model/revision/schema checks; 0 failed slates.
  Batch wall time: 402.14 seconds. Backend timing is informational, not a serving test.
- Corrected rank normalization to average ranks for ties before comparison,
  preventing candidate-position artifacts in fusion. Pilot candidate order is frozen
  and shuffled; pointwise requests see no labels or competing candidates.
- Pilot AUC/NDCG@10: item-CF 0.6500/0.5055; Jev 0.6289/0.3923;
  fixed fusion 0.6895/0.4578; user-CV fusion 0.6237/0.4083.
  Fixed-fusion paired AUC gain CI [−0.0816,+0.1684] crosses zero. No stable win claimed.
- Added an explicit 200-user protocol scope gate; a real 20-user pilot cannot satisfy
  the main MicroLens research targets. No true-CTR calibration claim on sampled negatives.
- **195 tests passed in 5.23s**. Scoring replay makes 0 HTTP requests and preserves
  score/cache hashes; rebuilding from raw CSVs exactly reproduces frozen pilot inputs.
- Full evidence and instructions:
  [`reports/microlens-real-2026-10-08`](../reports/microlens-real-2026-10-08/README.md).
  No real six-objective ParetoJev or student model was trained in this phase.

## 2026-10-08 — Local restoration and unified baseline harness

- Restored the user-provided Git Bundle with complete history. Verified upstream
  HEAD `26ff82c5f6fa14825745e7a229a6b8cdb8c68cc0`; the original **49 tests passed**
  locally before new implementation. The recovered original author history is preserved.
- Added separately trained project EMER-like/CMR-like variants and a full ParetoJev
  harness loss with explicit preference-sensitive shared-score supervision. These
  are documented variants, not claims of original-paper reproduction.
- Registered strict NPZ identities, source/data/environment/config hashes, seed
  lists, equal optimizer budget, train-fitted cold threshold, and validation-only
  selection. Added both fixed EMER-like and CMR-like research comparisons.
- Added one shared evaluator, per-slate prediction/identity archives, conditional
  metric coverage, common-box Monte Carlo hypervolume and paired user bootstrap.
- Added finite-only Target Mode statuses and evidence requirements; real gates
  cannot pass on a synthetic or missing-provenance report.
- Fixed a checkpoint serialization failure found by integration tests, rejected
  simultaneous output writers/source drift, normalized byte-string metadata, and
  copied read-only behavior-label arrays before tensor conversion.
- Independent review tested same-seed tensor/history equivalence and unchanged
  selected checkpoints after flipping only test labels. No test labels guide tuning.

Validation of the implementation: **179 passed in 7.64 seconds**, no warnings;
source compilation, shell-script syntax checks and `git diff --check` passed.
The pinned dependency snapshot is `requirements-lock.txt` (local Python 3.12).
Model implementation commit: `2dbfb62`; protocol/runner/evaluator/gates: `de8ab72`.
Final frozen-source experiment details follow.

### Registered synthetic comparison and replay

- Frozen clean code commit: `5202bc35f1641a01b7aa4221fbcc67950bdfede2`.
- Generator seed 123, train/validation/test slates 192/96/96, K=12, M=6, F=10.
- Model seeds 7/17/29, four epochs, 30 fixed preference vectors, top-k=5,
  100,000 shared hypervolume integration samples and 200 user-bootstrap replicates.
- All nine independent checkpoints trained and passed exact prediction reload.
- Mean synthetic hypervolume: EMER-like 0.036373; CMR-like 0.045207;
  ParetoJev 0.034407. ParetoJev mean relative gains: −5.39% / −23.75%.
- Cold-NDCG relative gains: +6.17% / +10.40%; regrets: 0.02940 / 0.05280;
  mean controllability −0.14948. No method/config was selected from test results.
- Engineering run exited 0. All seven real research targets were `not_evaluable`;
  the explicit target-check command exited 3, as required for synthetic evidence.
- Resume preserved exact SHA-256 hashes of all nine checkpoints, all three
  evaluation reports and the summary. Regenerating all three input NPZs also
  produced identical file hashes.
- Source compilation, shell syntax and Git whitespace checks passed. There was
  no real dataset/API call, GPU serving benchmark or statistical-win claim.

Full configuration, registered source/environment/input hashes, seed-level results
and verification evidence are saved in
[`reports/synthetic-baseline-2026-10-08`](../reports/synthetic-baseline-2026-10-08/README.md).
The original integration probe is retained locally under `artifacts/comparison-trial/`;
it used the same fixed data/budgets and produced identical metric values. Checkpoint
and prediction tensors are local ignored artifacts; clean source plus registered
commands reproduce them.

## 2026-10-08 — Phase 0 bootstrap

### Environment observed

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- pytest 9.0.2

### Completed

1. Created Git repository and Python package.
2. Implemented data, model, Jev, loss, metric and target modules.
3. Added CLI, configs, docs and CI.
4. Ran compile check.
5. Ran `pytest`: **14 passed**.
6. Ran end-to-end synthetic smoke experiment.

### Smoke result

- initial held-out loss: `1.06637`
- final held-out loss: `0.96978`
- reduction: `9.06%`
- behavior Brier: `0.03795`
- Jev-like Brier: `0.01665`
- fused Brier: `0.01156`
- cold mean Jev gate: `0.65949`
- warm mean Jev gate: `0.65611`
- status: passed

The cold/warm gate separation is small but in the intended direction. This is sufficient for a smoke test, not for a model claim.

### Not executed

- MicroLens download and real baseline reproduction;
- official Jev API calls;
- self-hosted JEV-27B/VL inference;
- KuaiRand preparation and training;
- GPU latency benchmark.

Reason: the execution container did not have the datasets, API credentials, a running open-JEV endpoint or GPU. No real metrics were fabricated.

### Additional verification

- Editable installation succeeded with local build isolation disabled after switching the build backend to installed `setuptools`.
- CLI help and all registered commands loaded successfully.
- Final test count after adding the precomputed-data training path: **14 passed**.
- A direct attempt to download the official MicroLens pairs file from this container failed because outbound file download was unavailable; the repository therefore records no fabricated real-data result.

## 2026-10-08 — Phase A reproducibility hardening

### Protocol audit and corrections

- Matched the public MicroLens demo's occurrence-weighted negative sampler instead of
  silently sampling uniformly from deduplicated items.
- Matched the public demo's right-open time-window boundary.
- Added an explicit `upstream_jitter` tie policy and retained `stable` and
  independently randomized alternatives.
- Made pointwise Noul genuinely pointwise by hiding competing candidates from state;
  candidate content is carried in the individual question.

### Scoring reliability

- Added durable JSONL append, run signatures, resume, overwrite protection, error logs,
  latency/usage aggregation and provenance fingerprints.
- Resuming an output now rejects provider/model/mode/objective/scope mismatches.
- Added objective sub-selection so MicroLens Phase A can request only `click` instead
  of paying for unrelated multi-objective questions.

### Research-grade Phase A evaluator

- Added reciprocal-rank fusion as a label-free baseline.
- Added nested user-CV logistic stacking with raw score, within-slate rank, margin,
  and pair interactions. Raw interactions allow slate-level Jev confidence/entropy
  to modulate candidate-level Jev evidence.
- Added out-of-fold Choice temperature scaling and Noul Platt calibration.
- Added Brier/ECE/NLL diagnostics and reliability curves.
- Added user-bootstrap confidence intervals, paired differences versus a declared
  reference and model-complementarity/oracle diagnostics.
- Added `configs/targets_microlens.yaml` and a complete Phase A shell workflow.

### Verification

- `python -m compileall -q src tests`: passed.
- `pytest -q`: **24 passed**.
- Synthetic 30-user/20-candidate CLI run completed through calibration, RRF, outer-CV
  hybrid, nested-CV stacker, bootstrap, pairwise comparison, complementarity and
  provenance output.
- Synthetic values are deliberately not committed as research outcomes.

## 2026-10-08 — KuaiRand annotation-budget pipeline

- Added strictly prior item-exposure and user-history counts during chronological
  pseudo-slate construction; these fields are persisted in NPZ and remain optional
  when loading older NPZ files.
- Added balanced decision-annotation sampling over four joint dimensions:
  cold/mid/warm item history, short/medium/long user history, standard/random
  exposure and none/sparse/rich positive feedback.
- Added per-cell caps, a global slate budget and a per-user cap to prevent a small
  group of heavy users from dominating expensive Jev calls.
- Added selected-slate manifests, bucket inventory reports, provenance and direct
  JSONL generation using only the sampled slate IDs.
- Reworked `scripts/prepare_kuairand.sh` so it no longer creates Jev prompts for the
  entire training log by default.
- Synthetic CLI verification prepared slates, sampled eight balanced users and
  emitted exactly eight Jev JSONL records.
- Current test count: **26 passed**.
- Added `--prefix-log` for later KuaiRand time splits so prior item/user counts begin
  with the earlier training history; overlapping timestamps are rejected.

## 2026-10-08 — Preference-sweep Pareto evaluator

- Added checkpoint-level `evaluate-pareto` execution over precomputed validation slates.
- Added deterministic uniform, one-hot, pairwise and Dirichlet preference sweeps.
- Added maximize-oriented objective utilities, shared-unit-box Monte Carlo
  hypervolume, Monte Carlo standard error, front size, controllability, cold-item
  positive NDCG, per-objective regret and paired dominance diagnostics.
- Fixed Spearman tie handling to use average ranks; this matters because one-hot
  preference sweeps contain many repeated zero weights.
- Added stable objective names to real-data training checkpoints.
- Added a strict train → evaluate → interim-target script and documented that its
  static signal baselines do not substitute for trained EMER-like/CMR-like baselines.
- `python -m compileall -q src tests`: passed.
- `pytest -q`: **31 passed**.
- A synthetic checkpoint + NPZ CLI integration test writes a provenance-stamped
  Pareto report; these synthetic values are not research outcomes.


## 2026-10-08 — KuaiRand score materialization pipeline

- Added a versioned `SlateScoreTensor` interchange contract for behavior-model
  probabilities, independent of the final reranker NPZ.
- Added a strict historical-Bayes CPU bootstrap. Every candidate in a pseudo-slate
  is scored from the same history strictly before the slate start; current-slate
  outcomes are queued until a later slate and overlapping windows are de-duplicated
  by `(user, item, timestamp)`.
- Added optional earlier-period prepared slates for initialization, with strict
  non-overlap validation.
- Added explicit Jev-to-KuaiRand objective aliases and hard failures for missing,
  duplicate or permuted slate/candidate/objective rows.
- Added train-fitted numeric candidate features and immediate reload through the
  `PrecomputedSlateDataset` contract.
- Changed the default feature contract to exclude pseudo-slate sequence position;
  the legacy v1 contract remains replayable and position is available only as an
  explicit ablation.
- Added `scripts/materialize_kuairand_probe.sh` for train/eval materialization with a
  shared feature specification.
- Added prequential Brier diagnostics with explicit warnings that they are neither
  independent held-out metrics nor a substitute for SASRec+MMoE.
- `python -m compileall -q src tests`: passed.
- `pytest -q`: **39 passed**.
- No real-data metric is claimed; all new integration checks use small synthetic
  prepared slates and fake decision rows.

## 2026-10-08 — Leakage-safe neural behavior branch

- Added `split-kuairand-log`, which creates chronological CSV partitions without
  dividing equal-timestamp groups and can emit cumulative `prefix_before_*` files.
- Added a train-fitted item vocabulary with explicit padding/OOV indices and a
  train-fitted pre-exposure context transform.
- Added strict event histories: every target sees only timestamps `< target_time`;
  simultaneous events neither enter one another's SASRec history nor prior counts.
- Added formal SASRec + MMoE training with chronological validation, early stopping,
  unweighted BCE, per-objective AUC/AP/Brier, macro diagnostics and provenance.
- Added checkpoint-based slate scoring into the versioned `SlateScoreTensor` contract,
  including candidate/history OOV fractions, Brier diagnostics, throughput and p50/p95
  batch latency.
- Changed SASRec histories to left padding and made the encoder select the actual last
  non-padding position, preserving recency alignment.
- Reworked `scripts/prepare_kuairand.sh` to reserve disjoint behavior-train,
  behavior-validation and reranker-train periods, preventing in-sample pXTR leakage
  into the downstream Pareto reranker.

### Verification

- `python -m compileall -q src tests`: passed.
- `bash -n scripts/*.sh`: passed.
- `pytest -q`: **46 passed**.
- A synthetic CLI chain completed:
  `split-kuairand-log → train-kuairand-behavior → prepare-kuairand →
  score-kuairand-behavior`, producing an aligned `[8,3,6]` score tensor.
- Synthetic quality and latency values are integration diagnostics only; no real-data
  recommendation metric is claimed.

## 2026-10-08 — Behavior-informed Jev annotation policy

- Extended the balanced KuaiRand annotation sampler with an optional full-slate
  `SlateScoreTensor` from the formal behavior model.
- Added three pre-Jev active-learning signals: normalized Bernoulli entropy,
  top-1/top-2 margin uncertainty and cross-objective winner disagreement.
- Negative-feedback winner/margin calculations use `1 - p(hate)` so all objectives
  share a maximize-oriented utility convention.
- Added a configurable interpolation between seeded random within-cell ordering and
  active priority. The recommended first probe uses strength `0.5`; strength `0`
  exactly keeps the structural sampler and strength `1` is a high-uncertainty ablation.
- Added exact slate/candidate/mask/objective alignment checks before a behavior score
  file can influence sampling.
- Added inventory-versus-selected priority diagnostics and priority-bucket counts to
  the manifest report.
- Added `scripts/sample_kuairand_behavior_informed.sh` and documented that this is a
  teacher-training annotation subset, not a population evaluation sample.

### Verification

- `python -m compileall -q src tests`: passed.
- `bash -n scripts/*.sh`: passed.
- `pytest -q`: **49 passed**.
- Synthetic tests verify that strength `1` selects the intended high-uncertainty
  slates and that malformed score shapes/alignments fail closed.
