# Execution Log

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
