# Execution Log

## 2026-10-13 — Expand the underpowered EMER/JeV pilot

- User approved larger chronological training, validation, independent testing,
  power planning and sustainable real Jev production. No new model framework.
- Recomputed paired user-cluster delta-method precision from the already-opened
  600-slate pilot, averaging the fixed three seeds first. A 1% long-view relative
  effect projects to 43,459 slates only under unchanged design/noise assumptions.
  Added user-correlation sensitivity; no claim that 50K rows guarantee power.
- Prepared April 19–20 training (30,000 slates, 11,427 users) and April 21
  validation (6,000 slates, 4,253 users); union 12,606 users. All 360,000 candidate
  records, labels and timestamps matched the source store; sampling weights and
  strict Jev histories verified. No April23+ observations opened.
- Generalized the existing sampler to two development splits; kept all original
  three-split defaults intact. Added explicit long-view question override and an
  optional metadata RAM copy to avoid random external-drive reads.
- Materialized 65 train-normalized common features and seven frozen behavior
  predictions. Started 12 EMER trials and 252,000 native Jev requests separately;
  shell dependency starts the 12 Jev-arm fits only after complete real scores.
  Missing Jev is an ignored structural baseline block, never a fabricated signal.
- Public service single-request probe succeeded with the registered self-reported
  revision. Total nominal train/validation/test request plan is 602,000; one-second
  pacing implies approximately seven days before failures/downtime/overhead.
  No new API credential is required for current public execution; faster resources
  remain an optional user input. Workers need this computer and external drive.
- Added tests for clustered information not growing with duplicate rows, power
  scaling, weighted two-split sampling and rejection of missing/misaligned Jev.
  A full run exposed an unrelated flaky harness test relying on id() uniqueness
  after object collection; retained model references in the test, not production.
  All 277 tests then passed. Source scripts compile and shell syntax checks pass.
- In a progress message, positive-list counts were mistakenly stated as 712/655;
  promptly corrected to audited validation follow/forward counts 506/521. Reports
  use only audited figures. No new effectiveness result or completion claim.

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


## 2026-10-08 — Real SASRec and semantic development controls

- Registered implementation and experiment at commit `611372a` before training.
  The 400 previously examined users became development users; 1,000 separate users
  select checkpoints. Both groups are excluded from SASRec and refitted CF training.
- Trained a real SASRec-style next-item model on 57,845 users / 503,713 interactions /
  442,815 valid targets. CPU, four threads, fixed seed, two learning rates, maximum
  30 epochs, patience five, unseen-item negative sampling. This is not six-objective
  MMoE training or an exact original SASRec paper reproduction.
- Learning rate 0.001 stopped after epoch 20 and selected epoch 15; rate 0.0003 ran
  30 epochs, best epoch 28. Validation NDCG@10 0.548039 vs 0.547204; select the former.
  CF on identical validation/training users is 0.485438. Both saved checkpoints reload
  with exact validation predictions. Total training 418.16 seconds.
- Downloaded the pinned MiniLM revision with safetensors; no remote model code.
  Encoded 19,738 titles into 384 dimensions in 24.67 seconds, including model load
  but excluding download. All 72 missing titles use zero vectors.
- Reused 8,000 genuine cached Jev candidate scores; **zero new Jev requests**.
- Development AUC / HR@5 / NDCG@10: CF 0.681447 / 0.538054 / 0.467663;
  SASRec 0.762237 / 0.590000 / 0.507418; SASRec+20% Jev
  0.770132 / 0.621250 / 0.513021; SASRec+20% MiniLM
  0.757763 / 0.598750 / 0.487814. All registered methods appear in the report.
- Development pointwise 95% AUC difference interval for 20% Jev vs SASRec is
  [0.001316, 0.014276], vs matched MiniLM [0.005592, 0.019145]. NDCG gain vs
  SASRec includes zero. These descriptive intervals are not independent or
  multiplicity-corrected confirmatory tests.
- SASRec has no within-user exact ties across 8,000 candidates. A 5% rank weight
  cannot strictly reverse an adjacent rank among 20 strictly ordered candidates;
  it can only introduce a limiting tie. Thus the old CF 5% rule barely changes
  SASRec. Registered development gates are **1/3**, with both 5% increment gates
  failing. The preregistered 20% result does not replace those gates.
- `203 passed in 7.33s`; model tests cover causal histories, right-padding
  invariance, empty/OOV inputs, strict-time labels, negative exclusion, finite
  gradients and actual learning. Raw audit rebuilt the training arrays, checked
  all 1,400 validation/development slates and verified train/validation/development
  user isolation. HTTP-blocked replay reproduces predictions, per-user metrics,
  aggregates and intervals exactly. Full training was not independently repeated.
- Report: `reports/microlens-sasrec-development-2026-10-08/README.md`.
  Original project targets remain unmet; no full ParetoJev performance claim.
  Next independent study requires a newly excluded user cohort and corresponding
  genuine Jev scores, not relabeling either development or checkpoint-validation
  users as unseen tests. No external input blocks the engineering completed here.


## 2026-10-09 — Fixed 700-user confirmation of 20% Jev fusion

- Registered at `8a4b708`: 700 previously unevaluated users, fixed 20% rank fusion,
  two AUC comparisons with 97.5% two-sided Bonferroni intervals; no interim test
  evaluation, optional stopping or sample expansion. Sample size was chosen from
  development variance; the planning approximation was not a success guarantee.
- Excluded 700 test + 400 development + 1,000 validation users from current model
  training. Refit SASRec from scratch on 57,145 users / 497,581 interactions /
  437,383 targets, fixed learning rate 0.001 and epoch 15 (119.43 seconds).
  Validation NDCG@10: 0.537947 vs matched CF 0.485753, diagnostic only.
- Behavior/semantic predictions frozen at `cf2f3d1` before test evaluation. Raw
  regeneration produces identical train NPZ, test candidates, Jev input and
  validation candidates. All real prompts fit the existing service limits.
- Initial collection stopped after one TLS unexpected EOF: 177 complete users,
  3,545 valid responses, 3,546 attempts. All cached responses verified before
  continuation; no interim outcome metrics examined. Commit `eeaf708` records a
  separate operational amendment adding at most 20 transport attempts. Research
  registration, users, prompts, model, weight and tests were unchanged. Certificate,
  HTTP and model-revision failures are not accepted by the resume exception.
- Completed 700 users / 14,000 real decisions using 14,001 attempts; the partially
  scored user's five cached responses were reused. Active collection time totals
  14,324.94 seconds (3.98 h), excluding repair waits and including the one-request/s
  limit; not an online serving benchmark.
- AUC / HR@5 / NDCG@10: CF 0.703233 / 0.569349 / 0.489881;
  SASRec 0.779549 / 0.641429 / 0.533152; MiniLM fusion
  0.783083 / 0.649048 / 0.530522; Jev fusion 0.785301 / 0.647381 / 0.539898.
- Primary Jev vs SASRec AUC +0.005752, 97.5% interval [0.000413, 0.011128];
  vs MiniLM +0.002218, interval [-0.003459, 0.007895]. **1/2 gates passed, overall
  failed.** Secondary HR/NDCG intervals contain zero. Do not claim equivalence,
  Jev superiority over the semantic control, or full ParetoJev architecture success.
- All final scores match raw responses at the pinned revision. Raw split/history
  audit and exact checkpoint/evaluation replay pass. HTTP-blocked collection
  replay skips all 700 users; score/cache/attempt-ledger hashes and historical
  collection costs remain unchanged. `209 passed in 7.64s`.
- Source and detailed evidence: `reports/microlens-sasrec-confirmatory-2026-10-09/`.
  No external input remains unresolved for this completed study.

- Exported a 52,842,522-byte offline data package containing 21 required ignored
  files, with per-file SHA-256s and no environment or credentials. Restored source
  commit `5a5feef` plus that package into a separate temporary directory; confirmed
  the restored package was imported and blocked HTTP throughout. Cached collection,
  checkpoint predictions, all metrics/intervals and historical costs replay exactly.
  Verification: `portable_replay.json`; full training and cross-hardware replay were
  not independently repeated.


## 2026-10-09 — User-selected KuaiRand-27K acquisition readiness

- Switched the active config to complete 27K standard shards and random track;
  moved the old Pure script/config to explicit legacy names. No user/item
  downsampling or automatic 1K/Pure fallback was introduced.
- Initial storage check: about 4.3 GiB free, later about 10 GiB after system space
  fluctuations; no large writable volume mounted. Requested a data directory
  with roughly 100 GiB workspace headroom while completing independent work.
- Verified official archive HTTP Range support and size 9,892,191,178 bytes.
  A 64 KiB prefix established layout; a 2 MiB prefix contained the entire
  3,519,028-byte user table. Actual 27,285 distinct users and 30 non-ID fields;
  is_lowactive_period is constant, onehot_feat4 has 874 missing entries, and
  onehot_feat12..17 each have 714. No snapshot timestamp exists in the table.
  Whole-archive MD5 remains unverified because the full archive was not downloaded.
- Implemented checksum-pinned/resumable downloads with disk checks, bounded-space
  writes and no TLS bypass; atomic extraction rejects unsafe paths and links.
  Full 27K audit uses bounded CSV chunks, checks labels/IDs/track/date ranges and
  emits feedback counts by date/tab, explicit prefix scope and source hashes.
- Whole-month item statistics are excluded from strict temporal model features.
  Published 30/62 feature counts are not treated as validated safe input counts.
  Current legacy behavior training remains in-memory with only a few context
  features; full rich-feature 27K model training is explicitly still unfinished.
- 226 tests passed in 7.44s. A real CLI download attempt was rejected before
  accessing the large file due to storage, leaving no large partial archive.
  No new Jev inference was requested. Report: reports/kuairand-27k-readiness-2026-10-09/.

- A later repeat prefix probe hit a read timeout. The previously acquired complete
  user table and matching SHA-256 remain intact; no new large download or model
  training was performed. Network reachability is intermittent, not continuously
  guaranteed by the earlier successful range response.


## 2026-10-11 — Full 27K acquired on user-provided external disk

- Verified write/fsync/atomic rename and locks on `/Volumes/曾诚的移动硬盘/`
  with about 883 GiB free. Created the dedicated `ParetoJev/kuairand-27k` directory;
  other drive files were not changed. The default project data path now links there.
- Downloaded and verified the published MD5 of the 9,892,191,178-byte core archive,
  3,222,932,768-byte captions and 3,687,928,520-byte categories. Two long category
  streams timed out; exact 32 MiB ranges with bounded transport retries completed
  the retained prefix. No TLS bypass, source substitution or checksum relaxation.
- Per-phase locks allow immutable core extraction alongside supplement download;
  partial extraction cannot be audited. Extracted all 11 files, 49,005,793,097 bytes.
  Original loader code was not executed.
- Full log scan: 136,296,576 early standard + 185,981,809 late standard + 1,186,059
  random exposures. The 322,278,385 standard and 1,186,059 random counts exactly
  match the publisher. All log SHA-256s match the extraction manifest.
- Full-period user exposure count median 8,646; all 27,285 users have at least 51
  full-period exposures, not necessarily 50 pre-event or positive-feedback events.
- Found 27,310,928 duration/play-time validity flags (8.4433% of exposures) and
  1,600,795 long_view/formula mismatches (0.5405% of remaining valid-duration rows).
  Raw labels remain unchanged. Further scene-specific definition checks are needed.
- Caption scan: 32,038,725 unique IDs, no duplicate rows; 26,420,471 have nonblank
  caption or cover text (82.4642% of catalog). Exposure-weighted join coverage and
  category value quality are not yet established. User-table SHA matches the prior
  bounded probe; metadata snapshot timing and whole-month aggregate risks remain.
- 233 tests passed in 7.84s. Plan output now reports recorded acquisition/extraction
  availability instead of hard-coding false; it does not claim a fresh hash check.
- No new Jev calls or trained 27K model. Next work is explicit feature/label timing
  and scalable rich-feature pXTR training, not a recommendation win claim.
  Evidence: `reports/kuairand-27k-acquisition-2026-10-11/`.

## 2026-10-11 — First real 27K seven-target behavior pilot

- Registered fixed periods, seed, target budgets, two feature regimes, three epochs,
  baselines and development gates before fitting (`9906097`). Training code/runtime
  provenance records `fa77eb1`; subsequent diagnostic-cutoff parameterization did
  not change this run's already materialized samples or model implementation.
- Re-read both complete early standard shards, verified their SHA-256 against the
  full audit, and built a bounded-memory per-user disk index for 136,296,576 events.
  Per-user stable sorting and strict `< target.time_ms` histories exclude all tied
  timestamps. Histories contain exposure IDs, not delayed feedback labels.
- Fixed per-user sampling yielded 426,832 training targets from 26,758 users and
  197,078 validation targets from 24,892 users. Inverse-inclusion weights represent
  99,550,635 and 18,036,928 exposures respectively. This is target-sampled training,
  not full-exposure fitting. History length is 50 in 94.83% / 99.63% of samples.
- Training-period validity flags are 8,902,416 zero-duration rows, with no negative
  durations or play times. On valid rows, 477,208 provided-long-view positives fail
  the comparison formula and 27 differences go the other way; 98,034 differences
  are within 100 ms of the threshold. Kept every published feedback label.
- Raw date partitions and time_ms are not fully aligned: 341 sampled training
  targets precede nominal April 8. Actual sampled training starts April 7 at
  23:11:25.486 +08:00. Kept the predeclared timestamp-only boundaries, and documented
  actual ranges rather than changing the protocol after looking at results.
- Trained SASRec-style + MMoE on seven raw labels, fixed 65,536 item hash buckets,
  a 32-dimensional one-layer encoder, CPU, three epochs and weighted BCE. Primary
  event-only inputs have seven fields; the sensitivity model adds 30 user profile
  fields whose snapshot timing is unknown. Month-wide item aggregates are excluded.
  User IDs and profile vocabularies use training-fitted OOV handling.
- Primary model: best epoch 1, AUCs 0.689849 / 0.705263 / 0.671814 / 0.627398 /
  0.652133 / 0.620929 / 0.554861 (click, long-view, like, follow, comment, forward,
  hate). Six of seven minimum development gates pass; forward fails. Only three
  of seven log losses beat the stronger tab-prevalence baseline. Click and long-view
  mean probabilities are 28.76% / 18.69% versus weighted prevalence 38.25% / 26.46%.
- Profile sensitivity: best epoch 1, seven of seven minimum gates pass, but this
  is not temporally verified evidence. Both regimes overfit: primary validation
  mean log loss 0.182351 -> 0.186803 -> 0.227860; profile 0.180060 -> 0.186155 ->
  0.222772. Do not extend epochs or lower gates to manufacture success.
- Both best checkpoints and all seven metrics replay exactly in a fresh CPU
  process. This is checkpoint/inference replay, not a second complete training
  run or cross-hardware reproduction. Source, config, samples, checkpoints and
  predictions have recorded hashes; heavy files stay on the dedicated external disk.
- 239 tests passed. Added coverage for unsorted shards, same-time exclusions,
  future-record invariance, inverse sampling weights, raw-label retention, source
  integrity, numerical gradients with empty histories, OOV users/profiles and
  training-bound diagnostic cache validation. Synthetic trainer/replay smoke also
  passed separately; its scores are not included in real evidence.
- Corrected the README to put the 27K pilot first and label Pure/in-memory commands
  as legacy. The seven raw outputs do not silently replace the six-objective
  reranker contract. Real Jev/ParetoJev comparison and research gates remain pending;
  this run made zero Jev requests. No external input currently blocks behavior work.
- Evidence: `reports/kuairand-27k-behavior-pilot-2026-10-11/`. Next: better training
  coverage and available item/history features, regularization and independently
  assessed calibration, followed by frozen out-of-period reranker features.

## 2026-10-11 — Richer 27K controls and separate calibration period

- Preregistered v2 (`eb9eafc`) before fitting: 64 training targets/user, eight per
  calibration/evaluation day, seven raw objectives, identical data for two feature
  regimes, 64 fixed boosting iterations, positive-slope Platt calibration and
  stricter gates against tab prevalence. Research target mode remains disabled.
- Verified all 32,038,725 unique basic-video IDs and source SHA-256. Built a dense
  external-disk index plus a local lookup cache. Unknown/invalid durations:
  6,758,072 catalog rows; unknown upload dates: 579. Future/missing upload dates
  mask candidate attributes, and historical authors are masked at their own times.
  Physical/author/upload attributes are explicitly assumed static, not proven so.
- Prepared 1,699,506 train targets (26,758 users), 188,642 April-18 calibration
  targets (24,045 users), and 187,419 April-19 evaluation targets (23,881 users).
  Weights represent 99,550,635 / 9,039,832 / 8,997,096 exposures. This is sampled
  target training over complete early histories, not full-exposure fitting.
- Exposure-only control: 18 request/history derived inputs; content control adds
  eight upload-property/author-match inputs. No user snapshots, month aggregates,
  past feedback labels or Jev signals. Raw user/item/author IDs are not ordinal
  tree features. This is a new independent-head tree baseline, not a SASRec refit.
- First startup (`run-v1`) was interrupted in repeated mmap-backed column access
  without completing a model group or evaluation. Changed only the loader to
  materialize bounded arrays (512 MiB cap per period). Synthetic predictions,
  calibration parameters and metrics match the mmap implementation exactly.
  Successful frozen run: `behavior-v2/run-v2`, training source `3facc4f`; fitting,
  calibration and evaluation took 258.6 seconds after loading/baseline setup.
- Both regimes pass 6/7 stricter point-estimate gates; is_hate fails log loss and
  Brier against tab prevalence. Content-model AUCs: click/valid-play 0.7567,
  long-view 0.7777, like 0.7681, follow 0.7446, comment 0.7181, forward 0.6771,
  hate 0.5745. Click/long-view calibrated means are 38.525% / 26.712%, versus
  weighted evaluation prevalences 38.334% / 26.689%. Positive-slope calibration
  changes no observed AUC; like log loss slightly worsens, which is retained.
- Content-minus-context calibrated log-loss point differences are negative for
  all tasks. Of seven exploratory 95% user-bootstrap intervals, only click, like
  and hate exclude zero. No multiplicity correction or unseen-test claim: April
  19 had already been used in v1 development. No Jev/ParetoJev gain is established.
- Shared evaluator exactly reproduces both v1 reports. 246 tests pass. Real raw and
  calibrated predictions plus objective/scene metrics replay exactly in a fresh
  process; the 1,000-replicate bootstrap results replay exactly as well.
  Exported a 48,981,162-byte frozen-evaluation package and independently restored
  it into a fresh directory; both models' predictions and metrics replay exactly.
  No complete retraining or cross-hardware replay was performed.
- Training-only consistency check: log/basic duration missingness agrees on
  426,823 of 426,832 v1 training targets; nine differ. This supports consistency,
  not proof of online attribute availability. Full-period post-experiment label
  audit revalidated the entire event-store SHA and exposure totals. It found
  48,688 training hate positives versus 856 sampled, 5,597 calibration positives
  versus 92 sampled, and 4,471 evaluation positives versus 85 sampled. True full
  hate prevalence changes from 0.06191% on April 18 to 0.04969% on April 19, while
  the sampled evaluation estimate is 0.0432%. Original model/gate decisions remain
  unchanged; full source-offset positive indices and hashes are prepared for a
  separately registered sampling study. This was a label audit, not full-population
  model evaluation. April 20 onward was not used for model selection/evaluation.
- Next: increase sparse positive coverage with correct inclusion weights and
  distinguish model weakness, sampling variability and temporal change. Later
  reranker exports must freeze a common slate decision time and explicitly map
  the seven raw objectives to any six-objective protocol. No new external input
  blocks this work; zero Jev requests were made this round.
- Evidence: `reports/kuairand-27k-behavior-v2-2026-10-11/`.

### 2026-10-12 — Aligned behavior controls (completed)

- User corrected the priority: match behavior-model data/information/evaluation
  before rare-feedback optimization. Registered `configs/kuairand_27k_aligned.json`
  and `reports/kuairand-27k-aligned-2026-10-12/REGISTRATION.md` before fitting.
- Recovered all exact v2 target occurrences from the SHA-verified early event
  store. User/time/seven labels/inclusion weights match every row; source record
  indices disambiguate ties. All last-50 histories are strictly earlier.
- Frozen v2 tree is reused, not refit. Tabular MMoE receives the same 26 source
  features; SASRec MMoE adds ordered item IDs and user/candidate IDs. The latter
  tests a richer package, not a pure architecture contrast. No rare oversampling.
- Both neural arms register fixed three epochs/final checkpoint, independent
  April 18 calibration, and unchanged April 19 development evaluator/gates.
- Initial source `899dbe5` completed/froze tabular MMoE. SASRec's unfinished first
  epoch was interrupted under system memory pressure; source `066b0c0` replaced
  resident arrays with local SSD file-backed arrays. Random gathering remained
  slow and was interrupted before any SASRec checkpoint/evaluation. Source
  `2ae8686` stages exactly the same three permutations and reads batches
  sequentially; tabular checkpoint is retained and SASRec restarts from seed.
  Exact synthetic predictions/calibration/metrics/bootstrap parity holds for both
  I/O changes. No new neural evaluation was inspected during these amendments.
- Sequential throughput restored to about 200 batches/33 seconds at startup.
  252 tests pass. Synthetic frozen archive restored into a fresh directory and
  all predictions/metrics/bootstrap replay exactly. This synthetic verification
  is engineering evidence only; real training was still in progress at that checkpoint.

- Real aligned comparison completed: tree/tabular MMoE/SASRec MMoE all pass 6/7
  unchanged development gates; hate fails all three. Identical-information MMoE
  is close to the tree, with worse click/long-view/like calibrated NLL (exploratory
  user-bootstrap intervals exclude zero). Richer SASRec has worse click/long-view
  but better like/follow/forward NLL; those five exploratory intervals exclude
  zero. Comment/hate differences remain uncertain. No multiplicity correction.
- AUC tree vs SASRec: click .7567 vs .7149; long-view .7777 vs .7364; like .7681
  vs .7816; follow .7446 vs .7661; comment .7181 vs .7049; forward .6771 vs .7359;
  hate .5745 vs .5672. No overall SASRec or architecture superiority claim.
- Tabular MMoE has 24,231 parameters; sequence/ID arm 2,383,639. SASRec training
  log loss declines .16624 -> .15523 -> .13332; this does not establish its best
  epoch or diagnose the cause of weaker development click/long-view. Final epoch
  and registered calibration are retained even when metrics worsen.
- All real raw/calibrated predictions, per-objective/scene metrics and all three
  1,000-replicate paired user-bootstrap comparisons match exactly on fresh-process
  replay. No full retraining or cross-hardware replication is claimed.
- Evidence and interpretation: `reports/kuairand-27k-aligned-2026-10-12/`.
  Next neural improvement should use a preregistered time holdout inside training
  and separate ID/sequence controls; do not prioritize making hate pass or claim
  a Jev/ParetoJev increment. Existing results remain frozen.
- Portable real frozen-evaluation archive restored into a fresh directory; all
  predictions, objective/scene metrics and bootstrap intervals again match exactly.
  Removed only this round's reconstructible local array/epoch cache after verification;
  original external-drive data, sequence artifacts, models and results are retained.


### 2026-10-12 — Core real Jev validation in progress

- User explicitly prioritized the core ParetoJev-vs-no-Jev-baseline question and
  delegated implementation choices. No further upstream behavior tuning was done.
- Registered a seven-raw-objective, six-arm experiment before sample selection:
  EMER-like/CMR-like/ParetoJev, each with/without Jev, plus a fixed behavior-sum
  sanity control. Shared frozen v2 tree, 3 seeds, 20 epochs, validation selection,
  original 2% HV / 3% cold NDCG / .005 regret / .60 controllability point thresholds.
- Selected 1,000/400/600 train/validation/test ten-candidate slates from full early
  27K logs, representing 499,429/152,285/351,756 eligible slates with exact inverse
  stratum weights. Common decision-time history prevents candidate-specific
  future history. Independent audit matched all 20,000 source records, labels,
  weights, candidates and strictly prior histories across 1,689 selected users.
- Joined 35,618 distinct candidate/history items to the complete caption table.
  Outcome-free prompts have max state length 2,371 characters. Missing/truncated
  text is recorded without replacing samples. Native Choice scores are relative
  distributions, explicitly not event pXTR. Pin the same public JEV-27B revision.
- Added no-Jev path/loss invariance and weighted evaluation. Weighted metrics and
  grouped bootstrap match explicit integer-weight replication. 260 tests pass;
  staged six-arm synthetic fit/evaluation/replay passes as engineering evidence.
- Nine real no-Jev checkpoints are frozen. Test model comparisons have not run.
  Real Choice scoring is active (14,000 planned decisions, bounded 15,400 total
  attempts, concurrency one). Durable reservation ledger and raw response cache
  preserve progress across transport failures.
- A verified-PID dependency driver automatically audits complete responses,
  materializes real-score tensors, trains remaining arms, compares all methods,
  and replays predictions/metrics/intervals. Final artifacts still require review;
  no research conclusion or goal completion is claimed at this checkpoint.
- Live handles, paths and anti-duplication notes are in
  `reports/kuairand-core-2026-10-12/STATUS.md`. Do not change frozen numerical code
  or dependent scripts while the run is in progress.


### 2026-10-12 — Core validation completed; negative overall conclusion

- All 14,000 registered real Choice decisions completed in 14,000 HTTP attempts,
  pinned to the same JEV-27B revision. Raw request/response/tensor audit passes.
- All 18 arms/seeds executed 20 epochs; checkpoints were selected using validation
  only. Seed 17 ParetoJev retained epoch zero, documenting unstable training benefit.
  No seed/checkpoint was replaced after seeing test results.
- Primary registered Monte Carlo result: mean HV relative gain +5.87%, cold NDCG
  +2.06%, worst regret .052266, controllability .117699; only 1/4 point gates pass.
- Completion review found small seven-dimensional HV (.0003–.0015): 200,000 unit-cube
  samples produce only tens/hundreds of hits, too coarse for narrow gain claims.
  Preserved every original report and performed a numerical-only exact-HV refinement
  using isolated moocore 0.3.2. Its maximization/origin semantics passed independently
  enumerable 2D/3D/7D inclusion-exclusion fixtures. NumPy remained 2.5.3 in the frozen
  training environment; no checkpoints, preferences, labels, reference or gates changed.
- Exact primary HV relative gains: +59.79%, -35.84%, +4.43%; registered mean-of-ratios
  +9.46%. Supplementary ratio of mean absolute HVs +0.27%, not substituted for the
  registered aggregation. Primary gate count remains 1/4. ParetoJev exact mean HV
  relative changes: -48.05% vs no-Jev CMR-like, -8.45% vs same-Jev EMER-like,
  -45.90% vs same-Jev CMR-like, -64.05% vs the fixed behavior weighted sum.
- Cold NDCG favors ParetoJev over several comparators, but within-ParetoJev Jev/no-Jev
  cold intervals cross zero for every seed. Thus local cold gains do not prove that
  Jev caused them or that the full framework wins overall. Full uncertainty and
  per-seed tables are retained. EMER-like/CMR-like are still project variants.
- Independently verified all 18 checkpoint/receipt hashes, earliest validation
  selection, real no-Jev checkpoint invariance, common tensors and source identities.
  Independently recomputed weighted objectives, cold metrics, regret and controllability
  from predictions. Original process replay is exact for predictions/metrics/intervals.
- Exported the frozen tensors/checkpoints/predictions and original real Jev cache.
  Verified archive/file hashes after fresh-directory extraction; all three seeds'
  predictions/metrics/intervals and all 14,000 raw decisions replay independently.
  Full upstream raw logs are deliberately excluded from the lightweight package;
  no complete retraining or cross-hardware bitwise reproducibility is claimed.
- Core hypothesis validation is finished; overall superiority is not supported.
  Quantitative research and deployment targets are not relabeled as achieved.
  Final interpretation and requirement-by-requirement evidence:
  `reports/kuairand-core-2026-10-12/README.md` and `completion_audit.json`.

## 2026-10-13 — EMER + Jev versus EMER

- Implemented an independent paper-aligned public-data EMER adaptation: normalized
  signal ranks, comparative Transformer, posterior feedback-count supervision,
  frozen-behavior prior ranking losses and previous-model Mean@3 advantage ratios.
  Both arms have 43,345 parameters; Jev changes inputs only, with a common search
  space and budget and separate validation-only selection.
- Discovered and fixed a training-constant feature scaling defect. Original core
  validation numeric values reached 678,447.875 after a weekday change; the new
  training-only encoder uses scale one for constant fields (maximum about 13.612).
  Preserved original results and added a post-run correction instead of rewriting
  their audited bytes or attributing all failures to this one defect.
- Completed 24 development fits (3 seeds, 4 settings, 2 arms, 30 epochs) on 1,000
  train / 400 validation pseudo-slates. Excluded the old core test from development.
  Frozen a fresh April 22 test of 600 ten-candidate pseudo-slates from complete
  late standard logs, with exact stratified inverse-inclusion weights.
- Collected 4,200 real Choice decisions in 4,200 HTTP attempts at a fixed JEV-27B
  revision. Source records, labels, weights, features and histories match; all
  requests/responses and candidate/objective mappings pass audit. Verified all
  24 training receipts and earliest validation epoch/config selection.
- Primary long-view NDCG averages 0.390850 -> 0.377369 (-3.45% relative), with
  absolute-gain 95% paired-user interval [-0.036197, +0.010153]. Click changes
  -1.43%; like/forward point gains have intervals crossing zero. Incremental
  superiority is not established; production success is not claimed.
- Cached-input M1 CPU ranking p95 is 0.428/0.467 ms. Seven serial public-service
  HTTP decisions have median 4.41 seconds, excluding pacing and backoff. These
  measurements do not establish complete recommendation RT or an online uplift.
- All 268 tests passed. Independently reconstructed scalar metrics, replayed all
  predictions/2,000-bootstrap intervals exactly, exported an 82-file portable
  archive and verified its 81 payload files and fresh-directory replay. The full
  source logs are excluded; no full retraining/cross-hardware bitwise proof claimed.
- Reported all four existing matched-config comparisons as post-confirmation
  diagnostics only; they do not replace the registered final models or conclusion.
  Evidence: `reports/emer-jev-2026-10-13/README.md` and `completion_audit.json`.

## 2026-10-13 — Signal diagnosis and long-view definition repair

- Audited weighted within-slate AUC, top-three metrics and complementary pair
  errors on existing April 20/21/22 data. April 22 is explicitly development now.
  Original Jev long-view AUC is about 0.50, below goal-specific behavior predictions.
  Correcting some wrong pairs is not evidence that an oracle-selected mixture is deployable.
- Registered a real 400-slate validation probe that changes only the long-view
  question. It now specifies completion for videos <=18 seconds, otherwise at
  least 18 seconds, and treats zero duration as unknown. Observed labels are not
  rewritten. Validation AUC improves 0.501152 -> 0.568800; NDCG improves
  0.331146 -> 0.402383; both paired-user difference intervals exclude zero.
- After the registered signal-quality gate passed, collected 1,000 training and
  600 reused-development corrected scores. In all three periods, old/new signal
  AUC and NDCG difference interval lower bounds are positive. These are development
  findings, not new confirmation or proof of incremental value over behavior.
- Tested bounded Jev-only corrections with frozen EMER checkpoints. Original seven
  signals with seven coefficients give -0.10% long-view change; corrected single
  signal with one coefficient gives -0.02% (absolute -0.000077802, 95% interval
  [-0.003867474, +0.003266609]). Click/hate guards pass, gain conditions do not.
  Coefficients/regularization are selected on April 21 only; zero correction is
  always eligible. No new date is opened when development evidence is absent.
- Added a configurable late-history start to future cohort preparation: April 23
  histories would include April 22 exposures. April 23 was not prepared/scored.
- All 2,000 corrected native responses and source candidate mappings audited;
  all three baseline checkpoint hashes unchanged. Both correction variants replay
  predictions, metrics and 2,000-bootstrap intervals exactly from saved coefficients.
- Verified 38 payload hashes in a 39-file portable archive and repeated both
  replays in a new directory. All 272 tests pass, including analytic-gradient,
  zero/uniform-signal, padding and permutation checks. The uniform-rank centering
  fix removes only floating-point offsets; the old residual results were replayed.
- Development phase complete. Signal repair is supported; stable EMER uplift,
  production RT and online business gains remain unproved. Evidence:
  `reports/jev-signal-diagnosis-2026-10-13/README.md`.


## 2026-10-13 — Native Jev grouped inference and authorized budget extension

- Replaced unstable cross-slate BF16 batching with within-slate shared-prefix
  inference, 64-token cache boundaries and independent seven-question branches.
  Pinned publisher checkpoint unchanged; FP16 compute and FP32 residual/head
  explicitly registered as a new producer, separate from public-service scores.
- v9 probe: 144 HTTP requests / 336 real decisions; maximum absolute probability
  difference 0.0009060204 versus uncached reference, below unchanged 0.01 guard.
  16-worker probe about17.32 decisions/s; no recommendation-effect claim.
- User authorized TOTAL65 CNY, excluding future independent test. Verified AutoDL
  shutdown 2026-10-14 04:20 +08; collector resumed with cutoff04:00. Immutable
  execution-extension receipt preserves the original scientific registration.
- Backed up 3,303 actual HTTP groups /23,121 child decisions at23:26. Collection
  still active toward36,000 slates /252,000 decisions, validation first. Source
  hashes match deployment. 283 local tests pass; independent test remains closed.
- Evidence and exact producer registration: reports/autodl-jev-2026-10-13/README.md.


## 2026-10-14 — Expanded native-Jev development training started

- Recovered all36,000 groups /252,000 decisions, verified score hashes and SQLite
  integrity, then audited every original HTTP envelope and child response against
  original prompts, objectives, candidates and materialized score tensors.
- Started equal-budget native-Jev arm: seeds7/17/29 ×four shared settings ×30epochs.
  Reuses12 previously completed baseline fits; no independent test or paid GPU.
- Added validation-only seven-goal evaluator with selected-checkpoint score replay,
  paired-user intervals, support counts and explicit selection-bias caveat. The
  sequential local finalizer will evaluate and replay after training finishes.
-12 relevant tests pass. Launch evidence: reports/emer-native-development-2026-10-14.
  This entry records running work and does not claim training or effect success.


## 2026-10-14 — Expanded native-Jev development comparison complete

- All12 native-Jev fits complete (matching12 baseline fits), selected by the
  original validation-only rule. Evaluation and independent-process exact replay
  complete; prediction/evaluation file hashes rechecked against replay receipt.
- On6,000 validation slates /4,253 users, seed-mean long-view NDCG@3 rises
  0.412540 to0.420807 (+2.00%); conditional paired-user absolute CI
  [+0.001819,+0.014960]. Only seed7 improves; seeds17/29 decrease.
- Three of five development diagnostics pass. The every-seed long-view gain and
  positive-mean lower-CI gates fail. Validation was reused for model selection;
  these intervals do not establish confirmatory superiority. Independent test
  remains unopened and no extra paid compute was started.
- Final evidence: reports/emer-native-development-2026-10-14/README.md.


## 2026-10-14 — Exhaustive native-Jev development diagnosis

- Replayed all24 existing checkpoints against original seven-goal validation
  receipts; evaluated all four matched learning-rate/prior settings, retaining
  each original selected epoch. Long-view changes: +0.56%, -3.90%, -0.26%,
  -0.15%; the sole positive interval crosses zero. Original main comparison
  remains unchanged; this is post-selection diagnosis, not a new winner.
- Goal-specific within-slate AUC: behavior/Jev long-view0.5928/0.5354. Strict
  behavior-error rescue19.84% versus newly incorrect pair ordering25.73%, both
  weighted within-slate pair fractions, not an implementable oracle policy.
  No-hate AUC0.4959/0.5532 gives a positive exploratory signal, not fusion proof.
- Fixed six complementary groups from historical counts and pre-label ranking
  disagreement. None improves long view in all three seeds; no post hoc
  subgroup gate was promoted. Inspected five fixed epochs in all four settings.
- All diagnostic statistics replay exactly from hashed cached predictions; six
  signal/statistics tests pass. All24 checkpoint hashes unchanged; no training,
  remote GPU, fresh test data or extra API calls. Decision: keep independent
  test sealed; first reconcile primary long-view guard with aggregate selection
  and test a bounded, baseline-preserving fusion hypothesis in development.
- Evidence: reports/emer-native-diagnosis-2026-10-14/README.md.


## 2026-10-14 — Equal-capacity bounded correction experiment complete

- Froze the original three EMER baselines. Registered two residual arms before
  fitting: seven behavior-rank coefficients versus seven Jev-rank coefficients,
  identical three ridge settings, train-only weighted long-view pair loss and
  0.25 within-slate baseline-standard-deviation score cap.18 small fits complete.
- Validation selects long view subject to long-view/positive-mean point
  noninferiority and original click/hate guards; zero correction wins ties and
  remains fallback. This is a new explicitly labeled development protocol, not
  a post-hoc rewrite of prior success criteria.
- Jev fallback for seeds7/17; seed29 accepts ridge0.001. Mean long-view +0.15%
  versus frozen EMER, absolute95%CI[-0.000542,+0.001833]. Behavior-only correction
  gives+1.23%, CI[+0.002786,+0.007348]. Jev versus behavior-only correction -1.07%,
  CI[-0.006990,-0.002066]. Validation reuse limits all these intervals.
- Jev/original gates2/5, behavior/original4/5, Jev/behavior2/5. New-test planning
  gate fails. Independent test remains sealed; no added GPU or API expenditure.
- Nine focused tests pass. Independent-process model/coefficients/predictions,
  every candidate's guarded selection and all comparison intervals replay
  exactly; optimizer refitting/cross-hardware equivalence not claimed. All
  registered inputs, source scripts and baseline hashes unchanged.
- Evidence: reports/emer-native-bounded-2026-10-14/README.md. Stop expanding this
  fusion grid; a future development proposal should first address signal input
  adequacy and relevance rather than spend on a larger confirmation set.


## 2026-10-14 — Active overall multi-objective goal

- User clarified overall multi-objective superiority, not long-view-only success.
  Preserved old reports and recorded prospective aggregate-primary protocol.
- Equal-budget three-model ensembles and18 seven-goal bounded correction fits
  completed and replayed; both fail stronger development screens. Original
  full model remains the confirmation candidate. All three original seeds
  improve six-goal mean, though only one improved long view.
- Froze original paired models and50,000-slate Apr23-25 confirmation sampling
  before new performance; source preparation running. Implemented producer-
  equality guard and label-free package generation; 11+9 focused tests pass.
- Existing65 CNY cap remains; no new GPU spend. Confirmation acquisition and
  evaluation remain incomplete. Goal active; next live handles in GOAL_PROGRESS.md.


## 2026-10-14 — Confirmation fully staged; additional budget requested

- Recovered from a host reboot after preparation. Replaced source audit random
  USB access/repeated tensor decompression with small user partitions and one
  raw-tensor load. Final50K histories/prompts and500K source occurrences match.
- Recomputed eligible populations across26,520 source users and exactly replayed
  all strata, random priorities, selected records and candidate permutations.
  Frozen test cohort has15,115 users and50,000 slates across Apr23-25.
- Input package54,981,316 bytes, no labels, uploaded to original instance in CPU
  mode; all27 producer files and recorded package versions verified. Instance
  then confirmed OFF. No newGPU spend or held-out effect computation.
- Completed real-score validator, common-tensor materializer, frozen paired
  evaluator, report and independent replay pipeline. Full300 tests pass.
- Verified current6.98 CNY/h and62.67 CNY existing settled project bills. Asked
  once to increase total cap65→115 CNY for this independent run. Pending human
  authorization; account balance is not permission. Goal still active.
- Evidence: reports/multiobjective-confirmation-ready-2026-10-14/README.md.


## 2026-10-14 — User maintains65 CNY total budget

- Explicitly declined the proposed115 CNY cap. Recorded65 as the continuing
  hard spending limit; no GPU restart or independent score acquisition initiated.
- Frozen models,50K source-audited cohort, uploaded input and producer-preflight
  receipts retained. Independent superiority remains unverified. Do not repeat
  the same spending request without a changed user instruction or circumstances.


## 2026-10-14 — No-new-fee retrospective user holdouts

- After explicit65 CNY cap, found a local route using the full existing36K
  real-score cohort: fresh three-fold nested user-disjoint retraining.
- Original architecture, identical budgets per arm, three seeds and four
  settings ×30epochs;72fits. Fold feature moments use training users only.
- Fixed splits cover every row once as outer test; never reinterpret old
  trained-model predictions as unseen-user evidence. Apr23-25 remains untouched
  by this driver; no newGPU/API spending.
-11 relevant tests pass; fold artifacts prepared and training started.
  Finalizer queues outer evaluation and exact replay only after all fits.
- Retrospective cohort reuse and overlapping fold-training uncertainty remain
  explicit limitations. This launch does not prove the requested superiority.


## 2026-10-14 — Saved CV tensor audit

- Read actual saved train/inner-validation/outer-test files across all three
  folds. Verified user exclusion, exactly-once outer coverage, original labels,
  behavior/Jev signals, sample weights and identities for all36K rows.
- Recomputed training-only weighted moments column by column independently of
  fit_feature_spec and checked all encoded candidate features. Passed.
- Training remains live; this audit computes no outer performance and makes
  no superiority claim. Receipt: reports/emer-user-cv-v1-start/data_audit.json.


## 2026-10-15 — First CV fold completed and inner selection audited

- All24 original-budget fits for outer fold0 complete; fold1 continues.
- Verified all checkpoint hashes and training budgets, reproduced initial
  checkpoint/earliest-best-epoch selection and max-inner-mean configuration
  selection, and recomputed seven metrics for all six selected models. Passed.
- No outer evaluation was performed. Completion of one fold is not evidence
  that the user's superiority objective has been achieved.


## 2026-10-15 — Second CV fold completed and inner selection audited

- All24 fits for outer fold1 complete; total48/72 complete before fold2 starts.
- Verified all trial hashes/budgets and earliest-best epoch/config rules;
  recomputed seven inner-validation metrics for all six selected models. Passed.
- No outer metrics computed. Third fold continues on the same process;
  no paidGPU/API activity.


## 2026-10-15 — Complete retrospective CV and preserve the failed guard

- All72 equal-budget fits completed; all three24-trial selection audits passed.
- Outer evaluation covered36,000 slates and12,606 users. Independent prediction,
  metric and bootstrap replay matched exactly. No new paidGPU/API activity.
- Aggregate six-positive NDCG@3 improves1.22%, conditional95% interval
  [+0.000818641,+0.006560752]; every seed and fold has a positive point gain.
- Follow declines0.005737264 absolute, exceeding the fixed0.005 guard.
  Screen4/5, full screen fails; goal remains unachieved. No threshold changes.
- Full seven-goal report, source/selection manifests, all audit receipts and
  replay evidence saved in reports/emer-user-cv-v1. Retrospective limitations
  and untouched chronological confirmation are retained explicitly.


## 2026-10-15 — Diagnose sparse-goal tradeoffs before another experiment

- Audited all36 existing Jev candidate receipts against each same-fold/seed
  selected EMER: none passes every inner point guard;5/9 selected candidates
  already fail the follow guard on inner validation. No new outer model scoring.
- Measured positive support, user-weight concentration and tie-aware within-slate
  signal AUC for every fold role. Follow training prevalence is roughly0.06%;
  its standalone Jev AUC is below the behavior signal in all three outer folds.
  These are descriptive findings, not a proven causal explanation.
- Added two metric tests covering weighted ties, padding, zero support, and
  user aggregation; both pass. Diagnostic report preserves the original result.
- Specified a fixed18-fit exploratory balanced-posterior follow-up, same recipe
  in both arms, identical anchor predictions and inner-only guarded selection.
  Implementation and execution registration are pending; no new training yet.
