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
