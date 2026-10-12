# ParetoJev

**Jev-style decision models × multi-objective recommendation reranking**

ParetoJev is a research/engineering starter kit for combining:

1. a conventional behavior recommender (`SASRec + MMoE`-style pXTR branch),
2. a fast typed decision model such as Jev,
3. confidence-aware semantic/behavior fusion,
4. a preference-conditioned set reranker,
5. explicit Pareto optimization and uncertainty modeling,
6. distillation into a smaller online model.

The repository is organized around **target mode**: every experiment is tied to a measurable gate rather than an open-ended “try a model and inspect the table” workflow.

## Current execution status

Verified locally on **2026-10-12**:

| Milestone | Status | Evidence |
|---|---|---|
| Product/research requirements | Complete | [`docs/PRD.md`](docs/PRD.md) |
| Target-mode gates | Complete | [`docs/TARGET_MODE.md`](docs/TARGET_MODE.md), [`configs/targets_research.yaml`](configs/targets_research.yaml) |
| MicroLens upstream-compatible protocol | Implemented and conformance-tested | `paretojev.data.microlens` |
| KuaiRand pseudo-slate builder | Implemented | `paretojev.data.kuairand` |
| Official TypeSafe Jev + open JEV clients | Implemented, cached and resumable | `paretojev.jev` |
| MicroLens research evaluator | Nested user-CV, calibration, bootstrap and complementarity | `paretojev.experiments.microlens` |
| SASRec + MMoE behavior branch | Implemented | `paretojev.models.behavior` |
| Confidence gate + Pareto reranker | Implemented | `paretojev.models.pareto_reranker` |
| Pareto/Tchebycheff/uncertainty losses | Implemented | `paretojev.losses.pareto` |
| Unified trainable-baseline harness | Registered EMER-like / CMR-like / ParetoJev comparison | [`docs/BASELINE_HARNESS.md`](docs/BASELINE_HARNESS.md) |
| Unit/integration tests | Current verified results in execution log | [`docs/EXECUTION_LOG.md`](docs/EXECUTION_LOG.md) |
| Synthetic end-to-end smoke run | **Passed** | [`artifacts/smoke_metrics.json`](artifacts/smoke_metrics.json) |
| Registered six-objective baseline comparison | 3 seeds × 3 methods; exact resume verified | [`reports/synthetic-baseline-2026-10-08/README.md`](reports/synthetic-baseline-2026-10-08/README.md) |
| Real MicroLens / AutoTrust JEV-27B | 200 users / 4,000 verified decisions; main targets 2/4 passed | [`reports/microlens-real-200-2026-10-08/README.md`](reports/microlens-real-200-2026-10-08/README.md) |
| New-user tie-breaking validation | 200 disjoint evaluation users; 2/2 predeclared AUC comparisons pass; request savings only 5.45% | [`reports/microlens-independent-2026-10-08/README.md`](reports/microlens-independent-2026-10-08/README.md) |
| Real SASRec + semantic development controls | Trained on 57,845 users; held-out validation improves; fixed 20% Jev shows developmental uplift; 5% gates fail | [`reports/microlens-sasrec-development-2026-10-08/README.md`](reports/microlens-sasrec-development-2026-10-08/README.md) |
| Fixed 700-user confirmation | 14,000 real decisions; AUC gain vs SASRec passes, superiority vs MiniLM does not; **1/2 gates** | [`reports/microlens-sasrec-confirmatory-2026-10-09/README.md`](reports/microlens-sasrec-confirmatory-2026-10-09/README.md) |
| KuaiRand-27K acquisition and audit | Full core and supplements verified; 323,464,444 exposure rows audited | [`reports/kuairand-27k-acquisition-2026-10-11/README.md`](reports/kuairand-27k-acquisition-2026-10-11/README.md) |
| Real KuaiRand-27K behavior pilot | 426,832 train / 197,078 validation targets; 7 tasks; event-only gates 6/7, calibration and overfitting remain | [`reports/kuairand-27k-behavior-pilot-2026-10-11/README.md`](reports/kuairand-27k-behavior-pilot-2026-10-11/README.md) |
| 27K richer controls + time-separated calibration | 1,699,506 train targets; both controls beat tab priors on 6/7 objectives; hate remains unresolved | [`reports/kuairand-27k-behavior-v2-2026-10-11/README.md`](reports/kuairand-27k-behavior-v2-2026-10-11/README.md) |
| Aligned 27K behavior comparison | Same 1,699,506 train targets; tree/MMoE close, richer SASRec trades worse click/long-view for better like/follow/forward; all 6/7 gates | [`reports/kuairand-27k-aligned-2026-10-12/README.md`](reports/kuairand-27k-aligned-2026-10-12/README.md) |
| Real KuaiRand core Jev/ParetoJev validation | Complete: 14,000 real decisions, 18 training runs, primary 1/4 gates; overall superiority not supported | [`final report`](reports/kuairand-core-2026-10-12/README.md), [`completion audit`](reports/kuairand-core-2026-10-12/completion_audit.json) |
| Serving/student experiments | Not run | Core offline validation takes priority |

Synthetic results validate engineering paths; they are **not recommendation research results**. The restored baseline and current execution evidence are recorded separately in [`docs/EXECUTION_LOG.md`](docs/EXECUTION_LOG.md). Full real MicroLens evaluation now supports an AUC increment from small-weight Jev fusion, while the complete target set remains unmet.

The registered synthetic comparison did **not** show ParetoJev superiority:
hypervolume was below both trainable baselines, despite better cold-NDCG. Those
synthetic results cannot satisfy real research gates; see the report for all results and limitations.

The full real text-only run used the author's public JEV-27B endpoint, verified each
deployment revision, and required no user API key. CV fusion reached AUC 0.7274,
HR@5 0.570 and NDCG@10 0.5188; AUC gains remain positive under the conventional
half-credit tie diagnostic. This is a simple fusion result, not evidence for the
neural ParetoJev architecture or a cover-image VL reproduction. The report explains
why the earlier positive-first upstream baseline AUC is not the same comparison.

On a fresh evaluation cohort, frozen Jev tie-breaking reached AUC 0.7176 versus
0.6650 for CF and 0.6738 for TF-IDF tie-breaking; both predeclared 97.5% paired AUC
intervals exclude zero. This supports a narrow incremental-signal hypothesis, not
completion of the original project targets. The tie-only call plan saves just
5.45% of pointwise requests; the experiment actually collected all 4,000 scores
for the matched fixed-weight comparison. No new-user outcomes tuned any weights.

A real SASRec-style next-item model has now been trained on MicroLens, with
1,000 separate users used only for checkpoint selection. On the 400 previously
examined development users, SASRec reaches AUC 0.7622 versus refitted CF 0.6814.
Fixed 5% Jev fusion barely changes its continuous-score ordering; the separately
preregistered 20% fusion reaches AUC 0.7701 and HR@5 0.6213, exceeding the matched
MiniLM control. These are development results, not new independent confirmation:
only 1/3 registered development gates passes and original project gates remain unmet.
No new Jev requests were made. This next-item model does not train the KuaiRand
six-objective MMoE heads or the full ParetoJev reranker.

The fixed 700-user confirmation is now complete: AUC is 0.7795 for SASRec,
0.7831 for its 20% MiniLM fusion and 0.7853 for its 20% Jev fusion. The registered
97.5% AUC difference interval is positive versus SASRec but includes zero versus
MiniLM. **Only 1/2 confirmation gates passes.** HR/NDCG increments remain uncertain.
This supports a narrow AUC increment, not a validated advantage over the semantic
control or success of the full ParetoJev architecture. All 14,000 genuine scores
were audited; one TLS interruption used one transparently recorded contingency
attempt. Offline collection and evaluation replay require no new requests.

## Research hypothesis

A classical recommender and a foundation decision model should be complementary:

- the behavior model is strong on warm items, collaborative statistics and platform-specific response patterns;
- the decision model is potentially strong on content semantics, sparse-history users and cold items;
- a learned reliability gate should decide when each signal deserves weight;
- a preference-conditioned reranker should expose not one fixed weighted score, but a controllable approximation to the objective trade-off surface.

The intended system is:

```text
User history + context + K candidates
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
SASRec + MMoE          Jev decisions
behavior pXTRs         semantic probabilities
        │                   │
        └─────────┬─────────┘
                  ▼
       confidence-aware fusion
                  │
                  ▼
 preference-conditioned Set Transformer
       + Pareto / uncertainty losses
                  │
                  ▼
       reranked candidate slate
                  │
                  ▼
         distilled online student
```

## Why these datasets

### MicroLens-100K: fast, directly comparable demo

The public JEV-27B-VL recommendation demo evaluates 200 users, the five most recent videos as history, and 20 candidates consisting of one hidden next item plus 19 time-local negatives. This repository reproduces the less-obvious details as well: occurrence-weighted negative sampling from the time window and the demo's deterministic position-wise tie jitter. A cleaner uniform-unique sampler and stable tie policy remain available as explicitly named research variants.

The evaluator now provides four fusion levels:

1. reciprocal-rank fusion without labels;
2. transparent simplex-weighted fusion with outer user folds;
3. nested user-CV logistic stacking with raw/rank/margin and confidence interactions;
4. a later learned confidence gate in the main ParetoJev model.

It also reports user-bootstrap confidence intervals, paired differences against a declared reference and error-complementarity diagnostics. Jev Choice probabilities can be temperature-calibrated out of fold; pointwise Noul probabilities can be Platt-calibrated out of fold.

The public demo reports JEV-27B-VL cover-only AUC `0.727`, HR@5 `0.590`, NDCG@10 `0.498`, and item-CF AUC `0.728`, HR@5 `0.490`, NDCG@10 `0.503`. These are **upstream self-reported reference values**, not results produced by this repository.

### KuaiRand-Pure: main multi-objective track

KuaiRand exposes click/valid-play, long-view, like, follow, comment, forward, hate, watch time and random-intervention logs. Supplementary caption/category files provide content text for decision-model prompts.

KuaiRand does **not** expose the original full candidate list for every serving request. The repository therefore labels its grouped construction precisely as an **exposed-session pseudo-slate** and keeps a separate random-exposure evaluation track. It must not be presented as native request-level reranking.

### Amazon Reviews 2023: follow-up compatibility track

The existing Jev reranking study uses Amazon Reviews 2023, SASRec hard candidates and `K ∈ {20, 50, 100, 200}`. It remains a planned compatibility track; the current bootstrap prioritizes MicroLens and KuaiRand.

## Installation

Python 3.11+ is required.

```bash
cd ParetoJev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

## Run the completed smoke workflow

```bash
paretojev synthetic-smoke \
  --config configs/synthetic_smoke.yaml \
  --output artifacts/smoke_metrics.json \
  --checkpoint artifacts/smoke_model.pt

paretojev check-targets \
  artifacts/smoke_metrics.json \
  configs/targets_smoke.yaml \
  --output artifacts/smoke_target_report.json
```

The smoke target report is expected to pass. The research target file is intentionally aspirational and should not pass until real experiments are run.

## Compare independently trained baselines

```bash
paretojev comparison-synthetic-data data/processed/comparison_synthetic \
  --seed 123 --train-slates 96 --validation-slates 48 --test-slates 48 \
  --slate-size 8 --objectives 3 --feature-dim 6

paretojev compare-baselines data/processed/comparison_synthetic/manifest.json \
  --output artifacts/comparison_synthetic \
  --config configs/comparison_smoke.yaml \
  --targets configs/targets_comparison.yaml
```

Use empty output directories. This runs separately trained `emer_like`, `cmr_like`
and `paretojev` on identical inputs, training budgets and held-out evaluation.
EMER-like and CMR-like are documented project variants, not paper reproductions.
Parameter counts are reported; equal epochs do not imply equal model size or FLOPs.
Checkpoints are selected only on validation, including epoch zero. Cold thresholds
come from training, and both comparator identities are fixed before test evaluation.

The run writes its registration, checkpoints, training histories, prediction audit
arrays, per-seed evaluations and descriptive seed-mean summary. Reuse the same
command with `--resume` to validate and reload completed methods; changed inputs,
source, environment or settings require a new directory. Experiment completion
returns exit code `0`; it does not assert that research targets passed.

Synthetic evidence always makes real research targets `not_evaluable`, even with
large synthetic gains. `paretojev check-targets ...` returns `3` for failed or
not-evaluable required gates. See [`docs/BASELINE_HARNESS.md`](docs/BASELINE_HARNESS.md)
for exact algorithms, metrics, uncertainty, configuration and the real-data manifest.
Prepared, aligned score artifacts are sufficient for a real run; no live API is
needed when scores are already supplied.

## MicroLens workflow

### 1. Download text/interaction data

```bash
paretojev download-microlens data/raw/microlens
```

Add `--covers --extract-covers` only for the multimodal JEV-27B-VL track; the cover archive is much larger.

### 2. Build deterministic candidate lists

```bash
paretojev prepare-microlens \
  data/raw/microlens \
  data/processed/microlens \
  --negative-sampling upstream_occurrence \
  --tie-policy upstream_jitter
```

Outputs:

- `candidates.csv`: candidate-level rows with popularity, title similarity and item-CF;
- `candidates.json`: upstream-compatible ID manifest;
- `jev_input.jsonl`: text-only slates consumable by this repository's Jev scorer;
- `prepare_report.json`: protocol and baseline metrics.

### 3. Score with official Jev

```bash
export TYPESAFE_API_KEY="..."
paretojev score-jsonl \
  data/processed/microlens/jev_input.jsonl \
  artifacts/microlens_jev_choice.jsonl \
  --provider typesafe \
  --mode choice \
  --objectives click \
  --model jev-1.13.0 \
  --report artifacts/microlens_jev_choice.report.json \
  --errors artifacts/microlens_jev_choice.errors.jsonl
```

Or point the same command at a self-hosted open JEV server:

```bash
paretojev score-jsonl \
  data/processed/microlens/jev_input.jsonl \
  artifacts/microlens_open_jev.jsonl \
  --provider open \
  --mode noul \
  --base-url http://localhost:8000
```

Every request is cached in SQLite and every successful slate is appended durably to the output JSONL. A compatible rerun resumes from completed slate IDs; it refuses to mix a different provider, model, mode, objective set or Noul scope into an existing output.

### 4. Merge scores and evaluate a leakage-safe hybrid

```bash
paretojev merge-decision-scores \
  data/processed/microlens/candidates.csv \
  artifacts/microlens_jev_choice.jsonl \
  data/processed/microlens/candidates_with_jev.csv \
  --objective click \
  --score-column jev_click

paretojev evaluate-microlens \
  data/processed/microlens/candidates_with_jev.csv \
  --scores itemcf title_sim jev_click \
  --choice-temperature jev_click \
  --probability-diagnostics jev_click \
  --rrf itemcf jev_click \
  --hybrid itemcf jev_click \
  --stack itemcf title_sim jev_click jev_click_confidence jev_click_entropy \
  --folds 5 \
  --inner-folds 4 \
  --weight-step 0.05 \
  --objective balanced \
  --reference itemcf \
  --complementarity itemcf jev_click \
  --bootstrap-samples 5000 \
  --output artifacts/microlens_metrics.json \
  --output-scores data/processed/microlens/evaluated_scores.csv
```

Weights are selected on K-1 user folds and applied only to the held-out fold. The stacker additionally selects regularization by inner user folds. This avoids fitting the reported 200 users and then scoring those same users with the fitted fusion.

Run the Phase A gate separately:

```bash
paretojev check-targets \
  artifacts/microlens_metrics.json \
  configs/targets_microlens.yaml \
  --output artifacts/microlens_target_report.json
```

Or execute the full local sequence after setting `TYPESAFE_API_KEY`:

```bash
scripts/run_microlens_phase_a.sh
```

The script exits non-zero when a required target fails or cannot be evaluated. Phase A research gates also require explicit real MicroLens evidence. Existing `evaluate-microlens` reports do not establish that origin automatically and therefore remain `not_evaluable` until reviewed evidence is supplied; a high metric alone is insufficient.

## Active KuaiRand-27K workflow

**Core real-signal validation is complete; superiority is not established.**
The six-arm experiment used 14,000 real JEV-27B Choice decisions, 18 training runs
and shared weighted evaluation. ParetoJev passes only 1/4 primary point gates
against no-Jev EMER-like. Exact-HV per-seed changes are +59.79%, -35.84%, +4.43%:
mean relative gain +9.46%, but the ratio of absolute HV means is only +0.27%.
It loses HV against no-Jev CMR-like, equal-Jev comparators and the fixed behavior
weighted sum, while some cold metrics improve. All raw responses, saved predictions,
metrics and intervals replay; independent arithmetic and exact-HV checks pass.
See [final findings and limits](reports/kuairand-core-2026-10-12/README.md).
This completes the current core validation, not the unachieved research/serving targets.

The active dataset is the complete **27K**, explicitly selected by the user. Its
core archive, captions and categories are acquired and verified on the external
drive. `data/raw/kuairand-27k` points to that directory. Do not load the full shards
with the older in-memory trainer below.

The completed tree-control run **v2** uses: 1,699,506 training targets,
188,642 calibration targets (April 18), and 187,419 evaluation targets (April 19).
The evaluation day was already seen during v1 development; this is not a fresh
confirmatory test. Two sets of seven independent histogram-boosted classifiers
compare exposure/history statistics with additional upload-property features.
SASRec was not retrained in this experiment. Both sets pass 6/7 stricter development
gates against tab prevalence; **hate remains unresolved**. Jev/ParetoJev research
targets remain pending. See the [report and feature assumptions](reports/kuairand-27k-behavior-v2-2026-10-11/README.md).

The completed **aligned v3** comparison reuses those exact rows and the
frozen tree, fits a 26-feature MMoE control, and fits SASRec + MMoE with those same
features plus user/candidate/history IDs. Only tree vs tabular MMoE has identical
information; the sequence arm tests an explicitly richer package. Both neural
arms use three fixed epochs, then the same April 18 calibration and April 19
evaluator. Rare-feedback enrichment is deferred. See the
[registration](reports/kuairand-27k-aligned-2026-10-12/REGISTRATION.md) and
[results](reports/kuairand-27k-aligned-2026-10-12/README.md). Tree and tabular MMoE
are close; the richer SASRec package has worse click/long-view but better
like/follow/forward predictions in this fixed-budget development run. All three
pass 6/7 gates; hate remains unresolved. This does not establish architecture
superiority or a Jev/ParetoJev increment.

```bash
.venv/bin/python scripts/prepare_kuairand_aligned.py \
  --store data/raw/kuairand-27k/behavior-pilot-v1 \
  --data data/raw/kuairand-27k/behavior-v2 \
  --output data/raw/kuairand-27k/behavior-aligned-v3
.venv/bin/python scripts/run_kuairand_aligned.py \
  --data data/raw/kuairand-27k/behavior-v2 \
  --sequences data/raw/kuairand-27k/behavior-aligned-v3 \
  --tree data/raw/kuairand-27k/behavior-v2/run-v2 \
  --output data/raw/kuairand-27k/behavior-aligned-v3/run-v1
```

Use `--replay` on the second command to verify frozen predictions, scene metrics
and paired bootstrap. Use a fresh output directory to retrain; completed data
and runs are never overwritten.

For a new data directory, prepare once (the existing `behavior-v2` is already
complete). This uses the v1 complete early-event store and verifies its hashes:

```bash
.venv/bin/python scripts/prepare_kuairand_behavior_v2.py \
  --root data/raw/kuairand-27k \
  --cache data/processed/kuairand-v2-metadata \
  --output data/raw/kuairand-27k/behavior-v2
```

The frozen successful run is `behavior-v2/run-v2`. Replay it without fitting:

```bash
.venv/bin/python scripts/replay_kuairand_behavior_v2.py \
  --data data/raw/kuairand-27k/behavior-v2 \
  --run data/raw/kuairand-27k/behavior-v2/run-v2 \
  --output data/raw/kuairand-27k/behavior-v2/run-v2/replay-check.json
```

To refit the registered experiment, choose a fresh output directory:

```bash
.venv/bin/python scripts/run_kuairand_behavior_v2.py \
  --data data/raw/kuairand-27k/behavior-v2 \
  --output data/raw/kuairand-27k/behavior-v2/run-reproduction
```

The loader checks hashes and caps each materialized period at 512 MiB. Its arrays
fit the registered target-sampling budget; do not substitute full-exposure arrays
without a new streaming plan. Calibration uses April 18 only. April 20 onward is
reserved for later work. The six-objective reranker protocol is not silently
equated to these seven raw feedback outputs; a common slate decision time is
required before exporting reranker features. Research target mode stays off.

The first real SASRec-style + MMoE pilot remains reproducible through
`scripts/prepare_kuairand_behavior_pilot.py`, `scripts/train_kuairand_behavior_pilot.py`
and `scripts/replay_kuairand_behavior_pilot.py`; its original protocol and results
are in the [v1 registration](reports/kuairand-27k-behavior-pilot-2026-10-11/REGISTRATION.md).

## Legacy small-file/Pure workflow

The examples below document the earlier in-memory interfaces for small files and
synthetic protocol checks. They are **not the active 27K procedure** and have not
been run as a complete real multi-objective reranker benchmark. Example inputs:

```text
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
log_random_4_22_to_5_08_pure.csv
kuairand_video_captions.csv
kuairand_video_categories.csv
```

### 1. Reserve disjoint periods for behavior training and reranker training

Do not train the behavior model on the same events whose pXTR values will become
features for the downstream reranker. Split the early standard log chronologically:

```bash
paretojev split-kuairand-log \
  data/raw/kuairand/log_standard_4_08_to_4_21_pure.csv \
  data/processed/kuairand/splits \
  --fractions 0.6,0.2,0.2 \
  --names behavior_train,behavior_validation,reranker_train \
  --write-prefixes
```

Boundaries never divide rows sharing the same timestamp. `--write-prefixes` also
creates `prefix_before_behavior_validation.csv` and
`prefix_before_reranker_train.csv`, which are used to initialize strictly earlier
histories and popularity counts.

Train the formal behavior branch only on the first period and early-stop on the
second:

```bash
scripts/train_kuairand_behavior.sh \
  data/processed/kuairand/splits/behavior_train.csv \
  data/processed/kuairand/splits/behavior_validation.csv \
  artifacts/kuairand_behavior.pt \
  artifacts/kuairand_behavior_metrics.json \
  cuda
```

The implementation uses a left-padded SASRec history encoder, shared item embeddings,
MMoE objective heads and pre-exposure context. New validation/test items map to an
explicit OOV index. Probabilities are trained with unweighted BCE; the command does
not fit a post-hoc calibrator on the validation labels and records this limitation.

### 2. Build reranker pseudo-slates from an unseen later period

```bash
paretojev prepare-kuairand \
  data/processed/kuairand/splits/reranker_train.csv \
  data/processed/kuairand/train_slates.npz \
  --prefix-log data/processed/kuairand/splits/prefix_before_reranker_train.csv \
  --slate-size 20 \
  --min-slate-size 5 \
  --session-gap-minutes 30 \
  --manifest data/processed/kuairand/train_manifest.jsonl \
  --report data/processed/kuairand/train_prepare_report.json
```

Slate construction records item exposure counts and user-history lengths using only
strictly earlier timestamps. Simultaneous events do not count one another as history.

When preparing a later 4/22–5/08 split, initialize counts from the complete earlier
standard log:

```bash
paretojev prepare-kuairand \
  data/raw/kuairand/log_standard_4_22_to_5_08_pure.csv \
  data/processed/kuairand/validation_slates.npz \
  --prefix-log data/raw/kuairand/log_standard_4_08_to_4_21_pure.csv \
  --report data/processed/kuairand/validation_prepare_report.json
```

The command rejects overlapping prefix/current timestamps, preventing accidental
future-contaminated histories.

### 3. Score every candidate slate with the out-of-period behavior model

```bash
paretojev score-kuairand-behavior \
  artifacts/kuairand_behavior.pt \
  data/processed/kuairand/train_slates.npz \
  data/processed/kuairand/splits/reranker_train.csv \
  data/processed/kuairand/train_behavior_scores.npz \
  --prefix-log data/processed/kuairand/splits/prefix_before_reranker_train.csv \
  --device cuda \
  --report artifacts/kuairand_train_behavior.report.json
```

Histories are cut with `searchsorted(..., side="left")`, so neither the current
candidate nor any simultaneous exposure enters the user state. Candidate/history OOV
rates, Brier diagnostics, throughput and batch latency are written to the report.

### 4. Select a cost-controlled Jev annotation batch

The preferred first probe preserves the structural cold/history/exposure/feedback
cells and, *within each cell*, softly prioritizes slates on which the behavior model
is uncertain or chooses different winners for different objectives:

```bash
scripts/sample_kuairand_behavior_informed.sh \
  data/processed/kuairand/train_slates.npz \
  data/processed/kuairand/train_behavior_scores.npz \
  data/processed/kuairand/splits/reranker_train.csv \
  data/raw/kuairand \
  data/processed/kuairand \
  5000 \
  0.5
```

Equivalent CLI flags are `--behavior-scores` and
`--behavior-priority-strength`. Priority combines normalized Bernoulli entropy,
low top-1/top-2 margin and cross-objective winner disagreement. Strength `0` recovers
random within-cell order; strength `1` takes the highest-priority available examples;
`0.5` deliberately keeps a randomized component so the teacher set does not collapse
to only ambiguous cases.

Without a neural score tensor, omit `--behavior-scores` to use the original purely
structural balanced sampler. The initial 5k budget is an annotation-training probe,
not an evaluation population. Because the structural sampler also balances observed
feedback density, final validation/test metrics must be computed on the complete
fixed evaluation protocol or with declared sampling weights, never directly on this
selected training subset.

### 5. Score the selected slates with Jev

After generating `train_jev_sample.jsonl`, score only that subset. For a first
text-only probe:

```bash
export TYPESAFE_API_KEY="..."
paretojev score-jsonl \
  data/processed/kuairand/train_jev_sample.jsonl \
  artifacts/kuairand_train_jev.jsonl \
  --provider typesafe \
  --mode noul \
  --objectives click long_view like follow deep_engagement negative_feedback \
  --model jev-1.13.0 \
  --report artifacts/kuairand_train_jev.report.json \
  --errors artifacts/kuairand_train_jev.errors.jsonl
```

The scorer's prompt objective names are explicitly mapped back to KuaiRand labels
(`click → is_click`, `like → is_like`, `negative_feedback → is_hate`) during
materialization. Candidate IDs, candidate order and objective order are validated;
a mismatch fails instead of silently permuting tensors.

### 6. Materialize a trainable NPZ

For the main experiment, pass `train_behavior_scores.npz` directly to
`materialize-kuairand`:

```bash
paretojev materialize-kuairand \
  data/processed/kuairand/train_slates.npz \
  artifacts/kuairand_train_jev.jsonl \
  data/processed/kuairand/train_behavior_scores.npz \
  data/processed/kuairand/train_features.npz \
  --manifest data/processed/kuairand/jev_sample_manifest.csv \
  --feature-spec data/processed/kuairand/candidate_feature_spec.json \
  --fit-feature-spec \
  --report artifacts/kuairand_train_materialize.report.json
```

### Bootstrap-only fallback

If no neural checkpoint is available yet, the repository can produce a
strictly pre-slate empirical-Bayes behavior signal. It uses only unique outcomes
represented in earlier prepared pseudo-slates, never an outcome from the current
slate. This is a CPU bootstrap and **not** the final SASRec+MMoE baseline.

```bash
scripts/materialize_kuairand_probe.sh \
  data/processed/kuairand/train_slates.npz \
  data/processed/kuairand/jev_sample_manifest.csv \
  artifacts/kuairand_train_jev.jsonl \
  data/processed/kuairand/train_features.npz \
  data/processed/kuairand/candidate_feature_spec.json \
  train
```

For a later validation period, reuse the train-fitted feature contract and seed the
historical prior with earlier prepared slates:

```bash
scripts/materialize_kuairand_probe.sh \
  data/processed/kuairand/validation_slates.npz \
  data/processed/kuairand/validation_jev_sample_manifest.csv \
  artifacts/kuairand_validation_jev.jsonl \
  data/processed/kuairand/validation_features.npz \
  data/processed/kuairand/candidate_feature_spec.json \
  eval \
  data/processed/kuairand/train_slates.npz
```

The train feature specification is fitted once and reused unchanged. By default,
candidate features exclude the position inside the constructed pseudo-slate because
that position can encode logged production order and break set invariance. The
`--include-sequence-position` switch exists only for an explicit ablation.

The materializer writes and immediately reloads the final artifact through the
`PrecomputedSlateDataset` contract, so malformed dimensions are rejected before a
long training run begins. The upstream behavior-score interchange format and final
training NPZ are documented in [`docs/NPZ_CONTRACT.md`](docs/NPZ_CONTRACT.md).

## Core model and losses

Implemented components include:

- **Behavior branch:** SASRec user encoder with shared item embeddings and MMoE objective heads.
- **Fusion:** a shared gate driven by behavior probability, Jev probability, disagreement, entropy, confidence, popularity and history length.
- **Reranker:** a set-aware Transformer with FiLM conditioning on runtime objective weights.
- **P-RAS:** pairwise supervision only when one item Pareto-dominates another across observed outcomes.
- **Augmented Tchebycheff:** covers trade-offs that a simple weighted sum may miss.
- **EMER-like feedback:** training-only detached objective-loss/EMA reweighting in the comparison variant; it is not a reproduction of the paper’s evaluator. See the harness specification for each method’s exact losses.
- **Uncertainty loss:** Gaussian pairwise probability based on mean and variance outputs.
- **Distillation:** candidate-distribution KL from Jev-derived utility distributions into objective heads.

## Train from precomputed real-data features

Once behavior/Jev features have been materialized, train the reranker with:

```bash
paretojev train-precomputed \
  data/processed/kuairand/train_features.npz \
  data/processed/kuairand/validation_features.npz \
  --directions 1,1,1,1,1,-1 \
  --objective-names is_click,long_view,is_like,is_follow,deep_engagement,is_hate \
  --config configs/train_precomputed.yaml \
  --output artifacts/train_metrics.json \
  --checkpoint artifacts/paretojev.pt

paretojev evaluate-pareto \
  artifacts/paretojev.pt \
  data/processed/kuairand/validation_features.npz \
  --top-k 10 \
  --random-preferences 32 \
  --include-pairwise-preferences \
  --hypervolume-samples 100000 \
  --output artifacts/pareto_metrics.json
```

`evaluate-pareto` sweeps uniform, one-hot, optional pairwise and Dirichlet runtime
preferences. It reports a shared-unit-box Monte Carlo hypervolume, objective-level
controllability, worst-objective regret, cold-item NDCG, dominance rates, and the
exact target namespace consumed by target mode. The in-report baselines are static
behavior/Jev/fusion scalarizations. This is an interim diagnostic evaluator; use
`compare-baselines` for the registered trainable-baseline comparison. Neither a
static evaluation nor a synthetic harness run can satisfy real research gates.

For the strict train → evaluate → target-gate workflow:

```bash
scripts/train_precomputed.sh \
  data/processed/kuairand/train_features.npz \
  data/processed/kuairand/validation_features.npz
```

The exact array contract is documented in [`docs/NPZ_CONTRACT.md`](docs/NPZ_CONTRACT.md).

## Target mode

Use the target file for the evidence being evaluated:

| Target file | Scope |
|---|---|
| `targets_smoke.yaml` | Synthetic engineering checks |
| `targets_kuairand_interim.yaml` | Numeric diagnostics against static signal scalarizations; no research claim |
| `targets_comparison.yaml` | Real KuaiRand comparison against both predeclared trainable variants |
| `targets_microlens.yaml` | Real MicroLens cross-validated hybrid and paired AUC interval |
| `targets_research.yaml` | Combined MicroLens, KuaiRand and serving plan; absent phases remain not evaluable |

The KuaiRand comparison targets require hypervolume gain ≥ `2%`, cold-item NDCG
gain ≥ `3%` and worst-objective regret ≤ `0.005` against **each** registered
EMER-like/CMR-like comparator, plus ParetoJev controllability ≥ `0.60`. The primary
reported comparison remains EMER-like; the second is fixed independently, with no
best-on-test selection. MicroLens targets remain AUC ≥ `0.735`, HR@5 ≥ `0.600`,
NDCG@10 ≥ `0.510`; serving targets remain ≥ `90%` uplift retention and p95 ≤ `30 ms`
at K=50 on declared hardware.

Statuses are `passed`, `failed` and `not_evaluable`. Missing/null/nonfinite metrics,
missing provenance and empty required target sets cannot pass. Real comparison
requires a real KuaiRand manifest, `baseline-harness-v1` and all three method
reports. A passing seed mean is descriptive, not proof of statistical significance.
See [`docs/TARGET_MODE.md`](docs/TARGET_MODE.md) for the exact gate semantics.

## Reproducibility rules

- Pin versioned Jev model IDs for experiments; do not use a moving alias in final tables.
- Cache every decision-model request and log resolved model ID plus request hash.
- Split by user/time before fitting blend weights or calibrators.
- Report user-level bootstrap confidence intervals.
- Separate standard-exposure and random-exposure results.
- Report Chinese caption and translated-English prompt ablations independently.
- Keep raw labels, utility-transformed labels and optimization directions distinct.
- Never describe pseudo-slates as production request candidate sets.

## Source references

Source details and verification notes are collected in [`docs/SOURCE_NOTES.md`](docs/SOURCE_NOTES.md). Key upstream resources:

- TypeSafe Jev models/API: <https://docs.typesafe.ai/models>, <https://docs.typesafe.ai/api>
- Jev recommendation reranking study: <https://arxiv.org/abs/2609.40241>
- EMER: <https://arxiv.org/abs/2508.05093>
- CMR: <https://arxiv.org/abs/2306.05118>
- KuaiRand: <https://github.com/chongminggao/KuaiRand>
- MicroLens: <https://github.com/westlake-repl/MicroLens>
- Community JEV-27B demo: <https://github.com/yuhai-china/JEV-27B-DEMO>

## License

Project code is Apache-2.0. Dataset/model licenses remain those of their respective providers. Do not redistribute MicroLens covers or KuaiRand files through this repository.
