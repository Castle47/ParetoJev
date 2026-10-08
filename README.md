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

Verified locally on **2026-10-08**:

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
| Unit/integration tests | **46 passed** | `pytest` |
| Synthetic end-to-end smoke run | **Passed** | [`artifacts/smoke_metrics.json`](artifacts/smoke_metrics.json) |
| Real MicroLens/KuaiRand/Jev experiments | Not run in this bootstrap | Requires datasets plus Jev endpoint/API or self-hosted model |

The smoke experiment reduced held-out composite loss by about **9.1%** and learned a fusion whose Brier error was lower than either synthetic source alone. Those values only validate the code path; they are **not recommendation research results**.

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

The script exits non-zero when a required research target is missed. That is an experiment result, not a pipeline error; inspect the generated metrics and target report before changing the method.

## KuaiRand workflow

Place the official files under `data/raw/kuairand/` or another local path. The minimum primary-track inputs are:

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
Select a balanced first annotation batch instead of sending every slate to an API:

```bash
paretojev sample-kuairand-jev \
  data/processed/kuairand/train_slates.npz \
  data/processed/kuairand/jev_sample_manifest.csv \
  --max-slates 5000 \
  --per-cell 250 \
  --max-per-user 2 \
  --log data/processed/kuairand/splits/reranker_train.csv \
  --content-directory data/raw/kuairand \
  --jev-input data/processed/kuairand/train_jev_sample.jsonl \
  --report data/processed/kuairand/jev_sample_report.json
```

The sampler balances joint cells over cold/mid/warm item history, short/medium/long
user history, standard/random exposure and no/sparse/rich positive-feedback density.
`scripts/prepare_kuairand.sh` runs the chronological split, pseudo-slate and sampling
stages. The initial 5k slate budget is a cost-controlled probe; expansion depends on
measured complementarity and bucket-level uplift.

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

### 3. Score the selected slates

After generating `train_jev_sample.jsonl`, score only that balanced subset. For a
first text-only probe:

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

### 4. Score the same sampled slates with the out-of-period behavior model

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
candidate nor any simultaneous exposure enters the user state. Candidate/context
OOV rates, Brier diagnostics and batch latency are written to the report.

### 5. Materialize a trainable NPZ

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
- **Advantage evaluator:** EMER-inspired metric feedback that corrects lagging objectives without replacing the external preference vector.
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
behavior/Jev/fusion scalarizations; separately trained EMER-like and CMR-like
checkpoints remain required for the final paper comparison.

For the strict train → evaluate → target-gate workflow:

```bash
scripts/train_precomputed.sh \
  data/processed/kuairand/train_features.npz \
  data/processed/kuairand/validation_features.npz
```

The exact array contract is documented in [`docs/NPZ_CONTRACT.md`](docs/NPZ_CONTRACT.md).

## Target mode

The isolated Phase A gates live in [`configs/targets_microlens.yaml`](configs/targets_microlens.yaml), while the full-project gates are maintained in [`configs/targets_research.yaml`](configs/targets_research.yaml):

- MicroLens hybrid AUC ≥ `0.735`, HR@5 ≥ `0.600`, NDCG@10 ≥ `0.510`;
- interim KuaiRand Pareto hypervolume relative gain ≥ `2%` over the strongest available static signal baseline;
- final KuaiRand comparison must separately clear the strongest trained EMER-like/CMR-like baseline;
- cold-item NDCG relative gain ≥ `3%`;
- worst-objective regret ≤ `0.5 percentage points`;
- preference controllability Spearman mean ≥ `0.60`;
- distilled student retains ≥ `90%` of uplift;
- student p95 latency at K=50 ≤ `30 ms` on the declared benchmark hardware.

These are **stop/go criteria**, not claimed outcomes. See [`docs/TARGET_MODE.md`](docs/TARGET_MODE.md) for escalation rules when a target is missed.

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
