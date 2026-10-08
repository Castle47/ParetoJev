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
| MicroLens deterministic protocol | Implemented | `paretojev.data.microlens` |
| KuaiRand pseudo-slate builder | Implemented | `paretojev.data.kuairand` |
| Official TypeSafe Jev + open JEV clients | Implemented | `paretojev.jev` |
| SASRec + MMoE behavior branch | Implemented | `paretojev.models.behavior` |
| Confidence gate + Pareto reranker | Implemented | `paretojev.models.pareto_reranker` |
| Pareto/Tchebycheff/uncertainty losses | Implemented | `paretojev.losses.pareto` |
| Unit/integration tests | **14 passed** | `pytest` |
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

The public JEV-27B-VL recommendation demo evaluates 200 users, the five most recent videos as history, and 20 candidates consisting of one hidden next item plus 19 time-local negatives. This repository reproduces that candidate construction and adds leakage-safe cross-validated hybrid weight selection.

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
  data/processed/microlens
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
  --model jev-1.13.0
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

Every request is cached in SQLite, so interrupted or repeated runs do not repay identical calls.

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
  --hybrid itemcf jev_click \
  --folds 5 \
  --weight-step 0.05 \
  --objective balanced \
  --output artifacts/microlens_metrics.json
```

Weights are selected on K-1 user folds and applied only to the held-out fold. This avoids selecting a blend on the same 200 users used for reporting.

## KuaiRand workflow

Place the official files under `data/raw/kuairand/` or another local path. The minimum primary-track inputs are:

```text
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
log_random_4_22_to_5_08_pure.csv
kuairand_video_captions.csv
kuairand_video_categories.csv
```

Build exposed-session pseudo-slates:

```bash
paretojev prepare-kuairand \
  data/raw/kuairand/log_standard_4_08_to_4_21_pure.csv \
  data/processed/kuairand/train_slates.npz \
  --slate-size 20 \
  --min-slate-size 5 \
  --session-gap-minutes 30 \
  --manifest data/processed/kuairand/train_manifest.jsonl \
  --jev-input data/processed/kuairand/train_jev_input.jsonl \
  --content-directory data/raw/kuairand \
  --report data/processed/kuairand/train_prepare_report.json
```

The main experiment sequence is specified in [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). The first Jev run should be a stratified subset, not the full dataset: cold/warm items, short/long histories, and standard/random exposure buckets.

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
  --config configs/train_precomputed.yaml \
  --output artifacts/train_metrics.json \
  --checkpoint artifacts/paretojev.pt
```

The exact array contract is documented in [`docs/NPZ_CONTRACT.md`](docs/NPZ_CONTRACT.md).

## Target mode

The top-level gates are maintained in [`configs/targets_research.yaml`](configs/targets_research.yaml):

- MicroLens hybrid AUC ≥ `0.735`, HR@5 ≥ `0.600`, NDCG@10 ≥ `0.510`;
- KuaiRand Pareto hypervolume relative gain ≥ `2%` over the strongest EMER-like baseline;
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
