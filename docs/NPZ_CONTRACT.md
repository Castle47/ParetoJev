# Precomputed NPZ Contract

`paretojev train-precomputed` decouples expensive feature production from final reranker training. A train/validation NPZ must contain:

| Key | Shape | Meaning |
|---|---|---|
| `candidate_features` | `[N,K,F]` | pre-exposure numeric/content features |
| `behavior_scores` | `[N,K,M]` | behavior model probabilities |
| `jev_scores` | `[N,K,M]` | Jev/decision probabilities |
| `labels` | `[N,K,M]` | observed multi-objective labels |
| `mask` | `[N,K]` | valid candidate mask |
| `jev_confidence` | `[N,K,M]` | official Choice confidence or documented proxy |
| `jev_entropy` | `[N,K,M]` | normalized uncertainty |
| `popularity` | `[N,K]` or `[N,K,1]` | history-only popularity feature |
| `history_length` | `[N]` or `[N,1]` | user history length |
| `preference_weights` | `[N,M]` | optional; generated from Dirichlet if absent |
| `target_probabilities` | `[N,K,M]` | optional soft targets for controlled experiments |

All probabilities use raw behavior semantics: for example `is_hate=1` remains high hate risk. Optimization directions are passed separately:

```bash
paretojev train-precomputed train.npz validation.npz \
  --directions 1,1,1,1,1,-1 \
  --config configs/train_precomputed.yaml
```

Do not place post-outcome fields such as watch time into `candidate_features` for a pre-ranking experiment.

## Checkpoint evaluation

Training can persist stable objective names:

```bash
paretojev train-precomputed train.npz validation.npz \
  --directions 1,1,1,1,1,-1 \
  --objective-names is_click,long_view,is_like,is_follow,deep_engagement,is_hate \
  --checkpoint artifacts/paretojev.pt
```

The checkpoint stores the model architecture, objective directions and names. A
runtime-preference sweep can then be reproduced without repeating those fields:

```bash
paretojev evaluate-pareto artifacts/paretojev.pt validation.npz \
  --top-k 10 \
  --random-preferences 32 \
  --include-pairwise-preferences \
  --output artifacts/pareto_metrics.json
```

All objective coordinates in the report are maximize-oriented utilities in `[0,1]`:
positive objectives use conditional NDCG@K, while negative objectives use one minus
the selected top-K negative-feedback rate. Hypervolume methods share the same unit
hypercube and the same Monte Carlo seed, avoiding method-specific integration boxes.

## Upstream behavior-score interchange contract

Before building the final training NPZ, `materialize-kuairand` consumes a separate
`SlateScoreTensor` NPZ. This keeps behavior-model training and scoring independent
from Jev annotation and final reranker training.

| Key | Shape | Meaning |
|---|---|---|
| `slate_ids` | `[N]` | row IDs in the source `PreparedKuaiRandSlates` file |
| `candidate_ids` | `[N,K]` | exact candidate IDs in source order |
| `probabilities` | `[N,K,M]` | raw behavior-risk probabilities |
| `mask` | `[N,K]` | valid candidate mask |
| `objective_names` | `[M]` | source objective names |

The materializer validates all of the following before copying a row:

- every requested `slate_id` occurs exactly once;
- the valid mask matches the prepared slate;
- valid candidate IDs and their order match exactly;
- each target objective has exactly one source objective or declared alias;
- valid probabilities are finite and in `[0,1]`.


Preferred producer for the main experiment:

```bash
paretojev score-kuairand-behavior \
  artifacts/kuairand_behavior.pt \
  data/processed/kuairand/train_slates.npz \
  data/processed/kuairand/splits/reranker_train.csv \
  data/processed/kuairand/train_behavior_scores.npz \
  --prefix-log data/processed/kuairand/splits/prefix_before_reranker_train.csv
```

The checkpoint vocabulary and context transform are fit only on the earlier behavior
training period. Histories and popularity are cut strictly before each slate start,
and OOV rates are recorded. The reranker-training period must not be part of behavior
model fitting.

The CPU command below creates a leakage-safe bootstrap tensor:

```bash
paretojev score-kuairand-behavior-prior \
  data/processed/kuairand/train_slates.npz \
  data/processed/kuairand/train_behavior_prior.npz
```

It is an empirical-Bayes prequential prior, not the final behavior baseline. Final
research tables must replace or compare it with a trained SASRec+MMoE/PLE branch.

## KuaiRand materialization

```bash
paretojev materialize-kuairand \
  data/processed/kuairand/train_slates.npz \
  artifacts/kuairand_train_jev.jsonl \
  data/processed/kuairand/train_behavior_scores.npz \
  data/processed/kuairand/train_features.npz \
  --manifest data/processed/kuairand/jev_sample_manifest.csv \
  --feature-spec data/processed/kuairand/candidate_feature_spec.json \
  --fit-feature-spec
```

The feature specification is fitted on training slates only and reused for later
splits. Its default features are strict prior item count, duration and tab indicators.
Pseudo-slate sequence position is excluded by default because it can carry the logged
production order; include it only as a declared ablation.

## Strict comparison protocol: `baseline-harness-v1`

The `compare-baselines` loader strengthens the legacy contract above. Every split
requires the same tensor inputs plus this metadata:

| Key | Shape | Meaning |
|---|---|---|
| `slate_ids` | `[N]` | nonempty integer/string IDs, unique within the split |
| `candidate_ids` | `[N,K]` | candidate IDs in exact order; valid positions nonempty |
| `user_ids` | `[N]` | nonempty integer/string IDs for paired user-cluster bootstrap |
| `timestamps` | `[N,K]` | nonnegative integer event times at valid positions |
| `objective_names` | `[M]` | unique ordered names |
| `directions` | `[M]` | ordered `+1` / `-1` optimization directions |
| `feature_names` | `[F]` | unique names in feature-column order |
| `feature_spec_hash` | scalar string | canonical JSON SHA-256 of the shared train-fitted transform |
| `data_kind` | scalar string | `synthetic` or `real`, matching the manifest |

`mask` must be stored as boolean. Every row needs at least two valid candidates,
with nonempty features and at least two objectives. All numeric arrays, including
padding, must be finite. Valid labels, behavior/Jev scores, confidence and entropy
are in `[0,1]`; popularity/history and preference weights are nonnegative. Request
preferences sum to one. The loader zeroes padded 3-D tensor values. If preferences
are omitted, the legacy loader deterministically generates them; every method sees
the same resulting preferences. `target_probabilities` is **forbidden** here: no
trainer receives privileged simulator targets.

Train, validation and test must share dimensions, objective directions/order,
feature names and feature hash. Valid timestamp ranges must be strictly separated;
a timestamp cannot cross a split boundary. Slate IDs are split-local, because
materialization may restart their row index. Audit identity therefore combines the
split hash with slate/user/candidate/timestamp metadata, rather than assuming a
slate ID is globally unique.

`materialize-kuairand` preserves these identity fields and emits the feature hash
and directions. Its `--data-kind {unverified,synthetic,real}` defaults to
`unverified`: materialization alone does not establish data origin. The strict
comparison loader rejects unverified inputs. Set `real` only for established real
inputs and preserve their upstream evidence; relabeling a fixture is not verification.

A manifest registers SHA-256 hashes for all three NPZ files. Real runs additionally
require hashed `feature_spec`, `behavior_provenance`, `decision_provenance` and
`evaluation_protocol` documents. The feature spec JSON's canonical content hash is
separate from its file-byte SHA-256. Hashes establish consistency, not the truth of
upstream claims. The manifest example and minimum handoff are in
[BASELINE_HARNESS.md](BASELINE_HARNESS.md).
