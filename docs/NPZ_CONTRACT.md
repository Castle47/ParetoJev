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
