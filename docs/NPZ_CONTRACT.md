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
