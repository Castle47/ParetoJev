#!/usr/bin/env bash
set -euo pipefail

TRAIN_NPZ="${1:?train NPZ required}"
VALIDATION_NPZ="${2:?validation NPZ required}"

paretojev train-precomputed "$TRAIN_NPZ" "$VALIDATION_NPZ" \
  --directions 1,1,1,1,1,-1 \
  --config configs/train_precomputed.yaml \
  --output artifacts/train_metrics.json \
  --checkpoint artifacts/paretojev.pt
