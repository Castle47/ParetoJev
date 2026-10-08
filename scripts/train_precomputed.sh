#!/usr/bin/env bash
set -euo pipefail

TRAIN_NPZ="${1:?train NPZ required}"
VALIDATION_NPZ="${2:?validation NPZ required}"
TRAIN_METRICS="${3:-artifacts/train_metrics.json}"
PARETO_METRICS="${4:-artifacts/pareto_metrics.json}"
TARGET_REPORT="${5:-artifacts/pareto_target_report.json}"
CHECKPOINT="${6:-artifacts/paretojev.pt}"

DIRECTIONS="1,1,1,1,1,-1"
OBJECTIVES="is_click,long_view,is_like,is_follow,deep_engagement,is_hate"

paretojev train-precomputed "$TRAIN_NPZ" "$VALIDATION_NPZ" \
  --directions "$DIRECTIONS" \
  --objective-names "$OBJECTIVES" \
  --config configs/train_precomputed.yaml \
  --output "$TRAIN_METRICS" \
  --checkpoint "$CHECKPOINT"

paretojev evaluate-pareto "$CHECKPOINT" "$VALIDATION_NPZ" \
  --batch-size 128 \
  --top-k 10 \
  --random-preferences 32 \
  --include-pairwise-preferences \
  --dirichlet-alpha 0.7 \
  --hypervolume-samples 100000 \
  --cold-quantile 0.25 \
  --seed 42 \
  --output "$PARETO_METRICS"

# Target mode is intentionally strict: a missed gate returns a non-zero status.
paretojev check-targets "$PARETO_METRICS" \
  configs/targets_kuairand_interim.yaml \
  --output "$TARGET_REPORT"
