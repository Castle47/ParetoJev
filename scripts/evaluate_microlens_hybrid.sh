#!/usr/bin/env bash
set -euo pipefail

CANDIDATES="${1:-data/processed/microlens/candidates.csv}"
JEV_SCORES="${2:-artifacts/microlens_jev_choice.jsonl}"
MERGED="${3:-data/processed/microlens/candidates_with_jev.csv}"
METRICS="${4:-artifacts/microlens_metrics.json}"
EVALUATED="${5:-data/processed/microlens/evaluated_scores.csv}"
TARGET_REPORT="${6:-artifacts/microlens_target_report.json}"

paretojev merge-decision-scores "$CANDIDATES" "$JEV_SCORES" "$MERGED" \
  --objective click --score-column jev_click

paretojev evaluate-microlens "$MERGED" \
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
  --confidence 0.95 \
  --tie-policy upstream_jitter \
  --tie-seed 1 \
  --output "$METRICS" \
  --output-scores "$EVALUATED"

paretojev check-targets "$METRICS" configs/targets_microlens.yaml \
  --output "$TARGET_REPORT"
