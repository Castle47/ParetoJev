#!/usr/bin/env bash
set -euo pipefail

CANDIDATES="${1:-data/processed/microlens/candidates.csv}"
JEV_SCORES="${2:-artifacts/microlens_jev_choice.jsonl}"
MERGED="${3:-data/processed/microlens/candidates_with_jev.csv}"
METRICS="${4:-artifacts/microlens_metrics.json}"

paretojev merge-decision-scores "$CANDIDATES" "$JEV_SCORES" "$MERGED" \
  --objective click --score-column jev_click
paretojev evaluate-microlens "$MERGED" \
  --scores itemcf title_sim jev_click \
  --hybrid itemcf jev_click \
  --folds 5 --weight-step 0.05 --objective balanced \
  --output "$METRICS"
