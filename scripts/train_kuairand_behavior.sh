#!/usr/bin/env bash
# Legacy in-memory workflow: do not pass complete 27K shards here.
# Current 27K entry point: scripts/train_kuairand_behavior_pilot.py.
set -euo pipefail

TRAIN_LOG="${1:?Chronologically early behavior-train log required}"
VALIDATION_LOG="${2:?Later behavior-validation log required}"
CHECKPOINT="${3:-artifacts/kuairand_behavior.pt}"
REPORT="${4:-artifacts/kuairand_behavior_metrics.json}"
DEVICE="${5:-cpu}"

mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$REPORT")"

paretojev train-kuairand-behavior \
  "$TRAIN_LOG" \
  "$VALIDATION_LOG" \
  "$CHECKPOINT" \
  --config configs/train_behavior.yaml \
  --device "$DEVICE" \
  --output "$REPORT"

echo "Behavior checkpoint: $CHECKPOINT"
echo "Behavior report: $REPORT"
