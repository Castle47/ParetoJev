#!/usr/bin/env bash
set -euo pipefail

SLATES="${1:?Prepared slates NPZ required}"
MANIFEST="${2:?Sample manifest CSV required}"
DECISION_SCORES="${3:?Decision score JSONL required}"
OUTPUT_NPZ="${4:?Output precomputed NPZ required}"
FEATURE_SPEC="${5:?Shared feature-spec JSON required}"
MODE="${6:-train}"
PREFIX_SLATES="${7:-}"

case "$MODE" in
  train|eval) ;;
  *) echo "MODE must be train or eval" >&2; exit 2 ;;
esac

OUTPUT_DIR="$(dirname "$OUTPUT_NPZ")"
mkdir -p "$OUTPUT_DIR"
STEM="${OUTPUT_NPZ%.npz}"
BEHAVIOR_SCORES="${STEM}.historical_bayes.npz"
BEHAVIOR_REPORT="${STEM}.historical_bayes.report.json"
MATERIALIZE_REPORT="${STEM}.materialize.report.json"

BEHAVIOR_ARGS=(
  score-kuairand-behavior-prior
  "$SLATES"
  "$BEHAVIOR_SCORES"
  --report "$BEHAVIOR_REPORT"
)
if [[ -n "$PREFIX_SLATES" ]]; then
  BEHAVIOR_ARGS+=(--prefix-slates "$PREFIX_SLATES")
fi
paretojev "${BEHAVIOR_ARGS[@]}"

MATERIALIZE_ARGS=(
  materialize-kuairand
  "$SLATES"
  "$DECISION_SCORES"
  "$BEHAVIOR_SCORES"
  "$OUTPUT_NPZ"
  --manifest "$MANIFEST"
  --feature-spec "$FEATURE_SPEC"
  --report "$MATERIALIZE_REPORT"
)
if [[ "$MODE" == "train" ]]; then
  MATERIALIZE_ARGS+=(--fit-feature-spec)
fi
paretojev "${MATERIALIZE_ARGS[@]}"

echo "Materialized: $OUTPUT_NPZ"
echo "Behavior report: $BEHAVIOR_REPORT"
echo "Materialization report: $MATERIALIZE_REPORT"
