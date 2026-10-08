#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/microlens}"
OUT_DIR="${2:-data/processed/microlens}"

scripts/prepare_microlens.sh "$RAW_DIR" "$OUT_DIR"
scripts/score_microlens_typesafe.sh \
  "$OUT_DIR/jev_input.jsonl" \
  artifacts/microlens_jev_choice.jsonl \
  artifacts/microlens_jev_choice.report.json \
  artifacts/microlens_jev_choice.errors.jsonl
scripts/evaluate_microlens_hybrid.sh \
  "$OUT_DIR/candidates.csv" \
  artifacts/microlens_jev_choice.jsonl \
  "$OUT_DIR/candidates_with_jev.csv" \
  artifacts/microlens_metrics.json \
  "$OUT_DIR/evaluated_scores.csv" \
  artifacts/microlens_target_report.json
