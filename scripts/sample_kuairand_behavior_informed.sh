#!/usr/bin/env bash
set -euo pipefail

SLATES="${1:?Prepared KuaiRand slates NPZ required}"
BEHAVIOR_SCORES="${2:?Full-slate neural behavior score NPZ required}"
LOG="${3:?Source reranker-period KuaiRand log required}"
CONTENT_DIR="${4:?KuaiRand caption/category directory required}"
OUTPUT_DIR="${5:-data/processed/kuairand}"
MAX_SLATES="${6:-5000}"
PRIORITY_STRENGTH="${7:-0.5}"

mkdir -p "$OUTPUT_DIR"

paretojev sample-kuairand-jev \
  "$SLATES" \
  "$OUTPUT_DIR/jev_sample_manifest.csv" \
  --max-slates "$MAX_SLATES" \
  --per-cell 250 \
  --max-per-user 2 \
  --seed 0 \
  --behavior-scores "$BEHAVIOR_SCORES" \
  --behavior-priority-strength "$PRIORITY_STRENGTH" \
  --log "$LOG" \
  --content-directory "$CONTENT_DIR" \
  --jev-input "$OUTPUT_DIR/train_jev_sample.jsonl" \
  --history-length 10 \
  --report "$OUTPUT_DIR/jev_sample_report.json"

echo "Behavior-informed manifest: $OUTPUT_DIR/jev_sample_manifest.csv"
echo "Jev input: $OUTPUT_DIR/train_jev_sample.jsonl"
