#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/kuairand}"
OUT_DIR="${2:-data/processed/kuairand}"
mkdir -p "$OUT_DIR"

TRAIN_LOG="$RAW_DIR/log_standard_4_08_to_4_21_pure.csv"
SLATES="$OUT_DIR/train_slates.npz"

paretojev prepare-kuairand \
  "$TRAIN_LOG" \
  "$SLATES" \
  --slate-size 20 \
  --min-slate-size 5 \
  --session-gap-minutes 30 \
  --manifest "$OUT_DIR/train_manifest.jsonl" \
  --report "$OUT_DIR/train_prepare_report.json"

# Do not send every training slate to an API. Build a balanced, auditable subset
# across cold/mid/warm items, short/medium/long histories, standard/random
# exposure and feedback density.
paretojev sample-kuairand-jev \
  "$SLATES" \
  "$OUT_DIR/jev_sample_manifest.csv" \
  --max-slates 5000 \
  --per-cell 250 \
  --max-per-user 2 \
  --seed 0 \
  --log "$TRAIN_LOG" \
  --content-directory "$RAW_DIR" \
  --jev-input "$OUT_DIR/train_jev_sample.jsonl" \
  --history-length 10 \
  --report "$OUT_DIR/jev_sample_report.json"
