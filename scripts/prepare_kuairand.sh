#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/kuairand}"
OUT_DIR="${2:-data/processed/kuairand}"
mkdir -p "$OUT_DIR"

paretojev prepare-kuairand \
  "$RAW_DIR/log_standard_4_08_to_4_21_pure.csv" \
  "$OUT_DIR/train_slates.npz" \
  --slate-size 20 \
  --min-slate-size 5 \
  --session-gap-minutes 30 \
  --manifest "$OUT_DIR/train_manifest.jsonl" \
  --jev-input "$OUT_DIR/train_jev_input.jsonl" \
  --content-directory "$RAW_DIR" \
  --report "$OUT_DIR/train_prepare_report.json"
