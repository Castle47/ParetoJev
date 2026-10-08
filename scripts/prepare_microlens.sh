#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/microlens}"
OUT_DIR="${2:-data/processed/microlens}"

paretojev download-microlens "$RAW_DIR"
paretojev prepare-microlens "$RAW_DIR" "$OUT_DIR" \
  --users 200 \
  --negatives 19 \
  --history 5 \
  --window-days 3 \
  --seed 0 \
  --negative-sampling upstream_occurrence \
  --tie-policy upstream_jitter \
  --tie-seed 1
