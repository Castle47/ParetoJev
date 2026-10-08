#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/microlens}"
OUT_DIR="${2:-data/processed/microlens}"

paretojev download-microlens "$RAW_DIR"
paretojev prepare-microlens "$RAW_DIR" "$OUT_DIR"
