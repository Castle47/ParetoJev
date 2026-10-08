#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/microlens/jev_input.jsonl}"
OUTPUT="${2:-artifacts/microlens_jev_choice.jsonl}"

: "${TYPESAFE_API_KEY:?Set TYPESAFE_API_KEY first}"
paretojev score-jsonl "$INPUT" "$OUTPUT" \
  --provider typesafe \
  --mode choice \
  --model jev-1.13.0
