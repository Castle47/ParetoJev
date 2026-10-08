#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/microlens/jev_input.jsonl}"
OUTPUT="${2:-artifacts/microlens_jev_choice.jsonl}"
REPORT="${3:-artifacts/microlens_jev_choice.report.json}"
ERRORS="${4:-artifacts/microlens_jev_choice.errors.jsonl}"

: "${TYPESAFE_API_KEY:?Set TYPESAFE_API_KEY first}"
paretojev score-jsonl "$INPUT" "$OUTPUT" \
  --provider typesafe \
  --mode choice \
  --objectives click \
  --model jev-1.13.0 \
  --durable-every 1 \
  --report "$REPORT" \
  --errors "$ERRORS"
