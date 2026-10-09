#!/usr/bin/env bash
set -euo pipefail
# Active 27K path. The earlier in-memory Pure workflow is explicitly retained in
# prepare_kuairand_pure_legacy.sh and is never an automatic fallback.
RAW_DIR="${1:-data/raw/kuairand-27k}"
REPORT="${2:-reports/kuairand-27k-readiness-2026-10-09/audit.json}"
"${PARETOJEV_PYTHON:-.venv/bin/python}" scripts/prepare_kuairand_27k.py audit \
  --root "$RAW_DIR" --report "$REPORT"
