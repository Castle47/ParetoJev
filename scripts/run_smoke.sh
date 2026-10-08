#!/usr/bin/env bash
set -euo pipefail

paretojev synthetic-smoke \
  --config configs/synthetic_smoke.yaml \
  --output artifacts/smoke_metrics.json \
  --checkpoint artifacts/smoke_model.pt

paretojev check-targets \
  artifacts/smoke_metrics.json \
  configs/targets_smoke.yaml \
  --output artifacts/smoke_target_report.json
