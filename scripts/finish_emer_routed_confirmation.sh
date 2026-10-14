#!/usr/bin/env bash
set -euo pipefail
root=${1:?KuaiRand root required}
scores=${2:?complete real-score directory required}
output="$root/emer-routed-confirmation-v1"
args=(--root "$root" --scores "$scores" --output "$output")
.venv/bin/python -u scripts/evaluate_emer_routed_confirmation.py "${args[@]}" --stage prepare
if [[ ! -f "$output/evaluation.json" ]]; then
  .venv/bin/python -u scripts/evaluate_emer_routed_confirmation.py "${args[@]}" --stage evaluate
fi
.venv/bin/python -u scripts/evaluate_emer_routed_confirmation.py "${args[@]}" --stage replay
.venv/bin/python scripts/report_emer_routed_confirmation.py --source "$output" --output reports/emer-routed-confirmation-v1
