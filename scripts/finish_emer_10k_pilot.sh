#!/usr/bin/env bash
set -euo pipefail
root=${1:?KuaiRand root required}
scores=${2:?complete pilot score directory required}
pilot="$root/emer-10k-pilot-v1"
development="$root/emer-jev-scale-v3"
bundle="$development/exports/emer-matched-blended-v1"
output="$root/emer-10k-pilot-evaluation-v1"
args=(--root "$root" --pilot "$pilot" --development "$development" --bundle "$bundle" --scores "$scores" --output "$output")
.venv/bin/python -u scripts/evaluate_emer_10k_pilot.py "${args[@]}" --stage prepare
if [[ ! -f "$output/evaluation.json" ]]; then
  .venv/bin/python -u scripts/evaluate_emer_10k_pilot.py "${args[@]}" --stage evaluate
fi
.venv/bin/python -u scripts/evaluate_emer_10k_pilot.py "${args[@]}" --stage replay
.venv/bin/python scripts/report_emer_10k_pilot.py --source "$output" --output reports/emer-10k-pilot-v1
