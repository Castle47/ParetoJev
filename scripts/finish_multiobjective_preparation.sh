#!/usr/bin/env bash
set -euo pipefail
preparation_pid=${1:?preparation PID required}
dataset_root=${2:?dataset root required}
cohort_dir="$dataset_root/emer-multiobjective-confirmation-v1"
case "$preparation_pid" in *[!0-9]*|'') exit 2;; esac
while kill -0 "$preparation_pid" 2>/dev/null; do sleep 30; done
test -f "$cohort_dir/preparation.json"
.venv/bin/python -u scripts/audit_multiobjective_cohort.py --root "$dataset_root" --data "$cohort_dir" \
  --config configs/emer_multiobjective_confirmation.json
.venv/bin/python -u scripts/package_multiobjective_confirmation.py --data "$cohort_dir" \
  --output "$cohort_dir/transfer" --config configs/emer_multiobjective_confirmation.json
printf '%s\n' 'Frozen source cohort audited and packaged; no inference or held-out metric evaluation performed.'
