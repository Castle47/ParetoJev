#!/usr/bin/env bash
set -euo pipefail
dataset_root=${1:?KuaiRand root required}
score_dir=${2:?downloaded complete confirmation score directory required}
cohort_dir="$dataset_root/emer-multiobjective-confirmation-v1"
args=(--root "$dataset_root" --development "$dataset_root/emer-jev-scale-v3" --data "$cohort_dir"
      --scores "$score_dir" --output "$cohort_dir/evaluation")
if [[ ! -f "$cohort_dir/evaluation/evaluation.json" ]]; then
  .venv/bin/python -u scripts/evaluate_multiobjective_confirmation.py "${args[@]}"
fi
.venv/bin/python -u scripts/evaluate_multiobjective_confirmation.py "${args[@]}" --replay
.venv/bin/python scripts/report_multiobjective_confirmation.py --source "$cohort_dir/evaluation" \
  --output reports/emer-multiobjective-confirmation-final
