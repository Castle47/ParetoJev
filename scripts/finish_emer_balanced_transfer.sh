#!/usr/bin/env bash
set -euo pipefail
training_pid=${1:?training PID required}
parent_run=${2:?completed CV directory required}
run_dir=${3:?balanced experiment directory required}
case "$training_pid" in *[!0-9]*|'') exit 2;; esac
while kill -0 "$training_pid" 2>/dev/null; do sleep 30; done
test -f "$run_dir/selection.json"
args=(--run "$parent_run" --output "$run_dir")
.venv/bin/python -u scripts/run_emer_balanced_transfer.py "${args[@]}" --stage audit
if [[ ! -f "$run_dir/evaluation.json" ]]; then
  .venv/bin/python -u scripts/run_emer_balanced_transfer.py "${args[@]}" --stage evaluate
fi
.venv/bin/python -u scripts/run_emer_balanced_transfer.py "${args[@]}" --stage replay
.venv/bin/python scripts/report_emer_balanced_transfer.py --source "$run_dir" --output reports/emer-balanced-transfer-v1
