#!/usr/bin/env bash
set -euo pipefail
training_pid=${1:?live training PID required}
source_dir=${2:?audited development directory required}
run_dir=${3:?CV output directory required}
case "$training_pid" in *[!0-9]*|'') exit 2;; esac
while kill -0 "$training_pid" 2>/dev/null; do sleep 30; done
test -f "$run_dir/selection.json"
args=(--data "$source_dir" --output "$run_dir")
if [[ ! -f "$run_dir/evaluation.json" ]]; then
  .venv/bin/python -u scripts/run_emer_user_cv.py "${args[@]}" --stage evaluate
fi
.venv/bin/python -u scripts/run_emer_user_cv.py "${args[@]}" --stage replay
.venv/bin/python scripts/report_emer_user_cv.py --source "$run_dir" --output reports/emer-user-cv-v1
