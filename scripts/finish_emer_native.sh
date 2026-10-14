#!/usr/bin/env bash
# One local run: wait for the specified training process, then evaluate/replay.
# Never launches remote compute or opens independent test data.
set -euo pipefail
train_pid=${1:?training PID required}
experiment_dir=${2:?expanded development directory required}
native_scores=${3:?verified native score directory required}
case "$train_pid" in *[!0-9]*|'') echo 'Expected numeric training PID' >&2; exit 2;; esac
while kill -0 "$train_pid" 2>/dev/null; do sleep 30; done
test -f "$experiment_dir/training-native/selection.json"
args=(--common "$experiment_dir/training" --training "$experiment_dir/training-native"
      --scores "$native_scores" --output "$experiment_dir/evaluation-native")
if [[ ! -f "$experiment_dir/evaluation-native/evaluation.json" ]]; then
  .venv/bin/python -u scripts/evaluate_emer_native.py "${args[@]}"
fi
.venv/bin/python -u scripts/evaluate_emer_native.py "${args[@]}" --replay
echo 'Development evaluation and exact replay complete; independent test unopened.'
