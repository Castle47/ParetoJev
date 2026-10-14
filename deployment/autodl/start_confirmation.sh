#!/usr/bin/env bash
# Run only after human budget authorization and platform timed-off are recorded.
set -euo pipefail
stop_at=${1:?timezone-aware collection cutoff required}
input_dir=${2:-/root/jev-confirmation-input}
run_dir=/root/autodl-tmp/jev-confirmation-v1
mkdir -p "$run_dir"
exec 9>/root/jev-confirmation-launch.lock
flock -n 9
exec >>"$run_dir/worker.log" 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
end_epoch=$(date -d "$stop_at" +%s)
(( $(date +%s) < end_epoch ))
/root/jev-torch-env/bin/python "$input_dir/verify_confirmation_bundle.py" --input-dir "$input_dir"
service_pid=''
if ! curl --silent --fail --max-time 3 http://127.0.0.1:8000/v1/decide/info >/dev/null; then
  nohup /root/jev-torch-env/bin/python -u /root/torch_head_server.py 9>&- >"$run_dir/server.log" 2>&1 &
  service_pid=$!
fi
while ! curl --silent --fail --max-time 3 http://127.0.0.1:8000/v1/decide/info >/dev/null; do
  if (( $(date +%s) >= end_epoch )); then echo 'No service before cutoff'; exit 2; fi
  if [[ -n "$service_pid" ]] && ! kill -0 "$service_pid" 2>/dev/null; then echo 'Inference server exited'; exit 3; fi
  sleep 10
done
status=0
/root/jev-torch-env/bin/python -u "$input_dir/score_confirmation.py" \
  --input "$input_dir/jev_input.jsonl.gz" --manifest "$input_dir/transfer.json" \
  --development-registration "$input_dir/development-registration.json" \
  --output "$run_dir" --stage score --workers 16 --stop-at "$stop_at" || status=$?
/root/jev-torch-env/bin/python "$input_dir/export_confirmation.py" --output "$run_dir"
exit "$status"
