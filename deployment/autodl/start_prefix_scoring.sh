#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/autodl-tmp/jev-native-v9-fp16
exec >>/root/autodl-tmp/jev-native-v9-fp16/worker.log 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
collection_end=$(date -d '2026-10-14T00:55:00+08:00' +%s)
while ! curl --silent --fail --max-time 3 http://127.0.0.1:8000/v1/decide/info | /root/jev-torch-env/bin/python -c 'import json,sys; sys.exit(json.load(sys.stdin).get("implementation")!="pytorch-peft-group-v3")' 2>/dev/null; do
  if (( $(date +%s) >= collection_end )); then echo 'No healthy service before deadline'; exit 2; fi
  sleep 30
done
exec /root/jev-torch-env/bin/python /root/score_native.py \
  --input /root/jev_input.jsonl.gz --manifest /root/transfer.json \
  --output /root/autodl-tmp/jev-native-v9-fp16 --stage score --workers 16 \
  --stop-at '2026-10-14T00:55:00+08:00'
