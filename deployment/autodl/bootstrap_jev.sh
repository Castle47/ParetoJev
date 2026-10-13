#!/usr/bin/env bash
# Run on the newly rented AutoDL instance, not on the local Mac.
# No experiment scores are generated or mixed by this bootstrap.
set -euo pipefail
jev_work=/root/autodl-tmp/jev-service
jev_env=/root/jev-env
jev_model=/root/autodl-tmp/JEV-27B
mkdir -p "$jev_work"
exec >>"$jev_work/setup.log" 2>&1
date -Is
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m venv "$jev_env"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
"$jev_env/bin/python" -m pip install --index-url https://pypi.org/simple --no-cache-dir 'vllm==0.31.0' huggingface_hub
# AutoDL's documented academic download proxy; do not print proxy credentials.
if test -f /etc/network_turbo; then source /etc/network_turbo >/dev/null; fi
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
"$jev_env/bin/python" - <<'PY'
from pathlib import Path
import json
from huggingface_hub import snapshot_download
repo='autotrust/JEV-27B'
revision='14b425269ed24452e0a5ad5a8f3e51d139647a88'
folder='/root/autodl-tmp/JEV-27B'
snapshot_download(repo, revision=revision, local_dir=folder, max_workers=4,
    allow_patterns=['model*.safetensors','model.safetensors.index.json','config.json',
        'tokenizer*','chat_template.jinja','adapter_vllm/*','calibration.json',
        'judge_config.json','serve_decide.py'])
Path('/root/autodl-tmp/jev-service/source.json').write_text(json.dumps({
    'repository':repo,'huggingface_revision':revision,'vllm':'0.31.0',
    'public_service_compatibility':'not yet established; do not reuse public cache identity',
    'experiment_scoring_started':False},indent=2)+chr(10))
PY
"$jev_env/bin/python" -m pip freeze >"$jev_work/requirements.lock.txt"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export JEV_DECIDE_THINKING=off
export HF_HUB_OFFLINE=1
exec "$jev_env/bin/python" "$jev_model/serve_decide.py" \
  --model "$jev_model" --served-model-name autotrust/JEV-27B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --enable-lora --max-lora-rank 32 \
  --lora-modules "jev-decision=$jev_model/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --enforce-eager
