#!/usr/bin/env bash
# Prepare the exact frozen producer. Does not score or open test outcomes.
set -euo pipefail
input_dir=${1:-/root/jev-pilot-input}
mkdir -p /root/autodl-tmp/jev-service
exec >>/root/autodl-tmp/jev-service/pilot-setup.log 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
python -m venv --system-site-packages /root/jev-torch-env
printf 'torch==2.8.0+cu128\n' >/root/jev-torch-constraints.txt
/root/jev-torch-env/bin/python -m pip install --no-cache-dir \
  -i https://mirrors.aliyun.com/pypi/simple -c /root/jev-torch-constraints.txt \
  'transformers==5.19.0' 'huggingface_hub==2.2.0' 'peft==0.21.2' 'accelerate==1.15.0' \
  'flash-linear-attention==0.5.2' 'ninja' 'wheel'
if command -v nvcc >/dev/null; then
  CAUSAL_CONV1D_FORCE_BUILD=TRUE /root/jev-torch-env/bin/python -m pip install --no-cache-dir --no-build-isolation \
    -i https://mirrors.aliyun.com/pypi/simple -c /root/jev-torch-constraints.txt 'causal-conv1d==1.7.0'
else
  /root/jev-torch-env/bin/python -m pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple -c /root/jev-torch-constraints.txt 'causal-conv1d==1.7.0'
fi
source /etc/network_turbo >/dev/null
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
/root/jev-torch-env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'autotrust/JEV-27B',
    revision='14b425269ed24452e0a5ad5a8f3e51d139647a88',
    local_dir='/root/autodl-tmp/JEV-27B',
    max_workers=4,
    allow_patterns=[
        'model.safetensors.index.json','config.json','tokenizer*',
        'chat_template.jinja','adapter/*','adapter_vllm/*','head.safetensors',
        'judge_config.json','calibration.json','serve_decide.py'
    ]
)
print('PINNED_PILOT_SMALL_FILES_COMPLETE', flush=True)
PY
model_url='https://hf-mirror.com/autotrust/JEV-27B/resolve/14b425269ed24452e0a5ad5a8f3e51d139647a88'
export model_url
seq -f '%05g' 1 13 | xargs -P 13 -I{} bash -c \
  'curl -L --fail --retry 20 --retry-all-errors --connect-timeout 10 -C - \
    -o "/root/autodl-tmp/JEV-27B/model-{}-of-00013.safetensors" \
    "$model_url/model-{}-of-00013.safetensors"'
echo PINNED_PILOT_MODEL_SHARDS_COMPLETE
cp "$input_dir/torch_head_server.py" /root/torch_head_server.py
/root/jev-torch-env/bin/python -m pip freeze >/root/autodl-tmp/jev-service/pilot-requirements.lock.txt
/root/jev-torch-env/bin/python "$input_dir/verify_pilot_bundle.py" --input-dir "$input_dir"
rm -rf /root/autodl-tmp/hf-cache
echo PILOT_PRODUCER_SETUP_COMPLETE
