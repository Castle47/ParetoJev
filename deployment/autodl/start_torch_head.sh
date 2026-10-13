#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/autodl-tmp/jev-service
exec >>/root/autodl-tmp/jev-service/torch-setup.log 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
python -m venv --system-site-packages /root/jev-torch-env
# Keep the paid instance's already-working Torch/CUDA instead of redownloading it.
printf 'torch==2.8.0+cu128\n' >/root/jev-torch-constraints.txt
/root/jev-torch-env/bin/python -m pip install --no-cache-dir \
  -i https://mirrors.aliyun.com/pypi/simple -c /root/jev-torch-constraints.txt \
  'transformers==5.19.0' 'peft==0.21.2' 'accelerate==1.15.0'
source /etc/network_turbo >/dev/null
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
/root/jev-torch-env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json
revision='14b425269ed24452e0a5ad5a8f3e51d139647a88'
snapshot_download('autotrust/JEV-27B',revision=revision,local_dir='/root/autodl-tmp/JEV-27B',max_workers=4,
    allow_patterns=['model*.safetensors','model.safetensors.index.json','config.json','tokenizer*',
        'chat_template.jinja','adapter/*','head.safetensors','judge_config.json','calibration.json'])
Path('/root/autodl-tmp/jev-service/source.json').write_text(json.dumps(dict(repository='autotrust/JEV-27B',
    huggingface_revision=revision,implementation='pytorch-peft-direct-head-v1',experiment_scoring_started=False),indent=2))
PY
/root/jev-torch-env/bin/python -m pip freeze >/root/autodl-tmp/jev-service/torch-requirements.lock.txt
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /root/jev-torch-env/bin/python /root/torch_head_server.py
