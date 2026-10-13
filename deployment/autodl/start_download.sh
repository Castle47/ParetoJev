#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/autodl-tmp/jev-service
exec >>/root/autodl-tmp/jev-service/download.log 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
python -m venv /root/jev-dl-env
/root/jev-dl-env/bin/python -m pip install --no-cache-dir \
  --index-url https://mirrors.aliyun.com/pypi/simple huggingface_hub
source /etc/network_turbo >/dev/null
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
exec /root/jev-dl-env/bin/python /root/download_jev.py
