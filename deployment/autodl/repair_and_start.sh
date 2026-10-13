#!/usr/bin/env bash
set -euo pipefail
exec >>/root/autodl-tmp/jev-service/repair.log 2>&1
source /etc/network_turbo >/dev/null
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_DISABLE_XET=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
/root/jev-dl-env/bin/python /root/verify_model.py --repair
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /root/jev-torch-env/bin/python /root/torch_head_server.py
