#!/usr/bin/env bash
set -euo pipefail
exec >>/root/autodl-tmp/jev-service/fast-install.log 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export MAX_JOBS=8
/root/jev-torch-env/bin/python -m pip install --no-cache-dir \
  -i https://mirrors.aliyun.com/pypi/simple -c /root/jev-torch-constraints.txt \
  'flash-linear-attention==0.5.2' ninja wheel
if command -v nvcc >/dev/null; then
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  timeout 720 /root/jev-torch-env/bin/python -m pip install --no-cache-dir --no-build-isolation \
    -i https://mirrors.aliyun.com/pypi/simple -c /root/jev-torch-constraints.txt 'causal-conv1d==1.7.0'
else
  echo 'No nvcc; use installed FLA with PyTorch convolution fallback.'
fi
/root/jev-torch-env/bin/python -m pip freeze >/root/autodl-tmp/jev-service/fast-requirements.lock.txt
echo FAST_KERNEL_SETUP_COMPLETE
