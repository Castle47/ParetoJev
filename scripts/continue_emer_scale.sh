#!/usr/bin/env bash
# Resume only after existing workers exit; their locks reject duplicate workers.
set -euo pipefail
cd "$(dirname "$0")/.."
dataset_root="${1:-/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k}"
scale_data="$dataset_root/emer-jev-scale-v3"
scale_config="configs/emer_jev_scale.json"
.venv/bin/python scripts/run_emer_scale.py --root "$dataset_root" --data "$scale_data" \
  --output "$scale_data/training" --config "$scale_config" --arm emer
.venv/bin/python scripts/score_kuairand_core.py --data "$scale_data" --output "$scale_data/jev" --config "$scale_config"
.venv/bin/python scripts/run_emer_scale.py --root "$dataset_root" --data "$scale_data" \
  --output "$scale_data/training" --config "$scale_config" --arm emer_jev
