#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw/kuairand}"
OUT_DIR="${2:-data/processed/kuairand}"
mkdir -p "$OUT_DIR"

FULL_EARLY_LOG="$RAW_DIR/log_standard_4_08_to_4_21_pure.csv"
SPLIT_DIR="$OUT_DIR/splits"
mkdir -p "$SPLIT_DIR"

# Reserve an early behavior-train period, a later behavior-validation period and
# an unseen reranker-train period. Equal timestamps are never split across files.
paretojev split-kuairand-log \
  "$FULL_EARLY_LOG" \
  "$SPLIT_DIR" \
  --fractions 0.6,0.2,0.2 \
  --names behavior_train,behavior_validation,reranker_train \
  --write-prefixes \
  --report "$SPLIT_DIR/temporal_split_report.json"

RERANKER_LOG="$SPLIT_DIR/reranker_train.csv"
RERANKER_PREFIX="$SPLIT_DIR/prefix_before_reranker_train.csv"
SLATES="$OUT_DIR/train_slates.npz"

paretojev prepare-kuairand \
  "$RERANKER_LOG" \
  "$SLATES" \
  --prefix-log "$RERANKER_PREFIX" \
  --slate-size 20 \
  --min-slate-size 5 \
  --session-gap-minutes 30 \
  --manifest "$OUT_DIR/train_manifest.jsonl" \
  --report "$OUT_DIR/train_prepare_report.json"

# Do not send every training slate to an API. Build a balanced, auditable subset
# across cold/mid/warm items, short/medium/long histories, standard/random
# exposure and feedback density.
paretojev sample-kuairand-jev \
  "$SLATES" \
  "$OUT_DIR/jev_sample_manifest.csv" \
  --max-slates 5000 \
  --per-cell 250 \
  --max-per-user 2 \
  --seed 0 \
  --log "$RERANKER_LOG" \
  --content-directory "$RAW_DIR" \
  --jev-input "$OUT_DIR/train_jev_sample.jsonl" \
  --history-length 10 \
  --report "$OUT_DIR/jev_sample_report.json"
