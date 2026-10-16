#!/usr/bin/env bash
set -euo pipefail

cd /root/MAIA
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${CUDA_VISIBLE_DEVICES:-0}
TASK=${TASK:-insert}
OBS_TEMPLATE=${OBS_TEMPLATE:-configs/baselines/single/logpzo_observed.yaml}
WM_TEMPLATE=${WM_TEMPLATE:-configs/baselines/single/logpzo_wm.yaml}
# Example: TASK=press bash scripts/logpzo.sh
OBS_CONFIG=$(mktemp "/tmp/maia_logpzo_observed_${TASK}.XXXXXX.yaml")
WM_CONFIG=$(mktemp "/tmp/maia_logpzo_wm_${TASK}.XXXXXX.yaml")
trap 'rm -f "$OBS_CONFIG" "$WM_CONFIG"' EXIT

# Reuse the two single-task YAMLs as templates; only task-specific paths/names change.
sed "s|insert|${TASK}|g" "$OBS_TEMPLATE" > "$OBS_CONFIG"
sed "s|insert|${TASK}|g" "$WM_TEMPLATE" > "$WM_CONFIG"
CACHE_ROOT=/root/MAIA/cache/$TASK
OBS_OUTPUT_ROOT=/root/MAIA/outputs/$TASK/logpzo_observed
WM_OUTPUT_ROOT=/root/MAIA/outputs/$TASK/logpzo_wm

latest_file() { find "$1" -type f -name "$2" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-; }

# 01. Build the observed feature cache used by observed logPZO.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" data/utils/precompute_observed_features.py \
  --config_file "$OBS_CONFIG" \
  --output_dir "$CACHE_ROOT/observed_feature" \
  --splits train val test \
  --batch_size 64 \
  --num_workers 4

# 02. Confirm the WM feature cache used by WM logPZO exists.
test -f "$CACHE_ROOT/video_action_force_anomaly/train/index.json"
test -f "$CACHE_ROOT/video_action_force_anomaly/val/index.json"
test -f "$CACHE_ROOT/video_action_force_anomaly/test/index.json"

# 03. Train observed logPZO on the configured task only.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29701} \
  bash scripts/train.sh \
  main.py \
  "$OBS_CONFIG" \
  runner/logpzo_observed_trainer.py
OBS_CKPT=$(latest_file "$OBS_OUTPUT_ROOT" logpzo_observed.pt)
OBS_RUN_DIR=$(dirname "$(dirname "$OBS_CKPT")")

# 04. Evaluate observed logPZO with validation calibration and that task's test data.
"$PYTHON_BIN" analyze/eval_logpzo_metrics.py \
  --config_file "$OBS_CONFIG" \
  --logpzo_checkpoint "$OBS_CKPT" \
  --output_dir "$OBS_RUN_DIR/eval" \
  --calibration_split val \
  --eval_split test \
  --cp_alpha 0.2

# 05. Train WM logPZO on the same configured task only.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29702} \
  bash scripts/train.sh \
  main.py \
  "$WM_CONFIG" \
  runner/logpzo_wm_trainer.py
WM_CKPT=$(latest_file "$WM_OUTPUT_ROOT" logpzo_wm.pt)
WM_RUN_DIR=$(dirname "$(dirname "$WM_CKPT")")

# 06. Evaluate WM logPZO with validation calibration and that task's test data.
"$PYTHON_BIN" analyze/eval_logpzo_metrics.py \
  --config_file "$WM_CONFIG" \
  --logpzo_checkpoint "$WM_CKPT" \
  --output_dir "$WM_RUN_DIR/eval" \
  --calibration_split val \
  --eval_split test \
  --cp_alpha 0.2
