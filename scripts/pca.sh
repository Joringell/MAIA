#!/usr/bin/env bash
set -euo pipefail

cd /root/MAIA
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${CUDA_VISIBLE_DEVICES:-0}
TASK=${TASK:-insert}
CONFIG_TEMPLATE=${CONFIG_TEMPLATE:-configs/baselines/single/pca_kmeans.yaml}
# Example: TASK=press bash scripts/pca.sh
CONFIG=$(mktemp "/tmp/maia_pca_${TASK}.XXXXXX.yaml")
trap 'rm -f "$CONFIG"' EXIT

# Reuse the single-task YAML as a template; only task-specific paths/names change.
sed "s|insert|${TASK}|g" "$CONFIG_TEMPLATE" > "$CONFIG"
CACHE_ROOT=/root/MAIA/cache/$TASK
OUTPUT_ROOT=/root/MAIA/outputs/$TASK/pca_kmeans_vsf

latest_file() { find "$1" -type f -name "$2" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-; }

# 01. Build the observed video/state/force feature cache for the configured task.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" data/utils/precompute_observed_features.py \
  --config_file "$CONFIG" \
  --output_dir "$CACHE_ROOT/observed_feature" \
  --splits train val test \
  --batch_size 64 \
  --num_workers 4

# 02. Confirm the WM feature cache required by the second PCA branch exists.
test -f "$CACHE_ROOT/video_action_force_anomaly/train/index.json"
test -f "$CACHE_ROOT/video_action_force_anomaly/val/index.json"
test -f "$CACHE_ROOT/video_action_force_anomaly/test/index.json"

# 03. Copy observed episode labels to the WM cache for normal-window filtering.
"$PYTHON_BIN" data/utils/build_anomaly_cache_episode_labels_from_observed.py \
  --observed_cache_dir "$CACHE_ROOT/observed_feature" \
  --target_cache_dir "$CACHE_ROOT/video_action_force_anomaly" \
  --splits train val test \
  --strict

# 04. Fit the observed and WM PCA-KMeans branches for this one task only.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29655} \
  bash scripts/train.sh \
  main.py \
  "$CONFIG" \
  runner/pca_kmeans_trainer.py

# 05. Evaluate the observed-feature branch with pointwise conformal calibration.
PCA_RUN_DIR=$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
"$PYTHON_BIN" analyze/eval_pca_kmeans_metrics.py \
  --config_file "$CONFIG" \
  --pca_kmeans_checkpoint "$PCA_RUN_DIR/step_0/pca_kmeans.pt" \
  --output_dir "$PCA_RUN_DIR/step_0/eval_pointwise" \
  --calibration_split val \
  --eval_split test \
  --cp_method pointwise \
  --cp_alpha 0.2

# 06. Evaluate the observed-feature branch with one global threshold.
"$PYTHON_BIN" analyze/eval_pca_kmeans_metrics.py \
  --config_file "$CONFIG" \
  --pca_kmeans_checkpoint "$PCA_RUN_DIR/step_0/pca_kmeans.pt" \
  --output_dir "$PCA_RUN_DIR/step_0/eval_global" \
  --calibration_split val \
  --eval_split test \
  --cp_method global \
  --cp_alpha 0.2

# 07. Evaluate the WM-feature branch with pointwise and global calibration.
"$PYTHON_BIN" analyze/eval_pca_kmeans_metrics.py \
  --config_file "$CONFIG" \
  --pca_kmeans_checkpoint "$PCA_RUN_DIR/step_1/pca_kmeans.pt" \
  --output_dir "$PCA_RUN_DIR/step_1/eval_pointwise" \
  --calibration_split val \
  --eval_split test \
  --cp_method pointwise \
  --cp_alpha 0.2
"$PYTHON_BIN" analyze/eval_pca_kmeans_metrics.py \
  --config_file "$CONFIG" \
  --pca_kmeans_checkpoint "$PCA_RUN_DIR/step_1/pca_kmeans.pt" \
  --output_dir "$PCA_RUN_DIR/step_1/eval_global" \
  --calibration_split val \
  --eval_split test \
  --cp_method global \
  --cp_alpha 0.2
