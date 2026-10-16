#!/usr/bin/env bash
set -euo pipefail

cd /root/MAIA
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${CUDA_VISIBLE_DEVICES:-0}
LSTM_CONFIG=configs/baselines/all/safe_pi05_lstm.yaml
MLP_CONFIG=configs/baselines/all/safe_pi05_mlp.yaml
LSTM_OUTPUT_ROOT=/root/MAIA/outputs/all/safe_pi05_lstm
MLP_OUTPUT_ROOT=/root/MAIA/outputs/all/safe_pi05_mlp

latest_file() { find "$1" -type f -name "$2" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-; }

# 01. Extract pi05 hidden-state features for train/validation/test.
cd /root/openpi
CUDA_VISIBLE_DEVICES="$GPU" UV_CACHE_DIR=/tmp/uv-cache uv run python \
  /root/MAIA/data/utils/precompute_pi05_feature_cache.py \
  --split_manifest /root/MAIA/data_splits/train13_seed42_val20_test30_train100.json \
  --output_dir /root/MAIA/cache/all/pi05 \
  --data_root /root/CRIA \
  --splits train val test \
  --openpi_root /root/openpi \
  --config pi05_part_gripper \
  --checkpoint_dir /root/openpi/checkpoints/pi05_part_gripper/pi05_part_gripper/29999 \
  --query_stride 10 \
  --num_steps 10 \
  --shard_size 128 \
  --feature_dtype float16

# 02. Train the SAFE LSTM baseline.
cd /root/MAIA
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29801} \
  bash scripts/train.sh \
  main.py \
  "$LSTM_CONFIG" \
  runner/safe_pi05_trainer.py
LSTM_CKPT=$(latest_file "$LSTM_OUTPUT_ROOT" safe_pi05.pt)
LSTM_RUN_DIR=$(dirname "$(dirname "$LSTM_CKPT")")

# 03. Evaluate SAFE LSTM on the test split with validation calibration.
"$PYTHON_BIN" analyze/eval_safe_pi05_metrics.py \
  --config_file "$LSTM_CONFIG" \
  --safe_checkpoint "$LSTM_CKPT" \
  --feature_cache_dir /root/MAIA/cache/all/pi05 \
  --calibration_split val \
  --eval_split test \
  --target_source frame \
  --cp_variants both \
  --single_band_normal_filter success_episodes \
  --single_band_method pointwise \
  --cp_alpha 0.1 \
  --output_dir "$LSTM_RUN_DIR/eval" \
  --batch_size 64 \
  --num_workers 4

# 04. Build one pi05 cache for the novel-task split; LSTM and MLP share it.
cd /root/openpi
CUDA_VISIBLE_DEVICES="$GPU" UV_CACHE_DIR=/tmp/uv-cache uv run python \
  /root/MAIA/data/utils/precompute_pi05_feature_cache.py \
  --split_manifest /root/MAIA/data_splits/novel5_seed42_val20_test30_train10.json \
  --output_dir /root/MAIA/cache/all/pi05_novel \
  --data_root /root/CRIA \
  --splits train val test \
  --openpi_root /root/openpi \
  --config pi05_part_gripper \
  --checkpoint_dir /root/openpi/checkpoints/pi05_part_gripper/pi05_part_gripper/29999 \
  --query_stride 10 \
  --num_steps 10 \
  --shard_size 128 \
  --feature_dtype float16

# 05. Test the trained SAFE LSTM on novel tasks with the training calibration.
cd /root/MAIA
"$PYTHON_BIN" analyze/eval_safe_pi05_metrics.py \
  --config_file "$LSTM_CONFIG" \
  --safe_checkpoint "$LSTM_CKPT" \
  --feature_cache_dir /root/MAIA/cache/all/pi05_novel \
  --calibration_split val \
  --eval_split test \
  --single_band_cp_metrics "$LSTM_RUN_DIR/eval/single_band/safe_pi05_cp_metrics.json" \
  --cp_variants single_band \
  --target_source frame \
  --cp_alpha 0.2 \
  --cp_alarm_margin 0.01 \
  --output_dir "$LSTM_RUN_DIR/eval_novel" \
  --batch_size 64 \
  --num_workers 4

# 06. Train the SAFE MLP baseline on the same offline task split.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29802} \
  bash scripts/train.sh \
  main.py \
  "$MLP_CONFIG" \
  runner/safe_pi05_trainer.py
MLP_CKPT=$(latest_file "$MLP_OUTPUT_ROOT" safe_pi05.pt)
MLP_RUN_DIR=$(dirname "$(dirname "$MLP_CKPT")")

# 07. Evaluate SAFE MLP on the test split with validation calibration.
"$PYTHON_BIN" analyze/eval_safe_pi05_metrics.py \
  --config_file "$MLP_CONFIG" \
  --safe_checkpoint "$MLP_CKPT" \
  --feature_cache_dir /root/MAIA/cache/all/pi05 \
  --calibration_split val \
  --eval_split test \
  --target_source frame \
  --cp_variants both \
  --single_band_normal_filter success_episodes \
  --single_band_method pointwise \
  --cp_alpha 0.1 \
  --output_dir "$MLP_RUN_DIR/eval" \
  --batch_size 64 \
  --num_workers 4

# 08. Test the trained SAFE MLP on novel tasks with the training calibration.
"$PYTHON_BIN" analyze/eval_safe_pi05_metrics.py \
  --config_file "$MLP_CONFIG" \
  --safe_checkpoint "$MLP_CKPT" \
  --feature_cache_dir /root/MAIA/cache/all/pi05_novel \
  --calibration_split val \
  --eval_split test \
  --single_band_cp_metrics "$MLP_RUN_DIR/eval/single_band/safe_pi05_cp_metrics.json" \
  --cp_variants single_band \
  --target_source frame \
  --cp_alpha 0.2 \
  --cp_alarm_margin 0.01 \
  --output_dir "$MLP_RUN_DIR/eval_novel" \
  --batch_size 64 \
  --num_workers 4
