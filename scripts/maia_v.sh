#!/usr/bin/env bash
set -euo pipefail

cd /root/MAIA
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${CUDA_VISIBLE_DEVICES:-0}

latest_file() {
  find "$1" -type f -name "$2" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

make_wm_config() {
  sed -E "s|^([[:space:]]*model_path: ).*|\1$2|" "$1" > "$3"
}

# 01. Train the video world model on the train13 split.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29511} \
  bash scripts/train.sh \
  main.py \
  configs/ltx_model/all/video_wm.yaml \
  runner/ge_trainer_video_wm.py

# 02. Locate the newly trained video WM checkpoint.
WM_CKPT=$(latest_file outputs/all/video_wm diffusion_pytorch_model.safetensors)
WM_RUN_DIR=$(dirname "$(dirname "$WM_CKPT")")

# 03. Point the video-anomaly config at that WM and materialize frozen features.
ANOMALY_CONFIG=$(mktemp /tmp/maia_video_anomaly.XXXXXX.yaml)
make_wm_config configs/ltx_model/all/video_anomaly.yaml "$WM_CKPT" "$ANOMALY_CONFIG"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" data/utils/precompute_video_anomaly_features.py \
  --config_file "$ANOMALY_CONFIG" \
  --output_dir /root/MAIA/cache/all/video_anomaly \
  --splits train val test \
  --batch_size 32 \
  --num_workers 4

# 04. Train the video anomaly head from the frozen feature cache.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29512} \
  bash scripts/train.sh \
  main.py \
  "$ANOMALY_CONFIG" \
  runner/ge_trainer_video_anomaly.py
ANOMALY_CKPT=$(latest_file outputs/all/video_anomaly anomaly_head.pt)

# 05. Compute video anomaly test metrics.
"$PYTHON_BIN" analyze/eval_video_anomaly_metrics.py \
  --config_file "$ANOMALY_CONFIG" \
  --anomaly_checkpoint "$ANOMALY_CKPT" \
  --eval_split test \
  --cache_split test \
  --window_stride 10 \
  --batch_size 32 \
  --output_dir "${WM_RUN_DIR}/video_anomaly_metrics"

# 06. Render one video anomaly test video for the microwave task.
"$PYTHON_BIN" analyze/eval_video_anomaly_video.py \
  --config_file "$ANOMALY_CONFIG" \
  --checkpoint_path "$WM_CKPT" \
  --anomaly_checkpoint "$ANOMALY_CKPT" \
  --output_dir "${WM_RUN_DIR}/video_anomaly_video" \
  --domain microwave \
  --dataset_split test \
  --dataset_index 27 \
  --window_stride 10 \
  --split_manifest /root/MAIA/data_splits/train13_seed42_val20_test30_train100.json

# 07. Build the novel-split feature cache for video anomaly finetuning.
FINETUNE_CONFIG=$(mktemp /tmp/maia_video_anomaly_finetune.XXXXXX.yaml)
make_wm_config configs/ltx_model/all/video_anomaly_finetune.yaml "$WM_CKPT" "$FINETUNE_CONFIG"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" data/utils/precompute_video_anomaly_features.py \
  --config_file "$FINETUNE_CONFIG" \
  --output_dir /root/MAIA/cache/all/video_anomaly_finetune \
  --splits train val test \
  --batch_size 32 \
  --num_workers 4

# 08. Finetune the anomaly head on novel tasks.
sed -i -E "s|^anomaly_head_checkpoint:.*|anomaly_head_checkpoint: $ANOMALY_CKPT|" "$FINETUNE_CONFIG"
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29513} \
  bash scripts/train.sh \
  main.py \
  "$FINETUNE_CONFIG" \
  runner/ge_trainer_video_anomaly.py
NOVEL_ANOMALY_CKPT=$(latest_file outputs/all/video_anomaly anomaly_head.pt)

# 09. Compute novel test metrics.
"$PYTHON_BIN" analyze/eval_video_anomaly_metrics.py \
  --config_file "$FINETUNE_CONFIG" \
  --anomaly_checkpoint "$NOVEL_ANOMALY_CKPT" \
  --eval_split test \
  --cache_split test \
  --window_stride 10 \
  --batch_size 32 \
  --output_dir "${WM_RUN_DIR}/video_anomaly_novel_metrics"

# 10. Render one novel test video for the pull_drawer task.
"$PYTHON_BIN" analyze/eval_video_anomaly_video.py \
  --config_file "$FINETUNE_CONFIG" \
  --checkpoint_path "$WM_CKPT" \
  --anomaly_checkpoint "$NOVEL_ANOMALY_CKPT" \
  --output_dir "${WM_RUN_DIR}/video_anomaly_novel_video" \
  --domain pull_drawer \
  --dataset_split test \
  --dataset_index 27 \
  --window_stride 10 \
  --split_manifest /root/MAIA/data_splits/novel5_seed42_val20_test30_train10.json
