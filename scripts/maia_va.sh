#!/usr/bin/env bash
set -euo pipefail

cd /root/MAIA
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${CUDA_VISIBLE_DEVICES:-0}

latest_file() { find "$1" -type f -name "$2" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-; }
make_wm_config() { sed -E "s|^([[:space:]]*model_path: ).*|\1$2|" "$1" > "$3"; }

# 01. Train the action-conditioned video world model.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29521} \
  bash scripts/train.sh \
  main.py \
  configs/ltx_model/all/video_action_wm.yaml \
  runner/ge_trainer_video_action_wm.py
WM_CKPT=$(latest_file outputs/all/video_action_wm diffusion_pytorch_model.safetensors)
WM_RUN_DIR=$(dirname "$(dirname "$WM_CKPT")")

# 02. Point the action-anomaly config at the trained action WM.
ANOMALY_CONFIG=$(mktemp /tmp/maia_video_action_anomaly.XXXXXX.yaml)
make_wm_config configs/ltx_model/all/video_action_anomaly.yaml "$WM_CKPT" "$ANOMALY_CONFIG"

# 03. Extract frozen action-WM features for train/validation/test.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" data/utils/precompute_video_action_anomaly_features.py \
  --config_file "$ANOMALY_CONFIG" \
  --output_dir /root/MAIA/cache/all/video_action_anomaly \
  --splits train val test \
  --batch_size 32 \
  --num_workers 4

# 04. Train the action anomaly head.
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29522} \
  bash scripts/train.sh \
  main.py \
  "$ANOMALY_CONFIG" \
  runner/ge_trainer_video_action_anomaly.py
ANOMALY_CKPT=$(latest_file outputs/all/video_action_anomaly anomaly_head.pt)

# 05. Evaluate the trained action-anomaly model on the training-task test split.
"$PYTHON_BIN" analyze/eval_action_anomaly_metrics.py \
  --config_file "$ANOMALY_CONFIG" \
  --checkpoint_path "$WM_CKPT" \
  --anomaly_checkpoint "$ANOMALY_CKPT" \
  --eval_split test \
  --cache_split test \
  --window_stride 10 \
  --split_manifest /root/MAIA/data_splits/train13_seed42_val20_test30_train100.json \
  --batch_size 32 \
  --output_dir "${WM_RUN_DIR}/video_action_anomaly_metrics"

# 06. Render one test episode from the training-task test split.
"$PYTHON_BIN" analyze/eval_action_anomaly_video.py \
  --config_file "$ANOMALY_CONFIG" \
  --checkpoint_path "$WM_CKPT" \
  --anomaly_checkpoint "$ANOMALY_CKPT" \
  --output_dir "${WM_RUN_DIR}/video_action_anomaly_video" \
  --domain microwave \
  --dataset_split test \
  --dataset_index 27 \
  --window_stride 10 \
  --split_manifest /root/MAIA/data_splits/train13_seed42_val20_test30_train100.json

# 07. Train the joint action-anomaly model on the novel split.
JOINT_CONFIG=$(mktemp /tmp/maia_video_action_anomaly_joint.XXXXXX.yaml)
sed -E "s|^([[:space:]]*model_path: ).*|\1$WM_CKPT|; s|^anomaly_head_checkpoint:.*|anomaly_head_checkpoint: $ANOMALY_CKPT|" \
  configs/ltx_model/all/video_action_anomaly_joint.yaml > "$JOINT_CONFIG"
CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT=${MASTER_PORT:-29523} \
  bash scripts/train.sh \
  main.py \
  "$JOINT_CONFIG" \
  runner/ge_trainer_video_action_anomaly_joint.py
JOINT_CKPT=$(latest_file outputs/all/video_action_anomaly diffusion_pytorch_model.safetensors)
JOINT_HEAD=$(latest_file outputs/all/video_action_anomaly anomaly_head.pt)

# 08. Evaluate the joint model on the novel test split.
"$PYTHON_BIN" analyze/eval_action_anomaly_metrics.py \
  --config_file "$JOINT_CONFIG" \
  --checkpoint_path "$JOINT_CKPT" \
  --anomaly_checkpoint "$JOINT_HEAD" \
  --eval_split test \
  --cache_split test \
  --window_stride 10 \
  --split_manifest /root/MAIA/data_splits/novel5_seed42_val20_test30_train10.json \
  --disable_cache \
  --batch_size 32 \
  --output_dir "${WM_RUN_DIR}/video_action_anomaly_joint_metrics"

# 09. Render one novel test video for the pull_drawer task.
"$PYTHON_BIN" analyze/eval_action_anomaly_video.py \
  --config_file "$JOINT_CONFIG" \
  --checkpoint_path "$JOINT_CKPT" \
  --anomaly_checkpoint "$JOINT_HEAD" \
  --output_dir "${WM_RUN_DIR}/video_action_anomaly_joint_video" \
  --domain pull_drawer \
  --dataset_split test \
  --split_manifest /root/MAIA/data_splits/novel5_seed42_val20_test30_train10.json
