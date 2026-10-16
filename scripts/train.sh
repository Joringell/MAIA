#!/usr/bin/bash


script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

runner_class_path=${3}
echo $runner_class_path

if [ -z $WORLD_SIZE ]; then
if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
  IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
  NGPU=${#DEV_ARR[@]}
else
  NGPU=`nvidia-smi --list-gpus | wc -l`
fi
echo "Training on 1 Nodes, $NGPU GPUs"
torchrun --nnodes=1 \
    --nproc_per_node=$NGPU \
    --node_rank=0 \
    --master-addr ${MASTER_ADDR:-127.0.0.1} \
    --master-port ${MASTER_PORT:-29511} \
    $script_path \
    --config_file $config_path \
    --runner_class_path $runner_class_path
else
echo "Training on $WORLD_SIZE Nodes, 8 GPU per Node"
if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
  IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
  NGPU=${#DEV_ARR[@]}
else
  NGPU=`nvidia-smi --list-gpus | wc -l`
fi
torchrun --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NGPU \
    --node_rank=$RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path \
    --runner_class_path $runner_class_path
fi
