#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
CONFIG=${CONFIG:-configs/reasoner_stage1.yaml}

DENOM=$((NUM_GPUS * PER_DEVICE_BATCH))
if (( GLOBAL_BATCH_SIZE % DENOM != 0 )); then
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS*PER_DEVICE_BATCH=$DENOM" >&2
  exit 2
fi
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / DENOM))

torchrun --standalone --nproc_per_node "$NUM_GPUS" train.py "$CONFIG" \
  trainer.per_device_train_batch_size="$PER_DEVICE_BATCH" \
  trainer.gradient_accumulation_steps="$GRAD_ACCUM" \
  "$@"
