#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned VAE defaults. Override these with environment variables.
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/vae-paper}
DATASET_NAME=${DATASET_NAME:-}
DATASET_CONFIG=${DATASET_CONFIG:-}
TRAIN_FILE=${TRAIN_FILE:-}
VALIDATION_FILE=${VALIDATION_FILE:-}
RESPONSE_FIELD=${RESPONSE_FIELD:-}
QUESTION_FIELD=${QUESTION_FIELD:-}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-46079}

LATENT_DIM=${LATENT_DIM:-512}
NUM_LATENT_TOKENS=${NUM_LATENT_TOKENS:-4}
LATENT_NOISE_STD=${LATENT_NOISE_STD:-3.0}
TOKEN_SUBSTITUTION_PROB=${TOKEN_SUBSTITUTION_PROB:-0.3}
BETA=${BETA:-1e-5}
ENCODER_TUNING=${ENCODER_TUNING:-full}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
MAX_BLOCK_TOKENS=${MAX_BLOCK_TOKENS:-256}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
PREPROCESSING_WORKERS=${PREPROCESSING_WORKERS:-8}

if [[ -z "${DATASET_NAME}" && -z "${TRAIN_FILE}" ]]; then
  echo "Set DATASET_NAME for a Hugging Face dataset or TRAIN_FILE for local JSON/JSONL/Parquet." >&2
  exit 2
fi
if [[ -n "${DATASET_NAME}" && -n "${TRAIN_FILE}" ]]; then
  echo "Set only one of DATASET_NAME and TRAIN_FILE." >&2
  exit 2
fi

ARGS=(
  --model_name_or_path "${MODEL_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --latent_dim "${LATENT_DIM}"
  --num_latent_tokens "${NUM_LATENT_TOKENS}"
  --latent_noise_std "${LATENT_NOISE_STD}"
  --token_substitution_prob "${TOKEN_SUBSTITUTION_PROB}"
  --beta "${BETA}"
  --encoder_tuning "${ENCODER_TUNING}"
  --block_mode sentence
  --reasoning_extraction think_or_full
  --max_block_tokens "${MAX_BLOCK_TOKENS}"
  --preprocessing_num_workers "${PREPROCESSING_WORKERS}"
  --num_train_epochs "${NUM_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --bf16 true
  --gradient_checkpointing true
  --do_train true
  --do_eval true
  --eval_strategy steps
  --eval_steps 1000
  --save_strategy steps
  --save_steps 1000
  --logging_steps 10
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --report_to wandb
)

if [[ -n "${DATASET_NAME}" ]]; then
  ARGS+=(--dataset_name "${DATASET_NAME}")
else
  ARGS+=(--train_file "${TRAIN_FILE}")
fi
if [[ -n "${DATASET_CONFIG}" ]]; then
  ARGS+=(--dataset_config "${DATASET_CONFIG}")
fi
if [[ -n "${VALIDATION_FILE}" ]]; then
  ARGS+=(--validation_file "${VALIDATION_FILE}")
fi
if [[ -n "${RESPONSE_FIELD}" ]]; then
  ARGS+=(--response_field "${RESPONSE_FIELD}")
fi
if [[ -n "${QUESTION_FIELD}" ]]; then
  ARGS+=(--question_field "${QUESTION_FIELD}")
fi

cd "$(dirname "$0")/../vae"
torchrun \
  --standalone \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  train_vae.py "${ARGS[@]}"
