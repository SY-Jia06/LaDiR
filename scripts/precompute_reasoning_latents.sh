#!/usr/bin/env bash
set -euo pipefail

INPUT=${INPUT:-data/train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-data/reasoner_train}
VAE_CHECKPOINT=${VAE_CHECKPOINT:-checkpoints/vae/model.safetensors}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}
BATCH_SIZE=${BATCH_SIZE:-64}
LATENT_MODE=${LATENT_MODE:-sample}

python scripts/precompute_reasoning_latents.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --model-name-or-path "$MODEL_NAME_OR_PATH" \
  --batch-size "$BATCH_SIZE" \
  --latent-mode "$LATENT_MODE" \
  "$@"
