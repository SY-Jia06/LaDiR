#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS" >&2
  exit 2
fi
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / NUM_GPUS))}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/vae_paper}"
REPORT_TO="${REPORT_TO:-wandb}"

mkdir -p "$OUTPUT_DIR" logs

# Paper recipe (Table 13): dz=512, four latent tokens per sentence block,
# beta=1e-5, LR=2e-5, global batch=128, two epochs.  Section 3.2.1 adds
# latent Gaussian noise k=3 and input-token substitution p=0.3.
torchrun --standalone --nproc_per_node "$NUM_GPUS" \
  vae/train_vae.py \
  --run_name vae_paper_reproduction \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --train_file data/vae_train.jsonl \
  --val_file data/vae_val.jsonl \
  --input_type cot_only \
  --answer_prefix "The answer is" \
  --require_answer_prefix true \
  --fixed_mem_size 4 \
  --latent_dim 512 \
  --beta 1e-5 \
  --latent_noise_std 3.0 \
  --token_substitution_prob 0.3 \
  --paper_block_mode true \
  --use_lora false \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --max_steps -1 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 1000 \
  --weight_decay 0.03 \
  --bf16 true \
  --save_strategy epoch \
  --eval_strategy no \
  --logging_steps 10 \
  --report_to "$REPORT_TO" \
  --remove_unused_columns false \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 8 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config '{"backward_prefetch":"backward_pre","forward_prefetch":true,"cpu_ram_efficient_loading":true,"sync_module_states":true,"transformer_layer_cls_to_wrap":["LlamaDecoderLayer"],"use_orig_params":true,"activation_checkpointing":true}'
