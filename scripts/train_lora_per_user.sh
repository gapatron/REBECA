#!/usr/bin/env bash
set -euo pipefail

META="/home/gabriel/Projects/REBECA/data/flickr/processed/lora/metadata.jsonl"
DATA_DIR="/home/gabriel/Projects/REBECA/data/flickr/processed/lora"
BASE_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"

for USER in $(seq 0 209); do
  # count captions like: "<u123> ..."
  N=$(grep -c "\"caption\": \"<u${USER}>" "$META" || true)
  N=${N//[[:space:]]/}

  echo "User u${USER} has ${N} examples"

  # optionally skip if none
  if [ "${N}" -eq 0 ]; then
    echo "  -> skipping u${USER} (no data)"
    continue
  fi

  # common args (keep TE frozen for per-user few-shot)
  COMMON_ARGS=(
    --pretrained_model_name_or_path "$BASE_MODEL"
    --train_data_dir "$DATA_DIR"
    --image_column image
    --caption_column caption
    --user_id "${USER}"
    --resolution 512
    --use_peft
    --gradient_checkpointing
    --mixed_precision fp16
    --allow_tf32
    --enable_xformers_memory_efficient_attention
    --train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 1e-4
    --lr_scheduler cosine
    --max_grad_norm 1.0
    --validation_prompt "<u${USER}> photo"
    --num_validation_images 4
    --validation_epochs 25
    --checkpointing_steps 2000
    --checkpoints_total_limit 6
    --output_dir "outputs/lora_users/u${USER}"
    --seed "${USER}"
  )

  if [ "$N" -lt 8 ]; then
    accelerate launch scripts/train_lora.py \
      "${COMMON_ARGS[@]}" \
      --lora_r 8  --lora_alpha 8  --lora_dropout 0.10 \
      --max_train_steps 1200 \
      --lr_warmup_steps 100

  elif [ "$N" -le 24 ]; then
    accelerate launch scripts/train_lora.py \
      "${COMMON_ARGS[@]}" \
      --lora_r 16 --lora_alpha 16 --lora_dropout 0.07 \
      --max_train_steps 1500 \
      --lr_warmup_steps 120

  elif [ "$N" -le 60 ]; then
    accelerate launch scripts/train_lora.py \
      "${COMMON_ARGS[@]}" \
      --lora_r 32 --lora_alpha 32 --lora_dropout 0.05 \
      --max_train_steps 1700 \
      --lr_warmup_steps 150

  elif [ "$N" -le 120 ]; then
    accelerate launch scripts/train_lora.py \
      "${COMMON_ARGS[@]}" \
      --lora_r 64 --lora_alpha 64 --lora_dropout 0.05 \
      --max_train_steps 2000 \
      --lr_warmup_steps 200

  else
    accelerate launch scripts/train_lora.py \
      "${COMMON_ARGS[@]}" \
      --lora_r 32 --lora_alpha 32 --lora_dropout 0.05 \
      --max_train_steps 2500 \
      --lr_warmup_steps 200
  fi
done
