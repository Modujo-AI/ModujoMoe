#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODUJO_QSA_MODEL:-$ROOT_DIR/models/modujo-1b-qsa-init}
DATA_FILE=${MODUJO_QSA_TRAIN_FILE:-$ROOT_DIR/datasets/modujo/train_bilingual_packed_2048.jsonl}
OUTPUT_DIR=${MODUJO_QSA_DISTILL_OUTPUT:-$ROOT_DIR/output/modujo-1b-qsa-indexer-distilled}

NPROC_PER_NODE=${NPROC_PER_NODE:-1} torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE:-1}" \
  "$ROOT_DIR/pretrain/train_transformers.py" \
  --model_name_or_path "$MODEL_DIR" \
  --train_file "$DATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --dtype bfloat16 \
  --bf16 true \
  --attn_implementation eager \
  --qsa_indexer_distill_weight 1.0 \
  --qsa_lm_loss_weight 0.0 \
  --max_length "${MODUJO_QSA_MAX_LENGTH:-8192}" \
  --per_device_train_batch_size "${MODUJO_QSA_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${MODUJO_QSA_GRAD_ACCUM_STEPS:-8}" \
  --gradient_checkpointing false \
  --ddp_find_unused_parameters false \
  --ddp_static_graph true \
  --learning_rate "${MODUJO_QSA_INDEXER_LR:-1e-3}" \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --warmup_steps "${MODUJO_QSA_WARMUP_STEPS:-100}" \
  --weight_decay 0.0 \
  --num_train_epochs "${MODUJO_QSA_DISTILL_EPOCHS:-1}" \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --report_to "${MODUJO_REPORT_TO:-none}" \
  --remove_unused_columns false \
  "$@"
