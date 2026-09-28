#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODUJO_QSA_DISTILLED_MODEL:-$ROOT_DIR/output/modujo-1b-qsa-indexer-distilled}
DATA_FILE=${MODUJO_QSA_TRAIN_FILE:-$ROOT_DIR/datasets/modujo/train_bilingual_packed_8192.jsonl}
OUTPUT_DIR=${MODUJO_QSA_OUTPUT_DIR:-$ROOT_DIR/output/modujo-1b-qsa-continued}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

torchrun --standalone --nproc_per_node 1 \
  "$ROOT_DIR/pretrain/train_transformers.py" \
  --model_name_or_path "$MODEL_DIR" \
  --train_file "$DATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --dtype bfloat16 \
  --experts_impl grouped_mm \
  --bf16 true \
  --tf32 true \
  --max_length "${MODUJO_QSA_MAX_LENGTH:-8192}" \
  --per_device_train_batch_size "${MODUJO_QSA_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${MODUJO_QSA_GRAD_ACCUM_STEPS:-8}" \
  --gradient_checkpointing true \
  --ddp_find_unused_parameters true \
  --num_train_epochs "${MODUJO_QSA_EPOCHS:-1}" \
  --learning_rate "${MODUJO_QSA_BACKBONE_LR:-3e-5}" \
  --warmup_steps "${MODUJO_QSA_WARMUP_STEPS:-100}" \
  --weight_decay 0.1 \
  --router_aux_loss_coef 0.001 \
  --use_muon true \
  --save_strategy steps \
  --save_steps "${MODUJO_SAVE_STEPS:-100}" \
  --save_total_limit 2 \
  --logging_steps 10 \
  --report_to "${MODUJO_REPORT_TO:-none}" \
  --remove_unused_columns false \
  "$@"
