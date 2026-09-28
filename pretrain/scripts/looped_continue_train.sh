#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

MODEL=${MODUJO_LOOP_MODEL:-$ROOT_DIR/models/modujo-1b-pretrain}
DATA=${MODUJO_LOOP_DATA:-$ROOT_DIR/datasets/modujo/train_mixture_1b_packed_2048.jsonl}
OUT=${MODUJO_LOOP_OUTPUT:-$ROOT_DIR/output/modujo-1b-looped-continue-full-from-78900}
NPROC=${NPROC_PER_NODE:-4}

[[ -f "$MODEL/config.json" ]] || { echo "Model missing: $MODEL" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "Data missing: $DATA" >&2; exit 1; }

RUNTIME_DIR=${MODUJO_RUNTIME_DIR:-$ROOT_DIR/.runtime}
mkdir -p "$RUNTIME_DIR/huggingface" "$RUNTIME_DIR/xdg-cache" "$RUNTIME_DIR/tmp" "$OUT"
export HF_HOME=${MODUJO_HF_HOME:-$RUNTIME_DIR/huggingface}
export XDG_CACHE_HOME=${MODUJO_XDG_CACHE_HOME:-$RUNTIME_DIR/xdg-cache}
export TMPDIR=${MODUJO_TMPDIR:-$RUNTIME_DIR/tmp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

torchrun --standalone --nproc_per_node "$NPROC" \
  "$ROOT_DIR/pretrain/train_transformers.py" \
  --model_name_or_path "$MODEL" \
  --train_file "$DATA" \
  --output_dir "$OUT" \
  --dtype bfloat16 \
  --bf16 true \
  --tf32 true \
  --max_length "${MODUJO_LOOP_MAX_LENGTH:-2048}" \
  --loop_start "${MODUJO_LOOP_START:-16}" \
  --loop_end "${MODUJO_LOOP_END:-19}" \
  --loop_repeats "${MODUJO_LOOP_REPEATS:-2}" \
  --per_device_train_batch_size "${MODUJO_LOOP_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --ddp_find_unused_parameters false \
  --ddp_static_graph true \
  --learning_rate "${MODUJO_LOOP_LR:-3e-5}" \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --warmup_steps "${MODUJO_LOOP_WARMUP_STEPS:-50}" \
  --weight_decay 0.1 \
  --num_train_epochs "${MODUJO_LOOP_EPOCHS:-1}" \
  --max_steps "${MODUJO_LOOP_MAX_STEPS:--1}" \
  --save_strategy steps \
  --save_steps "${MODUJO_LOOP_SAVE_STEPS:-1000}" \
  --save_total_limit 2 \
  --logging_steps 10 \
  --report_to "${MODUJO_REPORT_TO:-none}" \
  --remove_unused_columns false \
  --preprocessing_num_workers 4 \
  "$@"
