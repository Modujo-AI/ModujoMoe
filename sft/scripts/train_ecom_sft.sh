#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

RUNTIME_DIR=${MODUJO_RUNTIME_DIR:-$ROOT_DIR/.runtime}
mkdir -p "$RUNTIME_DIR/huggingface" "$RUNTIME_DIR/modelscope" "$RUNTIME_DIR/xdg-cache" "$RUNTIME_DIR/tmp" "$RUNTIME_DIR/torch"
export HF_HOME=${MODUJO_HF_HOME:-$RUNTIME_DIR/huggingface}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export MODELSCOPE_CACHE=${MODUJO_MODELSCOPE_CACHE:-$RUNTIME_DIR/modelscope}
export XDG_CACHE_HOME=${MODUJO_XDG_CACHE_HOME:-$RUNTIME_DIR/xdg-cache}
export TMPDIR=${MODUJO_TMPDIR:-$RUNTIME_DIR/tmp}
export TORCH_HOME=${MODUJO_TORCH_HOME:-$RUNTIME_DIR/torch}
export MLFLOW_TRACKING_URI=${MODUJO_MLFLOW_TRACKING_URI:-sqlite:///$ROOT_DIR/mlflow.db}
export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-modujo-1b-ecom-sft}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODUJO_SFT_MODEL:-$ROOT_DIR/models/modujo-1b-pretrain}
DATA=${MODUJO_SFT_DATA:-$ROOT_DIR/datasets/modujo/ecom_sft_en.jsonl}
OUT=${MODUJO_SFT_OUTPUT:-$ROOT_DIR/output/modujo-1b-ecom-sft}

[[ -f "$MODEL/config.json" ]] || { echo "Base model missing: $MODEL" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "SFT data missing: $DATA" >&2; exit 1; }

swift sft \
  --model "$MODEL" \
  --model_type modujo_qwen4_exp \
  --external_plugins "$ROOT_DIR/common/modujo_swift_plugin.py" \
  --model_kwargs '{"output_router_logits": true}' \
  --dataset "$DATA" \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --template qwen3 \
  --tuner_type full \
  --max_length 2048 \
  --packing false \
  --per_device_train_batch_size "${MODUJO_SFT_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${MODUJO_SFT_GRAD_ACCUM:-4}" \
  --num_train_epochs "${MODUJO_SFT_EPOCHS:-2}" \
  --learning_rate "${MODUJO_SFT_LR:-1e-5}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --router_aux_loss_coef 0.001 \
  --optim adamw_torch_fused \
  --optimizer modujo_muon \
  --bf16 true \
  --gradient_checkpointing true \
  --logging_steps 10 \
  --save_steps 250 \
  --save_total_limit 2 \
  --output_dir "$OUT" \
  --report_to "${MODUJO_REPORT_TO:-wandb}" \
  "$@"
