#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

RUNTIME_DIR=${MODUJO_RUNTIME_DIR:-$ROOT_DIR/.runtime}
mkdir -p \
  "$RUNTIME_DIR/huggingface" \
  "$RUNTIME_DIR/modelscope" \
  "$RUNTIME_DIR/xdg-cache" \
  "$RUNTIME_DIR/tmp" \
  "$RUNTIME_DIR/torch"

export HF_HOME=${MODUJO_HF_HOME:-$RUNTIME_DIR/huggingface}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export MODELSCOPE_CACHE=${MODUJO_MODELSCOPE_CACHE:-$RUNTIME_DIR/modelscope}
export XDG_CACHE_HOME=${MODUJO_XDG_CACHE_HOME:-$RUNTIME_DIR/xdg-cache}
export TMPDIR=${MODUJO_TMPDIR:-$RUNTIME_DIR/tmp}
export TORCH_HOME=${MODUJO_TORCH_HOME:-$RUNTIME_DIR/torch}
export MLFLOW_TRACKING_URI=${MODUJO_MLFLOW_TRACKING_URI:-sqlite:///$ROOT_DIR/mlflow.db}
export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-modujo-1b-pretrain}

MODEL_DIR=${MODUJO_1B_MODEL:-$ROOT_DIR/models/modujo-1b-dense-grouped}
DEFAULT_DATA_FILES=$ROOT_DIR/datasets/pretrain_standard/minimind_pretrain_t2t.jsonl:$ROOT_DIR/datasets/pretrain_standard/amazon_esci_products.jsonl:$ROOT_DIR/datasets/pretrain_standard/wayfair_wands_products.jsonl
DATA_FILE_SPEC=${MODUJO_TRAIN_FILES:-${MODUJO_TRAIN_FILE:-$DEFAULT_DATA_FILES}}
IFS=: read -r -a DATA_FILES <<< "$DATA_FILE_SPEC"
OUTPUT_DIR=${MODUJO_OUTPUT_DIR:-$ROOT_DIR/output/modujo-1b-dense-pretrain}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "1B model not found: $MODEL_DIR" >&2
  echo "Run: python pretrain/scripts/prepare_1b_dense_model.py" >&2
  exit 1
fi
for DATA_FILE in "${DATA_FILES[@]}"; do
  if [[ ! -f "$DATA_FILE" ]]; then
    echo "Training data not found: $DATA_FILE" >&2
    exit 1
  fi
done

python - "$MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)

expected = ["linear_attention"] * 3 + ["qwen_sparse_attention"]
layer_types = config.get("layer_types", [])
if not layer_types or any(t != expected[i % 4] for i, t in enumerate(layer_types)):
    raise SystemExit(f"Expected repeating 3x GDN + 1x dense attention, got {layer_types}")

indexer_fields = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)
if any(config.get(name) is not None for name in indexer_fields):
    raise SystemExit("Dense pretraining requires all QSA indexer fields to be null")
if config.get("experts_impl") != "grouped_mm":
    raise SystemExit("The 1B checkpoint must be created with experts_impl=grouped_mm")
PY

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
PER_DEVICE_BATCH_SIZE=${MODUJO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${MODUJO_GLOBAL_BATCH_SIZE:-32}
WORLD_BATCH_SIZE=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE))
if [[ -n ${MODUJO_GRAD_ACCUM_STEPS:-} ]]; then
  GRAD_ACCUM_STEPS=$MODUJO_GRAD_ACCUM_STEPS
else
  if (( GLOBAL_BATCH_SIZE % WORLD_BATCH_SIZE != 0 )); then
    echo "MODUJO_GLOBAL_BATCH_SIZE ($GLOBAL_BATCH_SIZE) must be divisible by" >&2
    echo "NPROC_PER_NODE * MODUJO_BATCH_SIZE ($WORLD_BATCH_SIZE)" >&2
    exit 1
  fi
  GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / WORLD_BATCH_SIZE))
fi
if (( GRAD_ACCUM_STEPS < 1 )); then
  echo "Gradient accumulation steps must be at least 1" >&2
  exit 1
fi
export NPROC_PER_NODE
export USE_HF=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

DATALOADER_WORKERS=${MODUJO_DATALOADER_WORKERS:-8}
MAX_LENGTH=${MODUJO_MAX_LENGTH:-4096}
PACKING=${MODUJO_PACKING:-false}
if [[ "$PACKING" == "true" ]]; then
  echo "MODUJO_PACKING=true is not supported by Qwen4-Exp attention." >&2
  echo "Transformers marks this architecture as incompatible with FlashAttention," >&2
  echo "which Swift requires for online packing. Use MODUJO_PACKING=false." >&2
  exit 1
fi
DATALOADER_ARGS=(--dataloader_num_workers "$DATALOADER_WORKERS")
if (( DATALOADER_WORKERS > 0 )); then
  DATALOADER_ARGS+=(
    --dataloader_persistent_workers true
    --dataloader_prefetch_factor "${MODUJO_DATALOADER_PREFETCH:-4}"
  )
fi

swift pt \
  --model "$MODEL_DIR" \
  --model_type modujo_qwen4_exp \
  --external_plugins "$ROOT_DIR/common/modujo_swift_plugin.py" \
  --model_kwargs '{"output_router_logits": true}' \
  --experts_impl grouped_mm \
  --dataset "${DATA_FILES[@]}" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --bf16 true \
  --tf32 true \
  --max_length "$MAX_LENGTH" \
  --packing "$PACKING" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --gradient_checkpointing true \
  --ddp_find_unused_parameters true \
  --num_train_epochs "${MODUJO_NUM_TRAIN_EPOCHS:-3}" \
  --learning_rate "${MODUJO_LEARNING_RATE:-3e-4}" \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --warmup_ratio 0.02 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --router_aux_loss_coef 0.001 \
  --optim adamw_torch_fused \
  --optimizer modujo_muon \
  "${DATALOADER_ARGS[@]}" \
  --save_strategy steps \
  --save_steps "${MODUJO_SAVE_STEPS:-100}" \
  --save_total_limit 2 \
  --logging_steps "${MODUJO_LOGGING_STEPS:-10}" \
  --report_to "${MODUJO_REPORT_TO:-wandb}" \
  --dataset_num_proc "${MODUJO_DATASET_NUM_PROC:-8}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
