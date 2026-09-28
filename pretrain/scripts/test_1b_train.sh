#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# This smoke test launches Swift itself. Do not inherit an outer torchrun or
# Accelerate process group, otherwise Swift treats the run as distributed and
# rejects a model loaded with device_map=auto.
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export MODUJO_BATCH_SIZE=1
export MODUJO_GLOBAL_BATCH_SIZE=8
export MODUJO_GRAD_ACCUM_STEPS=1
export MODUJO_TRAIN_FILE=$ROOT_DIR/pretrain/data/test_1b_sample.jsonl
export MODUJO_OUTPUT_DIR=${MODUJO_TEST_OUTPUT_DIR:-$ROOT_DIR/output/modujo-1b-smoke-test}
export MODUJO_DATALOADER_WORKERS=0
export MODUJO_DATASET_NUM_PROC=1
export MODUJO_REPORT_TO=wandb

exec bash "$ROOT_DIR/pretrain/scripts/train_1b_dense.sh" \
  --max_steps 1 \
  --max_length "${MODUJO_TEST_MAX_LENGTH:-256}" \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_persistent_workers false \
  "$@"
