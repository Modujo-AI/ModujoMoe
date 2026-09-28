# Modujo 1B pretraining

This directory contains one maintained training path: the roughly 1B
Qwen4-Exp MoE model. It uses a repeating `3x GDN + 1x dense attention`
backbone for ordinary pretraining. QSA is introduced only in later stages.

## 1. Build the dense 1B checkpoint

```bash
python pretrain/scripts/prepare_1b_dense_model.py
```

The default output is `models/modujo-1b-dense-grouped`. Dense attention uses
the Qwen4-Exp `qwen_sparse_attention` module with all `indexer_*` fields unset;
`common/qwen4_exp_dense_compat.py` supplies this pre-QSA execution mode.

## 2. Dense pretraining

Before a full run, verify the complete 1B pipeline with one sample and one
optimizer step:

```bash
bash pretrain/scripts/test_1b_train.sh
```

```bash
bash pretrain/scripts/train_1b_dense.sh
```

The launcher directly invokes `swift pt`; it does not wrap another shell
launcher. The standard `--optim` remains a valid Transformers value, while
Swift's `--optimizer modujo_muon` selects the plugin's Muon + AdamW callback.
It defaults to one GPU with a per-device batch size of 4 and automatically
chooses gradient accumulation 8 to keep global batch size 32. It also enables BF16, Muon for
eligible matrix parameters, AdamW for embeddings/router/norm/bias parameters,
and router auxiliary loss coefficient `0.001` with router logits enabled.

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MODUJO_BATCH_SIZE=4 \
MODUJO_GLOBAL_BATCH_SIZE=32 \
MODUJO_MAX_LENGTH=4096 \
MODUJO_PACKING=false \
MODUJO_TRAIN_FILE=/path/to/train.jsonl \
MODUJO_OUTPUT_DIR=/path/to/output \
  bash pretrain/scripts/train_1b_dense.sh
```

Pass multiple un-packed JSONL files with a colon-separated list. Swift merges
the datasets without creating another physical copy:

```bash
MODUJO_TRAIN_FILES=/path/to/general.jsonl:/path/to/ecommerce.jsonl \
  bash pretrain/scripts/train_1b_dense.sh
```

Packing is disabled. Swift requires a FlashAttention backend for online
packing, while Transformers marks Qwen4-Exp attention as not supporting
FlashAttention. The launcher rejects `MODUJO_PACKING=true` instead of forcing
an unsafe backend. Training therefore uses dynamic padding and keeps one
record, including its EOS boundary, per sequence.

## 3. Add QSA indexers

```bash
python pretrain/scripts/convert_dense_to_qsa.py \
  --source models/Modujo-1B-A0.75B \
  --output models/modujo-1b-qsa-init
```

## 4. Distill the QSA indexers

```bash
MODUJO_QSA_TRAIN_FILE=/path/to/packed_8192.jsonl \
  bash pretrain/scripts/qsa_indexer_distill.sh
```

## 5. Continue sparse QSA training

```bash
MODUJO_QSA_TRAIN_FILE=/path/to/packed_8192.jsonl \
  bash pretrain/scripts/qsa_continue_train.sh
```

Shared implementation files:

- `pretrain/train_transformers.py`: QSA distillation and sparse training.
- `pretrain/muon.py`: Muon plus auxiliary AdamW.
- `common/modujo_swift_plugin.py`: Swift model and optimizer registration.
- `common/qwen4_exp_dense_compat.py`: pre-QSA dense-attention compatibility.

## Looped continuation experiment

`pretrain/scripts/looped_continue_train.sh` starts from the dense stage-one
checkpoint by default and revisits physical layers 16–19 once, sharing their
weights. The four-layer group contains three GDN layers and one dense attention
layer. The experiment uses 4 GPUs, BF16, 2048-token samples, and one epoch of
the full packed pretraining corpus (504,514 samples). It writes to a separate
output directory. Set `MODUJO_LOOP_MODEL` to continue from a looped checkpoint,
and `MODUJO_REPORT_TO=wandb` to log to W&B.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 \
  bash pretrain/scripts/looped_continue_train.sh
```

The saved config records `modujo_loop_start`, `modujo_loop_end`, and
`modujo_loop_repeats`. Loading those weights with ordinary
`AutoModelForCausalLM.from_pretrained` does not install the loop; call
`common.qwen4_exp_loop.enable_qwen4_exp_loop` using those config values before
inference or resumed training. Generation must use `use_cache=False` because
the current Qwen4-Exp cache is indexed by physical layer rather than loop visit.
