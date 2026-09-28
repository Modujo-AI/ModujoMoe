# ModujoMoe

An end-to-end research workspace for the Modujo family of mixture-of-experts
language models. The current maintained base model is
[`Modujo-1B-A0.75B`](https://huggingface.co/Alexhu1999/Modujo-9B-A1B/tree/main/pretrain/Modujo-1B-A0.75B),
built with PyTorch, Hugging Face Transformers, and ModelScope ms-swift.

## Training stages

| Stage | Directory | Status | Input | Output |
| --- | --- | --- | --- | --- |
| 1 | [`pretrain/`](pretrain/) | Active | General and domain text | Base model |
| 2 | [`sft/`](sft/) | Active | Instruction conversations | Assistant model |
| 3 | [`rl/`](rl/) | Planned | Prompts, rewards and rollouts | Aligned policy |
| 4 | [`opd/`](opd/) | Planned | To be defined | Final model |

Shared ms-swift model registration lives in [`common/`](common/). Each stage
owns its data preparation, configuration, scripts, documentation, evaluation,
and output conventions. Generated datasets, logs, caches, and checkpoints are
excluded from Git.

## Current model paths

The repository keeps three experimental directions separate:

- **SFT** improves instruction following and response quality.
- **QSA** explores sparse attention for efficient long-context processing.
- **Looped Transformer** explores repeated computation with shared layers.

Generated datasets, checkpoints, caches, and experiment logs stay outside Git.
Published model weights live on Hugging Face; this repository contains the
reproducible preparation, training, and evaluation code.

## Quick start

Create the 1B dense-attention initialization and run a one-step pipeline test:

```bash
python -m pip install -r requirements.txt
python pretrain/scripts/prepare_1b_dense_model.py
bash pretrain/scripts/test_1b_train.sh
```

See [`pretrain/README.md`](pretrain/README.md) for dense pretraining, QSA, and
Looped Transformer workflows, and [`sft/README.md`](sft/README.md) for
instruction tuning. The public training report is available in
[`docs/pretraining-report.md`](docs/pretraining-report.md).

Regenerate the PDF report with:

```bash
python -m pip install -r requirements-docs.txt
python tools/build_pretraining_report_pdf.py
```
