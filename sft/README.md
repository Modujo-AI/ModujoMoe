# Supervised fine-tuning (SFT)

This stage will start from the selected pretraining checkpoint. Place SFT data
preparation under `data/`, Swift commands under `scripts/`, and stage-specific
configuration under `configs/`. Do not mix instruction data into `pretrain/`.

## Full SFT mixture

`data/prepare_full_sft.py` normalizes the following public datasets into
`datasets/modujo/sft_full/`. Its manifest records the actual usable row counts.

| Source | Public dataset | Usable rows |
| --- | --- | ---: |
| MiniMind SFT | `jingyaogong/minimind_dataset`, `sft_t2t.jsonl` | 5,108,713 |
| OpenGPT | `openchat/cogstack-opengpt-sharegpt`, `opengpt.jsonl` | 31,511 |
| Claude 3.5 Sonnet | `Data-Agora/magpie_general_claude3.5_sonnet_10000` | 10,000 |
| E-commerce | `rescommons/Ecom-Chatbot-Finetuning-Dataset`, all five splits | 40,098 |

OpenGPT is predominantly medical and NHS question answering. MiniMind contains
1,182,239 conversations with reasoning content and 385,102 with tool definitions.
The normalizer keeps those as Qwen style reasoning text and Swift compatible
tool calls. Rows without a final assistant response are excluded because they
have no SFT target (719 MiniMind and 21 OpenGPT rows).

`scripts/train_full_sft.sh` starts from the finished e-commerce SFT checkpoint
and trains the whole mixture for one epoch on four GPUs. The model's custom
attention does not support the Flash Attention implementation Swift requires
for sequence packing, so the script uses unpacked examples.

Model weights are intentionally not stored in Git. By default,
`train_ecom_sft.sh` reads the pretrained base model from
`models/modujo-1b-pretrain`, and `train_full_sft.sh` reads the e-commerce-tuned
model from `models/modujo-1b-ecom-sft`. Override either location with
`MODUJO_SFT_MODEL`.
