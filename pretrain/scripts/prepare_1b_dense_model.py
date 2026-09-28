#!/usr/bin/env python3
"""Build the roughly 1B dense-attention source model for Dense -> QSA training."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT_DIR))
from common.qwen4_exp_dense_compat import enable_dense_qwen4_exp_attention


DEFAULT_SOURCE = "Alexhu1999/Modujo-9B-A1B"
DEFAULT_OUTPUT = ROOT_DIR / "models/modujo-1b-dense-grouped"
LAYER_PATTERN = ("linear_attention",) * 3 + ("qwen_sparse_attention",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_config(source: str, seed: int):
    config = AutoConfig.from_pretrained(source)
    config.hidden_size = 768
    config.num_hidden_layers = 36
    config.num_attention_heads = 12
    config.num_key_value_heads = 2
    config.head_dim = 64
    config.linear_key_head_dim = 64
    config.linear_value_head_dim = 64
    config.linear_num_key_heads = 6
    config.linear_num_value_heads = 12
    config.num_experts = 8
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 512
    config.shared_expert_intermediate_size = 512
    config.ple_embed_dim = 768
    config.hc_lowrank = 96
    config.heads_per_ngram = 6
    config.layer_types = [LAYER_PATTERN[i % 4] for i in range(config.num_hidden_layers)]
    config.indexer_n_heads = None
    config.indexer_kv_heads = None
    config.indexer_head_dim = None
    config.indexer_budget = None
    config.indexer_compress_ratio = None
    config.experts_impl = "grouped_mm"
    config.seed = seed
    config.dtype = "bfloat16"
    return config


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output}")
    set_seed(args.seed)
    enable_dense_qwen4_exp_attention()
    config = make_config(args.source, args.seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 900_000_000 <= parameter_count <= 1_100_000_000:
        raise RuntimeError(f"Expected roughly 1B parameters, got {parameter_count:,}")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(args.source).save_pretrained(args.output)
    summary = {
        "source_model": args.source,
        "seed": args.seed,
        "total_parameters": parameter_count,
        "total_parameters_billions": parameter_count / 1e9,
        "experts_impl": config.experts_impl,
        "qsa_enabled": False,
        "dtype": "bfloat16",
    }
    (args.output / "parameter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
