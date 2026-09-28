#!/usr/bin/env python3
"""Add freshly initialized QSA indexers to a dense Qwen4-Exp checkpoint.

The dense attention projections remain at the same state-dict paths.  Only the
indexer projection and its two norms are new after conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


INDEXER_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indexer-n-heads", type=int, default=4)
    parser.add_argument("--indexer-kv-heads", type=int, default=1)
    parser.add_argument("--indexer-head-dim", type=int, default=64)
    parser.add_argument("--indexer-budget", type=int, default=1024)
    parser.add_argument("--indexer-compress-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output}")

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.source)
    source_values = {name: getattr(config, name, None) for name in INDEXER_FIELDS}
    if any(value is not None for value in source_values.values()):
        raise ValueError(f"Source already has QSA enabled: {source_values}")
    if "qwen_sparse_attention" not in config.layer_types:
        raise ValueError(
            "Source has no dense full-attention positions. Expected the repeating "
            "3x linear_attention + 1x qwen_sparse_attention stage-one layout."
        )

    config.indexer_n_heads = args.indexer_n_heads
    config.indexer_kv_heads = args.indexer_kv_heads
    config.indexer_head_dim = args.indexer_head_dim
    config.indexer_budget = args.indexer_budget
    config.indexer_compress_ratio = args.indexer_compress_ratio
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.source,
        config=config,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    missing = sorted(loading_info["missing_keys"])
    unexpected = sorted(loading_info["unexpected_keys"])
    if unexpected:
        raise RuntimeError(f"Unexpected source weights during QSA conversion: {unexpected}")
    if not missing or any(".indexer." not in name for name in missing):
        raise RuntimeError(f"Only newly introduced indexer weights may be missing, got: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    import accelerate.utils.other as accelerate_other

    accelerate_other.is_deepspeed_available = lambda: False
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(args.source).save_pretrained(args.output)
    summary = {
        "source_checkpoint": str(args.source),
        "qsa_enabled": True,
        "indexer_config": {name: getattr(config, name) for name in INDEXER_FIELDS},
        "new_indexer_weights": missing,
        "unexpected_source_weights": unexpected,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seed": args.seed,
    }
    (args.output / "qsa_conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
