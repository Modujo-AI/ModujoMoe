"""Compatibility support for pre-QSA dense Qwen4-Exp checkpoints.

Transformers 5.16 accepts ``None`` QSA fields in the configuration but still
constructs and calls the QSA indexer.  A pre-QSA checkpoint needs the same
attention module and parameter names, with the indexer absent and the ordinary
causal mask passed directly to the core attention operation.
"""

from __future__ import annotations

from typing import Any

import torch


_PATCHED = False


def qsa_is_enabled(config: Any) -> bool:
    fields = (
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
    )
    values = [getattr(config, name, None) for name in fields]
    if any(value is None for value in values) and any(value is not None for value in values):
        raise ValueError("QSA indexer fields must either all be set or all be None")
    return all(value is not None for value in values)


def enable_dense_qwen4_exp_attention() -> None:
    """Make null-indexer Qwen4-Exp attention execute as dense attention."""
    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    original_init = modeling.Qwen4ExpTextQSAIndexer.__init__
    original_forward = modeling.Qwen4ExpTextQSAIndexer.forward

    def patched_init(self, config, layer_idx):
        if qsa_is_enabled(config):
            original_init(self, config, layer_idx)
            self.qsa_enabled = True
        else:
            torch.nn.Module.__init__(self)
            self.layer_idx = layer_idx
            self.qsa_enabled = False

    def patched_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        if self.qsa_enabled:
            return original_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        if attention_mask.dtype == torch.bool:
            return torch.ones_like(attention_mask, dtype=torch.bool)
        return torch.zeros_like(attention_mask)

    modeling.Qwen4ExpTextQSAIndexer.__init__ = patched_init
    modeling.Qwen4ExpTextQSAIndexer.forward = patched_forward
    _PATCHED = True
