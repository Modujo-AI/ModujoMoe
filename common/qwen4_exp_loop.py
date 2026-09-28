"""Repeat a contiguous Qwen4-Exp layer block with shared weights.

The repeated block is inserted immediately after its ordinary execution.
This training experiment requires ``use_cache=False`` because Qwen4-Exp's
cache is keyed by physical layer index and cannot distinguish loop visits.
"""

from __future__ import annotations

from types import MethodType


def enable_qwen4_exp_loop(model, *, start: int, end: int, repeats: int) -> None:
    if repeats < 2:
        raise ValueError("Loop repeats must be at least 2")
    if start < 0 or end >= len(model.model.layers) or start >= end:
        raise ValueError(f"Invalid loop layer range: {start}..{end}")
    if getattr(model, "_modujo_loop_enabled", False):
        raise RuntimeError("Loop was already installed on this model")

    layers = model.model.layers
    # Preserve complete attention/GDN groups and avoid repeating layers with
    # per-layer embeddings, whose inputs refer to physical depth.
    if start % 4 != 0 or end % 4 != 3:
        raise ValueError("Loop boundaries must cover complete 3x GDN + 1x attention groups")
    if any(getattr(layer, "ple", None) is not None for layer in layers[start : end + 1]):
        raise ValueError("Looping layers with PLE is not supported")

    end_layer = layers[end]
    original_forward = end_layer.forward

    def looped_forward(self, hidden_states, *args, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Looped Qwen4-Exp currently requires use_cache=False")
        hidden_states = original_forward(hidden_states, *args, **kwargs)
        for _ in range(repeats - 1):
            for layer in layers[start:end]:
                hidden_states = layer(hidden_states, *args, **kwargs)
            hidden_states = original_forward(hidden_states, *args, **kwargs)
        return hidden_states

    end_layer.forward = MethodType(looped_forward, end_layer)
    model.config.modujo_loop_start = start
    model.config.modujo_loop_end = end
    model.config.modujo_loop_repeats = repeats
    model.config.use_cache = False
    model.generation_config.use_cache = False
    model._modujo_loop_enabled = True
