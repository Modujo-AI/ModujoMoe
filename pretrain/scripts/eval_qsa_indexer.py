#!/usr/bin/env python3
"""Check that trained QSA indexers affect attention and fit held-out dense attention."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.qwen4_exp_dense_compat import enable_dense_qwen4_exp_attention
from pretrain.train_transformers import QSAIndexerDistillation


def read_examples(path: Path, offset: int, count: int) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        for _ in range(offset):
            next(handle)
        return [json.loads(next(handle))["messages"][0]["content"] for _ in range(count)]


def load_initial_indexers(model: torch.nn.Module, initial: Path) -> None:
    index = json.loads((initial / "model.safetensors.index.json").read_text())["weight_map"]
    for name, module in model.named_modules():
        if not name.endswith(".indexer") or not hasattr(module, "index_qk_proj"):
            continue
        weights = {}
        for key in module.state_dict():
            full_name = f"{name}.{key}"
            with safe_open(initial / index[full_name], framework="pt", device="cpu") as shard:
                weights[key] = shard.get_tensor(full_name)
        module.load_state_dict(weights)


def indexers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [module for module in model.modules() if getattr(module, "qsa_enabled", False)]


def sparse_probe(model, tokenizer, text: str, length: int) -> dict:
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=length, return_tensors="pt")
    ids = {key: value.to(model.device) for key, value in ids.items()}
    captured = []
    handles = []

    def capture(module, inputs, output):
        selected = (output[0, 0, -1] == 0).nonzero(as_tuple=False).flatten().cpu().tolist()
        captured.append(selected)

    for module in indexers(model):
        handles.append(module.register_forward_hook(capture))
    start = time.monotonic()
    with torch.inference_mode():
        result = model(**ids, labels=ids["input_ids"], use_cache=False)
    for handle in handles:
        handle.remove()
    return {
        "tokens": ids["input_ids"].shape[1],
        "loss": result.loss.item(),
        "seconds": round(time.monotonic() - start, 2),
        "layers_called": len(captured),
        "last_query_selected_counts": [len(row) for row in captured],
        "last_query_selected": captured,
        "last_logits": result.logits[0, -1].float().cpu(),
    }


def dense_last_query_probs(distiller: QSAIndexerDistillation, record) -> torch.Tensor:
    attention, hidden_states, position_embeddings, attention_mask = record
    hidden_states = hidden_states.to(attention.q_proj.weight.dtype)
    batch_size, seq_length, _ = hidden_states.shape
    head_dim = attention.head_dim
    hidden_shape = (batch_size, seq_length, -1, head_dim)
    cos, sin = position_embeddings
    cos, sin = cos[:, -seq_length:], sin[:, -seq_length:]
    query, _ = torch.chunk(
        attention.q_proj(hidden_states).view(batch_size, seq_length, -1, head_dim * 2), 2, dim=-1
    )
    query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    query, key = distiller.apply_rotary_pos_emb(query, key, cos, sin)
    key = distiller.repeat_kv(key, attention.num_key_value_groups)
    scores = torch.matmul(query[:, :, -1:].float(), key.float().transpose(-1, -2)) * attention.scaling
    if attention_mask.dtype == torch.bool:
        scores = scores.masked_fill(~attention_mask[:, :, -1:], torch.finfo(scores.dtype).min)
    else:
        scores = scores + attention_mask[:, :, -1:].float()
    return scores.softmax(dim=-1).mean(dim=1)[0, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=40000)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--sparse-length", type=int, default=1536)
    parser.add_argument("--kl-length", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    enable_dense_qwen4_exp_attention()
    torch.set_num_threads(4)
    texts = read_examples(args.data, args.offset, args.examples)
    tokenizer = AutoTokenizer.from_pretrained(args.trained, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.trained, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda:0",
    ).eval()
    modules = indexers(model)
    if not modules:
        raise RuntimeError("No active QSA indexers in the trained checkpoint")
    trained_weights = [{key: value.detach().cpu().clone() for key, value in m.state_dict().items()} for m in modules]

    trained = sparse_probe(model, tokenizer, texts[0], args.sparse_length)
    load_initial_indexers(model, args.initial)
    initial = sparse_probe(model, tokenizer, texts[0], args.sparse_length)
    for module, weights in zip(modules, trained_weights):
        module.load_state_dict(weights)
        module.qsa_enabled = False
    dense = sparse_probe(model, tokenizer, texts[0], args.sparse_length)
    for module in modules:
        module.qsa_enabled = True
    if trained["layers_called"] != len(modules) or initial["layers_called"] != len(modules):
        raise RuntimeError("Not every QSA attention layer called its indexer")

    distiller = QSAIndexerDistillation(model)
    with torch.inference_mode():
        ids = tokenizer(texts[0], add_special_tokens=False, truncation=True, max_length=args.sparse_length, return_tensors="pt")
        ids = {key: value.to(model.device) for key, value in ids.items()}
        model(**ids, use_cache=False)
        reference_records = list(distiller.records)
        distiller.records.clear()
        attention_mass = []
        for record, trained_ids, initial_ids in zip(
            reference_records, trained["last_query_selected"], initial["last_query_selected"]
        ):
            probs = dense_last_query_probs(distiller, record)
            budget = len(trained_ids)
            attention_mass.append({
                "trained": probs[trained_ids].sum().item(),
                "initial": probs[initial_ids].sum().item(),
                "oracle_topk": probs.topk(budget).values.sum().item(),
            })
    kl_rows = []
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.kl_length, return_tensors="pt")
            ids = {key: value.to(model.device) for key, value in ids.items()}
            model(**ids, use_cache=False)
            records = list(distiller.records)
            distiller.records.clear()
            if len(records) != len(modules):
                raise RuntimeError(f"Expected {len(modules)} attention records, got {len(records)}")
            trained_kl = torch.stack([distiller._layer_loss(*record) for record in records]).mean().item()
            load_initial_indexers(model, args.initial)
            initial_kl = torch.stack([distiller._layer_loss(*record) for record in records]).mean().item()
            for module, weights in zip(modules, trained_weights):
                module.load_state_dict(weights)
            kl_rows.append({"tokens": ids["input_ids"].shape[1], "trained_kl": trained_kl, "initial_kl": initial_kl})

    trained_set = set(trained["last_query_selected"][0])
    initial_set = set(initial["last_query_selected"][0])
    report = {
        "trained_checkpoint": str(args.trained.resolve()),
        "initial_checkpoint": str(args.initial.resolve()),
        "heldout_data": str(args.data.resolve()),
        "heldout_offset": args.offset,
        "indexer_layers": len(modules),
        "sparse_probe": {
            "tokens": trained["tokens"],
            "trained_loss": trained["loss"],
            "initial_loss": initial["loss"],
            "dense_loss": dense["loss"],
            "trained_seconds": trained["seconds"],
            "initial_seconds": initial["seconds"],
            "trained_selected_counts": trained["last_query_selected_counts"],
            "initial_selected_counts": initial["last_query_selected_counts"],
            "trained_vs_initial_first_layer_jaccard": len(trained_set & initial_set) / len(trained_set | initial_set),
            "trained_vs_dense_last_logit_max_abs_diff": (trained["last_logits"] - dense["last_logits"]).abs().max().item(),
        },
        "heldout_indexer_kl": kl_rows,
        "last_query_dense_attention_mass_by_layer": attention_mass,
        "mean_last_query_mass_trained": sum(row["trained"] for row in attention_mass) / len(attention_mass),
        "mean_last_query_mass_initial": sum(row["initial"] for row in attention_mass) / len(attention_mass),
        "mean_last_query_mass_oracle": sum(row["oracle_topk"] for row in attention_mass) / len(attention_mass),
        "mean_trained_kl": sum(row["trained_kl"] for row in kl_rows) / len(kl_rows),
        "mean_initial_kl": sum(row["initial_kl"] for row in kl_rows) / len(kl_rows),
    }
    report["index_active"] = (
        all(count < trained["tokens"] for count in trained["last_query_selected_counts"])
        and report["sparse_probe"]["trained_vs_dense_last_logit_max_abs_diff"] > 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
