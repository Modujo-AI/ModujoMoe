"""Plain causal-language-model pretraining with Hugging Face Transformers.

Unlike the ms-swift entrypoint, this script deliberately does not apply a chat
template.  For the repository's JSONL files it extracts the content of the
single ``messages`` item and trains on the resulting tokens directly.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from common.qwen4_exp_dense_compat import enable_dense_qwen4_exp_attention
from common.qwen4_exp_loop import enable_qwen4_exp_loop
from pretrain.muon import MuonWithAuxAdamW, use_muon_for_parameter


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Local checkpoint or Hub model id."})
    dtype: str = field(default="bfloat16", metadata={"help": "float32, float16, bfloat16, or auto."})
    # Keep the checkpoint's native expert layout unless explicitly requested.
    # Switching a per-expert checkpoint to grouped_mm at load time changes state-dict keys.
    experts_impl: str | None = field(default=None)
    attn_implementation: str | None = field(default=None)
    router_aux_loss_coef: float = field(default=0.001)
    trust_remote_code: bool = field(default=False)
    qsa_indexer_distill_weight: float = field(
        default=0.0,
        metadata={"help": "Freeze the backbone and distill QSA indexers from dense attention when > 0."},
    )
    qsa_lm_loss_weight: float = field(default=0.0)
    use_muon: bool = field(default=False)
    muon_momentum: float = field(default=0.95)
    muon_ns_steps: int = field(default=5)
    loop_start: int | None = field(default=None)
    loop_end: int | None = field(default=None)
    loop_repeats: int = field(default=1)


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "A JSON or JSONL training file."})
    max_length: int = field(default=2048)
    preprocessing_num_workers: int = field(default=8)
    overwrite_cache: bool = field(default=False)
    add_eos_token: bool = field(default=False)


class CausalLMCollator:
    """Dynamically pad inputs and mask only padding positions in labels."""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels.masked_fill_(batch["attention_mask"].eq(0), -100)
        batch["labels"] = labels
        return batch


class MuonTrainer(Trainer):
    """Route hidden 2D weights to Muon and all other weights to AdamW."""

    def __init__(self, *args, muon_momentum: float, muon_ns_steps: int, **kwargs):
        self.muon_momentum = muon_momentum
        self.muon_ns_steps = muon_ns_steps
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer
        model = model or self.model
        muon_params, adamw_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            # grouped-mm stores the expert matrices as a 3D stack; Muon's
            # batched Newton-Schulz iteration handles each expert separately.
            (muon_params if use_muon_for_parameter(name, parameter) else adamw_params).append(parameter)
        self.optimizer = MuonWithAuxAdamW(
            muon_params,
            adamw_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=self.muon_momentum,
            ns_steps=self.muon_ns_steps,
            adam_betas=(self.args.adam_beta1, self.args.adam_beta2),
            adam_eps=self.args.adam_epsilon,
        )
        LOGGER.info("Muon parameters: %d; auxiliary AdamW parameters: %d", len(muon_params), len(adamw_params))
        return self.optimizer


class QSAIndexerDistillation:
    """Differentiable block-level supervision for QSA's discrete indexer."""

    def __init__(self, model: torch.nn.Module) -> None:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb, repeat_kv

        self.apply_rotary_pos_emb = apply_rotary_pos_emb
        self.repeat_kv = repeat_kv
        self.records: list[tuple[torch.nn.Module, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = []
        self.handles = []
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in model.modules():
            indexer = getattr(module, "indexer", None)
            if indexer is None or not getattr(indexer, "qsa_enabled", False):
                continue
            for parameter in indexer.parameters():
                parameter.requires_grad_(True)
            # Keep the core attention dense while its distribution supervises
            # the differentiable indexer scores. The saved config remains QSA.
            indexer.qsa_enabled = False
            self.handles.append(module.register_forward_pre_hook(self._capture, with_kwargs=True))
        if not self.handles:
            raise ValueError("QSA indexer distillation requested, but the model has no enabled QSA indexers")

    def _capture(self, module, args, kwargs) -> None:
        names = ("hidden_states", "position_embeddings", "attention_mask")
        values = [kwargs[name] if name in kwargs else args[index] for index, name in enumerate(names)]
        self.records.append(
            (module, *values)
        )

    def loss(self) -> torch.Tensor:
        losses = [self._layer_loss(*record) for record in self.records]
        self.records.clear()
        return torch.stack(losses).mean()

    def _layer_loss(self, attention, hidden_states, position_embeddings, attention_mask) -> torch.Tensor:
        indexer = attention.indexer
        # This runs after the model forward, outside Trainer's autocast scope.
        hidden_states = hidden_states.to(attention.q_proj.weight.dtype)
        batch_size, seq_length, _ = hidden_states.shape
        head_dim = attention.head_dim
        hidden_shape = (batch_size, seq_length, -1, head_dim)
        cos, sin = position_embeddings
        cos, sin = cos[:, -seq_length:], sin[:, -seq_length:]

        query, _gate = torch.chunk(
            attention.q_proj(hidden_states).view(batch_size, seq_length, -1, head_dim * 2), 2, dim=-1
        )
        query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        query, key = self.apply_rotary_pos_emb(query, key, cos, sin)
        key = self.repeat_kv(key, attention.num_key_value_groups)
        dense_scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attention.scaling
        if attention_mask.dtype == torch.bool:
            dense_scores = dense_scores.masked_fill(~attention_mask, torch.finfo(dense_scores.dtype).min)
            visible = attention_mask[:, 0]
        else:
            dense_scores = dense_scores + attention_mask.float()
            visible = attention_mask[:, 0].eq(0)
        dense_probs = dense_scores.softmax(dim=-1).mean(dim=1).detach()

        ratio = indexer.compress_ratio
        complete_length = seq_length // ratio * ratio
        if complete_length == 0:
            raise ValueError(f"Sequence length {seq_length} is shorter than QSA compression ratio {ratio}")
        target = dense_probs[..., :complete_length].view(batch_size, seq_length, -1, ratio).sum(dim=-1)
        block_visible = visible[..., :complete_length].view(batch_size, seq_length, -1, ratio).any(dim=-1)

        index_shape = (batch_size, seq_length, -1, indexer.index_head_dim)
        projected = indexer.index_qk_proj(hidden_states)
        index_query, raw_key = torch.split(
            projected,
            [indexer.index_n_heads * indexer.index_head_dim, indexer.index_kv_heads * indexer.index_head_dim],
            dim=-1,
        )
        index_query = indexer.q_layernorm(index_query.reshape(index_shape))
        index_query = self.apply_rotary_pos_emb(index_query, cos=cos, sin=sin, unsqueeze_dim=2)
        raw_key = raw_key.reshape(index_shape).squeeze(2)
        pooled_key = raw_key[:, :complete_length].view(batch_size, -1, ratio, indexer.index_head_dim).mean(dim=2)
        pooled_key = indexer.k_layernorm(pooled_key)
        block_cos = cos[:, :complete_length:ratio]
        block_sin = sin[:, :complete_length:ratio]
        pooled_key = self.apply_rotary_pos_emb(
            pooled_key.unsqueeze(2), cos=block_cos, sin=block_sin, unsqueeze_dim=2
        ).squeeze(2)
        predicted = torch.matmul(index_query.float(), pooled_key.float().unsqueeze(1).transpose(-1, -2))
        predicted = torch.relu(predicted).sum(dim=2) / math.sqrt(indexer.index_head_dim)

        valid_queries = block_visible.any(dim=-1)
        predicted = predicted.masked_fill(~block_visible, torch.finfo(predicted.dtype).min)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        per_query = F.kl_div(predicted.log_softmax(dim=-1), target, reduction="none").sum(dim=-1)
        return per_query.masked_select(valid_queries).mean()


class QSAIndexerTrainer(Trainer):
    def __init__(self, *args, qsa_distillation, distill_weight, lm_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.qsa_distillation = qsa_distillation
        self.distill_weight = distill_weight
        self.lm_weight = lm_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        indexer_loss = self.qsa_distillation.loss()
        loss = self.distill_weight * indexer_loss + self.lm_weight * outputs.loss
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"indexer_distill_loss": indexer_loss.detach().item()})
        return (loss, outputs) if return_outputs else loss


class QSAIndexerMuonTrainer(MuonTrainer, QSAIndexerTrainer):
    """QSA indexer distillation with the same Muon optimizer as other stages."""

    pass


def _extract_text(example: dict[str, Any]) -> str:
    if isinstance(example.get("text"), str):
        return example["text"]
    messages = example.get("messages")
    if isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0].get("content"), str):
        return messages[0]["content"]
    raise ValueError("Each row must contain `text` or exactly one `messages[].content` value.")


def _resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(dtypes)} or 'auto'.")
    return dtypes[name]


def main() -> None:
    enable_dense_qwen4_exp_attention()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_cache = False
    config.router_aux_loss_coef = model_args.router_aux_loss_coef
    config.output_router_logits = model_args.router_aux_loss_coef > 0
    if model_args.experts_impl:
        config.experts_impl = model_args.experts_impl

    load_kwargs: dict[str, Any] = {
        "config": config,
        "dtype": _resolve_dtype(model_args.dtype),
        "trust_remote_code": model_args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if model_args.attn_implementation:
        load_kwargs["attn_implementation"] = model_args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **load_kwargs)
    if model_args.loop_repeats > 1:
        if model_args.loop_start is None or model_args.loop_end is None:
            raise ValueError("--loop_start and --loop_end are required when --loop_repeats > 1")
        enable_qwen4_exp_loop(
            model, start=model_args.loop_start, end=model_args.loop_end, repeats=model_args.loop_repeats
        )
    elif getattr(config, "modujo_loop_repeats", 1) > 1:
        enable_qwen4_exp_loop(
            model,
            start=config.modujo_loop_start,
            end=config.modujo_loop_end,
            repeats=config.modujo_loop_repeats,
        )
    qsa_distillation = None
    if model_args.qsa_indexer_distill_weight > 0:
        if model_args.attn_implementation not in {None, "eager"}:
            raise ValueError("QSA indexer distillation requires --attn_implementation eager")
        qsa_distillation = QSAIndexerDistillation(model)

    suffix = Path(data_args.train_file).suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("--train_file must end in .json or .jsonl")
    raw_dataset = load_dataset("json", data_files={"train": data_args.train_file})["train"]
    columns = raw_dataset.column_names

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        rows = [dict(zip(batch, values)) for values in zip(*batch.values())]
        texts = [_extract_text(row) for row in rows]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=data_args.max_length,
        )
        if data_args.add_eos_token:
            eos = tokenizer.eos_token_id
            for input_ids, attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
                if len(input_ids) < data_args.max_length and (not input_ids or input_ids[-1] != eos):
                    input_ids.append(eos)
                    attention_mask.append(1)
        return encoded

    train_dataset = raw_dataset.map(
        tokenize,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=columns,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing pretraining data",
    )

    if qsa_distillation is not None:
        trainer_class = QSAIndexerMuonTrainer if model_args.use_muon else QSAIndexerTrainer
    else:
        trainer_class = MuonTrainer if model_args.use_muon else Trainer
    trainer_kwargs = {}
    if qsa_distillation is not None:
        trainer_kwargs.update(
            qsa_distillation=qsa_distillation,
            distill_weight=model_args.qsa_indexer_distill_weight,
            lm_weight=model_args.qsa_lm_loss_weight,
        )
    if model_args.use_muon:
        trainer_kwargs.update(muon_momentum=model_args.muon_momentum, muon_ns_steps=model_args.muon_ns_steps)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=CausalLMCollator(tokenizer),
        **trainer_kwargs,
    )

    checkpoint = training_args.resume_from_checkpoint
    if checkpoint is None and Path(training_args.output_dir).is_dir():
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint:
            LOGGER.info("Resuming from %s", checkpoint)

    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
