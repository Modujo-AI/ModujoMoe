#!/usr/bin/env python3
"""Generate the same prompts with trained QSA, dense attention, and initial QSA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.qwen4_exp_dense_compat import enable_dense_qwen4_exp_attention
from pretrain.scripts.eval_qsa_indexer import load_initial_indexers


def make_cases(tokenizer):
    filler = (
        "Catalog note: A blue ceramic mug is dishwasher safe. A cotton towel is machine washable. "
        "The garden lamp uses two AA batteries. Delivery dates vary by region. "
    )
    needle = "Order ZX-417 has tracking code LUMEN-7392 and is scheduled for delivery on Thursday. "
    question = "\nQuestion: What is the tracking code for order ZX-417? Answer with the code only.\nAnswer:"
    long_text = needle + filler * 80 + question
    while len(tokenizer(long_text, add_special_tokens=False).input_ids) < 1450:
        long_text = needle + filler * 110 + question
        break
    return [
        ("short_qa", "Question: What is the capital of China?\nAnswer:"),
        ("long_retrieval", long_text),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    enable_dense_qwen4_exp_attention()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.trained, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.trained, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda:0",
    ).eval()
    modules = [m for m in model.modules() if getattr(m, "qsa_enabled", False)]
    trained_weights = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in modules]
    cases = make_cases(tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for mode in ("trained_qsa", "dense", "initial_qsa"):
            if mode == "initial_qsa":
                load_initial_indexers(model, args.initial)
            if mode == "trained_qsa":
                for module, weights in zip(modules, trained_weights):
                    module.load_state_dict(weights)
            for module in modules:
                module.qsa_enabled = mode != "dense"
            for case, prompt in cases:
                inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                start = time.monotonic()
                try:
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                            use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        )
                    result = {
                        "mode": mode, "case": case, "prompt_tokens": inputs.input_ids.shape[1],
                        "generated_tokens": generated.shape[1] - inputs.input_ids.shape[1],
                        "text": tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True),
                        "seconds": round(time.monotonic() - start, 2),
                    }
                except Exception as error:
                    result = {"mode": mode, "case": case, "prompt_tokens": inputs.input_ids.shape[1],
                              "error": repr(error), "seconds": round(time.monotonic() - start, 2)}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
