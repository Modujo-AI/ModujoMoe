"""Small, reproducible generation probe for a local Modujo checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.qwen4_exp_dense_compat import enable_dense_qwen4_exp_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    enable_dense_qwen4_exp_attention()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda:0",
    ).eval()
    cases = [
        ("continuation", "今天天气很好，我决定"),
        ("continuation", "从前，有一只小猫，它每天最喜欢做的事情就是"),
        ("knowledge", "中国的首都是"),
        ("explanation", "植物需要阳光，是因为"),
        ("plain_qa", "问：你好，你是谁？\n答："),
        ("plain_qa", "问：1加1等于多少？\n答："),
        ("english", "The sun rises in the east and"),
        ("chat", "你好，请用中文介绍一下自己。"),
        ("chat", "请用一句话解释为什么天空是蓝色的。"),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle, torch.inference_mode():
        for index, (kind, prompt) in enumerate(cases):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) if kind == "chat" else prompt
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            for sample in (False, True):
                torch.manual_seed(42 + index)
                start = time.monotonic()
                kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=sample,
                              use_cache=True, pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=model.generation_config.eos_token_id,
                              repetition_penalty=1.0)
                if sample:
                    kwargs.update(temperature=0.7, top_p=0.9, top_k=50)
                tokens = model.generate(**inputs, **kwargs)[0, inputs.input_ids.shape[1]:]
                record = dict(checkpoint=str(Path(args.checkpoint).resolve()), kind=kind,
                              prompt=prompt, rendered_prompt=rendered,
                              decoding="sample_t0.7_p0.9" if sample else "greedy",
                              text=tokenizer.decode(tokens, skip_special_tokens=True),
                              tokens=tokens.tolist(), seconds=round(time.monotonic()-start, 2))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({k: v for k, v in record.items() if k not in ("tokens", "checkpoint", "rendered_prompt")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
