"""Offline-pack raw documents into near-full fixed-length samples."""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MODUJO_DATA_DIR", ROOT_DIR / "datasets/modujo"))
MODEL = os.getenv("MODUJO_BASE_MODEL", "Alexhu1999/Modujo-9B-A1B")


def flush(handle, tokenizer, token_buffer):
    text = tokenizer.decode(token_buffer, skip_special_tokens=False)
    record = {"messages": [{"role": "assistant", "content": text}]}
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "train_mixture_1b.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--reserved-tokens", type=int, default=64,
        help="Leave room for the chat template and special tokens.",
    )
    args = parser.parse_args()
    target_tokens = args.sequence_length - args.reserved_tokens
    if target_tokens < 1:
        raise ValueError("--sequence-length must exceed --reserved-tokens")
    if not args.input.is_file():
        raise SystemExit(
            f"Input corpus not found: {args.input}\n"
            "Create the expanded corpus first with:\n"
            "  python pretrain/data/prepare_train_data.py\n"
            "or explicitly pass an existing JSONL file with --input."
        )
    output_path = args.output or DATA_DIR / f"train_mixture_1b_packed_{args.sequence_length}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    buffer = []
    count = 0
    with args.input.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line in source:
            text = json.loads(line)["messages"][0]["content"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            while tokens:
                room = target_tokens - len(buffer)
                buffer.extend(tokens[:room])
                tokens = tokens[room:]
                if len(buffer) == target_tokens:
                    flush(output, tokenizer, buffer)
                    count += 1
                    buffer = []
                elif tokens or len(buffer) + len(separator) <= target_tokens:
                    buffer.extend(separator)
        if buffer:
            flush(output, tokenizer, buffer)
            count += 1
    print(f"Wrote {count} packed samples to {output_path}")


if __name__ == "__main__":
    main()
