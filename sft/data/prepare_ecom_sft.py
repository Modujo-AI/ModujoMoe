"""Download and normalize the public e-commerce SFT corpus for ms-swift."""

import argparse
import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


SPLITS = [
    "amazon_reviews",
    "amazon_meta",
    "asos_ecom_dataset",
    "bitext_customer_support",
    "bitext_retail_ecom",
]


def context_block(row: dict) -> str:
    context = json.loads(row["context"] or "{}")
    pieces = []
    docs = context.get("retrieved_docs") or []
    if docs:
        pieces.append("Retrieved product or policy context:\n" + "\n\n".join(docs))
    for key, label in (("user_profile", "User profile"), ("cart_state", "Cart state"), ("order_details", "Order details")):
        value = context.get(key) or {}
        if value:
            pieces.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")
    tools = json.loads(row["tools"] or "[]")
    if tools:
        pieces.append("Available tools:\n" + json.dumps(tools, ensure_ascii=False))
    return "\n\n".join(pieces)


def normalize(row: dict) -> dict:
    user = row["prompt"].strip()
    context = context_block(row)
    if context:
        user += "\n\n" + context
    history = json.loads(row["history"] or "[]")
    messages = [{"role": "system", "content": row["system"].strip()}]
    for turn in history:
        role = "assistant" if turn.get("role") in {"assistant", "agent"} else "user"
        messages.append({"role": role, "content": turn.get("content", turn.get("text", ""))})
    messages.extend([
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["response"].strip()},
    ])
    return {
        "messages": messages,
        "source": row["source"],
        "intent_category": row["intent_category"],
        "intent": row["intent"],
        "response_type": row["response_type"],
        "quality_score": row["quality_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/modujo/ecom_sft_en.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".runtime/huggingface"))
    parser.add_argument("--min-quality", type=float, default=0.8)
    args = parser.parse_args()
    parts = [load_dataset("rescommons/Ecom-Chatbot-Finetuning-Dataset", split=s, cache_dir=str(args.cache_dir)) for s in SPLITS]
    dataset = concatenate_datasets(parts).filter(lambda r: r["quality_score"] >= args.min_quality)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(normalize(row), ensure_ascii=False) + "\n")
    print(f"wrote {len(dataset)} records to {args.output}")


if __name__ == "__main__":
    main()
