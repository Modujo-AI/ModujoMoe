"""Build a token-budgeted bilingual pretraining mixture by streaming sources."""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# The repository has a top-level data directory named ``datasets``. Remove the
# repository root from import lookup here so it cannot shadow Hugging Face's
# installed ``datasets`` package.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path = [
    entry for entry in sys.path
    if Path(entry or os.getcwd()).resolve() != ROOT_DIR
]
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_DIR = Path(os.getenv("MODUJO_DATA_DIR", ROOT_DIR / "datasets/modujo"))


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    config: str | None
    weight: float


OPEN_RECIPE = (
    Source("fineweb_edu_en", "HuggingFaceFW/fineweb-edu", "sample-10BT", .40),
    Source("fineweb2_zh", "HuggingFaceFW/fineweb-2", "cmn_Hani", .40),
    Source("finemath", "HuggingFaceTB/finemath", "finemath-4plus", .10),
    Source("wikipedia_en", "wikimedia/wikipedia", "20231101.en", .05),
    Source("wikipedia_zh", "wikimedia/wikipedia", "20231101.zh", .05),
)
CCI_RECIPE = (
    Source("fineweb_edu_en", "HuggingFaceFW/fineweb-edu", "sample-10BT", .35),
    Source("fineweb2_zh", "HuggingFaceFW/fineweb-2", "cmn_Hani", .35),
    Source("cci3_hq_zh", "BAAI/CCI3-HQ", None, .10),
    Source("finemath", "HuggingFaceTB/finemath", "finemath-4plus", .10),
    Source("wikipedia_en", "wikimedia/wikipedia", "20231101.en", .05),
    Source("wikipedia_zh", "wikimedia/wikipedia", "20231101.zh", .05),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "train_mixture_1b.jsonl")
    parser.add_argument("--model", default=os.getenv("MODUJO_BASE_MODEL", "Alexhu1999/Modujo-9B-A1B"))
    parser.add_argument("--include-cci3", action="store_true")
    parser.add_argument("--min-chars", type=int, default=256)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true", help="Continue an existing partial output")
    args = parser.parse_args()
    if args.target_tokens < 1:
        raise ValueError("--target-tokens must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing corpus: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    recipe = CCI_RECIPE if args.include_cci3 else OPEN_RECIPE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen, stats = set(), {}
    if args.resume and args.output.exists():
        print(f"Scanning partial corpus before resume: {args.output}", flush=True)
        with args.output.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSON at {args.output}:{line_number}; repair the final partial line before resuming"
                    ) from error
                text = record["messages"][0]["content"]
                seen.add(hashlib.blake2b(text.encode(), digest_size=16).digest())
                source_stats = stats.setdefault(
                    record.get("source", "unknown"),
                    {"actual_tokens": 0, "documents": 0},
                )
                source_stats["actual_tokens"] += int(record.get("token_count", 0))
                source_stats["documents"] += 1
        print(f"Recovered {len(seen):,} documents: {stats}", flush=True)

    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        for index, source in enumerate(recipe):
            budget = round(args.target_tokens * source.weight)
            prior = stats.get(source.name, {"actual_tokens": 0, "documents": 0})
            if prior["actual_tokens"] >= budget:
                prior["target_tokens"] = budget
                print(f"Skipping completed source {source.name}: {prior}", flush=True)
                continue
            stream = load_dataset(source.repo, source.config, split="train", streaming=True)
            stream = stream.shuffle(seed=args.seed + index, buffer_size=args.shuffle_buffer)
            tokens_written = prior["actual_tokens"]
            documents = prior["documents"]
            for row in stream:
                text = str(row.get("text", "")).strip()
                if len(text) < args.min_chars:
                    continue
                digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                count = len(tokenizer.encode(text, add_special_tokens=False))
                if not count:
                    continue
                output.write(json.dumps({
                    "messages": [{"role": "assistant", "content": text}],
                    "source": source.name,
                    "token_count": count,
                }, ensure_ascii=False) + "\n")
                tokens_written += count
                documents += 1
                if tokens_written >= budget:
                    break
            if tokens_written < budget:
                raise RuntimeError(f"{source.name} exhausted at {tokens_written:,}/{budget:,} tokens")
            stats[source.name] = {"target_tokens": budget, "actual_tokens": tokens_written, "documents": documents}
            output.flush()
            print(source.name, stats[source.name], flush=True)

    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps({"target_tokens": args.target_tokens, "tokenizer": args.model, "sources": stats},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
