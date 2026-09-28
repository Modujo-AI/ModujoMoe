"""Stream a small reproducible bilingual pretraining sample."""

import json
import os
from pathlib import Path

from datasets import load_dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MODUJO_DATA_DIR", ROOT_DIR / "datasets/modujo"))
OUT = DATA_DIR / "smoke_bilingual.jsonl"
SOURCES = (
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 96),
    ("HuggingFaceFW/fineweb-2", "cmn_Hani", 96),
)


def write_dataset(out: Path, sources) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for repo, config, limit in sources:
            dataset = load_dataset(repo, config, split="train", streaming=True)
            written = 0
            for row in dataset:
                text = row.get("text", "").strip()
                if len(text) < 256:
                    continue
                record = {"messages": [{"role": "assistant", "content": text}]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= limit:
                    break
            if written != limit:
                raise RuntimeError(f"Only obtained {written}/{limit} records from {repo}:{config}")
    print(f"Wrote {sum(x[2] for x in sources)} records to {out}")


if __name__ == "__main__":
    write_dataset(OUT, SOURCES)
