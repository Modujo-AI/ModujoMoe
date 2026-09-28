#!/usr/bin/env python3
"""Normalize raw MiniMind and e-commerce records without packing or truncation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT_DIR / "datasets"


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def minimind_rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Invalid MiniMind text at line {line_number}")
            yield {
                "messages": [{"role": "assistant", "content": text}],
                "source": "minimind_pretrain_t2t",
            }


def wands_rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            parts = [
                ("Product", row.get("product_name")),
                ("Category", row.get("product_class")),
                ("Category hierarchy", row.get("category hierarchy")),
                ("Description", row.get("product_description")),
                ("Features", row.get("product_features")),
            ]
            text = "\n".join(f"{label}: {clean(value)}" for label, value in parts if clean(value))
            if text:
                yield {
                    "messages": [{"role": "assistant", "content": text}],
                    "source": "wayfair_wands_products",
                    "product_id": clean(row.get("product_id")),
                }


def esci_rows(path: Path, batch_size: int) -> Iterable[dict]:
    parquet = pq.ParquetFile(path)
    columns = [
        "product_id",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
        "product_locale",
    ]
    labels = [
        ("Product", "product_title"),
        ("Brand", "product_brand"),
        ("Color", "product_color"),
        ("Description", "product_description"),
        ("Bullet points", "product_bullet_point"),
    ]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            text = "\n".join(f"{label}: {clean(row.get(key))}" for label, key in labels if clean(row.get(key)))
            if text:
                yield {
                    "messages": [{"role": "assistant", "content": text}],
                    "source": "amazon_esci_products",
                    "product_id": clean(row.get("product_id")),
                    "locale": clean(row.get("product_locale")),
                }


def write_jsonl(rows: Iterable[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    count = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(output)
    print(f"Wrote {count:,} unmodified-length records to {output}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("all", "minimind", "wands", "esci"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DATASETS_DIR / "pretrain_standard")
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()

    jobs = {
        "minimind": (
            minimind_rows(DATASETS_DIR / "minimind/pretrain_t2t.jsonl"),
            args.output_dir / "minimind_pretrain_t2t.jsonl",
        ),
        "wands": (
            wands_rows(DATASETS_DIR / "ecommerce/wands/dataset/product.csv"),
            args.output_dir / "wayfair_wands_products.jsonl",
        ),
        "esci": (
            esci_rows(
                DATASETS_DIR
                / "ecommerce/esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet",
                args.batch_size,
            ),
            args.output_dir / "amazon_esci_products.jsonl",
        ),
    }
    selected = jobs if args.source == "all" else {args.source: jobs[args.source]}
    for rows, output in selected.values():
        write_jsonl(rows, output)


if __name__ == "__main__":
    main()
