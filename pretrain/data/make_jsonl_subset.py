#!/usr/bin/env python3
"""Create a deterministic prefix subset of a JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument(
        "--balance-by-source",
        action="store_true",
        help="Take an equal number of rows from each source found in the input.",
    )
    args = parser.parse_args()
    if args.rows < 1:
        raise ValueError("--rows must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    if args.balance_by_source:
        rows_by_source = {}
        with args.input.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                bucket = rows_by_source.setdefault(record.get("source", "unknown"), [])
                if len(bucket) < args.rows:
                    bucket.append(line)
                source_count = len(rows_by_source)
                if source_count > 1 and all(
                    len(rows) >= args.rows // source_count for rows in rows_by_source.values()
                ):
                    break
        source_names = sorted(rows_by_source)
        base, remainder = divmod(args.rows, len(source_names))
        with args.output.open("w", encoding="utf-8") as destination:
            for index, name in enumerate(source_names):
                selected = rows_by_source[name][: base + (index < remainder)]
                destination.writelines(selected)
                written += len(selected)
    else:
        with args.input.open("rb") as source, args.output.open("wb") as destination:
            for line in source:
                if not line.strip():
                    continue
                destination.write(line)
                written += 1
                if written == args.rows:
                    break
    if written != args.rows:
        raise ValueError(f"Requested {args.rows} rows but found {written}")
    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
