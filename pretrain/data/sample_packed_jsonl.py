#!/usr/bin/env python3
"""Uniformly sample packed JSONL rows from the entire file with bounded memory."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.rows < 1:
        raise ValueError("--rows must be positive")
    rng = random.Random(args.seed)
    chosen: list[tuple[int, bytes]] = []
    with args.input.open("rb") as source:
        for index, line in enumerate(source):
            if index < args.rows:
                chosen.append((index, line))
            else:
                slot = rng.randrange(index + 1)
                if slot < args.rows:
                    chosen[slot] = (index, line)
    if len(chosen) != args.rows:
        raise ValueError(f"Input contains only {len(chosen)} rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as destination:
        for _, line in sorted(chosen):
            destination.write(line)
    print(f"Sampled {args.rows} of {index + 1} rows into {args.output}")


if __name__ == "__main__":
    main()
