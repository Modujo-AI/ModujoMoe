"""Normalize the complete public SFT sources into ms-swift message JSONL files."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/sft_sources"
OUTPUT = ROOT / "datasets/modujo/sft_full"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            counts["rows"] += 1
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                counts["invalid"] += 1
                continue
            if messages[-1].get("role") != "assistant":
                counts["non_assistant_final"] += 1
                continue
            if not any(m.get("role") == "user" for m in messages):
                counts["no_user"] += 1
                continue
            counts["written"] += 1
            if row.get("tools") and row["tools"] != "[]":
                counts["with_tools"] += 1
            if any("<think>" in str(m.get("content", "")) for m in messages):
                counts["with_reasoning"] += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def minimind_rows(path: Path):
    def normalize(record):
        messages = []
        tools = None
        for message in record["conversations"]:
            role = message["role"]
            content = message.get("content") or ""
            if role == "system" and message.get("tools"):
                tool_defs = json.loads(message["tools"])
                tools = json.dumps([{"type": "function", **tool} for tool in tool_defs], ensure_ascii=False)
            reasoning = message.get("reasoning_content")
            if reasoning and role == "assistant":
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            result = {"role": role, "content": content}
            if message.get("tool_calls"):
                calls = []
                for call in json.loads(message["tool_calls"]):
                    function = call.get("function", call)
                    arguments = function.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    calls.append({"function": {"name": function["name"], "arguments": arguments}})
                result["tool_calls"] = calls
            messages.append(result)
        row = {"messages": messages, "source": "minimind_sft_t2t"}
        if tools and tools != "[]":
            row["tools"] = tools
        return row

    # The JSON loader infers nested columns from its first block. Start with a
    # real tool example so later tool calls match the inferred Arrow schema.
    anchor_index = None
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            record = json.loads(line)
            if any(message.get("tool_calls") for message in record["conversations"]):
                anchor_index = index
                yield normalize(record)
                break
    if anchor_index is None:
        raise ValueError("MiniMind source has no tool-call example for schema inference")
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index != anchor_index:
                yield normalize(json.loads(line))


def opengpt_rows(path: Path):
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            messages = [
                {"role": roles[message["from"]], "content": message["value"]}
                for message in record["conversations"]
            ]
            yield {"messages": messages, "source": "opengpt_sharegpt"}


def claude_rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024):
        for record in batch.to_pylist():
            yield {
                "messages": [
                    {"role": "user", "content": record["instruction"]},
                    {"role": "assistant", "content": record["response"]},
                ],
                "source": "claude_3_5_sonnet_magpie",
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sources = [
        ("minimind", minimind_rows(SOURCE / "minimind/sft_t2t.jsonl")),
        ("opengpt", opengpt_rows(SOURCE / "opengpt/opengpt.jsonl")),
        ("claude", claude_rows(SOURCE / "claude/data/train-00000-of-00001.parquet")),
    ]
    manifest = {}
    for name, rows in sources:
        path = args.output_dir / f"{name}.jsonl"
        manifest[name] = {"path": str(path), **write_jsonl(path, rows)}
        print(name, manifest[name], flush=True)
    manifest["ecom"] = {
        "path": str(ROOT / "datasets/modujo/ecom_sft_en.jsonl"),
        "rows": 40098,
        "written": 40098,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
