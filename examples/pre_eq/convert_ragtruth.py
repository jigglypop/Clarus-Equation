"""Convert raw RAGTruth JSONL files into the CE benchmark adapter schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def flatten_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        parts = []
        for key, nested in value.items():
            nested_text = flatten_context(nested)
            if nested_text:
                parts.append(f"{key}: {nested_text}")
        return "\n".join(parts)
    if isinstance(value, list):
        return "\n".join(flatten_context(item) for item in value)
    return str(value)


def load_sources(path: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sources[str(row["source_id"])] = row
    return sources


def convert(
    response_path: Path,
    source_path: Path,
    output_dir: Path,
    *,
    include_quality_issues: bool = False,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources(source_path)
    handles: dict[str, Any] = {}
    counts: dict[str, int] = {}
    try:
        with response_path.open("r", encoding="utf-8") as response_file:
            for line in response_file:
                if not line.strip():
                    continue
                response = json.loads(line)
                if not include_quality_issues and response.get("quality", "good") != "good":
                    continue
                split = str(response.get("split", "unknown"))
                source = sources.get(str(response["source_id"]), {})
                context = flatten_context(source.get("source_info", ""))
                output = {
                    "id": response["id"],
                    "source_id": response["source_id"],
                    "task_type": source.get("task_type", ""),
                    "model": response.get("model", ""),
                    "answer": response.get("response", ""),
                    "context": context,
                    "is_hallucinated": bool(response.get("labels")),
                    "labels": response.get("labels", []),
                    "label_count": len(response.get("labels", [])),
                    "label_types": sorted(
                        {
                            str(label.get("label_type", ""))
                            for label in response.get("labels", [])
                            if label.get("label_type")
                        }
                    ),
                }
                if split not in handles:
                    handles[split] = (output_dir / f"ragtruth_{split}.jsonl").open(
                        "w",
                        encoding="utf-8",
                    )
                handles[split].write(json.dumps(output, ensure_ascii=False) + "\n")
                counts[split] = counts.get(split, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    all_path = output_dir / "ragtruth_all.jsonl"
    with all_path.open("w", encoding="utf-8") as out:
        for split in sorted(counts):
            split_path = output_dir / f"ragtruth_{split}.jsonl"
            with split_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
    counts["all"] = sum(counts.values())
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--source-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-quality-issues", action="store_true")
    args = parser.parse_args()

    counts = convert(
        args.response,
        args.source_info,
        args.output_dir,
        include_quality_issues=args.include_quality_issues,
    )
    print("# RAGTruth conversion")
    for split, count in sorted(counts.items()):
        print(f"{split} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
