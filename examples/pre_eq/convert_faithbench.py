"""Convert FaithBench release batches into CE benchmark JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def positive_prefixes(policy: str) -> tuple[str, ...]:
    if policy == "unwanted-only":
        return ("Unwanted",)
    if policy == "unwanted-or-questionable":
        return ("Unwanted", "Questionable")
    raise ValueError(f"unknown FaithBench policy: {policy}")


def annotation_is_hallucinated(annotation: dict[str, Any], *, policy: str) -> bool:
    labels = annotation.get("label", [])
    prefixes = positive_prefixes(policy)
    return any(
        str(label).startswith(prefixes)
        for label in labels
    )


def sample_is_hallucinated(sample: dict[str, Any], *, policy: str) -> bool:
    return any(
        annotation_is_hallucinated(annotation, policy=policy)
        for annotation in sample.get("annotations", [])
    )


def convert(input_dir: Path, output_path: Path, *, policy: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for batch_path in sorted(input_dir.glob("batch_*.json")):
            data = json.loads(batch_path.read_text(encoding="utf-8"))
            for sample in data.get("samples", []):
                metadata = sample.get("metadata", {})
                row = {
                    "id": f"{batch_path.stem}_{sample.get('sample_id')}",
                    "answer": sample.get("summary", ""),
                    "context": sample.get("source", ""),
                    "is_hallucinated": sample_is_hallucinated(sample, policy=policy),
                    "label_policy": policy,
                    "summarizer": metadata.get("summarizer", ""),
                    "raw_sample_id": metadata.get("raw_sample_id", ""),
                    "annotation_count": len(sample.get("annotations", [])),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("unwanted-only", "unwanted-or-questionable"),
        default="unwanted-or-questionable",
    )
    args = parser.parse_args()

    count = convert(args.input_dir, args.output, policy=args.policy)
    print("# FaithBench conversion")
    print(f"records {count}")
    print(f"output {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
