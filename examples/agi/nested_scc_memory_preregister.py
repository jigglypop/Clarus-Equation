"""Create the V9 memory benchmark preregistration before any scored run."""

from __future__ import annotations

import argparse
from pathlib import Path

from reality_stone.clarus.nested_scc_memory_benchmark import (
    MemoryBenchmarkConfig,
    canonical_json,
    preregistration_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError("preregistration already exists; overwrite is forbidden")
    payload = preregistration_payload(repository_root=root, config=MemoryBenchmarkConfig())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(output.relative_to(root))


if __name__ == "__main__":
    main()
