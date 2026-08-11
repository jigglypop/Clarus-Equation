from __future__ import annotations

import argparse
import json
from pathlib import Path

from reality_stone.clarus.runtime_credit_benchmark import RuntimeCreditBenchConfig, evaluate_runtime_credit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matched-initial-manifold", action="store_true")
    args = parser.parse_args()
    result = evaluate_runtime_credit(
        config=RuntimeCreditBenchConfig(
            matched_initial_manifold=args.matched_initial_manifold
        )
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["hard_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
