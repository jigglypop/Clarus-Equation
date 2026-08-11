from __future__ import annotations

import argparse
import json
from pathlib import Path

from reality_stone.clarus.brain_geometry_benchmark import evaluate_stn_boundary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_stn_boundary()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["hard_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
