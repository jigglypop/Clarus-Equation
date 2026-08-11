"""Run the preregistered validation portion of the DPC Loop-1 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reality_stone.clarus.dpc_benchmark import evaluate_learned_validation, evaluate_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=920000)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learned", action="store_true")
    args = parser.parse_args()
    result = (
        evaluate_learned_validation(
            validation_start=args.start_seed,
            validation_episodes=args.episodes,
        )
        if args.learned
        else evaluate_validation(start_seed=args.start_seed, episodes=args.episodes)
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["hard_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
