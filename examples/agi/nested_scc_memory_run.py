"""Run one authorized phase of the locked V9 memory benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from reality_stone.clarus.nested_scc_memory_benchmark import run_locked_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("development", "confirmation"))
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-result")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    payload = run_locked_phase(
        repository_root=root,
        preregistration_path=root / args.preregistration,
        authorization_path=root / args.authorization,
        result_path=root / args.output,
        phase=args.phase,
        development_result_path=(
            None if args.development_result is None else root / args.development_result
        ),
    )
    result = payload["result"]
    print(
        {
            "phase": payload["phase"],
            "overall": result["overall"],
            "seed_count": result["seed_count"],
            "mean_accuracies": result["mean_accuracies"],
            "gates": result["gates"],
        }
    )


if __name__ == "__main__":
    main()
