#!/usr/bin/env python
"""Run the locked reduced dual-SCC costly-probe diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reality_stone.clarus.dual_scc_probe_benchmark import (
    DualSCCProbeBenchConfig,
    evaluate_dual_scc_probe_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("development", "validation", "test"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.seed_count < 2:
        raise SystemExit("--seed-count must be at least two")
    config = DualSCCProbeBenchConfig(
        episodes_per_seed=args.episodes,
        context_bias=1.10,
        cue_mean_id=0.28,
        cue_mean_ood=0.22,
        evidence_noise_id=1.35,
        evidence_noise_ood=1.50,
        probe_noise_id=0.12,
        probe_noise_ood=0.16,
        model_evidence_noise=1.35,
        model_probe_noise=0.12,
        probe_cost=0.30,
        quadrature_points=9,
        hidden_blocks_id=(29, 37, 31),
        hidden_blocks_ood=(23, 41, 29),
    )
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    result = evaluate_dual_scc_probe_benchmark(
        seeds=seeds,
        config=config,
        role=args.role,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "verdict": result["verdict"],
                    "diagnostic_verdict": result["diagnostic_verdict"],
                    "score": result["score"],
                },
                sort_keys=True,
            )
        )
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
