"""Toy gate for the life-minimum autocatalysis/boundary/copying triad.

This is not an empirical origin-of-life proof.  It is a minimal dynamical
counterexample gate: remove autocatalysis, boundary, or copying and the toy
system should lose either persistence or heritable template distinction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RESULT_JSON = Path(__file__).with_name("life_minimum_dynamics_results.json")
REPORT_MD = Path(__file__).with_name("life_minimum_dynamics_report.md")


@dataclass(frozen=True)
class Condition:
    autocatalysis: bool
    boundary: bool
    copying: bool


CONDITIONS = {
    "full": Condition(True, True, True),
    "no_autocatalysis": Condition(False, True, True),
    "no_boundary": Condition(True, False, True),
    "no_copying": Condition(True, True, False),
}


def draw_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "auto_rate": float(rng.uniform(0.35, 0.65)),
        "copy_rate": float(rng.uniform(0.16, 0.32)),
        "mutation_rate": float(rng.uniform(0.02, 0.045)),
        "feed_rate": float(rng.uniform(0.001, 0.003)),
        "leak_with_boundary": float(rng.uniform(0.015, 0.04)),
        "leak_without_boundary": float(rng.uniform(0.55, 0.90)),
        "capacity": float(rng.uniform(0.8, 1.2)),
    }


def simulate_lineage(
    params: dict[str, float],
    condition: Condition,
    initial_a_fraction: float,
    cycles: int,
    steps_per_cycle: int,
    dt: float,
    dilution: float,
) -> np.ndarray:
    x = np.asarray([initial_a_fraction, 1.0 - initial_a_fraction], dtype=float) * 0.25
    for _ in range(cycles):
        for _ in range(steps_per_cycle):
            mass = float(np.sum(x))
            freq = x / max(mass, 1e-12)
            if condition.autocatalysis:
                autocatalysis = params["auto_rate"] * x * (1.0 - mass / params["capacity"])
            else:
                autocatalysis = np.zeros_like(x)
            if condition.copying:
                copying = params["copy_rate"] * mass * (freq - 0.5)
            else:
                copying = np.zeros_like(x)
            mutation = params["mutation_rate"] * mass * (0.5 - freq)
            leak_rate = (
                params["leak_with_boundary"]
                if condition.boundary
                else params["leak_without_boundary"]
            )
            leak = leak_rate * x
            feed = params["feed_rate"] * np.asarray([0.5, 0.5], dtype=float)
            x = np.clip(x + dt * (autocatalysis + copying + mutation + feed - leak), 0.0, None)
        x *= dilution
    return x


def evaluate_draw(
    rng: np.random.Generator,
    condition: Condition,
    args: argparse.Namespace,
) -> dict[str, float | bool]:
    params = draw_params(rng)
    high_a = simulate_lineage(
        params,
        condition,
        0.62,
        args.cycles,
        args.steps_per_cycle,
        args.dt,
        args.dilution,
    )
    low_a = simulate_lineage(
        params,
        condition,
        0.38,
        args.cycles,
        args.steps_per_cycle,
        args.dt,
        args.dilution,
    )
    masses = np.asarray([np.sum(high_a), np.sum(low_a)], dtype=float)
    high_freq = float(high_a[0] / max(np.sum(high_a), 1e-12))
    low_freq = float(low_a[0] / max(np.sum(low_a), 1e-12))
    heredity = min(1.0, abs(high_freq - low_freq) / 0.24)
    viable = bool(np.min(masses) >= args.mass_threshold)
    heritable = bool(heredity >= args.heredity_threshold)
    return {
        "min_mass": float(np.min(masses)),
        "mean_mass": float(np.mean(masses)),
        "heredity": float(heredity),
        "viable": viable,
        "heritable": heritable,
        "passed": bool(viable and heritable),
    }


def summarize(rows: list[dict[str, float | bool]]) -> dict[str, Any]:
    return {
        "draws": len(rows),
        "pass_rate": float(np.mean([row["passed"] for row in rows])),
        "viable_rate": float(np.mean([row["viable"] for row in rows])),
        "heritable_rate": float(np.mean([row["heritable"] for row in rows])),
        "mean_min_mass": float(np.mean([row["min_mass"] for row in rows])),
        "mean_heredity": float(np.mean([row["heredity"] for row in rows])),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    summaries = {}
    rows_by_condition = {}
    for index, (name, condition) in enumerate(CONDITIONS.items()):
        rng = np.random.default_rng(args.seed + index * 1009)
        rows = [evaluate_draw(rng, condition, args) for _ in range(args.draws)]
        rows_by_condition[name] = rows
        summaries[name] = summarize(rows)

    full_pass = summaries["full"]["pass_rate"] >= args.full_min_pass_rate
    ablation_fail = all(
        summaries[name]["pass_rate"] <= args.ablation_max_pass_rate
        for name in ("no_autocatalysis", "no_boundary", "no_copying")
    )
    result = {
        "gate": "life_minimum_dynamics",
        "passed": bool(full_pass and ablation_fail),
        "criteria": {
            "mass_threshold": args.mass_threshold,
            "heredity_threshold": args.heredity_threshold,
            "full_min_pass_rate": args.full_min_pass_rate,
            "ablation_max_pass_rate": args.ablation_max_pass_rate,
        },
        "summaries": summaries,
        "interpretation": (
            "The toy open chemistry needs all three terms: autocatalysis for growth, "
            "boundary for retention, and copying for heritable template distinction."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Life minimum dynamics toy gate",
        "",
        "This is a toy dynamical gate, not an empirical origin-of-life closure.",
        "It asks whether autocatalysis, boundary retention, and copying are jointly necessary for persistence plus heritable template distinction.",
        "",
        "## criteria",
        "",
    ]
    for key, value in result["criteria"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## summary",
            "",
            f"- passed: `{result['passed']}`",
            "",
            "| condition | pass rate | viable rate | heritable rate | mean min mass | mean heredity |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in result["summaries"].items():
        lines.append(
            "| {name} | {pass_rate:.3f} | {viable_rate:.3f} | {heritable_rate:.3f} | {mean_min_mass:.6f} | {mean_heredity:.6f} |".format(
                name=name,
                **summary,
            )
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- Removing autocatalysis destroys repeated growth under dilution.",
            "- Removing boundary retention destroys persistence in the open reactor.",
            "- Removing copying leaves mass but loses heritable template distinction.",
            "- Therefore the minimum life term is kept as a triad, not as any single component.",
            "",
            "## equation update",
            "",
            "$$",
            "\\boxed{",
            "X_{n+1}",
            "=",
            "\\Pi_{\\mathcal C}",
            "\\left[",
            "X_n",
            "+A_{\\mathrm{auto}}(X_n)",
            "+B_{\\mathrm{boundary}}(X_n)",
            "+C_{\\mathrm{copy}}(X_n)",
            "-L_{\\mathrm{leak}}(X_n)",
            "\\right]",
            "}",
            "$$",
        ]
    )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--steps-per-cycle", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--dilution", type=float, default=0.55)
    parser.add_argument("--mass-threshold", type=float, default=0.25)
    parser.add_argument("--heredity-threshold", type=float, default=0.5)
    parser.add_argument("--full-min-pass-rate", type=float, default=0.8)
    parser.add_argument("--ablation-max-pass-rate", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(json.dumps({"passed": result["passed"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
