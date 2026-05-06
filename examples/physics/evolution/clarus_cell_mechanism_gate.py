"""Mechanistic Clarus-cell gate with explicit full/ablation dynamics.

This model is intentionally small.  It is not a biochemical origin proof; it is
an executable mechanism map for the Clarus cell hypothesis.  The question is:
which coupled parts are required for a bounded, heritable, dividing unit whose
own state recreates the next own state?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


RESULT_JSON = Path(__file__).with_name("clarus_cell_mechanism_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_mechanism_report.md")


@dataclass(frozen=True)
class Condition:
    boundary: bool
    autocatalytic_core: bool
    copying_template: bool
    gradient_ports: bool
    division_threshold: bool


CONDITIONS = {
    "full_clarus_cell": Condition(True, True, True, True, True),
    "no_boundary": Condition(False, True, True, True, True),
    "no_autocatalytic_core": Condition(True, False, True, True, True),
    "no_copying_template": Condition(True, True, False, True, True),
    "no_gradient_ports": Condition(True, True, True, False, True),
    "no_division_threshold": Condition(True, True, True, True, False),
}


@dataclass(frozen=True)
class Mechanism:
    name: str
    update: str
    role: str
    measurable_proxy: str


MECHANISMS = (
    Mechanism(
        name="boundary_retention",
        update="leak = leak_rate * resource / (membrane + eps)",
        role="keeps inside/outside distinction while allowing an open reactor",
        measurable_proxy="retention half-life, permeability, osmotic stability",
    ),
    Mechanism(
        name="resource_porting",
        update="influx = port_rate * membrane * max(external - resource, 0)",
        role="feeds metabolism without erasing compartment identity",
        measurable_proxy="monomer or nutrient uptake under gradient",
    ),
    Mechanism(
        name="autocatalytic_core",
        update="dA = k_auto * catalyst * resource / (K + resource)",
        role="turns inflow into self-maintenance and membrane growth drive",
        measurable_proxy="growth under dilution, catalytic amplification",
    ),
    Mechanism(
        name="template_copying",
        update="dT = k_copy * template * catalyst * resource / (K + resource)",
        role="preserves lineage-level distinction across cycles",
        measurable_proxy="template amplification, heritable sequence/state bias",
    ),
    Mechanism(
        name="growth_division",
        update="divide when catalyst + template + membrane exceeds threshold",
        role="turns a persistent unit into a selectable recurrence",
        measurable_proxy="division count, daughter retention, lineage growth rate",
    ),
)


def draw_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "external": float(rng.uniform(0.8, 1.2)),
        "port_rate": float(rng.uniform(0.015, 0.03)),
        "passive_influx": float(rng.uniform(0.001, 0.003)),
        "leak_with_boundary": float(rng.uniform(0.006, 0.014)),
        "leak_without_boundary": float(rng.uniform(0.16, 0.24)),
        "k_auto": float(rng.uniform(0.08, 0.13)),
        "k_copy": float(rng.uniform(0.04, 0.07)),
        "k_membrane": float(rng.uniform(0.035, 0.065)),
        "decay": float(rng.uniform(0.004, 0.01)),
        "template_decay": float(rng.uniform(0.003, 0.007)),
        "membrane_decay": float(rng.uniform(0.002, 0.006)),
        "mutation": float(rng.uniform(0.002, 0.007)),
        "division_threshold": float(rng.uniform(0.78, 0.94)),
        "daughter_loss": float(rng.uniform(0.01, 0.04)),
        "half_saturation": float(rng.uniform(0.12, 0.22)),
    }


def simulate_one(
    params: dict[str, float],
    condition: Condition,
    template_bias: float,
    steps: int,
    dt: float,
) -> dict[str, float | int]:
    resource = 0.16
    catalyst = 0.18
    template = 0.18
    membrane = 0.22 if condition.boundary else 0.03
    heredity = template_bias
    divisions = 0
    min_mass = catalyst + template + membrane

    for _ in range(steps):
        surface = max(membrane, 1e-6) ** (2.0 / 3.0)
        if condition.gradient_ports:
            influx = params["port_rate"] * surface * max(params["external"] - resource, 0.0)
        else:
            influx = params["passive_influx"] * max(params["external"] - resource, 0.0)

        leak_rate = (
            params["leak_with_boundary"] if condition.boundary else params["leak_without_boundary"]
        )
        leak = leak_rate * resource / max(membrane, 0.05)
        usable = resource / (params["half_saturation"] + resource)

        if condition.autocatalytic_core:
            auto_flux = params["k_auto"] * catalyst * usable
            membrane_flux = params["k_membrane"] * catalyst * usable
        else:
            auto_flux = 0.0
            membrane_flux = 0.0

        if condition.copying_template:
            copy_flux = params["k_copy"] * template * catalyst * usable
            heredity += dt * params["mutation"] * (template_bias - heredity)
        else:
            copy_flux = 0.0
            heredity += dt * params["mutation"] * (0.5 - heredity) * 6.0

        resource = max(
            0.0,
            resource
            + dt
            * (
                influx
                - leak
                - 0.55 * auto_flux
                - 0.45 * copy_flux
                - 0.35 * membrane_flux
            ),
        )
        catalyst = max(0.0, catalyst + dt * (auto_flux - params["decay"] * catalyst))
        template = max(
            0.0,
            template + dt * (copy_flux - params["template_decay"] * template),
        )
        if condition.boundary:
            membrane = max(
                0.0,
                membrane + dt * (membrane_flux - params["membrane_decay"] * membrane),
            )
        else:
            membrane = max(0.0, membrane - dt * params["membrane_decay"] * membrane * 3.0)

        mass = catalyst + template + membrane
        min_mass = min(min_mass, mass)
        if condition.division_threshold and mass >= params["division_threshold"]:
            retention = 0.5 * (1.0 - params["daughter_loss"])
            catalyst *= retention
            template *= retention
            membrane *= retention
            resource *= retention
            divisions += 1

    final_mass = catalyst + template + membrane
    return {
        "final_mass": float(final_mass),
        "min_mass": float(min_mass),
        "template_mass": float(template),
        "membrane_mass": float(membrane),
        "heredity": float(heredity),
        "divisions": divisions,
    }


def evaluate_draw(
    rng: np.random.Generator,
    condition: Condition,
    args: argparse.Namespace,
) -> dict[str, float | bool]:
    params = draw_params(rng)
    high = simulate_one(params, condition, 0.65, args.steps, args.dt)
    low = simulate_one(params, condition, 0.35, args.steps, args.dt)
    heredity_gap = abs(float(high["heredity"]) - float(low["heredity"]))
    min_final_mass = min(float(high["final_mass"]), float(low["final_mass"]))
    min_template_mass = min(float(high["template_mass"]), float(low["template_mass"]))
    min_membrane_mass = min(float(high["membrane_mass"]), float(low["membrane_mass"]))
    min_divisions = min(int(high["divisions"]), int(low["divisions"]))
    persistent = min_final_mass >= args.mass_threshold
    heritable = heredity_gap >= args.heredity_gap_threshold and min_template_mass > 0.03
    compartmental = min_membrane_mass >= args.membrane_threshold
    recurrent = min_divisions >= args.division_threshold
    return {
        "min_final_mass": min_final_mass,
        "heredity_gap": heredity_gap,
        "min_template_mass": min_template_mass,
        "min_membrane_mass": min_membrane_mass,
        "min_divisions": min_divisions,
        "persistent": persistent,
        "heritable": heritable,
        "compartmental": compartmental,
        "recurrent": recurrent,
        "passed": bool(persistent and heritable and compartmental and recurrent),
    }


def summarize(rows: list[dict[str, float | bool]]) -> dict[str, float]:
    return {
        "draws": float(len(rows)),
        "pass_rate": float(np.mean([row["passed"] for row in rows])),
        "persistent_rate": float(np.mean([row["persistent"] for row in rows])),
        "heritable_rate": float(np.mean([row["heritable"] for row in rows])),
        "compartmental_rate": float(np.mean([row["compartmental"] for row in rows])),
        "recurrent_rate": float(np.mean([row["recurrent"] for row in rows])),
        "mean_mass": float(np.mean([row["min_final_mass"] for row in rows])),
        "mean_heredity_gap": float(np.mean([row["heredity_gap"] for row in rows])),
        "mean_divisions": float(np.mean([row["min_divisions"] for row in rows])),
    }


def diagnose_ablation(name: str, summary: dict[str, float]) -> str:
    if name == "full_clarus_cell":
        return "all required mechanism loops close"
    weakest = min(
        (
            ("persistence", summary["persistent_rate"]),
            ("heredity", summary["heritable_rate"]),
            ("compartment", summary["compartmental_rate"]),
            ("recurrence", summary["recurrent_rate"]),
        ),
        key=lambda item: item[1],
    )
    return f"primary failure: {weakest[0]}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    summaries = {}
    rows_by_condition = {}
    for index, (name, condition) in enumerate(CONDITIONS.items()):
        rng = np.random.default_rng(args.seed + index * 4099)
        rows = [evaluate_draw(rng, condition, args) for _ in range(args.draws)]
        rows_by_condition[name] = rows
        summaries[name] = summarize(rows)

    full_closed = summaries["full_clarus_cell"]["pass_rate"] >= args.full_min_pass_rate
    ablations_blocked = all(
        summaries[name]["pass_rate"] <= args.ablation_max_pass_rate
        for name in CONDITIONS
        if name != "full_clarus_cell"
    )
    result = {
        "gate": "clarus_cell_mechanism",
        "passed": bool(full_closed and ablations_blocked),
        "criteria": {
            "full_min_pass_rate": args.full_min_pass_rate,
            "ablation_max_pass_rate": args.ablation_max_pass_rate,
            "mass_threshold": args.mass_threshold,
            "heredity_gap_threshold": args.heredity_gap_threshold,
            "membrane_threshold": args.membrane_threshold,
            "division_threshold": args.division_threshold,
        },
        "mechanisms": [asdict(mechanism) for mechanism in MECHANISMS],
        "summaries": summaries,
        "diagnoses": {name: diagnose_ablation(name, summary) for name, summary in summaries.items()},
        "minimal_cycle": [
            "environmental gradient feeds resource ports",
            "ports raise internal resource without erasing boundary",
            "autocatalytic core converts resource into catalyst and membrane growth",
            "template copying preserves lineage bias",
            "membrane growth reaches division threshold",
            "daughter compartments inherit catalyst/template/membrane state",
            "selection acts on recurrence rate and heredity retention",
        ],
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Clarus cell mechanism gate",
        "",
        f"- passed: `{result['passed']}`",
        "",
        "## mechanism cycle",
        "",
    ]
    for index, step in enumerate(result["minimal_cycle"], 1):
        lines.append(f"{index}. {step}")
    lines.extend(
        [
            "",
            "## mechanisms",
            "",
            "| mechanism | update | role | measurable proxy |",
            "|---|---|---|---|",
        ]
    )
    for row in result["mechanisms"]:
        lines.append(
            f"| `{row['name']}` | `{row['update']}` | {row['role']} | {row['measurable_proxy']} |"
        )
    lines.extend(
        [
            "",
            "## ablation summary",
            "",
            "| condition | pass rate | persistent | heritable | compartmental | recurrent | mean mass | mean heredity gap | mean divisions | diagnosis |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, summary in result["summaries"].items():
        lines.append(
            f"| `{name}` | {summary['pass_rate']:.3f} | {summary['persistent_rate']:.3f} | "
            f"{summary['heritable_rate']:.3f} | {summary['compartmental_rate']:.3f} | "
            f"{summary['recurrent_rate']:.3f} | {summary['mean_mass']:.6f} | "
            f"{summary['mean_heredity_gap']:.6f} | {summary['mean_divisions']:.3f} | "
            f"{result['diagnoses'][name]} |"
        )
    lines.extend(
        [
            "",
            "## mechanism verdict",
            "",
            "- The Clarus cell is not just a membrane or just a replicator.",
            "- It works only when retention, resource flow, autocatalysis, copying, and division are coupled.",
            "- The first selectable unit is the whole cycle, not any single molecular component.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=160)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--mass-threshold", type=float, default=0.32)
    parser.add_argument("--heredity-gap-threshold", type=float, default=0.18)
    parser.add_argument("--membrane-threshold", type=float, default=0.08)
    parser.add_argument("--division-threshold", type=int, default=1)
    parser.add_argument("--full-min-pass-rate", type=float, default=0.75)
    parser.add_argument("--ablation-max-pass-rate", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "full_pass_rate": result["summaries"]["full_clarus_cell"]["pass_rate"],
                "diagnoses": result["diagnoses"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
