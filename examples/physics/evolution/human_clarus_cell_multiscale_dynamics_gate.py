"""Multiscale dynamics gate for human Clarus-cell forms.

This gate refines the ladder result.  Human Clarus cells keep the primitive
kernel, but their recurrence operator splits:

- proliferative cells recur by cell-cycle/asymmetric division under tissue control;
- postmitotic neural cells recur by long-lived maintenance, repair, membrane and
  synaptic turnover, and glial/tissue support.

The model below is a compact stress/ablation simulator, not a clinical cell
biology model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


RESULT_JSON = Path(__file__).with_name("human_clarus_cell_multiscale_dynamics_results.json")
REPORT_MD = Path(__file__).with_name("human_clarus_cell_multiscale_dynamics_report.md")


@dataclass(frozen=True)
class HumanCondition:
    membrane_identity: bool
    mitochondrial_energy: bool
    genome_epigenome: bool
    organelle_traffic: bool
    repair_autophagy: bool
    tissue_support: bool
    recurrence_operator: bool


CONDITIONS = {
    "full": HumanCondition(True, True, True, True, True, True, True),
    "no_membrane_identity": HumanCondition(False, True, True, True, True, True, True),
    "no_mitochondrial_energy": HumanCondition(True, False, True, True, True, True, True),
    "no_genome_epigenome": HumanCondition(True, True, False, True, True, True, True),
    "no_organelle_traffic": HumanCondition(True, True, True, False, True, True, True),
    "no_repair_autophagy": HumanCondition(True, True, True, True, False, True, True),
    "no_tissue_support": HumanCondition(True, True, True, True, True, False, True),
    "no_recurrence_operator": HumanCondition(True, True, True, True, True, True, False),
}


@dataclass(frozen=True)
class HumanForm:
    name: str
    recurrence_mode: str
    stress_load: float
    division_required: bool
    maintenance_required: bool
    tissue_dependence: float
    turnover_need: float


FORMS = (
    HumanForm(
        name="human_proliferative_clarus_cell",
        recurrence_mode="division",
        stress_load=0.45,
        division_required=True,
        maintenance_required=False,
        tissue_dependence=0.55,
        turnover_need=0.45,
    ),
    HumanForm(
        name="human_postmitotic_neural_clarus_cell",
        recurrence_mode="postmitotic_maintenance",
        stress_load=0.70,
        division_required=False,
        maintenance_required=True,
        tissue_dependence=0.82,
        turnover_need=0.80,
    ),
)


@dataclass(frozen=True)
class Operator:
    name: str
    state_variable: str
    primitive_source: str
    human_role: str


OPERATORS = (
    Operator(
        "membrane_identity",
        "membrane",
        "boundary_retention",
        "keeps excitability, receptor state, osmotic identity, and adhesion",
    ),
    Operator(
        "mitochondrial_energy",
        "energy",
        "autocatalytic_core",
        "pays for biosynthesis, ion gradients, repair, firing, and division",
    ),
    Operator(
        "genome_epigenome",
        "identity",
        "copying_template",
        "keeps sequence plus chromatin/regulatory state as a cell-type template",
    ),
    Operator(
        "organelle_traffic",
        "traffic",
        "gradient_ports",
        "routes membrane, protein, nutrient, waste, and vesicle flux",
    ),
    Operator(
        "repair_autophagy",
        "damage",
        "daughter_retention_quality",
        "keeps damage below the identity-collapse boundary",
    ),
    Operator(
        "tissue_support",
        "support",
        "population_selection",
        "adds vascular, immune, endocrine, ECM, glial, and neighboring-cell context",
    ),
    Operator(
        "recurrence_operator",
        "recurrence",
        "division_threshold",
        "chooses division recurrence or postmitotic maintenance recurrence",
    ),
)


def draw_params(rng: np.random.Generator, form: HumanForm) -> dict[str, float]:
    return {
        "energy_supply": float(rng.uniform(0.072, 0.098)),
        "energy_use": float(rng.uniform(0.024, 0.036) * (1.0 + form.stress_load)),
        "identity_repair": float(rng.uniform(0.026, 0.044)),
        "identity_noise": float(rng.uniform(0.006, 0.014)),
        "traffic_flux": float(rng.uniform(0.040, 0.064) * (1.0 + form.turnover_need)),
        "traffic_decay": float(rng.uniform(0.008, 0.016)),
        "damage_rate": float(rng.uniform(0.010, 0.020) * (1.0 + form.stress_load)),
        "repair_rate": float(rng.uniform(0.050, 0.078)),
        "support_rate": float(rng.uniform(0.040, 0.068) * form.tissue_dependence),
        "support_decay": float(rng.uniform(0.010, 0.020)),
        "membrane_turnover": float(rng.uniform(0.040, 0.070)),
        "membrane_decay": float(rng.uniform(0.006, 0.014) * (1.0 + form.turnover_need)),
        "division_threshold": float(rng.uniform(0.92, 1.08)),
        "maintenance_threshold": float(rng.uniform(0.68, 0.76)),
    }


def simulate(
    params: dict[str, float],
    form: HumanForm,
    condition: HumanCondition,
    args: argparse.Namespace,
) -> dict[str, float | int | bool]:
    energy = 0.72
    identity = 0.86
    traffic = 0.70
    membrane = 0.78 if condition.membrane_identity else 0.25
    damage = 0.16
    support = 0.72 if condition.tissue_support else 0.20
    growth_or_maintenance = 0.55
    recurrences = 0
    min_energy = energy
    min_identity = identity
    min_membrane = membrane
    max_damage = damage

    for _ in range(args.steps):
        energy_supply = params["energy_supply"] if condition.mitochondrial_energy else 0.010
        traffic_flux = params["traffic_flux"] if condition.organelle_traffic else 0.001
        repair_rate = params["repair_rate"] if condition.repair_autophagy else 0.0
        identity_repair = params["identity_repair"] if condition.genome_epigenome else 0.0
        support_rate = params["support_rate"] if condition.tissue_support else 0.0
        recurrence_gain = (
            0.060 if condition.recurrence_operator and form.division_required else 0.046
        )
        if not condition.recurrence_operator:
            recurrence_gain = 0.0

        support = np.clip(
            support
            + args.dt * (support_rate - params["support_decay"] * support),
            0.0,
            1.25,
        )
        traffic = np.clip(
            traffic
            + args.dt
            * (
                traffic_flux * min(energy, 1.0) * max(support, 0.05)
                - params["traffic_decay"] * traffic
                - 0.18 * damage
            ),
            0.0,
            1.25,
        )
        energy = np.clip(
            energy
            + args.dt
            * (
                energy_supply * traffic
                - params["energy_use"] * (1.0 + damage)
                + 0.018 * support
            ),
            0.0,
            1.25,
        )
        membrane_drive = params["membrane_turnover"] if condition.membrane_identity else 0.002
        membrane = np.clip(
            membrane
            + args.dt
            * (
                membrane_drive * traffic * min(energy, 1.0)
                - params["membrane_decay"] * membrane
                - 0.10 * damage
            ),
            0.0,
            1.25,
        )
        damage_load = params["damage_rate"] * (1.0 + form.stress_load)
        if not condition.repair_autophagy:
            damage_load *= 1.55
        if not condition.organelle_traffic:
            damage_load *= 1.35
        damage = np.clip(
            damage
            + args.dt
            * (
                damage_load
                - repair_rate * traffic * min(energy, 1.0)
                - 0.016 * support
            ),
            0.0,
            1.25,
        )
        identity = np.clip(
            identity
            + args.dt
            * (
                identity_repair * min(energy, 1.0) * support
                - params["identity_noise"] * (1.0 + damage)
                - 0.08 * max(0.0, 0.45 - membrane)
            ),
            0.0,
            1.25,
        )
        if form.division_required:
            growth_or_maintenance += args.dt * (
                recurrence_gain * min(energy, identity, membrane, traffic)
                - 0.014 * damage
            )
            if growth_or_maintenance >= params["division_threshold"]:
                growth_or_maintenance *= 0.55
                damage *= 0.72
                recurrences += 1
        else:
            growth_or_maintenance = np.clip(
                growth_or_maintenance
                + args.dt
                * (
                    recurrence_gain * min(energy, identity, membrane, traffic, support)
                    - 0.020 * damage
                ),
                0.0,
                1.25,
            )
            if growth_or_maintenance >= params["maintenance_threshold"]:
                recurrences += 1
                growth_or_maintenance *= 0.96
                damage *= 0.94

        min_energy = min(min_energy, float(energy))
        min_identity = min(min_identity, float(identity))
        min_membrane = min(min_membrane, float(membrane))
        max_damage = max(max_damage, float(damage))

    passed = (
        min_energy >= args.energy_threshold
        and min_identity >= args.identity_threshold
        and min_membrane >= args.membrane_threshold
        and max_damage <= args.damage_threshold
        and recurrences >= args.recurrence_threshold
    )
    return {
        "min_energy": float(min_energy),
        "min_identity": float(min_identity),
        "min_membrane": float(min_membrane),
        "max_damage": float(max_damage),
        "recurrences": int(recurrences),
        "passed": bool(passed),
    }


def evaluate_form_condition(
    rng: np.random.Generator,
    form: HumanForm,
    condition: HumanCondition,
    args: argparse.Namespace,
) -> dict[str, float]:
    rows = [simulate(draw_params(rng, form), form, condition, args) for _ in range(args.draws)]
    return {
        "pass_rate": float(np.mean([row["passed"] for row in rows])),
        "mean_min_energy": float(np.mean([row["min_energy"] for row in rows])),
        "mean_min_identity": float(np.mean([row["min_identity"] for row in rows])),
        "mean_min_membrane": float(np.mean([row["min_membrane"] for row in rows])),
        "mean_max_damage": float(np.mean([row["max_damage"] for row in rows])),
        "mean_recurrences": float(np.mean([row["recurrences"] for row in rows])),
    }


def diagnose(summary: dict[str, float]) -> str:
    candidates = (
        ("energy", summary["mean_min_energy"]),
        ("identity", summary["mean_min_identity"]),
        ("membrane", summary["mean_min_membrane"]),
        ("damage_inverse", 1.0 - summary["mean_max_damage"]),
        ("recurrence", summary["mean_recurrences"] / 8.0),
    )
    return f"primary pressure: {min(candidates, key=lambda item: item[1])[0]}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    form_results = {}
    for form_index, form in enumerate(FORMS):
        condition_results = {}
        for condition_index, (condition_name, condition) in enumerate(CONDITIONS.items()):
            rng = np.random.default_rng(args.seed + form_index * 10007 + condition_index * 1009)
            summary = evaluate_form_condition(rng, form, condition, args)
            condition_results[condition_name] = summary | {"diagnosis": diagnose(summary)}
        form_results[form.name] = condition_results

    full_ok = all(
        form_results[form.name]["full"]["pass_rate"] >= args.full_min_pass_rate for form in FORMS
    )
    ablations_blocked = all(
        summary["pass_rate"] <= args.ablation_max_pass_rate
        for form_name, rows in form_results.items()
        for condition_name, summary in rows.items()
        if condition_name != "full"
    )
    result = {
        "gate": "human_clarus_cell_multiscale_dynamics",
        "passed": bool(full_ok and ablations_blocked),
        "criteria": {
            "full_min_pass_rate": args.full_min_pass_rate,
            "ablation_max_pass_rate": args.ablation_max_pass_rate,
            "energy_threshold": args.energy_threshold,
            "identity_threshold": args.identity_threshold,
            "membrane_threshold": args.membrane_threshold,
            "damage_threshold": args.damage_threshold,
            "recurrence_threshold": args.recurrence_threshold,
        },
        "operators": [asdict(operator) for operator in OPERATORS],
        "forms": [asdict(form) for form in FORMS],
        "form_results": form_results,
        "interpretation": (
            "Human Clarus-cell closure is multiscale: a cell-level state remains itself only when "
            "membrane identity, mitochondrial energy, genome/epigenome identity, organelle traffic, "
            "repair/autophagy, tissue support, and the appropriate recurrence operator are all present."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Human Clarus cell multiscale dynamics gate",
        "",
        f"- passed: `{result['passed']}`",
        "",
        "## operators",
        "",
        "| operator | state variable | primitive source | human role |",
        "|---|---|---|---|",
    ]
    for row in result["operators"]:
        lines.append(
            f"| `{row['name']}` | `{row['state_variable']}` | `{row['primitive_source']}` | {row['human_role']} |"
        )
    lines.extend(["", "## form results", ""])
    for form_name, rows in result["form_results"].items():
        lines.extend(
            [
                f"### {form_name}",
                "",
                "| condition | pass rate | min energy | min identity | min membrane | max damage | recurrences | diagnosis |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for condition_name, summary in rows.items():
            lines.append(
                f"| `{condition_name}` | {summary['pass_rate']:.3f} | "
                f"{summary['mean_min_energy']:.3f} | {summary['mean_min_identity']:.3f} | "
                f"{summary['mean_min_membrane']:.3f} | {summary['mean_max_damage']:.3f} | "
                f"{summary['mean_recurrences']:.3f} | {summary['diagnosis']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## verdict",
            "",
            result["interpretation"],
            "",
            "The advanced rule is: primitive recurrence becomes branch-specific human recurrence.",
            "A proliferative cell must close cell-cycle recurrence; a postmitotic neural cell must close maintenance recurrence.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=120)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--steps", type=int, default=520)
    parser.add_argument("--dt", type=float, default=0.06)
    parser.add_argument("--energy-threshold", type=float, default=0.45)
    parser.add_argument("--identity-threshold", type=float, default=0.70)
    parser.add_argument("--membrane-threshold", type=float, default=0.45)
    parser.add_argument("--damage-threshold", type=float, default=0.40)
    parser.add_argument("--recurrence-threshold", type=int, default=2)
    parser.add_argument("--full-min-pass-rate", type=float, default=0.70)
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
                "full_pass_rates": {
                    form: rows["full"]["pass_rate"]
                    for form, rows in result["form_results"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
