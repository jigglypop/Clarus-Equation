"""Exact Clarus-cell mechanism specification gate.

Exact here means exact within the current Clarus model, not a final biochemical
description of real cells.  The gate freezes the mechanism as state variables,
update order, primitive-to-human operator mapping, closure invariants, and
branch-specific recurrence rules.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("clarus_cell_exact_mechanism_spec_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_exact_mechanism_spec_report.md")


@dataclass(frozen=True)
class StateVar:
    symbol: str
    name: str
    meaning: str
    primitive_form: str
    human_form: str


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    update: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    invariant_role: str
    failure_signature: str


@dataclass(frozen=True)
class RecurrenceSpec:
    branch: str
    projection: str
    recurrence_rule: str
    closure_test: str


STATE_VARS = (
    StateVar(
        "B",
        "boundary_identity",
        "inside/outside distinction and membrane identity",
        "semi-permeable protocell boundary",
        "plasma membrane, channels, receptors, adhesion",
    ),
    StateVar(
        "U",
        "regulated_ports",
        "controlled exchange with the outside",
        "surface influx/efflux chemistry",
        "transporters, vesicle traffic, ER/Golgi/endosome/lysosome routing",
    ),
    StateVar(
        "E",
        "energy_resource",
        "usable internal free energy and resource state",
        "fed chemical resource pool",
        "ATP/redox/calcium/metabolic state",
    ),
    StateVar(
        "A",
        "autocatalytic_metabolism",
        "self-maintaining catalytic production loop",
        "autocatalytic reaction core",
        "mitochondria plus biosynthetic and maintenance metabolism",
    ),
    StateVar(
        "I",
        "identity_template",
        "heritable cell identity constraint",
        "copying template or heritable chemical state",
        "genome plus epigenome and transcriptional regulatory state",
    ),
    StateVar(
        "D",
        "damage_load",
        "accumulated entropy, waste, misfolding, and injury pressure",
        "leakage, decay, copying error",
        "ROS/proteotoxic stress/DNA damage/organelle damage",
    ),
    StateVar(
        "Q",
        "repair_quality_control",
        "damage removal and state restoration capacity",
        "daughter retention quality",
        "repair, autophagy, proteostasis, lysosomal clearance",
    ),
    StateVar(
        "S",
        "support_context",
        "external support that stabilizes the cell unit",
        "environmental gradient and population selection",
        "ECM, vascular, immune, endocrine, neighboring-cell and glial support",
    ),
    StateVar(
        "R",
        "recurrence_operator",
        "projection that makes the next unit count as the same cell type",
        "division threshold and daughter inheritance",
        "cell-cycle recurrence or postmitotic maintenance recurrence",
    ),
)


OPERATORS = (
    OperatorSpec(
        "context_to_ports",
        "U_{t+1}=f_U(U_t,S_t,B_t,D_t)",
        ("S", "B", "D"),
        ("U",),
        "outside influence enters only through regulated exchange",
        "uncontrolled exchange or starvation",
    ),
    OperatorSpec(
        "ports_to_energy",
        "E_{t+1}=f_E(E_t,U_t,A_t,S_t)-c_E(D_t)",
        ("U", "A", "S", "D"),
        ("E",),
        "resource flow must become usable free energy",
        "energy floor collapse",
    ),
    OperatorSpec(
        "energy_to_boundary",
        "B_{t+1}=f_B(B_t,E_t,U_t,Q_t)-l_B(D_t)",
        ("E", "U", "Q", "D"),
        ("B",),
        "identity needs active membrane maintenance",
        "membrane identity loss",
    ),
    OperatorSpec(
        "energy_to_identity",
        "I_{t+1}=f_I(I_t,E_t,Q_t,S_t)-n_I(D_t)",
        ("E", "Q", "S", "D"),
        ("I",),
        "same cell type must be constrained by template and regulatory memory",
        "identity drift",
    ),
    OperatorSpec(
        "metabolism_to_repair",
        "Q_{t+1}=f_Q(Q_t,E_t,U_t,I_t)-c_Q(D_t)",
        ("E", "U", "I", "D"),
        ("Q",),
        "maintenance must actively reduce accumulated damage",
        "damage accumulation",
    ),
    OperatorSpec(
        "repair_to_damage",
        "D_{t+1}=D_t+g_D(E_t,U_t,S_t)-r_D(Q_t,E_t)",
        ("E", "U", "S", "Q"),
        ("D",),
        "damage must stay below identity-collapse boundary",
        "damage exceeds closure threshold",
    ),
    OperatorSpec(
        "identity_to_metabolism",
        "A_{t+1}=f_A(A_t,I_t,E_t,U_t)",
        ("I", "E", "U"),
        ("A",),
        "template state must rebuild the metabolic machinery",
        "metabolic program decouples from identity",
    ),
    OperatorSpec(
        "recurrence_projection",
        "X_{t+1}=Pi_R[B,E,A,I,U,Q,D,S]",
        ("B", "E", "A", "I", "U", "Q", "D", "S"),
        ("R",),
        "the whole state is projected into the next same-type unit",
        "no self-continuing unit",
    ),
)


RECURRENCE = (
    RecurrenceSpec(
        "primitive_or_proliferative",
        "Pi_R = division/asymmetric-division projection",
        "mass_and_identity_cross_threshold -> daughter inherits B,E,A,I,U,Q",
        "division_count >= threshold and daughter identity retained",
    ),
    RecurrenceSpec(
        "human_postmitotic_neural",
        "Pi_R = maintenance projection",
        "no division; membrane/synaptic turnover, repair, autophagy, and glial support keep X_t in same identity basin",
        "energy, identity, membrane, damage, and maintenance recurrence all stay within thresholds",
    ),
)


INVARIANTS = {
    "bounded_identity": ("B", "I"),
    "powered_maintenance": ("E", "A", "Q"),
    "regulated_openness": ("U", "S", "B"),
    "damage_below_boundary": ("D", "Q", "E"),
    "same_type_recurrence": ("R", "B", "I", "S"),
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_symbols = {var.symbol for var in STATE_VARS}
    operator_inputs_outputs = {
        symbol for op in OPERATORS for symbol in (*op.inputs, *op.outputs)
    }
    invariant_symbols = {symbol for values in INVARIANTS.values() for symbol in values}
    recurrence_ok = {spec.branch for spec in RECURRENCE} == {
        "primitive_or_proliferative",
        "human_postmitotic_neural",
    }
    all_symbols_defined = operator_inputs_outputs | invariant_symbols <= state_symbols
    every_state_touched = state_symbols <= operator_inputs_outputs | invariant_symbols
    every_operator_has_failure = all(op.failure_signature for op in OPERATORS)
    primitive_human_mapping_complete = all(var.primitive_form and var.human_form for var in STATE_VARS)
    result = {
        "gate": "clarus_cell_exact_mechanism_spec",
        "passed": bool(
            all_symbols_defined
            and every_state_touched
            and every_operator_has_failure
            and primitive_human_mapping_complete
            and recurrence_ok
            and len(INVARIANTS) == 5
            and len(OPERATORS) == 8
            and len(STATE_VARS) == 9
        ),
        "exact_mechanism": (
            "A Clarus cell is an open bounded identity loop.  Context enters through regulated ports; "
            "ports feed energy; energy maintains boundary, identity, metabolism, and repair; repair keeps "
            "damage below the closure boundary; identity rebuilds the metabolic machinery; recurrence "
            "projects the whole state into either daughters or long-lived maintenance."
        ),
        "state_vars": [asdict(var) for var in STATE_VARS],
        "operators": [asdict(op) for op in OPERATORS],
        "recurrence": [asdict(spec) for spec in RECURRENCE],
        "invariants": INVARIANTS,
        "checks": {
            "all_symbols_defined": all_symbols_defined,
            "every_state_touched": every_state_touched,
            "every_operator_has_failure": every_operator_has_failure,
            "primitive_human_mapping_complete": primitive_human_mapping_complete,
            "recurrence_ok": recurrence_ok,
        },
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Clarus cell exact mechanism specification",
        "",
        f"- passed: `{result['passed']}`",
        f"- mechanism: {result['exact_mechanism']}",
        "",
        "## state variables",
        "",
        "| symbol | name | primitive form | human form | meaning |",
        "|---|---|---|---|---|",
    ]
    for row in result["state_vars"]:
        lines.append(
            f"| `{row['symbol']}` | `{row['name']}` | {row['primitive_form']} | "
            f"{row['human_form']} | {row['meaning']} |"
        )
    lines.extend(
        [
            "",
            "## operators",
            "",
            "| operator | update | invariant role | failure signature |",
            "|---|---|---|---|",
        ]
    )
    for row in result["operators"]:
        lines.append(
            f"| `{row['name']}` | `{row['update']}` | {row['invariant_role']} | {row['failure_signature']} |"
        )
    lines.extend(["", "## recurrence branches", ""])
    for row in result["recurrence"]:
        lines.extend(
            [
                f"### {row['branch']}",
                "",
                f"- projection: `{row['projection']}`",
                f"- recurrence rule: {row['recurrence_rule']}",
                f"- closure test: {row['closure_test']}",
                "",
            ]
        )
    lines.extend(
        [
            "## invariants",
            "",
            "| invariant | variables |",
            "|---|---|",
        ]
    )
    for name, variables in result["invariants"].items():
        lines.append(f"| `{name}` | {', '.join(f'`{symbol}`' for symbol in variables)} |")
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "exact_mechanism": result["exact_mechanism"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
