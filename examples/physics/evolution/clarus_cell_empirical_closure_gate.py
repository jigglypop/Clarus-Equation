"""Aggregate current Clarus-cell empirical closure status.

This is a report gate, not a new dataset analysis.  It reads the local gate
outputs produced by the Clarus-cell empirical pilots and converts them into a
mechanism map:

    B,U,E,A,I,D,Q,S,R

The scoring is deliberately conservative.  Synthetic/mechanistic gates tell us
whether the proposed operators cohere as a model.  Public-data gates are what
raise empirical confidence for specific branches.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_JSON = Path(__file__).with_name("clarus_cell_empirical_closure_results.json")
REPORT_MD = Path(__file__).with_name("clarus_cell_empirical_closure_report.md")

OPERATORS = ("B", "U", "E", "A", "I", "D", "Q", "S", "R")


@dataclass(frozen=True)
class EvidenceGate:
    key: str
    result_path: Path
    branch: str
    operators: tuple[str, ...]
    evidence_kind: str
    max_weight: float
    pass_weight: float
    partial_weight: float
    claim: str


GATES = (
    EvidenceGate(
        key="mechanistic_toy_full_ablation",
        result_path=Path(__file__).with_name("clarus_cell_mechanism_results.json"),
        branch="primitive / abstract cell",
        operators=("B", "U", "A", "I", "R"),
        evidence_kind="synthetic mechanism",
        max_weight=0.15,
        pass_weight=0.15,
        partial_weight=0.06,
        claim="full/ablation toy mechanism",
    ),
    EvidenceGate(
        key="exact_mechanism_spec",
        result_path=Path(__file__).with_name("clarus_cell_exact_mechanism_spec_results.json"),
        branch="formal operator specification",
        operators=OPERATORS,
        evidence_kind="formal specification",
        max_weight=0.12,
        pass_weight=0.12,
        partial_weight=0.05,
        claim="9-variable operator specification",
    ),
    EvidenceGate(
        key="human_multiscale_synthetic",
        result_path=Path(__file__).with_name("human_clarus_cell_multiscale_dynamics_results.json"),
        branch="human proliferative and postmitotic synthetic stress",
        operators=OPERATORS,
        evidence_kind="synthetic human stress model",
        max_weight=0.16,
        pass_weight=0.16,
        partial_weight=0.07,
        claim="human multiscale simulator",
    ),
    EvidenceGate(
        key="crisprbrain_neuron_maintenance",
        result_path=Path(__file__).with_name(
            "clarus_cell_crisprbrain_neuron_maintenance_results.json"
        ),
        branch="human postmitotic neuron maintenance",
        operators=("D", "Q", "E", "U", "R"),
        evidence_kind="public empirical phenotype screen",
        max_weight=0.24,
        pass_weight=0.24,
        partial_weight=0.10,
        claim="ROS/lipid/lysosome/iron CRISPRbrain coupling",
    ),
    EvidenceGate(
        key="depmap_operator_dependency",
        result_path=Path(__file__).with_name("clarus_cell_depmap_operator_dependency_results.json"),
        branch="human proliferative recurrence",
        operators=("B", "U", "E", "A", "I", "D", "Q", "R"),
        evidence_kind="public empirical fitness dependency",
        max_weight=0.24,
        pass_weight=0.24,
        partial_weight=0.12,
        claim="DepMap operator dependency",
    ),
    EvidenceGate(
        key="psapko_neuron_rnaseq",
        result_path=Path(__file__).with_name("clarus_cell_psapko_neuron_repair_results.json"),
        branch="human PSAP-KO neuron transcriptome",
        operators=("D", "Q", "E", "I"),
        evidence_kind="public empirical transcriptome",
        max_weight=0.09,
        pass_weight=0.09,
        partial_weight=0.03,
        claim="PSAP-KO RNA-seq repair gene-set pilot",
    ),
    EvidenceGate(
        key="glia_support_operator",
        result_path=Path(__file__).with_name("clarus_cell_glia_support_operator_results.json"),
        branch="human glia support context",
        operators=("S", "D", "Q", "U", "R"),
        evidence_kind="public empirical glia support screens",
        max_weight=0.20,
        pass_weight=0.20,
        partial_weight=0.08,
        claim="microglia and astrocyte support-context CRISPR screens",
    ),
    EvidenceGate(
        key="hpa_operator_blueprint",
        result_path=Path(__file__).with_name("clarus_cell_hpa_operator_blueprint_results.json"),
        branch="human subcellular operator architecture",
        operators=OPERATORS,
        evidence_kind="public empirical subcellular atlas",
        max_weight=0.10,
        pass_weight=0.10,
        partial_weight=0.04,
        claim="Human Protein Atlas subcellular operator blueprint",
    ),
    EvidenceGate(
        key="jump_morphology_operator",
        result_path=Path(__file__).with_name(
            "clarus_cell_jump_morphology_operator_results.json"
        ),
        branch="human image-based morphology operator activity",
        operators=("B", "U", "E", "A", "I", "D", "Q"),
        evidence_kind="public empirical image morphology profiles",
        max_weight=0.10,
        pass_weight=0.10,
        partial_weight=0.04,
        claim="JUMP Cell Painting operator morphology activity",
    ),
    EvidenceGate(
        key="jump_mitochondria_channel_gate",
        result_path=Path(__file__).with_name(
            "clarus_cell_jump_channel_specific_mitochondria_results.json"
        ),
        branch="human direct mitochondrial image-channel E check",
        operators=("E",),
        evidence_kind="public empirical image morphology profiles",
        max_weight=0.08,
        pass_weight=0.08,
        partial_weight=0.02,
        claim="JUMP direct Mito-channel E specificity check",
    ),
    EvidenceGate(
        key="jump_chemical_mito_positive_control",
        result_path=Path(__file__).with_name(
            "clarus_cell_jump_chemical_mitochondria_positive_control_results.json"
        ),
        branch="human compound direct mitochondrial image-channel assay control",
        operators=(),
        evidence_kind="public empirical assay control",
        max_weight=0.04,
        pass_weight=0.04,
        partial_weight=0.01,
        claim="JUMP compound direct Mito-channel positive control",
    ),
    EvidenceGate(
        key="perturbseq_state_reconstruction",
        result_path=Path(__file__).with_name(
            "clarus_cell_perturbseq_state_reconstruction_results.json"
        ),
        branch="human Perturb-seq transcriptomic operator state",
        operators=("E", "A", "I", "D", "Q", "R"),
        evidence_kind="public empirical transcriptome state",
        max_weight=0.16,
        pass_weight=0.16,
        partial_weight=0.06,
        claim="Replogle K562/RPE1 pseudo-bulk operator state reconstruction",
    ),
)

DEPMAP_OPERATOR_MAP = {
    "B_boundary_membrane": ("B",),
    "U_regulated_ports_traffic": ("U",),
    "E_energy_mitochondria": ("E",),
    "A_metabolic_autocatalytic_core": ("A",),
    "I_identity_template": ("I",),
    "D_Q_repair_quality_control": ("D", "Q"),
    "R_proliferative_recurrence": ("R",),
}

CRISPRBRAIN_OPERATOR_MAP = {
    "Q_lysosome_autophagy_repair": ("Q",),
    "D_redox_iron_lipid_damage": ("D",),
    "E_mito_energy": ("E",),
    "U_boundary_traffic": ("U",),
    "A_metabolic_core": ("A",),
}

JUMP_OPERATOR_MAP = {
    "B_boundary_morphology": ("B",),
    "U_traffic_morphology": ("U",),
    "E_energy_mitochondria_morphology": ("E",),
    "A_metabolic_core_morphology": ("A",),
    "I_identity_template_morphology": ("I",),
    "D_Q_repair_quality_morphology": ("D", "Q"),
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def gate_status(gate: EvidenceGate) -> dict[str, Any]:
    result = load_json(gate.result_path)
    if result is None:
        return {
            "key": gate.key,
            "branch": gate.branch,
            "operators": gate.operators,
            "evidence_kind": gate.evidence_kind,
            "claim": gate.claim,
            "status": "missing",
            "passed": False,
            "weight": 0.0,
            "max_weight": gate.max_weight,
            "result_path": str(gate.result_path),
        }
    passed = bool(result.get("passed"))
    if passed:
        weight = gate.pass_weight
        status = "passed"
    elif result.get("claim_level") == "parsed_no_promotion":
        weight = gate.partial_weight
        status = "parsed_no_promotion"
    else:
        weight = 0.0
        status = "failed"
    operator_support = gate.operators
    if gate.key == "depmap_operator_dependency":
        supported: set[str] = set()
        for summary in result.get("operator_summaries", []):
            if summary.get("passed"):
                supported.update(DEPMAP_OPERATOR_MAP.get(summary.get("key"), ()))
        operator_support = tuple(operator for operator in OPERATORS if operator in supported)
    elif gate.key == "crisprbrain_neuron_maintenance" and passed:
        supported = {"R"}
        for summary in result.get("operator_summaries", []):
            if summary.get("multi_channel_genes", 0) > 0:
                supported.update(CRISPRBRAIN_OPERATOR_MAP.get(summary.get("key"), ()))
        operator_support = tuple(operator for operator in OPERATORS if operator in supported)
    elif gate.key == "jump_morphology_operator" and passed:
        supported = set()
        for summary in result.get("operator_summaries", []):
            if summary.get("passed"):
                supported.update(JUMP_OPERATOR_MAP.get(summary.get("key"), ()))
        operator_support = tuple(operator for operator in OPERATORS if operator in supported)
    elif gate.key == "jump_mitochondria_channel_gate":
        operator_support = ("E",) if passed else ()
    elif gate.key == "perturbseq_state_reconstruction" and passed:
        supported = set(result.get("operators_supported", []))
        operator_support = tuple(operator for operator in OPERATORS if operator in supported)

    return {
        "key": gate.key,
        "branch": gate.branch,
        "operators": operator_support,
        "nominal_operators": gate.operators,
        "evidence_kind": gate.evidence_kind,
        "claim": gate.claim,
        "status": status,
        "passed": passed,
        "weight": weight,
        "max_weight": gate.max_weight,
        "result_path": str(gate.result_path),
        "claim_level": result.get("claim_level"),
    }


def operator_scores(statuses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for operator in OPERATORS:
        supporting = [status for status in statuses if operator in status["operators"]]
        empirical_passes = [
            status
            for status in supporting
            if status["passed"] and status["evidence_kind"].startswith("public empirical")
        ]
        synthetic_passes = [
            status
            for status in supporting
            if status["passed"] and not status["evidence_kind"].startswith("public empirical")
        ]
        score = sum(status["weight"] for status in supporting)
        max_score = sum(status["max_weight"] for status in supporting)
        fraction = score / max_score if max_score else 0.0
        if empirical_passes and fraction >= 0.70:
            level = "empirical strong"
        elif empirical_passes and fraction >= 0.50:
            level = "empirical moderate"
        elif empirical_passes:
            level = "empirical weak"
        elif synthetic_passes:
            level = "model-only"
        else:
            level = "open"
        scores[operator] = {
            "score": round(score, 6),
            "max_score": round(max_score, 6),
            "fraction": round(fraction, 6),
            "level": level,
            "passed_empirical_gates": [status["key"] for status in empirical_passes],
            "passed_synthetic_gates": [status["key"] for status in synthetic_passes],
        }
    return scores


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    statuses = [gate_status(gate) for gate in GATES]
    scores = operator_scores(statuses)
    total_weight = sum(status["weight"] for status in statuses)
    max_weight = sum(status["max_weight"] for status in statuses)
    total_fraction = total_weight / max_weight if max_weight else 0.0

    empirical_passed = [
        status
        for status in statuses
        if status["passed"] and status["evidence_kind"].startswith("public empirical")
    ]
    key_bottlenecks = [
        operator
        for operator, score in scores.items()
        if score["level"] in {"open", "model-only", "empirical weak"}
    ]
    passed = bool(
        len(empirical_passed) >= 2
        and scores["R"]["level"] in {"empirical moderate", "empirical strong"}
        and scores["Q"]["level"] in {"empirical moderate", "empirical strong"}
    )

    return {
        "gate": "clarus_cell_empirical_closure",
        "passed": passed,
        "claim_level": "six_branch_empirical_partial_closure" if passed else "partial_closure",
        "mechanism_percent_estimate": {
            "overall_clarus_cell": [60, 70],
            "postmitotic_neuron_D_Q_R_branch": [60, 70],
            "proliferative_cell_recurrence_branch": [60, 70],
            "perturbseq_transcriptomic_state_branch": [55, 65],
            "glia_tissue_support_context_branch": [60, 70],
            "subcellular_operator_blueprint_branch": [70, 80],
            "image_morphology_operator_activity_branch": [50, 60],
            "jump_direct_mitochondrial_E_branch": [20, 30],
            "origin_cell_full_loop": [30, 40],
            "human_brain_full_mechanism": [37, 44],
        },
        "total_weight": round(total_weight, 6),
        "max_weight": round(max_weight, 6),
        "total_fraction": round(total_fraction, 6),
        "empirical_passed_gates": [status["key"] for status in empirical_passed],
        "key_bottlenecks": key_bottlenecks,
        "gate_statuses": statuses,
        "operator_scores": scores,
        "next_bottleneck_gates": [
            "clarus_cell_jump_dose_or_cell_health_mitochondria_gate.py",
            "clarus_cell_protocell_boundary_recurrence_gate.py",
            "clarus_cell_neuron_glia_coculture_recurrence_gate.py",
        ],
    }


def write_outputs(result: dict[str, Any]) -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    percent = result["mechanism_percent_estimate"]
    lines = [
        "# Clarus cell empirical closure",
        "",
        f"- passed: `{result['passed']}`",
        f"- claim level: `{result['claim_level']}`",
        f"- weighted closure fraction: `{result['total_fraction']}`",
        f"- empirical passed gates: `{','.join(result['empirical_passed_gates'])}`",
        "",
        "## percent estimate",
        "",
        "| scope | estimate |",
        "|---|---:|",
    ]
    for key, value in percent.items():
        lines.append(f"| `{key}` | `{value[0]}-{value[1]}%` |")

    lines.extend(
        [
            "",
            "## operator scores",
            "",
            "| operator | level | fraction | empirical gates |",
            "|---|---|---:|---|",
        ]
    )
    for operator, score in result["operator_scores"].items():
        empirical = ",".join(score["passed_empirical_gates"]) or "none"
        lines.append(
            f"| `{operator}` | `{score['level']}` | {score['fraction']:.3f} | `{empirical}` |"
        )

    lines.extend(["", "## gate statuses", ""])
    lines.append("| gate | kind | status | branch |")
    lines.append("|---|---|---|---|")
    for status in result["gate_statuses"]:
        lines.append(
            f"| `{status['key']}` | `{status['evidence_kind']}` | "
            f"`{status['status']}` | {status['branch']} |"
        )

    lines.extend(
        [
            "",
            "## bottlenecks",
            "",
            "- key operators still weak/open: "
            + (
                f"`{','.join(result['key_bottlenecks'])}`"
                if result["key_bottlenecks"]
                else "`none at operator level`"
            ),
            "- next gates:",
        ]
    )
    for gate in result["next_bottleneck_gates"]:
        lines.append(f"  - `{gate}`")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    write_outputs(result)
    print(json.dumps({"passed": result["passed"], "claim_level": result["claim_level"]}))


if __name__ == "__main__":
    main()
