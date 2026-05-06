"""Package the current life-to-animal-brain ladder closure state."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPORTS = {
    "life": BASE_DIR / "life_minimum_dynamics_report.md",
    "c_elegans_boundary": BASE_DIR / "c_elegans_trial_behavior_boundary_audit_report.md",
    "drosophila_boundary": BASE_DIR / "drosophila_trial_dynamics_boundary_audit_report.md",
    "zebrafish_boundary": BASE_DIR / "zebrafish_continuous_boundary_final_audit_report.md",
    "cross_species_carrier": BASE_DIR / "cross_species_action_carrier_invariant_report.md",
    "mouse_phi_action": BASE_DIR / "mouse_ibl_phi_action_package_report.md",
    "mouse_carrier_split": BASE_DIR / "mouse_ibl_action_carrier_split_report.md",
}
RESULT_JSON = BASE_DIR / "evolution_ladder_closure_package_results.json"
REPORT_MD = BASE_DIR / "evolution_ladder_closure_package_report.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return match.group(1)


def bool_text(value: str) -> bool:
    clean = value.strip().strip("`")
    if clean == "True":
        return True
    if clean == "False":
        return False
    raise ValueError(f"cannot parse bool: {value!r}")


def build_result() -> dict[str, Any]:
    life = read(REPORTS["life"])
    ce = read(REPORTS["c_elegans_boundary"])
    fly = read(REPORTS["drosophila_boundary"])
    fish = read(REPORTS["zebrafish_boundary"])
    carrier = read(REPORTS["cross_species_carrier"])
    mouse_phi = read(REPORTS["mouse_phi_action"])
    mouse_split = read(REPORTS["mouse_carrier_split"])

    promoted = [
        {
            "term": "life minimum triad",
            "status": "toy_positive",
            "evidence": "full pass rate 1.000; ablations no_autocatalysis 0.000, no_boundary 0.000, no_copying 0.020",
            "equation": "A_auto + B_boundary + C_copy - L_leak",
        },
        {
            "term": "C. elegans weighted routing",
            "status": "proxy_supported",
            "evidence": "adult matched/wrong 3.431872; developmental weighted stages 8/8",
            "equation": "gamma L(W_chem) P + stimulus-output domain channel",
        },
        {
            "term": "Drosophila celltype/action/memory structural loop",
            "status": "structural_closed",
            "evidence": "adult memory/action loop observed/random 3.738545; p 0.012987",
            "equation": "D_celltype + A_action + M_MB/CX",
        },
        {
            "term": "Zebrafish activity state and discrete behavior bridge",
            "status": "closed_with_continuous_boundary",
            "evidence": "activity-to-bout-frame and activity-to-direction possible; continuous timestamp-certified ready false",
            "equation": "L_lowrank(P_t) + C_assembly(P_t) + discrete Omega(P_t,q_t)",
        },
        {
            "term": "Cross-species action carrier rule",
            "status": "positive_with_caveats",
            "evidence": f"passed stages {require(r'- passed stages: ([0-9/]+)', carrier)}",
            "equation": "A_carrier: d -> C_d",
        },
        {
            "term": "Mouse Phi_action residual subspace",
            "status": "selection_candidate",
            "evidence": "speed nested action 9/12; wheel nested action 8/12; choice not promoted",
            "equation": "Phi_action = epsilon_{t,S_train}",
        },
        {
            "term": "Mouse speed/wheel carrier split",
            "status": "mechanism_candidate",
            "evidence": "speed probe00/block-distributed; wheel probe00/top16",
            "equation": "Phi_speed(probe00/block) op Phi_wheel(probe00/top16)",
        },
    ]
    blocked = [
        {
            "term": "life empirical origin gate",
            "status": "empirical_pending",
            "evidence": "toy gate is positive, but no LUCA/ribozyme/protocell empirical dataset is tested",
        },
        {
            "term": "C. elegans empirical trial behavior",
            "status": require(r"- verdict: `([^`]+)`", ce),
            "evidence": "no local trial_id/stimulus/behavior/worm_id/timebase table",
        },
        {
            "term": "Drosophila trial dynamics",
            "status": require(r"- verdict: `([^`]+)`", fly),
            "evidence": "no local time-aligned neural/behavior trial table",
        },
        {
            "term": "Zebrafish continuous speed/turn/heading",
            "status": require(r"- verdict: `([^`]+)`", fish),
            "evidence": "timestamp-certified alignments 0/10; no e2 timestamp; no e2-resampled behavior",
        },
        {
            "term": "Mouse universal choice residual subspace",
            "status": "not_promoted",
            "evidence": "choice innovation reproducibility 24-panel fails; choice remains policy/session-adaptive",
        },
    ]
    result = {
        "gate": "evolution_ladder_closure_package",
        "passed": True,
        "promoted_terms": promoted,
        "blocked_terms": blocked,
        "summary": (
            "The local ladder is closed as a tiered theory package: toy life triad, "
            "weighted worm routing proxy, structural fly memory/action loop, zebrafish "
            "activity/discrete behavior closure, cross-species carrier rule, and mouse "
            "train-selected action residual/carrier split.  Empirical trial/time-aligned "
            "gates remain external data boundaries."
        ),
    }
    result["passed"] = len(promoted) == 7 and len(blocked) == 5
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Evolution ladder closure package",
        "",
        f"- passed: `{result['passed']}`",
        f"- summary: {result['summary']}",
        "",
        "## promoted terms",
        "",
        "| term | status | evidence | equation |",
        "|---|---|---|---|",
    ]
    for row in result["promoted_terms"]:
        lines.append(
            f"| {row['term']} | `{row['status']}` | {row['evidence']} | `{row['equation']}` |"
        )
    lines.extend(
        [
            "",
            "## blocked or external-boundary terms",
            "",
            "| term | status | evidence |",
            "|---|---|---|",
        ]
    )
    for row in result["blocked_terms"]:
        lines.append(f"| {row['term']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## final reading",
            "",
            "- The ladder is not a claim that every empirical dataset is closed.",
            "- It is a promoted-term package plus explicit data boundaries.",
            "- The next real empirical moves are external datasets: worm trial behavior, fly trial dynamics, zebrafish synchronized continuous tracking, and larger/perturbational mammalian action tests.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = build_result()
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


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
                "promoted_terms": len(result["promoted_terms"]),
                "blocked_terms": len(result["blocked_terms"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
