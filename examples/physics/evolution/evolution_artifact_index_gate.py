"""Index and verify lightweight evolution gate artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RESULT_JSON = BASE_DIR / "evolution_artifact_index_results.json"
REPORT_MD = BASE_DIR / "evolution_artifact_index_report.md"


@dataclass(frozen=True)
class Artifact:
    gate: str
    script: str
    report: str
    role: str


ARTIFACTS = (
    Artifact(
        "life_minimum_dynamics",
        "life_minimum_dynamics_gate.py",
        "life_minimum_dynamics_report.md",
        "toy_minimum_life",
    ),
    Artifact(
        "self_reference_origin_ladder",
        "self_reference_origin_ladder_gate.py",
        "self_reference_origin_ladder_report.md",
        "origin_boundary_audit",
    ),
    Artifact(
        "clarus_cell_origin_timeline",
        "clarus_cell_origin_timeline_gate.py",
        "clarus_cell_origin_timeline_report.md",
        "origin_timeline_morphology",
    ),
    Artifact(
        "clarus_cell_mechanism",
        "clarus_cell_mechanism_gate.py",
        "clarus_cell_mechanism_report.md",
        "mechanism_ablation",
    ),
    Artifact(
        "clarus_cell_to_human_ladder",
        "clarus_cell_to_human_ladder_gate.py",
        "clarus_cell_to_human_ladder_report.md",
        "human_mechanism_ladder",
    ),
    Artifact(
        "human_clarus_cell_multiscale",
        "human_clarus_cell_multiscale_dynamics_gate.py",
        "human_clarus_cell_multiscale_dynamics_report.md",
        "human_multiscale_ablation",
    ),
    Artifact(
        "clarus_cell_exact_mechanism",
        "clarus_cell_exact_mechanism_spec_gate.py",
        "clarus_cell_exact_mechanism_spec_report.md",
        "exact_mechanism_spec",
    ),
    Artifact(
        "c_elegans_trial_boundary",
        "c_elegans_trial_behavior_boundary_audit.py",
        "c_elegans_trial_behavior_boundary_audit_report.md",
        "empirical_boundary",
    ),
    Artifact(
        "drosophila_trial_boundary",
        "drosophila_trial_dynamics_boundary_audit.py",
        "drosophila_trial_dynamics_boundary_audit_report.md",
        "empirical_boundary",
    ),
    Artifact(
        "zebrafish_continuous_boundary",
        "zebrafish_continuous_boundary_final_audit.py",
        "zebrafish_continuous_boundary_final_audit_report.md",
        "empirical_boundary",
    ),
    Artifact(
        "mouse_action_carrier_split",
        "mouse_ibl_action_carrier_split_gate.py",
        "mouse_ibl_action_carrier_split_report.md",
        "mechanism_candidate",
    ),
    Artifact(
        "cross_species_action_carrier",
        "cross_species_action_carrier_invariant_gate.py",
        "cross_species_action_carrier_invariant_report.md",
        "invariant_package",
    ),
    Artifact(
        "evolution_ladder_package",
        "evolution_ladder_closure_package_gate.py",
        "evolution_ladder_closure_package_report.md",
        "closure_package",
    ),
    Artifact(
        "external_dataset_requirements",
        "external_dataset_requirements_gate.py",
        "external_dataset_requirements_report.md",
        "readiness_manifest",
    ),
    Artifact(
        "chapter9_consistency",
        "chapter9_consistency_audit.py",
        "chapter9_consistency_audit_report.md",
        "documentation_audit",
    ),
    Artifact(
        "chapter9_split_docs_consistency",
        "chapter9_split_docs_consistency_audit.py",
        "chapter9_split_docs_consistency_audit_report.md",
        "documentation_audit",
    ),
    Artifact(
        "evolution_lightweight_closure_suite",
        "evolution_lightweight_closure_suite.py",
        "evolution_lightweight_closure_suite_report.md",
        "reproducibility_suite",
    ),
)


def artifact_row(artifact: Artifact) -> dict[str, Any]:
    script_path = BASE_DIR / artifact.script
    report_path = BASE_DIR / artifact.report
    return {
        "gate": artifact.gate,
        "role": artifact.role,
        "script": artifact.script,
        "report": artifact.report,
        "script_exists": script_path.exists(),
        "report_exists": report_path.exists(),
        "script_bytes": script_path.stat().st_size if script_path.exists() else 0,
        "report_bytes": report_path.stat().st_size if report_path.exists() else 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [artifact_row(artifact) for artifact in ARTIFACTS]
    result = {
        "gate": "evolution_artifact_index",
        "passed": all(row["script_exists"] and row["report_exists"] for row in rows),
        "artifact_count": len(rows),
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Evolution artifact index",
        "",
        f"- passed: `{result['passed']}`",
        f"- artifacts: {result['artifact_count']}",
        "",
        "| gate | role | script | report | script exists | report exists |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['gate']}` | `{row['role']}` | `{row['script']}` | "
            f"`{row['report']}` | `{row['script_exists']}` | `{row['report_exists']}` |"
        )
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
                "artifact_count": result["artifact_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
