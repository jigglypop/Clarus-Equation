"""Index and verify self-recursive cosmology artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DOC = REPO_ROOT / "docs" / "3_상수" / "9_우주론_수식_의미와_후보.md"
SUITE = BASE_DIR / "self_recursive_cosmology_closure_suite.py"
RESULT_JSON = BASE_DIR / "self_recursive_cosmology_artifact_index_results.json"
REPORT_MD = BASE_DIR / "self_recursive_cosmology_artifact_index_report.md"


@dataclass(frozen=True)
class Artifact:
    gate: str
    script: str
    report: str
    doc_marker: str
    role: str


ARTIFACTS = (
    Artifact(
        "recursive_cosmology_research_audit",
        "examples/physics/recursive_cosmology_research_audit.py",
        "examples/physics/recursive_cosmology_research_audit_report.md",
        "recursive_cosmology_research_audit.py",
        "research_index",
    ),
    Artifact(
        "h0_selector_self_reference",
        "examples/physics/h0_readout/h0_recursive_selector_self_reference_gate.py",
        "examples/physics/h0_readout/h0_recursive_selector_self_reference_report.md",
        "h0_recursive_selector_self_reference_gate.py",
        "selector_self_reference",
    ),
    Artifact(
        "residual_cascade",
        "examples/physics/residual_cascade_invariant_gate.py",
        "examples/physics/residual_cascade_invariant_report.md",
        "residual_cascade_invariant_gate.py",
        "readout_cascade",
    ),
    Artifact(
        "kernel_no_free_parameter",
        "examples/physics/kernel_deformation_no_free_parameter_gate.py",
        "examples/physics/kernel_deformation_no_free_parameter_report.md",
        "kernel_deformation_no_free_parameter_gate.py",
        "kernel_guardrail",
    ),
    Artifact(
        "d0_measure_transport",
        "examples/physics/d0_measure_transport_gate.py",
        "examples/physics/d0_measure_transport_report.md",
        "d0_measure_transport_gate.py",
        "boundary_transport",
    ),
    Artifact(
        "early_late_measure_preservation",
        "examples/physics/early_late_measure_preservation_gate.py",
        "examples/physics/early_late_measure_preservation_report.md",
        "early_late_measure_preservation_gate.py",
        "horizon_bridge",
    ),
    Artifact(
        "self_recursive_package",
        "examples/physics/self_recursive_cosmology_package_gate.py",
        "examples/physics/self_recursive_cosmology_package_report.md",
        "self_recursive_cosmology_package_gate.py",
        "promotion_package",
    ),
    Artifact(
        "h0_fisher_real_readiness",
        "examples/physics/h0_readout/h0_fisher_real_readiness_gate.py",
        "examples/physics/h0_readout/h0_fisher_real_readiness_report.md",
        "h0_fisher_real_readiness_gate.py",
        "real_data_boundary",
    ),
    Artifact(
        "h0_real_covariance_requirements",
        "examples/physics/h0_readout/h0_real_covariance_requirements_gate.py",
        "examples/physics/h0_readout/h0_real_covariance_requirements_report.md",
        "h0_real_covariance_requirements_gate.py",
        "real_covariance_promotion_requirements",
    ),
    Artifact(
        "h0_real_covariance_promotion_decision",
        "examples/physics/h0_readout/h0_real_covariance_promotion_decision_gate.py",
        "examples/physics/h0_readout/h0_real_covariance_promotion_decision_report.md",
        "h0_real_covariance_promotion_decision_gate.py",
        "real_covariance_promotion_decision",
    ),
    Artifact(
        "prediction_ledger",
        "examples/physics/self_recursive_cosmology_prediction_ledger_gate.py",
        "examples/physics/self_recursive_cosmology_prediction_ledger_report.md",
        "self_recursive_cosmology_prediction_ledger_gate.py",
        "falsification_ledger",
    ),
    Artifact(
        "claim_audit",
        "examples/physics/self_recursive_cosmology_claim_audit_gate.py",
        "examples/physics/self_recursive_cosmology_claim_audit_report.md",
        "self_recursive_cosmology_claim_audit_gate.py",
        "claim_scope_audit",
    ),
)

FISHER_EXAMPLES = (
    "examples/physics/h0_readout/h0_fisher_io_examples/manifest.json",
    "examples/physics/h0_readout/h0_fisher_io_examples/gw_like_fisher.json",
    "examples/physics/h0_readout/h0_fisher_io_examples/gw_like_covariance.json",
    "examples/physics/h0_readout/h0_fisher_io_examples/cmb_global_fisher.json",
    "examples/physics/h0_readout/h0_fisher_io_examples/local_endpoint_fisher.json",
)

SUITE_MARKERS = (
    "research_audit",
    "h0_selector_self_reference",
    "residual_cascade",
    "kernel_no_free_parameter",
    "d0_measure_transport",
    "early_late_measure_preservation",
    "self_recursive_package",
    "h0_fisher_manifest_validate",
    "h0_fisher_io_validate",
    "h0_fisher_io_regression",
    "h0_fisher_io_batch",
    "h0_fisher_real_readiness",
    "h0_real_covariance_requirements",
    "h0_real_covariance_promotion_decision",
    "prediction_ledger",
    "claim_audit",
    "artifact_index",
)


def exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def artifact_row(artifact: Artifact, doc_text: str, suite_text: str) -> dict[str, Any]:
    script_exists = exists(artifact.script)
    report_exists = exists(artifact.report)
    return {
        **asdict(artifact),
        "script_exists": script_exists,
        "report_exists": report_exists,
        "doc_referenced": artifact.doc_marker in doc_text,
        "suite_referenced": artifact.gate in suite_text or artifact.script in suite_text,
    }


def run() -> dict[str, Any]:
    doc_text = DOC.read_text(encoding="utf-8")
    suite_text = SUITE.read_text(encoding="utf-8")
    artifact_rows = [artifact_row(artifact, doc_text, suite_text) for artifact in ARTIFACTS]
    fisher_rows = [{"path": path, "exists": exists(path)} for path in FISHER_EXAMPLES]
    suite_rows = [{"marker": marker, "present": marker in suite_text} for marker in SUITE_MARKERS]
    doc_rows = [
        {"marker": "17/17", "present": "17/17" in doc_text},
        {"marker": "real-ready covariance channel", "present": "real-ready covariance channel" in doc_text},
        {"marker": "not-promoted", "present": "not-promoted" in doc_text},
        {"marker": "blocking requirement", "present": "blocking requirement" in doc_text},
        {"marker": "Selection/Bridge package", "present": "Selection/Bridge package" in doc_text},
        {"marker": "prediction ledger", "present": "prediction ledger" in doc_text},
        {"marker": "claim audit", "present": "claim audit" in doc_text},
    ]
    passed = (
        all(row["script_exists"] and row["report_exists"] and row["doc_referenced"] for row in artifact_rows)
        and all(row["exists"] for row in fisher_rows)
        and all(row["present"] for row in suite_rows)
        and all(row["present"] for row in doc_rows)
    )
    return {
        "gate": "self_recursive_cosmology_artifact_index",
        "passed": passed,
        "artifact_count": len(artifact_rows),
        "fisher_example_count": len(fisher_rows),
        "suite_marker_count": len(suite_rows),
        "artifact_rows": artifact_rows,
        "fisher_rows": fisher_rows,
        "suite_rows": suite_rows,
        "doc_rows": doc_rows,
    }


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Self-recursive cosmology artifact index",
        "",
        f"- passed: `{result['passed']}`",
        f"- artifacts: {result['artifact_count']}",
        f"- Fisher examples: {result['fisher_example_count']}",
        f"- suite markers: {result['suite_marker_count']}",
        "",
        "## Artifacts",
        "",
        "| gate | role | script | report | script | report | doc | suite |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["artifact_rows"]:
        lines.append(
            f"| `{row['gate']}` | `{row['role']}` | `{row['script']}` | `{row['report']}` | "
            f"`{row['script_exists']}` | `{row['report_exists']}` | "
            f"`{row['doc_referenced']}` | `{row['suite_referenced']}` |"
        )
    lines.extend(["", "## Fisher examples", "", "| path | exists |", "|---|---|"])
    for row in result["fisher_rows"]:
        lines.append(f"| `{row['path']}` | `{row['exists']}` |")
    lines.extend(["", "## Suite markers", "", "| marker | present |", "|---|---|"])
    for row in result["suite_rows"]:
        lines.append(f"| `{row['marker']}` | `{row['present']}` |")
    lines.extend(["", "## Doc markers", "", "| marker | present |", "|---|---|"])
    for row in result["doc_rows"]:
        lines.append(f"| `{row['marker']}` | `{row['present']}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    result = run()
    RESULT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(build_report(result), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "artifact_count": result["artifact_count"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
