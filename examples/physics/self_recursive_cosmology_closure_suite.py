"""Run the lightweight self-recursive cosmology gates as one suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
RESULT_JSON = BASE_DIR / "self_recursive_cosmology_closure_suite_results.json"
REPORT_MD = BASE_DIR / "self_recursive_cosmology_closure_suite_report.md"

SUITE = (
    ("research_audit", "examples/physics/recursive_cosmology_research_audit.py"),
    ("h0_selector_self_reference", "examples/physics/h0_readout/h0_recursive_selector_self_reference_gate.py"),
    ("residual_cascade", "examples/physics/residual_cascade_invariant_gate.py"),
    ("kernel_no_free_parameter", "examples/physics/kernel_deformation_no_free_parameter_gate.py"),
    ("d0_measure_transport", "examples/physics/d0_measure_transport_gate.py"),
    ("early_late_measure_preservation", "examples/physics/early_late_measure_preservation_gate.py"),
    ("self_recursive_package", "examples/physics/self_recursive_cosmology_package_gate.py"),
    ("h0_fisher_manifest_validate", "examples/physics/h0_readout/h0_fisher_manifest_validate_gate.py"),
    ("h0_fisher_io_validate", "examples/physics/h0_readout/h0_fisher_io_validate_gate.py"),
    ("h0_fisher_io_regression", "examples/physics/h0_readout/h0_fisher_io_regression_gate.py"),
    ("h0_fisher_io_batch", "examples/physics/h0_readout/h0_fisher_io_batch_gate.py"),
    ("h0_fisher_real_readiness", "examples/physics/h0_readout/h0_fisher_real_readiness_gate.py"),
    ("h0_real_covariance_requirements", "examples/physics/h0_readout/h0_real_covariance_requirements_gate.py"),
    (
        "h0_real_covariance_promotion_decision",
        "examples/physics/h0_readout/h0_real_covariance_promotion_decision_gate.py",
    ),
    ("prediction_ledger", "examples/physics/self_recursive_cosmology_prediction_ledger_gate.py"),
    ("claim_audit", "examples/physics/self_recursive_cosmology_claim_audit_gate.py"),
    ("artifact_index", "examples/physics/self_recursive_cosmology_artifact_index_gate.py"),
)


def run_script(name: str, script: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "script": script,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "passed": proc.returncode == 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [run_script(name, script, args.timeout) for name, script in SUITE]
    result = {
        "gate": "self_recursive_cosmology_closure_suite",
        "passed": all(row["passed"] for row in rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "total": len(rows),
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def stdout_tail(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if line.strip()]
    return " / ".join(lines[-3:])


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Self-recursive cosmology closure suite",
        "",
        f"- passed: `{result['passed']}`",
        f"- passed gates: {result['passed_count']}/{result['total']}",
        "",
        "| gate | script | return code | seconds | stdout tail |",
        "|---|---|---:|---:|---|",
    ]
    for row in result["rows"]:
        tail = stdout_tail(str(row["stdout"])).replace("|", "\\|")
        lines.append(
            f"| `{row['name']}` | `{row['script']}` | {row['returncode']} | "
            f"{row['elapsed_seconds']:.3f} | {tail} |"
        )
    failed = [row for row in result["rows"] if not row["passed"]]
    if failed:
        lines.extend(["", "## failures", ""])
        for row in failed:
            lines.extend(
                [
                    f"### {row['name']}",
                    "",
                    "```text",
                    row["stderr"] or row["stdout"],
                    "```",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_count": result["passed_count"],
                "total": result["total"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
