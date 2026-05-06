"""Run the lightweight evolution closure/audit gates as a reproducibility suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RESULT_JSON = BASE_DIR / "evolution_lightweight_closure_suite_results.json"
REPORT_MD = BASE_DIR / "evolution_lightweight_closure_suite_report.md"

SUITE = (
    ("life_minimum_dynamics", "life_minimum_dynamics_gate.py"),
    ("self_reference_origin_ladder", "self_reference_origin_ladder_gate.py"),
    ("clarus_cell_origin_timeline", "clarus_cell_origin_timeline_gate.py"),
    ("clarus_cell_mechanism", "clarus_cell_mechanism_gate.py"),
    ("clarus_cell_to_human_ladder", "clarus_cell_to_human_ladder_gate.py"),
    ("human_clarus_cell_multiscale", "human_clarus_cell_multiscale_dynamics_gate.py"),
    ("clarus_cell_exact_mechanism", "clarus_cell_exact_mechanism_spec_gate.py"),
    ("brain_clarus_depth_hierarchy", "brain_clarus_depth_hierarchy_gate.py"),
    ("c_elegans_trial_boundary", "c_elegans_trial_behavior_boundary_audit.py"),
    ("drosophila_trial_boundary", "drosophila_trial_dynamics_boundary_audit.py"),
    ("zebrafish_continuous_boundary", "zebrafish_continuous_boundary_final_audit.py"),
    ("mouse_action_carrier_split", "mouse_ibl_action_carrier_split_gate.py"),
    ("cross_species_action_carrier", "cross_species_action_carrier_invariant_gate.py"),
    ("evolution_ladder_package", "evolution_ladder_closure_package_gate.py"),
    ("external_dataset_requirements", "external_dataset_requirements_gate.py"),
    ("chapter9_consistency", "chapter9_consistency_audit.py"),
    ("chapter9_split_docs_consistency", "chapter9_split_docs_consistency_audit.py"),
)


def run_script(name: str, script: str, timeout: int) -> dict[str, Any]:
    path = BASE_DIR / script
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    parsed: dict[str, Any] | None = None
    stdout = proc.stdout.strip()
    if stdout.startswith("{"):
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "name": name,
        "script": script,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout": stdout,
        "stderr": proc.stderr.strip(),
        "parsed": parsed,
        "passed": proc.returncode == 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [run_script(name, script, args.timeout) for name, script in SUITE]
    result = {
        "gate": "evolution_lightweight_closure_suite",
        "passed": all(row["passed"] for row in rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "total": len(rows),
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def summarize_parsed(parsed: dict[str, Any] | None) -> str:
    if not parsed:
        return ""
    keep = {
        key: value
        for key, value in parsed.items()
        if key
        in {
            "passed",
            "verdict",
            "ready_count",
            "total",
            "promoted_terms",
            "blocked_terms",
            "first_minimum_self_reference",
            "first_behavioral_self_reference",
            "first_local_neural_self_reference_proxy",
            "structural_window_ga",
            "evidence_by_window_ga",
            "minimal_form",
            "full_pass_rate",
            "human_clarus_cell_forms",
            "full_pass_rates",
            "exact_mechanism",
            "minimal_brain_depth",
            "minimal_brain_hypothesis",
            "mind_candidate_depth",
        }
    }
    return ", ".join(f"{key}={value}" for key, value in keep.items())


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Evolution lightweight closure suite",
        "",
        f"- passed: `{result['passed']}`",
        f"- passed gates: {result['passed_count']}/{result['total']}",
        "",
        "| gate | script | return code | seconds | parsed summary |",
        "|---|---|---:|---:|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['name']}` | `{row['script']}` | {row['returncode']} | "
            f"{row['elapsed_seconds']:.3f} | {summarize_parsed(row['parsed'])} |"
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
