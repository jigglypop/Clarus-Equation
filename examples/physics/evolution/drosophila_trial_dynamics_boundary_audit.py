"""Audit whether Drosophila connectome closure can be promoted to trial dynamics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DATA_ROOT = REPO_ROOT / "data"
LARVA_REPORT = BASE_DIR / "drosophila_larva_next_step_report.md"
ADULT_REPORT = BASE_DIR / "drosophila_adult_flywire_next_step_report.md"
RESULT_JSON = BASE_DIR / "drosophila_trial_dynamics_boundary_audit_results.json"
REPORT_MD = BASE_DIR / "drosophila_trial_dynamics_boundary_audit_report.md"

KEYWORDS = (
    "drosophila",
    "flywire",
    "fly",
    "larva",
    "adult",
    "behavior",
    "behaviour",
    "trial",
    "calcium",
    "ephys",
    "imaging",
    "dynamics",
)
REQUIRED_FIELDS = (
    "trial_id",
    "timebase",
    "neural_activity_or_spikes",
    "behavior_label_or_trace",
    "stimulus_or_task_epoch",
    "celltype_or_region_mapping",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return float(match.group(1))


def require_bool(pattern: str, text: str) -> bool:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    value = match.group(1).strip().strip("`")
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"cannot parse bool: {value!r}")


def candidate_files(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = str(path).lower()
        hits = [keyword for keyword in KEYWORDS if keyword in lower]
        if hits:
            rows.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "size_bytes": path.stat().st_size,
                    "keyword_hits": hits,
                }
            )
    return rows


def structural_evidence() -> dict[str, Any]:
    larva = read(LARVA_REPORT)
    adult = read(ADULT_REPORT)
    return {
        "larva_memory_touched_fraction": require_float(
            r"\| total memory-loop touched fraction \| ([0-9.]+) \|", larva
        ),
        "larva_memory_internal_over_boundary": require_float(
            r"\| memory internal / boundary \| ([0-9.]+) \|", larva
        ),
        "adult_refined_model_gate": require_bool(
            r"- adult refined model gate: `(True|False)`", adult
        ),
        "adult_memory_action_loop_gate": require_bool(
            r"- memory/action loop gate: `(True|False)`", adult
        ),
        "adult_closed": require_bool(r"- closed: `(True|False)`", adult),
        "adult_observed_random_mean": require_float(
            r"\| observed/random mean \| ([0-9.]+) \|", adult
        ),
        "adult_loop_p": require_float(r"\| random >= observed p \| ([0-9.]+) \|", adult),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    files = candidate_files(args.data_root)
    evidence = structural_evidence()
    found_trial_data = len(files) > 0
    found_fields = REQUIRED_FIELDS if found_trial_data else ()
    missing_fields = [field for field in REQUIRED_FIELDS if field not in found_fields]
    empirical_ready = found_trial_data and not missing_fields
    result = {
        "gate": "drosophila_trial_dynamics_boundary_audit",
        "empirical_trial_dynamics_ready": bool(empirical_ready),
        "verdict": "data_boundary" if not empirical_ready else "ready",
        "candidate_files": files,
        "required_fields": list(REQUIRED_FIELDS),
        "missing_fields": missing_fields,
        "structural_evidence": evidence,
        "structural_closed": bool(evidence["adult_closed"]),
        "interpretation": (
            "Adult FlyWire closes the celltype/action/memory structural loop, but no local "
            "time-aligned Drosophila neural/behavior trial dataset is available to test "
            "memory-action dynamics."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    evidence = result["structural_evidence"]
    missing = set(result["missing_fields"])
    lines = [
        "# Drosophila trial-dynamics boundary audit",
        "",
        "This audit separates adult FlyWire structural closure from empirical trial dynamics.",
        "",
        "## verdict",
        "",
        f"- empirical trial dynamics ready: `{result['empirical_trial_dynamics_ready']}`",
        f"- verdict: `{result['verdict']}`",
        f"- structural closed: `{result['structural_closed']}`",
        "",
        "## structural evidence",
        "",
        "| item | value |",
        "|---|---:|",
        f"| larva memory-loop touched fraction | {evidence['larva_memory_touched_fraction']:.6f} |",
        f"| larva memory internal / boundary | {evidence['larva_memory_internal_over_boundary']:.6f} |",
        f"| adult observed/random memory-action loop | {evidence['adult_observed_random_mean']:.6f} |",
        f"| adult loop p | {evidence['adult_loop_p']:.6f} |",
        "",
        "## required trial-dynamics fields",
        "",
        "| required field | found |",
        "|---|---|",
    ]
    for field in result["required_fields"]:
        lines.append(f"| `{field}` | `{field not in missing}` |")
    lines.extend(["", "## candidate local files", ""])
    if result["candidate_files"]:
        lines.extend(["| path | size bytes | keyword hits |", "|---|---:|---|"])
        for row in result["candidate_files"]:
            lines.append(
                f"| `{row['path']}` | {row['size_bytes']} | {', '.join(row['keyword_hits'])} |"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## interpretation",
            "",
            "- Drosophila remains closed at the structural connectome level: celltype/action/memory co-differentiation plus a memory/action loop.",
            "- Trial dynamics are not falsified; they are not testable from the local files.",
            "- Promotion to a temporal behavior equation requires time-aligned neural activity or spikes, behavior traces, task/stimulus epochs, and celltype/region mapping.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "structural_closed": result["structural_closed"],
                "candidate_file_count": len(result["candidate_files"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
