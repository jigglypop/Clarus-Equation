"""Prioritize external datasets needed after the local evolution ladder package."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DATA_ROOT = REPO_ROOT / "data"
RESULT_JSON = BASE_DIR / "external_dataset_requirements_results.json"
REPORT_MD = BASE_DIR / "external_dataset_requirements_report.md"


@dataclass(frozen=True)
class Requirement:
    gate: str
    reason: str
    required_fields: tuple[str, ...]
    keywords: tuple[str, ...]
    next_script: str


REQUIREMENTS = (
    Requirement(
        gate="life_empirical_origin",
        reason="Toy life triad is positive, but origin-of-life evidence is not tested.",
        required_fields=(
            "reaction_network_or_sequence",
            "autocatalysis_or_growth_measure",
            "boundary_or_compartment_condition",
            "copying_or_template_measure",
            "control_or_ablation_condition",
        ),
        keywords=("luca", "ribozyme", "protocell", "autocatal", "vesicle", "origin"),
        next_script="life_empirical_origin_gate.py",
    ),
    Requirement(
        gate="c_elegans_empirical_trial_behavior",
        reason="Connectome proxy is supported, but empirical stimulus-behavior trials are absent.",
        required_fields=(
            "trial_id",
            "stimulus_label_or_time",
            "behavior_label_or_trace",
            "worm_id",
            "timebase",
        ),
        keywords=("elegans", "celegans", "worm", "stimulus", "behavior", "trial"),
        next_script="c_elegans_empirical_trial_behavior_gate.py",
    ),
    Requirement(
        gate="drosophila_trial_dynamics",
        reason="Adult FlyWire structural loop is closed, but trial dynamics are absent.",
        required_fields=(
            "trial_id",
            "timebase",
            "neural_activity_or_spikes",
            "behavior_label_or_trace",
            "stimulus_or_task_epoch",
            "celltype_or_region_mapping",
        ),
        keywords=("drosophila", "fly", "larva", "trial", "behavior", "calcium", "ephys"),
        next_script="drosophila_trial_dynamics_gate.py",
    ),
    Requirement(
        gate="zebrafish_continuous_movement",
        reason="Discrete bridges pass, but e2-to-continuous tracking alignment is missing.",
        required_fields=(
            "e2_frame_timestamp",
            "e2_resampled_speed_or_position",
            "turn_or_heading_trace",
            "fish_or_session_id",
            "alignment_quality_or_sync_marker",
        ),
        keywords=("zebrafish", "e2", "tail", "stage", "tracking", "speed", "heading", "turn"),
        next_script="zebrafish_timestamp_certified_continuous_gate.py",
    ),
    Requirement(
        gate="mammalian_action_replication_or_perturbation",
        reason="Mouse Phi_action and carrier split are candidate-panel/mechanism candidates.",
        required_fields=(
            "registered_sessions_or_subjects",
            "spike_or_activity_matrix",
            "action_targets",
            "region_or_probe_metadata",
            "perturbation_or_larger_panel_indicator",
        ),
        keywords=("ibl", "mouse", "neuropixels", "perturb", "optogen", "action", "wheel"),
        next_script="mammalian_phi_action_replication_gate.py",
    ),
)


def scan_files(root: Path, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = str(path).lower()
        hits = [keyword for keyword in keywords if keyword in lower]
        if hits:
            rows.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "size_bytes": path.stat().st_size,
                    "keyword_hits": hits,
                }
            )
    return rows


def infer_found_fields(files: list[dict[str, Any]], fields: tuple[str, ...]) -> set[str]:
    found = set()
    joined = " ".join(row["path"].lower() for row in files)
    for field in fields:
        tokens = field.lower().replace("_", " ").split()
        if any(token in joined for token in tokens):
            found.add(field)
    return found


def evaluate_requirement(req: Requirement, data_root: Path) -> dict[str, Any]:
    files = scan_files(data_root, req.keywords)
    found_fields = infer_found_fields(files, req.required_fields)
    missing = [field for field in req.required_fields if field not in found_fields]
    readiness = (len(req.required_fields) - len(missing)) / len(req.required_fields)
    return {
        "gate": req.gate,
        "reason": req.reason,
        "readiness": readiness,
        "ready": readiness == 1.0,
        "required_fields": list(req.required_fields),
        "found_fields": sorted(found_fields),
        "missing_fields": missing,
        "candidate_file_count": len(files),
        "candidate_files": files[:20],
        "next_script": req.next_script,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [evaluate_requirement(req, args.data_root) for req in REQUIREMENTS]
    rows.sort(key=lambda row: (-row["readiness"], row["gate"]))
    result = {
        "gate": "external_dataset_requirements",
        "data_root": str(args.data_root),
        "ready_count": sum(bool(row["ready"]) for row in rows),
        "total": len(rows),
        "requirements": rows,
        "next_action": (
            "No external empirical boundary is locally ready; acquire one dataset with the "
            "listed required fields, then run the corresponding next_script."
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# External dataset requirements",
        "",
        f"- data root: `{result['data_root']}`",
        f"- ready gates: {result['ready_count']}/{result['total']}",
        f"- next action: {result['next_action']}",
        "",
        "## readiness",
        "",
        "| gate | readiness | candidate files | missing fields | next script |",
        "|---|---:|---:|---|---|",
    ]
    for row in result["requirements"]:
        lines.append(
            f"| `{row['gate']}` | {row['readiness']:.3f} | {row['candidate_file_count']} | "
            f"{', '.join(f'`{field}`' for field in row['missing_fields'])} | `{row['next_script']}` |"
        )
    lines.extend(["", "## details", ""])
    for row in result["requirements"]:
        lines.extend(
            [
                f"### {row['gate']}",
                "",
                f"- reason: {row['reason']}",
                f"- ready: `{row['ready']}`",
                f"- found fields: {', '.join(f'`{field}`' for field in row['found_fields']) or 'none'}",
                "",
            ]
        )
        if row["candidate_files"]:
            lines.extend(["| candidate file | size bytes | keyword hits |", "|---|---:|---|"])
            for file_row in row["candidate_files"]:
                lines.append(
                    f"| `{file_row['path']}` | {file_row['size_bytes']} | {', '.join(file_row['keyword_hits'])} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(json.dumps({"ready_count": result["ready_count"], "total": result["total"]}, indent=2))


if __name__ == "__main__":
    main()
