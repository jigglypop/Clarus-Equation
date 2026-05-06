"""Cross-species action carrier invariant from existing evolution reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPORTS = {
    "c_elegans": BASE_DIR / "c_elegans_developmental_stimulus_behavior_report.md",
    "drosophila": BASE_DIR / "drosophila_adult_flywire_next_step_report.md",
    "zebrafish_behavior": BASE_DIR / "zebrafish_laser_behavior_report.md",
    "zebrafish_activity": BASE_DIR / "zebrafish_activity_direction_report.md",
    "mouse": BASE_DIR / "mouse_ibl_action_carrier_split_report.md",
}
RESULT_JSON = BASE_DIR / "cross_species_action_carrier_invariant_results.json"
REPORT_MD = BASE_DIR / "cross_species_action_carrier_invariant_report.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return float(match.group(1))


def require_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return int(match.group(1))


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


def c_elegans_evidence(text: str) -> dict[str, Any]:
    passed = require_int(r"\| passed chemical weighted stages \| (\d+) \|", text)
    stages = require_int(r"\| stages \| (\d+) \|", text)
    return {
        "stage": "C. elegans",
        "carrier": "weighted stimulus-output domain channel",
        "passed": passed == stages,
        "status": "proxy",
        "evidence": {
            "passed_weighted_stages": passed,
            "stages": stages,
            "mean_matched_wrong": require_float(
                r"\| mean chemical matched/wrong \| ([0-9.]+) \|", text
            ),
            "min_matched_wrong": require_float(
                r"\| min chemical matched/wrong \| ([0-9.]+) \|", text
            ),
            "mean_p": require_float(r"\| mean p value \| ([0-9.]+) \|", text),
        },
    }


def drosophila_evidence(text: str) -> dict[str, Any]:
    return {
        "stage": "Drosophila adult",
        "carrier": "celltype/action/memory loop",
        "passed": require_bool(r"- closed: `(True|False)`", text),
        "status": "connectome",
        "evidence": {
            "memory_action_loop_fraction": require_float(
                r"\| memory_action_loop_fraction \| ([0-9.]+) \|", text
            ),
            "observed_random_mean": require_float(
                r"\| observed/random mean \| ([0-9.]+) \|", text
            ),
            "loop_p": require_float(r"\| random >= observed p \| ([0-9.]+) \|", text),
        },
    }


def zebrafish_evidence(behavior_text: str, activity_text: str) -> dict[str, Any]:
    behavior_pass = require_bool(r"- pass: (True|False)", behavior_text)
    activity_pass = require_bool(r"- pass: (True|False)", activity_text)
    return {
        "stage": "Zebrafish",
        "carrier": "left/right perturbation-to-direction activity channel",
        "passed": behavior_pass and activity_pass,
        "status": "discrete_direction",
        "evidence": {
            "behavior_effect_ratio": require_float(
                r"experimental/control effect ratio: ([0-9.]+)", behavior_text
            ),
            "activity_auc": require_float(r"- AUC: ([0-9.]+)", activity_text),
            "activity_balanced_accuracy": require_float(
                r"- balanced accuracy: ([0-9.]+)", activity_text
            ),
            "activity_p": require_float(r"- p: ([0-9.]+)", activity_text),
        },
    }


def mouse_evidence(text: str) -> dict[str, Any]:
    speed_pass = require_bool(
        r"\| `first_movement_speed` \| .*? \| `(True|False)` \|", text
    )
    wheel_pass = require_bool(
        r"\| `wheel_action_direction` \| .*? \| `(True|False)` \|", text
    )
    return {
        "stage": "Mouse IBL",
        "carrier": "split speed probe00/block and wheel probe00/top16 carrier",
        "passed": speed_pass and wheel_pass,
        "status": "candidate_panel",
        "evidence": {
            "speed_pattern": "full 9/12; drop_top_ccf 4/11; drop_probe 3/6; only_probe 7/9; only_top_units 6/9",
            "wheel_pattern": "full 8/12; drop_top_ccf 10/11; drop_probe 4/6; only_probe 6/9; only_top_units 7/9",
        },
    }


def build_result() -> dict[str, Any]:
    entries = [
        c_elegans_evidence(read(REPORTS["c_elegans"])),
        drosophila_evidence(read(REPORTS["drosophila"])),
        zebrafish_evidence(read(REPORTS["zebrafish_behavior"]), read(REPORTS["zebrafish_activity"])),
        mouse_evidence(read(REPORTS["mouse"])),
    ]
    passed_count = sum(1 for entry in entries if entry["passed"])
    return {
        "gate": "cross_species_action_carrier_invariant",
        "passed": passed_count == len(entries),
        "passed_count": passed_count,
        "total": len(entries),
        "invariant": (
            "action output is carried by a restricted, weighted, target-linked channel; "
            "the carrier becomes more specialized from domain flow to action/memory loop "
            "to direction channel to split mammalian unit/probe carriers"
        ),
        "entries": entries,
        "caveats": [
            "C. elegans evidence is connectome proxy, not trial behavior.",
            "Drosophila evidence is structural connectome, not trial dynamics.",
            "Zebrafish evidence is discrete laser/activity direction, not continuous movement.",
            "Mouse evidence is an IBL candidate-panel carrier split, not a universal mammalian atlas.",
        ],
    }


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# Cross-species action carrier invariant",
        "",
        "This gate reads existing species reports and asks whether action output is repeatedly carried by a restricted target-linked channel.",
        "",
        "## summary",
        "",
        f"- passed: `{result['passed']}`",
        f"- passed stages: {result['passed_count']}/{result['total']}",
        f"- invariant: {result['invariant']}",
        "",
        "## stage evidence",
        "",
        "| stage | carrier | status | key evidence | passed |",
        "|---|---|---|---|---|",
    ]
    for entry in result["entries"]:
        evidence = entry["evidence"]
        if entry["stage"] == "C. elegans":
            key = (
                f"{evidence['passed_weighted_stages']}/{evidence['stages']} weighted stages; "
                f"mean matched/wrong {evidence['mean_matched_wrong']:.6f}"
            )
        elif entry["stage"] == "Drosophila adult":
            key = (
                f"memory/action loop observed/random {evidence['observed_random_mean']:.6f}; "
                f"p {evidence['loop_p']:.6f}"
            )
        elif entry["stage"] == "Zebrafish":
            key = (
                f"behavior effect ratio {evidence['behavior_effect_ratio']:.6f}; "
                f"activity AUC {evidence['activity_auc']:.6f}"
            )
        else:
            key = f"{evidence['speed_pattern']}; {evidence['wheel_pattern']}"
        lines.append(
            f"| {entry['stage']} | {entry['carrier']} | `{entry['status']}` | {key} | `{entry['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in result["caveats"])
    lines.extend(
        [
            "",
            "## equation update",
            "",
            "$$",
            "\\boxed{",
            "\\mathcal A_{\\mathrm{carrier}}",
            ":",
            "d\\mapsto C_d,",
            "\\qquad",
            "y_{d,t}=g_d(C_d(t),X_t,\\hat H_t)",
            "}",
            "$$",
            "",
            "The invariant is not a fixed anatomical location. It is a rule: action variables become readable when the model respects the stage-specific carrier.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = build_result()
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report_md.write_text(build_report(result), encoding="utf-8")
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main() -> None:
    result = run(build_argparser().parse_args())
    print(json.dumps({"passed": result["passed"], "passed_count": result["passed_count"]}, indent=2))


if __name__ == "__main__":
    main()
