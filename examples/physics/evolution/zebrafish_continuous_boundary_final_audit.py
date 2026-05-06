"""Final boundary audit for zebrafish continuous movement decoding.

This consolidates the existing zebrafish alignment/continuous-decoding reports.
It does not retry arbitrary alignments.  It asks whether the current local files
contain enough certified bridge information to promote the weak candidate speed
signal to a final continuous movement gate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
ALIGNMENT_REPORT = BASE_DIR / "zebrafish_continuous_alignment_audit_report.md"
E2_LR_REPORT = BASE_DIR / "zebrafish_e2_lr_alignment_probe_report.md"
CANDIDATE_REPORT = BASE_DIR / "zebrafish_candidate_continuous_decoding_report.md"
SUPPLEMENTARY_REPORT = BASE_DIR / "zebrafish_supplementary_continuous_closure_audit_report.md"
RESULT_JSON = BASE_DIR / "zebrafish_continuous_boundary_final_audit_results.json"
REPORT_MD = BASE_DIR / "zebrafish_continuous_boundary_final_audit_report.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def require_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return float(match.group(1))


def require_int_pair(pattern: str, text: str) -> tuple[int, int]:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing pattern: {pattern}")
    return int(match.group(1)), int(match.group(2))


def parse_candidate_target(text: str, target: str) -> dict[str, Any]:
    pattern = (
        rf"\| {target} \| ([0-9-]+) \| ([0-9.\-]+) \| ([0-9.\-]+) "
        rf"\| ([0-9.\-]+) \| (True|False) \|"
    )
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing candidate target row: {target}")
    return {
        "target": target,
        "best_lag_e2_frames": int(match.group(1)),
        "r2": float(match.group(2)),
        "mse_over_base": float(match.group(3)),
        "shift_p": float(match.group(4)),
        "candidate": match.group(5) == "True",
    }


def build_result() -> dict[str, Any]:
    alignment = read(ALIGNMENT_REPORT)
    e2_lr = read(E2_LR_REPORT)
    candidate = read(CANDIDATE_REPORT)
    supplementary = read(SUPPLEMENTARY_REPORT)
    laser_matches, laser_total = require_int_pair(r"- laser-schedule matches: (\d+) / (\d+)", e2_lr)
    certified_matches, certified_total = require_int_pair(
        r"- timestamp-certified alignments: (\d+) / (\d+)", e2_lr
    )
    inferred_matches, inferred_total = require_int_pair(
        r"- candidate inferred alignments: (\d+) / (\d+)", e2_lr
    )
    speed = parse_candidate_target(candidate, "speed")
    turn = parse_candidate_target(candidate, "turn")
    certified_ready = (
        require_bool(r"- 현재 partial만으로 continuous movement decoding 가능: (True|False)", alignment)
        or require_bool(r"- certified continuous decoding ready: (True|False)", e2_lr)
        or require_bool(r"- timestamp-certified continuous ready: `(True|False)`", supplementary)
    )
    result = {
        "gate": "zebrafish_continuous_boundary_final_audit",
        "timestamp_certified_continuous_ready": bool(certified_ready),
        "verdict": "data_boundary" if not certified_ready else "ready",
        "discrete_activity_behavior_supported": {
            "activity_to_behavior_frame_possible": require_bool(
                r"- activity -> behavior-frame gate 가능: (True|False)", alignment
            ),
            "activity_to_direction_possible": require_bool(
                r"- activity -> direction gate 가능: (True|False)", alignment
            ),
        },
        "alignment_evidence": {
            "e2_neural_matrix": require_bool(r"- e2 neural matrix 있음: (True|False)", alignment),
            "behavior_bout_frame_label": require_bool(
                r"- behavior bout frame label 있음: (True|False)", alignment
            ),
            "stage_tracking_txt": require_bool(
                r"- stage/head/yolk tracking txt 있음: (True|False)", alignment
            ),
            "neural_mat_has_coordinates": require_bool(
                r"- neural mat 안에 stage/head/tail 좌표 있음: (True|False)", alignment
            ),
            "neural_mat_has_e2_timestamp": require_bool(
                r"- neural mat 안에 e2 column별 absolute timestamp 있음: (True|False)",
                alignment,
            ),
            "laser_schedule_matches": laser_matches,
            "laser_schedule_total": laser_total,
            "timestamp_certified_alignments": certified_matches,
            "timestamp_certified_total": certified_total,
            "candidate_inferred_alignments": inferred_matches,
            "candidate_inferred_total": inferred_total,
            "supplementary_has_e2_timestamp_variable": require_bool(
                r"\| has e2 timestamp variable \| (True|False) \|", supplementary
            ),
            "supplementary_has_e2_resampled_behavior": require_bool(
                r"\| has e2-resampled behavior \| (True|False) \|", supplementary
            ),
        },
        "candidate_decoding": {
            "status": re.search(r"- status: ([^\n]+)", candidate).group(1).strip(),
            "speed": speed,
            "turn": turn,
            "final_continuous_gate_pass": require_bool(
                r"- final continuous gate pass: (True|False)", candidate
            ),
        },
        "interpretation": (
            "Discrete zebrafish activity-behavior links are supported.  The inferred "
            "alignment has a weak speed candidate, but there are zero timestamp-certified "
            "e2-to-tracking alignments and no e2-resampled behavior trace, so continuous "
            "speed/turn/heading remains a data boundary."
        ),
    }
    return result


def build_report(result: dict[str, Any]) -> str:
    align = result["alignment_evidence"]
    candidate = result["candidate_decoding"]
    speed = candidate["speed"]
    turn = candidate["turn"]
    lines = [
        "# Zebrafish continuous boundary final audit",
        "",
        "This consolidates the existing continuous-alignment probes into one final boundary verdict.",
        "",
        "## verdict",
        "",
        f"- timestamp-certified continuous ready: `{result['timestamp_certified_continuous_ready']}`",
        f"- verdict: `{result['verdict']}`",
        f"- final continuous gate pass: `{candidate['final_continuous_gate_pass']}`",
        "",
        "## supported discrete bridges",
        "",
        "| bridge | possible |",
        "|---|---|",
    ]
    for key, value in result["discrete_activity_behavior_supported"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## alignment evidence",
            "",
            "| item | value |",
            "|---|---:|",
            f"| e2 neural matrix | {align['e2_neural_matrix']} |",
            f"| behavior bout frame label | {align['behavior_bout_frame_label']} |",
            f"| stage tracking txt | {align['stage_tracking_txt']} |",
            f"| neural mat has coordinates | {align['neural_mat_has_coordinates']} |",
            f"| neural mat has e2 timestamp | {align['neural_mat_has_e2_timestamp']} |",
            f"| laser-schedule matches | {align['laser_schedule_matches']}/{align['laser_schedule_total']} |",
            f"| timestamp-certified alignments | {align['timestamp_certified_alignments']}/{align['timestamp_certified_total']} |",
            f"| candidate inferred alignments | {align['candidate_inferred_alignments']}/{align['candidate_inferred_total']} |",
            f"| supplementary has e2 timestamp variable | {align['supplementary_has_e2_timestamp_variable']} |",
            f"| supplementary has e2-resampled behavior | {align['supplementary_has_e2_resampled_behavior']} |",
            "",
            "## candidate inferred decoding",
            "",
            "| target | best lag e2 frames | R2 | mse/base | shift p | candidate |",
            "|---|---:|---:|---:|---:|---|",
            f"| speed | {speed['best_lag_e2_frames']} | {speed['r2']:.6f} | {speed['mse_over_base']:.6f} | {speed['shift_p']:.6f} | `{speed['candidate']}` |",
            f"| turn | {turn['best_lag_e2_frames']} | {turn['r2']:.6f} | {turn['mse_over_base']:.6f} | {turn['shift_p']:.6f} | `{turn['candidate']}` |",
            "",
            "## interpretation",
            "",
            "- Activity-to-bout-frame and activity-to-direction gates remain supported.",
            "- The inferred alignment gives a weak speed candidate, but it is not timestamp-certified.",
            "- Turn is not supported even under the inferred alignment.",
            "- Continuous movement decoding therefore remains blocked by missing e2 timestamp or e2-resampled speed/turn/heading trace.",
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
                "verdict": result["verdict"],
                "timestamp_certified_continuous_ready": result[
                    "timestamp_certified_continuous_ready"
                ],
                "final_continuous_gate_pass": result["candidate_decoding"][
                    "final_continuous_gate_pass"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
