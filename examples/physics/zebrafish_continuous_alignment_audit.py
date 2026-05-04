"""Audit whether current partial zebrafish files support continuous decoding.

The desired final gate is:

    neural activity frame -> continuous speed / heading / turn angle

The partial Figshare chunks already support several gates, but continuous
decoding requires a shared per-frame alignment between neural activity frames
and behavior tracking frames. This script inspects the downloaded partial files
and records what is present and what is missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from zebrafish_freely_swimming_activity_gate import load_mat_numeric


BASE = Path("data/evolution/zebrafish/freely_swimming")
RESULT_JSON = Path(__file__).with_name("zebrafish_continuous_alignment_audit_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_continuous_alignment_audit_report.md")


def stage_file_summary(path: Path, max_lines: int = 100000) -> dict[str, object]:
    pattern = re.compile(
        r"^(?P<frame>\d+), x: (?P<x>-?\d+), y: (?P<y>-?\d+), detection: (?P<det>\d+), "
        r"head: \[(?P<hx>-?\d+), (?P<hy>-?\d+)\], yolk: \[(?P<yx>-?\d+), (?P<yy>-?\d+)\]"
    )
    frames = []
    xy = []
    head = []
    yolk = []
    with path.open(errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            match = pattern.search(line)
            if not match:
                continue
            frames.append(int(match.group("frame")))
            xy.append((float(match.group("x")), float(match.group("y"))))
            head.append((float(match.group("hx")), float(match.group("hy"))))
            yolk.append((float(match.group("yx")), float(match.group("yy"))))
    if len(frames) < 2:
        return {"path": str(path), "parsed_rows": len(frames)}
    xy_arr = np.asarray(xy, dtype=float)
    head_arr = np.asarray(head, dtype=float)
    yolk_arr = np.asarray(yolk, dtype=float)
    speed = np.linalg.norm(np.diff(xy_arr, axis=0), axis=1)
    heading = np.arctan2(head_arr[:, 1] - yolk_arr[:, 1], head_arr[:, 0] - yolk_arr[:, 0])
    return {
        "path": str(path),
        "parsed_rows": int(len(frames)),
        "frame_min": int(min(frames)),
        "frame_max": int(max(frames)),
        "stage_speed_mean_first_rows": float(np.mean(speed)),
        "stage_speed_nonzero_fraction_first_rows": float(np.mean(speed > 0)),
        "heading_std_first_rows": float(np.std(heading)),
    }


def mat_summary(path: Path) -> dict[str, object]:
    data = load_mat_numeric(path)
    return {
        "path": str(path),
        "keys": sorted(data.keys()),
        "shapes": {key: list(value.shape) for key, value in data.items()},
    }


def audit() -> dict[str, object]:
    figure8_f = BASE / "figure8/f/newPropa2.mat"
    figure8_g = BASE / "figure8/g/Corr_FrameActive3_230830.mat"
    lr_root = BASE / "figure8/c/LR"

    stage_files = sorted(lr_root.rglob("Stage_postion*.txt"))
    timestamp_files = sorted(lr_root.rglob("timestamp.mat"))
    bout_files = sorted(lr_root.rglob("boutInfo.mat"))

    neural_mats = [figure8_f, figure8_g]
    neural_summaries = [mat_summary(path) for path in neural_mats if path.exists()]
    stage_summaries = [stage_file_summary(path, max_lines=50000) for path in stage_files[:3]]
    timestamp_summaries = [mat_summary(path) for path in timestamp_files]

    has_e2 = any("e2" in summary["keys"] for summary in neural_summaries)
    has_frame_bout = any("FrameBout" in summary["keys"] for summary in neural_summaries)
    has_left_right = any(
        "LeftLS" in summary["keys"] and "RightLS" in summary["keys"] for summary in neural_summaries
    )
    has_stage_tracking = len(stage_files) > 0
    has_timestamp = len(timestamp_files) > 0
    neural_has_stage_xy = any(
        any(key.lower() in {"stage", "stagepos", "stageposition", "head", "tail", "yolk", "speed", "heading"} for key in summary["keys"])
        for summary in neural_summaries
    )
    neural_has_absolute_timestamp = any(
        any(key.lower() in {"timestamp", "timestamps", "ts340", "ts50", "time"} for key in summary["keys"])
        for summary in neural_summaries
    )

    verdict = {
        "activity_behavior_frame_gate_possible": bool(has_e2 and has_frame_bout),
        "activity_direction_gate_possible": bool(has_e2 and has_left_right),
        "continuous_decoding_possible_from_current_partial": bool(
            has_e2 and has_stage_tracking and (neural_has_stage_xy or neural_has_absolute_timestamp)
        ),
    }
    return {
        "base": str(BASE),
        "downloaded_neural_mats": neural_summaries,
        "stage_file_count": len(stage_files),
        "timestamp_file_count": len(timestamp_files),
        "bout_info_count": len(bout_files),
        "stage_examples": stage_summaries,
        "timestamp_summaries": timestamp_summaries,
        "presence": {
            "has_e2_neural_matrix": has_e2,
            "has_behavior_frame_labels": has_frame_bout,
            "has_left_right_laser_frame_labels": has_left_right,
            "has_stage_tracking_txt": has_stage_tracking,
            "has_lr_timestamp_files": has_timestamp,
            "neural_mats_have_stage_xy": neural_has_stage_xy,
            "neural_mats_have_absolute_timestamp": neural_has_absolute_timestamp,
        },
        "verdict": verdict,
        "next_required_data": [
            "a per-neural-frame timestamp matching e2 columns",
            "or a stage/head/tail trajectory already resampled to e2 frame indices",
            "or the larger raw chunk that contains both light-field neural frames and synchronized tracking output",
        ],
    }


def write_report(output: dict[str, object], path: Path) -> None:
    presence = output["presence"]  # type: ignore[index]
    verdict = output["verdict"]  # type: ignore[index]
    lines = [
        "# Zebrafish continuous decoding alignment audit",
        "",
        "목표는 neural activity frame으로 speed, heading, turn angle을 직접 예측하는 continuous decoding gate다.",
        "",
        "현재 받은 partial chunk가 그 목표를 지원하는지 점검했다.",
        "",
        "## 현재 가능한 것",
        "",
        f"- e2 neural matrix 있음: {presence['has_e2_neural_matrix']}",
        f"- behavior bout frame label 있음: {presence['has_behavior_frame_labels']}",
        f"- left/right laser frame label 있음: {presence['has_left_right_laser_frame_labels']}",
        f"- stage/head/yolk tracking txt 있음: {presence['has_stage_tracking_txt']}",
        f"- LR 일부 폴더 timestamp.mat 있음: {presence['has_lr_timestamp_files']}",
        "",
        "## 빠진 것",
        "",
        f"- neural mat 안에 stage/head/tail 좌표 있음: {presence['neural_mats_have_stage_xy']}",
        f"- neural mat 안에 e2 column별 absolute timestamp 있음: {presence['neural_mats_have_absolute_timestamp']}",
        "",
        "## 판정",
        "",
        f"- activity -> behavior-frame gate 가능: {verdict['activity_behavior_frame_gate_possible']}",
        f"- activity -> direction gate 가능: {verdict['activity_direction_gate_possible']}",
        f"- 현재 partial만으로 continuous movement decoding 가능: {verdict['continuous_decoding_possible_from_current_partial']}",
        "",
        "## 해석",
        "",
        "- 지금 partial 자료는 neural activity와 discrete behavior labels는 연결한다.",
        "- 하지만 e2의 각 column이 stage tracking의 어느 시간/프레임에 해당하는지 알려주는 per-frame alignment가 없다.",
        "- 그래서 현재 partial만으로 speed, heading, turn angle을 직접 예측하면 임의 정렬이 되어 검증이 무효가 된다.",
        "- 다음에는 더 큰 raw chunk나 e2-frame timestamp가 포함된 파일을 받아야 continuous decoding으로 넘어갈 수 있다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output = audit()
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, REPORT_MD)
    print("Zebrafish continuous decoding alignment audit")
    for key, value in output["verdict"].items():  # type: ignore[index]
        print(f"  {key}={value}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
