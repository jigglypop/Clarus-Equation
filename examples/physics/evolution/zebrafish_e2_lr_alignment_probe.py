"""Probe whether figure8 e2 neural frames align to LR raw tracking sessions.

The final desired zebrafish gate is continuous movement decoding:

    e2[:, t] -> speed / heading / turn angle at the same time t

The current Figshare partial has two relevant pieces:

- figure8/f,g,S11: processed neural matrix ``e2`` with 3360 columns and laser
  event frames ``LeftLS``/``RightLS``.
- figure8/c/LR: per-fish raw laser sessions with stage/head/yolk tracking,
  raw ``laserOn`` frames, and a few timestamp.mat files.

This script checks the simplest possible bridge: can any LR raw laser-onset
sequence be mapped to the e2 laser-onset sequence by a contiguous affine frame
mapping? If not, continuous decoding remains blocked unless another file gives
an explicit e2 timestamp or resampled behavior trace.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from zebrafish_freely_swimming_activity_gate import load_mat_numeric


BASE = Path("data/evolution/zebrafish/freely_swimming")
DEFAULT_E2_MAT = BASE / "figure8/f/newPropa2.mat"
DEFAULT_LR_ROOT = BASE / "figure8/c/LR"
RESULT_JSON = Path(__file__).with_name("zebrafish_e2_lr_alignment_probe_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_e2_lr_alignment_probe_report.md")


FRAME_RE = re.compile(r"^frameNum:(?P<frame>\d+)")
LASER_RE = re.compile(r"^laserOn:(?P<laser>-?\d+)")


def load_e2_events(path: Path) -> dict[str, object]:
    data = load_mat_numeric(path)
    left = data["LeftLS"].astype(int).ravel()
    right = data["RightLS"].astype(int).ravel()
    events = np.sort(np.concatenate([left, right]))
    return {
        "mat_path": str(path),
        "e2_shape": list(data["e2"].shape),
        "left": left.tolist(),
        "right": right.tolist(),
        "events_sorted": events.tolist(),
        "event_count": int(len(events)),
        "event_gaps": np.diff(events).astype(int).tolist(),
    }


def raw_laser_onsets(path: Path) -> list[int]:
    current_frame: int | None = None
    laser_rows: list[tuple[int, int]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            frame_match = FRAME_RE.match(line)
            if frame_match:
                current_frame = int(frame_match.group("frame"))
                continue
            laser_match = LASER_RE.match(line)
            if laser_match and current_frame is not None:
                laser_rows.append((current_frame, int(laser_match.group("laser"))))
    onsets = []
    previous = 0
    for frame, laser in laser_rows:
        if laser != 0 and previous == 0:
            onsets.append(frame)
        previous = laser
    return onsets


def raw_frame_count(path: Path) -> int:
    count = 0
    with path.open(errors="ignore") as handle:
        for line in handle:
            frame_match = FRAME_RE.match(line)
            if frame_match:
                count = max(count, int(frame_match.group("frame")))
    return count


def fit_affine(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = slope * x + intercept
    residual = y - pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))
    denom = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_e2_frames": rmse,
        "mae_e2_frames": mae,
        "max_abs_e2_frames": max_abs,
        "rmse_over_e2_std": rmse / max(denom, 1e-12),
    }


def best_contiguous_match(raw: list[int], e2_events: list[int]) -> dict[str, object]:
    target = np.asarray(e2_events, dtype=float)
    count = len(target)
    if len(raw) < count:
        return {"possible": False, "reason": "raw onset count shorter than e2 event count"}
    best: dict[str, object] | None = None
    for start in range(0, len(raw) - count + 1):
        window = np.asarray(raw[start : start + count], dtype=float)
        fit = fit_affine(window, target)
        row = {
            "raw_window_start": int(start),
            "raw_window_onsets": window.astype(int).tolist(),
            **fit,
        }
        if best is None or float(row["rmse_e2_frames"]) < float(best["rmse_e2_frames"]):
            best = row
    assert best is not None
    # A permissive threshold: if the mapping is off by more than ~1 s worth of
    # 20 Hz e2 frames, it is not a credible direct alignment for continuous
    # decoding.
    best["credible_under_20_frame_rmse"] = bool(float(best["rmse_e2_frames"]) < 20.0)
    return best


def session_rows(lr_root: Path, e2_events: list[int]) -> list[dict[str, object]]:
    rows = []
    for txt in sorted(lr_root.rglob("*.txt")):
        if txt.name.startswith("Stage_postion"):
            continue
        onsets = raw_laser_onsets(txt)
        raw_count = raw_frame_count(txt)
        match = best_contiguous_match(onsets, e2_events) if onsets else {"possible": False}
        session = txt.parent.relative_to(lr_root)
        stage_files = sorted(txt.parent.glob("Stage_postion*.txt"))
        stage_line_count = 0
        if stage_files:
            with stage_files[0].open(errors="ignore") as handle:
                stage_line_count = sum(1 for _ in handle)
        stage_over_raw = stage_line_count / max(raw_count, 1)
        inferred_raw_stage_ratio = bool(stage_files and 4.8 <= stage_over_raw <= 5.2)
        laser_schedule_match = bool(match.get("credible_under_20_frame_rmse", False))
        timestamp_certified = bool(laser_schedule_match and (txt.parent / "timestamp.mat").exists())
        rows.append(
            {
                "session": str(session),
                "raw_txt": str(txt),
                "raw_frame_count": raw_count,
                "raw_laser_onset_count": len(onsets),
                "raw_laser_onsets_first_30": onsets[:30],
                "has_stage_tracking": bool(stage_files),
                "stage_files": [str(path) for path in stage_files],
                "stage_line_count": stage_line_count,
                "stage_over_raw_frame_ratio": stage_over_raw,
                "inferred_raw_stage_5x_ratio": inferred_raw_stage_ratio,
                "has_timestamp_mat": (txt.parent / "timestamp.mat").exists(),
                "laser_schedule_match": laser_schedule_match,
                "timestamp_certified_alignment": timestamp_certified,
                "candidate_inferred_alignment": bool(laser_schedule_match and inferred_raw_stage_ratio),
                "best_contiguous_affine_match": match,
            }
        )
    return rows


def write_report(output: dict[str, object], path: Path) -> None:
    e2 = output["e2"]  # type: ignore[index]
    rows = output["sessions"]  # type: ignore[index]
    laser_matches = [row for row in rows if row["laser_schedule_match"]]
    timestamp_certified = [row for row in rows if row["timestamp_certified_alignment"]]
    inferred_candidates = [row for row in rows if row["candidate_inferred_alignment"]]
    best = min(
        rows,
        key=lambda row: float(row["best_contiguous_affine_match"].get("rmse_e2_frames", math.inf)),
    )
    best_match = best["best_contiguous_affine_match"]
    lines = [
        "# Zebrafish e2-LR alignment probe",
        "",
        "목표는 `e2[:, t]` neural frame을 같은 시각의 stage/head/yolk movement에 붙일 수 있는지 확인하는 것이다.",
        "",
        "## e2 event sequence",
        "",
        f"- e2 shape: {e2['e2_shape']}",
        f"- event count: {e2['event_count']}",
        f"- sorted events: {e2['events_sorted']}",
        f"- event gaps: {e2['event_gaps']}",
        "",
        "## LR session match summary",
        "",
        "| session | raw frames | raw laser onsets | stage/raw | timestamp.mat | best RMSE e2 frames | laser match | candidate |",
        "|---|---:|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        match = row["best_contiguous_affine_match"]
        rmse = match.get("rmse_e2_frames")
        rmse_text = f"{float(rmse):.3f}" if rmse is not None else "nan"
        lines.append(
            f"| {row['session']} | {row['raw_frame_count']} | "
            f"{row['raw_laser_onset_count']} | "
            f"{float(row['stage_over_raw_frame_ratio']):.3f} | "
            f"{row['has_timestamp_mat']} | {rmse_text} | "
            f"{row['laser_schedule_match']} | {row['candidate_inferred_alignment']} |"
        )
    lines.extend(
        [
            "",
            "## Best attempted match",
            "",
            f"- session: {best['session']}",
            f"- RMSE e2 frames: {float(best_match.get('rmse_e2_frames', math.nan)):.3f}",
            f"- MAE e2 frames: {float(best_match.get('mae_e2_frames', math.nan)):.3f}",
            f"- max abs error e2 frames: {float(best_match.get('max_abs_e2_frames', math.nan)):.3f}",
            f"- slope raw->e2: {float(best_match.get('slope', math.nan)):.6f}",
            f"- laser schedule match: {best['laser_schedule_match']}",
            f"- stage/raw frame ratio: {float(best['stage_over_raw_frame_ratio']):.6f}",
            f"- timestamp certified: {best['timestamp_certified_alignment']}",
            f"- candidate inferred alignment: {best['candidate_inferred_alignment']}",
            "",
            "## Verdict",
            "",
            f"- laser-schedule matches: {len(laser_matches)} / {len(rows)}",
            f"- timestamp-certified alignments: {len(timestamp_certified)} / {len(rows)}",
            f"- candidate inferred alignments: {len(inferred_candidates)} / {len(rows)}",
            f"- certified continuous decoding ready: {output['certified_continuous_decoding_ready']}",
            f"- candidate inferred decoding ready: {output['candidate_inferred_decoding_ready']}",
            "",
            "`exp/20221016_1556...`은 `e2` event sequence와 raw laser schedule이 정확히 맞고 stage/raw frame count도 5배에 가깝다.",
            "하지만 그 session에는 `timestamp.mat`가 없으므로 이것은 candidate inferred alignment이지 final timestamp-certified alignment는 아니다.",
            "최종 결론에는 explicit e2 timestamp, e2-resampled tracking, 또는 raw synchronized bundle verification이 필요하다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-mat", type=Path, default=DEFAULT_E2_MAT)
    parser.add_argument("--lr-root", type=Path, default=DEFAULT_LR_ROOT)
    args = parser.parse_args()

    e2 = load_e2_events(args.e2_mat)
    rows = session_rows(args.lr_root, e2["events_sorted"])  # type: ignore[arg-type]
    laser_matches = [row for row in rows if row["laser_schedule_match"]]
    timestamp_certified = [row for row in rows if row["timestamp_certified_alignment"]]
    inferred_candidates = [row for row in rows if row["candidate_inferred_alignment"]]
    output = {
        "dataset": "All-optical interrogation of brain-wide activity in freely swimming larval zebrafish",
        "question": "Can figure8 e2 neural frames be aligned to LR raw tracking by laser schedule alone?",
        "e2": e2,
        "sessions": rows,
        "laser_schedule_match_count": len(laser_matches),
        "timestamp_certified_alignment_count": len(timestamp_certified),
        "candidate_inferred_alignment_count": len(inferred_candidates),
        "certified_continuous_decoding_ready": bool(len(timestamp_certified) > 0),
        "candidate_inferred_decoding_ready": bool(len(inferred_candidates) > 0),
        "next_required_data": [
            "per-e2-column timestamp",
            "or behavior traces resampled to e2 frame indices",
            "or a raw synchronized neural/tracking bundle for the exact e2 session",
        ],
    }
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, REPORT_MD)

    print("Zebrafish e2-LR alignment probe")
    print(f"  e2_events={e2['event_count']}")
    print(f"  sessions={len(rows)}")
    print(f"  laser_schedule_matches={len(laser_matches)}")
    print(f"  timestamp_certified={len(timestamp_certified)}")
    print(f"  inferred_candidates={len(inferred_candidates)}")
    best = min(
        rows,
        key=lambda row: float(row["best_contiguous_affine_match"].get("rmse_e2_frames", math.inf)),
    )
    best_match = best["best_contiguous_affine_match"]
    print(
        "  best="
        f"{best['session']} rmse={float(best_match.get('rmse_e2_frames', math.nan)):.3f}"
    )
    print(f"  certified_continuous_decoding_ready={bool(len(timestamp_certified) > 0)}")
    print(f"  candidate_inferred_decoding_ready={bool(len(inferred_candidates) > 0)}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
