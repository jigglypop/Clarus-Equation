"""Candidate continuous zebrafish movement decoding with inferred alignment.

This is not the final timestamp-certified gate. It uses the candidate alignment
found by ``zebrafish_e2_lr_alignment_probe.py``:

    exp/20221016_1556... raw_laser_onset = e2_event + 9339
    stage tracking rows ~= 5 * raw frame

The goal is to see whether this inferred bridge carries any continuous movement
signal before spending effort on the larger raw synchronized bundles. The final
gate still requires an explicit e2 timestamp or e2-resampled behavior trace.
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
DEFAULT_RAW_TXT = (
    BASE
    / "figure8/c/LR/exp/20221016_1556_g8s-lssm-chriR_8dpf"
    / "20221016_1556_g8s-lssm-chriR_8dpf.txt"
)
DEFAULT_STAGE_TXT = (
    BASE
    / "figure8/c/LR/exp/20221016_1556_g8s-lssm-chriR_8dpf"
    / "Stage_postion2022_10_16-15_58_5.txt"
)
RESULT_JSON = Path(__file__).with_name("zebrafish_candidate_continuous_decoding_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_candidate_continuous_decoding_report.md")

STAGE_RE = re.compile(
    r"^(?P<frame>\d+), x: (?P<x>-?\d+), y: (?P<y>-?\d+), detection: (?P<det>\d+), "
    r"head: \[(?P<hx>-?\d+), (?P<hy>-?\d+)\], yolk: \[(?P<yx>-?\d+), (?P<yy>-?\d+)\]"
)


def raw_frame_count(path: Path) -> int:
    count = 0
    with path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith("frameNum:"):
                count = max(count, int(line.split(":", 1)[1]))
    return count


def load_stage(path: Path) -> dict[str, np.ndarray]:
    xy = []
    heading = []
    detection = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = STAGE_RE.search(line)
            if not match:
                continue
            xy.append((float(match.group("x")), float(match.group("y"))))
            detection.append(int(match.group("det")))
            heading.append(
                math.atan2(
                    float(match.group("hy")) - float(match.group("yy")),
                    float(match.group("hx")) - float(match.group("yx")),
                )
            )
    return {
        "xy": np.asarray(xy, dtype=float),
        "heading": np.unwrap(np.asarray(heading, dtype=float)),
        "detection": np.asarray(detection, dtype=int),
    }


def behavior_at_e2_frames(
    stage: dict[str, np.ndarray],
    *,
    e2_count: int,
    raw_offset: int,
    stage_over_raw: float,
    lag: int,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy = stage["xy"]
    heading = stage["heading"]
    detection = stage["detection"]
    raw_frames = np.arange(1, e2_count + 1) + raw_offset
    idx = np.round((raw_frames - 1) * stage_over_raw + lag * stage_over_raw).astype(int)
    idx = np.clip(idx, half_window, len(xy) - half_window - 1)
    speed = np.linalg.norm(xy[idx + half_window] - xy[idx - half_window], axis=1)
    turn = heading[idx + half_window] - heading[idx - half_window]
    valid = detection[idx] == 1
    return np.column_stack([speed, turn]), valid


def lowrank_design(x: np.ndarray, train: np.ndarray, test: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x[train], axis=0, keepdims=True)
    sd = np.std(x[train], axis=0, keepdims=True)
    xz = np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > 0)
    u, s, vh = np.linalg.svd(xz[train], full_matrices=False)
    k = min(rank, len(s))
    train_low = u[:, :k] * s[:k]
    test_low = xz[test] @ vh[:k, :].T
    return (
        np.column_stack([np.ones(train_low.shape[0]), train_low]),
        np.column_stack([np.ones(test_low.shape[0]), test_low]),
    )


def fit_predict(design_train: np.ndarray, design_test: np.ndarray, y_train: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(design_train.shape[1])
    penalty[0, 0] = 0.0
    coeff = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ y_train)
    return design_test @ coeff


def score_targets(
    design_train: np.ndarray,
    design_test: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge: float,
) -> dict[str, object]:
    y_train = y[train]
    y_test = y[test]
    y_mu = np.mean(y_train, axis=0, keepdims=True)
    y_sd = np.std(y_train, axis=0, keepdims=True)
    yz = np.divide(y - y_mu, y_sd, out=np.zeros_like(y), where=y_sd > 0)
    pred = fit_predict(design_train, design_test, yz[train], ridge)
    baseline = np.mean((yz[test] - np.mean(yz[train], axis=0, keepdims=True)) ** 2, axis=0)
    mse = np.mean((yz[test] - pred) ** 2, axis=0)
    ratio = mse / np.maximum(baseline, 1e-12)
    r2 = 1.0 - ratio
    return {
        "speed_mse_over_baseline": float(ratio[0]),
        "turn_mse_over_baseline": float(ratio[1]),
        "speed_r2": float(r2[0]),
        "turn_r2": float(r2[1]),
    }


def evaluate(
    e2_mat: Path,
    raw_txt: Path,
    stage_txt: Path,
    *,
    raw_offset: int,
    half_window: int,
    rank: int,
    ridge: float,
    block_size: int,
    lag_min: int,
    lag_max: int,
    lag_step: int,
) -> dict[str, object]:
    data = load_mat_numeric(e2_mat)
    e2 = data["e2"].astype(float).T
    raw_count = raw_frame_count(raw_txt)
    stage = load_stage(stage_txt)
    stage_over_raw = len(stage["xy"]) / max(raw_count, 1)

    n = e2.shape[0]
    blocks = np.arange(n) // block_size
    test = (blocks % 5) == 0
    train = ~test
    design_train, design_test = lowrank_design(e2, train, test, rank)

    lag_rows = []
    for lag in range(lag_min, lag_max + 1, lag_step):
        y, valid = behavior_at_e2_frames(
            stage,
            e2_count=n,
            raw_offset=raw_offset,
            stage_over_raw=stage_over_raw,
            lag=lag,
            half_window=half_window,
        )
        if not np.all(valid):
            # The current matched interval is fully detected, but keep this
            # branch for robustness if another session is supplied.
            y = y.copy()
            y[~valid] = np.nanmean(y[valid], axis=0)
        row = {"lag_e2_frames": int(lag), **score_targets(design_train, design_test, y, train, test, ridge)}
        lag_rows.append(row)

    best_speed = max(lag_rows, key=lambda row: float(row["speed_r2"]))
    best_turn = max(lag_rows, key=lambda row: float(row["turn_r2"]))

    y_best, valid_best = behavior_at_e2_frames(
        stage,
        e2_count=n,
        raw_offset=raw_offset,
        stage_over_raw=stage_over_raw,
        lag=int(best_speed["lag_e2_frames"]),
        half_window=half_window,
    )
    shifts = [160, 240, 320, 480, 640, 800, 960, 1120, 1280, 1440, 1600, 1920, 2240, 2560]
    shift_rows = []
    for shift in shifts:
        shifted = np.roll(y_best, shift, axis=0)
        shift_rows.append(
            {"shift": shift, **score_targets(design_train, design_test, shifted, train, test, ridge)}
        )
    shift_speed = np.asarray([row["speed_r2"] for row in shift_rows], dtype=float)
    shift_turn = np.asarray([row["turn_r2"] for row in shift_rows], dtype=float)
    speed_p = float((np.sum(shift_speed >= float(best_speed["speed_r2"])) + 1) / (len(shift_speed) + 1))
    turn_p = float((np.sum(shift_turn >= float(best_turn["turn_r2"])) + 1) / (len(shift_turn) + 1))

    speed_candidate = bool(float(best_speed["speed_r2"]) > 0.05 and speed_p < 0.1)
    turn_candidate = bool(float(best_turn["turn_r2"]) > 0.05 and turn_p < 0.1)
    return {
        "dataset": "All-optical interrogation of brain-wide activity in freely swimming larval zebrafish",
        "e2_mat": str(e2_mat),
        "raw_txt": str(raw_txt),
        "stage_txt": str(stage_txt),
        "alignment_status": "candidate_inferred_not_timestamp_certified",
        "raw_offset_e2_to_raw": raw_offset,
        "raw_frame_count": raw_count,
        "stage_line_count": int(len(stage["xy"])),
        "stage_over_raw_frame_ratio": float(stage_over_raw),
        "e2_frame_count": int(n),
        "valid_detection_fraction_at_best_speed_lag": float(np.mean(valid_best)),
        "rank": int(rank),
        "ridge": float(ridge),
        "block_size": int(block_size),
        "lag_rows": lag_rows,
        "best_speed": best_speed,
        "best_turn": best_turn,
        "shift_control_rows_at_best_speed_lag": shift_rows,
        "shift_control_speed_r2_mean": float(np.mean(shift_speed)),
        "shift_control_speed_r2_max": float(np.max(shift_speed)),
        "shift_control_turn_r2_mean": float(np.mean(shift_turn)),
        "shift_control_turn_r2_max": float(np.max(shift_turn)),
        "speed_shift_p_ge_observed": speed_p,
        "turn_shift_p_ge_observed": turn_p,
        "speed_candidate": speed_candidate,
        "turn_candidate": turn_candidate,
        "passed_final_continuous_gate": bool(speed_candidate and turn_candidate),
        "caveat": (
            "Uses inferred raw/e2/stage alignment. Treat as a scouting result, "
            "not a final timestamp-certified movement decoding gate."
        ),
    }


def write_report(output: dict[str, object], path: Path) -> None:
    best_speed = output["best_speed"]  # type: ignore[index]
    best_turn = output["best_turn"]  # type: ignore[index]
    lines = [
        "# Zebrafish candidate continuous decoding gate",
        "",
        "이 gate는 timestamp-certified 최종 검증이 아니라 inferred alignment 위에서 신호가 있는지 보는 정찰 검증이다.",
        "",
        "## Alignment",
        "",
        f"- raw offset e2 -> raw: {output['raw_offset_e2_to_raw']}",
        f"- raw frames: {output['raw_frame_count']}",
        f"- stage rows: {output['stage_line_count']}",
        f"- stage/raw ratio: {output['stage_over_raw_frame_ratio']:.6f}",
        f"- status: {output['alignment_status']}",
        "",
        "## Best Decoding",
        "",
        "| target | best lag e2 frames | R2 | mse/base | shift p | candidate |",
        "|---|---:|---:|---:|---:|---|",
        (
            f"| speed | {best_speed['lag_e2_frames']} | {best_speed['speed_r2']:.6f} | "
            f"{best_speed['speed_mse_over_baseline']:.6f} | "
            f"{output['speed_shift_p_ge_observed']:.6f} | {output['speed_candidate']} |"
        ),
        (
            f"| turn | {best_turn['lag_e2_frames']} | {best_turn['turn_r2']:.6f} | "
            f"{best_turn['turn_mse_over_baseline']:.6f} | "
            f"{output['turn_shift_p_ge_observed']:.6f} | {output['turn_candidate']} |"
        ),
        "",
        f"- final continuous gate pass: {output['passed_final_continuous_gate']}",
        "",
        "## Interpretation",
        "",
        "- speed는 inferred alignment에서 약한 후보 신호가 있다.",
        "- turn/heading은 닫히지 않았다.",
        "- 따라서 현재 partial로는 최종 continuous movement gate를 통과했다고 보지 않는다.",
        "- 다음에는 explicit e2 timestamp 또는 e2-resampled behavior trace가 필요하다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-mat", type=Path, default=DEFAULT_E2_MAT)
    parser.add_argument("--raw-txt", type=Path, default=DEFAULT_RAW_TXT)
    parser.add_argument("--stage-txt", type=Path, default=DEFAULT_STAGE_TXT)
    parser.add_argument("--raw-offset", type=int, default=9339)
    parser.add_argument("--half-window", type=int, default=2)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--block-size", type=int, default=80)
    parser.add_argument("--lag-min", type=int, default=-120)
    parser.add_argument("--lag-max", type=int, default=240)
    parser.add_argument("--lag-step", type=int, default=10)
    args = parser.parse_args()

    output = evaluate(
        args.e2_mat,
        args.raw_txt,
        args.stage_txt,
        raw_offset=args.raw_offset,
        half_window=args.half_window,
        rank=args.rank,
        ridge=args.ridge,
        block_size=args.block_size,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        lag_step=args.lag_step,
    )
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, REPORT_MD)

    best_speed = output["best_speed"]
    best_turn = output["best_turn"]
    print("Zebrafish candidate continuous decoding gate")
    print(f"  alignment={output['alignment_status']}")
    print(
        f"  speed: lag={best_speed['lag_e2_frames']}, "
        f"r2={best_speed['speed_r2']:.6f}, p={output['speed_shift_p_ge_observed']:.6f}, "
        f"candidate={output['speed_candidate']}"
    )
    print(
        f"  turn: lag={best_turn['lag_e2_frames']}, "
        f"r2={best_turn['turn_r2']:.6f}, p={output['turn_shift_p_ge_observed']:.6f}, "
        f"candidate={output['turn_candidate']}"
    )
    print(f"  passed_final_continuous_gate={output['passed_final_continuous_gate']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
