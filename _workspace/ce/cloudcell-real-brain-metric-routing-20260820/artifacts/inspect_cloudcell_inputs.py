"""Read-only eligibility audit for the locally cached CloudCell recordings.

This script never extracts or mutates source data.  It hashes the three source
archives and verifies the released ``heatDataMS.mat`` schema, recording-local
clock alignment, signal class, and unit/time dimensions for every selected
recording.  The generated JSON is an input receipt, not an empirical result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data" / "external" / "cloudcell"

DATASETS = (
    {
        "name": "AML310",
        "archive": "AML310_moving.tar.gz",
        "extracted_root": "AKS297.51_moving",
        "log": "AKS297.51_moving_datasets.txt",
        "signal_class": "gcamp",
        "expected_sha256": "144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a",
    },
    {
        "name": "AML32",
        "archive": "AML32_moving.tar.gz",
        "extracted_root": "AML32_moving",
        "log": "AML32_moving_datasets.txt",
        "signal_class": "gcamp",
        "expected_sha256": "6b71a6ba1a5d2f1ef3bf9661e845e1e52634bae217fc0c2630a83fca07daed63",
    },
    {
        "name": "AML18",
        "archive": "AML18_moving.tar.gz",
        "extracted_root": "AML18_moving",
        "log": "AML18_moving_datasets.txt",
        "signal_class": "gfp_control",
        "expected_sha256": "588d7666f4e8afebad1ab9b8483244a6de0303251d862425522c2b8dd78bbd82",
    },
)

REQUIRED_VARIABLES = {
    "rRaw",
    "gRaw",
    "Ratio2",
    "acorr",
    "cgIdx",
    "cgIdxRev",
    "XYZcoord",
    "hasPointsTime",
    "clTime",
    "behavior",
}
BEHAVIOR_FIELDS = {"ethogram", "x_pos", "y_pos", "v", "pc1_2", "pc_3"}
LEADING_GUARD = 12
HISTORY_VOLUMES = 6
FUTURE_VOLUMES = 6
SPLIT_EMBARGO = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_rows(log_path: Path) -> list[tuple[str, int | None]]:
    rows: list[tuple[str, int | None]] = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        rows.append((fields[0], int(fields[1]) if len(fields) > 1 else None))
    return rows


def _clock_audit(values: np.ndarray, *, leading_guard: int = LEADING_GUARD) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    differences = np.diff(flat)
    positive = differences[differences > 0]
    median_dt = float(np.median(positive)) if positive.size else float("nan")
    nonincreasing = np.flatnonzero(differences <= 0)
    gaps = np.flatnonzero(differences > 3.0 * median_dt) if np.isfinite(median_dt) else np.array([])
    guarded = flat[leading_guard:]
    return {
        "finite": bool(np.isfinite(flat).all()),
        "strict_after_leading_guard": bool(
            guarded.size > 1 and np.isfinite(guarded).all() and (np.diff(guarded) > 0).all()
        ),
        "leading_guard_volumes": leading_guard,
        "nonincreasing_count_full": int(nonincreasing.size),
        "nonincreasing_indices_full": nonincreasing.astype(int).tolist(),
        "large_gap_count_full": int(gaps.size),
        "large_gap_indices_full": gaps.astype(int).tolist(),
        "median_positive_dt_seconds": median_dt,
        "max_dt_seconds": float(np.max(differences)),
    }


def _admissible_windows(neural_time: np.ndarray, usable_t: int) -> dict[str, Any]:
    """Freeze clock-only anchor indices for the later empirical preregistration."""

    time = np.asarray(neural_time, dtype=np.float64).reshape(-1)[:usable_t]
    differences = np.diff(time)
    positive = differences[differences > 0]
    if positive.size == 0:
        raise ValueError("clock has no positive intervals")
    gap_threshold = 3.0 * float(np.median(positive))

    analysis_span = usable_t - LEADING_GUARD
    train_boundary = LEADING_GUARD + int(np.floor(0.60 * analysis_span))
    validation_boundary = LEADING_GUARD + int(np.floor(0.80 * analysis_span))
    split_ranges = {
        "train": (LEADING_GUARD, train_boundary - SPLIT_EMBARGO),
        "validation": (train_boundary + SPLIT_EMBARGO, validation_boundary - SPLIT_EMBARGO),
        "test": (validation_boundary + SPLIT_EMBARGO, usable_t),
    }
    anchors: dict[str, list[int]] = {name: [] for name in split_ranges}
    excluded_for_clock = 0
    for anchor in range(LEADING_GUARD + HISTORY_VOLUMES, usable_t - FUTURE_VOLUMES):
        window_start = anchor - HISTORY_VOLUMES
        window_stop = anchor + FUTURE_VOLUMES + 1
        window_differences = np.diff(time[window_start:window_stop])
        clock_valid = bool(
            np.isfinite(window_differences).all()
            and (window_differences > 0).all()
            and (window_differences <= gap_threshold).all()
        )
        if not clock_valid:
            excluded_for_clock += 1
            continue
        for split, (split_start, split_stop) in split_ranges.items():
            if window_start >= split_start and window_stop <= split_stop:
                anchors[split].append(anchor)
                break

    counts = {name: len(indices) for name, indices in anchors.items()}
    return {
        "history_volumes": HISTORY_VOLUMES,
        "future_volumes": FUTURE_VOLUMES,
        "leading_guard_volumes": LEADING_GUARD,
        "split_fractions": [0.60, 0.20, 0.20],
        "split_embargo_volumes_each_side": SPLIT_EMBARGO,
        "gap_threshold_multiple_of_median_positive_dt": 3.0,
        "gap_threshold_seconds": gap_threshold,
        "train_boundary_index": train_boundary,
        "validation_boundary_index": validation_boundary,
        "split_ranges_half_open": {
            name: [start, stop] for name, (start, stop) in split_ranges.items()
        },
        "excluded_anchor_count_for_clock": excluded_for_clock,
        "admissible_anchor_counts": counts,
        "admissible_anchor_indices": anchors,
        "all_splits_nonempty": all(count > 0 for count in counts.values()),
        "scope": "clock_only; finite neural/output filtering remains mandatory",
    }


def _recording_audit(mat_path: Path, recording: str, cut_volume: int | None) -> dict[str, Any]:
    payload = loadmat(
        mat_path,
        variable_names=sorted(REQUIRED_VARIABLES),
        struct_as_record=True,
        squeeze_me=False,
    )
    present = set(payload) - {"__header__", "__version__", "__globals__"}
    missing = sorted(REQUIRED_VARIABLES - present)
    if missing:
        raise ValueError(f"{recording}: missing variables {missing}")

    r_raw = np.asarray(payload["rRaw"])
    g_raw = np.asarray(payload["gRaw"])
    ratio = np.asarray(payload["Ratio2"])
    xyz = np.asarray(payload["XYZcoord"])
    acorr = np.asarray(payload["acorr"])
    neural_time = np.asarray(payload["hasPointsTime"]).reshape(-1)
    centerline_time = np.asarray(payload["clTime"]).reshape(-1)
    behavior = payload["behavior"]

    if r_raw.ndim != 2:
        raise ValueError(f"{recording}: rRaw is not N x T")
    neurons, full_t = r_raw.shape
    behavior_names = set(behavior.dtype.names or ())
    if behavior.shape != (1, 1) or behavior_names != BEHAVIOR_FIELDS:
        raise ValueError(f"{recording}: unexpected behavior schema {sorted(behavior_names)}")

    behavior_shapes = {
        name: list(np.asarray(behavior[0, 0][name]).shape) for name in sorted(BEHAVIOR_FIELDS)
    }
    scalar_behavior_aligned = all(
        behavior_shapes[name] == [full_t, 1]
        for name in ("ethogram", "x_pos", "y_pos", "v", "pc_3")
    )
    pc_aligned = behavior_shapes["pc1_2"] == [full_t, 2]
    matrices_aligned = (
        g_raw.shape == (neurons, full_t)
        and ratio.shape == (neurons, full_t)
        and xyz.shape == (neurons, 3)
        and acorr.shape == (neurons, neurons)
        and neural_time.size == full_t
    )
    usable_t = min(full_t, cut_volume + 1) if cut_volume is not None else full_t
    finite_both = np.isfinite(r_raw[:, :usable_t]) & np.isfinite(g_raw[:, :usable_t])
    majority_valid_t = int((finite_both.mean(axis=0) > 0.5).sum())
    units_75 = int((finite_both.mean(axis=1) >= 0.75).sum())

    neural_clock = _clock_audit(neural_time)
    centerline_clock = _clock_audit(centerline_time, leading_guard=0)
    window_policy = _admissible_windows(neural_time, usable_t)
    checks = {
        "required_variables": not missing,
        "matrix_and_clock_shapes": bool(matrices_aligned),
        "behavior_shapes": bool(scalar_behavior_aligned and pc_aligned),
        "neural_clock_finite_and_strict_after_guard": bool(
            neural_clock["finite"] and neural_clock["strict_after_leading_guard"]
        ),
        "centerline_clock_strictly_increasing": bool(
            centerline_clock["finite"] and centerline_clock["strict_after_leading_guard"]
        ),
        "acorr_finite_symmetric": bool(
            np.isfinite(acorr).all() and np.allclose(acorr, acorr.T, atol=1e-8, rtol=0.0)
        ),
        "clock_window_splits_nonempty": bool(window_policy["all_splits_nonempty"]),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"{recording}: failed checks {failed}")

    return {
        "recording": recording,
        "mat_path": mat_path.relative_to(REPO_ROOT).as_posix(),
        "neurons": neurons,
        "full_timepoints": full_t,
        "cut_volume_zero_based": cut_volume,
        "usable_timepoints": usable_t,
        "majority_valid_timepoints": majority_valid_t,
        "units_at_least_75pct_finite": units_75,
        "finite_xyz_rows": int(np.isfinite(xyz).all(axis=1).sum()),
        "neural_clock": neural_clock,
        "centerline_clock": centerline_clock,
        "clock_window_policy": window_policy,
        "behavior_shapes": behavior_shapes,
        "checks": checks,
    }


def inspect() -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    all_recordings: list[dict[str, Any]] = []
    for spec in DATASETS:
        archive = DATA_ROOT / spec["archive"]
        extracted = DATA_ROOT / "extracted" / spec["extracted_root"]
        log_path = extracted / spec["log"]
        archive_sha = _sha256(archive)
        if archive_sha != spec["expected_sha256"]:
            raise ValueError(f"{archive.name}: SHA-256 mismatch")

        recordings = []
        for recording, cut_volume in _dataset_rows(log_path):
            mat_path = extracted / f"{recording}_MS" / "heatDataMS.mat"
            if not mat_path.is_file():
                raise FileNotFoundError(mat_path)
            row = _recording_audit(mat_path, recording, cut_volume)
            row["dataset"] = spec["name"]
            row["signal_class"] = spec["signal_class"]
            recordings.append(row)
            all_recordings.append(row)

        datasets.append(
            {
                "name": spec["name"],
                "signal_class": spec["signal_class"],
                "archive_path": archive.relative_to(REPO_ROOT).as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": archive_sha,
                "extracted_root": extracted.relative_to(REPO_ROOT).as_posix(),
                "recording_count": len(recordings),
                "recordings": recordings,
            }
        )

    gcamp = [row for row in all_recordings if row["signal_class"] == "gcamp"]
    gfp = [row for row in all_recordings if row["signal_class"] == "gfp_control"]
    return {
        "schema": "clarus.cloudcell.input-audit.v1",
        "status": "PASS_INPUT_SCHEMA",
        "scope": "read_only_input_receipt_not_empirical_result",
        "dataset_count": len(datasets),
        "recording_count": len(all_recordings),
        "gcamp_recording_count": len(gcamp),
        "gfp_control_recording_count": len(gfp),
        "all_recording_checks_pass": all(
            all(row["checks"].values()) for row in all_recordings
        ),
        "datasets": datasets,
        "claim_boundary": {
            "output_fisher_input": "PASS_INPUT",
            "anatomical_source_target": "BLOCKED_SOURCE_TARGET_DEFINITION",
            "causal_routing": "BLOCKED_INTERVENTION",
        },
        "window_policy": {
            "leading_guard_volumes": LEADING_GUARD,
            "history_volumes": HISTORY_VOLUMES,
            "future_volumes": FUTURE_VOLUMES,
            "chronological_split": [0.60, 0.20, 0.20],
            "split_embargo_volumes_each_side": SPLIT_EMBARGO,
            "gap_rule": "exclude anchors whose full history/future window has dt<=0 or dt>3*median_positive_dt",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
