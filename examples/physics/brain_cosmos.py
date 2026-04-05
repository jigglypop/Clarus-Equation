"""
Reproducible CE brain pilots for MSC, ds004902, and ds000201.

- `msc`: stage-1 `x_a/x_b` estimate from Midnight Scan Club task/rest fMRI
- `ds004902`: `q_sleep` pilot from resting EEG plus deprivation metadata
- `sleepybrain`: joint stage-1 `x_a/x_b/q_sleep` pilot from Sleepy Brain Project I

Running the script without a subcommand preserves the original MSC behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.integrate import trapezoid
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import wilcoxon

ALPHA_S = 0.11789
EPS2 = 0.048647
SIN2_THETA_W = 4 * ALPHA_S ** (4 / 3)
DELTA = SIN2_THETA_W * (1 - SIN2_THETA_W)
D_EFF = 3 + DELTA
P_STAR_DOC = np.array([0.0487, 0.2623, 0.6891], dtype=np.float64)
P_STAR = P_STAR_DOC / P_STAR_DOC.sum()
RHO = D_EFF * EPS2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "brain" / "msc" / "sub-MSC01" / "ses-func01"
DEFAULT_DS004902_ROOT = REPO_ROOT / "data" / "brain" / "ds004902"
DEFAULT_DS000201_ROOT = REPO_ROOT / "data" / "brain" / "ds000201"
DEFAULT_TASK_BOLD = [
    DEFAULT_DATA_ROOT / "sub-MSC01_ses-func01_task-motor_run-01_bold.nii.gz",
    DEFAULT_DATA_ROOT / "sub-MSC01_ses-func01_task-motor_run-02_bold.nii.gz",
]
DEFAULT_TASK_EVENTS = [
    DEFAULT_DATA_ROOT / "sub-MSC01_ses-func01_task-motor_run-01_events.tsv",
    DEFAULT_DATA_ROOT / "sub-MSC01_ses-func01_task-motor_run-02_events.tsv",
]
DEFAULT_REST_BOLD = DEFAULT_DATA_ROOT / "sub-MSC01_ses-func01_task-rest_bold.nii.gz"
DEFAULT_CONTRAST = DEFAULT_DATA_ROOT / "sub-MSC01-motor_contrasts_32k_fsLR.dscalar.nii"
REST_PARCEL_BIN_SIZE = 6
REST_PARCEL_MIN_VOXELS = 24
POSTERIOR_CHANNELS = frozenset({"Pz", "P1", "P2", "PO3", "PO4", "POz", "O1", "Oz", "O2"})
PVT_LAPSE_THRESHOLD_SEC = 0.5
BEHAVIOR_METRICS = [
    ("stanford_sleepiness_scale", "Stanford Sleepiness Scale", "SSS_NS", "SSS_SD"),
    ("pvt_mean_rt_proxy", "PVT mean RT proxy", "PVT_item2_NS", "PVT_item2_SD"),
    ("pvt_lapse_proxy", "PVT lapse proxy", "PVT_item3_NS", "PVT_item3_SD"),
    ("panas_positive", "PANAS positive", "PANAS_P_NS", "PANAS_P_SD"),
    ("panas_negative", "PANAS negative", "PANAS_N_NS", "PANAS_N_SD"),
]


def add_msc_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rest-bold",
        type=Path,
        default=DEFAULT_REST_BOLD,
        help="Raw resting-state 4D NIfTI.",
    )
    parser.add_argument(
        "--task-bold",
        action="append",
        type=Path,
        help="Raw task 4D NIfTI. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--task-events",
        action="append",
        type=Path,
        help="TSV events file aligned with --task-bold. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--contrast-map",
        type=Path,
        default=DEFAULT_CONTRAST,
        help="Motor contrast CIFTI dscalar used for an independent responsive-fraction check.",
    )
    parser.add_argument(
        "--contrast-name",
        default="Allcondition_avg",
        help="Contrast map name inside the dscalar file.",
    )
    parser.add_argument(
        "--hrf-lag-sec",
        type=float,
        default=4.4,
        help="Shift applied to events before building the active task mask.",
    )
    parser.add_argument(
        "--mask-mean-quantile",
        type=float,
        default=0.35,
        help="Lower mean-intensity quantile used to reject obvious non-brain voxels.",
    )
    parser.add_argument(
        "--mask-std-quantile",
        type=float,
        default=0.35,
        help="Lower temporal-std quantile used to reject static voxels.",
    )
    parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=3.0,
        help="MAD multiplier used for responsive-fraction thresholds.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save the result payload as JSON.",
    )


def add_ds004902_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DS004902_ROOT,
        help="Downloaded ds004902 root with participants.tsv and sub-* EEG folders.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="Optional subject ID to include. Repeat to pin a specific subset.",
    )
    parser.add_argument(
        "--subject-limit",
        type=int,
        help="Optional cap on the number of readable complete subjects.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save the result payload as JSON.",
    )


def add_sleepybrain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DS000201_ROOT,
        help="Downloaded ds000201 root with participants.tsv and sub-* MRI folders.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="Optional subject ID to include. Repeat to pin a specific subset.",
    )
    parser.add_argument(
        "--subject-limit",
        type=int,
        help="Optional cap on the number of readable complete subjects.",
    )
    parser.add_argument(
        "--hrf-lag-sec",
        type=float,
        default=4.4,
        help="Shift applied to task events before building the active task mask.",
    )
    parser.add_argument(
        "--mask-mean-quantile",
        type=float,
        default=0.35,
        help="Lower mean-intensity quantile used to reject obvious non-brain voxels.",
    )
    parser.add_argument(
        "--mask-std-quantile",
        type=float,
        default=0.35,
        help="Lower temporal-std quantile used to reject static voxels.",
    )
    parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=3.0,
        help="MAD multiplier used for responsive-fraction thresholds.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to save the result payload as JSON.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["msc"]
    elif argv[0] not in {"msc", "ds004902", "sleepybrain", "ds000201", "-h", "--help"}:
        argv = ["msc", *argv]

    parser = argparse.ArgumentParser(description="Reproducible CE brain pilots.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    msc_parser = subparsers.add_parser(
        "msc",
        help="Midnight Scan Club stage-1 x_a/x_b pilot.",
        description="Estimate stage-1 CE brain proxies from one MSC subject/session.",
    )
    add_msc_arguments(msc_parser)

    ds004902_parser = subparsers.add_parser(
        "ds004902",
        help="Sleep deprivation q_sleep pilot from resting EEG.",
        description="Estimate q_sleep proxies from the ds004902 sleep-deprivation EEG dataset.",
    )
    add_ds004902_arguments(ds004902_parser)

    sleepybrain_parser = subparsers.add_parser(
        "sleepybrain",
        aliases=["ds000201"],
        help="Sleepy Brain stage-1 x_a/x_b/q_sleep pilot.",
        description="Estimate joint stage-1 CE brain proxies from the ds000201 sleep-restriction fMRI dataset.",
    )
    add_sleepybrain_arguments(sleepybrain_parser)

    return parser.parse_args(argv)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return path


def write_json(payload: dict[str, object], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def infer_task_name(path: Path) -> str:
    for token in path.name.split("_"):
        if token.startswith("task-"):
            return token.removeprefix("task-")
    return path.stem


def load_events(path: Path) -> list[dict[str, str]]:
    return load_tsv_rows(path)


def event_duration_sec(row: dict[str, str], fallback_sec: float) -> float:
    duration = parse_optional_float(row.get("duration"))
    if duration is not None and duration > 0:
        return duration
    response_time = parse_optional_float(row.get("response_time"))
    if response_time is not None and response_time > 0:
        return response_time
    return fallback_sec


def event_is_active(task_name: str, row: dict[str, str]) -> bool:
    trial_type = (row.get("trial_type") or "").strip().lower()
    event_type = (row.get("event_type") or "").strip().lower()

    if task_name == "sleepiness":
        return "response" in trial_type or "response" in event_type

    baseline_tokens = ("iti", "fixation", "rest")
    if any(token in trial_type for token in baseline_tokens):
        return False
    if any(token in event_type for token in baseline_tokens):
        return False
    return bool(trial_type or event_type)


def build_task_mask(
    n_scans: int,
    tr: float,
    event_rows: list[dict[str, str]],
    lag_sec: float,
    task_name: str,
) -> np.ndarray:
    active = np.zeros(n_scans, dtype=bool)
    for row in event_rows:
        if not event_is_active(task_name, row):
            continue
        onset_text = (row.get("onset") or "").strip()
        if not onset_text:
            continue
        onset_sec = float(onset_text)
        duration_sec = event_duration_sec(row, fallback_sec=tr)
        start = max(0, int(round((onset_sec + lag_sec) / tr)))
        stop = max(start + 1, int(round((onset_sec + lag_sec + duration_sec) / tr)))
        stop = min(n_scans, stop)
        if stop > start:
            active[start:stop] = True
    if not np.any(active):
        raise ValueError(f"Task mask is empty for task={task_name}. Check events and lag.")
    return active


def load_4d_nifti(path: Path) -> tuple[np.ndarray, float]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    tr = float(image.header.get_zooms()[3])
    return data, tr


def build_brain_mask(
    rest_data: np.ndarray,
    mean_quantile: float,
    std_quantile: float,
) -> tuple[np.ndarray, float, float]:
    mean_img = rest_data.mean(axis=-1)
    std_img = rest_data.std(axis=-1)
    positive = mean_img > 0
    if not np.any(positive):
        raise ValueError("Rest image has no positive voxels.")
    mean_threshold = float(np.quantile(mean_img[positive], mean_quantile))
    std_threshold = float(np.quantile(std_img[positive], std_quantile))
    mask = positive & (mean_img >= mean_threshold) & (std_img >= std_threshold)
    if not np.any(mask):
        raise ValueError("Brain mask is empty. Relax the mask quantiles.")
    return mask, mean_threshold, std_threshold


def masked_timeseries(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return data[mask].reshape(-1, data.shape[-1]).astype(np.float32, copy=False)


def percent_signal_change(data_2d: np.ndarray) -> np.ndarray:
    baseline = data_2d.mean(axis=1, keepdims=True)
    safe_baseline = np.where(np.abs(baseline) > 1e-3, baseline, 1.0)
    return 100.0 * (data_2d - baseline) / safe_baseline


def robust_threshold_fraction(values: np.ndarray, mad_multiplier: float) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("No finite values available for thresholding.")
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    threshold = median + mad_multiplier * max(mad, 1e-8)
    fraction = float(np.mean(valid > threshold))
    return threshold, fraction


def rest_background_proxy(
    rest_bold: Path,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    rest_data, tr = load_4d_nifti(rest_bold)
    rest_matrix = masked_timeseries(rest_data, mask)
    del rest_data
    rest_psc = percent_signal_change(rest_matrix)
    del rest_matrix
    rest_rms = np.sqrt(np.mean(rest_psc**2, axis=1))
    return rest_rms.astype(np.float32, copy=False), tr


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Correlation inputs must have the same shape.")
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 2:
        return 0.0
    left_valid = np.asarray(left[mask], dtype=np.float64)
    right_valid = np.asarray(right[mask], dtype=np.float64)
    left_centered = left_valid - left_valid.mean()
    right_centered = right_valid - right_valid.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def global_connectivity_map(data_2d: np.ndarray) -> np.ndarray:
    centered = data_2d - data_2d.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    normalized = centered / np.where(scale > 1e-6, scale, 1.0)
    global_signal = normalized.mean(axis=0)
    global_centered = global_signal - global_signal.mean()
    global_scale = float(global_centered.std())
    if global_scale <= 1e-6:
        return np.zeros(normalized.shape[0], dtype=np.float32)
    global_normalized = global_centered / global_scale
    connectivity = np.mean(normalized * global_normalized[None, :], axis=1)
    return connectivity.astype(np.float32, copy=False)


def rest_background_stability_proxy(
    rest_bold: Path,
    mask: np.ndarray,
) -> dict[str, float]:
    rest_data, tr = load_4d_nifti(rest_bold)
    rest_matrix = masked_timeseries(rest_data, mask)
    del rest_data
    rest_psc = percent_signal_change(rest_matrix)
    del rest_matrix
    midpoint = rest_psc.shape[1] // 2
    if midpoint == 0 or midpoint == rest_psc.shape[1]:
        raise ValueError(f"Rest run is too short for split-half stability: {rest_bold}")
    first_half_rms = np.sqrt(np.mean(rest_psc[:, :midpoint] ** 2, axis=1))
    second_half_rms = np.sqrt(np.mean(rest_psc[:, midpoint:] ** 2, axis=1))
    split_half_r = safe_correlation(first_half_rms, second_half_rms)
    first_half_connectivity = global_connectivity_map(rest_psc[:, :midpoint])
    second_half_connectivity = global_connectivity_map(rest_psc[:, midpoint:])
    split_half_connectivity_r = safe_correlation(first_half_connectivity, second_half_connectivity)
    return {
        "tr_sec": tr,
        "split_half_spatial_r": split_half_r,
        "split_half_spatial_r_clipped": max(split_half_r, 0.0),
        "split_half_global_connectivity_r": split_half_connectivity_r,
        "split_half_global_connectivity_r_clipped": max(split_half_connectivity_r, 0.0),
        "first_half_rms_mean_psc": float(first_half_rms.mean()),
        "second_half_rms_mean_psc": float(second_half_rms.mean()),
        "first_half_global_connectivity_mean": float(first_half_connectivity.mean()),
        "second_half_global_connectivity_mean": float(second_half_connectivity.mean()),
    }


def coarse_parcel_timeseries(
    data_2d: np.ndarray,
    mask: np.ndarray,
    bin_size: int,
    min_voxels: int,
) -> tuple[np.ndarray, int]:
    coords = np.column_stack(np.nonzero(mask))
    bins = coords // bin_size
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    keep_ids = np.flatnonzero(counts >= min_voxels)
    if keep_ids.size < 6:
        raise ValueError("Not enough populated parcels for rest network segregation.")
    parcel_timeseries = np.empty((keep_ids.size, data_2d.shape[1]), dtype=np.float32)
    for out_index, parcel_id in enumerate(keep_ids):
        parcel_timeseries[out_index] = data_2d[inverse == parcel_id].mean(axis=0)
    return parcel_timeseries, int(keep_ids.size)


def connectivity_correlation_matrix(data_2d: np.ndarray) -> np.ndarray:
    centered = data_2d - data_2d.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    normalized = centered / np.where(scale > 1e-6, scale, 1.0)
    return np.clip(np.corrcoef(normalized), -1.0, 1.0)


def connectivity_partition_labels(correlation_matrix: np.ndarray) -> np.ndarray:
    if correlation_matrix.ndim != 2 or correlation_matrix.shape[0] < 6:
        raise ValueError("Connectivity partition requires at least six parcels.")
    _, eigenvectors = np.linalg.eigh(correlation_matrix)
    partition_axis = eigenvectors[:, -2] if correlation_matrix.shape[0] >= 2 else eigenvectors[:, -1]
    labels = partition_axis >= np.median(partition_axis)
    if np.all(labels) or np.all(~labels):
        raise ValueError("Connectivity partition collapsed to one module.")
    return labels


def rest_network_segregation_proxy(
    rest_bold: Path,
    mask: np.ndarray,
    bin_size: int = REST_PARCEL_BIN_SIZE,
    min_voxels: int = REST_PARCEL_MIN_VOXELS,
) -> dict[str, float | int]:
    rest_data, tr = load_4d_nifti(rest_bold)
    rest_matrix = masked_timeseries(rest_data, mask)
    del rest_data
    rest_psc = percent_signal_change(rest_matrix)
    del rest_matrix
    parcel_timeseries, parcel_count = coarse_parcel_timeseries(rest_psc, mask, bin_size, min_voxels)
    correlation_matrix = connectivity_correlation_matrix(parcel_timeseries)
    labels = connectivity_partition_labels(correlation_matrix)
    upper = np.triu_indices(correlation_matrix.shape[0], 1)
    same_module = labels[upper[0]] == labels[upper[1]]
    within = correlation_matrix[upper][same_module]
    between = correlation_matrix[upper][~same_module]
    if within.size == 0 or between.size == 0:
        raise ValueError("Rest network segregation requires both within- and between-module edges.")
    segregation = float(within.mean() - between.mean())
    return {
        "tr_sec": tr,
        "parcel_bin_size": bin_size,
        "parcel_min_voxels": min_voxels,
        "parcel_count": parcel_count,
        "within_module_mean_r": float(within.mean()),
        "between_module_mean_r": float(between.mean()),
        "network_segregation": segregation,
        "network_segregation_clipped": max(segregation, 0.0),
    }


def task_increment_proxy(
    task_bold: Path,
    task_events: Path,
    mask: np.ndarray | None,
    lag_sec: float,
    mask_mean_quantile: float = 0.35,
    mask_std_quantile: float = 0.35,
    mad_multiplier: float = 3.0,
) -> dict[str, float | np.ndarray]:
    task_data, tr = load_4d_nifti(task_bold)
    if mask is None:
        mask, mean_threshold, std_threshold = build_brain_mask(
            rest_data=task_data,
            mean_quantile=mask_mean_quantile,
            std_quantile=mask_std_quantile,
        )
    else:
        mean_threshold = None
        std_threshold = None
    task_matrix = masked_timeseries(task_data, mask)
    del task_data

    task_psc = percent_signal_change(task_matrix)
    del task_matrix

    task_name = infer_task_name(task_bold)
    active_mask = build_task_mask(task_psc.shape[1], tr, load_events(task_events), lag_sec, task_name)
    baseline_mask = ~active_mask
    if not np.any(baseline_mask):
        raise ValueError(f"Baseline mask is empty for {task_bold}")

    active_mean = task_psc[:, active_mask].mean(axis=1)
    baseline_mean = task_psc[:, baseline_mask].mean(axis=1)
    increment = np.clip(active_mean - baseline_mean, 0.0, None).astype(np.float32, copy=False)
    responsive_threshold, responsive_fraction = robust_threshold_fraction(increment, mad_multiplier)

    return {
        "task_name": task_name,
        "task_bold": str(task_bold),
        "task_events": str(task_events),
        "increment": increment,
        "tr": float(tr),
        "active_scan_fraction": float(active_mask.mean()),
        "increment_mean": float(increment.mean()),
        "increment_median": float(np.median(increment)),
        "responsive_threshold_increment": responsive_threshold,
        "responsive_fraction_increment": responsive_fraction,
        "mask_voxels": int(mask.sum()),
        "mask_mean_threshold": mean_threshold,
        "mask_std_threshold": std_threshold,
    }


def contrast_responsive_fraction(
    contrast_map: Path,
    contrast_name: str,
    mad_multiplier: float,
) -> tuple[float, float, str]:
    image = nib.load(str(contrast_map))
    axis0 = image.header.get_axis(0)
    names = [str(name) for name in axis0.name]
    if contrast_name in names:
        index = names.index(contrast_name)
    else:
        index = 0
        contrast_name = names[0]
    contrast = np.asanyarray(image.dataobj)[index].astype(np.float32, copy=False)
    threshold, fraction = robust_threshold_fraction(contrast, mad_multiplier)
    return threshold, fraction, contrast_name


def stage1_simplex(active_proxy: float, background_proxy: float) -> np.ndarray:
    pair_total = active_proxy + background_proxy
    if pair_total <= 0:
        raise ValueError("Stage-1 pair total must be positive.")
    non_struct_mass = 1.0 - P_STAR[1]
    x_a = non_struct_mass * active_proxy / pair_total
    x_b = non_struct_mass * background_proxy / pair_total
    return np.array([x_a, P_STAR[1], x_b], dtype=np.float64)


def stage1_summary(active_proxy: float, background_proxy: float) -> dict[str, object]:
    pair_total = active_proxy + background_proxy
    if pair_total <= 0:
        raise ValueError("Stage-1 pair total must be positive.")
    active_share = active_proxy / pair_total
    background_share = background_proxy / pair_total
    p_hat = stage1_simplex(active_proxy, background_proxy)
    return {
        "stage1_pair": {
            "active_share": active_share,
            "background_share": background_share,
            "lambda_share": active_share,
        },
        "stage1_simplex_with_xs_prior": {
            "p_hat": [float(x) for x in p_hat],
            "l2_to_p_star": float(np.linalg.norm(p_hat - P_STAR)),
        },
    }


def resolve_task_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    task_bolds = args.task_bold or DEFAULT_TASK_BOLD
    task_events = args.task_events or DEFAULT_TASK_EVENTS
    if len(task_bolds) != len(task_events):
        raise ValueError("--task-bold and --task-events must have the same count.")
    return [require_file(path) for path in task_bolds], [require_file(path) for path in task_events]


def load_tsv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows.extend(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "n/a":
        return None
    return float(text)


def mean_optional(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(np.mean(np.asarray(valid, dtype=np.float64)))


def empty_paired_summary() -> dict[str, float | int | None]:
    return {
        "n": 0,
        "median_ns": None,
        "median_sd": None,
        "mean_delta_sd_minus_ns": None,
        "positive_fraction_sd_minus_ns": None,
        "wilcoxon_p": None,
    }


def paired_summary_or_empty(ns_values: list[float], sd_values: list[float]) -> dict[str, float | int | None]:
    if not ns_values or len(ns_values) != len(sd_values):
        return empty_paired_summary()
    return paired_summary(ns_values, sd_values)


def _safe_delta(a: float, b: float) -> float | None:
    import math
    a_f, b_f = float(a), float(b)
    if math.isnan(a_f) or math.isnan(b_f):
        return None
    return a_f - b_f


def parse_sleep_deprived_label(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if not text or text == "n/a" or text == "equivocal":
        return None
    if "not" in text:
        return False
    if text in {"surely", "likely"}:
        return True
    return None


def session_index(session_id: str) -> int:
    prefix, _, suffix = session_id.partition("-")
    if prefix != "ses" or not suffix.isdigit():
        raise ValueError(f"Unrecognized session ID: {session_id}")
    return int(suffix)


def load_ds000201_subject_sessions(data_root: Path, subject_id: str) -> dict[str, dict[str, str]]:
    subject_path = data_root / subject_id / f"{subject_id}_sessions.tsv"
    fallback_path = data_root / f"{subject_id}_sessions.tsv"
    path = subject_path if subject_path.exists() else require_file(fallback_path)
    return {
        str(row["session_id"]): row
        for row in load_tsv_rows(path)
        if row.get("session_id")
    }


def session_kss_mean(session_row: dict[str, str]) -> float | None:
    return mean_optional(
        [
            parse_optional_float(session_row.get(f"preMRI_KSS_Q{index}_rating"))
            for index in range(6)
        ]
    )


def pvt_response_summary(path: Path) -> dict[str, float | int]:
    rows = load_tsv_rows(require_file(path))
    response_times = [
        parse_optional_float(row.get("response_time"))
        for row in rows
    ]
    valid = [value for value in response_times if value is not None]
    if not valid:
        raise ValueError(f"No readable PVT response times found in {path}")
    values = np.asarray(valid, dtype=np.float64)
    return {
        "trial_count": int(values.size),
        "mean_rt_sec": float(values.mean()),
        "median_rt_sec": float(np.median(values)),
        "lapse_fraction_ge_500ms": float(np.mean(values >= PVT_LAPSE_THRESHOLD_SEC)),
    }


def summarize_sleepybrain_metric(
    subject_records: list[dict[str, object]],
    key_path: tuple[str, ...],
) -> dict[str, float | int | None]:
    ns_values: list[float] = []
    sd_values: list[float] = []
    for record in subject_records:
        ns_current: object = record["normal_sleep"]
        sd_current: object = record["sleep_deprived"]
        for key in key_path:
            if not isinstance(ns_current, dict) or key not in ns_current:
                ns_current = None
                break
            ns_current = ns_current[key]
        for key in key_path:
            if not isinstance(sd_current, dict) or key not in sd_current:
                sd_current = None
                break
            sd_current = sd_current[key]
        if ns_current is None or sd_current is None:
            continue
        import math as _math
        ns_f, sd_f = float(ns_current), float(sd_current)
        if _math.isnan(ns_f) or _math.isnan(sd_f):
            continue
        ns_values.append(ns_f)
        sd_values.append(sd_f)
    return paired_summary_or_empty(ns_values, sd_values)


def paired_summary(ns_values: list[float], sd_values: list[float]) -> dict[str, float | int | None]:
    ns = np.asarray(ns_values, dtype=np.float64)
    sd = np.asarray(sd_values, dtype=np.float64)
    if ns.size == 0 or ns.size != sd.size:
        raise ValueError("Paired summaries require equally sized non-empty vectors.")
    delta = sd - ns
    try:
        p_value = float(wilcoxon(sd, ns).pvalue)
    except ValueError:
        p_value = None
    return {
        "n": int(ns.size),
        "median_ns": float(np.median(ns)),
        "median_sd": float(np.median(sd)),
        "mean_delta_sd_minus_ns": float(delta.mean()),
        "positive_fraction_sd_minus_ns": float(np.mean(delta > 0)),
        "wilcoxon_p": p_value,
    }


def summarize_behavior_metric(
    participant_rows: list[dict[str, str]],
    ns_key: str,
    sd_key: str,
) -> dict[str, float | int | None]:
    ns_values: list[float] = []
    sd_values: list[float] = []
    for row in participant_rows:
        ns_value = parse_optional_float(row.get(ns_key))
        sd_value = parse_optional_float(row.get(sd_key))
        if ns_value is None or sd_value is None:
            continue
        ns_values.append(ns_value)
        sd_values.append(sd_value)
    return paired_summary(ns_values, sd_values)


def parse_eeglab_set_header(set_path: Path) -> tuple[Path, int, float, list[str]]:
    mat = loadmat(str(set_path), squeeze_me=True, struct_as_record=False)
    if "EEG" in mat:
        eeg = mat["EEG"]
        data_ref = str(eeg.data)
        nbchan = int(eeg.nbchan)
        srate = float(eeg.srate)
        chanlocs = np.atleast_1d(eeg.chanlocs)
    else:
        required = ("data", "nbchan", "srate", "chanlocs")
        missing = [name for name in required if name not in mat]
        if missing:
            raise ValueError(f"Missing EEGLAB fields in {set_path}: {', '.join(missing)}")
        data_ref = str(mat["data"])
        nbchan = int(mat["nbchan"])
        srate = float(mat["srate"])
        chanlocs = np.atleast_1d(mat["chanlocs"])

    labels = [str(getattr(chanloc, "labels", "")).strip() for chanloc in chanlocs]
    return require_file(set_path.with_name(data_ref)), nbchan, srate, labels


def load_eeglab_matrix(set_path: Path) -> tuple[np.ndarray, float, list[str]]:
    fdt_path, nbchan, srate, labels = parse_eeglab_set_header(require_file(set_path))
    raw = np.fromfile(fdt_path, dtype="<f4")
    usable = (raw.size // nbchan) * nbchan
    if usable == 0:
        raise ValueError(f"Empty or invalid FDT payload: {fdt_path}")
    data = raw[:usable].reshape(nbchan, -1, order="F").astype(np.float32, copy=False)
    return data, srate, labels


def mean_bandpower(
    psd: np.ndarray,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
    picks: list[int] | None = None,
) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        raise ValueError(f"No frequencies available in [{low_hz}, {high_hz}).")
    matrix = psd if not picks else psd[picks]
    return float(trapezoid(matrix[:, mask], freqs[mask], axis=1).mean())


def posterior_alpha_fraction(set_path: Path) -> tuple[float, int]:
    data, srate, labels = load_eeglab_matrix(set_path)
    nperseg = min(2048, data.shape[1])
    noverlap = min(1024, max(0, nperseg // 2))
    freqs, psd = welch(data, fs=srate, nperseg=nperseg, noverlap=noverlap, axis=1)
    posterior_picks = [index for index, label in enumerate(labels) if label in POSTERIOR_CHANNELS]
    alpha = mean_bandpower(psd, freqs, 8.0, 12.0, posterior_picks)
    total = mean_bandpower(psd, freqs, 1.0, 30.0, posterior_picks)
    return alpha / max(total, 1e-12), int(data.shape[1])


def ds004902_subject_record(subject_id: str, data_root: Path) -> dict[str, float | int | str]:
    base = data_root / subject_id
    paths = {
        "ns_open": base / "ses-1" / "eeg" / f"{subject_id}_ses-1_task-eyesopen_eeg.set",
        "ns_closed": base / "ses-1" / "eeg" / f"{subject_id}_ses-1_task-eyesclosed_eeg.set",
        "sd_open": base / "ses-2" / "eeg" / f"{subject_id}_ses-2_task-eyesopen_eeg.set",
        "sd_closed": base / "ses-2" / "eeg" / f"{subject_id}_ses-2_task-eyesclosed_eeg.set",
    }
    ns_open, ns_open_samples = posterior_alpha_fraction(paths["ns_open"])
    ns_closed, ns_closed_samples = posterior_alpha_fraction(paths["ns_closed"])
    sd_open, sd_open_samples = posterior_alpha_fraction(paths["sd_open"])
    sd_closed, sd_closed_samples = posterior_alpha_fraction(paths["sd_closed"])
    return {
        "subject": subject_id,
        "ns_open_posterior_alpha_fraction": ns_open,
        "ns_closed_posterior_alpha_fraction": ns_closed,
        "sd_open_posterior_alpha_fraction": sd_open,
        "sd_closed_posterior_alpha_fraction": sd_closed,
        "ns_open_samples": ns_open_samples,
        "ns_closed_samples": ns_closed_samples,
        "sd_open_samples": sd_open_samples,
        "sd_closed_samples": sd_closed_samples,
        "ns_reactivity_difference": ns_closed - ns_open,
        "sd_reactivity_difference": sd_closed - sd_open,
        "ns_reactivity_ratio": ns_closed / max(ns_open, 1e-12),
        "sd_reactivity_ratio": sd_closed / max(sd_open, 1e-12),
    }


def load_ds000201_task_payloads(
    base: Path,
    subject_id: str,
    session_id: str,
    args: argparse.Namespace,
) -> list[dict[str, float | np.ndarray]]:
    payloads: list[dict[str, float | np.ndarray]] = []
    for task_name in ("hands", "sleepiness"):
        task_bold = base / "func" / f"{subject_id}_{session_id}_task-{task_name}_bold.nii.gz"
        task_events = base / "func" / f"{subject_id}_{session_id}_task-{task_name}_events.tsv"
        if not task_bold.exists() or not task_events.exists():
            continue
        payloads.append(
            task_increment_proxy(
                task_bold=task_bold,
                task_events=task_events,
                mask=None,
                lag_sec=args.hrf_lag_sec,
                mask_mean_quantile=args.mask_mean_quantile,
                mask_std_quantile=args.mask_std_quantile,
                mad_multiplier=args.mad_multiplier,
            )
        )
    if not payloads:
        raise FileNotFoundError(f"No readable ds000201 task BOLD/events pairs found for {subject_id} {session_id}")
    return payloads


def ds000201_session_record(
    subject_id: str,
    session_id: str,
    data_root: Path,
    participant_row: dict[str, str],
    session_rows: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, object]:
    base = data_root / subject_id / session_id
    rest_bold = require_file(base / "func" / f"{subject_id}_{session_id}_task-rest_bold.nii.gz")
    pvt_events = base / "beh" / f"{subject_id}_{session_id}_task-PVT_events.tsv"

    rest_data, _rest_tr = load_4d_nifti(rest_bold)
    mask, mean_threshold, std_threshold = build_brain_mask(
        rest_data=rest_data,
        mean_quantile=args.mask_mean_quantile,
        std_quantile=args.mask_std_quantile,
    )
    del rest_data

    rest_rms, rest_tr = rest_background_proxy(rest_bold, mask)
    rest_stability = rest_background_stability_proxy(rest_bold, mask)
    rest_network = rest_network_segregation_proxy(rest_bold, mask)
    task_payloads = load_ds000201_task_payloads(base, subject_id, session_id, args)
    task_summaries = [{key: value for key, value in payload.items() if key != "increment"} for payload in task_payloads]
    task_payload_map = {str(payload["task_name"]): payload for payload in task_payloads}
    active_increment_means = np.asarray(
        [float(payload["increment_mean"]) for payload in task_payloads],
        dtype=np.float64,
    )
    active_increment_medians = np.asarray(
        [float(payload["increment_median"]) for payload in task_payloads],
        dtype=np.float64,
    )
    active_fractions = np.asarray(
        [float(payload["responsive_fraction_increment"]) for payload in task_payloads],
        dtype=np.float64,
    )
    if "sleepiness" in task_payload_map:
        primary_active_payload = task_payload_map["sleepiness"]
        active_source_task = "sleepiness"
    else:
        primary_active_payload = task_payloads[0]
        active_source_task = "mean_available"
    active_proxy = (
        float(primary_active_payload["responsive_fraction_increment"])
        if active_source_task == "sleepiness"
        else float(active_fractions.mean())
    )
    active_increment_mean = (
        float(primary_active_payload["increment_mean"])
        if active_source_task == "sleepiness"
        else float(active_increment_means.mean())
    )
    active_increment_median = (
        float(primary_active_payload["increment_median"])
        if active_source_task == "sleepiness"
        else float(np.median(active_increment_medians))
    )
    background_proxy = float(rest_network["network_segregation_clipped"])
    legacy_stage1 = stage1_summary(active_increment_mean, float(rest_rms.mean()))
    session_row = session_rows[session_id]
    sleep_column = f"SleepDeprivedSession{session_index(session_id)}"
    sleep_deprived_label = participant_row.get(sleep_column)
    sleep_deprived = parse_sleep_deprived_label(sleep_deprived_label)
    kss_mean = session_kss_mean(session_row)
    pvt_summary = pvt_response_summary(pvt_events) if pvt_events.exists() else {
        "mean_rt_sec": float("nan"),
        "median_rt_sec": float("nan"),
        "lapse_count_ge_500ms": 0,
        "total_responses": 0,
        "lapse_fraction_ge_500ms": float("nan"),
    }
    stage1 = stage1_summary(active_proxy, background_proxy)

    return {
        "subject": subject_id,
        "session": session_id,
        "sleep_pressure": {
            "sleep_deprived": sleep_deprived,
            "sleep_deprived_label": sleep_deprived_label or "",
            "pre_mri_kss_mean": kss_mean,
        },
        "behavior": pvt_summary,
        "mask": {
            "voxels": int(mask.sum()),
            "mean_threshold": mean_threshold,
            "std_threshold": std_threshold,
        },
        "rest_proxy": {
            "tr_sec": rest_tr,
            "background_rms_mean_psc": float(rest_rms.mean()),
            "background_rms_median_psc": float(np.median(rest_rms)),
            **rest_stability,
            **rest_network,
        },
        "task_proxy": {
            "task_names": [str(summary["task_name"]) for summary in task_summaries],
            "task_summaries": task_summaries,
            "active_source_task": active_source_task,
            "active_increment_mean_psc": active_increment_mean,
            "active_increment_median_psc": active_increment_median,
            "active_responsive_fraction": active_proxy,
            "active_task_count": len(task_summaries),
        },
        "stage1_proxy_definition": {
            "active_proxy": "sleepiness responsive fraction when available, otherwise mean across available task increments",
            "background_proxy": "clipped coarse rest network segregation from parcel connectivity",
        },
        **stage1,
        "stage1_legacy_mean_psc_pair": legacy_stage1,
    }


def run_msc(args: argparse.Namespace) -> dict[str, object]:
    rest_bold = require_file(args.rest_bold)
    contrast_map = require_file(args.contrast_map)
    task_bolds, task_events = resolve_task_paths(args)

    rest_data, _rest_tr = load_4d_nifti(rest_bold)
    mask, mean_threshold, std_threshold = build_brain_mask(
        rest_data=rest_data,
        mean_quantile=args.mask_mean_quantile,
        std_quantile=args.mask_std_quantile,
    )
    del rest_data

    rest_rms, rest_tr = rest_background_proxy(rest_bold, mask)
    run_payloads = [
        task_increment_proxy(
            task_bold=task_bold,
            task_events=task_event,
            mask=mask,
            lag_sec=args.hrf_lag_sec,
            mad_multiplier=args.mad_multiplier,
        )
        for task_bold, task_event in zip(task_bolds, task_events, strict=True)
    ]

    increment_stack = np.stack([payload["increment"] for payload in run_payloads], axis=0)
    mean_increment = np.mean(increment_stack, axis=0)
    run_summaries = [
        {key: value for key, value in payload.items() if key != "increment"}
        for payload in run_payloads
    ]
    del increment_stack
    del run_payloads

    contrast_threshold, contrast_fraction, contrast_name = contrast_responsive_fraction(
        contrast_map=contrast_map,
        contrast_name=args.contrast_name,
        mad_multiplier=args.mad_multiplier,
    )

    active_proxy = float(mean_increment.mean())
    background_proxy = float(rest_rms.mean())
    stage1 = stage1_summary(active_proxy, background_proxy)

    return {
        "ce": {
            "p_star": [float(x) for x in P_STAR_DOC],
            "rho": float(RHO),
            "d_eff": float(D_EFF),
        },
        "inputs": {
            "rest_bold": str(rest_bold),
            "task_bold": [str(path) for path in task_bolds],
            "task_events": [str(path) for path in task_events],
            "contrast_map": str(contrast_map),
            "contrast_name": contrast_name,
        },
        "mask": {
            "voxels": int(mask.sum()),
            "mean_threshold": mean_threshold,
            "std_threshold": std_threshold,
        },
        "rest_proxy": {
            "tr_sec": rest_tr,
            "background_rms_mean_psc": background_proxy,
            "background_rms_median_psc": float(np.median(rest_rms)),
        },
        "task_proxy": {
            "run_summaries": run_summaries,
            "active_increment_mean_psc": active_proxy,
            "active_increment_median_psc": float(np.median(mean_increment)),
            "responsive_fraction_contrast": contrast_fraction,
            "responsive_threshold_contrast": contrast_threshold,
        },
        **stage1,
    }


def run_ds004902(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root
    participant_rows = load_tsv_rows(require_file(data_root / "participants.tsv"))

    behavior_summary: dict[str, dict[str, float | int | str | None]] = {}
    for key, label, ns_key, sd_key in BEHAVIOR_METRICS:
        summary = summarize_behavior_metric(participant_rows, ns_key, sd_key)
        summary["label"] = label
        behavior_summary[key] = summary

    if args.subject:
        subject_ids = args.subject
    else:
        subject_ids = sorted(path.name for path in data_root.glob("sub-*") if path.is_dir())

    subject_records: list[dict[str, float | int | str]] = []
    for subject_id in subject_ids:
        try:
            subject_records.append(ds004902_subject_record(subject_id, data_root))
        except (FileNotFoundError, ValueError):
            continue
        if args.subject_limit is not None and len(subject_records) >= args.subject_limit:
            break

    if not subject_records:
        raise ValueError("No readable ds004902 open/closed subject pairs were found.")

    reactivity_difference_summary = paired_summary(
        [float(record["ns_reactivity_difference"]) for record in subject_records],
        [float(record["sd_reactivity_difference"]) for record in subject_records],
    )
    reactivity_ratio_summary = paired_summary(
        [float(record["ns_reactivity_ratio"]) for record in subject_records],
        [float(record["sd_reactivity_ratio"]) for record in subject_records],
    )

    return {
        "dataset": "ds004902",
        "inputs": {
            "data_root": str(data_root),
            "subject_filter": args.subject or [],
            "subject_limit": args.subject_limit,
        },
        "behavioral_condition_check": behavior_summary,
        "sleep_reactivity": {
            "subjects_used": [str(record["subject"]) for record in subject_records],
            "subject_count": len(subject_records),
            "subject_records": subject_records,
            "posterior_alpha_reactivity_difference": reactivity_difference_summary,
            "posterior_alpha_reactivity_ratio": reactivity_ratio_summary,
        },
        "notes": [
            "Behavioral deltas are computed from the full metadata table in participants.tsv.",
            "EEG deltas only use subjects with readable eyes-open and eyes-closed session pairs.",
            "In this pilot, open-only resting EEG fractions were weaker than closed/open alpha reactivity.",
        ],
    }


def run_sleepybrain(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root
    participant_rows = load_tsv_rows(require_file(data_root / "participants.tsv"))
    participant_map = {
        str(row["participant_id"]): row
        for row in participant_rows
        if row.get("participant_id")
    }

    if args.subject:
        subject_ids = args.subject
    else:
        subject_ids = sorted(path.name for path in data_root.glob("sub-*") if path.is_dir())

    subject_records: list[dict[str, object]] = []
    for subject_id in subject_ids:
        participant_row = participant_map.get(subject_id)
        if participant_row is None:
            continue
        try:
            session_rows = load_ds000201_subject_sessions(data_root, subject_id)
        except (FileNotFoundError, ValueError):
            continue

        session_records: list[dict[str, object]] = []
        for session_id in sorted(session_rows):
            try:
                session_records.append(
                    ds000201_session_record(
                        subject_id=subject_id,
                        session_id=session_id,
                        data_root=data_root,
                        participant_row=participant_row,
                        session_rows=session_rows,
                        args=args,
                    )
                )
            except (FileNotFoundError, ValueError):
                continue

        if len(session_records) < 2:
            continue

        normal_sleep = [
            record
            for record in session_records
            if record["sleep_pressure"]["sleep_deprived"] is False
        ]
        sleep_deprived = [
            record
            for record in session_records
            if record["sleep_pressure"]["sleep_deprived"] is True
        ]
        unlabeled = [
            record
            for record in session_records
            if record["sleep_pressure"]["sleep_deprived"] is None
        ]
        if len(normal_sleep) == 1 and len(sleep_deprived) == 0 and len(unlabeled) == 1:
            sleep_deprived = unlabeled
        elif len(sleep_deprived) == 1 and len(normal_sleep) == 0 and len(unlabeled) == 1:
            normal_sleep = unlabeled
        elif len(normal_sleep) == 0 and len(sleep_deprived) == 0 and len(unlabeled) == 2:
            kss_values = [
                (rec, rec["sleep_pressure"].get("pre_mri_kss_mean")) for rec in unlabeled
            ]
            kss_a, kss_b = kss_values[0][1], kss_values[1][1]
            if kss_a is not None and kss_b is not None and kss_a != kss_b:
                if kss_a < kss_b:
                    normal_sleep, sleep_deprived = [kss_values[0][0]], [kss_values[1][0]]
                else:
                    normal_sleep, sleep_deprived = [kss_values[1][0]], [kss_values[0][0]]

        if len(normal_sleep) != 1 or len(sleep_deprived) != 1:
            continue

        ns_record = normal_sleep[0]
        sd_record = sleep_deprived[0]
        ns_kss = ns_record["sleep_pressure"]["pre_mri_kss_mean"]
        sd_kss = sd_record["sleep_pressure"]["pre_mri_kss_mean"]
        if ns_kss is not None and sd_kss is not None and float(sd_kss) < float(ns_kss):
            continue
        subject_records.append(
            {
                "subject": subject_id,
                "normal_sleep": ns_record,
                "sleep_deprived": sd_record,
                "delta_sd_minus_ns": {
                    "pre_mri_kss_mean": None
                    if ns_kss is None or sd_kss is None
                    else float(sd_kss) - float(ns_kss),
                    "pvt_mean_rt_sec": _safe_delta(
                        sd_record["behavior"]["mean_rt_sec"],
                        ns_record["behavior"]["mean_rt_sec"],
                    ),
                    "pvt_lapse_fraction_ge_500ms": _safe_delta(
                        sd_record["behavior"]["lapse_fraction_ge_500ms"],
                        ns_record["behavior"]["lapse_fraction_ge_500ms"],
                    ),
                    "active_responsive_fraction": float(sd_record["task_proxy"]["active_responsive_fraction"])
                    - float(ns_record["task_proxy"]["active_responsive_fraction"]),
                    "active_increment_mean_psc": float(sd_record["task_proxy"]["active_increment_mean_psc"])
                    - float(ns_record["task_proxy"]["active_increment_mean_psc"]),
                    "background_segregation_proxy": float(
                        sd_record["rest_proxy"]["network_segregation_clipped"]
                    )
                    - float(ns_record["rest_proxy"]["network_segregation_clipped"]),
                    "background_rms_mean_psc": float(sd_record["rest_proxy"]["background_rms_mean_psc"])
                    - float(ns_record["rest_proxy"]["background_rms_mean_psc"]),
                    "active_share": float(sd_record["stage1_pair"]["active_share"])
                    - float(ns_record["stage1_pair"]["active_share"]),
                    "background_share": float(sd_record["stage1_pair"]["background_share"])
                    - float(ns_record["stage1_pair"]["background_share"]),
                },
            }
        )
        if args.subject_limit is not None and len(subject_records) >= args.subject_limit:
            break

    if not subject_records:
        raise ValueError("No readable ds000201 subject pairs with explicit sleep labels were found.")

    cohort_summary = {
        "subject_count": len(subject_records),
        "subjects_used": [str(record["subject"]) for record in subject_records],
        "pre_mri_kss_mean": summarize_sleepybrain_metric(subject_records, ("sleep_pressure", "pre_mri_kss_mean")),
        "pvt_mean_rt_sec": summarize_sleepybrain_metric(subject_records, ("behavior", "mean_rt_sec")),
        "pvt_lapse_fraction_ge_500ms": summarize_sleepybrain_metric(
            subject_records,
            ("behavior", "lapse_fraction_ge_500ms"),
        ),
        "active_responsive_fraction": summarize_sleepybrain_metric(
            subject_records,
            ("task_proxy", "active_responsive_fraction"),
        ),
        "active_increment_mean_psc": summarize_sleepybrain_metric(
            subject_records,
            ("task_proxy", "active_increment_mean_psc"),
        ),
        "background_segregation_proxy": summarize_sleepybrain_metric(
            subject_records,
            ("rest_proxy", "network_segregation_clipped"),
        ),
        "background_rms_mean_psc": summarize_sleepybrain_metric(
            subject_records,
            ("rest_proxy", "background_rms_mean_psc"),
        ),
        "active_share": summarize_sleepybrain_metric(subject_records, ("stage1_pair", "active_share")),
        "background_share": summarize_sleepybrain_metric(subject_records, ("stage1_pair", "background_share")),
    }

    return {
        "dataset": "ds000201",
        "inputs": {
            "data_root": str(data_root),
            "subject_filter": args.subject or [],
            "subject_limit": args.subject_limit,
            "hrf_lag_sec": float(args.hrf_lag_sec),
        },
        "subject_records": subject_records,
        "cohort_summary": cohort_summary,
        "notes": [
            "This pilot uses within-subject rest and available hands/sleepiness task fMRI from ds000201.",
            "Sleep condition labels come from participants.tsv SleepDeprivedSession1/2 columns.",
            "Session-level q_sleep is proxied by the pre-MRI KSS mean in the subject sessions TSV.",
            "Stage-1 active proxy uses the sleepiness task responsive fraction when available, with other tasks as support.",
            "Stage-1 background proxy uses clipped coarse rest network segregation from parcel connectivity.",
            "Stage-1 simplex still holds x_s fixed to the CE prior.",
        ],
    }


def print_msc_result(result: dict[str, object]) -> None:
    ce = result["ce"]
    mask = result["mask"]
    rest_proxy = result["rest_proxy"]
    task_proxy = result["task_proxy"]
    stage1_pair = result["stage1_pair"]
    stage1_simplex = result["stage1_simplex_with_xs_prior"]

    print("=" * 72)
    print("CE brain pilot: Midnight Scan Club")
    print("=" * 72)
    print(
        "p* = ({:.4f}, {:.4f}, {:.4f})".format(
            *[float(value) for value in ce["p_star"]]
        )
    )
    print(f"rho = {float(ce['rho']):.3f}, D_eff = {float(ce['d_eff']):.3f}")
    print()
    print("Mask")
    print(f"  voxels kept: {int(mask['voxels'])}")
    print(f"  mean threshold: {float(mask['mean_threshold']):.3f}")
    print(f"  std threshold:  {float(mask['std_threshold']):.3f}")
    print()
    print("x_b proxy from raw rest")
    print(
        "  mean RMS(percent-signal-change):   "
        f"{float(rest_proxy['background_rms_mean_psc']):.4f}"
    )
    print(
        "  median RMS(percent-signal-change): "
        f"{float(rest_proxy['background_rms_median_psc']):.4f}"
    )
    print()
    print("x_a proxy from raw motor runs")
    for index, payload in enumerate(task_proxy["run_summaries"], start=1):
        print(
            "  run {idx}: mean increment={mean:.4f}, median increment={median:.4f}, "
            "active scans={active:.3f}".format(
                idx=index,
                mean=float(payload["increment_mean"]),
                median=float(payload["increment_median"]),
                active=float(payload["active_scan_fraction"]),
            )
        )
    print(
        "  mean increment across runs:        "
        f"{float(task_proxy['active_increment_mean_psc']):.4f}"
    )
    print(
        "  median increment across runs:      "
        f"{float(task_proxy['active_increment_median_psc']):.4f}"
    )
    print(
        "  responsive fraction (contrast):    "
        f"{float(task_proxy['responsive_fraction_contrast']):.4f}"
    )
    print()
    print("Stage-1 pair and simplex")
    print(f"  active share u_a/(u_a+u_b):        {float(stage1_pair['active_share']):.4f}")
    print(
        f"  background share u_b/(u_a+u_b):    {float(stage1_pair['background_share']):.4f}"
    )
    print(
        "  p_hat with x_s fixed to p*:        "
        "({:.4f}, {:.4f}, {:.4f})".format(
            *[float(value) for value in stage1_simplex["p_hat"]]
        )
    )
    print(f"  ||p_hat - p*||_2:                  {float(stage1_simplex['l2_to_p_star']):.4f}")
    print()
    print("Notes")
    print("  - x_s is held at the CE prior in this stage-1 pilot.")
    print("  - This is a within-subject pilot, not a cohort-level estimate.")


def print_ds004902_result(result: dict[str, object]) -> None:
    behavior_summary = result["behavioral_condition_check"]
    sleep_reactivity = result["sleep_reactivity"]
    ratio_summary = sleep_reactivity["posterior_alpha_reactivity_ratio"]
    diff_summary = sleep_reactivity["posterior_alpha_reactivity_difference"]

    print("=" * 72)
    print("CE brain pilot: OpenNeuro ds004902")
    print("=" * 72)
    print("Behavioral condition check (full metadata table)")
    for key, _, _, _ in BEHAVIOR_METRICS:
        summary = behavior_summary[key]
        print(
            "  {label}: n={n}, median NS={median_ns:.3f}, median SD={median_sd:.3f}, "
            "mean delta={mean_delta:.3f}, positive frac={positive_frac:.3f}, p={p_value}".format(
                label=summary["label"],
                n=int(summary["n"]),
                median_ns=float(summary["median_ns"]),
                median_sd=float(summary["median_sd"]),
                mean_delta=float(summary["mean_delta_sd_minus_ns"]),
                positive_frac=float(summary["positive_fraction_sd_minus_ns"]),
                p_value="n/a"
                if summary["wilcoxon_p"] is None
                else f"{float(summary['wilcoxon_p']):.4g}",
            )
        )

    print()
    print("Resting EEG posterior alpha reactivity")
    print(f"  readable subject pairs:            {int(sleep_reactivity['subject_count'])}")
    print(
        "  subjects used:                     "
        + ", ".join(sleep_reactivity["subjects_used"])
    )
    print("  ratio = eyes-closed / eyes-open")
    print(f"    median NS:                       {float(ratio_summary['median_ns']):.4f}")
    print(f"    median SD:                       {float(ratio_summary['median_sd']):.4f}")
    print(
        "    mean delta (SD-NS):              "
        f"{float(ratio_summary['mean_delta_sd_minus_ns']):.4f}"
    )
    print(
        "    positive frac (SD>NS):           "
        f"{float(ratio_summary['positive_fraction_sd_minus_ns']):.4f}"
    )
    print(
        "    Wilcoxon p:                      "
        + (
            "n/a"
            if ratio_summary["wilcoxon_p"] is None
            else f"{float(ratio_summary['wilcoxon_p']):.4g}"
        )
    )
    print("  difference = eyes-closed - eyes-open")
    print(f"    median NS:                       {float(diff_summary['median_ns']):.4f}")
    print(f"    median SD:                       {float(diff_summary['median_sd']):.4f}")
    print(
        "    mean delta (SD-NS):              "
        f"{float(diff_summary['mean_delta_sd_minus_ns']):.4f}"
    )
    print(
        "    positive frac (SD>NS):           "
        f"{float(diff_summary['positive_fraction_sd_minus_ns']):.4f}"
    )
    print(
        "    Wilcoxon p:                      "
        + (
            "n/a"
            if diff_summary["wilcoxon_p"] is None
            else f"{float(diff_summary['wilcoxon_p']):.4g}"
        )
    )
    print()
    print("Notes")
    for note in result["notes"]:
        print(f"  - {note}")


def format_optional(value: object, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def print_sleepybrain_result(result: dict[str, object]) -> None:
    cohort_summary = result["cohort_summary"]
    subject_records = result["subject_records"]
    kss_summary = cohort_summary["pre_mri_kss_mean"]
    pvt_summary = cohort_summary["pvt_mean_rt_sec"]
    active_sensor_summary = cohort_summary["active_responsive_fraction"]
    background_sensor_summary = cohort_summary["background_segregation_proxy"]
    active_summary = cohort_summary["active_share"]
    background_summary = cohort_summary["background_share"]

    print("=" * 72)
    print("CE brain pilot: Sleepy Brain Project I")
    print("=" * 72)
    print(f"  readable subject pairs:            {int(cohort_summary['subject_count'])}")
    print(
        "  subjects used:                     "
        + ", ".join(cohort_summary["subjects_used"])
    )
    print()
    print("Within-subject sleep pressure")
    print(f"  pre-MRI KSS median NS:             {format_optional(kss_summary['median_ns'])}")
    print(f"  pre-MRI KSS median SD:             {format_optional(kss_summary['median_sd'])}")
    print(
        "  mean delta (SD-NS):                "
        f"{format_optional(kss_summary['mean_delta_sd_minus_ns'])}"
    )
    print(
        "  Wilcoxon p:                        "
        + (
            "n/a"
            if kss_summary["wilcoxon_p"] is None
            else f"{float(kss_summary['wilcoxon_p']):.4g}"
        )
    )
    print()
    print("Behavioral condition check")
    print(f"  PVT mean RT median NS:             {format_optional(pvt_summary['median_ns'])} sec")
    print(f"  PVT mean RT median SD:             {format_optional(pvt_summary['median_sd'])} sec")
    print(
        "  mean delta (SD-NS):                "
        f"{format_optional(pvt_summary['mean_delta_sd_minus_ns'])} sec"
    )
    print()
    print("Selected stage-1 sensors")
    print(
        "  active responsive frac NS:         "
        f"{format_optional(active_sensor_summary['median_ns'])}"
    )
    print(
        "  active responsive frac SD:         "
        f"{format_optional(active_sensor_summary['median_sd'])}"
    )
    print(
        "  mean delta (SD-NS):                "
        f"{format_optional(active_sensor_summary['mean_delta_sd_minus_ns'])}"
    )
    print(
        "  background segregation NS:         "
        f"{format_optional(background_sensor_summary['median_ns'])}"
    )
    print(
        "  background segregation SD:         "
        f"{format_optional(background_sensor_summary['median_sd'])}"
    )
    print(
        "  mean delta (SD-NS):                "
        f"{format_optional(background_sensor_summary['mean_delta_sd_minus_ns'])}"
    )
    print()
    print("Stage-1 x_a/x_b state")
    print(f"  active share median NS:            {format_optional(active_summary['median_ns'])}")
    print(f"  active share median SD:            {format_optional(active_summary['median_sd'])}")
    print(
        "  mean delta (SD-NS):                "
        f"{format_optional(active_summary['mean_delta_sd_minus_ns'])}"
    )
    print(f"  background share median NS:        {format_optional(background_summary['median_ns'])}")
    print(f"  background share median SD:        {format_optional(background_summary['median_sd'])}")
    print(
        "  mean delta (SD-NS):                "
        f"{format_optional(background_summary['mean_delta_sd_minus_ns'])}"
    )
    print()
    print("Per-subject sessions")
    for record in subject_records:
        ns_record = record["normal_sleep"]
        sd_record = record["sleep_deprived"]
        print(
            "  {subject}: NS {ns_session} sensor_a={ns_sensor_a:.4f}, sensor_b={ns_sensor_b:.4f}, "
            "share_a={ns_active:.4f}; SD {sd_session} sensor_a={sd_sensor_a:.4f}, "
            "sensor_b={sd_sensor_b:.4f}, share_a={sd_active:.4f}, KSS {ns_kss}->{sd_kss}".format(
                subject=record["subject"],
                ns_session=ns_record["session"],
                ns_sensor_a=float(ns_record["task_proxy"]["active_responsive_fraction"]),
                ns_sensor_b=float(ns_record["rest_proxy"]["network_segregation_clipped"]),
                ns_active=float(ns_record["stage1_pair"]["active_share"]),
                ns_kss=format_optional(ns_record["sleep_pressure"]["pre_mri_kss_mean"]),
                sd_session=sd_record["session"],
                sd_sensor_a=float(sd_record["task_proxy"]["active_responsive_fraction"]),
                sd_sensor_b=float(sd_record["rest_proxy"]["network_segregation_clipped"]),
                sd_active=float(sd_record["stage1_pair"]["active_share"]),
                sd_kss=format_optional(sd_record["sleep_pressure"]["pre_mri_kss_mean"]),
            )
        )
    print()
    print("Notes")
    for note in result["notes"]:
        print(f"  - {note}")


def main() -> None:
    args = parse_args()

    if args.dataset == "msc":
        result = run_msc(args)
        print_msc_result(result)
    elif args.dataset == "ds004902":
        result = run_ds004902(args)
        print_ds004902_result(result)
    else:
        result = run_sleepybrain(args)
        print_sleepybrain_result(result)

    write_json(result, args.json_out)


if __name__ == "__main__":
    main()
