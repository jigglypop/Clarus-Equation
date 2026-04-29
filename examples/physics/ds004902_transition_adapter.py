"""Convert ds004902-derived summary rows into brain-equation transition JSON.

This adapter does not process raw EEG. It expects a per-subject summary table
with normal-sleep and sleep-deprivation rows and converts the measured burden
proxies into the real-transition schema used by brain_equation_integrated_gate.py.

Required columns, with accepted aliases:

- subject: subject, participant_id, participant, sub
- session: session, condition
- posterior alpha reactivity: posterior_alpha_reactivity_ratio, alpha_reactivity
- PVT mean reaction time: pvt_mean_rt_ms, pvt_mean_rt, mean_rt_ms
- sleepiness: stanford_sleepiness_scale, sss, sleepiness

Optional region-resolved state columns:

- region: region
- active component proxy: active_proxy, x_a, active
- structural component proxy: structural_proxy, x_s, structural
- background component proxy: background_proxy, x_b, background

The output is a first-pass transition set for the sleep/homeostasis block. It is
claim-ready only when region-resolved state columns are supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from brain_equation_integrated_gate import (
    EDGES,
    REGIONS,
    P_STAR,
    homeostatic_forcing_from_q_delta,
    project_simplex,
    slow_forcing_for_d_sleep,
)


ALIASES = {
    "subject": ("subject", "participant_id", "participant", "sub"),
    "session": ("session", "condition"),
    "alpha": ("posterior_alpha_reactivity_ratio", "alpha_reactivity", "alpha_ratio"),
    "pvt": ("pvt_mean_rt_ms", "pvt_mean_rt", "mean_rt_ms"),
    "sleepiness": ("stanford_sleepiness_scale", "sss", "sleepiness"),
    "region": ("region",),
    "active": ("active_proxy", "x_a", "active"),
    "structural": ("structural_proxy", "x_s", "structural"),
    "background": ("background_proxy", "x_b", "background"),
}
NORMAL_SESSION_LABELS = {"ns", "normal", "normal_sleep", "baseline"}
DEPRIVED_SESSION_LABELS = {"sd", "sleep_deprived", "sleep_deprivation", "deprivation"}


def read_text(path: Path) -> str:
    return sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")


def delimiter_for(path: Path, text: str) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    sample = text[:2048]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return ","


def read_rows(path: Path) -> list[dict[str, str]]:
    text = read_text(path)
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter_for(path, text))
    if reader.fieldnames is None:
        raise ValueError("summary table must have a header row")
    rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("summary table contains no rows")
    return rows


def first_present(row: dict[str, str], key: str) -> str:
    for alias in ALIASES[key]:
        if alias in row and row[alias] not in {"", "n/a", "NA", "NaN"}:
            return row[alias]
    raise ValueError(f"missing required column for {key}: {ALIASES[key]}")


def has_alias(row: dict[str, str], key: str) -> bool:
    return any(alias in row for alias in ALIASES[key])


def has_region_state_columns(rows: list[dict[str, str]]) -> bool:
    first = rows[0]
    return all(
        has_alias(first, key)
        for key in ("region", "active", "structural", "background")
    )


def float_value(row: dict[str, str], key: str) -> float:
    value = float(first_present(row, key))
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def session_kind(row: dict[str, str]) -> str:
    label = first_present(row, "session").strip().lower().replace("-", "_")
    if label in NORMAL_SESSION_LABELS:
        return "normal_sleep"
    if label in DEPRIVED_SESSION_LABELS:
        return "sleep_deprived"
    raise ValueError(f"unknown session label: {label}")


def grouped_sessions(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        subject = first_present(row, "subject")
        grouped.setdefault(subject, {})[session_kind(row)] = row
    return grouped


def grouped_region_sessions(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        subject = first_present(row, "subject")
        grouped.setdefault(subject, {}).setdefault(session_kind(row), []).append(row)
    return grouped


def positive_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("normal-sleep baseline must be positive")
    return max(numerator / denominator, 0.0)


def sleep_burden(normal: dict[str, str], deprived: dict[str, str]) -> dict[str, object]:
    alpha_ns = float_value(normal, "alpha")
    alpha_sd = float_value(deprived, "alpha")
    pvt_ns = float_value(normal, "pvt")
    pvt_sd = float_value(deprived, "pvt")
    sleepiness_ns = float_value(normal, "sleepiness")
    sleepiness_sd = float_value(deprived, "sleepiness")

    z_alpha = positive_fraction(alpha_ns - alpha_sd, alpha_ns)
    z_pvt = positive_fraction(pvt_sd - pvt_ns, pvt_ns)
    z_sleepiness = max((sleepiness_sd - sleepiness_ns) / 6.0, 0.0)
    z_mean = (z_alpha + z_pvt + z_sleepiness) / 3.0

    return {
        "alpha_normal_sleep": alpha_ns,
        "alpha_sleep_deprived": alpha_sd,
        "pvt_normal_sleep": pvt_ns,
        "pvt_sleep_deprived": pvt_sd,
        "sleepiness_normal_sleep": sleepiness_ns,
        "sleepiness_sleep_deprived": sleepiness_sd,
        "z_alpha": z_alpha,
        "z_pvt": z_pvt,
        "z_sleepiness": z_sleepiness,
        "z_mean": z_mean,
        "passed_alpha_direction": alpha_sd < alpha_ns,
        "passed_pvt_direction": pvt_sd > pvt_ns,
        "passed_sleepiness_direction": sleepiness_sd > sleepiness_ns,
    }


def p_star_matrix() -> np.ndarray:
    return np.asarray([P_STAR[region] for region in REGIONS], dtype=np.float64)


def state_from_burden(q_delta: dict[str, float], d_sleep: float) -> np.ndarray:
    raw = (
        p_star_matrix()
        + homeostatic_forcing_from_q_delta(q_delta)
        + slow_forcing_for_d_sleep(d_sleep)
    )
    return np.asarray([project_simplex(row) for row in raw], dtype=np.float64)


def state_from_region_rows(
    rows: list[dict[str, str]],
    *,
    subject: str,
    session: str,
    fill_missing_regions: bool,
) -> np.ndarray:
    by_region = {first_present(row, "region"): row for row in rows}
    missing = [region for region in REGIONS if region not in by_region]
    if missing and not fill_missing_regions:
        raise ValueError(f"{subject}.{session} is missing region rows: {missing}")

    state_rows = []
    for region in REGIONS:
        if region not in by_region:
            state_rows.append(P_STAR[region])
            continue
        row = by_region[region]
        values = np.asarray(
            [
                float_value(row, "active"),
                float_value(row, "structural"),
                float_value(row, "background"),
            ],
            dtype=np.float64,
        )
        if np.any(values < 0.0):
            raise ValueError(f"{subject}.{session}.{region} state proxies must be non-negative")
        state_rows.append(project_simplex(values))
    return np.asarray(state_rows, dtype=np.float64)


def region_coverage_row(
    subject: str,
    normal_rows: list[dict[str, str]],
    deprived_rows: list[dict[str, str]],
) -> dict[str, object]:
    normal_regions = {first_present(row, "region") for row in normal_rows}
    deprived_regions = {first_present(row, "region") for row in deprived_rows}
    observed = sorted(normal_regions.intersection(deprived_regions))
    missing = [region for region in REGIONS if region not in observed]
    return {
        "subject": subject,
        "observed_regions": observed,
        "missing_regions": missing,
        "observed_region_count": len(observed),
        "required_region_count": len(REGIONS),
        "complete": not missing,
    }


def region_coverage_diagnostics(
    rows: list[dict[str, object]],
    *,
    fill_missing_regions: bool,
) -> dict[str, object]:
    if not rows:
        return {
            "available": False,
            "reason": "region-resolved state columns were not supplied",
        }
    return {
        "available": True,
        "fill_missing_regions": fill_missing_regions,
        "subject_count": len(rows),
        "complete_subject_rate": float(np.mean([bool(row["complete"]) for row in rows])),
        "mean_observed_region_count": float(
            np.mean([int(row["observed_region_count"]) for row in rows])
        ),
        "subjects": rows,
    }


def transition_for_subject(
    subject: str,
    burden: dict[str, object],
    *,
    d_sleep: float,
    current_p: np.ndarray | None = None,
    observed_next_p: np.ndarray | None = None,
) -> dict[str, object]:
    q_delta = {
        "sleep": float(burden["z_mean"]),
        "arousal": float(burden["z_sleepiness"]),
        "metabolic": 0.0,
    }
    observation_source = (
        "region_resolved_state"
        if current_p is not None and observed_next_p is not None
        else "proxy_derived_from_control"
    )
    current = p_star_matrix() if current_p is None else current_p
    observed = (
        state_from_burden(q_delta, d_sleep)
        if observed_next_p is None
        else observed_next_p
    )
    return {
        "case_id": f"{subject}_normal_to_deprived",
        "subject": subject,
        "session": "normal_to_deprived",
        "task": "resting_eeg",
        "current_p": current.tolist(),
        "observed_next_p": observed.tolist(),
        "task_input": 0.0,
        "q_delta": q_delta,
        "plasticity_input": 0.0,
        "d_sleep": d_sleep,
        "graph_edges": [[left, right, float(weight)] for left, right, weight in EDGES],
        "source_proxies": burden,
        "observation_source": observation_source,
        "adapter_note": (
            "region-resolved state transition"
            if observation_source == "region_resolved_state"
            else "proxy-derived transition; validate with region-resolved EEG/fMRI before biological claims"
        ),
    }


def burden_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    if count == 0:
        raise ValueError("cannot diagnose an empty burden row set")
    return {
        "subject_count": count,
        "mean_z_alpha": float(np.mean([float(row["z_alpha"]) for row in rows])),
        "mean_z_pvt": float(np.mean([float(row["z_pvt"]) for row in rows])),
        "mean_z_sleepiness": float(
            np.mean([float(row["z_sleepiness"]) for row in rows])
        ),
        "mean_z_sleep": float(np.mean([float(row["z_mean"]) for row in rows])),
        "alpha_direction_rate": float(
            np.mean([bool(row["passed_alpha_direction"]) for row in rows])
        ),
        "pvt_direction_rate": float(
            np.mean([bool(row["passed_pvt_direction"]) for row in rows])
        ),
        "sleepiness_direction_rate": float(
            np.mean([bool(row["passed_sleepiness_direction"]) for row in rows])
        ),
        "all_proxy_direction_rate": float(
            np.mean(
                [
                    bool(row["passed_alpha_direction"])
                    and bool(row["passed_pvt_direction"])
                    and bool(row["passed_sleepiness_direction"])
                    for row in rows
                ]
            )
        ),
    }


def state_delta_row(
    subject: str,
    current_p: np.ndarray,
    observed_next_p: np.ndarray,
) -> dict[str, object]:
    delta = observed_next_p - current_p
    region_rows = [
        {
            "region": region,
            "delta_active": float(delta[index, 0]),
            "delta_structural": float(delta[index, 1]),
            "delta_background": float(delta[index, 2]),
            "passed_sleep_deprivation_direction": bool(
                delta[index, 0] > 0.0 and delta[index, 2] < 0.0
            ),
        }
        for index, region in enumerate(REGIONS)
    ]
    return {
        "subject": subject,
        "mean_delta_active": float(np.mean(delta[:, 0])),
        "mean_delta_structural": float(np.mean(delta[:, 1])),
        "mean_delta_background": float(np.mean(delta[:, 2])),
        "active_increase_rate": float(np.mean(delta[:, 0] > 0.0)),
        "background_decrease_rate": float(np.mean(delta[:, 2] < 0.0)),
        "sleep_deprivation_direction_rate": float(
            np.mean((delta[:, 0] > 0.0) & (delta[:, 2] < 0.0))
        ),
        "regions": region_rows,
    }


def state_delta_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "available": False,
            "reason": "region-resolved state columns were not supplied",
        }
    return {
        "available": True,
        "subject_count": len(rows),
        "mean_delta_active": float(
            np.mean([float(row["mean_delta_active"]) for row in rows])
        ),
        "mean_delta_structural": float(
            np.mean([float(row["mean_delta_structural"]) for row in rows])
        ),
        "mean_delta_background": float(
            np.mean([float(row["mean_delta_background"]) for row in rows])
        ),
        "mean_active_increase_rate": float(
            np.mean([float(row["active_increase_rate"]) for row in rows])
        ),
        "mean_background_decrease_rate": float(
            np.mean([float(row["background_decrease_rate"]) for row in rows])
        ),
        "mean_sleep_deprivation_direction_rate": float(
            np.mean([float(row["sleep_deprivation_direction_rate"]) for row in rows])
        ),
        "subjects": rows,
    }


def build_transitions(
    rows: list[dict[str, str]],
    *,
    d_sleep: float,
    fill_missing_regions: bool,
) -> dict[str, object]:
    transitions = []
    burden_rows = []
    state_delta_rows = []
    region_coverage_rows = []
    skipped = []
    uses_region_state = has_region_state_columns(rows)
    grouped = grouped_region_sessions(rows) if uses_region_state else grouped_sessions(rows)
    for subject, sessions in grouped.items():
        if "normal_sleep" not in sessions or "sleep_deprived" not in sessions:
            skipped.append(
                {
                    "subject": subject,
                    "reason": "need both normal_sleep and sleep_deprived rows",
                }
            )
            continue
        normal_rows = (
            sessions["normal_sleep"]
            if uses_region_state
            else [sessions["normal_sleep"]]
        )
        deprived_rows = (
            sessions["sleep_deprived"]
            if uses_region_state
            else [sessions["sleep_deprived"]]
        )
        burden = sleep_burden(normal_rows[0], deprived_rows[0])
        burden_rows.append({"subject": subject, **burden})
        if uses_region_state:
            region_coverage_rows.append(
                region_coverage_row(subject, normal_rows, deprived_rows)
            )
        current_p = (
            state_from_region_rows(
                normal_rows,
                subject=subject,
                session="normal_sleep",
                fill_missing_regions=fill_missing_regions,
            )
            if uses_region_state
            else None
        )
        observed_next_p = (
            state_from_region_rows(
                deprived_rows,
                subject=subject,
                session="sleep_deprived",
                fill_missing_regions=fill_missing_regions,
            )
            if uses_region_state
            else None
        )
        if current_p is not None and observed_next_p is not None:
            state_delta_rows.append(state_delta_row(subject, current_p, observed_next_p))
        transitions.append(
            transition_for_subject(
                subject,
                burden,
                d_sleep=d_sleep,
                current_p=current_p,
                observed_next_p=observed_next_p,
            )
        )
    if not transitions:
        raise ValueError("no complete normal_sleep -> sleep_deprived subject pairs found")
    return {
        "region_order": REGIONS,
        "transitions": transitions,
        "adapter": {
            "source": "OpenNeuro ds004902 summary table",
            "d_sleep": d_sleep,
            "observation_mode": (
                "region_resolved_state" if uses_region_state else "control_proxy_only"
            ),
            "skipped": skipped,
            "burden_diagnostics": burden_diagnostics(burden_rows),
            "region_coverage": region_coverage_diagnostics(
                region_coverage_rows,
                fill_missing_regions=fill_missing_regions,
            ),
            "state_delta_diagnostics": state_delta_diagnostics(state_delta_rows),
            "burden_rows": burden_rows,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_table", type=Path, help="CSV/TSV summary table or '-' for stdin.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("-"),
        help="Output JSON path. Defaults to stdout.",
    )
    parser.add_argument(
        "--d-sleep",
        type=float,
        default=30.0,
        help="Sleep deprivation duration in hours for d_sleep.",
    )
    parser.add_argument(
        "--fill-missing-regions",
        action="store_true",
        help="Fill missing region rows with P_STAR and report them in region_coverage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_transitions(
        read_rows(args.summary_table),
        d_sleep=args.d_sleep,
        fill_missing_regions=args.fill_missing_regions,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if str(args.output) == "-":
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
