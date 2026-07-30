#!/usr/bin/env python
"""Run the CloudCell operational gate on human Sternberg-task NWB files.

The loader is the only part of the gate that imports ``h5py``.  It converts
ragged NWB spike times into causal, per-trial rate windows and passes plain
NumPy arrays to :mod:`reality_stone.clarus.cloudcell_evidence`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reality_stone.clarus.cloudcell_evidence import (
    CloudCellGateConfig,
    TrialPopulation,
    build_cloudcell_artifact,
    evaluate_panel,
)


MTL_LOCATION_TOKENS = ("amygdala", "hippocampus", "entorhinal", "parahippocampal")


def load_sternberg_nwb(
    path: str | Path,
    *,
    mtl_only: bool = True,
) -> tuple[TrialPopulation | None, dict[str, object]]:
    """Load one Sternberg-task NWB file without requiring PyNWB."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-dependent path
        raise RuntimeError("NWB loading requires h5py; install it only for this CLI") from exc

    input_path = Path(path)
    with h5py.File(input_path, "r") as handle:
        units = handle["units"]
        trials = handle["intervals/trials"]
        flat_spikes = np.asarray(units["spike_times"], dtype=float)
        spike_ends = np.asarray(units["spike_times_index"], dtype=np.int64)
        spike_starts = np.concatenate(([0], spike_ends[:-1]))
        all_unit_spikes = [
            flat_spikes[int(start) : int(end)]
            for start, end in zip(spike_starts, spike_ends, strict=True)
        ]
        electrode_rows = np.asarray(units["electrodes"], dtype=np.int64)
        electrode_locations = [
            _decode_scalar(value)
            for value in np.asarray(
                handle["general/extracellular_ephys/electrodes/location"]
            ).reshape(-1)
        ]
        unit_locations = [electrode_locations[int(row)] for row in electrode_rows]
        if mtl_only:
            selected = np.asarray(
                [
                    any(token in location.lower() for token in MTL_LOCATION_TOKENS)
                    for location in unit_locations
                ],
                dtype=bool,
            )
        else:
            selected = np.ones(len(unit_locations), dtype=bool)
        unit_spikes = [
            spikes for spikes, keep in zip(all_unit_spikes, selected, strict=True) if keep
        ]

        trial_ids = np.asarray(trials["id"], dtype=np.int64)
        trial_start = np.asarray(trials["start_time"], dtype=float)
        encoding_start = np.asarray(trials["timestamps_Encoding1"], dtype=float)
        maintenance_start = np.asarray(trials["timestamps_Maintenance"], dtype=float)
        probe_start = np.asarray(trials["timestamps_Probe"], dtype=float)
        response_start = np.asarray(trials["timestamps_Response"], dtype=float)
        memory_load = np.asarray(trials["loads"], dtype=np.int64)
        probe_in_out = np.asarray(trials["probe_in_out"], dtype=np.int64)

        required_lengths = {
            trial_ids.size,
            trial_start.size,
            encoding_start.size,
            maintenance_start.size,
            probe_start.size,
            response_start.size,
            memory_load.size,
            probe_in_out.size,
        }
        if len(required_lengths) != 1:
            raise ValueError(f"{input_path}: NWB trial columns have inconsistent lengths")
        order = np.argsort(trial_start, kind="stable")
        trial_ids = trial_ids[order]
        encoding_start = encoding_start[order]
        maintenance_start = maintenance_start[order]
        probe_start = probe_start[order]
        response_start = response_start[order]
        memory_load = memory_load[order]
        probe_in_out = probe_in_out[order]
        maintenance_midpoint = maintenance_start + 0.5 * (
            probe_start - maintenance_start
        )
        if np.any(maintenance_start <= encoding_start):
            raise ValueError(f"{input_path}: invalid encoding window")
        if np.any(probe_start <= maintenance_start):
            raise ValueError(f"{input_path}: invalid maintenance window")
        if np.any(response_start <= probe_start):
            raise ValueError(f"{input_path}: invalid probe window")

        encoding = _window_rates(unit_spikes, encoding_start, maintenance_start)
        maintenance_early = _window_rates(
            unit_spikes,
            maintenance_start,
            maintenance_midpoint,
        )
        maintenance_late = _window_rates(
            unit_spikes,
            maintenance_midpoint,
            probe_start,
        )
        probe = _window_rates(unit_spikes, probe_start, response_start)
        subject_id = _decode_scalar(handle["general/subject/subject_id"][()])
        identifier = _decode_scalar(handle["identifier"][()])
        related = [
            _decode_scalar(value)
            for value in np.asarray(handle["general/related_publications"]).reshape(-1)
        ]
        description = _decode_scalar(handle["general/experiment_description"][()])

    selected_locations = [
        location for location, keep in zip(unit_locations, selected, strict=True) if keep
    ]
    metadata = {
        "subject_id": subject_id,
        "identifier": identifier,
        "experiment_description": description,
        "related_publications": related,
        "n_trials": int(trial_ids.size),
        "all_unit_count": len(unit_locations),
        "selected_unit_count": len(selected_locations),
        "unit_selection": (
            "MTL location token filter: " + ", ".join(MTL_LOCATION_TOKENS)
            if mtl_only
            else "all recorded locations"
        ),
        "selected_unit_locations": selected_locations,
    }
    if len(selected_locations) < 3:
        metadata["analysis_included"] = False
        metadata["exclusion_reason"] = (
            "fewer than three selected units; population and leave-one-unit-out "
            "comparisons are not identifiable"
        )
        return None, metadata
    data = TrialPopulation(
        subject_id=subject_id,
        encoding=encoding,
        maintenance_early=maintenance_early,
        maintenance_late=maintenance_late,
        probe=probe,
        memory_load=memory_load,
        probe_in_out=probe_in_out,
        trial_ids=trial_ids,
    )
    metadata["analysis_included"] = True
    return data, metadata


def _window_rates(
    unit_spikes: Sequence[np.ndarray],
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Count spikes in half-open windows and divide by trial-specific duration."""

    durations = np.asarray(stops, dtype=float) - np.asarray(starts, dtype=float)
    if np.any(durations <= 0.0):
        raise ValueError("all causal windows must have positive duration")
    rates = np.empty((starts.size, len(unit_spikes)), dtype=float)
    for unit, spikes in enumerate(unit_spikes):
        left = np.searchsorted(spikes, starts, side="left")
        right = np.searchsorted(spikes, stops, side="left")
        rates[:, unit] = (right - left) / durations
    return rates


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one input artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_expected_sha256(
    entries: Sequence[str],
    manifest_path: str | None,
) -> dict[str, str]:
    """Parse ``NAME=HEX`` entries and an optional JSON name-to-hash manifest."""

    expected: dict[str, str] = {}
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("SHA-256 manifest must be a JSON object")
        for name, value in payload.items():
            expected[str(name)] = str(value).lower()
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--expected-sha256 must use NAME=HEX")
        name, value = entry.split("=", 1)
        expected[name] = value.lower()
    for name, value in expected.items():
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid SHA-256 for {name!r}")
    return expected


def _expected_for_path(path: Path, expected: Mapping[str, str]) -> str | None:
    for key in (str(path), path.as_posix(), path.name):
        if key in expected:
            return expected[key]
    return None


def _decode_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-safe CloudCell coding/persistence gate on human MTL NWB files."
        )
    )
    parser.add_argument("nwb", nargs="+", help="One or more Sternberg-task NWB files")
    parser.add_argument("--output", help="Optional JSON artifact output path")
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        metavar="NAME=HEX",
        help="Validate an input by file name/path and SHA-256 (repeatable)",
    )
    parser.add_argument(
        "--sha256-manifest",
        help="Optional JSON object mapping input names/paths to SHA-256 values",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--inner-train-fraction", type=float, default=0.75)
    parser.add_argument("--n-shifts", type=int, default=19)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--min-population-gain", type=float, default=0.02)
    parser.add_argument("--min-dropout-gain", type=float, default=0.0)
    parser.add_argument("--min-local-gain", type=float, default=0.01)
    parser.add_argument("--min-full-over-best-gain", type=float, default=0.01)
    parser.add_argument("--max-null-p", type=float, default=0.05)
    parser.add_argument("--min-subject-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--include-all-locations",
        action="store_true",
        help="Diagnostic override: include MFC as well as MTL units",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        expected = parse_expected_sha256(args.expected_sha256, args.sha256_manifest)
        config = CloudCellGateConfig(
            train_fraction=args.train_fraction,
            inner_train_fraction=args.inner_train_fraction,
            n_shifts=args.n_shifts,
            block_size=args.block_size,
            min_population_gain=args.min_population_gain,
            min_dropout_gain=args.min_dropout_gain,
            min_local_gain=args.min_local_gain,
            min_full_over_best_gain=args.min_full_over_best_gain,
            max_null_p=args.max_null_p,
            min_subject_fraction=args.min_subject_fraction,
        )
    except ValueError as exc:
        parser.error(str(exc))

    datasets: list[TrialPopulation] = []
    provenance: list[dict[str, object]] = []
    matched_expected: set[str] = set()
    for raw_path in args.nwb:
        path = Path(raw_path)
        actual_sha = sha256_file(path)
        expected_sha = _expected_for_path(path, expected)
        if expected_sha is not None and actual_sha != expected_sha:
            parser.error(
                f"SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
        for key in (str(path), path.as_posix(), path.name):
            if key in expected:
                matched_expected.add(key)
        data, metadata = load_sternberg_nwb(
            path,
            mtl_only=not args.include_all_locations,
        )
        if data is not None:
            datasets.append(data)
        provenance.append(
            {
                "input_file": str(path),
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": actual_sha,
                "sha256_verified": True if expected_sha is not None else None,
                **metadata,
            }
        )
    unmatched = set(expected) - matched_expected
    if unmatched:
        parser.error(f"SHA-256 expectations did not match an input: {sorted(unmatched)}")
    if not datasets:
        parser.error("no input retained at least three units after location filtering")

    panel = evaluate_panel(datasets, config)
    artifact = build_cloudcell_artifact(panel, config=config, provenance=provenance)
    rendered = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if artifact["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
