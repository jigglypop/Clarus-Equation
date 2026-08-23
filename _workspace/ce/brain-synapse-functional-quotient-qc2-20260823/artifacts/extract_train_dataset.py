"""Extract the frozen BA-SRM3 eligible train cohort into dimensionless arrays."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np


VERSION = "BA-SRM3-TRAIN-DATASET-V1"
EXPECTED_SUPPORT_RECEIPT_SHA256 = (
    "0f19eb14d3d133b9a181fb69aa5497b191d392313d3c1c830abdb0b95cd83247"
)
EXPECTED_ELIGIBLE_MANIFEST_SHA256 = (
    "74d6d3b142e48d7906305e133983b91cc8227a40748335985414e674dd1fd81c"
)

TIME_REF_S = 1e-3
VOLTAGE_REF_V = 1e-3
CURRENT_REF_A = 1e-12
RESISTANCE_REF_OHM = 1e6
CAPACITANCE_REF_F = 1e-12
LENGTH_REF_M = 1e-4
TEMPERATURE_REF_K = 310.0

PULSE_NUMERIC_SPECS = (
    ("previous_pulse_dt", "time"),
    ("stim_amplitude", "current"),
    ("stim_duration", "time"),
    ("n_spikes", "count"),
    ("first_spike_after_onset", "time"),
    ("dec_fit_reconv_amp", "voltage"),
    ("baseline_dec_fit_reconv_amp", "voltage"),
    ("dec_fit_latency", "time"),
    ("dec_fit_rise_time", "time"),
    ("dec_fit_decay_tau", "time"),
    ("dec_fit_nrmse", "count"),
)

STATIC_NUMERIC_SPECS = (
    ("induction_frequency", "frequency"),
    ("recovery_delay", "time"),
    ("bath_temperature", "temperature_c"),
    ("baseline_potential", "voltage"),
    ("baseline_current", "current"),
    ("baseline_noise_stdev", "voltage"),
    ("pair_soma_distance", "length"),
    ("post_input_resistance", "resistance"),
    ("post_capacitance", "capacitance"),
    ("post_time_constant", "time"),
)

CATEGORY_NAMES = (
    "post_stim_name",
    "pre_target_layer",
    "post_target_layer",
    "pre_cell_class",
    "post_cell_class",
)

TARGET_SPECS = (
    ("dec_fit_reconv_amp", "voltage"),
    ("dec_fit_latency", "time"),
    ("dec_fit_rise_time", "time"),
    ("dec_fit_decay_tau", "time"),
)


class ExtractionFailure(RuntimeError):
    """Raised when frozen train extraction cannot be reproduced."""


def _load_support_module():
    path = Path(__file__).with_name("audit_response_qc_support.py")
    spec = importlib.util.spec_from_file_location("ba_srm3_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load BA-SRM3 support module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = _load_support_module()
PARENT = SUPPORT.PARENT


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dimensionless(value: Any, kind: str) -> float:
    if value is None:
        return float("nan")
    try:
        raw = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    if not math.isfinite(raw):
        return float("nan")
    if kind == "time":
        return raw / TIME_REF_S
    if kind == "voltage":
        return raw / VOLTAGE_REF_V
    if kind == "current":
        return raw / CURRENT_REF_A
    if kind == "resistance":
        return raw / RESISTANCE_REF_OHM
    if kind == "capacitance":
        return raw / CAPACITANCE_REF_F
    if kind == "length":
        return raw / LENGTH_REF_M
    if kind == "temperature_c":
        return (raw + 273.15) / TEMPERATURE_REF_K
    if kind == "frequency":
        return raw * TIME_REF_S
    if kind == "count":
        return raw
    raise ExtractionFailure(f"unknown unit kind: {kind}")


def load_eligible_manifest(path: Path, support_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_ELIGIBLE_MANIFEST_SHA256:
        raise ExtractionFailure("eligible manifest SHA-256 mismatch")
    if support_receipt.get("eligible_manifest_sha256") != digest:
        raise ExtractionFailure("support receipt eligible-manifest mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    expected_keys = {
        "version",
        "split",
        "synapse_type",
        "slice_id",
        "slice_ext_id",
        "sequence_key",
    }
    seen: set[str] = set()
    for row in rows:
        if set(row) != expected_keys or row["split"] != "train":
            raise ExtractionFailure("eligible manifest schema/split mismatch")
        if row["synapse_type"] not in ("ex", "in"):
            raise ExtractionFailure("eligible manifest E/I mismatch")
        if PARENT.split_bucket(str(row["slice_ext_id"])) > 5:
            raise ExtractionFailure("eligible manifest contains non-train slice")
        key = str(row["sequence_key"])
        if key in seen:
            raise ExtractionFailure("duplicate eligible sequence key")
        seen.add(key)
    if len(rows) != int(support_receipt.get("eligible_manifest_rows", -1)):
        raise ExtractionFailure("eligible manifest row count mismatch")
    return rows


def load_support_receipt(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != EXPECTED_SUPPORT_RECEIPT_SHA256:
        raise ExtractionFailure("support receipt SHA-256 mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_TRAIN_SUPPORT" or not receipt.get(
        "model_fit_unlocked"
    ):
        raise ExtractionFailure("support receipt does not unlock model fit")
    if receipt.get("development_outcomes_read") is not False or receipt.get(
        "confirmation_outcomes_read"
    ) is not False:
        raise ExtractionFailure("support receipt sealing boundary mismatch")
    return receipt


def build_arrays(
    manifest_rows: list[dict[str, Any]],
    event_rows: list[sqlite3.Row],
) -> dict[str, np.ndarray]:
    by_sequence: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in event_rows:
        by_sequence[str(row["sequence_key"])].append(row)
    source_by_key = {str(row["sequence_key"]): row for row in manifest_rows}
    if set(by_sequence) != set(source_by_key):
        raise ExtractionFailure("event keys differ from eligible manifest")

    numeric_names = [
        f"p{pulse}:{field}"
        for pulse in PARENT.HISTORY_PULSES
        for field, _ in PULSE_NUMERIC_SPECS
    ] + [f"static:{field}" for field, _ in STATIC_NUMERIC_SPECS]
    target_names = [
        f"p{pulse}:{field}"
        for pulse in PARENT.PRIMARY_TARGET_PULSES
        for field, _ in TARGET_SPECS
    ]

    numeric_rows: list[list[float]] = []
    category_rows: list[list[object]] = []
    target_rows: list[list[float]] = []
    sequence_keys: list[str] = []
    slice_ids: list[str] = []
    synapse_types: list[str] = []

    ordered_manifest = sorted(
        manifest_rows,
        key=lambda row: (
            row["synapse_type"],
            str(row["slice_ext_id"]).encode("utf-8"),
            row["sequence_key"],
        ),
    )
    for source in ordered_manifest:
        key = str(source["sequence_key"])
        rows = by_sequence[key]
        if not SUPPORT.exact_zero_based_sequence(rows):
            raise ExtractionFailure("eligible event sequence lost 0..11 order")
        if not SUPPORT.type_matched_response_qc(rows, source["synapse_type"]):
            raise ExtractionFailure("eligible sequence lost response QC")
        if not PARENT.complete_target(rows, PARENT.PRIMARY_TARGET_PULSES):
            raise ExtractionFailure("eligible sequence lost complete target")
        pulses = {int(row["pulse_number"]): row for row in rows}

        numeric: list[float] = []
        for pulse in PARENT.HISTORY_PULSES:
            row = pulses[pulse]
            numeric.extend(
                dimensionless(row[field], kind)
                for field, kind in PULSE_NUMERIC_SPECS
            )
        first = pulses[0]
        numeric.extend(
            dimensionless(first[field], kind)
            for field, kind in STATIC_NUMERIC_SPECS
        )
        categories = [
            source["post_stim_name"],
            first["pre_target_layer"],
            first["post_target_layer"],
            first["pre_cell_class"],
            first["post_cell_class"],
        ]
        target: list[float] = []
        for pulse in PARENT.PRIMARY_TARGET_PULSES:
            row = pulses[pulse]
            target.extend(
                dimensionless(row[field], kind) for field, kind in TARGET_SPECS
            )
        if not np.all(np.isfinite(target)):
            raise ExtractionFailure("eligible target became nonfinite after unit conversion")
        numeric_rows.append(numeric)
        category_rows.append(categories)
        target_rows.append(target)
        sequence_keys.append(key)
        slice_ids.append(str(source["slice_ext_id"]))
        synapse_types.append(str(source["synapse_type"]))

    return {
        "numeric": np.asarray(numeric_rows, dtype=float),
        "categorical": np.asarray(category_rows, dtype=str),
        "target": np.asarray(target_rows, dtype=float),
        "sequence_key": np.asarray(sequence_keys, dtype=str),
        "slice_ext_id": np.asarray(slice_ids, dtype=str),
        "synapse_type": np.asarray(synapse_types, dtype=str),
        "numeric_feature_names": np.asarray(numeric_names, dtype=str),
        "categorical_feature_names": np.asarray(CATEGORY_NAMES, dtype=str),
        "target_names": np.asarray(target_names, dtype=str),
    }


def verify_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as observed:
        if set(observed.files) != set(arrays):
            raise ExtractionFailure("dataset NPZ key mismatch after write")
        for key, expected in arrays.items():
            actual = observed[key]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ExtractionFailure(f"dataset NPZ shape/dtype mismatch: {key}")
            if np.issubdtype(expected.dtype, np.floating):
                equal = np.array_equal(actual, expected, equal_nan=True)
            else:
                equal = np.array_equal(actual, expected)
            if not equal:
                raise ExtractionFailure(f"dataset NPZ value mismatch: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-receipt", type=Path, required=True)
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--support-receipt", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.dataset.exists() or args.output.exists():
            raise ExtractionFailure("refusing to overwrite frozen dataset/receipt")
        if args.dataset.suffix.lower() != ".npz":
            raise ExtractionFailure("dataset output must end in .npz")
        support_receipt = load_support_receipt(args.support_receipt)
        eligible = load_eligible_manifest(args.eligible_manifest, support_receipt)
        full_manifest, _, _ = PARENT.load_manifest(
            args.source_manifest, args.source_manifest_receipt
        )
        full_by_key = {str(row["sequence_key"]): row for row in full_manifest}
        selected_full = []
        for item in eligible:
            source = full_by_key.get(str(item["sequence_key"]))
            if source is None:
                raise ExtractionFailure("eligible key absent from source manifest")
            for field in ("slice_id", "slice_ext_id", "synapse_type"):
                if source[field] != item[field]:
                    raise ExtractionFailure("eligible/source manifest identity mismatch")
            selected_full.append(source)

        resolved = args.database.resolve(strict=True)
        if PARENT.sha256_file(resolved) != SUPPORT.EXPECTED_DATABASE_SHA256:
            raise ExtractionFailure("database SHA-256 mismatch")
        con = sqlite3.connect(
            f"file:{resolved.as_posix()}?mode=ro&immutable=1", uri=True
        )
        con.row_factory = sqlite3.Row
        try:
            PARENT.create_selected_table(con, selected_full)
            PARENT.assert_train_extraction_sql(PARENT.TRAIN_EXTRACTION_SQL)
            events = list(con.execute(PARENT.TRAIN_EXTRACTION_SQL))
        finally:
            con.close()
        arrays = build_arrays(selected_full, events)
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial_dataset = args.dataset.with_name(args.dataset.name + ".partial.npz")
        if partial_dataset.exists():
            raise ExtractionFailure("stale partial dataset exists")
        np.savez_compressed(partial_dataset, **arrays)
        verify_npz(partial_dataset, arrays)
        partial_dataset.replace(args.dataset)
        verify_npz(args.dataset, arrays)
        dataset_hash = sha256_file(args.dataset)
        receipt = {
            "status": "PASS_TRAIN_DATASET",
            "version": VERSION,
            "database_sha256": SUPPORT.EXPECTED_DATABASE_SHA256,
            "support_receipt_sha256": EXPECTED_SUPPORT_RECEIPT_SHA256,
            "eligible_manifest_sha256": EXPECTED_ELIGIBLE_MANIFEST_SHA256,
            "dataset": str(args.dataset.resolve(strict=True)),
            "dataset_sha256": dataset_hash,
            "rows": int(arrays["target"].shape[0]),
            "numeric_features": int(arrays["numeric"].shape[1]),
            "categorical_features": int(arrays["categorical"].shape[1]),
            "target_coordinates": int(arrays["target"].shape[1]),
            "strata": {
                label: {
                    "rows": int(np.sum(arrays["synapse_type"] == label)),
                    "slice_groups": int(
                        np.unique(
                            arrays["slice_ext_id"][arrays["synapse_type"] == label]
                        ).size
                    ),
                }
                for label in ("ex", "in")
            },
            "numeric_finite_counts": {
                name: int(np.sum(np.isfinite(arrays["numeric"][:, idx])))
                for idx, name in enumerate(arrays["numeric_feature_names"])
            },
            "all_targets_finite": bool(np.all(np.isfinite(arrays["target"]))),
            "train_outcomes_read": True,
            "development_outcomes_read": False,
            "confirmation_outcomes_read": False,
            "waveform_blobs_read": False,
        }
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ExtractionFailure, OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_TRAIN_DATASET", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
