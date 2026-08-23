"""BA-SRM3 train-only support gate using official type-matched response QC.

This is a new post-BA-SRM2 candidate.  It reuses the exact frozen target-blind
3,000-sequence manifest, reads no development/confirmation rows or BLOBs, and
derives the complete 16-target train cohort without resampling.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any, Iterable


VERSION = "BA-SRM3-RESPONSE-QC-SUPPORT-V1"
EXPECTED_MANIFEST_SHA256 = (
    "4ddb4a52294a55b011c5118a02432ca28c057ca5b5ebb63d8d7c945923aa62c2"
)
EXPECTED_DATABASE_SHA256 = (
    "dbf19786f9e0d0d73c26351dc29d69ef8c10a2e67e32e19ac73034a5624d48c5"
)
EXPECTED_PARENT_HELPER_SHA256 = (
    "d0f521a48c22f532cbdd0ff808d70647da627c585c84af2a4d8addff1b941a0d"
)
EXPECTED_SCHEMA_RECEIPT_SHA256 = (
    "a94c940e1426d9968bcb48ac20343c00b106d944d4620d55ccad0d7106eb9cc0"
)
MIN_DISTINCT_SLICE_GROUPS = 160


def _load_parent_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "brain-synapse-functional-quotient-20260823"
        / "artifacts"
        / "audit_train_support.py"
    )
    observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash != EXPECTED_PARENT_HELPER_SHA256:
        raise RuntimeError("frozen BA-SRM2 helper SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("ba_srm2_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen BA-SRM2 support helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARENT = _load_parent_module()


class SupportFailure(RuntimeError):
    """Raised when BA-SRM3 support provenance or relations fail."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def positive_finite_mad(values: Iterable[Any]) -> bool:
    observed = [float(value) for value in values if PARENT.finite(value)]
    if not observed:
        return False
    center = statistics.median(observed)
    mad = statistics.median(abs(value - center) for value in observed)
    return math.isfinite(mad) and mad > 0.0


def type_matched_response_qc(rows: list[sqlite3.Row], synapse_type: str) -> bool:
    field = "ex_qc_pass" if synapse_type == "ex" else "in_qc_pass"
    return all(row[field] == 1 for row in rows)


def exact_zero_based_sequence(rows: list[sqlite3.Row]) -> bool:
    if len(rows) != 12:
        return False
    ordered = sorted(rows, key=lambda row: int(row["pulse_number"]))
    if [int(row["pulse_number"]) for row in ordered] != list(range(12)):
        return False
    onsets = [row["onset_time"] for row in ordered]
    if not all(PARENT.finite(value) for value in onsets):
        return False
    return all(float(right) > float(left) for left, right in zip(onsets, onsets[1:]))


def validate_schema_receipt(path: Path) -> str:
    resolved = path.resolve(strict=True)
    digest = PARENT.sha256_file(resolved)
    if digest != EXPECTED_SCHEMA_RECEIPT_SHA256:
        raise SupportFailure("schema receipt SHA-256 mismatch")
    receipt = json.loads(resolved.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_SCHEMA_ONLY":
        raise SupportFailure("schema receipt did not pass")
    if receipt.get("sha256") != EXPECTED_DATABASE_SHA256:
        raise SupportFailure("schema receipt database SHA-256 mismatch")
    if not receipt.get("relations_and_order", {}).get("relation_order_pass"):
        raise SupportFailure("schema receipt relation/order gate did not pass")
    if receipt.get("outcome_values_read") is not False:
        raise SupportFailure("schema receipt outcome boundary is invalid")
    return digest


def eligible_manifest_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def audit(
    database: Path,
    manifest_path: Path,
    manifest_receipt_path: Path,
    schema_receipt_path: Path,
) -> tuple[dict[str, Any], bytes]:
    schema_receipt_hash = validate_schema_receipt(schema_receipt_path)
    manifest_rows, manifest_receipt, manifest_hash = PARENT.load_manifest(
        manifest_path, manifest_receipt_path
    )
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise SupportFailure("manifest is not the frozen BA-SRM3 input")
    if manifest_receipt.get("database_sha256") != EXPECTED_DATABASE_SHA256:
        raise SupportFailure("manifest receipt has unexpected database SHA-256")

    resolved = database.resolve(strict=True)
    database_hash = PARENT.sha256_file(resolved)
    if database_hash != EXPECTED_DATABASE_SHA256:
        raise SupportFailure("database SHA-256 mismatch")

    uri = f"file:{resolved.as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        PARENT.create_selected_table(con, manifest_rows)
        PARENT.assert_train_extraction_sql(PARENT.TRAIN_EXTRACTION_SQL)
        extracted = list(con.execute(PARENT.TRAIN_EXTRACTION_SQL))
    finally:
        con.close()

    by_sequence: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in extracted:
        by_sequence[str(row["sequence_key"])].append(row)
    manifest_by_key = {str(row["sequence_key"]): row for row in manifest_rows}
    if set(by_sequence) != set(manifest_by_key):
        raise SupportFailure("extracted sequence keys differ from frozen manifest")

    response_qc: dict[str, bool] = {}
    primary_complete: dict[str, bool] = {}
    sensitivity_complete: dict[str, bool] = {}
    stimulus_qc_rows: Counter[str] = Counter()
    for key, manifest_row in manifest_by_key.items():
        rows = by_sequence[key]
        if not exact_zero_based_sequence(rows):
            raise SupportFailure("selected sequence is not exact pulse 0..11")
        if any(int(row["observed_slice_id"]) != int(manifest_row["slice_id"]) for row in rows):
            raise SupportFailure("observed slice ID differs from manifest")
        if any(str(row["observed_slice_ext_id"]) != manifest_row["slice_ext_id"] for row in rows):
            raise SupportFailure("observed slice ext ID differs from manifest")
        if any(row["observed_synapse_type"] != manifest_row["synapse_type"] for row in rows):
            raise SupportFailure("observed E/I type differs from manifest")

        response_qc[key] = type_matched_response_qc(
            rows, manifest_row["synapse_type"]
        )
        primary_complete[key] = PARENT.complete_target(
            rows, PARENT.PRIMARY_TARGET_PULSES
        )
        sensitivity_complete[key] = PARENT.complete_target(
            rows, PARENT.SENSITIVITY_TARGET_PULSES
        )
        for row in rows:
            value = row["stim_qc_pass"]
            label = "NULL" if value is None else str(int(value))
            stimulus_qc_rows[label] += 1

    eligible_rows: list[dict[str, Any]] = []
    strata: dict[str, Any] = {}
    for label in ("ex", "in"):
        keys = sorted(
            (
                str(row["sequence_key"])
                for row in manifest_rows
                if row["synapse_type"] == label
            )
        )
        eligible = [
            key for key in keys if response_qc[key] and primary_complete[key]
        ]
        sensitivity = [
            key for key in keys if response_qc[key] and sensitivity_complete[key]
        ]
        slices = {manifest_by_key[key]["slice_ext_id"] for key in eligible}
        coordinates: dict[str, list[float]] = {
            f"p{pulse}:{field}": []
            for pulse in PARENT.PRIMARY_TARGET_PULSES
            for field in PARENT.TARGET_FIELDS
        }
        for key in eligible:
            pulse_rows = {
                int(row["pulse_number"]): row for row in by_sequence[key]
            }
            for pulse in PARENT.PRIMARY_TARGET_PULSES:
                for field in PARENT.TARGET_FIELDS:
                    coordinates[f"p{pulse}:{field}"].append(
                        float(pulse_rows[pulse][field])
                    )
            source = manifest_by_key[key]
            eligible_rows.append(
                {
                    "version": VERSION,
                    "split": "train",
                    "synapse_type": label,
                    "slice_id": int(source["slice_id"]),
                    "slice_ext_id": source["slice_ext_id"],
                    "sequence_key": key,
                }
            )
        mad_pass = {
            coordinate: positive_finite_mad(values)
            for coordinate, values in coordinates.items()
        }
        all_mad_pass = all(mad_pass.values())
        support_pass = len(slices) >= MIN_DISTINCT_SLICE_GROUPS and all_mad_pass
        strata[label] = {
            "manifest_sequences": len(keys),
            "response_qc_pass_sequences": sum(response_qc[key] for key in keys),
            "primary_complete_sequences": sum(primary_complete[key] for key in keys),
            "response_qc_and_primary_complete_sequences": len(eligible),
            "response_qc_and_primary_complete_slices": len(slices),
            "sensitivity_eligible_sequences": len(sensitivity),
            "target_coordinate_count": len(coordinates),
            "target_coordinate_n": {
                coordinate: len(values) for coordinate, values in coordinates.items()
            },
            "target_positive_finite_mad_pass": mad_pass,
            "all_target_positive_finite_mad_pass": all_mad_pass,
            "geometry_support_pass": support_pass,
        }

    eligible_rows.sort(
        key=lambda row: (
            row["synapse_type"],
            row["slice_ext_id"].encode("utf-8"),
            row["sequence_key"],
        )
    )
    rendered_manifest = eligible_manifest_bytes(eligible_rows)
    passed = all(strata[label]["geometry_support_pass"] for label in ("ex", "in"))
    receipt = {
        "status": "PASS_TRAIN_SUPPORT" if passed else "STOP_TRAIN_SUPPORT",
        "version": VERSION,
        "parent_ba_srm2_stop_preserved": True,
        "qc_rule": "all 12 type-matched PulseResponse QC values equal 1",
        "stim_pulse_qc_used_for_selection": False,
        "stim_pulse_qc_row_counts_diagnostic_only": dict(stimulus_qc_rows),
        "database": str(resolved),
        "database_sha256": database_hash,
        "schema_receipt": str(schema_receipt_path.resolve(strict=True)),
        "schema_receipt_sha256": schema_receipt_hash,
        "parent_helper_sha256": EXPECTED_PARENT_HELPER_SHA256,
        "manifest": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": manifest_hash,
        "manifest_rows": len(manifest_rows),
        "eligible_manifest_rows": len(eligible_rows),
        "eligible_manifest_sha256": sha256_bytes(rendered_manifest),
        "split": "train-only",
        "train_outcome_values_read": True,
        "development_outcomes_read": False,
        "confirmation_outcomes_read": False,
        "waveform_blobs_read": False,
        "minimum_distinct_slice_groups": MIN_DISTINCT_SLICE_GROUPS,
        "extracted_event_rows": len(extracted),
        "strata": strata,
        "model_fit_unlocked": passed,
    }
    return receipt, rendered_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--schema-receipt", type=Path, required=True)
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt, manifest_bytes = audit(
            args.database,
            args.manifest,
            args.manifest_receipt,
            args.schema_receipt,
        )
        args.eligible_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.eligible_manifest.write_bytes(manifest_bytes)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (SupportFailure, OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_TRAIN_SUPPORT", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS_TRAIN_SUPPORT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
