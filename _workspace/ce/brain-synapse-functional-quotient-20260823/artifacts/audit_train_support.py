"""Unlock and audit BA-SRM2 *train-only* event support.

Prerequisite: a frozen, target-blind manifest and its receipt.  This program
reads fitted response values and QC only for manifest-listed train sequences.
It never constructs a query over development or confirmation slices, never
reads waveform BLOBs, and emits counts/missingness only (not response values).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct
from typing import Any, Iterable


SPLIT_SALT = "BA-SRM2-MEDIUM-R21-20260823-V1"
CAP_SALT = "BA-SRM2-MEDIUM-SEQUENCE-CAP-V1:"
MANIFEST_VERSION = "BA-SRM2-TRAIN-MANIFEST-V2-ZERO-BASED"
SUPPORT_AUDITOR_VERSION = "BA-SRM2-TRAIN-SUPPORT-V2.1-QC-DIAGNOSTIC"
MIN_INDEPENDENT_SLICES = 160
HISTORY_PULSES = tuple(range(0, 8))
PRIMARY_TARGET_PULSES = tuple(range(8, 12))
SENSITIVITY_TARGET_PULSES = tuple(range(4, 8))

MANIFEST_KEYS = {
    "version",
    "split",
    "synapse_type",
    "slice_id",
    "slice_ext_id",
    "pair_id",
    "post_recording_id",
    "pre_recording_id",
    "post_stim_name",
    "induction_frequency",
    "recovery_delay",
    "sequence_key",
    "cap_hash",
    "round_robin_index",
}

TARGET_FIELDS = (
    "dec_fit_reconv_amp",
    "dec_fit_latency",
    "dec_fit_rise_time",
    "dec_fit_decay_tau",
)

PAST_FIT_FIELDS = (
    "dec_fit_reconv_amp",
    "baseline_dec_fit_reconv_amp",
    "dec_fit_latency",
    "dec_fit_rise_time",
    "dec_fit_decay_tau",
    "dec_fit_nrmse",
)

PAST_STIM_FIELDS = (
    "previous_pulse_dt",
    "stim_amplitude",
    "stim_duration",
    "n_spikes",
    "first_spike_after_onset",
)


TRAIN_EXTRACTION_SQL = """
SELECT
    ss.sequence_key,
    ss.slice_ext_id AS manifest_slice_ext_id,
    ss.synapse_type AS manifest_synapse_type,
    sl.id AS observed_slice_id,
    sl.ext_id AS observed_slice_ext_id,
    sy.synapse_type AS observed_synapse_type,
    pa.id AS pair_id,
    post_r.id AS post_recording_id,
    pre_r.id AS pre_recording_id,
    sp.id AS stim_pulse_id,
    sp.pulse_number,
    sp.onset_time,
    sp.previous_pulse_dt,
    sp.amplitude AS stim_amplitude,
    sp.duration AS stim_duration,
    sp.n_spikes,
    sp.first_spike_time - sp.onset_time AS first_spike_after_onset,
    sp.qc_pass AS stim_qc_pass,
    pr.ex_qc_pass,
    pr.in_qc_pass,
    prf.dec_fit_reconv_amp,
    prf.baseline_dec_fit_reconv_amp,
    prf.dec_fit_latency,
    prf.dec_fit_rise_time,
    prf.dec_fit_decay_tau,
    prf.dec_fit_nrmse,
    mpp.induction_frequency,
    mpp.recovery_delay,
    post_sr.temperature AS bath_temperature,
    post_pcr.baseline_potential,
    post_pcr.baseline_current,
    post_pcr.baseline_noise_stdev,
    pa.distance AS pair_soma_distance,
    post_tp.input_resistance AS post_input_resistance,
    post_tp.capacitance AS post_capacitance,
    post_tp.time_constant AS post_time_constant,
    pre_c.target_layer AS pre_target_layer,
    post_c.target_layer AS post_target_layer,
    pre_c.cell_class_nonsynaptic AS pre_cell_class,
    post_c.cell_class_nonsynaptic AS post_cell_class
FROM selected_train_sequence ss
JOIN pair pa ON pa.id = ss.pair_id
JOIN experiment ex ON ex.id = pa.experiment_id
JOIN slice sl ON sl.id = ex.slice_id
JOIN synapse sy
  ON sy.pair_id = pa.id AND sy.synapse_type = ss.synapse_type
JOIN cell pre_c ON pre_c.id = pa.pre_cell_id
JOIN cell post_c ON post_c.id = pa.post_cell_id
JOIN pulse_response pr ON pr.pair_id = pa.id
JOIN stim_pulse sp
  ON sp.id = pr.stim_pulse_id
 AND sp.recording_id = ss.pre_recording_id
JOIN recording post_r
  ON post_r.id = pr.recording_id
 AND post_r.id = ss.post_recording_id
JOIN recording pre_r ON pre_r.id = sp.recording_id
JOIN sync_rec post_sr ON post_sr.id = post_r.sync_rec_id
JOIN sync_rec pre_sr ON pre_sr.id = pre_r.sync_rec_id
JOIN patch_clamp_recording post_pcr
  ON post_pcr.recording_id = post_r.id
JOIN multi_patch_probe mpp
  ON mpp.patch_clamp_recording_id = post_pcr.id
LEFT JOIN pulse_response_fit prf ON prf.pulse_response_id = pr.id
LEFT JOIN test_pulse post_tp ON post_tp.id = post_pcr.nearest_test_pulse_id
WHERE post_r.stim_name IS ss.post_stim_name
  AND mpp.induction_frequency = ss.induction_frequency
  AND mpp.recovery_delay = ss.recovery_delay
  AND post_r.id <> pre_r.id
  AND post_r.sync_rec_id = pre_r.sync_rec_id
  AND post_sr.experiment_id = pa.experiment_id
  AND pre_sr.experiment_id = pa.experiment_id
  AND (sp.cell_id IS NULL OR sp.cell_id = pa.pre_cell_id)
  AND pre_r.electrode_id = pre_c.electrode_id
  AND post_r.electrode_id = post_c.electrode_id
ORDER BY ss.sequence_key, sp.pulse_number, sp.onset_time, sp.id
"""


class SupportFailure(RuntimeError):
    """Raised when train-only support cannot be audited safely."""


def assert_train_extraction_sql(sql: str) -> None:
    normalized = " ".join(sql.lower().split())
    if "from selected_train_sequence" not in normalized:
        raise SupportFailure("train extraction must originate from frozen manifest")
    for forbidden in (
        "stim_pulse.data",
        "pulse_response.data",
        "baseline.data",
        " development ",
        " confirmation ",
    ):
        if forbidden in f" {normalized} ":
            raise SupportFailure(f"forbidden train extraction token: {forbidden.strip()}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def split_bucket(group_id: str) -> int:
    material = f"{SPLIT_SALT}:{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10


def canonical_sequence_key(row: dict[str, Any]) -> str:
    stim_name = row["post_stim_name"]
    encoded_stim = (
        "NULL"
        if stim_name is None
        else "UTF8HEX:" + str(stim_name).encode("utf-8").hex()
    )
    frequency = struct.pack(">d", float(row["induction_frequency"])).hex()
    delay = struct.pack(">d", float(row["recovery_delay"])).hex()
    return (
        f"pair_id={int(row['pair_id'])}"
        f"|post_recording_id={int(row['post_recording_id'])}"
        f"|pre_recording_id={int(row['pre_recording_id'])}"
        f"|post_stim_name={encoded_stim}"
        f"|induction_frequency_f64be={frequency}"
        f"|recovery_delay_f64be={delay}"
    )


def expected_cap_hash(sequence_key: str) -> str:
    return sha256_bytes((CAP_SALT + sequence_key).encode("utf-8"))


def finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def load_manifest(
    manifest_path: Path, receipt_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = sha256_bytes(manifest_bytes)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "TRAIN_MANIFEST_FROZEN":
        raise SupportFailure("manifest receipt is not frozen")
    if receipt.get("manifest_sha256") != manifest_hash:
        raise SupportFailure("manifest SHA-256 does not match receipt")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(manifest_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if set(row) != MANIFEST_KEYS:
            extra = sorted(set(row) - MANIFEST_KEYS)
            missing = sorted(MANIFEST_KEYS - set(row))
            raise SupportFailure(
                f"manifest schema mismatch at line {line_no}: extra={extra}, missing={missing}"
            )
        if row["version"] != MANIFEST_VERSION or row["split"] != "train":
            raise SupportFailure("manifest version/split mismatch")
        if row["synapse_type"] not in ("ex", "in"):
            raise SupportFailure("unexpected manifest E/I label")
        if split_bucket(str(row["slice_ext_id"])) > 5:
            raise SupportFailure("manifest includes non-train slice")
        key = str(row["sequence_key"])
        if key != canonical_sequence_key(row):
            raise SupportFailure("manifest canonical sequence key mismatch")
        if row["cap_hash"] != expected_cap_hash(key):
            raise SupportFailure("manifest cap hash mismatch")
        if type(row["round_robin_index"]) is not int or row["round_robin_index"] < 0:
            raise SupportFailure("invalid round-robin index")
        if key in seen:
            raise SupportFailure("duplicate manifest sequence key")
        seen.add(key)
        rows.append(row)
    if len(rows) != int(receipt.get("manifest_rows", -1)):
        raise SupportFailure("manifest row count does not match receipt")
    return rows, receipt, manifest_hash


def create_selected_table(
    con: sqlite3.Connection, manifest_rows: Iterable[dict[str, Any]]
) -> None:
    con.execute(
        """
        CREATE TEMP TABLE selected_train_sequence(
            sequence_key TEXT PRIMARY KEY,
            slice_ext_id TEXT NOT NULL,
            synapse_type TEXT NOT NULL,
            pair_id INTEGER NOT NULL,
            post_recording_id INTEGER NOT NULL,
            pre_recording_id INTEGER NOT NULL,
            post_stim_name TEXT,
            induction_frequency REAL NOT NULL,
            recovery_delay REAL NOT NULL
        )
        """
    )
    con.executemany(
        """
        INSERT INTO selected_train_sequence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                row["sequence_key"],
                row["slice_ext_id"],
                row["synapse_type"],
                row["pair_id"],
                row["post_recording_id"],
                row["pre_recording_id"],
                row["post_stim_name"],
                row["induction_frequency"],
                row["recovery_delay"],
            )
            for row in manifest_rows
        ),
    )


def stimulus_qc_pass(rows: list[sqlite3.Row]) -> bool:
    return all(row["stim_qc_pass"] == 1 for row in rows)


def response_qc_pass(rows: list[sqlite3.Row], synapse_type: str) -> bool:
    response_field = "ex_qc_pass" if synapse_type == "ex" else "in_qc_pass"
    return all(row[response_field] == 1 for row in rows)


def sequence_qc_pass(rows: list[sqlite3.Row], synapse_type: str) -> bool:
    return stimulus_qc_pass(rows) and response_qc_pass(rows, synapse_type)


def complete_target(rows: list[sqlite3.Row], pulses: range) -> bool:
    selected = {int(row["pulse_number"]): row for row in rows}
    return all(
        pulse in selected and all(finite(selected[pulse][field]) for field in TARGET_FIELDS)
        for pulse in pulses
    )


def positive_finite_mad(values: Iterable[Any]) -> bool:
    finite_values = [float(value) for value in values if finite(value)]
    if not finite_values:
        return False
    center = statistics.median(finite_values)
    mad = statistics.median(abs(value - center) for value in finite_values)
    return math.isfinite(mad) and mad > 0.0


def audit_support(
    database: Path, manifest_path: Path, manifest_receipt_path: Path
) -> dict[str, Any]:
    manifest_rows, manifest_receipt, manifest_hash = load_manifest(
        manifest_path, manifest_receipt_path
    )
    resolved = database.resolve(strict=True)
    expected_db_hash = manifest_receipt.get("database_sha256")
    observed_db_hash = sha256_file(resolved)
    if expected_db_hash != observed_db_hash:
        raise SupportFailure("database SHA-256 differs from frozen manifest input")

    uri = f"file:{resolved.as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        create_selected_table(con, manifest_rows)
        assert_train_extraction_sql(TRAIN_EXTRACTION_SQL)
        extracted = list(con.execute(TRAIN_EXTRACTION_SQL))
    finally:
        con.close()

    by_sequence: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in extracted:
        by_sequence[str(row["sequence_key"])].append(row)

    manifest_by_key = {str(row["sequence_key"]): row for row in manifest_rows}
    structural_valid: dict[str, bool] = {}
    stimulus_qc_valid: dict[str, bool] = {}
    response_qc_valid: dict[str, bool] = {}
    qc_valid: dict[str, bool] = {}
    primary_complete: dict[str, bool] = {}
    sensitivity_complete: dict[str, bool] = {}
    target_finite_counts: Counter[str] = Counter()
    past_finite_counts: Counter[str] = Counter()
    extraction_failures: Counter[str] = Counter()

    for key, manifest_row in manifest_by_key.items():
        rows = by_sequence.get(key, [])
        pulse_numbers = [int(row["pulse_number"]) for row in rows]
        structural = len(rows) == 12 and sorted(pulse_numbers) == list(range(0, 12))
        structural_valid[key] = structural
        if not structural:
            extraction_failures["not_exact_12_unique_pulses"] += 1
            stimulus_qc_valid[key] = False
            response_qc_valid[key] = False
            qc_valid[key] = False
            primary_complete[key] = False
            sensitivity_complete[key] = False
            continue
        if any(str(row["observed_slice_ext_id"]) != manifest_row["slice_ext_id"] for row in rows):
            raise SupportFailure("observed slice differs from frozen manifest")
        if any(int(row["observed_slice_id"]) != int(manifest_row["slice_id"]) for row in rows):
            raise SupportFailure("observed slice ID differs from frozen manifest")
        if any(row["observed_synapse_type"] != manifest_row["synapse_type"] for row in rows):
            raise SupportFailure("observed synapse type differs from frozen manifest")

        stimulus_qc_valid[key] = stimulus_qc_pass(rows)
        response_qc_valid[key] = response_qc_pass(
            rows, manifest_row["synapse_type"]
        )
        qc_valid[key] = (
            stimulus_qc_valid[key] and response_qc_valid[key]
        )
        primary_complete[key] = complete_target(rows, PRIMARY_TARGET_PULSES)
        sensitivity_complete[key] = complete_target(
            rows, SENSITIVITY_TARGET_PULSES
        )
        for row in rows:
            pulse = int(row["pulse_number"])
            if pulse in PRIMARY_TARGET_PULSES:
                for field in TARGET_FIELDS:
                    if finite(row[field]):
                        target_finite_counts[f"p{pulse}:{field}"] += 1
            if pulse in HISTORY_PULSES:
                for field in PAST_STIM_FIELDS + PAST_FIT_FIELDS:
                    if finite(row[field]):
                        past_finite_counts[f"p{pulse}:{field}"] += 1

    def stratum_summary(label: str) -> dict[str, Any]:
        keys = [
            str(row["sequence_key"])
            for row in manifest_rows
            if row["synapse_type"] == label
        ]
        eligible = [
            key
            for key in keys
            if structural_valid.get(key, False)
            and qc_valid.get(key, False)
            and primary_complete.get(key, False)
        ]
        sensitivity = [
            key
            for key in keys
            if structural_valid.get(key, False)
            and qc_valid.get(key, False)
            and sensitivity_complete.get(key, False)
        ]
        slices = {manifest_by_key[key]["slice_ext_id"] for key in eligible}
        coordinate_values: dict[str, list[float]] = {
            f"p{pulse}:{field}": []
            for pulse in PRIMARY_TARGET_PULSES
            for field in TARGET_FIELDS
        }
        for key in eligible:
            pulse_rows = {
                int(row["pulse_number"]): row for row in by_sequence[key]
            }
            for pulse in PRIMARY_TARGET_PULSES:
                for field in TARGET_FIELDS:
                    coordinate_values[f"p{pulse}:{field}"].append(
                        float(pulse_rows[pulse][field])
                    )
        target_mad_pass = {
            coordinate: positive_finite_mad(values)
            for coordinate, values in coordinate_values.items()
        }
        all_target_mads_pass = all(target_mad_pass.values())
        geometry_support_pass = (
            len(slices) >= MIN_INDEPENDENT_SLICES and all_target_mads_pass
        )
        return {
            "manifest_sequences": len(keys),
            "structural_sequences": sum(structural_valid.get(key, False) for key in keys),
            "all_stimulus_qc_pass_sequences": sum(
                stimulus_qc_valid.get(key, False) for key in keys
            ),
            "all_type_matched_response_qc_pass_sequences": sum(
                response_qc_valid.get(key, False) for key in keys
            ),
            "qc_pass_sequences": sum(qc_valid.get(key, False) for key in keys),
            "primary_complete_sequences": sum(primary_complete.get(key, False) for key in keys),
            "primary_complete_fraction_of_manifest": (
                sum(primary_complete.get(key, False) for key in keys) / len(keys)
                if keys else None
            ),
            "primary_eligible_sequences": len(eligible),
            "primary_eligible_slices": len(slices),
            "strict_primary_eligible_fraction_of_manifest": (
                len(eligible) / len(keys) if keys else None
            ),
            "sensitivity_eligible_sequences": len(sensitivity),
            "target_coordinate_count": len(target_mad_pass),
            "target_coordinate_n": {
                coordinate: len(values)
                for coordinate, values in coordinate_values.items()
            },
            "target_positive_finite_mad_count": sum(target_mad_pass.values()),
            "target_positive_finite_mad_pass": target_mad_pass,
            "all_target_positive_finite_mad_pass": all_target_mads_pass,
            "geometry_support_pass": geometry_support_pass,
        }

    strata = {label: stratum_summary(label) for label in ("ex", "in")}
    passed = all(item["geometry_support_pass"] for item in strata.values())
    return {
        "status": "PASS_TRAIN_SUPPORT" if passed else "STOP_INSUFFICIENT_TRAIN_SUPPORT",
        "version": SUPPORT_AUDITOR_VERSION,
        "database": str(resolved),
        "database_sha256": observed_db_hash,
        "manifest": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": manifest_hash,
        "manifest_rows": len(manifest_rows),
        "split": "train-only",
        "train_outcome_values_read": True,
        "development_outcomes_read": False,
        "confirmation_outcomes_read": False,
        "waveform_blobs_read": False,
        "minimum_independent_slices": MIN_INDEPENDENT_SLICES,
        "extracted_event_rows": len(extracted),
        "extracted_sequence_count": len(by_sequence),
        "strata": strata,
        "extraction_failure_counts": dict(sorted(extraction_failures.items())),
        "target_finite_counts": dict(sorted(target_finite_counts.items())),
        "past_input_finite_counts": dict(sorted(past_finite_counts.items())),
        "model_fit_unlocked": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = audit_support(
            args.database, args.manifest, args.manifest_receipt
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
