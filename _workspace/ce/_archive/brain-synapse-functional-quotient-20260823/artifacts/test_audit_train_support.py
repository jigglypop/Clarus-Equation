from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("audit_train_support.py")
SPEC = importlib.util.spec_from_file_location("audit_train_support", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def event(pulse, *, ex_qc=1, in_qc=0, stim_qc=1, target=1.0):
    row = {
        "pulse_number": pulse,
        "ex_qc_pass": ex_qc,
        "in_qc_pass": in_qc,
        "stim_qc_pass": stim_qc,
    }
    for field in support.TARGET_FIELDS:
        row[field] = target
    return row


def manifest_row(**updates):
    row = {
        "version": support.MANIFEST_VERSION,
        "split": "train",
        "synapse_type": "ex",
        "slice_id": 1,
        "slice_ext_id": "slice-a",
        "pair_id": 2,
        "post_recording_id": 3,
        "pre_recording_id": 4,
        "post_stim_name": None,
        "induction_frequency": 50.0,
        "recovery_delay": 0.25,
        "round_robin_index": 0,
    }
    row.update(updates)
    if "sequence_key" not in updates:
        row["sequence_key"] = support.canonical_sequence_key(row)
    if "cap_hash" not in updates:
        row["cap_hash"] = support.expected_cap_hash(row["sequence_key"])
    return row


def write_manifest(tmp_path, rows):
    raw = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    manifest_path = tmp_path / "manifest.jsonl"
    receipt_path = tmp_path / "receipt.json"
    manifest_path.write_bytes(raw)
    receipt_path.write_text(
        json.dumps(
            {
                "status": "TRAIN_MANIFEST_FROZEN",
                "manifest_sha256": support.sha256_bytes(raw),
                "manifest_rows": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, receipt_path


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf"), "x"])
def test_finite_rejects_missing_and_nonfinite(value):
    assert support.finite(value) is False


@pytest.mark.parametrize("value", [0.0, -1.0, 2.5])
def test_finite_keeps_signed_and_zero_values(value):
    assert support.finite(value) is True


def test_complete_target_requires_all_four_future_pulses():
    rows = [event(pulse) for pulse in range(0, 12)]
    assert support.complete_target(rows, range(8, 12))
    rows[9][support.TARGET_FIELDS[2]] = None
    assert not support.complete_target(rows, range(8, 12))


def test_zero_based_history_primary_and_sensitivity_ranges_are_frozen():
    assert support.HISTORY_PULSES == tuple(range(0, 8))
    assert support.PRIMARY_TARGET_PULSES == tuple(range(8, 12))
    assert support.SENSITIVITY_TARGET_PULSES == tuple(range(4, 8))


def test_qc_uses_only_matching_synapse_type():
    rows = [event(pulse, ex_qc=1, in_qc=0) for pulse in range(0, 12)]
    assert support.sequence_qc_pass(rows, "ex")
    assert not support.sequence_qc_pass(rows, "in")


def test_qc_requires_every_stimulus_pulse():
    rows = [event(pulse) for pulse in range(0, 12)]
    rows[-1]["stim_qc_pass"] = 0
    assert not support.sequence_qc_pass(rows, "ex")


def test_positive_finite_mad_rejects_constant_coordinate():
    assert not support.positive_finite_mad([1.0, 1.0, 1.0])
    assert support.positive_finite_mad([0.0, 1.0, 2.0])


def test_train_sql_is_manifest_scoped_and_blob_free():
    support.assert_train_extraction_sql(support.TRAIN_EXTRACTION_SQL)
    lowered = support.TRAIN_EXTRACTION_SQL.lower()
    assert "stim_pulse.data" not in lowered
    assert "pulse_response.data" not in lowered


def test_train_sql_rejects_unscoped_or_blob_query():
    with pytest.raises(support.SupportFailure):
        support.assert_train_extraction_sql("SELECT x FROM pulse_response")
    with pytest.raises(support.SupportFailure):
        support.assert_train_extraction_sql(
            "SELECT stim_pulse.data FROM selected_train_sequence"
        )


def test_manifest_hash_and_schema_are_verified(tmp_path):
    row = manifest_row()
    while support.split_bucket(row["slice_ext_id"]) > 5:
        row["slice_ext_id"] += "x"
    manifest_path, receipt_path = write_manifest(tmp_path, [row])
    rows, _, digest = support.load_manifest(manifest_path, receipt_path)
    assert rows == [row]
    assert digest == support.sha256_file(manifest_path)


def test_manifest_rejects_unknown_outcome_like_field(tmp_path):
    row = manifest_row(dec_fit_reconv_amp=1.0)
    while support.split_bucket(row["slice_ext_id"]) > 5:
        row["slice_ext_id"] += "x"
    manifest_path, receipt_path = write_manifest(tmp_path, [row])
    with pytest.raises(support.SupportFailure, match="schema mismatch"):
        support.load_manifest(manifest_path, receipt_path)


def test_manifest_rejects_confirmation_slice(tmp_path):
    row = manifest_row()
    while support.split_bucket(row["slice_ext_id"]) <= 5:
        row["slice_ext_id"] += "x"
    manifest_path, receipt_path = write_manifest(tmp_path, [row])
    with pytest.raises(support.SupportFailure, match="non-train"):
        support.load_manifest(manifest_path, receipt_path)


def test_manifest_rejects_tamper(tmp_path):
    row = manifest_row()
    while support.split_bucket(row["slice_ext_id"]) > 5:
        row["slice_ext_id"] += "x"
    manifest_path, receipt_path = write_manifest(tmp_path, [row])
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(support.SupportFailure, match="SHA-256"):
        support.load_manifest(manifest_path, receipt_path)


def test_manifest_rejects_semantic_key_tamper(tmp_path):
    row = manifest_row(sequence_key="wrong")
    row["cap_hash"] = support.expected_cap_hash(row["sequence_key"])
    while support.split_bucket(row["slice_ext_id"]) > 5:
        row["slice_ext_id"] += "x"
    manifest_path, receipt_path = write_manifest(tmp_path, [row])
    with pytest.raises(support.SupportFailure, match="canonical"):
        support.load_manifest(manifest_path, receipt_path)


def test_manifest_rejects_cap_hash_tamper(tmp_path):
    row = manifest_row(cap_hash="0" * 64)
    while support.split_bucket(row["slice_ext_id"]) > 5:
        row["slice_ext_id"] += "x"
    manifest_path, receipt_path = write_manifest(tmp_path, [row])
    with pytest.raises(support.SupportFailure, match="cap hash"):
        support.load_manifest(manifest_path, receipt_path)
