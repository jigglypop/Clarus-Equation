from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_train_manifest.py")
SPEC = importlib.util.spec_from_file_location("build_train_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def row(**updates):
    value = {
        "pair_id": 1,
        "post_recording_id": 2,
        "pre_recording_id": 3,
        "post_stim_name": None,
        "induction_frequency": 50.0,
        "recovery_delay": 0.25,
    }
    value.update(updates)
    return value


def candidate(group, key, digest, synapse_type="ex"):
    return {
        "slice_ext_id": group,
        "sequence_key": key,
        "cap_hash": digest,
        "synapse_type": synapse_type,
    }


def test_sequence_key_distinguishes_null_empty_and_float_bits():
    null_key = manifest.canonical_sequence_key(row(post_stim_name=None))
    empty_key = manifest.canonical_sequence_key(row(post_stim_name=""))
    next_float_key = manifest.canonical_sequence_key(
        row(induction_frequency=float.fromhex("0x1.9000000000001p+5"))
    )
    assert null_key != empty_key
    assert null_key != next_float_key
    assert "post_stim_name=NULL" in null_key
    assert "post_stim_name=UTF8HEX:" in empty_key


def test_sequence_key_is_stable():
    assert manifest.canonical_sequence_key(row()) == manifest.canonical_sequence_key(row())
    assert manifest.cap_hash("abc") == manifest.cap_hash("abc")
    assert len(manifest.cap_hash("abc")) == 64


def test_split_is_deterministic_and_bounded():
    first = manifest.split_bucket("slice-17")
    assert first == manifest.split_bucket("slice-17")
    assert 0 <= first <= 9


def test_round_robin_takes_one_per_slice_before_second():
    rows = [
        candidate("a", "a2", "2"),
        candidate("a", "a1", "1"),
        candidate("b", "b1", "1"),
        candidate("c", "c1", "1"),
    ]
    selected = manifest.select_round_robin(rows, cap=4)
    assert [item["sequence_key"] for item in selected] == ["a1", "b1", "c1", "a2"]
    assert [item["round_robin_index"] for item in selected] == [0, 0, 0, 1]


def test_round_robin_never_reads_target_fields():
    rows = [candidate("a", "a1", "1"), candidate("b", "b1", "1")]
    selected = manifest.select_round_robin(rows, cap=1)
    assert len(selected) == 1
    assert all("target" not in key for key in selected[0])


@pytest.mark.parametrize("token", manifest.FORBIDDEN_SQL_TOKENS)
def test_target_blind_sql_rejects_locked_tokens(token):
    with pytest.raises(manifest.ManifestFailure):
        manifest.assert_target_blind_sql(f"SELECT {token} FROM x")


def test_structural_sql_is_target_blind():
    manifest.assert_target_blind_sql(manifest.STRUCTURAL_SEQUENCE_SQL)


def test_structural_sql_locks_zero_based_twelve_pulses():
    normalized = " ".join(manifest.STRUCTURAL_SEQUENCE_SQL.split())
    assert "min_pulse = 0" in normalized
    assert "max_pulse = 11" in normalized
    assert "event_rows = 12" in normalized
    assert "distinct_pulses = 12" in normalized
