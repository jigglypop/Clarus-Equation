from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_response_qc_support.py")
SPEC = importlib.util.spec_from_file_location("audit_response_qc_support", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def event(pulse, *, ex=1, in_=0, stim=None, target=1.0):
    row = {
        "pulse_number": pulse,
        "ex_qc_pass": ex,
        "in_qc_pass": in_,
        "stim_qc_pass": stim,
        "onset_time": float(pulse) * 0.01,
    }
    for field in support.PARENT.TARGET_FIELDS:
        row[field] = target
    return row


def test_response_qc_uses_sign_matched_field_only():
    rows = [event(pulse, ex=1, in_=0, stim=None) for pulse in range(12)]
    assert support.type_matched_response_qc(rows, "ex")
    assert not support.type_matched_response_qc(rows, "in")


def test_stimulus_qc_is_not_part_of_response_qc():
    rows = [event(pulse, ex=1, stim=None) for pulse in range(12)]
    assert support.type_matched_response_qc(rows, "ex")


def test_response_qc_requires_all_twelve_rows():
    rows = [event(pulse, ex=1) for pulse in range(12)]
    rows[7]["ex_qc_pass"] = 0
    assert not support.type_matched_response_qc(rows, "ex")


def test_exact_zero_based_sequence():
    rows = [event(pulse) for pulse in range(12)]
    assert support.exact_zero_based_sequence(rows)
    assert not support.exact_zero_based_sequence(rows[1:])
    rows[-1]["pulse_number"] = 10
    assert not support.exact_zero_based_sequence(rows)


def test_exact_sequence_rejects_nonincreasing_onset():
    rows = [event(pulse) for pulse in range(12)]
    rows[8]["onset_time"] = rows[7]["onset_time"]
    assert not support.exact_zero_based_sequence(rows)


def test_parent_helper_is_hash_pinned():
    assert len(support.EXPECTED_PARENT_HELPER_SHA256) == 64
    assert (
        support.PARENT.SUPPORT_AUDITOR_VERSION
        == "BA-SRM2-TRAIN-SUPPORT-V2.1-QC-DIAGNOSTIC"
    )


def test_positive_mad_and_constant_rejection():
    assert support.positive_finite_mad([0.0, 1.0, 2.0])
    assert not support.positive_finite_mad([2.0, 2.0, 2.0])


def test_eligible_manifest_has_no_outcomes():
    rows = [
        {
            "version": support.VERSION,
            "split": "train",
            "synapse_type": "ex",
            "slice_id": 1,
            "slice_ext_id": "s",
            "sequence_key": "k",
        }
    ]
    rendered = support.eligible_manifest_bytes(rows)
    assert b"dec_fit" not in rendered
    assert b"qc_pass" not in rendered
    assert rendered.endswith(b"\n")
