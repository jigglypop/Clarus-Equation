from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("extract_train_dataset.py")
SPEC = importlib.util.spec_from_file_location("extract_train_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
extract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extract
SPEC.loader.exec_module(extract)


def test_exact_dimensionless_conversions():
    assert extract.dimensionless(1e-3, "time") == 1.0
    assert extract.dimensionless(1e-3, "voltage") == 1.0
    assert extract.dimensionless(1e-12, "current") == 1.0
    assert extract.dimensionless(1e6, "resistance") == 1.0
    assert extract.dimensionless(1e-12, "capacitance") == 1.0
    assert extract.dimensionless(1e-4, "length") == 1.0
    assert extract.dimensionless(1000.0, "frequency") == 1.0
    assert extract.dimensionless(36.85, "temperature_c") == 1.0
    assert extract.dimensionless(3, "count") == 3.0


def test_nonfinite_values_become_missing_not_clipped():
    assert np.isnan(extract.dimensionless(None, "time"))
    assert np.isnan(extract.dimensionless(float("inf"), "time"))
    assert extract.dimensionless(-1e-3, "voltage") == -1.0
    assert extract.dimensionless(0.0, "time") == 0.0


def test_feature_dimensions_and_causal_pulse_ranges_are_frozen():
    assert len(extract.PULSE_NUMERIC_SPECS) == 11
    assert len(extract.PARENT.HISTORY_PULSES) == 8
    assert len(extract.STATIC_NUMERIC_SPECS) == 10
    assert 8 * 11 + 10 == 98
    assert extract.PARENT.PRIMARY_TARGET_PULSES == tuple(range(8, 12))
    assert len(extract.TARGET_SPECS) * 4 == 16


def fake_event(pulse):
    row = {
        "sequence_key": "k",
        "pulse_number": pulse,
        "onset_time": pulse * 0.01,
        "ex_qc_pass": 1,
        "in_qc_pass": 0,
        "stim_qc_pass": None,
        "induction_frequency": 50.0,
        "recovery_delay": 0.25,
        "bath_temperature": 32.0,
        "baseline_potential": -0.06,
        "baseline_current": 0.0,
        "baseline_noise_stdev": 1e-4,
        "pair_soma_distance": 50e-6,
        "post_input_resistance": 100e6,
        "post_capacitance": 20e-12,
        "post_time_constant": 10e-3,
        "pre_target_layer": "2/3",
        "post_target_layer": "5",
        "pre_cell_class": "ex",
        "post_cell_class": "in",
    }
    raw_unit = {
        "count": 1.0,
        "time": 1e-3,
        "voltage": 1e-3,
        "current": 1e-12,
    }
    for field, kind in extract.PULSE_NUMERIC_SPECS:
        row[field] = raw_unit[kind]
    for field, kind in extract.TARGET_SPECS:
        row[field] = 9999.0 if pulse >= 8 else (1.0 if kind == "count" else 1e-3)
    row["baseline_dec_fit_reconv_amp"] = 1e-3
    row["dec_fit_nrmse"] = 1.0
    return row


def test_sentinel_future_values_never_enter_inputs_and_category_order_is_fixed():
    manifest = [
        {
            "sequence_key": "k",
            "synapse_type": "ex",
            "slice_id": 1,
            "slice_ext_id": "slice",
            "post_stim_name": "protocol",
        }
    ]
    arrays = extract.build_arrays(manifest, [fake_event(pulse) for pulse in range(12)])
    assert arrays["numeric"].shape == (1, 98)
    assert arrays["target"].shape == (1, 16)
    assert np.max(np.abs(arrays["target"])) > 1e6
    assert np.max(np.abs(arrays["numeric"])) < 1e6
    assert arrays["categorical"].tolist() == [
        ["protocol", "2/3", "5", "ex", "in"]
    ]


def test_sql_is_manifest_scoped_and_blob_free():
    extract.PARENT.assert_train_extraction_sql(extract.PARENT.TRAIN_EXTRACTION_SQL)
    sql = extract.PARENT.TRAIN_EXTRACTION_SQL.lower()
    assert "stim_pulse.data" not in sql
    assert "pulse_response.data" not in sql


def test_npz_roundtrip_and_tamper_detection(tmp_path):
    arrays = {
        "x": np.asarray([[1.0, np.nan]]),
        "label": np.asarray(["a"]),
    }
    path = tmp_path / "data.npz"
    np.savez_compressed(path, **arrays)
    extract.verify_npz(path, arrays)
    with pytest.raises(extract.ExtractionFailure):
        extract.verify_npz(path, {"x": arrays["x"]})


def test_frozen_receipt_and_manifest_reject_wrong_hash(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(extract.ExtractionFailure, match="SHA-256"):
        extract.load_support_receipt(bad)
