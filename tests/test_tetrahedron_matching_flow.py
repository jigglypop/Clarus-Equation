"""형상 일치·주변 분산·보존 방출을 독립 텐서 및 유한 사슬과 대조한다."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "verify/Q-0020/tetrahedron_matching_flow.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("tetra_matching_checks", SOURCE)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_recoupling_matches_actual_four_spin_tensor(report):
    for row in report["spectra"]:
        assert row["tensor_error"] < 1e-10
        assert row["gauss_closure_error"] < 1e-10
        assert row["minimum_neighbor"] > 0


def test_matching_kernel_and_small_exact_spectrum(report):
    assert report["spectra"][0]["spectrum"] == pytest.approx([0, 8/9, 8/3, 32/9], abs=1e-10)
    for row in report["spectra"]:
        assert row["zero_count"] == 1
        assert row["gap"] > 0
        assert row["kernel_residual"] < 1e-10
        assert row["ground_overlap"] == pytest.approx(1, abs=1e-10)
    for row in report["gap_exploration"]:
        assert row["gap"] > 0
        assert row["eigen_residual"] < 1e-9


def test_oriented_volume_needs_dual_transpose(report):
    for row in report["spectra"]:
        assert row["dual_volume_residual"] < 1e-10
        assert row["wrong_orientation_ground"] > 1e-4


def test_exact_matching_does_not_select_local_classical_shape(report):
    for row in report["spectra"]:
        assert row["marginal_error"] < 1e-12
        assert row["mean_shape"] == pytest.approx(-1/3, abs=1e-12)
        assert row["variance_shape"] == pytest.approx(row["variance_formula"], abs=1e-12)
        assert row["variance_shape"] > 16/45
    for n in (8, 32, 128):
        first, _ = checks.operators(n)
        mean = np.trace(first) / len(first)
        variance = np.trace(first @ first) / len(first) - mean**2
        assert variance == pytest.approx(16/45 + 1/(15*(n/2)*(n/2+1)), abs=1e-12)


def test_autonomous_emission_keeps_system_bath_and_interaction_energy(report):
    for row in report["cooling"] + report["superposition"]:
        assert row["positive_energy_lower_bound"] > 0
        assert row["norm_error"] < 1e-10
        assert row["total_energy_error"] < 1e-10
        assert row["energy_law_error"] < 1e-10
        assert row["survival_error"] < 1e-10
        assert row["trace_distance_error"] < 1e-10
        assert abs(row["interaction_energy"]) < 1e-10
        assert abs(row["bath_hopping_energy"]) < 1e-10
    last = [r for r in report["cooling"] if r["tau"] == 20.]
    assert all(r["system_energy"] < 1e-4*r["initial_energy"] for r in last)
    assert all(r["bath_energy"] > .999*r["initial_energy"] for r in last)


def test_finite_cutoff_and_isolated_negative_controls(report):
    assert max(r["difference"] for r in report["cutoff"]) < 1e-8
    for row in report["isolated"]["rows"]:
        assert row["energy_error"] < 1e-10
        assert row["ground_probability_error"] < 1e-10
    assert report["isolated"]["initial_energy"] > .01
    rows = report["recurrence"]
    assert any(r["finite_survival"] > r["infinite_survival"] + .01 for r in rows)


def test_emission_keeps_reference_and_nonselected_input_information(report):
    for row in report["information"]:
        assert row["isometry_error"] < 1e-10
        assert row["reference_marginal_error"] < 1e-10
        assert row["local_excited_probability"] == pytest.approx(
            row["predicted_excited_probability"], abs=1e-10)
        assert row["deleting_bath_retained_norm"] < .3
        assert row["erasing_channel_label_defect"] > 1.


def test_reusing_one_shape_space_cannot_match_two_bell_partners(report):
    for row in report["shared_copy"]:
        d = row["dimension"]
        assert row["projector_overlap"] == pytest.approx(1/d, abs=1e-10)
        assert row["projector_sum_norm"] == pytest.approx(1+1/d, abs=1e-10)
        assert row["ground_energy"] >= row["lower_bound"] - 1e-10
        assert row["ground_energy"] > 1e-4
        assert row["operator_bound_residual"] > -1e-10
        assert row["zero_count"] == 0


def test_invalid_dimensions_and_zero_time():
    for n in (True, 0, -1, .5):
        with pytest.raises(ValueError):
            checks.operators(n)
    for n in (False, 0, 1, 2.5):
        with pytest.raises(ValueError):
            checks.chain_basis(n)
    assert checks.survival(0) == 1.
    with pytest.raises(ValueError):
        checks.survival(float("nan"))


def test_saved_artifact_sources():
    path = SOURCE.with_suffix(".json")
    if not path.exists():
        pytest.skip("최초 선별 뒤 산출물을 고정한다")
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, value in data["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == value
