"""위상 보정의 물리 선택 부재를 실제 기하와 독립 대조로 검산한다."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "verify/Q-0020/regge_boundary_embeddings.py"
SPEC = importlib.util.spec_from_file_location("ce_boundary_embedding_checks", SOURCE)
model = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(model)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def domain(report):
    return model.base.Domain(report["cases"][0]["boundary"])


def unpack(values):
    a = np.asarray(values)
    return a[..., 0]+1j*a[..., 1]


def test_artifact_matches_exact_source_and_dependencies(report):
    assert report["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, digest in report["dependencies"].items():
        assert digest == hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["length", "squared"])
def test_different_actual_cones_pass_but_change_the_natural_state(report, name):
    for case in report["cases"]:
        for sample in case["samples"]:
            row = sample["results"][name]
            kernel = unpack(row["kernel"])
            minimum = np.asarray(row["minimum_distance_squared"])
            assert minimum[0] == pytest.approx(0, abs=1e-13)
            assert np.all(minimum[1:] > .005)
            assert row["minimum_formula_error"] < 1e-12
            assert row["refined_128_256_error"] < 1e-6
            a, b = row["cones"]
            assert abs(a["coarse_action"]-b["coarse_action"]) > 10
            for cone in (a, b):
                assert cone["boundary_preserved"]
                assert cone["minimum_gram_eigenvalue"] > 0
                for key in ("amplitude_error", "norm_error", "distance_formula_error", "composition_error"):
                    assert cone[key] < 1e-10
                assert np.all(np.asarray(cone["distance_squared"]) >= minimum-1e-12)
                assert cone["natural_amplitude_error"][2] > .1
                assert cone["opposite_phase_error"][2] > .1
                assert cone["frozen_beta1_error_at_beta0"] >= minimum[1]/2-1e-12
            assert np.all(abs(kernel) <= 1+1e-13)


def test_initial_quadrature_failure_is_retained(report):
    rows = [s["results"][n] for c in report["cases"] for s in c["samples"]
            for n in ("length", "squared")]
    assert max(r["initial_64_128_error"] for r in rows) > 1e-6
    assert max(r["refined_128_256_error"] for r in rows) < 1e-6


@pytest.mark.parametrize("name", ["length", "squared"])
def test_live_natural_kernel_matches_mass_and_amplitude_pair(domain, name):
    actual = model.compare_fiber(domain, .7, (0, 5, -5), order=128)["results"][name]
    pair = model.boundary.pair_at(domain, .7, (0, 5, -5), order=128)[name]
    kernel = unpack(actual["kernel"])
    np.testing.assert_allclose(kernel, pair["amplitude"]/pair["mass"], atol=1e-13, rtol=0)
    assert kernel[2] == pytest.approx(kernel[1].conjugate(), abs=1e-13)
    assert actual["minimum_distance_squared"][2] == pytest.approx(
        actual["minimum_distance_squared"][1], abs=1e-13)


def test_fixed_local_derivative_detects_action_cancellation(domain, report):
    for case in report["cases"]:
        for sample in case["samples"]:
            derivative = sample["derivative"]
            assert derivative["commutator_magnitude"] > 3
            assert derivative["derivative_error"] < 1e-8
            assert derivative["squared_measure_commutator_error"] < 1e-8
    actual = model.derivative_check(domain, .7)
    assert actual["commutator_magnitude"] > 4
    assert actual["derivative_error"] < 1e-8
    zero = model.derivative_check(domain, .7, beta=0)
    assert zero["commutator_magnitude"] == 0
    assert zero["derivative_error"] == 0


def test_unit_norm_wrong_phase_does_not_saturate_target_amplitude(domain):
    data = model.fiber_data(domain, .7, order=128)
    phase = np.exp(5j*data["action"])
    coarse = model.boundary.cone_completion(domain, .7, 1)["action"]
    exact = model.phase_insertion(data["action"], coarse, 5)
    perturbation = np.exp(3j*data["e"])
    for weights in data["weights"].values():
        assert weights @ abs(exact*perturbation)**2 == pytest.approx(1, abs=1e-13)
        assert abs(weights @ (phase*exact*perturbation)) < .98
        assert abs(weights @ (phase*exact)) == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("f", [float("nan"), -.7, 10])
def test_invalid_fiber_is_rejected(domain, f):
    with pytest.raises(ValueError):
        model.fiber_data(domain, f)
