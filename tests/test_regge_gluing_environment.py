"""Regge 경계 부호, 같은 환경의 감쇠·잡음 및 대안 처방을 검산한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"
original = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("regge_environment_under_test", HERE/"regge_gluing_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = original


def test_exact_coarse_hessian_has_johnson_spectrum_and_negative_cycle():
    h = module.exact_coarse_hessian()
    expected = module.exact_coarse_spectrum()
    np.testing.assert_allclose(np.linalg.eigvalsh(h), expected, atol=1e-13)
    cycle = np.array([1., -1, 0, 0, 0, -1, 0, 1, 0, 0])/2
    np.testing.assert_allclose(h @ cycle, expected[0]*cycle, atol=1e-13)
    assert np.count_nonzero(expected < 0) == 9
    assert expected[0] < -4.7
    # 모든 변의 균등한 길이 변화로 이차 동차성을 독립 검산한다.
    theta = math.pi-math.acos(.25)
    np.testing.assert_allclose(h @ np.ones(10), math.sqrt(3)*theta/2, atol=1e-13)


@pytest.mark.parametrize("step", [2e-3, 1e-3])
def test_actual_regge_action_preserves_negative_glued_cycle_under_penalty(step):
    row = module.certificate(step)
    assert row["coarse_hessian_difference"] < 2e-6
    assert row["section_pullback_difference"] < 2e-6
    assert row["cycle_gluing_residual"] < 1e-13
    assert row["cycle_internal_section_residual"] < 1e-13
    for entry in row["penalty_curvatures"]:
        assert entry["curvature"] == pytest.approx(row["cycle_exact_curvature"], abs=2e-6)
    assert row["unconstrained_flow_mixing_residual"] > 1.
    assert row["fixed_boundary_radial_curvature"] == pytest.approx(40*math.sqrt(5), abs=2e-5)
    assert row["fixed_boundary_gauge_residual"] < 2e-6


@pytest.mark.parametrize("frequency", [0., .3, 2.])
def test_direct_environment_integration_matches_frequency_kernel(frequency):
    data = module.regge_geometry()
    c, n = data["c"], data["n"]
    omega, g = np.linspace(.8, 1.7, len(c)), np.linspace(.2, .8, len(c))
    full, effective, correction = module.euclidean_kernels(n, np.ones(len(n)), c, omega, g, frequency)
    size = len(n)
    direct = full[:size, :size]-full[:size, size:] @ np.linalg.solve(full[size:, size:], full[size:, :size])
    np.testing.assert_allclose(direct, effective, atol=2e-13)
    np.testing.assert_allclose(correction @ data["a"], 0, atol=1e-13)
    if frequency == 0:
        np.testing.assert_allclose(effective, n, atol=1e-13)


def test_quantum_noise_matches_free_thermal_oscillator_covariance():
    data = module.regge_geometry()
    c = data["c"]
    omega, g = np.linspace(.8, 1.7, len(c)), np.linspace(.2, .8, len(c))
    temperature, time = .4, .3
    factor = 1/np.tanh(omega/(2*temperature))
    covariance = np.diag(np.r_[factor/(2*omega), factor*omega/2])
    force_t = -c.T @ np.hstack((np.diag(g*np.cos(omega*time)), np.diag(g*np.sin(omega*time)/omega)))
    force_0 = -c.T @ np.hstack((np.diag(g), np.zeros((len(c), len(c)))))
    memory, noise, mass_shift = module.environment_kernels(c, omega, g, time, temperature)
    np.testing.assert_allclose(force_t @ covariance @ force_0.T, noise, atol=1e-13)
    for kernel in (memory, noise, mass_shift):
        np.testing.assert_allclose(kernel @ data["a"], 0, atol=1e-13)


def test_zero_temperature_noise_and_classical_high_temperature_limit():
    c = np.array([[1., -1.]])
    omega, g = np.array([1.3]), np.array([.7])
    memory, zero_noise, _ = module.environment_kernels(c, omega, g, temperature=0.)
    np.testing.assert_allclose(zero_noise, g[0]**2/(2*omega[0])*c.T @ c, atol=1e-14)
    _, hot_noise, _ = module.environment_kernels(c, omega, g, temperature=1e4)
    np.testing.assert_allclose(hot_noise/1e4, memory, rtol=2e-9, atol=1e-12)


def test_initial_slip_is_required_by_full_coupled_trajectory():
    result = module.langevin_slip_case()
    assert result["with_initial_slip_residual"] < 1e-9
    assert result["omitting_initial_slip_residual"] > 1e-3
    assert result["omitting_initial_slip_residual"] == pytest.approx(result["slip_norm"], rel=1e-7)


def test_boundary_precision_and_contour_are_distinct_extra_inputs():
    row = module.certificate()
    alternatives = row["boundary_state_alternatives"]
    assert alternatives[0]["smallest_curvature"] == pytest.approx(.5, abs=1e-12)
    assert alternatives[1]["smallest_curvature"] == pytest.approx(1., abs=1e-12)
    assert alternatives[0]["cycle_variance"] == pytest.approx(2., abs=1e-12)
    assert alternatives[1]["cycle_variance"] == pytest.approx(1., abs=1e-12)
    h = module.exact_coarse_hessian()
    cycle = module.regge_geometry()["cycle"]
    assert cycle @ (h+(row["boundary_precision_threshold"]-.01)*np.eye(10)) @ cycle < 0
    assert row["rotated_curvature_residual"] < 1e-12
    assert row["contour_phase_in_spectral_coordinates"] == [0., 1.]
    assert row["conjugate_contour_phase"] == [0., -1.]


@pytest.mark.parametrize("omega, temperature", [(0., 0.), (math.nan, 0.), (1., -1.), (1., math.inf)])
def test_invalid_environment_inputs_are_rejected(omega, temperature):
    with pytest.raises(ValueError):
        module.environment_kernels([[1., -1.]], [omega], [.5], temperature=temperature)


def test_artifact_hashes_and_scope_are_current():
    report = json.loads((HERE/"regge_gluing_environment.json").read_text(encoding="utf-8"))
    for name, digest in report["source_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    scope = report["scope"]
    assert scope["damping_and_noise_computed_from_same_supplied_action"]
    assert scope["fixed_boundary_glued_gauge_quotient_positive"]
    for name in ("gluing_environment_selects_system_kinetic_mass",
                 "boundary_integrated_real_quadratic_measure_stabilized",
                 "boundary_state_precision_uniquely_derived", "complex_contour_is_real_length_probability",
                 "physical_regge_time_dynamics_derived", "common_metric_selected",
                 "continuum_einstein_limit_derived"):
        assert scope[name] is False
