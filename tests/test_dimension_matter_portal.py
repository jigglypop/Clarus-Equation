"""Force/phase and shared-amplitude counterexamples, not an observational fit."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_matter_portal", HERE/"dimension_matter_portal.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


@pytest.mark.parametrize("r", [.01, .1, 1., 3., 10.])
def test_subordination_matches_positive_mass_quadrature(r):
    p = model.MatterPortal()
    x, w, _ = p.probe.positive_quadrature(n_u=24, n_x=64)
    mass = np.sqrt(1+x)
    result = p.profile(r)
    assert result["potential_shape"] == pytest.approx(np.sum(w*np.exp(-r*mass)), rel=1e-5, abs=1e-10)
    assert result["force_shape"] == pytest.approx(np.sum(w*(1+r*mass)*np.exp(-r*mass)), rel=1e-5, abs=1e-10)
    assert 0 < result["potential_shape"] <= math.exp(-r)
    assert 0 < result["force_shape"] <= (1+r)*math.exp(-r)


def test_force_is_potential_derivative_and_phase_has_same_amplitude():
    p = model.MatterPortal()
    r, h = .7, 1e-4
    plus = p.profile(r+h)["potential_shape"]/(r+h)
    minus = p.profile(r-h)["potential_shape"]/(r-h)
    assert -(plus-minus)/(2*h)*r*r == pytest.approx(p.profile(r)["force_shape"], rel=1e-6)
    phase = p.held_arm_phase(.01, .02, .01, 2e-25, .02, 100., .1)
    assert phase > 0
    assert p.held_arm_phase(.01, .02, .01, 2e-25, .02, 100., .2) == pytest.approx(2*phase)
    assert p.held_arm_phase(.02, .01, .01, 2e-25, .02, 100., .1) == pytest.approx(-phase)
    assert p.point_source(.01, .01, 100., 0.)["inward_acceleration_m_s2"] == 0


def test_shared_interval_matches_direct_correlated_sse():
    r = np.array([2., 1., 3.])
    k = np.array([1., .5, 1.])
    c = np.array([[2., .2, .1], [.2, 1., .1], [.1, .1, 1.]])
    labels = ["quantum", "quantum", "macro"]
    result = model.common_amplitude_screen(r, k, c, labels, required_groups=["quantum", "macro"])
    assert result["joint_strict_improvement_possible"]
    for amplitude in (.2, 1., 2., 8.):
        before = r@np.linalg.solve(c, r)
        after = (r-amplitude*k)@np.linalg.solve(c, r-amplitude*k)
        coeff = result["overall"]
        assert after-before == pytest.approx(amplitude**2*coeff["quadratic_A"]-2*amplitude*coeff["alignment_B"])
    upper = result["joint_open_interval"][1]
    for data in [result["overall"], *result["groups"].values()]:
        assert (.5*upper)**2*data["quadratic_A"]-upper*data["alignment_B"] < 0


def test_total_gain_cannot_override_wrong_sign_quantum_residual():
    result = model.common_amplitude_screen([-.7, 10.], [1., 1.], np.eye(2),
                                          ["quantum", "macro"], required_groups=["quantum", "macro"])
    assert result["overall"]["strict_improvement_possible"]
    assert not result["groups"]["quantum"]["strict_improvement_possible"]
    assert not result["joint_strict_improvement_possible"]


def test_positive_group_alignments_do_not_replace_full_covariance():
    result = model.common_amplitude_screen([1., 10.], [10., 1.], [[1., .9], [.9, 1.]],
                                          ["quantum", "macro"], required_groups=["quantum", "macro"])
    assert all(g["strict_improvement_possible"] for g in result["groups"].values())
    assert result["overall"]["alignment_B"] < 0
    assert not result["joint_strict_improvement_possible"]


def test_strict_interval_boundary_is_not_reported_as_attained_optimum():
    result = model.common_amplitude_screen([1., 10.], [1., 1.], np.eye(2),
                                          ["quantum", "macro"], required_groups=["quantum", "macro"])
    assert result["joint_open_interval"] == [0., 2.]
    assert result["constrained_infimum_amplitude"] == 2.
    assert not result["constrained_infimum_attained"]


def test_corrected_published_sign_and_raw_value_control():
    result = model.report()
    corrected, raw = result["published_marginal_cases"]
    assert not corrected["screen"]["joint_strict_improvement_possible"]
    assert raw["screen"]["joint_strict_improvement_possible"]
    assert not result["all_domain_rmse_reduced"]


def test_zero_shape_missing_group_and_bad_covariance():
    result = model.common_amplitude_screen([1.], [0.], [[1.]], ["q"], required_groups=["q"])
    assert not result["joint_strict_improvement_possible"]
    with pytest.raises(ValueError):
        model.common_amplitude_screen([1.], [1.], [[1.]], ["q"], required_groups=["q", "m"])
    with pytest.raises(ValueError):
        model.common_amplitude_screen([1., 1.], [1., 1.], [[1., 2.], [2., 1.]], ["q", "m"], required_groups=["q", "m"])
