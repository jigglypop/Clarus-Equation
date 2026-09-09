"""Independent nonlinear boundary-value checks and tiny-bound arithmetic."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import lambertw

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_isolated_screening", HERE/"dimension_isolated_screening.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


@pytest.mark.parametrize("beta", [.1,.8])
def test_integral_equation_matches_independent_bvp_under_refinement(beta):
    errors = []
    for order in (96,192):
        result = model.radial_fixed_point(beta,order=order)
        reference = model.independent_massless_bvp(beta,result["radius"])
        errors.append(np.max(np.abs(result["u"]-reference)))
        assert np.all(result["u"] <= result["linear_u"]+1e-12)
        assert np.all(result["u"] >= np.exp(-beta)*result["linear_u"]-1e-12)
        assert result["discrete_solution_error_upper_from_residual"] < 1e-12
    assert errors[1] < .3*errors[0]
    assert errors[1] < 5e-6
    # A tiny discrete residual must NOT be misreported as continuum accuracy.
    assert errors[1] > 1000*result["discrete_solution_error_upper_from_residual"]


def test_positive_yukawa_kernel_respects_contraction_and_charge_bounds():
    beta = .7
    result = model.radial_fixed_point(beta,inverse_range=3.)
    assert result["discrete_operator_norm"] < beta
    assert np.all(result["u"] >= np.exp(-beta)*result["linear_u"]-1e-12)
    assert np.exp(-beta) <= result["relative_total_charge"] <= 1


def test_physical_tiny_nonlinearity_is_bounded_without_rounding_to_zero():
    s = model.cassini_band()["maximum_s"]
    result = model.isolated_source_bound(s=s)
    beta = result["beta_Newton_potential_upper"]
    assert 0 < beta < 2e-32
    assert result["fractional_source_density_suppression_upper"] == pytest.approx(beta, rel=1e-12,abs=0)
    assert result["near_minus_far_acceleration_upper_m_s2"] > 1e-45
    assert result["full_environment_or_calibrated_mass_bound"] is False
    zero = model.isolated_source_bound(s=0.)
    assert zero["near_minus_far_phase_upper_rad"] == 0.


def test_contraction_condition_is_not_claimed_for_arbitrarily_strong_sources():
    result = model.isolated_source_bound(s=1e30)
    assert result["beta_Newton_potential_upper"] > 1
    assert result["contraction_certified_under_assumptions"] is False
    with pytest.raises(ValueError):
        model.radial_fixed_point(1.1)


@pytest.mark.parametrize("beta_e,beta_s", [(0.,.1),(.2,.05),(.7,.2)])
def test_environment_response_bound_against_exact_single_site_lambert_solution(beta_e,beta_s):
    # Independent positive rank-one algebraic model: u=beta exp(-u).
    # This checks the norm inequality, not an actual spatial environment.
    u_e = float(lambertw(beta_e).real)
    u_total = float(lambertw(beta_e+beta_s).real)
    gamma_e,gamma_s = .3,.07
    result = model.environment_response_bound(beta_environment=beta_e,beta_source=beta_s,
                gradient_environment=gamma_e,gradient_source=gamma_s)
    actual = (gamma_e+gamma_s)*np.exp(-u_total)-gamma_e*np.exp(-u_e)-gamma_s
    assert abs(actual) <= result["one_position_response_minus_linear_source_upper_m_s2"]
    assert abs(u_total-u_e) <= result["source_induced_field_sup_upper"]


def test_environment_zero_reduces_to_isolated_and_strong_case_fails_closed():
    isolated = model.isolated_source_bound(s=.02)
    beta = isolated["beta_Newton_potential_upper"]
    gamma = model.G_UPPER*.02*isolated["conserved_source_mass_kg_input"]/.003**2
    result = model.environment_response_bound(beta_environment=0.,beta_source=beta,
                                              gradient_environment=0.,gradient_source=gamma)
    assert result["near_minus_far_response_minus_linear_upper_m_s2"] == pytest.approx(
        isolated["near_minus_far_acceleration_upper_m_s2"], rel=1e-12, abs=0)
    with pytest.raises(ValueError,match="contraction"):
        model.environment_response_bound(beta_environment=.8,beta_source=.3,
                                          gradient_environment=1.,gradient_source=1.)


def test_environment_feedback_is_retained_instead_of_only_rescaling_source():
    result = model.bounded_environment_example(model.cassini_band()["maximum_s"])
    bound = result["result"]
    assert bound["induced_environment_acceleration_upper_m_s2"] > 0
    assert bound["one_position_response_minus_linear_source_upper_m_s2"] > bound["direct_nonlinear_source_acceleration_upper_m_s2"]
    assert bound["full_calibrated_observational_bound"] is False
