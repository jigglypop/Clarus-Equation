"""Independent integrals and adversarial cases, not physical confirmation."""

import importlib.util
import math

import numpy as np
import pytest
from scipy.integrate import quad

from test_support.paths import VERIFY_ROOT

PATH = VERIFY_ROOT / "dimension_joint_candidate.py"
spec = importlib.util.spec_from_file_location("dimension_joint_candidate", PATH)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


@pytest.mark.parametrize("t", [1e-6, 0.1, 1.0, 10.0, 1e6])
def test_continuous_dimension_matches_independent_integral(t):
    eta, b, d = .2, .7, 4.
    norm = 2*math.sqrt(b/math.pi)
    def integrand(u):
        return norm*math.exp(-b*u*u-u*math.log(t))
    z = quad(integrand, 0, math.inf, epsabs=1e-10)[0]
    first = quad(lambda u: u*integrand(u), 0, math.inf, epsabs=1e-10)[0]
    expected = (1-eta)+eta*z
    actual = model.dimension_kernel(t, eta=eta, b=b, d_ir=d)
    assert actual["log_heat_trace"] == pytest.approx(-d/2*math.log(t)+math.log(expected))
    assert actual["dimension"] == pytest.approx(d+2*eta*first/expected)


@pytest.mark.parametrize("t", [1e-100, 1e-3, 1., 1e20, 1e100])
def test_dimension_is_log_derivative_and_baseline_recovers(t):
    h = 1e-4
    plus = model.dimension_kernel(t*math.exp(h))["log_heat_trace"]
    minus = model.dimension_kernel(t*math.exp(-h))["log_heat_trace"]
    actual = model.dimension_kernel(t)
    assert actual["dimension"] == pytest.approx(-(plus-minus)/h, rel=1e-7)
    assert model.dimension_kernel(t, eta=0)["dimension"] == 4
    assert actual["dimension"] >= 4


def test_ultraviolet_growth_and_infrared_limit():
    assert model.dimension_kernel(1e-100)["dimension"] > model.dimension_kernel(1e-20)["dimension"] > 40
    assert model.dimension_kernel(1e100)["dimension"] < 4.001
    assert model.dimension_kernel(1)["log_heat_trace"] == pytest.approx(0, abs=1e-15)


def test_normalized_infinite_spectrum_is_not_infinite_dimension():
    # Exponential probability density on the UNBOUNDED spectrum [0,infinity).
    for t in (1., 1e-3, 1e-6):
        p = quad(lambda x: math.exp(-(1+t)*x), 0, math.inf)[0]
        numerator = quad(lambda x: t*x*math.exp(-(1+t)*x), 0, math.inf)[0]
        assert 2*numerator/p == pytest.approx(2*t/(1+t))


def evaluate(y, baseline, candidate, c, groups, required=("quantum", "macro")):
    return model.joint_residuals(y, baseline, candidate, c, groups, required_groups=required)


@pytest.mark.parametrize("missing", model.REQUIRED_OBSERVATION_GROUPS)
def test_selected_scope_rejects_each_missing_observation_domain(missing):
    labels = [g for g in model.REQUIRED_OBSERVATION_GROUPS if g != missing]
    with pytest.raises(ValueError, match="all and only declared groups"):
        model.scoped_joint_residuals([0]*3, [1]*3, [.5]*3, np.eye(3), labels)


def test_selected_scope_does_not_hide_unchanged_flavor_in_aggregate_gain():
    result = model.scoped_joint_residuals(
        [0]*4, [2]*4, [0, 2, 2, 0], np.eye(4), model.REQUIRED_OBSERVATION_GROUPS)
    assert result["overall"]["strictly_reduced"]
    assert not result["joint_arithmetic_reduction"]
    assert not result["scientific_success"]


def test_overall_improvement_can_hide_macro_failure():
    result = evaluate(np.zeros(101), [10]*100+[1], [0]*100+[20],
                      np.eye(101), ["quantum"]*100+["macro"])
    assert result["overall"]["strictly_reduced"]
    assert not result["groups"]["macro"]["strictly_reduced"]
    assert not result["joint_arithmetic_reduction"]


def test_full_covariance_and_invariance_under_units_and_order():
    y, ref, pred = np.array([1., 2., 3.]), np.array([2., 4., 5.]), np.array([1.2, 2.4, 3.4])
    c = np.array([[2., .3, .2], [.3, 1., -.1], [.2, -.1, 3.]])
    labels = ["quantum", "macro", "macro"]
    a = evaluate(y, ref, pred, c, labels)
    r = ref-y
    assert a["overall"]["baseline_rmse"] == pytest.approx(math.sqrt(r@np.linalg.solve(c, r)/3))
    scale, order = np.array([1e9, 1e-5, 7.]), np.array([2, 0, 1])
    c2 = (c*np.outer(scale, scale))[np.ix_(order, order)]
    z = evaluate((y*scale)[order], (ref*scale)[order], (pred*scale)[order], c2,
                 [labels[i] for i in order])
    assert z["overall"]["candidate_rmse"] == pytest.approx(a["overall"]["candidate_rmse"])
    for g in labels:
        assert z["groups"][g]["candidate_rmse"] == pytest.approx(a["groups"][g]["candidate_rmse"])
    assert a["joint_arithmetic_reduction"] and not a["scientific_success"]


@pytest.mark.parametrize("c", [[[1, 2], [2, 1]], [[1, 1], [1, 1]], [[1, .1], [.2, 1]], [[1, 0], [0, float('nan')]]])
def test_invalid_covariance_rejected(c):
    with pytest.raises(ValueError):
        evaluate([0, 0], [1, 1], [.5, .5], c, ["quantum", "macro"])


def test_missing_domain_and_zero_baseline_cannot_pass():
    with pytest.raises(ValueError):
        evaluate([0], [1], [.5], [[1]], ["quantum"])
    result = evaluate([0, 0], [0, 1], [0, .5], np.eye(2), ["quantum", "macro"])
    assert not result["joint_arithmetic_reduction"]


@pytest.mark.parametrize("kwargs", [{"t": 0}, {"t": float('nan')}, {"t": 1, "eta": 2}, {"t": 1, "b": 0}])
def test_invalid_kernel_parameters(kwargs):
    with pytest.raises(ValueError):
        model.dimension_kernel(**kwargs)


def test_relative_action_matches_unweighted_coleman_weinberg_limit():
    # Independent closed integral: cancelled first two moments remove scheme terms.
    angles = 2*math.pi*np.arange(3)
    x = 3+.5*np.cos(angles/3)
    xr = 3+.5*np.cos((math.pi+angles)/3)
    expected = (np.sum(x*x*np.log(x))-np.sum(xr*xr*np.log(xr)))/(64*math.pi**2)
    actual = model.relative_flat_action(1e-8, eta=0)
    assert actual["relative_action_density"] == pytest.approx(expected, rel=1e-6)


def test_unbounded_dimension_restores_cutoff_dependence():
    standard = [model.relative_flat_action(t, eta=0)["relative_action_density"]
                for t in (1e-6, 1e-12)]
    unbounded = [model.relative_flat_action(t, eta=.1)["relative_action_density"]
                 for t in (1e-6, 1e-12)]
    assert standard[1] == pytest.approx(standard[0], rel=1e-5)
    assert unbounded[1] > unbounded[0]*1e30


def test_equal_spectrum_has_zero_relative_action():
    assert model.relative_flat_action(1e-8, phase=.3, reference_phase=.3)["relative_action_density"] == 0
