"""Exact Gaussian limits and composition, independent of Regge finite differences."""

import math

import numpy as np
import pytest

from examples.physics.gravity.gaussian_fold import log_density_at_zero, soft_fold, split_log_density, stationary_scale


def test_correlations_require_conditional_not_reset_measure():
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
    first, second = split_log_density(covariance, 1)
    assert first + second == pytest.approx(-math.log(2 * math.pi) - 0.5 * math.log(0.75))
    reset = 2 * log_density_at_zero(np.eye(1))
    assert first + second - reset == pytest.approx(-0.5 * math.log(0.75))


def test_soft_fold_composes_with_posterior_transport():
    covariance = np.array([[2.0, 0.6], [0.6, 1.0]])
    constraints = np.array([[1.0, 0.0], [0.3, 1.0]])
    direct, final = soft_fold(covariance, constraints, 0.7)
    first, middle = soft_fold(covariance, constraints[:1], 0.7)
    second, sequential = soft_fold(middle, constraints[1:], 0.7)
    assert direct == pytest.approx(first + second, abs=1e-14)
    np.testing.assert_allclose(final, sequential, atol=1e-14)


def test_soft_fold_scalar_analytic_limit_and_unconstrained_direction():
    log_weight, posterior = soft_fold(np.diag([4.0, 9.0]), np.array([[1.0, 0.0]]), 2.0)
    assert log_weight == pytest.approx(-0.5 * math.log(2))
    np.testing.assert_allclose(posterior, np.diag([2.0, 9.0]))
    assert log_weight <= 0


def test_no_constraints_leaves_measure_unchanged():
    covariance = np.diag([2.0, 3.0])
    log_weight, posterior = soft_fold(covariance, np.empty((0, 2)), 0.5)
    assert log_weight == 0
    np.testing.assert_array_equal(posterior, covariance)


@pytest.mark.parametrize("matrix", [np.diag([1.0, -1.0]), np.zeros((2, 2)), np.array([[1.0, 2.0], [0.0, 1.0]])])
def test_indefinite_singular_and_asymmetric_measures_rejected(matrix):
    with pytest.raises(ValueError):
        log_density_at_zero(matrix)


def test_stationary_scale_matches_scalar_quadratic_and_depends_on_resolution():
    action, eigenvalue, resolution = 2.0, 0.3, 1.2
    a = 8 * math.pi * eigenvalue / resolution**2
    expected = (-a + math.sqrt(a*a + 16*math.pi*a/action))/2
    assert stationary_scale(np.array([eigenvalue]), action, resolution) == pytest.approx(expected)
    small = stationary_scale(np.array([eigenvalue]), action, 1e-5)
    large = stationary_scale(np.array([eigenvalue]), action, 1e5)
    assert small == pytest.approx(4*math.pi/action, rel=1e-8)
    assert 0 < large < 1e-3


def test_fixed_absolute_resolution_removes_scale_dependence_of_fold_cost():
    covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
    constraints = np.eye(2)
    values = [soft_fold(covariance/t, constraints, 0.7/math.sqrt(t))[0] for t in (0.2, 1.0, 9.0)]
    assert values == pytest.approx([values[0]] * 3, abs=1e-14)


def test_tight_resolution_preserves_positive_variance_and_repeated_fold():
    weight, posterior = soft_fold(np.eye(1), np.eye(1), 1e-10)
    assert posterior[0, 0] == pytest.approx(1e-20, rel=1e-14, abs=0)
    second, repeated = soft_fold(posterior, np.eye(1), 1e-10)
    assert repeated[0, 0] == pytest.approx(5e-21, rel=1e-14, abs=0)
    assert weight + second == pytest.approx(-0.5 * math.log(1 + 2e20))


def test_stationary_scale_extreme_finite_inputs_have_finite_roots():
    # Scalar root: t(t+a)=4*pi*a/action; here t >> a.
    actual = stationary_scale(np.array([1.0]), 1e-320, 1.0)
    expected = math.exp(0.5 * (math.log(32 * math.pi**2) - math.log(1e-320)))
    assert actual == pytest.approx(expected, rel=1e-12)
    tiny = stationary_scale(np.array([1.0]), 1.0, 1e100)
    assert tiny == pytest.approx(math.sqrt(32) * math.pi * 1e-100, rel=1e-12, abs=0)
