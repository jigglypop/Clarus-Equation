from __future__ import annotations

import math

import numpy as np
import pytest

from examples.pre_eq.fraction_threshold import (
    gamma_mean_field_ratio,
    mean_field_ratio,
    minimal_modes_for_budget,
    mode_action_moments,
    normal_cdf,
    normal_pdf,
    path_energy_gap,
    threshold_fraction,
)
from reality_stone.clarus.pre_eq import (
    conditioned_prior,
    gibbs_reweight,
    layer_cake_survival,
    manifest_indices,
    mean_field_bounds,
    survival_fraction,
    tilt_survival,
)


def test_layer_cake_identity_is_exact() -> None:
    prior = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    phi = np.array([0.0, 0.7, 0.7, 2.4, np.inf])

    assert math.isclose(
        layer_cake_survival(prior, phi),
        tilt_survival(prior, phi),
        rel_tol=1e-12,
    )


def test_mean_field_bounds_bracket_tilt() -> None:
    prior = np.array([0.4, 0.35, 0.25])
    phi = np.array([0.2, 1.1, 2.7])

    lower, upper = mean_field_bounds(prior, phi)
    tilt = tilt_survival(prior, phi)

    assert lower < tilt < upper


def test_mean_field_is_exact_only_for_constant_phi() -> None:
    prior = np.array([0.5, 0.3, 0.2])
    phi = np.array([1.3, 1.3, 1.3])

    lower, upper = mean_field_bounds(prior, phi)

    assert math.isclose(lower, tilt_survival(prior, phi), rel_tol=1e-12)
    assert math.isclose(upper, lower, rel_tol=1e-12)


def test_zero_mass_hard_constraint_raises() -> None:
    prior = np.array([0.5, 0.5])
    energy = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        conditioned_prior(prior, energy, threshold=0.5)


def test_survival_fraction_is_monotone_in_threshold() -> None:
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    energy = np.array([0.0, 1.0, 2.0, 3.0])

    fractions = [survival_fraction(prior, energy, t) for t in (0.5, 1.5, 2.5, 3.5)]

    assert fractions == [0.25, 0.5, 0.75, 1.0]


def test_conditioning_above_minimum_preserves_manifest() -> None:
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    energy = np.array([0.0, 1.0, 2.0, 3.0])

    conditioned = conditioned_prior(prior, energy, threshold=2.5)

    assert manifest_indices(prior, energy).tolist() == [0]
    assert manifest_indices(conditioned, energy).tolist() == [0]

    posterior = gibbs_reweight(conditioned, energy, beta=50.0)
    assert int(np.argmax(posterior)) == 0
    assert posterior[0] > 1.0 - 1e-12


def test_threshold_scaling_matches_normal_quantile() -> None:
    z = -1.66
    target = normal_cdf(z)

    coarse = threshold_fraction(shape=32, z=z, samples=200_000, seed=0)
    fine = threshold_fraction(shape=512, z=z, samples=200_000, seed=0)

    assert abs(coarse - target) < 0.03
    assert abs(fine - target) < 0.006
    assert abs(fine - target) < abs(coarse - target)


def test_path_energy_gap_scales_like_inverse_sqrt_shape() -> None:
    z = -1.0
    gap_small = path_energy_gap(shape=128, z=z, samples=400_000, seed=1)
    gap_large = path_energy_gap(shape=512, z=z, samples=400_000, seed=1)

    for shape, gap in ((128, gap_small), (512, gap_large)):
        assert 0.20 < gap * math.sqrt(shape) < 0.29
    assert 1.7 < gap_small / gap_large < 2.4
    assert abs(gap_small * math.sqrt(128) - normal_pdf(z)) < 0.03


def test_mode_action_moments_sit_on_uncertainty_floor() -> None:
    mean, var = mode_action_moments(samples=200_000, seed=2)

    assert abs(mean - 0.5) < 0.01
    assert abs(var - 0.5) < 0.02


def test_mean_field_ratio_converges_at_rate_one_over_n() -> None:
    ratios = {
        n: mean_field_ratio(n, target_mean=3.0, samples=400_000, seed=3)
        for n in (16, 64, 256)
    }

    assert all(ratio > 1.0 for ratio in ratios.values())
    assert ratios[16] > ratios[64] > ratios[256]
    for n in (64, 256):
        assert 7.0 < (ratios[n] - 1.0) * n < 11.0


PHI_MEAN = (1.0 - 0.04865) * 3.17776
OMEGA_B_REL_BUDGET = 0.0010 / 0.0486


def test_closed_form_matches_monte_carlo_ratio() -> None:
    for n in (16, 64):
        mc = mean_field_ratio(n, target_mean=3.0, samples=400_000, seed=3)
        exact = gamma_mean_field_ratio(n, target_mean=3.0)

        assert abs(mc - exact) / exact < 0.02


def test_dimension_as_mode_reading_is_excluded() -> None:
    ratio = gamma_mean_field_ratio(3.17776, PHI_MEAN)
    implied_omega_b = ratio * 0.04865

    assert 3.7 < ratio < 3.9
    assert abs(implied_omega_b - 0.0486) / 0.0010 > 100.0


def test_observation_budget_forces_neff_in_hundreds() -> None:
    n_min = minimal_modes_for_budget(PHI_MEAN, OMEGA_B_REL_BUDGET)

    assert 430 <= n_min <= 460
    assert gamma_mean_field_ratio(n_min, PHI_MEAN) <= 1.0 + OMEGA_B_REL_BUDGET
    assert gamma_mean_field_ratio(n_min - 1, PHI_MEAN) > 1.0 + OMEGA_B_REL_BUDGET


def test_correlation_cap_reciprocal_matches_neff_bound() -> None:
    rho_cap = 2.0 * math.log1p(OMEGA_B_REL_BUDGET) / (2.0 * PHI_MEAN**2)
    n_min = minimal_modes_for_budget(PHI_MEAN, OMEGA_B_REL_BUDGET)

    assert 0.0020 < rho_cap < 0.0025
    assert abs(1.0 / rho_cap - n_min) / n_min < 0.02
