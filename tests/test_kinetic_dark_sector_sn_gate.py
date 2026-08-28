from __future__ import annotations

import math

import pytest

from examples.physics.kinetic_dark_sector_gate import KineticClockConfig, solve_background
from examples.physics.kinetic_dark_sector_sn_gate import (
    SupernovaDataset,
    compare_pantheon_binned,
    compare_pantheon_binned_holdout,
    load_pantheon_binned,
    profiled_intercept_holdout_fit,
)


def test_hash_pinned_pantheon_binned_dimensions_and_covariance() -> None:
    dataset = load_pantheon_binned()

    assert len(dataset.redshift) == 40
    assert len(dataset.covariance) == 40
    assert all(len(row) == 40 for row in dataset.covariance)
    assert all(dataset.covariance[i][i] > 0.0 for i in range(40))


def test_pantheon_shape_comparison_profiles_only_the_intercept() -> None:
    solution = solve_background(KineticClockConfig(steps=600))
    result = compare_pantheon_binned(solution)

    assert result.kinetic.dof == 39
    assert result.lcdm.dof == 39
    assert math.isfinite(result.kinetic.chi2)
    assert math.isfinite(result.lcdm.chi2)
    assert result.kinetic.role.endswith("NOT_PANTHEON_PLUS")


def test_correlated_holdout_is_finite_and_disjoint() -> None:
    solution = solve_background(KineticClockConfig(steps=600))
    result = compare_pantheon_binned_holdout(solution)

    assert len(result.kinetic.training_indices) == 30
    assert len(result.kinetic.holdout_indices) == 10
    assert not set(result.kinetic.training_indices) & set(
        result.kinetic.holdout_indices
    )
    assert math.isfinite(result.kinetic.predictive_chi2)
    assert math.isfinite(result.lcdm.predictive_chi2)
    assert math.isfinite(result.delta_predictive_chi2_kinetic_minus_lcdm)
    assert result.kinetic.role.endswith("NOT_PREREGISTERED")


def test_diagonal_covariance_holdout_propagates_intercept_uncertainty() -> None:
    dataset = SupernovaDataset(
        redshift=(0.1, 0.2, 0.3, 0.4),
        apparent_magnitude=(1.0, 1.0, 1.0, 2.0),
        covariance=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        source="synthetic",
    )
    fit = profiled_intercept_holdout_fit(
        (0.0, 0.0, 0.0, 0.0),
        dataset,
        holdout_indices=(3,),
        label="synthetic",
    )

    assert fit.training_intercept == pytest.approx(1.0)
    # Predictive variance is 1 (held-out noise) + 1/3 (intercept posterior).
    assert fit.predictive_chi2 == pytest.approx(0.75)


def test_cross_covariance_enters_conditional_mean_and_variance() -> None:
    dataset = SupernovaDataset(
        redshift=(0.1, 0.2, 0.3),
        apparent_magnitude=(1.0, 1.0, 2.0),
        covariance=(
            (1.0, 0.2, 0.1),
            (0.2, 1.0, 0.1),
            (0.1, 0.1, 1.0),
        ),
        source="correlated synthetic",
    )
    fit = profiled_intercept_holdout_fit(
        (0.0, 0.0, 0.0),
        dataset,
        holdout_indices=(2,),
        label="correlated synthetic",
    )

    # A direct Schur-complement calculation gives predictive variance 7/5
    # and residual -1 after integrating the train-only intercept posterior.
    assert fit.training_intercept == pytest.approx(1.0)
    assert fit.predictive_log_determinant == pytest.approx(math.log(7.0 / 5.0))
    assert fit.predictive_chi2 == pytest.approx(5.0 / 7.0)
