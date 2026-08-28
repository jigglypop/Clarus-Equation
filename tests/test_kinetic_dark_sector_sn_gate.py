from __future__ import annotations

import math

from examples.physics.kinetic_dark_sector_gate import KineticClockConfig, solve_background
from examples.physics.kinetic_dark_sector_sn_gate import (
    compare_pantheon_binned,
    load_pantheon_binned,
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
