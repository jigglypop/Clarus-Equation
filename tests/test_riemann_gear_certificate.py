from __future__ import annotations

import math

import numpy as np

from examples.brain.riemann_gear_certificate import (
    minimum_frustration,
    pair_residual_exact,
    run_certificate,
)


def test_pair_residual_decays_on_principal_branch() -> None:
    initial = 1.2
    result = pair_residual_exact(initial, coupling_rate=2.5, time=1.7)
    assert 0.0 < result < initial
    assert math.isclose(
        math.tan(result / 2.0),
        math.tan(initial / 2.0) * math.exp(-2.5 * 1.7),
        rel_tol=1e-13,
    )


def test_cycle_frustration_is_zero_only_for_consistent_target() -> None:
    incidence = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]])
    stiffness = np.eye(3)
    consistent = np.array([0.2, 0.3, -0.5])
    inconsistent = np.array([0.2, 0.3, -0.4])
    consistent_value, _, _ = minimum_frustration(incidence, stiffness, consistent)
    inconsistent_value, _, _ = minimum_frustration(incidence, stiffness, inconsistent)
    assert consistent_value < 1e-28
    assert inconsistent_value > 1e-4


def test_combined_certificate() -> None:
    result = run_certificate()
    assert result.pair_exact_error < 1e-9
    assert result.frustration_projection_error < 1e-9
    assert result.iss_bound_margin >= 0.0
    assert result.spectral_bound_margin >= 0.0
