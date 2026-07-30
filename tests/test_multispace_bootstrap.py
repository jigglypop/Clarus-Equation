from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.core_axioms import low_bootstrap_fixed_point
from reality_stone.clarus.multispace_bootstrap import (
    branching_regime,
    fixed_point_stability_radius,
    identity_branch_radius,
    is_irreducible,
    minimal_multispace_fixed_point,
    multispace_bootstrap_map,
    multispace_residual,
    nearest_neighbor_coupling,
    normalized_transfer_coupling,
    symmetric_reduction_depth,
)


def test_scalar_self_recursion_is_the_one_type_special_case() -> None:
    depth = 3.17776
    result = minimal_multispace_fixed_point([[depth]])

    assert math.isclose(
        result.survival[0],
        low_bootstrap_fixed_point(depth),
        abs_tol=1e-12,
    )
    assert result.residual < 1e-12
    assert result.stability_radius < 1.0


def test_cross_space_recursion_exists_without_any_self_recursion() -> None:
    cross_depth = 1.8
    coupling = np.array([[0.0, cross_depth], [cross_depth, 0.0]])
    result = minimal_multispace_fixed_point(coupling)
    scalar = low_bootstrap_fixed_point(cross_depth)

    assert np.all(np.diag(coupling) == 0.0)
    assert np.allclose(result.survival, (scalar, scalar), atol=1e-12)
    assert branching_regime(coupling) == "supercritical"
    assert identity_branch_radius(coupling) > 1.0
    assert result.stability_radius < 1.0


def test_cross_recursion_threshold_is_spectral_not_entrywise() -> None:
    subcritical = np.array([[0.0, 0.8], [0.8, 0.0]])
    result = minimal_multispace_fixed_point(subcritical)

    assert branching_regime(subcritical) == "subcritical"
    assert np.allclose(result.survival, (1.0, 1.0), atol=1e-12)
    assert identity_branch_radius(subcritical) == pytest.approx(0.8)


def test_large_one_way_influence_is_not_a_closed_recursive_cycle() -> None:
    acyclic = np.array([[0.0, 5.0], [0.0, 0.0]])
    result = minimal_multispace_fixed_point(acyclic)

    assert identity_branch_radius(acyclic) == 0.0
    assert branching_regime(acyclic) == "subcritical"
    assert np.allclose(result.survival, (1.0, 1.0), atol=1e-12)
    assert not is_irreducible(acyclic)


def test_asymmetric_neighbor_coupling_produces_vector_not_scalar_survival() -> None:
    coupling = np.array([[1.6, 0.9], [0.3, 1.2]])
    result = minimal_multispace_fixed_point(coupling)
    survival = result.as_array()

    assert is_irreducible(coupling)
    assert branching_regime(coupling) == "supercritical"
    assert survival[0] < survival[1]
    assert np.max(np.abs(multispace_residual(survival, coupling))) < 1e-12
    assert fixed_point_stability_radius(survival, coupling) < 1.0
    with pytest.raises(ValueError, match="row sums differ"):
        symmetric_reduction_depth(coupling)


def test_equal_row_sums_are_exact_scalar_reduction_condition() -> None:
    coupling = np.array([[2.0, 0.5], [0.2, 2.3]])
    depth = symmetric_reduction_depth(coupling)
    scalar = low_bootstrap_fixed_point(depth)
    vector = np.array([scalar, scalar])

    assert depth == pytest.approx(2.5)
    assert np.allclose(multispace_bootstrap_map(vector, coupling), vector, atol=1e-12)


def test_identity_vector_is_always_a_fixed_point() -> None:
    coupling = np.array([[1.6, 0.9], [0.3, 1.2]])
    identity = np.ones(2)

    assert np.allclose(multispace_bootstrap_map(identity, coupling), identity)
    assert np.max(np.abs(multispace_residual(identity, coupling))) == 0.0


def test_periodic_neighbor_space_has_exact_homogeneous_sector() -> None:
    coupling = nearest_neighbor_coupling(
        7,
        self_depth=1.2,
        neighbor_depth=0.6,
        periodic=True,
    )
    depth = symmetric_reduction_depth(coupling)
    result = minimal_multispace_fixed_point(coupling)
    scalar = low_bootstrap_fixed_point(depth)

    assert depth == pytest.approx(2.4)
    assert np.allclose(result.survival, scalar, atol=1e-12)


def test_open_neighbor_space_retains_a_boundary_profile() -> None:
    coupling = nearest_neighbor_coupling(
        7,
        self_depth=1.2,
        neighbor_depth=0.6,
        periodic=False,
    )
    result = minimal_multispace_fixed_point(coupling)
    survival = result.as_array()

    with pytest.raises(ValueError, match="row sums differ"):
        symmetric_reduction_depth(coupling)
    assert survival[0] == pytest.approx(survival[-1])
    assert survival[1] == pytest.approx(survival[-2])
    assert survival[0] > survival[len(survival) // 2]


def test_additive_effective_depth_is_perron_mode_of_normalized_transfer() -> None:
    spatial_depth = 3.0
    cross_depth = 0.17776
    transfer = np.array([[0.0, 1.0], [1.0, 0.0]])
    coupling = normalized_transfer_coupling(
        spatial_depth,
        cross_depth,
        transfer,
    )
    effective_depth = spatial_depth + cross_depth
    result = minimal_multispace_fixed_point(coupling)
    scalar = low_bootstrap_fixed_point(effective_depth)

    assert symmetric_reduction_depth(coupling) == pytest.approx(effective_depth)
    assert identity_branch_radius(coupling) == pytest.approx(effective_depth)
    assert np.allclose(result.survival, scalar, atol=1e-12)
    assert np.linalg.eigvalsh(coupling)[0] == pytest.approx(
        spatial_depth - cross_depth
    )


def test_additive_coefficient_requires_transfer_normalization() -> None:
    with pytest.raises(ValueError, match="row-stochastic"):
        normalized_transfer_coupling(
            3.0,
            0.17776,
            [[0.0, 2.0], [2.0, 0.0]],
        )
