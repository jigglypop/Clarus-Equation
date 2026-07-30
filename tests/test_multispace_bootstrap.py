from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.core_axioms import low_bootstrap_fixed_point
from reality_stone.clarus.multispace_bootstrap import (
    branching_regime,
    fixed_point_stability_radius,
    homogeneous_reduction_depth,
    identity_branch_radius,
    is_irreducible,
    linear_stability_class,
    minimal_multispace_fixed_point,
    multispace_jacobian,
    multispace_bootstrap_map,
    multispace_residual,
    nearest_neighbor_coupling,
    normalized_transfer_coupling,
    strongly_connected_components,
    supercritical_components,
    symmetric_reduction_depth,
    types_reaching_supercritical,
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
    depth = homogeneous_reduction_depth(coupling)
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


def test_critical_poisson_component_returns_extinction_analytically() -> None:
    coupling = np.array([[1.0]])
    result = minimal_multispace_fixed_point(coupling)

    assert result.survival == (1.0,)
    assert result.iterations == 0
    assert result.residual == 0.0
    assert result.stability_radius == pytest.approx(1.0)
    assert linear_stability_class(result.stability_radius) == (
        "linearization_inconclusive"
    )


def test_near_critical_supercritical_branch_uses_accelerated_minimal_solve() -> None:
    depth = 1.0001
    result = minimal_multispace_fixed_point([[depth]], max_iterations=128)

    assert result.survival[0] < 1.0
    assert result.survival[0] == pytest.approx(
        low_bootstrap_fixed_point(depth),
        abs=1e-10,
    )
    assert result.residual < 1e-12
    assert result.iterations < 128


def test_reducible_critical_component_does_not_stall_solver() -> None:
    coupling = np.diag([2.0, 1.0])
    result = minimal_multispace_fixed_point(coupling)

    assert result.survival[0] == pytest.approx(
        low_bootstrap_fixed_point(2.0),
        abs=1e-12,
    )
    assert result.survival[1] == 1.0
    assert result.stability_radius == pytest.approx(1.0)


def test_upstream_type_inherits_survival_from_reachable_supercritical_scc() -> None:
    coupling = np.array([[0.0, 1.0], [0.0, 2.0]])
    result = minimal_multispace_fixed_point(coupling)
    downstream = low_bootstrap_fixed_point(2.0)
    upstream = math.exp(-(1.0 - downstream))

    assert strongly_connected_components(coupling) == ((0,), (1,))
    assert supercritical_components(coupling) == ((1,),)
    assert types_reaching_supercritical(coupling) == (0, 1)
    assert result.survival == pytest.approx((upstream, downstream), abs=1e-12)


def test_disconnected_subcritical_type_has_certain_extinction() -> None:
    coupling = np.diag([2.0, 0.5])
    result = minimal_multispace_fixed_point(coupling)

    assert types_reaching_supercritical(coupling) == (0,)
    assert result.survival[0] == pytest.approx(
        low_bootstrap_fixed_point(2.0),
        abs=1e-12,
    )
    assert result.survival[1] == 1.0


def test_multispace_jacobian_matches_finite_difference() -> None:
    coupling = np.array([[1.6, 0.9], [0.3, 1.2]])
    result = minimal_multispace_fixed_point(coupling)
    survival = result.as_array()
    step = 1e-6
    numerical = np.empty_like(coupling)

    for column in range(coupling.shape[1]):
        offset = np.zeros(coupling.shape[0])
        offset[column] = step
        numerical[:, column] = (
            multispace_bootstrap_map(survival + offset, coupling)
            - multispace_bootstrap_map(survival - offset, coupling)
        ) / (2.0 * step)

    assert np.allclose(
        multispace_jacobian(survival, coupling),
        numerical,
        rtol=1e-9,
        atol=1e-9,
    )


def test_nonsymmetric_normalized_transfer_keeps_additive_perron_depth() -> None:
    spatial_depth = 3.0
    cross_depth = 0.17776
    coupling = normalized_transfer_coupling(
        spatial_depth,
        cross_depth,
        [[0.2, 0.8], [0.6, 0.4]],
    )

    assert identity_branch_radius(coupling) == pytest.approx(
        spatial_depth + cross_depth
    )
