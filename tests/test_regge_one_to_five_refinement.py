from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.regge_one_to_five_refinement import (
    FINE_SIMPLICES,
    INTERNAL_TRIANGLES,
    audit_regge_one_to_five_refinement,
    euclidean_regge_action,
    four_simplex_volume,
    simplex_dihedral_angle,
)


def test_barycentric_1_to_5_geometry_has_the_declared_volumes_and_counts() -> None:
    audit = audit_regge_one_to_five_refinement()

    assert audit.boundary_squared_length == 2.0
    assert audit.internal_squared_lengths == pytest.approx((4.0 / 5.0,) * 5)
    assert audit.coarse_four_volume == pytest.approx(math.sqrt(5.0) / 24.0)
    assert audit.refined_four_volumes == pytest.approx((math.sqrt(5.0) / 120.0,) * 5)
    assert sum(audit.refined_four_volumes) == pytest.approx(audit.coarse_four_volume)
    assert audit.internal_edge_count == 5
    assert audit.internal_triangle_count == 10


def test_actual_gram_geometry_has_flat_internal_deficits() -> None:
    lengths = np.full(5, math.sqrt(4.0 / 5.0))
    assert four_simplex_volume(FINE_SIMPLICES[0], lengths) == pytest.approx(
        math.sqrt(5.0) / 120.0
    )
    for triangle in INTERNAL_TRIANGLES:
        angle_sum = sum(
            simplex_dihedral_angle(simplex, triangle, lengths)
            for simplex in FINE_SIMPLICES
            if set(triangle).issubset(simplex)
        )
        assert angle_sum == pytest.approx(2.0 * math.pi, abs=2.0e-12)


def test_regge_action_with_boundary_term_is_stationary_at_flat_block() -> None:
    audit = audit_regge_one_to_five_refinement()

    assert audit.maximum_internal_deficit < 2.0e-12
    assert audit.maximum_internal_gradient < 2.0e-5
    assert audit.status == "EUCLIDEAN_REGGE_1_TO_5_FLAT_GAUGE_BLOCK_CLOSED"


def test_removing_the_barycentric_equal_length_condition_changes_the_action() -> None:
    flat = np.full(5, math.sqrt(4.0 / 5.0))
    distorted = flat.copy()
    distorted[0] *= 1.03

    assert euclidean_regge_action(distorted) != pytest.approx(euclidean_regge_action(flat))


def test_internal_hessian_has_four_vertex_displacement_null_modes_and_one_radial_mode() -> None:
    audit = audit_regge_one_to_five_refinement()

    assert audit.internal_hessian_rank == 1
    assert audit.internal_hessian_nullity == 4
    assert audit.raw_hessian_symmetry_residual < 2.0e-6
    assert audit.raw_hessian_gauge_residual < 2.0e-4
    assert max(abs(value) for value in audit.hessian_eigenvalues[:4]) < 1.0e-4
    assert audit.hessian_eigenvalues[-1] > 1.0
    assert not audit.ordinary_internal_inverse_exists
    assert audit.gauge_basis_orthogonality_residual < 1.0e-12


def test_only_the_collective_u_mode_has_a_projected_physical_inverse() -> None:
    audit = audit_regge_one_to_five_refinement()
    unit = np.ones(5) / math.sqrt(5.0)
    projector = np.outer(unit, unit)
    physical_inverse = projector / audit.projected_physical_curvature

    assert audit.projected_physical_schur_defined
    assert audit.projected_physical_curvature > 1.0
    # This is the gauge-reduced (one-dimensional) inverse, not inv(H_internal).
    assert np.linalg.norm(projector @ physical_inverse @ projector - physical_inverse) < 1.0e-14


def test_claim_ceiling_keeps_boundary_matching_and_spin_foam_claims_open() -> None:
    audit = audit_regge_one_to_five_refinement()

    assert not audit.boundary_hessian_equality_checked
    assert not audit.gauge_reduced_boundary_hessian_equals_coarse
    assert not audit.proper_eprl_amplitude_derived
    assert not audit.full_gaussian_path_integral_defined

