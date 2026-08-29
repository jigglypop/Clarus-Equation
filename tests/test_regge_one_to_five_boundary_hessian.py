from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.regge_one_to_five_boundary_hessian import (
    audit_regge_one_to_five_boundary_hessian,
    barycentric_internal_lengths,
    barycentric_section_jacobian,
    coarse_euclidean_regge_boundary_action,
    euclidean_regge_one_to_five_action,
    interior_point_internal_lengths,
)
from examples.physics.regge_one_to_five_refinement import (
    analytic_barycentric_internal_hessian,
)


def test_flat_interior_point_subdivisions_have_the_coarse_boundary_action() -> None:
    boundary = np.full(10, math.sqrt(2.0))
    perturbed = boundary * (
        1.0
        + np.asarray((0.006, -0.004, 0.003, -0.002, 0.005, -0.003, 0.002, 0.004, -0.005, 0.001))
    )
    weights = (
        np.full(5, 0.2),
        np.asarray((0.24, 0.18, 0.19, 0.17, 0.22)),
    )

    for sample_boundary in (boundary, perturbed):
        coarse = coarse_euclidean_regge_boundary_action(sample_boundary)
        for sample_weights in weights:
            internal = interior_point_internal_lengths(sample_boundary, sample_weights)
            assert euclidean_regge_one_to_five_action(
                sample_boundary, internal
            ) == pytest.approx(coarse, abs=1.0e-12)


def test_barycentric_length_formula_and_jacobian_match_coordinate_and_difference_checks() -> None:
    boundary = np.full(10, math.sqrt(2.0))
    formula = barycentric_internal_lengths(boundary)
    coordinate = interior_point_internal_lengths(boundary, np.full(5, 0.2))
    jacobian = barycentric_section_jacobian(boundary)
    step = 1.0e-6
    numerical = np.empty((5, 10))
    for edge in range(10):
        positive = boundary.copy()
        negative = boundary.copy()
        positive[edge] += step
        negative[edge] -= step
        numerical[:, edge] = (
            barycentric_internal_lengths(positive)
            - barycentric_internal_lengths(negative)
        ) / (2.0 * step)

    assert formula == pytest.approx((math.sqrt(4.0 / 5.0),) * 5)
    assert coordinate == pytest.approx(formula, abs=1.0e-12)
    assert np.linalg.norm(jacobian - numerical) < 1.0e-9


def test_analytic_internal_pseudoinverse_removes_only_the_collective_mode() -> None:
    internal = analytic_barycentric_internal_hessian()
    unit = np.ones(5) / math.sqrt(5.0)
    radial = np.outer(unit, unit)
    gauge = np.eye(5) - radial
    pseudoinverse = radial / (40.0 * math.sqrt(5.0))

    assert np.linalg.norm(internal @ gauge) < 1.0e-12
    assert np.linalg.norm(internal @ pseudoinverse @ internal - internal) < 1.0e-12
    assert np.linalg.norm(pseudoinverse @ internal @ pseudoinverse - pseudoinverse) < 1.0e-12


def test_actual_regge_blocks_converge_to_the_gauge_reduced_coarse_hessian() -> None:
    audit = audit_regge_one_to_five_boundary_hessian()

    assert audit.boundary_dimension == 10
    assert audit.internal_dimension == 5
    assert audit.full_dimension == 15
    assert audit.maximum_flat_section_action_residual < 1.0e-12
    assert audit.barycentric_formula_coordinate_residual < 1.0e-12
    assert audit.raw_internal_gauge_residual > audit.half_step_internal_gauge_residual
    assert audit.raw_mixing_gauge_residual > audit.half_step_mixing_gauge_residual
    assert (
        audit.raw_section_stationarity_derivative_residual
        > audit.half_step_section_stationarity_derivative_residual
    )
    assert audit.raw_on_shell_pullback_residual > audit.half_step_on_shell_pullback_residual
    assert (
        audit.raw_gauge_reduced_schur_coarse_residual
        > audit.half_step_gauge_reduced_schur_coarse_residual
    )
    assert audit.half_step_relative_schur_coarse_residual < 2.0e-6
    assert audit.finite_difference_residuals_decrease
    assert audit.classical_on_shell_boundary_hessian_identity_closed
    assert (
        audit.status
        == 'FLAT_REGGE_1_TO_5_GAUGE_REDUCED_BOUNDARY_HESSIAN_IDENTITY_CLOSED'
    )


def test_raw_finite_difference_internal_inverse_is_rejected_as_gauge_lifting() -> None:
    audit = audit_regge_one_to_five_boundary_hessian()

    assert audit.raw_finite_difference_internal_rank == 5
    assert audit.analytic_internal_rank == 1
    assert audit.analytic_internal_nullity == 4
    assert audit.analytic_internal_radial_curvature == pytest.approx(40.0 * math.sqrt(5.0))
    assert audit.gauge_reduced_internal_pseudoinverse_used
    assert not audit.raw_finite_difference_pseudoinverse_used


def test_claim_ceiling_excludes_gaussian_spinfoam_and_continuum_claims() -> None:
    audit = audit_regge_one_to_five_boundary_hessian()

    assert audit.reduced_geometric_action_normalization_used
    assert not audit.physical_gravitational_prefactor_included
    assert not audit.conditional_gaussian_integral_defined
    assert not audit.proper_eprl_multicell_hessian_computed
    assert not audit.spin_foam_measure_and_contour_derived
    assert not audit.curved_refinement_identity_derived
    assert not audit.continuum_einstein_hilbert_dominance_derived
