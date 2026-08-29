from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.gaussian_refinement_schur_kernel import (
    audit_gaussian_refinement_schur_kernel,
    certify_gaussian_schur_ward_inheritance,
    schur_complement_effective_hessian,
)


def test_scalar_schur_complement_recovers_declared_effective_hessian() -> None:
    target = 4.0
    mixing = 3.0
    internal = 2.0
    boundary = target + mixing**2 / internal

    effective = schur_complement_effective_hessian(
        ((boundary,),), ((mixing,),), ((internal,),)
    )

    assert effective == pytest.approx(np.asarray(((target,),)))


def test_scalar_internal_gaussian_matches_schur_exponent_and_normalization() -> None:
    boundary = 5.0
    mixing = 1.3
    internal = 2.4
    boundary_value = 0.7
    effective = boundary - mixing**2 / internal
    internal_values = np.linspace(-9.0, 9.0, 40001)
    integrand = np.exp(
        -0.5 * boundary * boundary_value**2
        - mixing * boundary_value * internal_values
        - 0.5 * internal * internal_values**2
    )
    numerical = float(np.trapezoid(integrand, internal_values))
    analytic = math.sqrt(2.0 * math.pi / internal) * math.exp(
        -0.5 * effective * boundary_value**2
    )

    assert numerical == pytest.approx(analytic, rel=1.0e-10)


def test_saddle_embedding_pulls_fine_hessian_back_to_schur_complement() -> None:
    boundary = np.asarray(((5.0, 1.0), (1.0, 4.0)))
    mixing = np.asarray(((1.0,), (2.0,)))
    internal = np.asarray(((3.0,),))
    effective = schur_complement_effective_hessian(boundary, mixing, internal)
    saddle_embedding = np.vstack((np.eye(2), -np.linalg.solve(internal, mixing.T)))
    fine = np.block([[boundary, mixing], [mixing.T, internal]])

    assert fine @ saddle_embedding == pytest.approx(
        np.vstack((effective, np.zeros((1, 2))))
    )
    assert saddle_embedding.T @ fine @ saddle_embedding == pytest.approx(effective)


def test_fine_gauge_null_vector_descends_through_schur_complement() -> None:
    effective = np.diag((2.0, 0.0))
    mixing = np.asarray(((1.0,), (2.0,)))
    internal = np.asarray(((4.0,),))
    boundary = effective + mixing @ np.linalg.solve(internal, mixing.T)
    coarse_gauge = np.asarray(((0.0,), (1.0,)))
    saddle_embedding = np.vstack((np.eye(2), -np.linalg.solve(internal, mixing.T)))
    fine_gauge = saddle_embedding @ coarse_gauge
    certificate = certify_gaussian_schur_ward_inheritance(
        boundary, mixing, internal, coarse_gauge, fine_gauge
    )

    assert certificate.exact_schur_pullback
    assert certificate.exact_fine_to_effective_ward_inheritance
    assert certificate.effective_ward_residual < 1.0e-12
    assert certificate.naive_boundary_block_ward_residual > 1.0e-6
    assert certificate.naive_boundary_block_is_not_the_effective_kernel


def test_wrong_fine_gauge_lift_is_detected() -> None:
    effective = np.diag((2.0, 0.0))
    mixing = np.asarray(((1.0,), (2.0,)))
    internal = np.asarray(((4.0,),))
    boundary = effective + mixing @ np.linalg.solve(internal, mixing.T)
    coarse_gauge = np.asarray(((0.0,), (1.0,)))
    wrong_fine_gauge = np.asarray(((0.0,), (1.0,), (0.0,)))
    certificate = certify_gaussian_schur_ward_inheritance(
        boundary, mixing, internal, coarse_gauge, wrong_fine_gauge
    )

    assert certificate.supplied_gauge_lift_residual > 1.0e-6
    assert certificate.fine_ward_residual > 1.0e-6
    assert not certificate.exact_fine_to_effective_ward_inheritance


def test_naive_boundary_block_can_accidentally_keep_ward_when_mixing_misses_gauge() -> None:
    effective = np.diag((2.0, 0.0))
    mixing = np.asarray(((1.0,), (0.0,)))
    internal = np.asarray(((4.0,),))
    boundary = effective + mixing @ np.linalg.solve(internal, mixing.T)
    coarse_gauge = np.asarray(((0.0,), (1.0,)))
    saddle_embedding = np.vstack((np.eye(2), -np.linalg.solve(internal, mixing.T)))
    fine_gauge = saddle_embedding @ coarse_gauge
    certificate = certify_gaussian_schur_ward_inheritance(
        boundary, mixing, internal, coarse_gauge, fine_gauge
    )

    assert certificate.exact_fine_to_effective_ward_inheritance
    assert certificate.naive_boundary_block_ward_residual < 1.0e-12
    assert not certificate.naive_boundary_block_is_not_the_effective_kernel


@pytest.mark.parametrize(
    "momentum",
    (
        (1.2, 0.3, -0.4, 0.8),
        (0.7, -0.2, 0.5, 0.1),
        (2.0, 0.4, 0.3, -0.9),
    ),
)
def test_spin2_gaussian_refinement_witness_recovers_fp_and_ward(
    momentum: tuple[float, float, float, float],
) -> None:
    audit = audit_gaussian_refinement_schur_kernel(momentum)

    assert audit.boundary_dimension == 10
    assert audit.internal_refinement_dimension == 3
    assert audit.internal_hessian_eigenvalues == (2.0, 3.0, 5.0)
    assert audit.constructed_target_fierz_pauli_hessian_recovery_residual < 1.0e-10
    assert audit.full_fine_hessian_minimum_eigenvalue < 0.0
    assert audit.full_fine_hessian_nullity >= 4
    assert audit.exact_internal_gaussian_elimination_closed
    assert audit.constructed_target_fierz_pauli_kernel_recovered
    assert audit.effective_ward_identity_preserved
    assert audit.omitting_schur_term_breaks_ward_identity
    assert audit.status == "CONSTRUCTED_GAUSSIAN_REFINEMENT_SCHUR_WARD_INTERFACE_CLOSED"


def test_audit_keeps_actual_spin_foam_and_eh_claims_false() -> None:
    audit = audit_gaussian_refinement_schur_kernel()

    assert not audit.actual_proper_vertex_multicell_hessian_blocks_computed
    assert not audit.spin_foam_measure_and_contour_matched_to_real_gaussian
    assert not audit.full_real_euclidean_partition_integral_defined
    assert audit.certificate.gaussian_normalization_is_internal_conditional_only
    assert not audit.microscopic_higher_derivative_terms_excluded
    assert not audit.nonlinear_einstein_hilbert_effective_action_derived
    assert audit.claim_ceiling.endswith("NOT_PROPER_VERTEX_BLOCK_CALCULATION")


@pytest.mark.parametrize(
    "boundary,mixing,internal,message",
    (
        (((1.0, 2.0), (0.0, 1.0)), ((1.0,), (0.0,)), ((1.0,),), "symmetric"),
        (((1.0,),), ((1.0,),), ((0.0,),), "positive definite"),
        (((1.0,),), ((1.0,),), ((-1.0,),), "positive definite"),
        (((1.0,),), ((1.0, 2.0),), ((1.0,),), "incompatible shape"),
    ),
)
def test_schur_complement_rejects_invalid_blocks(
    boundary: tuple[tuple[float, ...], ...],
    mixing: tuple[tuple[float, ...], ...],
    internal: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        schur_complement_effective_hessian(boundary, mixing, internal)


@pytest.mark.parametrize("tolerance", (0.0, -1.0, math.inf, math.nan))
def test_audit_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        audit_gaussian_refinement_schur_kernel(tolerance=tolerance)
