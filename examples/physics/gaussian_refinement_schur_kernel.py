"""Gaussian internal-mode elimination and coarse Ward inheritance.

Split a dimensionless fine quadratic action into boundary variables ``x`` and
internal refinement variables ``y``:

``S_f = 1/2 x.T A x + x.T B y + 1/2 y.T C y``.

For symmetric ``C`` whose smallest eigenvalue exceeds the declared numerical
conditioning threshold, the internal saddle is
``y_*(x)=-C^{-1}B.T x`` and exact Gaussian integration produces the effective
boundary Hessian

``H_eff = A - B C^{-1} B.T``.

Equivalently, with the saddle embedding ``J=[I; -C^{-1}B.T]``, one has
``J.T H_f J=H_eff`` and the internal block of ``H_f J`` vanishes.  If a coarse
gauge vector ``G_c`` lifts to ``G_f=J G_c`` and ``H_f G_f=0``, then
``H_eff G_c=0``.  Taking only the boundary block ``A`` generally breaks this
identity; the Schur term is essential.

The concrete audit deliberately completes a supplied Fierz--Pauli Hessian into
a finer Gaussian witness and recovers that preselected target exactly by
elimination.  This establishes the
multi-cell Hessian-to-effective-kernel calculation once the blocks are known.
It does not calculate those blocks from a proper/EPRL spin-foam amplitude.
Positive ``C`` makes only the internal ``y`` integral converge for fixed
boundary ``x``; the full Minkowski/gauge-degenerate integral still needs its
own contour, gauge fixing, and measure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from examples.physics.lattice_fierz_pauli_refinement import (
    linearized_gauge_direction_matrix,
)
from examples.physics.two_derivative_spin2_uniqueness import (
    general_two_derivative_spin2_symbol,
)


_ETA_DIAGONAL = np.asarray((-1.0, 1.0, 1.0, 1.0))
_COMPONENTS = tuple(
    (first, second) for first in range(4) for second in range(first, 4)
)


def _finite_matrix(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return matrix


def schur_complement_effective_hessian(
    boundary_hessian_block: Sequence[Sequence[float]],
    boundary_internal_mixing: Sequence[Sequence[float]],
    internal_hessian_block: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return ``A-B C^{-1}B.T`` for a convergent internal Gaussian."""

    boundary = _finite_matrix("boundary_hessian_block", boundary_hessian_block)
    mixing = _finite_matrix("boundary_internal_mixing", boundary_internal_mixing)
    internal = _finite_matrix("internal_hessian_block", internal_hessian_block)
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    boundary_dimension = boundary.shape[0]
    internal_dimension = internal.shape[0]
    if boundary.shape != (boundary_dimension, boundary_dimension):
        raise ValueError("boundary_hessian_block must be square")
    if internal.shape != (internal_dimension, internal_dimension):
        raise ValueError("internal_hessian_block must be square")
    if mixing.shape != (boundary_dimension, internal_dimension):
        raise ValueError("boundary_internal_mixing has incompatible shape")
    if np.linalg.norm(boundary - boundary.T) > tolerance:
        raise ValueError("boundary_hessian_block must be symmetric")
    if np.linalg.norm(internal - internal.T) > tolerance:
        raise ValueError("internal_hessian_block must be symmetric")
    if float(np.min(np.linalg.eigvalsh(internal))) <= tolerance:
        raise ValueError("internal_hessian_block must be positive definite")
    solved_mixing_transpose = np.linalg.solve(internal, mixing.T)
    return np.asarray(boundary - mixing @ solved_mixing_transpose, dtype=float)


@dataclass(frozen=True)
class GaussianSchurWardCertificate:
    boundary_dimension: int
    internal_dimension: int
    gauge_parameter_count: int
    internal_minimum_eigenvalue: float
    fine_self_adjoint_residual: float
    effective_self_adjoint_residual: float
    internal_saddle_block_residual: float
    saddle_pullback_residual: float
    supplied_gauge_lift_residual: float
    fine_ward_residual: float
    effective_ward_residual: float
    naive_boundary_block_ward_residual: float
    gaussian_log_normalization: float
    gaussian_normalization_is_internal_conditional_only: bool
    exact_schur_pullback: bool
    exact_fine_to_effective_ward_inheritance: bool
    naive_boundary_block_is_not_the_effective_kernel: bool


def certify_gaussian_schur_ward_inheritance(
    boundary_hessian_block: Sequence[Sequence[float]],
    boundary_internal_mixing: Sequence[Sequence[float]],
    internal_hessian_block: Sequence[Sequence[float]],
    coarse_gauge_generators: Sequence[Sequence[float]],
    supplied_fine_gauge_generators: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> GaussianSchurWardCertificate:
    """Certify Gaussian elimination, saddle pullback, and Ward inheritance."""

    boundary = _finite_matrix("boundary_hessian_block", boundary_hessian_block)
    mixing = _finite_matrix("boundary_internal_mixing", boundary_internal_mixing)
    internal = _finite_matrix("internal_hessian_block", internal_hessian_block)
    coarse_gauge = _finite_matrix(
        "coarse_gauge_generators", coarse_gauge_generators
    )
    supplied_fine_gauge = _finite_matrix(
        "supplied_fine_gauge_generators", supplied_fine_gauge_generators
    )
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    effective = schur_complement_effective_hessian(
        boundary, mixing, internal, tolerance=tolerance
    )
    boundary_dimension = boundary.shape[0]
    internal_dimension = internal.shape[0]
    if coarse_gauge.shape[0] != boundary_dimension:
        raise ValueError("coarse_gauge_generators have incompatible shape")
    if supplied_fine_gauge.shape != (
        boundary_dimension + internal_dimension,
        coarse_gauge.shape[1],
    ):
        raise ValueError("supplied_fine_gauge_generators have incompatible shape")

    solved_mixing_transpose = np.linalg.solve(internal, mixing.T)
    saddle_embedding = np.vstack(
        (np.eye(boundary_dimension), -solved_mixing_transpose)
    )
    fine_hessian = np.block(
        [[boundary, mixing], [mixing.T, internal]]
    )
    lifted_gauge = saddle_embedding @ coarse_gauge
    fine_times_embedding = fine_hessian @ saddle_embedding
    internal_saddle_residual = float(
        np.linalg.norm(fine_times_embedding[boundary_dimension:, :])
    )
    pullback_residual = float(
        np.linalg.norm(saddle_embedding.T @ fine_hessian @ saddle_embedding - effective)
    )
    gauge_lift_residual = float(np.linalg.norm(supplied_fine_gauge - lifted_gauge))
    fine_ward_residual = float(np.linalg.norm(fine_hessian @ supplied_fine_gauge))
    effective_ward_residual = float(np.linalg.norm(effective @ coarse_gauge))
    naive_ward_residual = float(np.linalg.norm(boundary @ coarse_gauge))
    sign, log_determinant = np.linalg.slogdet(internal)
    if sign <= 0.0:
        raise ValueError("internal_hessian_block must have positive determinant")
    gaussian_log_normalization = 0.5 * (
        internal_dimension * math.log(2.0 * math.pi) - log_determinant
    )
    schur_closed = (
        internal_saddle_residual <= tolerance
        and pullback_residual <= tolerance
        and np.linalg.norm(fine_hessian - fine_hessian.T) <= tolerance
        and np.linalg.norm(effective - effective.T) <= tolerance
    )
    ward_closed = (
        gauge_lift_residual <= tolerance
        and fine_ward_residual <= tolerance
        and effective_ward_residual <= tolerance
    )

    return GaussianSchurWardCertificate(
        boundary_dimension=boundary_dimension,
        internal_dimension=internal_dimension,
        gauge_parameter_count=coarse_gauge.shape[1],
        internal_minimum_eigenvalue=float(np.min(np.linalg.eigvalsh(internal))),
        fine_self_adjoint_residual=float(np.linalg.norm(fine_hessian - fine_hessian.T)),
        effective_self_adjoint_residual=float(np.linalg.norm(effective - effective.T)),
        internal_saddle_block_residual=internal_saddle_residual,
        saddle_pullback_residual=pullback_residual,
        supplied_gauge_lift_residual=gauge_lift_residual,
        fine_ward_residual=fine_ward_residual,
        effective_ward_residual=effective_ward_residual,
        naive_boundary_block_ward_residual=naive_ward_residual,
        gaussian_log_normalization=float(gaussian_log_normalization),
        gaussian_normalization_is_internal_conditional_only=True,
        exact_schur_pullback=schur_closed,
        exact_fine_to_effective_ward_inheritance=ward_closed,
        naive_boundary_block_is_not_the_effective_kernel=(
            naive_ward_residual > tolerance
        ),
    )


@dataclass(frozen=True)
class GaussianRefinementSchurKernelAudit:
    dimensionless_momentum_up: tuple[float, float, float, float]
    boundary_dimension: int
    internal_refinement_dimension: int
    internal_hessian_eigenvalues: tuple[float, ...]
    constructed_target_fierz_pauli_hessian_recovery_residual: float
    full_fine_hessian_minimum_eigenvalue: float
    full_fine_hessian_nullity: int
    certificate: GaussianSchurWardCertificate
    exact_internal_gaussian_elimination_closed: bool
    constructed_target_fierz_pauli_kernel_recovered: bool
    effective_ward_identity_preserved: bool
    omitting_schur_term_breaks_ward_identity: bool
    actual_proper_vertex_multicell_hessian_blocks_computed: bool
    spin_foam_measure_and_contour_matched_to_real_gaussian: bool
    full_real_euclidean_partition_integral_defined: bool
    microscopic_higher_derivative_terms_excluded: bool
    nonlinear_einstein_hilbert_effective_action_derived: bool
    status: str
    claim_ceiling: str = (
        "EXACT_GAUSSIAN_SCHUR_WARD_INTERFACE_NOT_PROPER_VERTEX_BLOCK_CALCULATION"
    )


def audit_gaussian_refinement_schur_kernel(
    dimensionless_momentum_up: Sequence[float] = (1.2, 0.3, -0.4, 0.8),
    *,
    tolerance: float = 1.0e-10,
) -> GaussianRefinementSchurKernelAudit:
    """Complete and eliminate a dimensionless three-mode refinement witness."""

    momentum = np.asarray(dimensionless_momentum_up, dtype=float)
    tolerance = float(tolerance)
    if momentum.shape != (4,) or not np.all(np.isfinite(momentum)):
        raise ValueError("dimensionless_momentum_up must contain four finite values")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    equation_kernel = general_two_derivative_spin2_symbol(
        momentum, (1.0, -1.0, 1.0, 1.0, -1.0)
    )
    component_weights = np.asarray(
        [
            (1.0 if mu == nu else 2.0)
            * _ETA_DIAGONAL[mu]
            * _ETA_DIAGONAL[nu]
            for mu, nu in _COMPONENTS
        ]
    )
    target_hessian = np.diag(component_weights) @ equation_kernel
    boundary_dimension = target_hessian.shape[0]
    internal = np.diag((2.0, 3.0, 5.0))
    mixing = np.asarray(
        [
            [
                ((row + 1) * (column + 2) % 7 - 3) / 11.0
                for column in range(internal.shape[0])
            ]
            for row in range(boundary_dimension)
        ],
        dtype=float,
    )
    solved_mixing_transpose = np.linalg.solve(internal, mixing.T)
    boundary = target_hessian + mixing @ solved_mixing_transpose
    coarse_gauge = linearized_gauge_direction_matrix(momentum)
    saddle_embedding = np.vstack(
        (np.eye(boundary_dimension), -solved_mixing_transpose)
    )
    fine_gauge = saddle_embedding @ coarse_gauge
    certificate = certify_gaussian_schur_ward_inheritance(
        boundary,
        mixing,
        internal,
        coarse_gauge,
        fine_gauge,
        tolerance=tolerance,
    )
    recovered = schur_complement_effective_hessian(
        boundary, mixing, internal, tolerance=tolerance
    )
    recovery_residual = float(np.linalg.norm(recovered - target_hessian))
    recovered_fp = recovery_residual <= tolerance
    fine_hessian = np.block([[boundary, mixing], [mixing.T, internal]])
    fine_eigenvalues = np.linalg.eigvalsh(fine_hessian)
    fine_nullity = int(np.count_nonzero(np.abs(fine_eigenvalues) <= tolerance))
    closed = (
        certificate.exact_schur_pullback
        and certificate.exact_fine_to_effective_ward_inheritance
        and certificate.naive_boundary_block_is_not_the_effective_kernel
        and recovered_fp
    )

    return GaussianRefinementSchurKernelAudit(
        dimensionless_momentum_up=tuple(float(value) for value in momentum),
        boundary_dimension=boundary_dimension,
        internal_refinement_dimension=internal.shape[0],
        internal_hessian_eigenvalues=tuple(
            float(value) for value in np.linalg.eigvalsh(internal)
        ),
        constructed_target_fierz_pauli_hessian_recovery_residual=recovery_residual,
        full_fine_hessian_minimum_eigenvalue=float(np.min(fine_eigenvalues)),
        full_fine_hessian_nullity=fine_nullity,
        certificate=certificate,
        exact_internal_gaussian_elimination_closed=certificate.exact_schur_pullback,
        constructed_target_fierz_pauli_kernel_recovered=recovered_fp,
        effective_ward_identity_preserved=(
            certificate.exact_fine_to_effective_ward_inheritance
        ),
        omitting_schur_term_breaks_ward_identity=(
            certificate.naive_boundary_block_is_not_the_effective_kernel
        ),
        actual_proper_vertex_multicell_hessian_blocks_computed=False,
        spin_foam_measure_and_contour_matched_to_real_gaussian=False,
        full_real_euclidean_partition_integral_defined=False,
        microscopic_higher_derivative_terms_excluded=False,
        nonlinear_einstein_hilbert_effective_action_derived=False,
        status=(
            "CONSTRUCTED_GAUSSIAN_REFINEMENT_SCHUR_WARD_INTERFACE_CLOSED"
            if closed
            else "GAUSSIAN_REFINEMENT_SCHUR_KERNEL_AUDIT_FAILED"
        ),
    )
