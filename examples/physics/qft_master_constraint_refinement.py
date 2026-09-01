'''Exact master-constraint refinement toy for the QFT-next M2 gate.

At level N the normalized generalized branch vector u_N defines the zero
projector Pi_N and master constraint I-Pi_N. Zero extension does not carry
the finite kernel into the next finite kernel, and Pi_N converges strongly to
zero on the direct-limit Hilbert space. The rescaled forms N Pi_N are instead
cylindrically consistent on finite-support test vectors. This is an exact
distributional rigging witness, not a gravity regulator or an anomaly-free HDA.
'''

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from examples.physics.distributional_rigging_map import (
    generalized_branch_coefficients,
    zero_refinement_embedding,
)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def zero_projector(
    dimension: int, *, dimensionless_phase_increment: float
) -> np.ndarray:
    '''Return the normalized rank-one zero spectral projector at one level.'''

    dimension = _positive_integer('dimension', dimension)
    branch = generalized_branch_coefficients(
        dimension, dimensionless_phase_increment=dimensionless_phase_increment
    )
    unit_branch = branch / math.sqrt(dimension)
    return np.outer(unit_branch, unit_branch.conjugate())


def master_constraint(
    dimension: int, *, dimensionless_phase_increment: float
) -> np.ndarray:
    '''Return I-Pi_N, whose exact kernel is the finite branch line.'''

    projector = zero_projector(
        dimension, dimensionless_phase_increment=dimensionless_phase_increment
    )
    return np.eye(dimension, dtype=complex) - projector


@dataclass(frozen=True)
class MasterConstraintRefinementAudit:
    coarse_dimension: int
    refined_dimension: int
    coarse_kernel_dimension: int
    refined_kernel_dimension: int
    coarse_master_minimum_eigenvalue: float
    refined_master_minimum_eigenvalue: float
    embedded_kernel_residual: float
    coarse_projector_probe_pairing: float
    refined_projector_probe_pairing: float
    coarse_renormalized_pairing: float
    refined_renormalized_pairing: float
    renormalized_pairing_residual: float
    projector_strong_probe_norm: float
    required_scale_ratio: float
    exact_scale_ratio: float
    finite_kernels_nontrivial: bool
    finite_kernel_embedding_inconsistent: bool
    normalized_projector_limit_trivial_on_probe: bool
    renormalized_forms_cylindrically_consistent: bool
    normalization_linear_in_dimension: bool
    gravity_regulator_defined: bool
    original_hda_anomaly_checked: bool
    continuum_physical_hilbert_proved: bool
    status: str
    claim_ceiling: str = 'EXACT_RANK_ONE_DISTRIBUTIONAL_TOY_NOT_QFT_GR_CONTINUUM'


def audit_master_constraint_refinement(
    coarse_dimension: int,
    refined_dimension: int,
    *,
    dimensionless_phase_increment: float,
    tolerance: float = 1.0e-12,
) -> MasterConstraintRefinementAudit:
    '''Audit finite kernels, projector loss, and distributional recovery.'''

    coarse = _positive_integer('coarse_dimension', coarse_dimension)
    refined = _positive_integer('refined_dimension', refined_dimension)
    if coarse >= refined:
        raise ValueError('dimensions must satisfy coarse_dimension < refined_dimension')
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    coarse_projector = zero_projector(
        coarse, dimensionless_phase_increment=dimensionless_phase_increment
    )
    refined_projector = zero_projector(
        refined, dimensionless_phase_increment=dimensionless_phase_increment
    )
    coarse_master = np.eye(coarse, dtype=complex) - coarse_projector
    refined_master = np.eye(refined, dtype=complex) - refined_projector
    coarse_eigenvalues = np.linalg.eigvalsh(coarse_master)
    refined_eigenvalues = np.linalg.eigvalsh(refined_master)
    coarse_kernel = int(np.count_nonzero(np.abs(coarse_eigenvalues) <= tolerance))
    refined_kernel = int(np.count_nonzero(np.abs(refined_eigenvalues) <= tolerance))

    coarse_branch = generalized_branch_coefficients(
        coarse, dimensionless_phase_increment=dimensionless_phase_increment
    ) / math.sqrt(coarse)
    embedded_branch = zero_refinement_embedding(
        coarse_branch, refined_dimension=refined
    )
    kernel_embedding_residual = float(
        np.linalg.norm(refined_master @ embedded_branch)
    )

    probe = np.zeros(coarse, dtype=complex)
    probe[0] = 1.0
    embedded_probe = zero_refinement_embedding(probe, refined_dimension=refined)
    coarse_pairing = float(np.vdot(probe, coarse_projector @ probe).real)
    refined_pairing = float(
        np.vdot(embedded_probe, refined_projector @ embedded_probe).real
    )
    coarse_renormalized = coarse * coarse_pairing
    refined_renormalized = refined * refined_pairing
    renormalized_residual = abs(coarse_renormalized - refined_renormalized)
    refined_projector_norm = float(
        np.linalg.norm(refined_projector @ embedded_probe)
    )
    coarse_projector_norm = float(np.linalg.norm(coarse_projector @ probe))
    if refined_pairing <= tolerance:
        raise ArithmeticError('reference projector pairing must stay positive')
    required_ratio = coarse_pairing / refined_pairing
    exact_ratio = refined / coarse

    kernels_nontrivial = coarse_kernel == 1 and refined_kernel == 1
    embedding_inconsistent = kernel_embedding_residual > tolerance
    projector_trivial = refined_projector_norm < coarse_projector_norm
    rigging_consistent = renormalized_residual <= tolerance
    linear_normalization = math.isclose(
        required_ratio, exact_ratio, rel_tol=0.0, abs_tol=tolerance
    )
    closed = (
        kernels_nontrivial
        and embedding_inconsistent
        and projector_trivial
        and rigging_consistent
        and linear_normalization
    )

    return MasterConstraintRefinementAudit(
        coarse_dimension=coarse,
        refined_dimension=refined,
        coarse_kernel_dimension=coarse_kernel,
        refined_kernel_dimension=refined_kernel,
        coarse_master_minimum_eigenvalue=float(coarse_eigenvalues[0]),
        refined_master_minimum_eigenvalue=float(refined_eigenvalues[0]),
        embedded_kernel_residual=kernel_embedding_residual,
        coarse_projector_probe_pairing=coarse_pairing,
        refined_projector_probe_pairing=refined_pairing,
        coarse_renormalized_pairing=coarse_renormalized,
        refined_renormalized_pairing=refined_renormalized,
        renormalized_pairing_residual=renormalized_residual,
        projector_strong_probe_norm=refined_projector_norm,
        required_scale_ratio=required_ratio,
        exact_scale_ratio=exact_ratio,
        finite_kernels_nontrivial=kernels_nontrivial,
        finite_kernel_embedding_inconsistent=embedding_inconsistent,
        normalized_projector_limit_trivial_on_probe=projector_trivial,
        renormalized_forms_cylindrically_consistent=rigging_consistent,
        normalization_linear_in_dimension=linear_normalization,
        gravity_regulator_defined=False,
        original_hda_anomaly_checked=False,
        continuum_physical_hilbert_proved=False,
        status=(
            'EXACT_MASTER_CONSTRAINT_DISTRIBUTIONAL_TOY_CLOSED'
            if closed
            else 'MASTER_CONSTRAINT_DISTRIBUTIONAL_TOY_AUDIT_FAILED'
        ),
    )
