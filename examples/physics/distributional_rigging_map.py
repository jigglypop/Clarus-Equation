"""Exact distributional rigging-map witness on a refinement direct system.

The Hilbert spaces are ``H_N = C^N`` and refinement embeds a vector by
appending zeros.  For a declared dimensionless branch phase ``s`` define the
generalized vector

``Omega_s = (exp(i n s))_{n >= 1}``.

Its finite truncations have norm squared ``N`` and therefore do not converge
in the direct-limit Hilbert norm.  On the dense space of finite-support
vectors, however, their pairings are eventually constant.  The positive
sesquilinear form

``P_s(phi)[psi] = conjugate(L_s(phi)) L_s(psi)``

with ``L_s(psi)=sum_n exp(-i n s) psi_n`` is cylindrically consistent.  Its
null-space quotient is one dimensional and completes to ``C``.  This is an
exact soluble witness for the distributional rigging-map route; it is not a
geometric refinement, an EPRL amplitude, or a proof of Einstein dynamics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import cmath
import math

import numpy as np


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_phase(value: float) -> float:
    phase = float(value)
    if not math.isfinite(phase):
        raise ValueError("dimensionless_phase_increment must be finite")
    return phase


def _finite_vector(name: str, values: Sequence[complex]) -> np.ndarray:
    vector = np.asarray(values, dtype=complex)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite nonempty one-dimensional vector")
    return vector


def generalized_branch_coefficients(
    dimension: int, *, dimensionless_phase_increment: float
) -> np.ndarray:
    """Return the first ``dimension`` components of ``Omega_s``."""

    dimension = _positive_integer("dimension", dimension)
    phase = _finite_phase(dimensionless_phase_increment)
    labels = np.arange(1, dimension + 1, dtype=float)
    return np.exp(1j * labels * phase)


def zero_refinement_embedding(
    vector: Sequence[complex], *, refined_dimension: int
) -> np.ndarray:
    """Embed ``C^N`` isometrically in ``C^M`` by appending zeros."""

    source = _finite_vector("vector", vector)
    refined_dimension = _positive_integer("refined_dimension", refined_dimension)
    if refined_dimension < source.size:
        raise ValueError("refined_dimension cannot be smaller than the source")
    result = np.zeros(refined_dimension, dtype=complex)
    result[: source.size] = source
    return result


def branch_functional(
    vector: Sequence[complex], *, dimensionless_phase_increment: float
) -> complex:
    """Evaluate ``L_s`` on one finite-support representative."""

    state = _finite_vector("vector", vector)
    coefficients = generalized_branch_coefficients(
        state.size, dimensionless_phase_increment=dimensionless_phase_increment
    )
    return complex(np.vdot(coefficients, state))


def rigging_pairing(
    first: Sequence[complex],
    second: Sequence[complex],
    *,
    dimensionless_phase_increment: float,
) -> complex:
    """Return the positive antilinear-first-slot pairing ``P_s``."""

    first_value = branch_functional(
        first, dimensionless_phase_increment=dimensionless_phase_increment
    )
    second_value = branch_functional(
        second, dimensionless_phase_increment=dimensionless_phase_increment
    )
    return first_value.conjugate() * second_value


@dataclass(frozen=True)
class DistributionalRiggingMapAudit:
    coarse_dimension: int
    intermediate_dimension: int
    refined_dimension: int
    dimensionless_phase_increment: float
    coarse_truncation_norm_squared: float
    refined_truncation_norm_squared: float
    embedded_truncation_difference_norm_squared: float
    cylindrical_pairing_residual: float
    embedding_composition_residual: float
    embedding_isometry_residual: float
    rigging_gram_hermiticity_residual: float
    rigging_gram_minimum_eigenvalue: float
    rigging_gram_rank: int
    quotient_dimension: int
    unit_norm_coarse_probe_pairing: float
    unit_norm_refined_probe_pairing: float
    phase_increment_declared_dimensionless: bool
    direct_system_identity_composition_isometric: bool
    cylindrical_consistency_exact: bool
    finite_truncations_fail_hilbert_norm_cauchy_criterion: bool
    finite_support_distributional_limit_eventually_constant: bool
    rigging_pairing_hermitian_positive_semidefinite: bool
    physical_completion_isomorphic_to_complex: bool
    unit_norm_truncation_pairings_break_cylindrical_consistency: bool
    topological_limit_excluded: bool
    geometric_refinement_defined: bool
    eprl_amplitude_used: bool
    renormalized_cutoff_removal_proved: bool
    einstein_hilbert_dominance_proved: bool
    anomaly_free_constraint_algebra_proved: bool
    exactly_two_graviton_helicities_proved: bool
    status: str
    claim_ceiling: str = (
        "EXACT_DISTRIBUTIONAL_RIGGING_MAP_MODEL_NOT_4D_GR_CONTINUUM"
    )


def audit_distributional_rigging_map(
    coarse_dimension: int,
    intermediate_dimension: int,
    refined_dimension: int,
    *,
    dimensionless_phase_increment: float,
    tolerance: float = 1.0e-12,
) -> DistributionalRiggingMapAudit:
    """Audit the exact direct-system and rank-one rigging-map identities.

    The dimensions must obey ``N < K < M``.  Numerical residuals regress the
    closed-form identities; the general proof follows from zero extension and
    the eventual constancy of every finite-support pairing.
    """

    coarse = _positive_integer("coarse_dimension", coarse_dimension)
    intermediate = _positive_integer(
        "intermediate_dimension", intermediate_dimension
    )
    refined = _positive_integer("refined_dimension", refined_dimension)
    if not coarse < intermediate < refined:
        raise ValueError("dimensions must satisfy coarse < intermediate < refined")
    phase = _finite_phase(dimensionless_phase_increment)
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    probe = np.asarray(
        [complex(index + 1, (-1) ** index) for index in range(coarse)],
        dtype=complex,
    )
    direct = zero_refinement_embedding(probe, refined_dimension=refined)
    via_intermediate = zero_refinement_embedding(
        zero_refinement_embedding(probe, refined_dimension=intermediate),
        refined_dimension=refined,
    )
    composition_residual = float(np.linalg.norm(direct - via_intermediate))
    isometry_residual = abs(
        float(np.vdot(probe, probe).real) - float(np.vdot(direct, direct).real)
    )

    coarse_pairing = rigging_pairing(
        probe, probe, dimensionless_phase_increment=phase
    )
    refined_pairing = rigging_pairing(
        direct, direct, dimensionless_phase_increment=phase
    )
    cylindrical_residual = abs(refined_pairing - coarse_pairing)

    coefficients_coarse = generalized_branch_coefficients(
        coarse, dimensionless_phase_increment=phase
    )
    coefficients_refined = generalized_branch_coefficients(
        refined, dimensionless_phase_increment=phase
    )
    embedded_coefficients = zero_refinement_embedding(
        coefficients_coarse, refined_dimension=refined
    )
    difference_norm_squared = float(
        np.vdot(
            coefficients_refined - embedded_coefficients,
            coefficients_refined - embedded_coefficients,
        ).real
    )

    basis = np.eye(coarse, dtype=complex)
    gram = np.asarray(
        [
            [
                rigging_pairing(
                    basis[row], basis[column], dimensionless_phase_increment=phase
                )
                for column in range(coarse)
            ]
            for row in range(coarse)
        ],
        dtype=complex,
    )
    hermiticity_residual = float(np.linalg.norm(gram - gram.conjugate().T))
    eigenvalues = np.linalg.eigvalsh(gram)
    minimum_eigenvalue = float(eigenvalues[0])
    rank = int(np.linalg.matrix_rank(gram, tol=tolerance))

    coarse_unit_vector = coefficients_coarse / math.sqrt(coarse)
    refined_unit_vector = coefficients_refined / math.sqrt(refined)
    coarse_first_basis = np.zeros(coarse, dtype=complex)
    refined_first_basis = np.zeros(refined, dtype=complex)
    coarse_first_basis[0] = 1.0
    refined_first_basis[0] = 1.0
    normalized_coarse_functional = np.vdot(
        coarse_unit_vector, coarse_first_basis
    )
    normalized_refined_functional = np.vdot(
        refined_unit_vector, refined_first_basis
    )
    normalized_coarse_pairing = float(abs(normalized_coarse_functional) ** 2)
    normalized_refined_pairing = float(abs(normalized_refined_functional) ** 2)
    normalized_breaks_consistency = not math.isclose(
        normalized_coarse_pairing,
        normalized_refined_pairing,
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    direct_system_closed = (
        composition_residual <= tolerance and isometry_residual <= tolerance
    )
    residuals_closed = (
        direct_system_closed
        and cylindrical_residual <= tolerance
        and hermiticity_residual <= tolerance
    )
    positive_semidefinite = minimum_eigenvalue >= -tolerance and rank == 1
    exact_difference = math.isclose(
        difference_norm_squared,
        float(refined - coarse),
        rel_tol=0.0,
        abs_tol=tolerance,
    )
    closed = residuals_closed and positive_semidefinite and exact_difference

    return DistributionalRiggingMapAudit(
        coarse_dimension=coarse,
        intermediate_dimension=intermediate,
        refined_dimension=refined,
        dimensionless_phase_increment=phase,
        coarse_truncation_norm_squared=float(coarse),
        refined_truncation_norm_squared=float(refined),
        embedded_truncation_difference_norm_squared=difference_norm_squared,
        cylindrical_pairing_residual=float(cylindrical_residual),
        embedding_composition_residual=composition_residual,
        embedding_isometry_residual=isometry_residual,
        rigging_gram_hermiticity_residual=hermiticity_residual,
        rigging_gram_minimum_eigenvalue=minimum_eigenvalue,
        rigging_gram_rank=rank,
        quotient_dimension=1,
        unit_norm_coarse_probe_pairing=normalized_coarse_pairing,
        unit_norm_refined_probe_pairing=normalized_refined_pairing,
        phase_increment_declared_dimensionless=True,
        direct_system_identity_composition_isometric=direct_system_closed,
        cylindrical_consistency_exact=cylindrical_residual <= tolerance,
        finite_truncations_fail_hilbert_norm_cauchy_criterion=exact_difference,
        finite_support_distributional_limit_eventually_constant=True,
        rigging_pairing_hermitian_positive_semidefinite=positive_semidefinite,
        physical_completion_isomorphic_to_complex=positive_semidefinite,
        unit_norm_truncation_pairings_break_cylindrical_consistency=(
            normalized_breaks_consistency
        ),
        topological_limit_excluded=False,
        geometric_refinement_defined=False,
        eprl_amplitude_used=False,
        renormalized_cutoff_removal_proved=False,
        einstein_hilbert_dominance_proved=False,
        anomaly_free_constraint_algebra_proved=False,
        exactly_two_graviton_helicities_proved=False,
        status=(
            "EXACT_DISTRIBUTIONAL_RIGGING_MAP_MODEL_CLOSED"
            if closed
            else "DISTRIBUTIONAL_RIGGING_MAP_MODEL_AUDIT_FAILED"
        ),
    )
