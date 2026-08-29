"""Two-channel distributional rigging model with a declared TT identification.

This module extends the exact ``c_00`` rigging construction by a two-dimensional
channel fibre.  The quotient of the positive cylinder pairing is then ``C^2``.
The two quotient coordinates are identified, by an explicit isometry, with the
plus and cross representatives of the supplied linearized Fierz--Pauli model.

The construction proves compatibility of two already declared structures.  It
does not derive the two-channel fibre from a spin foam, construct a geometric
refinement, or prove Einstein--Hilbert dominance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from examples.physics.distributional_rigging_map import (
    generalized_branch_coefficients,
)
from examples.physics.linearized_spin2_acceptance import (
    transverse_traceless_basis,
)


def _finite_two_channel_state(
    name: str, values: Sequence[Sequence[complex]]
) -> np.ndarray:
    state = np.asarray(values, dtype=complex)
    if (
        state.ndim != 2
        or state.shape[0] == 0
        or state.shape[1] != 2
        or not np.all(np.isfinite(state))
    ):
        raise ValueError(f"{name} must be a finite nonempty N-by-2 state")
    return state


def zero_refine_two_channel_state(
    state: Sequence[Sequence[complex]], *, refined_label_count: int
) -> np.ndarray:
    """Append zero refinement labels while preserving both channels."""

    source = _finite_two_channel_state("state", state)
    if (
        isinstance(refined_label_count, bool)
        or not isinstance(refined_label_count, int)
        or refined_label_count < source.shape[0]
    ):
        raise ValueError(
            "refined_label_count must be an integer at least the source size"
        )
    result = np.zeros((refined_label_count, 2), dtype=complex)
    result[: source.shape[0], :] = source
    return result


def two_channel_functional(
    state: Sequence[Sequence[complex]], *, dimensionless_phase_increment: float
) -> np.ndarray:
    """Return ``(L_+(psi), L_x(psi))`` for a finite-support state."""

    finite_state = _finite_two_channel_state("state", state)
    coefficients = generalized_branch_coefficients(
        finite_state.shape[0],
        dimensionless_phase_increment=dimensionless_phase_increment,
    )
    return np.asarray(np.conjugate(coefficients) @ finite_state, dtype=complex)


def two_channel_rigging_pairing(
    first: Sequence[Sequence[complex]],
    second: Sequence[Sequence[complex]],
    *,
    dimensionless_phase_increment: float,
) -> complex:
    """Return ``sum_a conjugate(L_a(first)) L_a(second)``."""

    first_values = two_channel_functional(
        first, dimensionless_phase_increment=dimensionless_phase_increment
    )
    second_values = two_channel_functional(
        second, dimensionless_phase_increment=dimensionless_phase_increment
    )
    return complex(np.vdot(first_values, second_values))


def normalized_tt_representative(channel_amplitudes: Sequence[complex]) -> np.ndarray:
    """Map two channel amplitudes isometrically to plus/cross TT components."""

    amplitudes = np.asarray(channel_amplitudes, dtype=complex)
    if amplitudes.shape != (2,) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("channel_amplitudes must contain two finite values")
    basis = np.asarray(transverse_traceless_basis(), dtype=complex)
    return basis @ amplitudes / math.sqrt(2.0)


def symmetric_tensor_component_inner_product(
    first: Sequence[complex], second: Sequence[complex]
) -> complex:
    """Use Frobenius weights for the ten independent symmetric components."""

    first_vector = np.asarray(first, dtype=complex)
    second_vector = np.asarray(second, dtype=complex)
    if (
        first_vector.shape != (10,)
        or second_vector.shape != (10,)
        or not np.all(np.isfinite(first_vector))
        or not np.all(np.isfinite(second_vector))
    ):
        raise ValueError("tensor component vectors must contain ten finite values")
    component_labels = tuple(
        (left, right) for left in range(4) for right in range(left, 4)
    )
    weights = np.asarray(
        [1.0 if left == right else 2.0 for left, right in component_labels]
    )
    return complex(np.vdot(first_vector, weights * second_vector))


@dataclass(frozen=True)
class DistributionalHelicityBridgeAudit:
    coarse_label_count: int
    refined_label_count: int
    channel_count: int
    dimensionless_phase_increment: float
    each_coarse_generalized_channel_norm_squared: tuple[float, float]
    each_refined_generalized_channel_norm_squared: tuple[float, float]
    each_embedded_difference_norm_squared: tuple[float, float]
    cylindrical_pairing_residual: float
    finite_cutoff_rigging_gram_rank: int
    finite_cutoff_kernel_dimension: int
    distributional_quotient_dimension: int
    collapsed_channel_negative_control_rank: int
    normalized_tt_basis_gram: tuple[tuple[complex, complex], tuple[complex, complex]]
    two_channel_distributional_pairing_closed: bool
    quotient_isomorphic_to_complex_two: bool
    declared_tt_identification_isometric: bool
    collapsing_channels_loses_one_physical_coordinate: bool
    two_channel_fibre_derived_from_spin_foam: bool
    geometric_refinement_defined: bool
    refinement_limit_fierz_pauli_kernel_derived: bool
    ward_identities_and_extra_pole_exclusion_proved: bool
    einstein_hilbert_dominance_proved: bool
    status: str
    claim_ceiling: str = (
        "EXACT_DECLARED_TWO_CHANNEL_RIGGING_TT_BRIDGE_NOT_EH_EMERGENCE"
    )


def audit_distributional_helicity_bridge(
    coarse_label_count: int,
    refined_label_count: int,
    *,
    dimensionless_phase_increment: float,
    tolerance: float = 1.0e-12,
) -> DistributionalHelicityBridgeAudit:
    """Audit the two-channel distributional quotient and TT isometry."""

    if (
        isinstance(coarse_label_count, bool)
        or isinstance(refined_label_count, bool)
        or not isinstance(coarse_label_count, int)
        or not isinstance(refined_label_count, int)
        or not 0 < coarse_label_count < refined_label_count
    ):
        raise ValueError("label counts must be positive integers with coarse < refined")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    coefficients_coarse = generalized_branch_coefficients(
        coarse_label_count,
        dimensionless_phase_increment=dimensionless_phase_increment,
    )
    coefficients_refined = generalized_branch_coefficients(
        refined_label_count,
        dimensionless_phase_increment=dimensionless_phase_increment,
    )

    coarse_omegas = []
    refined_omegas = []
    difference_norms = []
    for channel in range(2):
        coarse_omega = np.zeros((coarse_label_count, 2), dtype=complex)
        refined_omega = np.zeros((refined_label_count, 2), dtype=complex)
        coarse_omega[:, channel] = coefficients_coarse
        refined_omega[:, channel] = coefficients_refined
        embedded = zero_refine_two_channel_state(
            coarse_omega, refined_label_count=refined_label_count
        )
        difference = refined_omega - embedded
        coarse_omegas.append(coarse_omega)
        refined_omegas.append(refined_omega)
        difference_norms.append(float(np.vdot(difference, difference).real))

    probe = np.asarray(
        [
            (complex(index + 1, 1 - index), complex(-index, index + 2))
            for index in range(coarse_label_count)
        ]
    )
    embedded_probe = zero_refine_two_channel_state(
        probe, refined_label_count=refined_label_count
    )
    cylindrical_residual = abs(
        two_channel_rigging_pairing(
            embedded_probe,
            embedded_probe,
            dimensionless_phase_increment=dimensionless_phase_increment,
        )
        - two_channel_rigging_pairing(
            probe,
            probe,
            dimensionless_phase_increment=dimensionless_phase_increment,
        )
    )

    finite_basis = []
    for label in range(coarse_label_count):
        for channel in range(2):
            basis_state = np.zeros((coarse_label_count, 2), dtype=complex)
            basis_state[label, channel] = 1.0
            finite_basis.append(basis_state)
    gram = np.asarray(
        [
            [
                two_channel_rigging_pairing(
                    first,
                    second,
                    dimensionless_phase_increment=dimensionless_phase_increment,
                )
                for second in finite_basis
            ]
            for first in finite_basis
        ]
    )
    gram_rank = int(np.linalg.matrix_rank(gram, tol=tolerance))

    first_label_channels = finite_basis[:2]
    collapsed_values = []
    for state in first_label_channels:
        values = two_channel_functional(
            state, dimensionless_phase_increment=dimensionless_phase_increment
        )
        collapsed_values.append(values.sum())
    collapsed_gram = np.asarray(
        [
            [first.conjugate() * second for second in collapsed_values]
            for first in collapsed_values
        ]
    )
    collapsed_rank = int(np.linalg.matrix_rank(collapsed_gram, tol=tolerance))

    tt_plus = normalized_tt_representative((1.0, 0.0))
    tt_cross = normalized_tt_representative((0.0, 1.0))
    tt_gram_array = np.asarray(
        [
            [
                symmetric_tensor_component_inner_product(first, second)
                for second in (tt_plus, tt_cross)
            ]
            for first in (tt_plus, tt_cross)
        ]
    )
    tt_isometry = bool(np.allclose(tt_gram_array, np.eye(2), atol=tolerance, rtol=0.0))
    expected_difference = float(refined_label_count - coarse_label_count)
    difference_closed = all(
        math.isclose(value, expected_difference, rel_tol=0.0, abs_tol=tolerance)
        for value in difference_norms
    )
    pairing_closed = cylindrical_residual <= tolerance and gram_rank == 2
    closed = pairing_closed and difference_closed and tt_isometry and collapsed_rank == 1

    return DistributionalHelicityBridgeAudit(
        coarse_label_count=coarse_label_count,
        refined_label_count=refined_label_count,
        channel_count=2,
        dimensionless_phase_increment=float(dimensionless_phase_increment),
        each_coarse_generalized_channel_norm_squared=(
            float(np.vdot(coarse_omegas[0], coarse_omegas[0]).real),
            float(np.vdot(coarse_omegas[1], coarse_omegas[1]).real),
        ),
        each_refined_generalized_channel_norm_squared=(
            float(np.vdot(refined_omegas[0], refined_omegas[0]).real),
            float(np.vdot(refined_omegas[1], refined_omegas[1]).real),
        ),
        each_embedded_difference_norm_squared=(
            difference_norms[0],
            difference_norms[1],
        ),
        cylindrical_pairing_residual=float(cylindrical_residual),
        finite_cutoff_rigging_gram_rank=gram_rank,
        finite_cutoff_kernel_dimension=2 * coarse_label_count - gram_rank,
        distributional_quotient_dimension=2,
        collapsed_channel_negative_control_rank=collapsed_rank,
        normalized_tt_basis_gram=(
            (complex(tt_gram_array[0, 0]), complex(tt_gram_array[0, 1])),
            (complex(tt_gram_array[1, 0]), complex(tt_gram_array[1, 1])),
        ),
        two_channel_distributional_pairing_closed=pairing_closed,
        quotient_isomorphic_to_complex_two=gram_rank == 2,
        declared_tt_identification_isometric=tt_isometry,
        collapsing_channels_loses_one_physical_coordinate=collapsed_rank == 1,
        two_channel_fibre_derived_from_spin_foam=False,
        geometric_refinement_defined=False,
        refinement_limit_fierz_pauli_kernel_derived=False,
        ward_identities_and_extra_pole_exclusion_proved=False,
        einstein_hilbert_dominance_proved=False,
        status=(
            "EXACT_TWO_CHANNEL_DISTRIBUTIONAL_TT_BRIDGE_CLOSED"
            if closed
            else "TWO_CHANNEL_DISTRIBUTIONAL_TT_BRIDGE_AUDIT_FAILED"
        ),
    )
