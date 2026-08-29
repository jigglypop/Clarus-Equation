import math

import numpy as np
import pytest

from examples.physics.distributional_helicity_bridge import (
    audit_distributional_helicity_bridge,
    normalized_tt_representative,
    symmetric_tensor_component_inner_product,
    two_channel_functional,
    two_channel_rigging_pairing,
    zero_refine_two_channel_state,
)


def test_two_channel_functional_is_eventually_constant() -> None:
    state = ((1.0, 2.0j), (-0.5j, 3.0))
    refined = zero_refine_two_channel_state(state, refined_label_count=8)

    assert two_channel_functional(
        refined, dimensionless_phase_increment=0.4
    ) == pytest.approx(
        two_channel_functional(state, dimensionless_phase_increment=0.4)
    )


def test_two_channel_pairing_is_hermitian_and_positive() -> None:
    first = np.asarray(((1.0, 2.0j), (0.2, -0.3j)))
    second = np.asarray(((-1.0j, 0.5), (2.0, 0.1j)))
    phase = -0.3
    scalar = 0.4 + 1.2j

    pairing = two_channel_rigging_pairing(
        first, second, dimensionless_phase_increment=phase
    )
    assert pairing == pytest.approx(
        two_channel_rigging_pairing(
            second, first, dimensionless_phase_increment=phase
        ).conjugate()
    )
    diagonal = two_channel_rigging_pairing(
        first, first, dimensionless_phase_increment=phase
    )
    assert diagonal.imag == pytest.approx(0.0)
    assert diagonal.real >= 0.0
    assert two_channel_rigging_pairing(
        scalar * first, second, dimensionless_phase_increment=phase
    ) == pytest.approx(scalar.conjugate() * pairing)
    assert two_channel_rigging_pairing(
        first, scalar * second, dimensionless_phase_increment=phase
    ) == pytest.approx(scalar * pairing)


def test_tt_channel_map_is_an_isometry_for_frobenius_component_metric() -> None:
    amplitudes = np.asarray((0.7 - 0.2j, -1.1 + 0.4j))
    tensor = normalized_tt_representative(amplitudes)

    assert symmetric_tensor_component_inner_product(tensor, tensor) == pytest.approx(
        np.vdot(amplitudes, amplitudes)
    )
    plus = normalized_tt_representative((1.0, 0.0))
    cross = normalized_tt_representative((0.0, 1.0))
    assert symmetric_tensor_component_inner_product(plus, plus) == pytest.approx(1.0)
    assert symmetric_tensor_component_inner_product(cross, cross) == pytest.approx(1.0)
    assert symmetric_tensor_component_inner_product(plus, cross) == pytest.approx(0.0)


def test_two_channel_audit_has_two_dimensional_quotient() -> None:
    audit = audit_distributional_helicity_bridge(
        3, 9, dimensionless_phase_increment=0.37
    )

    assert audit.each_coarse_generalized_channel_norm_squared == pytest.approx(
        (3.0, 3.0)
    )
    assert audit.each_refined_generalized_channel_norm_squared == pytest.approx(
        (9.0, 9.0)
    )
    assert audit.each_embedded_difference_norm_squared == pytest.approx((6.0, 6.0))
    assert audit.cylindrical_pairing_residual == pytest.approx(0.0, abs=1.0e-12)
    assert audit.finite_cutoff_rigging_gram_rank == 2
    assert audit.finite_cutoff_kernel_dimension == 4
    assert audit.distributional_quotient_dimension == 2
    assert audit.quotient_isomorphic_to_complex_two
    assert audit.declared_tt_identification_isometric
    assert audit.collapsed_channel_negative_control_rank == 1
    assert audit.collapsing_channels_loses_one_physical_coordinate
    assert audit.status == "EXACT_TWO_CHANNEL_DISTRIBUTIONAL_TT_BRIDGE_CLOSED"


def test_two_channel_bridge_keeps_eh_emergence_claims_false() -> None:
    audit = audit_distributional_helicity_bridge(
        2, 5, dimensionless_phase_increment=-0.8
    )

    assert not audit.two_channel_fibre_derived_from_spin_foam
    assert not audit.geometric_refinement_defined
    assert not audit.refinement_limit_fierz_pauli_kernel_derived
    assert not audit.ward_identities_and_extra_pole_exclusion_proved
    assert not audit.einstein_hilbert_dominance_proved
    assert audit.claim_ceiling.endswith("NOT_EH_EMERGENCE")


@pytest.mark.parametrize("label_count", (1, 3, 6))
def test_full_finite_basis_gram_has_rank_two_and_expected_kernel(
    label_count: int,
) -> None:
    phase = 0.21
    basis = []
    for label in range(label_count):
        for channel in range(2):
            state = np.zeros((label_count, 2), dtype=complex)
            state[label, channel] = 1.0
            basis.append(state)
    gram = np.asarray(
        [
            [
                two_channel_rigging_pairing(
                    first, second, dimensionless_phase_increment=phase
                )
                for second in basis
            ]
            for first in basis
        ]
    )

    assert np.linalg.matrix_rank(gram, tol=1.0e-12) == 2
    assert gram.shape[0] - np.linalg.matrix_rank(gram, tol=1.0e-12) == 2 * label_count - 2


def test_collapsing_both_channels_is_rank_one_on_the_full_finite_basis() -> None:
    label_count = 4
    phase = -0.17
    collapsed_values = []
    for label in range(label_count):
        for channel in range(2):
            state = np.zeros((label_count, 2), dtype=complex)
            state[label, channel] = 1.0
            collapsed_values.append(
                two_channel_functional(
                    state, dimensionless_phase_increment=phase
                ).sum()
            )
    gram = np.asarray(
        [
            [first.conjugate() * second for second in collapsed_values]
            for first in collapsed_values
        ]
    )

    assert np.linalg.matrix_rank(gram, tol=1.0e-12) == 1


@pytest.mark.parametrize("counts", ((0, 2), (2, 2), (4, 3), (True, 3)))
def test_invalid_label_counts_are_rejected(counts: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="label counts"):
        audit_distributional_helicity_bridge(
            *counts, dimensionless_phase_increment=0.0
        )


@pytest.mark.parametrize("tolerance", (0.0, -1.0, math.inf, math.nan))
def test_invalid_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        audit_distributional_helicity_bridge(
            1,
            2,
            dimensionless_phase_increment=0.0,
            tolerance=tolerance,
        )


def test_invalid_channel_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="N-by-2"):
        two_channel_functional(((1.0,),), dimensionless_phase_increment=0.0)
    with pytest.raises(ValueError, match="two finite"):
        normalized_tt_representative((1.0,))
    with pytest.raises(ValueError, match="ten finite"):
        symmetric_tensor_component_inner_product((1.0,), (1.0,))
