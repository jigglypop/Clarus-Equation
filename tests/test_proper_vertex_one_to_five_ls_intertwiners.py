from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_coherent_boundary import (
    direction_spinor,
)
from examples.physics.proper_vertex_one_to_five_ls_intertwiners import (
    allowed_four_valent_recoupling_channels,
    certify_lorentzian_one_to_five_local_ls_intertwiners,
    four_valent_invariant_basis_amplitudes,
    four_valent_ls_recoupling_coefficients,
    integer_clebsch_gordan,
    spin_coherent_state_coefficients,
)


def test_integer_clebsch_gordan_matches_known_condon_shortley_values() -> None:
    assert integer_clebsch_gordan(1, 1, 1, -1, 0, 0) == pytest.approx(
        1.0 / math.sqrt(3.0)
    )
    assert integer_clebsch_gordan(1, 0, 1, 0, 0, 0) == pytest.approx(
        -1.0 / math.sqrt(3.0)
    )
    assert integer_clebsch_gordan(1, 1, 1, 0, 1, 1) == pytest.approx(
        1.0 / math.sqrt(2.0)
    )
    assert integer_clebsch_gordan(1, 1, 1, 1, 0, 0) == 0.0


def test_integer_clebsch_gordan_coupled_states_are_normalized() -> None:
    first_spin = 2
    second_spin = 1

    for coupled_spin in range(1, 4):
        for coupled_magnetic in range(-coupled_spin, coupled_spin + 1):
            norm = sum(
                integer_clebsch_gordan(
                    first_spin,
                    first_magnetic,
                    second_spin,
                    second_magnetic,
                    coupled_spin,
                    coupled_magnetic,
                )
                ** 2
                for first_magnetic in range(-first_spin, first_spin + 1)
                for second_magnetic in range(-second_spin, second_spin + 1)
            )
            assert norm == pytest.approx(1.0, abs=1.0e-14)


def test_spin_coherent_state_is_normalized_and_points_along_the_normal() -> None:
    normal = np.asarray((0.3, -0.4, math.sqrt(0.75)))
    spin_j = 5
    coefficients = np.asarray(
        spin_coherent_state_coefficients(spin_j, direction_spinor(normal)),
        dtype=complex,
    )
    magnetic_values = np.arange(-spin_j, spin_j + 1)
    raising_expectation = sum(
        np.conjugate(coefficients[index + 1])
        * coefficients[index]
        * math.sqrt(
            (spin_j - magnetic_m) * (spin_j + magnetic_m + 1)
        )
        for index, magnetic_m in enumerate(range(-spin_j, spin_j))
    )

    assert np.vdot(coefficients, coefficients).real == pytest.approx(1.0)
    assert np.sum(abs(coefficients) ** 2 * magnetic_values) == pytest.approx(
        spin_j * normal[2]
    )
    assert raising_expectation.real == pytest.approx(spin_j * normal[0])
    assert raising_expectation.imag == pytest.approx(spin_j * normal[1])


def test_four_spin_zero_projection_is_the_identity_witness() -> None:
    channels, coefficients = four_valent_ls_recoupling_coefficients(
        (0, 0, 0, 0),
        ((1.0, 0.0),) * 4,
    )

    assert allowed_four_valent_recoupling_channels((0, 0, 0, 0)) == (0,)
    assert channels == (0,)
    assert coefficients == (1.0 + 0.0j,)


def test_nontrivial_recoupling_basis_is_orthonormal_and_su2_invariant() -> None:
    spins = (1, 1, 1, 1)
    basis = {
        channel: four_valent_invariant_basis_amplitudes(spins, channel)
        for channel in allowed_four_valent_recoupling_channels(spins)
    }

    for left_channel, left in basis.items():
        for right_channel, right in basis.items():
            inner = sum(
                np.conjugate(left.get(label, 0.0j))
                * right.get(label, 0.0j)
                for label in set(left) | set(right)
            )
            assert inner == pytest.approx(
                1.0 if left_channel == right_channel else 0.0,
                abs=1.0e-14,
            )
        assert all(
            abs(sum(label) * amplitude) < 1.0e-14
            for label, amplitude in left.items()
        )
        raised: dict[tuple[int, int, int, int], complex] = {}
        for label, amplitude in left.items():
            for index, (spin, magnetic) in enumerate(zip(spins, label)):
                if magnetic < spin:
                    target = list(label)
                    target[index] += 1
                    target_label = tuple(target)
                    raised[target_label] = raised.get(target_label, 0.0j) + (
                        amplitude
                        * math.sqrt((spin - magnetic) * (spin + magnetic + 1))
                    )
        assert math.sqrt(sum(abs(value) ** 2 for value in raised.values())) < 1.0e-14


def test_recoupling_coefficients_equal_direct_magnetic_basis_projection() -> None:
    spins = (1, 1, 1, 1)
    spinors = (
        direction_spinor((0.0, 0.0, 1.0)),
        direction_spinor((1.0, 0.0, 0.0)),
        direction_spinor((0.0, 1.0, 0.0)),
        direction_spinor((0.0, 0.0, -1.0)),
    )
    channels, coefficients = four_valent_ls_recoupling_coefficients(
        spins,
        spinors,
    )
    coherent_states = tuple(
        spin_coherent_state_coefficients(spin, spinor)
        for spin, spinor in zip(spins, spinors)
    )

    for channel, coefficient in zip(channels, coefficients):
        basis = four_valent_invariant_basis_amplitudes(spins, channel)
        direct = sum(
            np.conjugate(amplitude)
            * math.prod(
                coherent_states[index][magnetic + spins[index]]
                for index, magnetic in enumerate(label)
            )
            for label, amplitude in basis.items()
        )
        assert coefficient == pytest.approx(direct, abs=1.0e-14)


def test_all_fifteen_level_three_local_ls_intertwiners_are_materialized() -> None:
    certificate = certify_lorentzian_one_to_five_local_ls_intertwiners()

    assert certificate.spin_level == 3
    assert certificate.tetrahedron_count == 15
    assert certificate.global_triangle_spin_count == 20
    assert certificate.max_spin_j == 9
    assert certificate.all_fifteen_invariant_spaces_nonzero
    assert certificate.all_fifteen_local_ls_group_averages_numerically_nonzero
    assert certificate.min_unnormalized_group_average_norm > 0.07
    assert certificate.minimum_norm_to_tolerance_ratio > 1.0e12
    assert certificate.max_normalized_coefficient_norm_residual < 1.0e-14
    assert certificate.max_product_coherent_state_norm_residual < 1.0e-13
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_LOCAL_NUMERICAL_LS_INTERTWINERS_MATERIALIZED'
    )
    for tetrahedron in certificate.tetrahedron_data:
        assert tetrahedron.invariant_space_dimension == len(
            tetrahedron.recoupling_channel_k_values
        )
        assert tetrahedron.invariant_space_dimension == len(
            tetrahedron.unnormalized_recoupling_coefficients
        )
        assert (
            tetrahedron.numerically_nonzero_group_averaged_intertwiner_materialized
        )
        assert sum(
            abs(value) ** 2
            for value in tetrahedron.normalized_recoupling_coefficients
        ) == pytest.approx(1.0, abs=1.0e-14)


def test_level_three_face_spins_remain_global_across_local_intertwiners() -> None:
    certificate = certify_lorentzian_one_to_five_local_ls_intertwiners()
    observed: dict[tuple[int, int, int], set[int]] = {}

    for tetrahedron in certificate.tetrahedron_data:
        for triangle, spin in zip(
            tetrahedron.face_triangles,
            tetrahedron.face_spin_j_labels,
        ):
            observed.setdefault(triangle, set()).add(spin)
    assert len(observed) == 20
    assert all(len(spins) == 1 for spins in observed.values())


def test_computational_cap_and_nonadmissible_small_level_are_rejected() -> None:
    with pytest.raises(ValueError, match='positive integers'):
        certify_lorentzian_one_to_five_local_ls_intertwiners(spin_level=1)
    with pytest.raises(ValueError, match='computational cap'):
        certify_lorentzian_one_to_five_local_ls_intertwiners(
            spin_level=118,
            computational_spin_cap=12,
        )


def test_claim_ceiling_is_local_ls_not_eprl_or_proper_gluing() -> None:
    certificate = certify_lorentzian_one_to_five_local_ls_intertwiners()

    assert certificate.haar_projector_algebraically_reconstructed_from_orthonormal_basis
    assert not certificate.haar_quadrature_numerically_performed
    assert not certificate.spin_weighted_geometric_closure_exact
    assert not certificate.independent_tetrahedron_su2_frames_constructed
    assert not certificate.tetrahedron_time_orientations_assigned
    assert not certificate.shared_bra_ket_dualization_constructed
    assert not certificate.globally_glued_spin_network_constructed
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.lorentzian_sl2c_group_integrals_evaluated
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling == 'LOCAL_NUMERICAL_LS_INTERTWINERS_ONLY'
