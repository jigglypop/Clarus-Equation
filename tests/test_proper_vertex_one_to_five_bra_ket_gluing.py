from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_bra_ket_gluing import (
    cell_local_bra_ket_gluing,
    certify_lorentzian_one_to_five_bra_ket_gluing,
    j_dual_spinor,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    FINE_SIMPLICES,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_coherent_boundary import (
    direction_spinor,
    spinor_pauli_direction,
)


def test_j_dual_spinor_is_normalized_antipodal_and_squares_to_minus_one() -> None:
    spinor = np.asarray(direction_spinor((0.3, -0.4, np.sqrt(0.75))))
    dual = j_dual_spinor(spinor)

    assert np.vdot(dual, dual).real == pytest.approx(1.0)
    assert np.allclose(
        spinor_pauli_direction(dual),
        -spinor_pauli_direction(spinor),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.allclose(j_dual_spinor(dual), -spinor, atol=1.0e-14)


def test_all_fifty_cell_local_bra_ket_phase_equations_close() -> None:
    certificate = certify_lorentzian_one_to_five_bra_ket_gluing()

    assert certificate.fine_cell_count == 5
    assert certificate.gluing_count == 50
    assert (
        certificate.second_tangent_preserving_count
        + certificate.second_tangent_reversing_count
        == 50
    )
    assert certificate.second_tangent_preserving_count > 0
    assert certificate.second_tangent_reversing_count > 0
    assert certificate.all_relative_rotations_proper
    assert certificate.all_outward_normals_mapped_antipodally
    assert certificate.all_su2_lifts_verified
    assert certificate.all_j_dualized_phase_equations_verified
    assert certificate.max_residual < 3.0e-12
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_CELL_LOCAL_BRA_KET_GLUING_CLOSED'
    )


def test_positive_rational_scaling_preserves_gluing_sections() -> None:
    unit = certify_lorentzian_one_to_five_bra_ket_gluing()
    scaled = certify_lorentzian_one_to_five_bra_ket_gluing(
        scale=Fraction(7, 3)
    )
    tiny = certify_lorentzian_one_to_five_bra_ket_gluing(
        scale=Fraction(1, 10**500)
    )
    huge = certify_lorentzian_one_to_five_bra_ket_gluing(
        scale=Fraction(10**500)
    )

    for left, right, reduced, enlarged in zip(
        unit.gluing_data,
        scaled.gluing_data,
        tiny.gluing_data,
        huge.gluing_data,
    ):
        assert left.cell == right.cell == reduced.cell == enlarged.cell
        assert (
            left.triangle
            == right.triangle
            == reduced.triangle
            == enlarged.triangle
        )
        assert left.second_tangent_transport_sign == (
            right.second_tangent_transport_sign
        ) == reduced.second_tangent_transport_sign == (
            enlarged.second_tangent_transport_sign
        )
        assert np.allclose(
            left.relative_rotation,
            right.relative_rotation,
            rtol=0.0,
            atol=3.0e-14,
        )
        assert np.allclose(
            left.relative_rotation,
            reduced.relative_rotation,
            rtol=0.0,
            atol=3.0e-14,
        )
        assert np.allclose(
            left.relative_rotation,
            enlarged.relative_rotation,
            rtol=0.0,
            atol=3.0e-14,
        )


def test_reverse_links_are_inverse_rotations_and_su2_lifts_up_to_sign() -> None:
    coordinates = lorentzian_one_to_five_coordinates()

    for cell in FINE_SIMPLICES:
        for left_vertex, right_vertex in combinations(sorted(cell), 2):
            forward = cell_local_bra_ket_gluing(
                cell,
                left_vertex,
                right_vertex,
                coordinates,
            )
            reverse = cell_local_bra_ket_gluing(
                cell,
                right_vertex,
                left_vertex,
                coordinates,
            )
            assert np.allclose(
                reverse.relative_rotation,
                forward.relative_rotation.T,
                rtol=0.0,
                atol=3.0e-14,
            )
            inverse_lift = np.conjugate(forward.relative_su2_lift.T)
            assert min(
                np.linalg.norm(reverse.relative_su2_lift - inverse_lift),
                np.linalg.norm(reverse.relative_su2_lift + inverse_lift),
            ) < 3.0e-14


def test_pairwise_face_sections_do_not_imply_trivial_triangle_loops() -> None:
    coordinates = lorentzian_one_to_five_coordinates()
    loop_residuals: list[float] = []

    for cell in FINE_SIMPLICES:
        for first, second, third in combinations(sorted(cell), 3):
            first_to_second = cell_local_bra_ket_gluing(
                cell, first, second, coordinates
            ).relative_rotation
            second_to_third = cell_local_bra_ket_gluing(
                cell, second, third, coordinates
            ).relative_rotation
            third_to_first = cell_local_bra_ket_gluing(
                cell, third, first, coordinates
            ).relative_rotation
            loop_residuals.append(
                float(
                    np.linalg.norm(
                        third_to_first
                        @ second_to_third
                        @ first_to_second
                        - np.eye(3)
                    )
                )
            )

    assert len(loop_residuals) == 50
    assert max(loop_residuals) > 1.0


def test_claim_ceiling_stops_before_global_regge_or_eprl_amplitude() -> None:
    certificate = certify_lorentzian_one_to_five_bra_ket_gluing()

    assert certificate.edge_aligned_face_transport_sections_constructed
    assert certificate.cell_local_j_dualized_matching_constructed
    assert certificate.face_spinor_phase_gluing_verified
    assert not certificate.global_regge_levi_civita_holonomy_derived
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.proper_projectors_materialized
    assert not certificate.lorentzian_sl2c_group_integrals_evaluated
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('J_DUALIZED_GLUING_ONLY')
