from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_regge_faces import (
    certify_lorentzian_one_to_five_regge_faces,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    su2_lift_of_rotation,
)


def test_exact_full_shape_maps_split_into_twenty_four_rotations_and_reflections() -> None:
    certificate = certify_lorentzian_one_to_five_regge_faces()

    assert certificate.face_transport_count == 50
    assert certificate.proper_full_shape_count == 24
    assert certificate.improper_full_shape_count == 26
    assert certificate.all_exact_face_orientation_signs_match_numeric_frames
    assert certificate.all_full_shape_maps_orthogonal
    assert certificate.all_full_shape_maps_transport_both_labelled_edges
    assert certificate.all_full_shape_maps_reverse_by_transpose
    assert certificate.all_and_only_proper_full_shape_maps_admit_su2_lifts
    assert certificate.max_residual < 8.0e-12
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_FULL_SHAPE_PARITY_NO_GO_CLOSED'
    )

    for record in certificate.full_shape_transports:
        assert record.source_exact_orientation_determinant != 0
        assert record.target_exact_orientation_determinant != 0
        assert record.exact_map_determinant_sign in (-1, 1)
        assert np.linalg.det(record.full_shape_map) == pytest.approx(
            record.exact_map_determinant_sign
        )
        if record.exact_map_determinant_sign == 1:
            assert record.su2_lift is not None
        else:
            assert record.su2_lift is None
            assert record.improper_su2_rejection_verified
            with pytest.raises(ValueError, match=r'SO\(3\)'):
                su2_lift_of_rotation(record.full_shape_map)


def test_explicit_negative_cycle_forbids_global_tetrahedron_parity_repair() -> None:
    certificate = certify_lorentzian_one_to_five_regge_faces()
    cycle = certificate.negative_parity_cycle

    assert cycle.tetrahedra == (
        (1, 3, 4, 5),
        (2, 3, 4, 5),
        (0, 3, 4, 5),
    )
    assert cycle.edge_products == (1, 1, -1)
    assert cycle.cycle_product == -1
    assert certificate.explicit_negative_parity_cycle_verified
    assert certificate.parity_constraint_violation_count > 0
    assert not certificate.global_tetrahedron_parity_assignment_exists


def test_full_lorentz_transport_closes_while_wigner_sections_do_not() -> None:
    certificate = certify_lorentzian_one_to_five_regge_faces()

    assert certificate.all_lorentz_transitions_proper_orthochronous
    assert certificate.all_lorentz_transitions_transport_both_four_edges
    assert certificate.all_wigner_factors_proper_rotations
    assert certificate.full_lorentz_transitions_form_flat_cocycle
    assert certificate.max_lorentz_cocycle_residual < 8.0e-12
    assert not certificate.wigner_factors_form_global_cocycle
    assert certificate.min_wigner_loop_residual == pytest.approx(
        1.3500831831660113e-6
    )
    assert certificate.max_wigner_loop_residual == pytest.approx(
        2.291688313597141e-4
    )
    assert certificate.max_wigner_loop_residual > 1.0e-5
    assert certificate.max_wigner_loop_cell == (0, 1, 5, 3, 4)
    assert certificate.max_wigner_loop_omitted_vertices == (1, 3, 4)
    assert certificate.wigner_existing_local_agreement_count == 24
    assert certificate.wigner_existing_local_maximal_mismatch_count == 26

    mismatches = tuple(
        record.wigner_to_existing_local_residual
        for record in certificate.lorentz_wigner_transports
        if record.wigner_to_existing_local_residual > 8.0e-12
    )
    assert len(mismatches) == 26
    assert all(value == pytest.approx(math.sqrt(8.0)) for value in mismatches)


def test_exact_parity_and_transport_split_are_scale_invariant() -> None:
    unit = certify_lorentzian_one_to_five_regge_faces()
    tiny = certify_lorentzian_one_to_five_regge_faces(
        scale=Fraction(1, 10**500)
    )
    huge = certify_lorentzian_one_to_five_regge_faces(
        scale=Fraction(10**500)
    )

    for reference, reduced, enlarged in zip(
        unit.full_shape_transports,
        tiny.full_shape_transports,
        huge.full_shape_transports,
    ):
        assert reference.triangle == reduced.triangle == enlarged.triangle
        assert reference.exact_map_determinant_sign == (
            reduced.exact_map_determinant_sign
        ) == enlarged.exact_map_determinant_sign
        assert np.allclose(
            reference.full_shape_map,
            reduced.full_shape_map,
            rtol=0.0,
            atol=8.0e-12,
        )
        assert np.allclose(
            reference.full_shape_map,
            enlarged.full_shape_map,
            rtol=0.0,
            atol=8.0e-12,
        )


def test_no_go_claim_ceiling_stops_before_regge_phase_and_eprl() -> None:
    with pytest.raises(ValueError, match='scale cannot'):
        certify_lorentzian_one_to_five_regge_faces(
            coordinates={}, scale=Fraction(2)
        )
    with pytest.raises(ValueError, match='tolerance'):
        certify_lorentzian_one_to_five_regge_faces(tolerance=0.0)

    certificate = certify_lorentzian_one_to_five_regge_faces()
    assert not certificate.global_pointwise_labelled_su2_face_transport_constructed
    assert not certificate.global_regge_spinor_phase_constructed
    assert not certificate.global_eprl_boundary_state_constructed
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('SPLIT_LORENTZ_TRANSPORT_ONLY')
