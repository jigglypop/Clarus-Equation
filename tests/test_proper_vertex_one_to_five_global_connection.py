from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    MINKOWSKI_METRIC,
    canonical_pure_boost,
)
from examples.physics.proper_vertex_one_to_five_global_connection import (
    certify_lorentzian_one_to_five_global_connection,
)


def test_global_flat_connection_closes_all_cell_and_hinge_counts() -> None:
    certificate = certify_lorentzian_one_to_five_global_connection()

    assert certificate.cell_count == 5
    assert certificate.shared_tetrahedron_transition_count == 10
    assert certificate.internal_triangle_holonomy_count == 10
    assert certificate.all_shared_shapes_match_exactly
    assert certificate.common_global_affine_frame_declared
    assert certificate.all_cell_frames_proper_orthochronous
    assert certificate.all_sl2c_cell_lifts_verified
    assert certificate.all_shared_transitions_proper_orthochronous
    assert certificate.all_transition_inverse_relations_verified
    assert certificate.all_transition_cocycles_verified
    assert certificate.all_shared_tangents_and_future_normals_preserved
    assert certificate.all_shared_outward_normals_mapped_antipodally
    assert certificate.all_internal_hinge_tangent_planes_fixed
    assert certificate.all_internal_triangle_holonomies_identity
    assert certificate.all_internal_regge_boost_deficits_zero
    assert certificate.max_cell_frame_residual < 4.0e-12
    assert certificate.max_transition_residual < 4.0e-12
    assert certificate.max_cocycle_residual < 4.0e-12
    assert certificate.max_holonomy_residual < 4.0e-12
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_GLOBAL_FLAT_COFRAME_CONNECTION_CLOSED'
    )


def test_each_cell_frame_and_shared_transition_is_proper_orthochronous() -> None:
    certificate = certify_lorentzian_one_to_five_global_connection()

    for frame in certificate.cell_coframes:
        assert np.allclose(
            frame.lorentz_frame.T
            @ MINKOWSKI_METRIC
            @ frame.lorentz_frame,
            MINKOWSKI_METRIC,
            rtol=0.0,
            atol=4.0e-12,
        )
        assert np.linalg.det(frame.lorentz_frame) == pytest.approx(1.0)
        assert frame.lorentz_frame[0, 0] >= 1.0

    for transition in certificate.shared_transitions:
        assert np.allclose(
            transition.lorentz_transition.T
            @ MINKOWSKI_METRIC
            @ transition.lorentz_transition,
            MINKOWSKI_METRIC,
            rtol=0.0,
            atol=4.0e-12,
        )
        assert np.linalg.det(transition.lorentz_transition) == pytest.approx(1.0)
        assert transition.lorentz_transition[0, 0] >= 1.0
        assert transition.source_outward_sign == -transition.target_outward_sign
        assert transition.inverse_residual < 4.0e-12
        assert transition.sl2c_inverse_residual < 4.0e-12
        assert transition.global_future_normal_agreement_residual < 4.0e-12
        assert transition.shared_future_normal_residual < 4.0e-12
        assert transition.outward_antipode_residual < 4.0e-12


def test_all_internal_triangle_dual_loops_are_identity_not_local_face_loops() -> None:
    certificate = certify_lorentzian_one_to_five_global_connection()

    assert len(certificate.internal_triangle_holonomies) == 10
    for record in certificate.internal_triangle_holonomies:
        assert len(record.ordered_cells) == 3
        assert np.allclose(
            record.lorentz_holonomy, np.eye(4), rtol=0.0, atol=4.0e-12
        )
        assert np.allclose(
            record.sl2c_holonomy, np.eye(2), rtol=0.0, atol=4.0e-12
        )
        assert record.hinge_tangent_residual < 4.0e-12
        assert record.boost_trace_domain_residual < 4.0e-12
        assert record.boost_deficit < 4.0e-12

    assert certificate.max_cell_local_pairwise_loop_residual > 1.0
    assert not certificate.cell_local_pairwise_links_form_global_connection


def test_positive_rational_scale_preserves_global_connection() -> None:
    unit = certify_lorentzian_one_to_five_global_connection()
    tiny = certify_lorentzian_one_to_five_global_connection(
        scale=Fraction(1, 10**500)
    )
    huge = certify_lorentzian_one_to_five_global_connection(
        scale=Fraction(10**500)
    )

    for reference, reduced, enlarged in zip(
        unit.shared_transitions,
        tiny.shared_transitions,
        huge.shared_transitions,
    ):
        assert reference.tetrahedron == reduced.tetrahedron == enlarged.tetrahedron
        assert np.allclose(
            reference.lorentz_transition,
            reduced.lorentz_transition,
            rtol=0.0,
            atol=4.0e-12,
        )
        assert np.allclose(
            reference.lorentz_transition,
            enlarged.lorentz_transition,
            rtol=0.0,
            atol=4.0e-12,
        )


def test_cell_frame_gauge_changes_conjugate_loops_without_changing_flatness() -> None:
    certificate = certify_lorentzian_one_to_five_global_connection()
    frames = [record.lorentz_frame for record in certificate.cell_coframes]
    gauges = [
        canonical_pure_boost(
            (np.cosh(index / 20.0), np.sinh(index / 20.0), 0.0, 0.0)
        )
        for index in range(5)
    ]
    transformed = [frame @ gauge for frame, gauge in zip(frames, gauges)]

    first, second, third = 0, 1, 2
    first_to_second = np.linalg.solve(transformed[second], transformed[first])
    second_to_third = np.linalg.solve(transformed[third], transformed[second])
    third_to_first = np.linalg.solve(transformed[first], transformed[third])
    loop = third_to_first @ second_to_third @ first_to_second

    original = np.linalg.solve(frames[second], frames[first])
    expected = (
        np.linalg.solve(gauges[second], original @ gauges[first])
    )
    assert np.allclose(first_to_second, expected, rtol=0.0, atol=4.0e-12)
    assert np.allclose(loop, np.eye(4), rtol=0.0, atol=4.0e-12)


def test_invalid_inputs_and_claim_ceiling() -> None:
    with pytest.raises(ValueError, match='scale cannot'):
        certify_lorentzian_one_to_five_global_connection(
            coordinates={}, scale=Fraction(2)
        )
    with pytest.raises(ValueError, match='tolerance'):
        certify_lorentzian_one_to_five_global_connection(tolerance=0.0)

    certificate = certify_lorentzian_one_to_five_global_connection()
    assert certificate.global_flat_affine_levi_civita_connection_constructed
    assert not certificate.global_regge_spinor_phase_constructed
    assert not certificate.global_eprl_boundary_state_constructed
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.proper_projectors_materialized
    assert not certificate.lorentzian_sl2c_group_integrals_evaluated
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('GLOBAL_COFRAME_CONNECTION_ONLY')
