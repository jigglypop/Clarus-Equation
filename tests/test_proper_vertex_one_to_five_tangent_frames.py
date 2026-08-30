from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    certify_lorentzian_one_to_five_tangent_frames,
    su2_lift_of_rotation,
    su2_rotation_matrix,
)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


@pytest.mark.parametrize('angle', (0.0, 0.2, 1.7, np.pi))
def test_su2_lift_reconstructs_proper_rotation(angle: float) -> None:
    rotation = _axis_angle_rotation(np.asarray((1.0, -2.0, 3.0)), angle)
    lift = su2_lift_of_rotation(rotation)

    assert np.allclose(
        np.conjugate(lift.T) @ lift,
        np.eye(2),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.linalg.det(lift) == pytest.approx(1.0, abs=2.0e-14)
    assert np.allclose(
        su2_rotation_matrix(lift),
        rotation,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_improper_rotation_has_no_su2_lift() -> None:
    with pytest.raises(ValueError, match=r'SO\(3\)'):
        su2_lift_of_rotation(np.diag((1.0, 1.0, -1.0)))


def test_all_fifteen_oriented_tangent_frames_and_lifts_close() -> None:
    certificate = certify_lorentzian_one_to_five_tangent_frames()

    assert certificate.tetrahedron_count == 15
    assert certificate.orientation_preserving_sorted_charts == 8
    assert certificate.orientation_reversing_sorted_charts == 7
    assert not certificate.naive_all_cholesky_charts_admit_su2_lifts
    assert certificate.all_right_handed_tangent_frames_constructed
    assert certificate.all_tangent_su2_lifts_verified
    assert certificate.all_full_sl2c_frame_sections_verified
    assert certificate.max_residual < 2.0e-12
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_ORIENTED_TANGENT_SU2_FRAMES_CLOSED'
    )
    improper = tuple(
        frame.sorted_cholesky_to_rest_comparison
        for frame in certificate.frame_data
        if frame.sorted_cholesky_comparison_determinant < 0.0
    )
    assert len(improper) == 7
    for comparison in improper:
        with pytest.raises(ValueError, match=r'SO\(3\)'):
            su2_lift_of_rotation(comparison)


def test_positive_rational_scaling_preserves_frame_sections() -> None:
    unit = certify_lorentzian_one_to_five_tangent_frames()
    scaled = certify_lorentzian_one_to_five_tangent_frames(
        scale=Fraction(7, 3)
    )
    tiny = certify_lorentzian_one_to_five_tangent_frames(
        scale=Fraction(1, 10**500)
    )
    huge = certify_lorentzian_one_to_five_tangent_frames(
        scale=Fraction(10**500)
    )

    for left, right, reduced, enlarged in zip(
        unit.frame_data,
        scaled.frame_data,
        tiny.frame_data,
        huge.frame_data,
    ):
        assert (
            left.tetrahedron
            == right.tetrahedron
            == reduced.tetrahedron
            == enlarged.tetrahedron
        )
        assert (
            left.sorted_edge_orientation_sign_in_rest_space
            == right.sorted_edge_orientation_sign_in_rest_space
            == reduced.sorted_edge_orientation_sign_in_rest_space
            == enlarged.sorted_edge_orientation_sign_in_rest_space
        )
        assert np.allclose(
            left.right_handed_tangent_rotation,
            right.right_handed_tangent_rotation,
            rtol=0.0,
            atol=2.0e-14,
        )
        assert np.allclose(
            left.full_sl2c_frame,
            right.full_sl2c_frame,
            rtol=0.0,
            atol=2.0e-14,
        )
        assert np.allclose(
            left.full_sl2c_frame,
            reduced.full_sl2c_frame,
            rtol=0.0,
            atol=2.0e-14,
        )
        assert np.allclose(
            left.full_sl2c_frame,
            enlarged.full_sl2c_frame,
            rtol=0.0,
            atol=2.0e-14,
        )
        assert np.allclose(
            left.sorted_cholesky_to_rest_comparison,
            enlarged.sorted_cholesky_to_rest_comparison,
            rtol=0.0,
            atol=2.0e-14,
        )


def test_claim_ceiling_stops_before_regge_phase_or_eprl_data() -> None:
    certificate = certify_lorentzian_one_to_five_tangent_frames()

    assert certificate.local_tangent_su2_frames_constructed
    assert not certificate.face_bivectors_constructed
    assert not certificate.relative_regge_transports_constructed
    assert not certificate.face_spinor_phase_gluing_verified
    assert not certificate.shared_bra_ket_dualization_constructed
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('TANGENT_SU2_FRAME_SECTIONS_ONLY')
