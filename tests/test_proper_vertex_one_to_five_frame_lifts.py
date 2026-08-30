from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_boundary import (
    BOUNDARY_TETRAHEDRA,
    INTERNAL_TETRAHEDRA,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    IDENTITY_TWO,
    MINKOWSKI_METRIC,
    PAULI_MATRICES,
    canonical_pure_boost,
    certify_lorentzian_one_to_five_frame_lifts,
    exact_tetrahedron_future_normal,
    hermitian_sl2c_boost_lift,
    sl2c_lorentz_matrix,
)


def test_all_fifteen_exact_normal_lines_are_timelike_and_tangent_orthogonal() -> None:
    coordinates = lorentzian_one_to_five_coordinates()

    assert all(tuple(sorted(item)) == item for item in INTERNAL_TETRAHEDRA)
    for tetrahedron in BOUNDARY_TETRAHEDRA + INTERNAL_TETRAHEDRA:
        normal = exact_tetrahedron_future_normal(tetrahedron, coordinates)
        assert normal.tetrahedron == tuple(sorted(tetrahedron))
        assert normal.exact_tangent_annihilations == (0, 0, 0)
        assert normal.exact_vector_squared < 0
        assert normal.exact_future_contravariant_vector[0] > 0
        assert normal.future_unit_normal[0] > 0.0
        assert normal.unit_timelike_residual < 1.0e-14


def test_all_twenty_five_future_and_outward_incidence_data_close() -> None:
    certificate = certify_lorentzian_one_to_five_frame_lifts()

    assert certificate.incidence_count == 25
    assert certificate.boundary_tetrahedron_incidence_count == 5
    assert certificate.internal_tetrahedron_incidence_count == 20
    assert certificate.unique_tetrahedron_count == 15
    assert certificate.all_exact_normal_covectors_annihilate_tangents
    assert certificate.all_normal_lines_timelike
    assert certificate.all_future_unit_normal_representatives
    assert certificate.all_outward_side_evaluations_negative
    assert certificate.all_pure_boosts_proper_orthochronous
    assert certificate.all_hermitian_sl2c_normal_lifts_verified
    assert {record.outward_is_future for record in certificate.incidence_data} == {
        False,
        True,
    }
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_FUTURE_NORMAL_COSET_LIFTS_CLOSED'
    )


def test_pure_boost_and_hermitian_sl2c_lift_are_the_same_lorentz_map() -> None:
    certificate = certify_lorentzian_one_to_five_frame_lifts()
    time_axis = np.asarray((1.0, 0.0, 0.0, 0.0))

    for incidence in certificate.incidence_data:
        normal = incidence.future_unit_normal
        boost = canonical_pure_boost(normal)
        lift = hermitian_sl2c_boost_lift(normal)
        target = normal[0] * IDENTITY_TWO
        for component, pauli in zip(normal[1:], PAULI_MATRICES):
            target = target + component * pauli
        assert np.allclose(boost @ time_axis, normal, rtol=0.0, atol=1.0e-14)
        assert np.allclose(
            boost.T @ MINKOWSKI_METRIC @ boost,
            MINKOWSKI_METRIC,
            rtol=0.0,
            atol=1.0e-14,
        )
        assert np.linalg.det(boost) == pytest.approx(1.0, abs=1.0e-14)
        assert np.allclose(
            lift @ np.conjugate(lift.T),
            target,
            rtol=0.0,
            atol=1.0e-14,
        )
        assert np.linalg.det(lift) == pytest.approx(1.0, abs=1.0e-14)
        assert np.allclose(
            sl2c_lorentz_matrix(lift),
            boost,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_shared_internal_tetrahedra_have_opposite_outward_sides_exactly() -> None:
    certificate = certify_lorentzian_one_to_five_frame_lifts()

    assert certificate.shared_internal_tetrahedron_incidence_counts == (2,) * 10
    assert set(certificate.shared_internal_absolute_face_evaluations) == {
        Fraction(9, 2500)
    }
    assert certificate.shared_future_normal_representatives_agree
    assert certificate.shared_outward_normals_are_opposite
    assert certificate.shared_exact_face_evaluations_are_opposite
    for tetrahedron in INTERNAL_TETRAHEDRA:
        incidences = tuple(
            record
            for record in certificate.incidence_data
            if record.tetrahedron == tetrahedron
        )
        assert len(incidences) == 2
        left, right = incidences
        assert left.exact_face_evaluation == -right.exact_face_evaluation
        assert left.outward_side_sign == -right.outward_side_sign
        assert np.array_equal(
            left.future_unit_normal,
            right.future_unit_normal,
        )
        assert np.array_equal(
            left.outward_unit_normal,
            -right.outward_unit_normal,
        )


def test_positive_rational_scaling_preserves_normals_boosts_and_lifts() -> None:
    unit = certify_lorentzian_one_to_five_frame_lifts()
    scale = Fraction(7, 3)
    scaled = certify_lorentzian_one_to_five_frame_lifts(scale=scale)
    tiny = certify_lorentzian_one_to_five_frame_lifts(
        scale=Fraction(1, 10**100)
    )

    assert scaled.rotation_free_future_normal_coset_representatives_materialized
    assert tiny.rotation_free_future_normal_coset_representatives_materialized
    for base, enlarged, reduced in zip(
        unit.incidence_data,
        scaled.incidence_data,
        tiny.incidence_data,
    ):
        assert enlarged.cell == base.cell == reduced.cell
        assert enlarged.tetrahedron == base.tetrahedron == reduced.tetrahedron
        assert enlarged.outward_side_sign == base.outward_side_sign
        assert reduced.outward_side_sign == base.outward_side_sign
        assert enlarged.exact_face_evaluation == (
            base.exact_face_evaluation * scale**4
        )
        assert enlarged.exact_normal_vector_squared == (
            base.exact_normal_vector_squared * scale**6
        )
        assert np.allclose(
            enlarged.future_unit_normal,
            base.future_unit_normal,
            rtol=0.0,
            atol=1.0e-14,
        )
        assert np.allclose(
            reduced.future_unit_normal,
            base.future_unit_normal,
            rtol=0.0,
            atol=1.0e-14,
        )
        assert np.allclose(enlarged.pure_boost, base.pure_boost, atol=1.0e-14)
        assert np.allclose(reduced.pure_boost, base.pure_boost, atol=1.0e-14)
        assert np.allclose(
            enlarged.hermitian_sl2c_lift,
            base.hermitian_sl2c_lift,
            atol=1.0e-14,
        )
        assert np.allclose(
            reduced.hermitian_sl2c_lift,
            base.hermitian_sl2c_lift,
            atol=1.0e-14,
        )


def test_degenerate_tetrahedron_and_invalid_future_normal_are_rejected() -> None:
    coordinates = lorentzian_one_to_five_coordinates()
    coordinates[5] = coordinates[0]

    with pytest.raises(ValueError, match='future timelike'):
        exact_tetrahedron_future_normal((0, 1, 2, 5), coordinates)
    with pytest.raises(ValueError, match='classical Lorentzian gluing skeleton'):
        certify_lorentzian_one_to_five_frame_lifts(coordinates)
    with pytest.raises(ValueError, match='future unit timelike'):
        canonical_pure_boost((1.0, 1.0, 0.0, 0.0))


def test_exact_near_null_normal_is_rejected_before_float_normalization() -> None:
    epsilon = Fraction(1, 10**30)
    coordinates = {
        0: (Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        1: (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        2: (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        3: (
            Fraction(1) - epsilon,
            Fraction(0),
            Fraction(0),
            Fraction(1),
        ),
    }

    with pytest.raises(ValueError, match='too near-null'):
        exact_tetrahedron_future_normal((0, 1, 2, 3), coordinates)

    with pytest.raises(ValueError, match='positive Fraction'):
        exact_tetrahedron_future_normal(
            (0, 1, 2, 3),
            coordinates,
            max_normal_condition_squared=Fraction(0),
        )


def test_claim_ceiling_is_normal_coset_data_not_an_eprl_boundary_frame() -> None:
    certificate = certify_lorentzian_one_to_five_frame_lifts()

    assert certificate.rotation_free_future_normal_coset_representatives_materialized
    assert not certificate.full_engle_zipfel_boundary_frames_constructed
    assert not certificate.local_su2_tangent_frames_constructed
    assert not certificate.face_bivectors_constructed
    assert not certificate.eprl_orientation_equation_verified
    assert not certificate.relative_regge_transports_constructed
    assert not certificate.face_spinor_phase_gluing_verified
    assert not certificate.local_ls_intertwiners_integrated_with_frames
    assert not certificate.shared_bra_ket_dualization_constructed
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('ROTATION_FREE_COSET_LIFTS_ONLY')
