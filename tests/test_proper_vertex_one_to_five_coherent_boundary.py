from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_boundary import (
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_coherent_boundary import (
    certify_lorentzian_one_to_five_intrinsic_direction_spinors,
    direction_spinor,
    relative_closure_residual,
    spinor_pauli_direction,
)


def test_all_fifteen_tetrahedra_close_and_have_direction_spinors() -> None:
    certificate = certify_lorentzian_one_to_five_intrinsic_direction_spinors()

    assert certificate.tetrahedron_count == 15
    assert certificate.boundary_tetrahedron_count == 5
    assert certificate.internal_tetrahedron_count == 10
    assert certificate.classical_gluing_skeleton_closed
    assert certificate.all_geometric_face_closures_verified
    assert certificate.all_normalized_direction_spinors_materialized
    assert certificate.max_relative_closure_residual < 5.0e-15
    assert certificate.max_spinor_norm_residual < 5.0e-15
    assert certificate.max_pauli_map_residual < 5.0e-15
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_INTRINSIC_CLOSURE_AND_DIRECTION_SPINORS_CLOSED'
    )


def test_every_face_record_reproduces_closure_and_the_pauli_map() -> None:
    certificate = certify_lorentzian_one_to_five_intrinsic_direction_spinors()

    for tetrahedron in certificate.tetrahedron_data:
        assert tetrahedron.nondegenerate_spacelike
        assert tetrahedron.all_face_areas_positive
        assert len(tetrahedron.face_data) == 4
        area_vectors = np.asarray(
            [face.normalized_area_vector for face in tetrahedron.face_data]
        )
        assert np.allclose(
            np.sum(area_vectors, axis=0),
            tetrahedron.normalized_closure_vector,
            rtol=0.0,
            atol=1.0e-15,
        )
        assert relative_closure_residual(area_vectors) < 5.0e-15
        for face in tetrahedron.face_data:
            spinor = np.asarray(face.direction_spinor, dtype=complex)
            assert face.area_squared_exact > 0
            assert np.isclose(np.vdot(spinor, spinor).real, 1.0)
            assert np.allclose(
                spinor_pauli_direction(spinor),
                face.unit_normal,
                rtol=0.0,
                atol=1.0e-14,
            )


def test_canonical_shared_tetrahedron_reconstruction_is_repeatable() -> None:
    certificate = certify_lorentzian_one_to_five_intrinsic_direction_spinors()

    assert certificate.internal_tetrahedron_incidence_counts == (2,) * 10
    assert certificate.repeated_area_squared_labels_match_exactly
    assert certificate.max_repeated_canonical_unit_normal_residual == 0.0
    assert certificate.max_repeated_canonical_direction_spinor_residual == 0.0
    assert certificate.canonical_shared_tetrahedron_reconstruction_is_repeatable


def test_positive_rational_scaling_changes_areas_but_not_directions() -> None:
    unit = certify_lorentzian_one_to_five_intrinsic_direction_spinors()
    scale = Fraction(7, 3)
    scaled = certify_lorentzian_one_to_five_intrinsic_direction_spinors(
        scale=scale
    )
    tiny = certify_lorentzian_one_to_five_intrinsic_direction_spinors(
        scale=Fraction(1, 10**100)
    )

    assert scaled.all_geometric_face_closures_verified
    assert tiny.all_geometric_face_closures_verified
    assert tiny.all_normalized_direction_spinors_materialized
    for unit_tetrahedron, scaled_tetrahedron, tiny_tetrahedron in zip(
        unit.tetrahedron_data,
        scaled.tetrahedron_data,
        tiny.tetrahedron_data,
    ):
        assert scaled_tetrahedron.intrinsic_gram_scale_exact == (
            unit_tetrahedron.intrinsic_gram_scale_exact * scale**2
        )
        for unit_face, scaled_face, tiny_face in zip(
            unit_tetrahedron.face_data,
            scaled_tetrahedron.face_data,
            tiny_tetrahedron.face_data,
        ):
            assert scaled_face.area_squared_exact == (
                unit_face.area_squared_exact * scale**4
            )
            assert np.allclose(
                scaled_face.unit_normal,
                unit_face.unit_normal,
                rtol=0.0,
                atol=1.0e-14,
            )
            assert np.allclose(
                tiny_face.unit_normal,
                unit_face.unit_normal,
                rtol=0.0,
                atol=1.0e-14,
            )
            assert np.allclose(
                scaled_face.direction_spinor,
                unit_face.direction_spinor,
                rtol=0.0,
                atol=1.0e-14,
            )


def test_spinor_south_pole_and_broken_closure_controls() -> None:
    south_spinor = direction_spinor((0.0, 0.0, -1.0))

    assert south_spinor == (0.0j, 1.0 + 0.0j)
    assert np.array_equal(
        spinor_pauli_direction(south_spinor),
        np.asarray((0.0, 0.0, -1.0)),
    )
    near_south_z = -1.0 + 5.0e-13
    near_south = (
        float(np.sqrt(1.0 - near_south_z**2)),
        0.0,
        near_south_z,
    )
    assert np.allclose(
        spinor_pauli_direction(direction_spinor(near_south)),
        near_south,
        rtol=0.0,
        atol=1.0e-14,
    )

    certificate = certify_lorentzian_one_to_five_intrinsic_direction_spinors()
    vectors = np.asarray(
        [
            face.normalized_area_vector
            for face in certificate.tetrahedron_data[0].face_data
        ]
    )
    broken = vectors.copy()
    broken[0] *= -1.0
    assert relative_closure_residual(vectors) < 5.0e-15
    assert relative_closure_residual(broken) > 1.0e-2


def test_degenerate_gluing_is_rejected_before_direction_data_are_claimed() -> None:
    coordinates = lorentzian_one_to_five_coordinates()
    coordinates[5] = coordinates[0]

    with pytest.raises(
        ValueError,
        match='classical Lorentzian gluing skeleton must be closed',
    ):
        certify_lorentzian_one_to_five_intrinsic_direction_spinors(coordinates)


def test_claim_ceiling_keeps_quantum_and_proper_steps_open() -> None:
    certificate = certify_lorentzian_one_to_five_intrinsic_direction_spinors()

    assert not certificate.half_integer_spin_assignment_constructed
    assert not certificate.area_spectrum_scale_and_immirzi_parameter_selected
    assert not certificate.livine_speziale_coherent_intertwiners_constructed
    assert not certificate.tetrahedron_time_orientations_assigned
    assert not certificate.shared_bra_ket_dualization_constructed
    assert not certificate.independent_frame_su2_lifts_constructed
    assert not certificate.lorentzian_sl2c_lifts_constructed
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_single_vertex_integrals_evaluated
    assert not certificate.standard_proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('DIRECTION_SPINORS_ONLY')
