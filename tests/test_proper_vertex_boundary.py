from fractions import Fraction
import math

import pytest

from examples.physics.proper_vertex_boundary import (
    SHARED_TETRAHEDRA,
    THREE_CELLS,
    constructive_proper_vertex_boundary_certificate,
    four_spin_invariant_exists,
    lorentzian_four_simplex_audit,
    proper_compatible_vertex_coordinates,
    spacelike_tetrahedron_audit,
    spin_quantized_proper_vertex_certificate,
    three_cell_rank_one_proper_amplitude_certificate,
)


EXPECTED_TETRAHEDRON_MINORS = {
    (0, 1, 2, 3): (Fraction(9999, 10000), Fraction(9983, 10000), Fraction(4951, 5000)),
    (0, 1, 2, 4): (Fraction(9999, 10000), Fraction(9983, 10000), Fraction(4931, 5000)),
    (0, 1, 2, 5): (Fraction(9999, 10000), Fraction(9983, 10000), Fraction(44379, 5000)),
    (0, 1, 3, 4): (Fraction(9999, 10000), Fraction(4959, 5000), Fraction(4941, 5000)),
    (0, 1, 3, 5): (Fraction(9999, 10000), Fraction(4959, 5000), Fraction(43479, 5000)),
    (0, 2, 3, 4): (Fraction(624, 625), Fraction(9903, 10000), Fraction(4947, 5000)),
    (0, 2, 3, 5): (Fraction(624, 625), Fraction(9903, 10000), Fraction(9647, 625)),
    (1, 2, 3, 4): (Fraction(19991, 10000), Fraction(14951, 5000), Fraction(9871, 2500)),
    (1, 2, 3, 5): (Fraction(19991, 10000), Fraction(14951, 5000), Fraction(41679, 5000)),
    (1, 2, 4, 5): (Fraction(19991, 10000), Fraction(14811, 5000), Fraction(39033, 5000)),
    (1, 3, 4, 5): (Fraction(1246, 625), Fraction(14831, 5000), Fraction(37143, 5000)),
    (2, 3, 4, 5): (Fraction(799, 400), Fraction(14891, 5000), Fraction(600251, 5000)),
}


def test_all_twelve_unique_boundary_tetrahedra_are_exactly_spacelike() -> None:
    coordinates = proper_compatible_vertex_coordinates()

    for tetrahedron, expected in EXPECTED_TETRAHEDRON_MINORS.items():
        audit = spacelike_tetrahedron_audit(tetrahedron, coordinates)
        assert audit.leading_principal_minors == expected
        assert audit.gram_determinant == expected[-1]
        assert audit.minimum_gram_eigenvalue > 0.0
        assert audit.nondegenerate_spacelike


def test_all_three_cells_are_nondegenerate_lorentzian_four_simplices() -> None:
    coordinates = proper_compatible_vertex_coordinates()
    expected_determinants = (
        Fraction(-1, 2500),
        Fraction(-3969, 2500),
        Fraction(-9, 25),
    )

    for cell, expected in zip(THREE_CELLS, expected_determinants):
        audit = lorentzian_four_simplex_audit(cell, coordinates)
        assert audit.gram_determinant == expected
        assert audit.negative_eigenvalue_count == 1
        assert audit.positive_eigenvalue_count == 3
        assert audit.all_boundary_tetrahedra_spacelike
        assert audit.nondegenerate_lorentzian


def test_certificate_preserves_the_three_shared_tetrahedron_cycle() -> None:
    certificate = constructive_proper_vertex_boundary_certificate()

    assert certificate.cells == THREE_CELLS
    assert certificate.shared_tetrahedra == SHARED_TETRAHEDRA
    assert certificate.unique_tetrahedron_count == 12
    assert certificate.shared_tetrahedra_have_two_incident_cells_in_global_coordinates
    assert certificate.all_twelve_tetrahedra_spacelike
    assert certificate.all_three_cells_lorentzian
    assert certificate.classical_boundary_geometry_ready_for_standard_proper_vertex
    assert certificate.status == "STANDARD_PROPER_VERTEX_CLASSICAL_BOUNDARY_GEOMETRY_CLOSED"


def test_certificate_does_not_claim_quantum_gluing_or_curved_amplitude() -> None:
    certificate = constructive_proper_vertex_boundary_certificate()

    assert not certificate.quantized_regge_coherent_states_constructed
    assert not certificate.three_vertex_glued_amplitude_derived
    assert not certificate.curved_de_sitter_proper_amplitude_derived
    assert certificate.claim_ceiling.endswith("NOT_CURVED_OR_GLUED_AMPLITUDE")


def test_positive_rational_rescaling_preserves_all_signatures() -> None:
    certificate = constructive_proper_vertex_boundary_certificate(
        scale=Fraction(1, 10)
    )

    assert certificate.classical_boundary_geometry_ready_for_standard_proper_vertex
    assert certificate.simplex_audits[0].gram_determinant == Fraction(
        -1, 2500 * 10**8
    )


def test_zero_time_coordinates_destroy_lorentzian_four_volume() -> None:
    coordinates = proper_compatible_vertex_coordinates()
    flattened = {
        vertex: (Fraction(0), coordinate[1], coordinate[2], coordinate[3])
        for vertex, coordinate in coordinates.items()
    }
    audit = lorentzian_four_simplex_audit(THREE_CELLS[0], flattened)

    assert audit.gram_determinant == 0
    assert not audit.nondegenerate_lorentzian


def test_previous_curved_coordinates_fail_standard_spacelike_boundary_hypothesis() -> None:
    previous = {
        0: (Fraction(1), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)),
        1: (Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        2: (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        3: (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        4: (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        5: (Fraction(-1), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)),
    }
    audit = lorentzian_four_simplex_audit(THREE_CELLS[0], previous)

    assert not audit.all_boundary_tetrahedra_spacelike
    assert any(not face.nondegenerate_spacelike for face in audit.boundary_tetrahedra)


@pytest.mark.parametrize("scale", (Fraction(0), Fraction(-1), 1.0))
def test_scale_must_be_a_positive_exact_fraction(scale: object) -> None:
    with pytest.raises(ValueError, match="positive Fraction"):
        proper_compatible_vertex_coordinates(scale=scale)  # type: ignore[arg-type]


def test_symmetric_simplex_has_exact_quantized_area_ratio_and_signature() -> None:
    certificate = spin_quantized_proper_vertex_certificate()

    assert certificate.time_height_squared_over_base_edge_squared == Fraction(5, 256)
    assert certificate.apex_edge_squared_over_base_edge_squared == Fraction(91, 256)
    assert certificate.base_triangle_area_squared_over_base_edge_fourth == Fraction(3, 16)
    assert certificate.apex_triangle_area_squared_over_base_edge_fourth == Fraction(27, 1024)
    assert certificate.base_triangle_spin == 8
    assert certificate.apex_triangle_spin == 3
    assert certificate.base_tetrahedron_leading_minors == (
        Fraction(1),
        Fraction(3, 4),
        Fraction(1, 2),
    )
    assert certificate.side_tetrahedron_leading_minors == (
        Fraction(1),
        Fraction(3, 4),
        Fraction(17, 1024),
    )
    assert certificate.normalized_four_simplex_gram_determinant == Fraction(-5, 512)
    assert certificate.all_five_boundary_tetrahedra_spacelike
    assert certificate.nondegenerate_lorentzian_four_simplex


def test_minimal_integer_rescaling_repairs_side_intertwiner_parity() -> None:
    assert not four_spin_invariant_exists(
        (Fraction(4), Fraction(3, 2), Fraction(3, 2), Fraction(3, 2))
    )
    assert four_spin_invariant_exists(
        (Fraction(8), Fraction(3), Fraction(3), Fraction(3))
    )


def test_one_spin_quantized_proper_vertex_invokes_single_term_theorem() -> None:
    certificate = spin_quantized_proper_vertex_certificate(
        barbero_immirzi_parameter=0.2
    )

    expected_base_angle = math.acosh(4.0 * math.sqrt(2.0 / 17.0))
    expected_apex_angle = math.acosh(37.0 / 17.0)
    expected_phase = 0.2 * (32.0 * expected_base_angle + 18.0 * expected_apex_angle)
    assert certificate.base_tetrahedron_intertwiner_admissible
    assert certificate.side_tetrahedron_intertwiner_admissible
    assert certificate.regge_coherent_boundary_state_exists
    assert certificate.base_triangle_boost_angle == pytest.approx(expected_base_angle)
    assert certificate.apex_triangle_boost_angle == pytest.approx(expected_apex_angle)
    assert certificate.dimensionless_regge_phase_coefficient == pytest.approx(
        expected_phase
    )
    assert certificate.published_proper_vertex_definition_applicable_to_scaling_family
    assert certificate.published_single_term_asymptotic_theorem_applies
    assert certificate.large_spin_limit_is_multiplier_to_infinity
    assert not certificate.explicit_boundary_spinors_and_regge_phases_materialized
    assert not certificate.proper_vertex_amplitude_numerically_evaluated
    assert not certificate.hessian_prefactor_evaluated
    assert certificate.large_spin_power == -12
    assert certificate.status == "ONE_SPIN_QUANTIZED_PROPER_VERTEX_SCALING_FAMILY_CLOSED"


def test_spin_multiplier_scales_spins_and_regge_phase_but_not_shape() -> None:
    first = spin_quantized_proper_vertex_certificate(spin_multiplier=1)
    third = spin_quantized_proper_vertex_certificate(spin_multiplier=3)

    assert third.base_triangle_spin == 3 * first.base_triangle_spin
    assert third.apex_triangle_spin == 3 * first.apex_triangle_spin
    assert third.dimensionless_regge_phase_coefficient == pytest.approx(
        3.0 * first.dimensionless_regge_phase_coefficient
    )
    assert third.normalized_four_simplex_gram_determinant == (
        first.normalized_four_simplex_gram_determinant
    )


def test_one_vertex_result_does_not_claim_gluing_or_curved_amplitude() -> None:
    certificate = spin_quantized_proper_vertex_certificate()

    assert not certificate.three_vertex_glued_amplitude_derived
    assert not certificate.curved_de_sitter_proper_amplitude_derived
    assert certificate.claim_ceiling.endswith("NOT_CURVED_OR_GLUED")
    assert "Eq. (53)" in certificate.amplitude_definition_source
    assert "Theorem 3" in certificate.asymptotic_theorem_source


@pytest.mark.parametrize(
    ("multiplier", "gamma"),
    ((0, 1.0), (1.5, 1.0), (1, 0.0), (1, math.inf)),
)
def test_quantized_proper_vertex_rejects_invalid_scaling_data(
    multiplier: object,
    gamma: float,
) -> None:
    with pytest.raises(ValueError):
        spin_quantized_proper_vertex_certificate(
            spin_multiplier=multiplier,  # type: ignore[arg-type]
            barbero_immirzi_parameter=gamma,
        )


def test_common_apex_gives_one_global_spin_and_edge_shape_assignment() -> None:
    certificate = three_cell_rank_one_proper_amplitude_certificate()

    assert certificate.cell_apices == (1, 1, 1)
    assert certificate.unique_triangle_count == 19
    assert certificate.global_edge_shape_assignment_consistent
    assert certificate.global_triangle_spin_assignment_consistent
    assert certificate.shared_side_tetrahedra_shape_matched
    assert certificate.every_cell_in_single_vertex_proper_scaling_family
    for triangle, spins in certificate.triangle_spin_assignments:
        assert spins == ((Fraction(3),) if 1 in triangle else (Fraction(8),))


def test_rank_one_internal_projectors_make_conditioned_product_asymptotic() -> None:
    single = spin_quantized_proper_vertex_certificate(
        spin_multiplier=2,
        barbero_immirzi_parameter=0.3,
    )
    certificate = three_cell_rank_one_proper_amplitude_certificate(
        spin_multiplier=2,
        barbero_immirzi_parameter=0.3,
    )

    assert certificate.normalized_rank_one_internal_intertwiner_projectors_declared
    assert certificate.internal_projector_definition.startswith("P_e=")
    assert certificate.shared_bra_ket_dualization_declared
    assert certificate.compatible_local_time_orientations_declared
    assert certificate.compatible_positive_regge_phase_branches_declared
    assert not certificate.projector_coherent_spinors_materialized
    assert certificate.fixed_internal_face_amplitudes
    assert certificate.product_haar_measure_and_per_vertex_gauge_fixing_declared
    assert certificate.independent_vertex_gauge_fixings_declared
    assert (
        certificate.conditioned_rank_one_contraction_asymptotic_derived_under_declared_model
    )
    assert certificate.conditioned_large_spin_power == -36
    assert certificate.conditioned_regge_phase_coefficient == pytest.approx(
        3.0 * single.dimensionless_regge_phase_coefficient
    )
    assert not certificate.contraction_numerically_evaluated
    assert certificate.status == "THREE_CELL_RANK_ONE_PROPER_CONTRACTION_ASYMPTOTIC_CLOSED"


def test_rank_one_model_does_not_claim_standard_state_sum_or_curved_vertex() -> None:
    certificate = three_cell_rank_one_proper_amplitude_certificate()

    assert not certificate.internal_spins_summed
    assert not certificate.internal_intertwiners_integrated
    assert not certificate.standard_eprl_multi_vertex_state_sum_derived
    assert not certificate.curved_de_sitter_proper_amplitude_derived
    assert certificate.claim_ceiling.endswith("NOT_STANDARD_STATE_SUM")


def test_mismatched_cell_apices_break_global_internal_data_negative_control() -> None:
    certificate = three_cell_rank_one_proper_amplitude_certificate(
        cell_apices=(1, 2, 1)
    )

    assert not certificate.global_edge_shape_assignment_consistent
    assert not certificate.global_triangle_spin_assignment_consistent
    assert not certificate.shared_side_tetrahedra_shape_matched
    assert not (
        certificate.conditioned_rank_one_contraction_asymptotic_derived_under_declared_model
    )
    assert certificate.conditioned_large_spin_power == 0
    assert math.isnan(certificate.conditioned_regge_phase_coefficient)
    assert certificate.status == "THREE_CELL_PROPER_BOUNDARY_MATCHING_FAILED"


def test_every_declared_cell_apex_must_belong_to_its_cell() -> None:
    with pytest.raises(ValueError, match="belong"):
        three_cell_rank_one_proper_amplitude_certificate(cell_apices=(5, 0, 4))
