from __future__ import annotations

from fractions import Fraction

from examples.physics.proper_vertex_one_to_five_boundary import (
    BOUNDARY_TETRAHEDRA,
    FINE_SIMPLICES,
    INTERNAL_TETRAHEDRA,
    certify_lorentzian_one_to_five_proper_boundary,
    lorentzian_one_to_five_coordinates,
)


def test_exact_internal_vertex_is_the_boundary_barycentre() -> None:
    coordinates = lorentzian_one_to_five_coordinates()
    expected = tuple(
        sum(coordinates[vertex][component] for vertex in range(5)) / 5
        for component in range(4)
    )

    assert coordinates[5] == expected
    assert expected == (
        Fraction(3, 500),
        Fraction(-2, 5),
        Fraction(-1, 5),
        Fraction(2, 5),
    )


def test_one_to_five_incidence_counts_are_the_actual_four_dimensional_counts() -> None:
    audit = certify_lorentzian_one_to_five_proper_boundary()

    assert len(FINE_SIMPLICES) == 5
    assert len(BOUNDARY_TETRAHEDRA) == 5
    assert len(INTERNAL_TETRAHEDRA) == 10
    assert audit.vertex_count == 6
    assert audit.boundary_edge_count == 10
    assert audit.internal_edge_count == 5
    assert audit.boundary_triangle_count == 10
    assert audit.internal_triangle_count == 10
    assert audit.boundary_tetrahedron_count == 5
    assert audit.internal_tetrahedron_count == 10
    assert audit.fine_four_simplex_count == 5
    assert audit.internal_tetrahedron_incidence_counts == (2,) * 10
    assert audit.internal_triangle_incidence_counts == (3,) * 10
    assert audit.boundary_triangle_incidence_counts == (2,) * 10


def test_coarse_and_all_fine_cells_are_lorentzian_with_spacelike_tetrahedra() -> None:
    audit = certify_lorentzian_one_to_five_proper_boundary()

    assert audit.coarse_simplex_audit.nondegenerate_lorentzian
    assert audit.coarse_simplex_audit.all_boundary_tetrahedra_spacelike
    assert all(item.nondegenerate_lorentzian for item in audit.fine_simplex_audits)
    assert all(item.all_boundary_tetrahedra_spacelike for item in audit.fine_simplex_audits)
    assert all(item.nondegenerate_spacelike for item in audit.boundary_tetrahedron_audits)
    assert all(item.nondegenerate_spacelike for item in audit.internal_tetrahedron_audits)
    assert audit.all_five_cells_nondegenerate_lorentzian
    assert audit.all_fifteen_unique_tetrahedra_spacelike


def test_barycentric_fine_cells_have_consistent_orientation_and_exact_volume_ratio() -> None:
    audit = certify_lorentzian_one_to_five_proper_boundary()

    assert audit.coarse_coordinate_determinant == Fraction(-9, 500)
    assert audit.fine_to_coarse_coordinate_determinant_ratios == (Fraction(1, 5),) * 5
    assert audit.fine_to_coarse_gram_determinant_ratios == (Fraction(1, 25),) * 5
    assert audit.all_fine_cells_share_coarse_orientation
    assert audit.all_fine_four_volumes_are_one_fifth_of_coarse


def test_shared_intrinsic_geometry_and_triangle_labels_match_globally() -> None:
    audit = certify_lorentzian_one_to_five_proper_boundary()

    assert audit.internal_tetrahedra_have_two_incident_cells
    assert audit.internal_triangles_have_three_incident_cells
    assert audit.boundary_triangles_have_two_incident_cells
    assert audit.shared_tetrahedron_intrinsic_shape_matching
    assert audit.global_triangle_area_squared_labels_consistent
    assert audit.all_triangle_area_squared_positive
    assert audit.classical_proper_boundary_geometry_prerequisite_closed
    assert (
        audit.status
        == 'LORENTZIAN_PROPER_COMPATIBLE_1_TO_5_CLASSICAL_BOUNDARY_SKELETON_CLOSED'
    )


def test_positive_rational_scaling_preserves_the_certificate() -> None:
    unit = certify_lorentzian_one_to_five_proper_boundary()
    scaled = certify_lorentzian_one_to_five_proper_boundary(scale=Fraction(7, 3))

    assert scaled.classical_proper_boundary_geometry_prerequisite_closed
    assert scaled.coarse_coordinate_determinant == (
        unit.coarse_coordinate_determinant * Fraction(7, 3) ** 4
    )
    assert scaled.fine_to_coarse_coordinate_determinant_ratios == (Fraction(1, 5),) * 5
    assert scaled.fine_to_coarse_gram_determinant_ratios == (Fraction(1, 25),) * 5


def test_collapsing_the_internal_vertex_is_rejected() -> None:
    coordinates = lorentzian_one_to_five_coordinates()
    coordinates[5] = coordinates[0]
    audit = certify_lorentzian_one_to_five_proper_boundary(coordinates)

    assert not audit.inserted_vertex_is_exact_barycentre
    assert not audit.all_five_cells_nondegenerate_lorentzian
    assert not audit.classical_proper_boundary_geometry_prerequisite_closed
    assert audit.status == 'LORENTZIAN_1_TO_5_BOUNDARY_SKELETON_FAILED'


def test_claim_ceiling_keeps_all_quantum_and_glued_amplitude_steps_open() -> None:
    audit = certify_lorentzian_one_to_five_proper_boundary()

    assert not audit.regge_coherent_spinors_materialized
    assert not audit.half_integer_spin_assignment_constructed
    assert not audit.shared_bra_ket_orientation_data_constructed
    assert not audit.proper_projectors_materialized
    assert not audit.proper_single_vertex_integrals_evaluated
    assert not audit.internal_spins_summed
    assert not audit.internal_intertwiners_integrated
    assert not audit.standard_proper_eprl_five_vertex_amplitude_derived
    assert not audit.proper_eprl_multicell_hessian_computed
