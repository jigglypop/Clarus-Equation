from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_incidence_spinors import (
    certify_lorentzian_one_to_five_incidence_spinors,
)


def test_all_hundred_oriented_incidences_and_ninety_links_close() -> None:
    certificate = certify_lorentzian_one_to_five_incidence_spinors()

    assert certificate.incidence_count == 100
    assert certificate.within_cell_link_count == 50
    assert certificate.cross_cell_link_count == 40
    assert certificate.internal_triangle_cycle_count == 10
    assert certificate.boundary_triangle_path_component_count == 10
    assert certificate.all_incidence_normals_unit
    assert certificate.all_exact_incidence_sign_relations_hold
    assert certificate.all_within_cell_maps_proper_and_full_shape
    assert certificate.all_cross_cell_links_use_common_tetrahedron_rest_gauge
    assert certificate.all_link_spinor_equations_verified
    assert certificate.all_reverse_links_verified
    assert certificate.max_link_residual < 8.0e-12
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_INCIDENCE_SPINOR_LINE_SECTION_CLOSED'
    )


def test_cell_orientation_sign_removes_the_old_twenty_six_reflections() -> None:
    certificate = certify_lorentzian_one_to_five_incidence_spinors()

    assert len(certificate.within_cell_links) == 50
    for link in certificate.within_cell_links:
        assert link.exact_incidence_sign_relation_holds
        assert np.linalg.det(link.rotation) == pytest.approx(1.0)
        assert link.first_labelled_edge_residual < 8.0e-12
        assert link.second_labelled_edge_residual < 8.0e-12
        assert link.normal_antipode_residual < 8.0e-12


def test_all_internal_triangle_spin_lift_six_cycles_are_identity() -> None:
    certificate = certify_lorentzian_one_to_five_incidence_spinors()

    assert certificate.all_internal_so3_cycles_identity
    assert certificate.all_internal_su2_lift_signs_globally_consistent
    assert certificate.all_internal_phase_corrected_spinor_cycles_identity
    assert certificate.max_so3_cycle_residual < 8.0e-12
    assert certificate.max_su2_cycle_residual < 8.0e-12
    assert certificate.max_phase_corrected_spinor_cycle_residual < 8.0e-12
    for cycle in certificate.internal_triangle_cycles:
        assert len(cycle.ordered_nodes) == 6
        assert np.allclose(cycle.so3_holonomy, np.eye(3), atol=8.0e-12, rtol=0.0)
        assert np.allclose(cycle.su2_holonomy, np.eye(2), atol=8.0e-12, rtol=0.0)
        assert np.allclose(
            cycle.phase_corrected_spinor_holonomy,
            np.eye(2),
            atol=8.0e-12,
            rtol=0.0,
        )


def test_incidence_spinor_section_survives_extreme_positive_scales() -> None:
    unit = certify_lorentzian_one_to_five_incidence_spinors()
    tiny = certify_lorentzian_one_to_five_incidence_spinors(
        scale=Fraction(1, 10**500)
    )
    huge = certify_lorentzian_one_to_five_incidence_spinors(
        scale=Fraction(10**500)
    )

    for reference, reduced, enlarged in zip(
        unit.within_cell_links, tiny.within_cell_links, huge.within_cell_links
    ):
        assert reference.source_node == reduced.source_node == enlarged.source_node
        assert reference.target_node == reduced.target_node == enlarged.target_node
        assert np.allclose(reference.rotation, reduced.rotation, atol=8.0e-12, rtol=0.0)
        assert np.allclose(reference.rotation, enlarged.rotation, atol=8.0e-12, rtol=0.0)


def test_phase_ceiling_remains_conventional_not_regge_or_eprl() -> None:
    with pytest.raises(ValueError, match='scale cannot'):
        certify_lorentzian_one_to_five_incidence_spinors(
            coordinates={}, scale=Fraction(2)
        )
    with pytest.raises(ValueError, match='tolerance'):
        certify_lorentzian_one_to_five_incidence_spinors(tolerance=0.0)

    certificate = certify_lorentzian_one_to_five_incidence_spinors()
    assert certificate.incidence_oriented_spinor_line_section_constructed
    assert certificate.linkwise_u1_phase_convention_constructed
    assert not certificate.physical_regge_state_phase_constructed
    assert not certificate.regge_action_phase_derived
    assert not certificate.global_eprl_boundary_state_constructed
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('SPINOR_LINE_SECTION_ONLY')
