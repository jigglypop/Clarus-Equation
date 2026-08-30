from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_bivectors import (
    certify_lorentzian_one_to_five_bivectors,
)


def test_all_fifty_exact_first_bivector_routes_agree() -> None:
    certificate = certify_lorentzian_one_to_five_bivectors()

    assert certificate.cell_wedge_count == 50
    assert certificate.cross_cell_transport_count == 40
    assert certificate.internal_triangle_loop_count == 10
    assert certificate.all_exact_area_identities_hold
    assert certificate.all_sigma_bivectors_antisymmetric
    assert certificate.all_reverse_label_antisymmetries_verified
    assert certificate.all_cross_cell_exact_canonical_bivectors_agree
    assert certificate.all_three_bivector_routes_agree
    assert certificate.all_classical_signed_orientation_equations_verified
    assert certificate.all_cross_cell_bivector_transports_verified
    assert certificate.all_internal_triangle_bivector_loops_verified
    assert certificate.max_wedge_residual < 5.0e-12
    assert certificate.max_cross_cell_transport_residual < 5.0e-12
    assert certificate.max_internal_triangle_loop_residual < 5.0e-12
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_CLASSICAL_ORIENTED_BIVECTORS_CLOSED'
    )


def test_exact_matrix_component_area_and_antisymmetry_identities() -> None:
    certificate = certify_lorentzian_one_to_five_bivectors()

    for record in certificate.wedge_data:
        assert record.sigma_area_squared_exact > 0
        assert record.exact_area_identity_holds
        for i in range(4):
            assert record.sigma_exact[i][i] == 0
            for j in range(4):
                assert record.sigma_exact[i][j] == -record.sigma_exact[j][i]
                assert record.canonical_b0_exact[i][j] == (
                    -record.canonical_b0_exact[j][i]
                )
        assert record.cell_orientation_sign in (-1, 1)
        assert record.reverse_label_antisymmetry_holds


def test_rest_frame_orientation_is_left_equals_minus_right() -> None:
    certificate = certify_lorentzian_one_to_five_bivectors()

    for record in certificate.wedge_data:
        assert np.allclose(
            record.normal_plane_unit_bivector,
            record.cell_oriented_unit_bivector,
            rtol=0.0,
            atol=5.0e-12,
        )
        assert record.left_rest_route_residual < 5.0e-12
        assert record.right_rest_route_residual < 5.0e-12
        assert record.linear_simplicity_residual < 5.0e-12
        assert record.signed_orientation_residual < 5.0e-12


def test_existing_local_section_has_twenty_six_full_shape_counterexamples() -> None:
    certificate = certify_lorentzian_one_to_five_bivectors()

    assert certificate.full_labeled_triangle_shape_gluing_successes == 24
    assert certificate.full_labeled_triangle_shape_gluing_failures == 26
    assert certificate.minimum_failed_second_edge_mismatch > 0.6
    assert certificate.maximum_second_edge_mismatch == pytest.approx(2.0)

    witness = next(
        record
        for record in certificate.wedge_data
        if record.cell == (5, 1, 2, 3, 4)
        and record.omitted_left == 1
        and record.omitted_right == 5
    )
    assert witness.triangle == (2, 3, 4)
    assert witness.second_shared_edge_shape_mismatch_residual == pytest.approx(
        1.52, rel=0.02
    )


def test_scale_free_bivector_routes_survive_extreme_positive_scales() -> None:
    unit = certify_lorentzian_one_to_five_bivectors()
    tiny = certify_lorentzian_one_to_five_bivectors(
        scale=Fraction(1, 10**500)
    )
    huge = certify_lorentzian_one_to_five_bivectors(
        scale=Fraction(10**500)
    )

    for reference, reduced, enlarged in zip(
        unit.wedge_data, tiny.wedge_data, huge.wedge_data
    ):
        assert reference.triangle == reduced.triangle == enlarged.triangle
        assert reference.cell_orientation_sign == (
            reduced.cell_orientation_sign
        ) == enlarged.cell_orientation_sign
        assert np.allclose(
            reference.cell_oriented_unit_bivector,
            reduced.cell_oriented_unit_bivector,
            rtol=0.0,
            atol=5.0e-12,
        )
        assert np.allclose(
            reference.cell_oriented_unit_bivector,
            enlarged.cell_oriented_unit_bivector,
            rtol=0.0,
            atol=5.0e-12,
        )


def test_invalid_inputs_and_quantum_claim_ceiling() -> None:
    with pytest.raises(ValueError, match='scale cannot'):
        certify_lorentzian_one_to_five_bivectors(
            coordinates={}, scale=Fraction(2)
        )
    with pytest.raises(ValueError, match='tolerance'):
        certify_lorentzian_one_to_five_bivectors(tolerance=0.0)

    certificate = certify_lorentzian_one_to_five_bivectors()
    assert certificate.classical_oriented_bivector_geometry_constructed
    assert not certificate.old_local_phase_is_regge_phase
    assert not certificate.global_regge_spinor_phase_constructed
    assert not certificate.full_eprl_critical_orientation_equation_verified
    assert not certificate.global_eprl_state_constructed
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('CLASSICAL_ORIENTED_BIVECTORS_ONLY')
