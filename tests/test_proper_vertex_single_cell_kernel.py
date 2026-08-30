from __future__ import annotations

import numpy as np
import pytest

from examples.physics.proper_vertex_single_cell_kernel import (
    certify_proper_vertex_single_cell_kernel,
    spin_j_anti_linear_dual,
)


def test_single_cell_ls_boundary_tensor_and_face_contract_counts() -> None:
    certificate = certify_proper_vertex_single_cell_kernel()

    assert certificate.tetrahedron_count == 5
    assert certificate.unoriented_face_count == 10
    assert certificate.directed_incidence_count == 20
    assert certificate.all_five_ls_intertwiners_nonzero_and_normalized
    assert certificate.all_ten_face_spins_match_at_endpoints
    assert certificate.single_cell_ls_boundary_tensor_constructed
    assert not certificate.finite_level_spin_weighted_closure_exact
    assert max(
        item.spin_weighted_closure_defect
        for item in certificate.ls_intertwiners
    ) > 0.0


def test_spin_j_dual_has_integer_spin_square_plus_one() -> None:
    for spin in (1, 2, 3, 5):
        state = np.arange(2 * spin + 1) + 1.0j * np.arange(2 * spin + 1)[::-1]
        assert np.allclose(
            spin_j_anti_linear_dual(
                spin_j_anti_linear_dual(state, spin), spin
            ),
            state,
            atol=1.0e-12,
            rtol=0.0,
        )
    with pytest.raises(ValueError, match='spin-j'):
        spin_j_anti_linear_dual(np.ones(2), 1)


def test_eq53_target_projectors_preserve_cartan_dual_critical_kets() -> None:
    certificate = certify_proper_vertex_single_cell_kernel()

    assert certificate.all_ten_cartan_dual_projectors_preserve_eq53_target_kets
    assert certificate.all_ten_cartan_dual_sector_scalars_positive
    assert certificate.all_ten_critical_spinor_equations_verified
    for face in certificate.face_kernel_contracts:
        assert face.projector_endpoint.endswith('Eq_53')
        assert face.endpoint_spin_match
        assert face.source_endpoint_spin == face.target_endpoint_spin == face.spin
        assert face.spin_space_dimension == 2 * face.spin + 1
        assert face.source_coherent_state.shape == (2 * face.spin + 1,)
        assert face.target_coherent_state.shape == (2 * face.spin + 1,)
        assert face.principal_series_p == pytest.approx(
            certificate.gamma * face.spin
        )
        assert face.sector_scalar_q > 0.0
        assert face.target_projected_norm == pytest.approx(1.0, abs=1.0e-11)
        assert face.target_projector_residual < 1.0e-11
        assert face.critical_spinor_equation_residual < 1.0e-11
        assert face.epsilon_j_square_sign == 1


def test_root_gauge_fixing_and_common_left_invariance() -> None:
    certificate = certify_proper_vertex_single_cell_kernel()
    critical = certificate.critical_point

    assert certificate.integration_group_count_before_gauge_fixing == 5
    assert certificate.integration_group_count_after_gauge_fixing == 4
    assert certificate.integration_real_dimension_after_gauge_fixing == 24
    assert certificate.gauge_fixed_cartan_dual_critical_point_constructed
    assert critical.root_identity_residual < 1.0e-11
    assert critical.relative_element_gauge_invariance_residual < 1.0e-11
    assert critical.common_left_projector_invariance_residual < 1.0e-11
    assert critical.common_left_beta_invariant


def test_kernel_contract_ceiling_stops_before_integrand_and_haar_integral() -> None:
    with pytest.raises(ValueError, match='cell_index'):
        certify_proper_vertex_single_cell_kernel(cell_index=5)

    certificate = certify_proper_vertex_single_cell_kernel()
    assert certificate.proper_face_kernel_types_and_endpoint_policy_defined
    assert not certificate.physical_regge_state_phase_constructed
    assert not certificate.full_principal_series_matrix_coefficients_materialized
    assert not certificate.pointwise_proper_vertex_integrand_evaluated
    assert not certificate.noncompact_haar_integral_evaluated
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.status.endswith('CRITICAL_POINT_CLOSED')
    assert certificate.claim_ceiling.endswith('CRITICAL_POINT_ONLY')
