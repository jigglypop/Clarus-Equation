from __future__ import annotations

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_eprl_scaffold import (
    certify_lorentzian_one_to_five_eprl_scaffold,
)


def test_all_fifty_y_gamma_samples_and_projectors_materialize() -> None:
    certificate = certify_lorentzian_one_to_five_eprl_scaffold()

    assert certificate.level == 3
    assert certificate.gamma == pytest.approx(0.274)
    assert certificate.face_record_count == 50
    assert certificate.all_beta_diagnostics_nondegenerate
    assert certificate.all_projector_generators_nontrivial
    assert certificate.all_fifty_y_gamma_samples_materialized
    assert certificate.all_fifty_proper_projector_matrices_materialized
    assert certificate.max_projector_residual < 1.0e-11
    assert certificate.max_y_gamma_sample_residual < 1.0e-11
    for record in certificate.face_records:
        assert record.beta in (-1, 1)
        assert record.spin >= 1
        assert record.projector.shape == (2 * record.spin + 1,) * 2
        assert record.projector_rank == record.spin
        assert record.projector_zero_eigenvalue_count == 1
        assert np.allclose(
            record.projector @ record.projector,
            record.projector,
            atol=1.0e-11,
            rtol=0.0,
        )


def test_current_incidence_normals_are_strictly_in_opposite_sector() -> None:
    certificate = certify_lorentzian_one_to_five_eprl_scaffold()

    assert certificate.all_chosen_incidence_sector_scalars_negative
    assert certificate.minimum_absolute_sector_scalar > 0.002
    assert certificate.all_chosen_coherent_states_removed_by_positive_projector
    assert certificate.all_j_dual_coherent_states_preserved_by_positive_projector
    assert certificate.all_chosen_spinor_bivectors_match_classical_branch
    assert certificate.all_antipodal_spinor_bivectors_match_globally_negated_branch
    for record in certificate.face_records:
        assert record.sector_scalar_q < 0.0
        assert record.chosen_coherent_projected_norm < 1.0e-11
        assert record.j_dual_coherent_projected_norm == pytest.approx(
            1.0, abs=1.0e-11
        )
        assert record.chosen_simple_bivector_residual < 1.0e-11
        assert record.antipodal_negated_bivector_residual < 1.0e-11


def test_cartan_dual_is_inequivalent_projector_positive_reconstruction_candidate() -> None:
    certificate = certify_lorentzian_one_to_five_eprl_scaffold()

    assert certificate.all_cartan_dual_frames_proper_orthochronous
    assert certificate.all_cartan_dual_beta_signs_match_original
    assert certificate.all_cartan_dual_orientation_equations_verified
    assert certificate.all_original_and_cartan_dual_critical_equations_verified
    assert certificate.all_cartan_dual_bivectors_match_parity_transform
    assert certificate.all_cartan_dual_sector_scalars_positive
    assert certificate.all_cartan_dual_projectors_preserve_chosen_coherent_states
    assert certificate.all_five_cartan_dual_solutions_inequivalent_to_original
    assert min(certificate.cartan_dual_solution_inequivalence_residuals) > 1.0e-5
    for record in certificate.face_records:
        assert record.cartan_dual_beta == record.beta
        assert record.cartan_dual_sector_scalar_q > 0.0
        assert record.cartan_dual_orientation_equation_residual < 1.0e-11
        assert record.cartan_dual_critical_equation_residual < 1.0e-11
        assert record.cartan_dual_parity_bivector_residual < 1.0e-11
        assert record.cartan_dual_projector_rank == record.spin
        assert record.cartan_dual_coherent_projected_norm == pytest.approx(
            1.0, abs=1.0e-11
        )


def test_independent_continuum_mu_omega_gate_selects_cartan_dual_eh_branch() -> None:
    certificate = certify_lorentzian_one_to_five_eprl_scaffold()

    assert certificate.independent_mu_omega_einstein_hilbert_gate_verified
    assert len(certificate.continuum_sector_audits) == 5
    for audit in certificate.continuum_sector_audits:
        assert audit.coordinate_face_matrix_rank == 6
        assert audit.original_continuum_reconstruction_residual < 1.0e-11
        assert audit.cartan_dual_continuum_reconstruction_residual < 1.0e-11
        assert audit.original_hodge_tetrad_residual < 1.0e-11
        assert audit.cartan_dual_parity_hodge_tetrad_residual < 1.0e-11
        assert audit.original_dynamical_orientation_scalar == pytest.approx(-96.0)
        assert audit.cartan_dual_dynamical_orientation_scalar == pytest.approx(96.0)
        assert audit.original_omega == -1
        assert audit.cartan_dual_omega == 1
        assert audit.original_plebanski_sector_nu == 1
        assert audit.cartan_dual_plebanski_sector_nu == 1
        assert audit.original_mu == -1
        assert audit.cartan_dual_mu == 1


def test_scaffold_ceiling_stops_before_proper_boundary_or_integral() -> None:
    with pytest.raises(ValueError, match='level'):
        certify_lorentzian_one_to_five_eprl_scaffold(level=0)
    with pytest.raises(ValueError, match='gamma'):
        certify_lorentzian_one_to_five_eprl_scaffold(gamma=np.inf)

    certificate = certify_lorentzian_one_to_five_eprl_scaffold()
    assert certificate.local_y_gamma_evaluation_scaffold_constructed
    assert certificate.local_proper_projector_scaffold_constructed
    assert not certificate.chosen_boundary_data_in_positive_einstein_hilbert_sector
    assert certificate.cartan_dual_boundary_data_in_positive_einstein_hilbert_sector
    assert certificate.globally_negated_bivector_boundary_candidate_constructed
    assert not certificate.proper_sector_boundary_state_constructed
    assert not certificate.full_principal_series_representations_materialized
    assert not certificate.gauge_fixed_single_vertex_integral_evaluated
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.parity_related_projector_positive_reconstruction_candidate_constructed
    assert certificate.independent_mu_omega_einstein_hilbert_gate_verified
    assert certificate.status.endswith('EINSTEIN_HILBERT_BRANCH_CLOSED')
    assert certificate.claim_ceiling.endswith('PROJECTOR_SCAFFOLD_ONLY')
