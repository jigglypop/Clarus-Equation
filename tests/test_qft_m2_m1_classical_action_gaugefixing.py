from dataclasses import replace

import pytest

from examples.physics.qft_m2_m1_classical_action_gaugefixing import (
    BULK_TERM_SPECS,
    CONTRACT_SHA256,
    bad_coordinate_density_algebra,
    bulk_density_residuals,
    bulk_dimension_residuals,
    contract_payload_sha256,
    density_weight_residual_coefficient,
    evaluate_m1_classical_action_gaugefixing_gate,
    gauge_fermion_algebra,
    gauge_quantum_number_audit,
    m1_classical_action_gaugefixing_contract,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_classical_action_gaugefixing_gate()


def test_contract_source_fields_hash_and_claims_fail_closed() -> None:
    contract = m1_classical_action_gaugefixing_contract()
    validate_contract(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'source_relation': 'literal source transcription'},
        {'source_contains_m1_chi_plus_four_x': True},
        {'matter_scalar_labels': ()},
        {'reference_scalar_labels': contract.reference_scalar_labels[:-1]},
        {'bulk_term_specs': contract.bulk_term_specs[:-1]},
        {'bulk_density_type_naturality_certificate_computed': False},
        {'full_coordinate_jet_action_variation_computed': True},
        {'boundary_flux_retained': False},
        {'boundary_discarded': True},
        {'ghy_boundary_variation_computed': True},
        {'integrated_action_invariance_proved': True},
        {'m1_gauge_condition_derived': True},
        {'bv_antifields_constructed': True},
        {'classical_master_equation_computed': True},
        {'quantum_master_equation_computed': True},
        {'functional_measure_computed': True},
        {'loop_st_anomaly_cancellation_computed': True},
        {'positive_physical_hilbert_proved': True},
        {'quantum_hda_m2_proved': True},
        {'m3_relational_observables_unlocked': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_full_m1_bulk_terms_are_dimension_four_weight_one(receipt) -> None:
    assert len(BULK_TERM_SPECS) == 6
    assert bulk_dimension_residuals() == tuple(
        (name, 0) for name, *_ in BULK_TERM_SPECS
    )
    broken_dimension_specs = (
        ('broken', 2, 1, 4, 1),
    )
    assert bulk_dimension_residuals(broken_dimension_specs) == (
        ('broken', -1),
    )
    assert bulk_density_residuals() == tuple(
        (name, 0) for name, *_ in BULK_TERM_SPECS
    )
    assert receipt.bulk_term_count == 6
    assert receipt.maximum_bulk_dimension_residual == 0
    assert receipt.maximum_bulk_density_weight_residual == 0
    assert receipt.bulk_boundary_current_component_count == 24
    assert receipt.bulk_density_type_naturality_certificate_computed
    assert not receipt.full_coordinate_jet_action_variation_computed


def test_density_weight_identity_and_measure_controls(receipt) -> None:
    assert density_weight_residual_coefficient(1) == 0
    assert density_weight_residual_coefficient(0) == -1
    assert density_weight_residual_coefficient(2) == 1
    assert receipt.dropped_sqrt_g_density_residual_term_count == 6
    assert receipt.missing_reference_scale_dimension_residual == -2
    assert receipt.omitted_bulk_term_rejected
    assert receipt.boundary_flux_retained
    assert not receipt.boundary_discarded
    assert not receipt.ghy_boundary_variation_computed
    assert not receipt.integrated_action_invariance_proved


def test_nilpotent_bad_coordinate_density_is_not_a_divergence(receipt) -> None:
    algebra = bad_coordinate_density_algebra()
    assert algebra.variation == algebra.locked_variation
    assert algebra.second_variation.is_zero
    assert algebra.euler_chi_residual == algebra.locked_euler_chi_residual
    assert algebra.euler_chi_residual.term_count == 4
    assert receipt.bad_coordinate_density_mass_dimension == 4
    assert receipt.bad_coordinate_density_weight == 0
    assert receipt.bad_density_locked_variation_mismatch_term_count == 0
    assert receipt.bad_density_second_variation_term_count == 0
    assert receipt.bad_density_euler_chi_residual_term_count == 4
    assert receipt.bad_density_not_horizontal_divergence
    assert receipt.nilpotency_does_not_imply_action_invariance


def test_gauge_fermion_expands_exactly_and_is_nilpotent(receipt) -> None:
    algebra = gauge_fermion_algebra()
    assert len(algebra.fermion_components) == 4
    assert all(
        expanded == target
        for expanded, target in zip(
            algebra.expanded_components,
            algebra.locked_target_components,
        )
    )
    assert all(item.is_zero for item in algebra.second_variation_components)
    assert receipt.gauge_fixing_locked_mismatch_term_count == 0
    assert receipt.gauge_fixing_second_variation_term_count == 0
    assert receipt.gauge_fixing_fermion_constructed
    assert receipt.gauge_fixing_brst_exactness_computed


def test_gauge_quantum_numbers_and_negative_controls_are_live(receipt) -> None:
    dimensions_ok, ghost_numbers_ok, count = gauge_quantum_number_audit(
        gauge_fermion_algebra()
    )
    assert dimensions_ok
    assert ghost_numbers_ok
    assert count == 8
    assert receipt.gauge_quantum_number_audit_map_count == 8
    assert receipt.wrong_gauge_ghost_sign_mismatch_term_count > 0
    assert receipt.omitted_auxiliary_square_mismatch_term_count > 0
    assert receipt.commuting_gauge_ghost_mismatch_term_count > 0
    assert receipt.broken_gauge_doublet_second_variation_term_count > 0


def test_scope_is_local_classical_not_bv_qme_or_m2(receipt) -> None:
    assert receipt.primary_source == 'arXiv:2206.00780v2'
    assert 'not literal source transcriptions' in receipt.source_relation
    assert not receipt.source_contains_m1_chi_plus_four_x
    assert receipt.upstream_classical_brst_gate_passed
    assert receipt.upstream_base_nilpotency_component_count == 27
    assert not receipt.m1_gauge_condition_derived
    assert not receipt.gauge_fixed_density_diffeomorphism_covariance_proved
    assert not receipt.bv_antifields_constructed
    assert not receipt.classical_master_equation_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.functional_measure_computed
    assert not receipt.loop_st_anomaly_cancellation_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.declared_m1_classical_action_gaugefixing_gate_passed
    assert receipt.derivation_status == (
        'exact_bulk_type_naturality_mod_dh_and_abstract_gauge_fermion_only'
    )
