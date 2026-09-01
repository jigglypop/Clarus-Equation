from dataclasses import replace

import pytest

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    CONTRACT_SHA256,
    MATTER_SCALAR_LABELS,
    REFERENCE_SCALAR_LABELS,
    SCALAR_LABELS,
    SparseSuperPolynomial,
    build_jet_differential,
    contract_payload_sha256,
    evaluate_four_scalar_classical_brst_jet_gate,
    four_scalar_classical_brst_jet_contract,
    nilpotency_residuals,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_four_scalar_classical_brst_jet_gate()


def test_sparse_super_polynomial_has_exact_exterior_signs() -> None:
    a = SparseSuperPolynomial.generator('a', odd=True)
    b = SparseSuperPolynomial.generator('b', odd=True)
    x = SparseSuperPolynomial.generator('x', odd=False)
    assert (a * a).is_zero
    assert a * b == -(b * a)
    assert a * x == x * a
    assert (a * b).term_count == 1


def test_all_declared_base_transformations_are_exactly_nilpotent() -> None:
    differential = build_jet_differential()
    residuals = nilpotency_residuals(differential)
    assert len(residuals) == 27
    assert all(polynomial.is_zero for _, polynomial in residuals)
    assert len(differential.generator_names) == 293
    assert len(differential.odd_generator_names) == 64


def test_full_m1_scalar_content_and_nontrivial_first_transformations(
    receipt,
) -> None:
    counts = dict(receipt.first_transformation_term_counts)
    assert receipt.matter_scalar_labels == MATTER_SCALAR_LABELS
    assert receipt.reference_scalar_labels == REFERENCE_SCALAR_LABELS
    assert receipt.scalar_labels == SCALAR_LABELS
    assert receipt.matter_scalar_field_count == 1
    assert receipt.reference_scalar_field_count == 4
    assert receipt.scalar_field_count == 5
    assert MATTER_SCALAR_LABELS == ('chi',)
    assert REFERENCE_SCALAR_LABELS == ('X0', 'X1', 'X2', 'X3')
    assert all(counts[label] == 4 for label in SCALAR_LABELS)
    assert all(counts[f'c{mu}'] == 4 for mu in range(4))
    assert all(counts[f'barc{mu}'] == 1 for mu in range(4))
    assert all(counts[f'B{mu}'] == 0 for mu in range(4))
    assert receipt.required_first_transformations_nonzero
    assert receipt.auxiliary_transformations_zero
    assert receipt.metric_symmetry_preserved
    assert receipt.scalar_multiplet_preserved
    assert receipt.locked_base_transformation_mismatch_term_count == 0


def test_dimensions_ghost_numbers_and_exact_receipt(receipt) -> None:
    assert receipt.maximum_nilpotency_residual_term_count == 0
    assert receipt.dimension_audit_map_count == 95
    assert receipt.ghost_number_audit_map_count == 103
    assert receipt.all_geometric_map_dimensions_correct
    assert receipt.all_map_ghost_numbers_correct
    assert receipt.declared_exact_classical_brst_jet_gate_passed


def test_each_independent_negative_control_is_live(receipt) -> None:
    assert receipt.wrong_ghost_sign_residual_term_count > 0
    assert receipt.commuting_ghost_residual_term_count > 0
    assert receipt.ungraded_leibniz_residual_term_count > 0
    assert receipt.unsymmetrized_second_jet_residual_term_count > 0
    assert receipt.missing_scalar_transport_mismatch_term_count > 0
    assert receipt.missing_metric_lie_slot_mismatch_term_count > 0
    assert receipt.broken_doublet_residual_term_count > 0
    assert receipt.wrong_reference_scalar_multiplicity_rejected
    assert receipt.missing_matter_scalar_rejected
    assert receipt.wrong_scalar_multiplicity_rejected


def test_contract_and_claim_status_fail_closed() -> None:
    contract = four_scalar_classical_brst_jet_contract()
    validate_contract(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'source_relation': 'literal source transcription'},
        {'source_contains_four_reference_scalars': True},
        {'sign_convention': contract.sign_convention.replace('s c^mu=', 's c^mu=-')},
        {
            'reference_scalar_labels': contract.reference_scalar_labels[:-1],
            'scalar_labels': (
                contract.matter_scalar_labels
                + contract.reference_scalar_labels[:-1]
            ),
        },
        {
            'matter_scalar_labels': (),
            'scalar_labels': contract.reference_scalar_labels,
        },
        {'spacetime_dimension': 3},
        {'maximum_jet_order': 1},
        {'classical_brst_nilpotency_computed': False},
        {'second_jet_generator_images_defined': True},
        {'action_density_invariance_computed': True},
        {'gauge_fixing_fermion_constructed': True},
        {'bv_antifields_constructed': True},
        {'classical_master_equation_computed': True},
        {'quantum_master_equation_computed': True},
        {'loop_anomaly_cancellation_computed': True},
        {'positive_physical_hilbert_proved': True},
        {'quantum_hda_m2_proved': True},
        {'relational_observable_interpretation_proved': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_scope_is_classical_nilpotency_not_bv_or_m2(receipt) -> None:
    assert receipt.primary_source == 'arXiv:2206.00780v2'
    assert receipt.primary_source_date == '2025-06-01'
    assert 'not literal transcription' in receipt.source_relation
    assert not receipt.source_contains_four_reference_scalars
    assert not receipt.action_density_invariance_computed
    assert not receipt.gauge_fixing_fermion_constructed
    assert not receipt.bv_antifields_constructed
    assert not receipt.classical_master_equation_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.loop_anomaly_cancellation_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.relational_observable_interpretation_proved
    assert not receipt.second_jet_generator_images_defined
    assert receipt.derivation_status == (
        'exact_27_base_component_second_jet_classical_brst_nilpotency_only'
    )
