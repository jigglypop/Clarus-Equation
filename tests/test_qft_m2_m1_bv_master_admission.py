from dataclasses import replace

import pytest

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
)
from examples.physics.qft_m2_m1_bv_master_admission import (
    CONTRACT_SHA256,
    bv_antibracket,
    bv_left_derivative,
    bv_right_derivative,
    contract_payload_sha256,
    evaluate_m1_bv_master_admission_gate,
    finite_bv_toy_algebra,
    m1_bv_master_admission_contract,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_bv_master_admission_gate()


def test_contract_source_ledger_hash_and_claims_fail_closed() -> None:
    contract = m1_bv_master_admission_contract()
    validate_contract(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'source_relation': 'literal M1 BV source transcription'},
        {'antibracket_convention': 'ungraded Poisson bracket'},
        {'base_field_specs': contract.base_field_specs[:-1]},
        {'antifield_specs': contract.antifield_specs[:-1]},
        {'antifield_ledger_constructed': False},
        {'finite_canonical_antibracket_calibrated': False},
        {'finite_toy_classical_master_equation_computed': False},
        {'formal_m1_master_residual_decomposition_admitted': False},
        {'full_m1_antifield_functional_constructed': True},
        {'jet_antifield_variational_calculus_constructed': True},
        {'local_functional_boundary_quotient_constructed': True},
        {'boundary_completion_proved': True},
        {'full_m1_classical_master_equation_computed': True},
        {'functional_measure_computed': True},
        {'quantum_master_equation_computed': True},
        {'continuum_loop_st_computed': True},
        {'positive_physical_hilbert_proved': True},
        {'quantum_hda_m2_proved': True},
        {'m3_relational_observables_unlocked': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_all_27_fields_have_locked_antifield_quantum_numbers(receipt) -> None:
    contract = m1_bv_master_admission_contract()
    assert len(contract.base_field_specs) == 27
    assert len(contract.antifield_specs) == 27
    for field, antifield in zip(
        contract.base_field_specs,
        contract.antifield_specs,
        strict=True,
    ):
        assert antifield.name == f'{field.name}_star'
        assert antifield.parity == (field.parity + 1) % 2
        assert antifield.ghost_number == -field.ghost_number - 1
        assert antifield.mass_dimension + field.mass_dimension == 4
        assert antifield.antifield_number == 1
    assert (receipt.base_even_count, receipt.base_odd_count) == (19, 8)
    assert (receipt.antifield_even_count, receipt.antifield_odd_count) == (8, 19)
    assert receipt.base_name_coverage_mismatch_count == 0
    assert receipt.antifield_name_coverage_mismatch_count == 0
    assert receipt.antifield_rule_mismatch_count == 0
    assert receipt.antifield_action_term_type_audit_count == 27
    assert receipt.antifield_action_term_type_mismatch_count == 0
    assert receipt.omitted_field_antifield_pair_rejected
    assert receipt.wrong_antifield_parity_rejected


def test_left_right_derivatives_and_canonical_bracket_are_calibrated(receipt) -> None:
    toy = finite_bv_toy_algebra()
    c = SparseSuperPolynomial.generator('c', odd=True)
    x_star = SparseSuperPolynomial.generator('x_star', odd=True)
    two_odd = c * x_star
    left = bv_left_derivative(two_odd, 'c', odd=True)
    right = bv_right_derivative(two_odd, 'c', odd=True)
    assert left == -right

    x = SparseSuperPolynomial.generator('x', odd=False)
    one = SparseSuperPolynomial.scalar(1)
    assert bv_antibracket(x, x_star, toy.pairs) == one
    assert bv_antibracket(x_star, x, toy.pairs) == -one
    u = SparseSuperPolynomial.generator('u', odd=True)
    v = SparseSuperPolynomial.generator('v', odd=True)
    barc = SparseSuperPolynomial.generator('barc', odd=True)
    b = SparseSuperPolynomial.generator('B', odd=False)
    assert bv_antibracket(toy.action, u, toy.pairs) == u * v
    assert bv_antibracket(toy.action, barc, toy.pairs) == b
    assert toy.transformation_mismatch.is_zero
    assert receipt.finite_toy_field_star_calibration_residual_term_count == 0
    assert receipt.finite_toy_star_field_calibration_residual_term_count == 0
    assert receipt.finite_toy_transformation_mismatch_term_count == 0


def test_nontrivial_finite_bv_toy_solves_the_cme(receipt) -> None:
    toy = finite_bv_toy_algebra()
    assert len(toy.pairs) == 7
    assert toy.action.term_count > 0
    assert toy.master_residual.is_zero
    assert receipt.finite_toy_canonical_pair_count == 7
    assert receipt.finite_toy_master_residual_term_count == 0
    assert receipt.finite_toy_classical_master_equation_computed


def test_bad_action_broken_doublet_and_wrong_sign_controls_are_live(receipt) -> None:
    toy = finite_bv_toy_algebra()
    assert toy.bad_action_master_residual.term_count > 0
    assert toy.broken_doublet_master_residual.term_count > 0
    assert toy.wrong_antibracket_sign_residual.term_count > 0
    assert toy.wrong_odd_antifield_sign_transformation_mismatch.term_count > 0
    assert receipt.bad_action_master_residual_term_count == 1
    assert receipt.broken_doublet_master_residual_term_count == 1
    assert receipt.wrong_antibracket_sign_residual_term_count == 1
    assert (
        receipt.wrong_odd_antifield_sign_transformation_mismatch_term_count
        == 2
    )


def test_formal_afn_split_uses_all_classical_inputs_without_overclaim(receipt) -> None:
    assert receipt.upstream_contracts_verified
    assert receipt.upstream_base_nilpotency_component_count == 27
    assert receipt.upstream_base_nilpotency_maximum_residual_term_count == 0
    assert receipt.upstream_bulk_type_naturality_term_count == 6
    assert receipt.upstream_bulk_type_naturality_maximum_residual == 0
    assert receipt.upstream_boundary_flux_retained
    assert not receipt.upstream_full_coordinate_jet_action_variation_computed
    assert receipt.formal_afn0_input_status == (
        'six_bulk_type_naturality_residuals_zero_boundary_flux_retained'
    )
    assert receipt.formal_afn1_input_status == (
        'twenty_seven_base_nilpotency_residuals_zero'
    )
    assert receipt.formal_master_residual_input_count == 33
    assert receipt.formal_m1_master_residual_decomposition_admitted


def test_scope_is_antifield_admission_not_full_m1_cme_qme_or_m2(receipt) -> None:
    assert receipt.primary_source == 'hep-th/0506098'
    assert receipt.secondary_source == 'arXiv:2206.00780v2'
    assert 'not a literal source transcription' in receipt.source_relation
    assert 'sF=(S,F)' in receipt.antibracket_convention
    assert 'standard left graded Leibniz rule' in receipt.antibracket_convention
    assert receipt.antifield_ledger_constructed
    assert receipt.finite_canonical_antibracket_calibrated
    assert not receipt.full_m1_antifield_functional_constructed
    assert not receipt.jet_antifield_variational_calculus_constructed
    assert not receipt.local_functional_boundary_quotient_constructed
    assert not receipt.boundary_completion_proved
    assert not receipt.full_m1_classical_master_equation_computed
    assert not receipt.functional_measure_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.continuum_loop_st_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.derivation_status == (
        'exact_antifield_ledger_and_standard_left_finite_bv_toy_m1_cme_incomplete'
    )
    assert receipt.declared_m1_bv_master_admission_gate_passed
