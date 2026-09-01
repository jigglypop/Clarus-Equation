from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
)
from examples.physics.qft_m2_m1_reparam_local_bv_quotient import (
    ALL_VARIABLE_SPECS,
    CONTRACT_SHA256,
    JET_LOOKUP,
    MAXIMUM_JET_ORDER,
    JetOrderExceeded,
    all_euler_residuals,
    antifield_number_components,
    apply_local_brst,
    canonical_contract_payload,
    classical_noether_identity,
    contract_payload_sha256,
    derived_transformation_mismatch,
    euler_derivative,
    evaluate_m1_reparam_local_bv_quotient_gate,
    generator,
    graded_antisymmetry_residual,
    graded_jacobi_residual,
    jet_partial_derivative,
    local_antifield_specs,
    local_bv_antibracket_density,
    local_field_specs,
    locked_master_boundary_current,
    m1_reparam_local_bv_quotient_contract,
    master_density,
    reparam_local_bv_model,
    total_derivative,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_reparam_local_bv_quotient_gate()


def test_contract_hash_sources_basis_and_claims_fail_closed() -> None:
    contract = m1_reparam_local_bv_quotient_contract()
    validate_contract(contract)
    assert canonical_contract_payload(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'local_functional_source': 'unlocked'},
        {'secondary_source': 'unlocked'},
        {'model_relation': 'literal 4D M1 source action'},
        {'normalization': 'dimensionful unspecified variables'},
        {'scalar_labels': contract.scalar_labels[:-1]},
        {'maximum_jet_order': 3},
        {'antibracket_convention': 'mixed right Hamiltonian convention'},
        {'field_specs': contract.field_specs[:-1]},
        {'antifield_specs': contract.antifield_specs[:-1]},
        {'upstream_hashes': (('E70-D', '0' * 64),)},
        {'bounded_jet_euler_calculus_constructed': False},
        {'terminal_jet_rejection_enforced': False},
        {'bounded_local_functional_antibracket_constructed': False},
        {'classical_noether_identity_computed': False},
        {'nonzero_horizontal_currents_retained': False},
        {'reparam_toy_cme_mod_dh_computed': False},
        {'explicit_master_current_constructed': False},
        {'graded_identities_sampled': False},
        {'silent_terminal_truncation_allowed': True},
        {'open_boundary_action_invariance_proved': True},
        {'unbounded_jet_closure_proved': True},
        {'general_local_functional_theorem_proved': True},
        {'four_dimensional_m1_action_used': True},
        {'full_m1_antifield_functional_constructed': True},
        {'full_m1_classical_master_equation_computed': True},
        {'boundary_completion_proved': True},
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


def test_14_pairs_and_140_bounded_jets_are_complete_and_terminal_safe(receipt) -> None:
    fields = local_field_specs()
    antifields = local_antifield_specs()
    assert len(fields) == len(antifields) == 14
    for field, antifield in zip(fields, antifields, strict=True):
        assert antifield.name == f'{field.name}_star'
        assert antifield.parity == (field.parity + 1) % 2
        assert antifield.ghost_number == -field.ghost_number - 1
        assert antifield.antifield_number == 1
    assert len(ALL_VARIABLE_SPECS) == 28
    assert len(JET_LOOKUP) == 140
    assert receipt.bounded_even_jet_generator_count == 70
    assert receipt.bounded_odd_jet_generator_count == 70
    with pytest.raises(JetOrderExceeded):
        total_derivative(generator('q_chi', MAXIMUM_JET_ORDER))
    assert receipt.terminal_jet_derivative_rejected


def test_classical_density_has_exact_retained_current_and_noether_identity(receipt) -> None:
    model = reparam_local_bv_model()
    variation = apply_local_brst(model.classical_density, model.transformations)
    current_derivative = total_derivative(model.classical_boundary_current)
    assert variation == current_derivative
    assert not model.classical_boundary_current.is_zero
    assert classical_noether_identity(model).is_zero
    assert receipt.classical_density_term_count == 15
    assert receipt.classical_boundary_current_term_count == 15
    assert receipt.classical_variation_term_count == 45
    assert receipt.classical_current_derivative_term_count == 45
    assert receipt.classical_density_identity_mismatch_term_count == 0
    assert receipt.classical_noether_identity_residual_term_count == 0


def test_left_brst_is_nilpotent_and_generated_by_the_extended_functional(receipt) -> None:
    model = reparam_local_bv_model()
    assert all(
        apply_local_brst(image, model.transformations).is_zero
        for image in model.transformations.values()
    )
    assert derived_transformation_mismatch(model).is_zero
    assert receipt.base_nilpotency_component_count == 14
    assert receipt.base_nilpotency_maximum_residual_term_count == 0
    assert receipt.derived_transformation_component_count == 14
    assert receipt.derived_transformation_mismatch_term_count == 0


def test_local_antibracket_signs_antisymmetry_and_jacobi_are_live(receipt) -> None:
    q = generator('q_chi')
    q1 = generator('q_chi', 1)
    q_star = generator('q_chi_star')
    c = generator('c')
    c1 = generator('c', 1)
    c_star = generator('c_star')
    one = SparseSuperPolynomial.scalar(1)
    assert local_bv_antibracket_density(q, q_star) == one
    assert local_bv_antibracket_density(q_star, q) == -one
    two_odd = c * q_star
    left = jet_partial_derivative(two_odd, 'c', 0, side='left')
    right = jet_partial_derivative(two_odd, 'c', 0, side='right')
    assert left == -right

    first = q_star * c * q1
    second = -(c_star * c * c1)
    third = q
    assert graded_antisymmetry_residual(first, second).is_zero
    nested = (
        local_bv_antibracket_density(
            first,
            local_bv_antibracket_density(second, third),
        ),
        local_bv_antibracket_density(
            second,
            local_bv_antibracket_density(third, first),
        ),
        local_bv_antibracket_density(
            third,
            local_bv_antibracket_density(first, second),
        ),
    )
    assert sum(not value.is_zero for value in nested) == 2
    assert graded_jacobi_residual(first, second, third).is_zero
    assert receipt.jacobi_nonzero_nested_bracket_count == 2


def test_master_density_is_nonzero_but_exactly_a_retained_total_derivative(receipt) -> None:
    model = reparam_local_bv_model()
    master = master_density(model)
    current = locked_master_boundary_current(model)
    assert not master.is_zero
    assert not current.is_zero
    assert master == total_derivative(current)
    assert all(residual.is_zero for _, residual in all_euler_residuals(master))
    components = antifield_number_components(master)
    assert components[0].term_count == 30
    assert 1 not in components
    assert receipt.master_density_term_count == 30
    assert receipt.master_boundary_current_term_count == 10
    assert receipt.master_current_derivative_term_count == 30
    assert receipt.master_current_mismatch_term_count == 0
    assert receipt.master_euler_audit_count == 28
    assert receipt.master_euler_maximum_residual_term_count == 0
    assert receipt.master_afn0_term_count == 30
    assert receipt.master_afn1_term_count == 0


def test_independent_action_ghost_antifield_and_bracket_controls_fail(receipt) -> None:
    assert receipt.bad_missing_lapse_identity_mismatch_term_count == 10
    assert receipt.bad_missing_lapse_nonzero_euler_count == 11
    assert receipt.missing_lapse_weight_identity_mismatch_term_count == 10
    assert receipt.missing_lapse_weight_nonzero_euler_count == 12
    assert receipt.wrong_ghost_sign_nonzero_nilpotency_component_count == 11
    assert receipt.wrong_ghost_sign_maximum_nilpotency_residual_term_count == 2
    assert receipt.wrong_ghost_sign_master_nonzero_euler_count == 23
    assert receipt.omitted_ghost_antifield_master_term_count == 41
    assert receipt.omitted_ghost_antifield_master_nonzero_euler_count == 23
    assert receipt.uniform_plus_antifield_transformation_mismatch_term_count == 2
    assert receipt.uniform_plus_antifield_master_nonzero_euler_count == 23
    assert receipt.wrong_antibracket_canonical_residual_term_count == 1
    assert receipt.wrong_antibracket_antisymmetry_residual_term_count == 1
    assert receipt.wrong_antibracket_jacobi_residual_term_count == 1
    assert receipt.naive_partial_vs_euler_difference_term_count == 1


def test_scope_is_dimensionless_bounded_toy_not_four_dimensional_m1_or_qme(receipt) -> None:
    assert receipt.primary_source == 'hep-th/0506098'
    assert receipt.local_functional_source == 'hep-th/0002245v3'
    assert receipt.secondary_source == 'arXiv:2206.00780v2'
    assert 'dimensionless exact coordinates' in receipt.normalization
    assert 'convention-adapted polynomial toy' in receipt.model_relation
    assert receipt.upstream_e70_d_verified
    assert receipt.bounded_jet_euler_calculus_constructed
    assert receipt.bounded_local_functional_antibracket_constructed
    assert receipt.nonzero_horizontal_currents_retained
    assert receipt.reparam_toy_cme_mod_dh_computed
    assert not receipt.silent_terminal_truncation_allowed
    assert not receipt.open_boundary_action_invariance_proved
    assert not receipt.unbounded_jet_closure_proved
    assert not receipt.general_local_functional_theorem_proved
    assert not receipt.four_dimensional_m1_action_used
    assert not receipt.full_m1_antifield_functional_constructed
    assert not receipt.full_m1_classical_master_equation_computed
    assert not receipt.boundary_completion_proved
    assert not receipt.functional_measure_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.continuum_loop_st_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.declared_m1_reparam_local_bv_quotient_gate_passed
