from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
)
from examples.physics.qft_m2_m1_4d_densitized_scalar_bv import (
    ALL_VARIABLE_SPECS,
    CONTRACT_SHA256,
    JET_LOOKUP,
    MAXIMUM_TOTAL_JET_ORDER,
    MULTIINDICES,
    MultiJetOrderExceeded,
    ZERO_MULTIINDEX,
    apply_densitized_brst,
    canonical_contract_payload,
    contract_payload_sha256,
    densitized_antifield_specs,
    densitized_field_specs,
    densitized_scalar_bv_model,
    divergence,
    evaluate_m1_4d_densitized_scalar_bv_gate,
    generator,
    graded_antisymmetry_residual,
    graded_jacobi_residual,
    horizontal_derivative,
    jet_partial_derivative,
    local_bv_antibracket_density,
    m1_4d_densitized_scalar_bv_contract,
    nonperiodic_boundary_fixture,
    scalar_label_permutation_image,
    spacetime_axis_permutation_image,
    unit_multiindex,
    validate_contract,
    variational_homotopy_current,
    variational_homotopy_euler_remainder,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_4d_densitized_scalar_bv_gate()


def test_contract_hash_sources_basis_weights_and_claims_fail_closed() -> None:
    contract = m1_4d_densitized_scalar_bv_contract()
    validate_contract(contract)
    assert canonical_contract_payload(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    bad_weight_field = replace(contract.field_specs[5], density_weight=0)
    bad_weight_fields = (
        contract.field_specs[:5]
        + (bad_weight_field,)
        + contract.field_specs[6:]
    )
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'local_functional_source': 'unlocked'},
        {'diffeomorphism_source': 'unlocked'},
        {'source_boundary': 'literal source model'},
        {'model_relation': 'h=sqrt(-g) g^-1 and rho=sqrt(-g)'},
        {'normalization': 'dimensionful unspecified variables'},
        {'dimension': 3},
        {'scalar_labels': contract.scalar_labels[:-1]},
        {'maximum_total_jet_order': 2},
        {'antibracket_convention': 'right Hamiltonian convention'},
        {'field_specs': contract.field_specs[:-1]},
        {'field_specs': bad_weight_fields},
        {'antifield_specs': contract.antifield_specs[:-1]},
        {'upstream_hashes': (('E70-E', '0' * 64),)},
        {'four_dimensional_densitized_scalar_toy_constructed': False},
        {'exact_multiindex_euler_calculus_constructed': False},
        {'terminal_jet_rejection_enforced': False},
        {'local_functional_antibracket_constructed': False},
        {'density_and_ghost_gradings_computed': False},
        {'classical_density_identity_computed': False},
        {'base_brst_nilpotency_computed': False},
        {'explicit_afn0_noether_current_constructed': False},
        {'explicit_afn1_homotopy_current_constructed': False},
        {'scalar_toy_cme_mod_dh_computed': False},
        {'basis_permutation_covariance_sampled': False},
        {'nonperiodic_boundary_retained': False},
        {'graded_identities_sampled': False},
        {'silent_terminal_truncation_allowed': True},
        {'h_metric_determinant_relation_imposed': True},
        {'rho_metric_determinant_relation_imposed': True},
        {'einstein_hilbert_action_used': True},
        {'full_m1_functional_constructed': True},
        {'global_boundary_completion_proved': True},
        {'unbounded_variational_bicomplex_proved': True},
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


def test_28_pairs_1960_jets_and_density_gradings_are_complete(receipt) -> None:
    fields = densitized_field_specs()
    antifields = densitized_antifield_specs()
    assert len(fields) == len(antifields) == 28
    for field, antifield in zip(fields, antifields, strict=True):
        assert antifield.name == f'{field.name}_star'
        assert antifield.parity == (field.parity + 1) % 2
        assert antifield.ghost_number == -field.ghost_number - 1
        assert antifield.antifield_number == 1
        assert antifield.density_weight == 1 - field.density_weight
    assert len(ALL_VARIABLE_SPECS) == 56
    assert len(MULTIINDICES) == 35
    assert len(JET_LOOKUP) == 1960
    assert receipt.bounded_even_jet_generator_count == 980
    assert receipt.bounded_odd_jet_generator_count == 980
    assert receipt.classical_density_weights == (1,)
    assert receipt.antifield_density_weights == (1,)
    assert receipt.extended_density_ghost_numbers == (0,)
    with pytest.raises(MultiJetOrderExceeded):
        horizontal_derivative(
            generator('phi_chi', (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0)),
            0,
        )
    assert receipt.terminal_jet_derivative_rejected


def test_classical_density_and_left_brst_close_exactly(receipt) -> None:
    model = densitized_scalar_bv_model()
    variation = apply_densitized_brst(
        model.classical_density,
        model.transformations,
    )
    assert variation == divergence(model.classical_boundary_current)
    assert all(
        apply_densitized_brst(image, model.transformations).is_zero
        for image in model.transformations.values()
    )
    assert receipt.potential_term_count == 2
    assert receipt.classical_density_term_count == 52
    assert receipt.antifield_density_term_count == 164
    assert receipt.extended_density_term_count == 216
    assert receipt.classical_variation_term_count == 744
    assert receipt.classical_current_term_count == 208
    assert receipt.classical_current_divergence_term_count == 744
    assert receipt.classical_identity_mismatch_term_count == 0
    assert receipt.base_nilpotency_component_count == 28
    assert receipt.base_nilpotency_nonzero_component_count == 0
    assert receipt.derived_transformation_component_count == 28
    assert receipt.derived_transformation_mismatch_term_count == 0


def test_standard_left_local_bv_signs_and_graded_identities_are_live(receipt) -> None:
    phi = generator('phi_chi')
    phi_one = generator('phi_chi', unit_multiindex(0))
    phi_star = generator('phi_chi_star')
    ghost = generator('c0')
    ghost_one = generator('c0', unit_multiindex(0))
    ghost_star = generator('c0_star')
    one = SparseSuperPolynomial.scalar(1)
    assert local_bv_antibracket_density(phi, phi_star) == one
    assert local_bv_antibracket_density(phi_star, phi) == -one
    two_odd = ghost * phi_star
    left = jet_partial_derivative(
        two_odd, 'c0', ZERO_MULTIINDEX, side='left'
    )
    right = jet_partial_derivative(
        two_odd, 'c0', ZERO_MULTIINDEX, side='right'
    )
    assert left == -right
    first = phi_star * ghost * phi_one
    second = -(ghost_star * ghost * ghost_one)
    assert graded_antisymmetry_residual(first, second).is_zero
    assert graded_jacobi_residual(first, second, phi).is_zero
    assert receipt.canonical_field_star_residual_term_count == 0
    assert receipt.canonical_star_field_residual_term_count == 0
    assert receipt.jacobi_nonzero_nested_bracket_count == 2


def test_master_density_has_explicit_afn0_and_afn1_boundary_currents(receipt) -> None:
    assert receipt.master_density_term_count == 2604
    assert receipt.master_density_maximum_total_jet_order == 2
    assert receipt.master_density_weights == (1,)
    assert receipt.master_density_ghost_numbers == (1,)
    assert receipt.master_afn0_term_count == 984
    assert receipt.master_afn1_term_count == 1620
    assert receipt.analytic_afn0_current_term_count == 388
    assert receipt.analytic_afn0_current_divergence_term_count == 984
    assert receipt.analytic_afn0_mismatch_term_count == 0
    assert receipt.homotopy_afn1_current_term_count == 1752
    assert receipt.homotopy_afn1_current_divergence_term_count == 1620
    assert receipt.homotopy_afn1_remainder_term_count == 0
    assert receipt.homotopy_afn1_identity_mismatch_term_count == 0
    assert receipt.homotopy_afn1_direct_mismatch_term_count == 0
    assert receipt.full_master_current_term_count == 2140
    assert receipt.full_master_current_divergence_term_count == 2604
    assert receipt.full_master_current_mismatch_term_count == 0
    assert receipt.master_euler_audit_count == 56
    assert receipt.master_euler_nonzero_count == 0

    fixture = horizontal_derivative(
        generator('phi_chi') * generator('h00'),
        0,
    )
    current = variational_homotopy_current(fixture)
    assert variational_homotopy_euler_remainder(fixture).is_zero
    assert fixture == divergence(current)


def test_scalar_and_axis_relabelling_and_open_boundary_are_retained(receipt) -> None:
    model = densitized_scalar_bv_model()
    scalar_swap = {
        'chi': 'chi',
        'X0': 'X3',
        'X1': 'X1',
        'X2': 'X2',
        'X3': 'X0',
    }
    assert scalar_label_permutation_image(
        model.extended_density,
        scalar_swap,
    ) == model.extended_density
    assert spacetime_axis_permutation_image(
        model.extended_density,
        (1, 0, 2, 3),
    ) == model.extended_density
    assert receipt.scalar_label_permutation_mismatch_term_count == 0
    assert receipt.spacetime_axis_permutation_mismatch_term_count == 0
    assert receipt.symmetric_h_transpose_name_locked
    assert nonperiodic_boundary_fixture() == (
        Fraction(0),
        Fraction(1),
        Fraction(1),
    )
    assert receipt.nonperiodic_boundary_endpoint_difference == 1


def test_independent_density_action_ghost_antifield_and_bracket_controls_fail(receipt) -> None:
    assert receipt.missing_h_weight_identity_mismatch_term_count == 200
    assert receipt.missing_h_weight_phi_euler_term_count == 168
    assert receipt.missing_rho_weight_identity_mismatch_term_count == 8
    assert receipt.missing_rho_weight_phi_euler_term_count == 8
    assert receipt.missing_second_h_index_identity_mismatch_term_count == 200
    assert receipt.missing_second_h_index_nonzero_nilpotency_component_count == 6
    assert receipt.bad_density_potential_identity_mismatch_term_count == 8
    assert receipt.bad_density_potential_phi_euler_term_count == 8
    assert receipt.wrong_ghost_sign_nonzero_nilpotency_component_count == 16
    assert receipt.wrong_ghost_sign_maximum_nilpotency_residual_term_count == 76
    assert receipt.wrong_ghost_sign_master_phi_euler_term_count == 32
    assert receipt.omitted_ghost_antifield_transformation_mismatch_term_count == 16
    assert receipt.omitted_ghost_antifield_master_phi_euler_term_count == 32
    assert receipt.uniform_plus_antifield_transformation_mismatch_term_count == 20
    assert receipt.uniform_plus_antifield_master_phi_euler_term_count == 32
    assert receipt.wrong_antibracket_canonical_residual_term_count == 1
    assert receipt.wrong_antibracket_antisymmetry_residual_term_count == 1
    assert receipt.wrong_antibracket_jacobi_residual_term_count == 1
    assert receipt.naive_partial_vs_euler_difference_term_count == 26


def test_scope_is_bounded_independent_density_scalar_toy_not_eh_or_quantum(receipt) -> None:
    assert receipt.upstream_e70_e_verified
    assert receipt.dimension == 4
    assert receipt.scalar_count == 5
    assert receipt.symmetric_h_component_count == 10
    assert 'dimensionless exact coordinates' in receipt.normalization
    assert 'independent weight-one' in receipt.model_relation
    assert 'convention-adapted density geometry' in receipt.source_boundary
    assert receipt.four_dimensional_densitized_scalar_toy_constructed
    assert receipt.exact_multiindex_euler_calculus_constructed
    assert receipt.explicit_afn0_noether_current_constructed
    assert receipt.explicit_afn1_homotopy_current_constructed
    assert receipt.scalar_toy_cme_mod_dh_computed
    assert receipt.nonperiodic_boundary_retained
    assert not receipt.silent_terminal_truncation_allowed
    assert not receipt.h_metric_determinant_relation_imposed
    assert not receipt.rho_metric_determinant_relation_imposed
    assert not receipt.einstein_hilbert_action_used
    assert not receipt.full_m1_functional_constructed
    assert not receipt.global_boundary_completion_proved
    assert not receipt.unbounded_variational_bicomplex_proved
    assert not receipt.functional_measure_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.continuum_loop_st_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.declared_m1_4d_densitized_scalar_bv_gate_passed
