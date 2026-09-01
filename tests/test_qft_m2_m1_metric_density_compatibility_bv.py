from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    SparseSuperPolynomial,
)
from examples.physics.qft_m2_m1_metric_density_compatibility_bv import (
    ALL_VARIABLE_SPECS,
    CONTRACT_SHA256,
    JET_LOOKUP,
    MAXIMUM_TOTAL_JET_ORDER,
    MULTIINDICES,
    CompatibilityJetOrderExceeded,
    ZERO_MULTIINDEX,
    adjugate_identity_residuals,
    apply_compatibility_brst,
    canonical_contract_payload,
    compatibility_antifield_specs,
    compatibility_field_specs,
    component_factor_weight_sums,
    contract_payload_sha256,
    density_transformation_target,
    divergence,
    evaluate_m1_metric_density_compatibility_bv_gate,
    generator,
    graded_antisymmetry_residual,
    graded_jacobi_residual,
    horizontal_derivative,
    jet_partial_derivative,
    local_bv_antibracket_density,
    m1_metric_density_compatibility_bv_contract,
    metric_density_compatibility_bv_model,
    reconstruct_metric_patch,
    unit_multiindex,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_metric_density_compatibility_bv_gate()


def test_contract_hash_sources_basis_and_claim_ceiling_fail_closed() -> None:
    contract = m1_metric_density_compatibility_bv_contract()
    validate_contract(contract)
    assert canonical_contract_payload(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    bad_ell = replace(contract.field_specs[-1], density_weight=0)
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'density_source': 'unlocked'},
        {'metric_density_precedent': 'literal 4D source'},
        {'parametrization_source': 'literal determinant source'},
        {'source_boundary': 'sources literally prove this full 4D model'},
        {'normalization': 'dimensionful unspecified variables'},
        {'compatibility_relation': 'det(h) has weight four'},
        {'dimension': 3},
        {'scalar_labels': contract.scalar_labels[:-1]},
        {'maximum_total_jet_order': 2},
        {'antibracket_convention': 'right Hamiltonian convention'},
        {'field_specs': contract.field_specs[:-1]},
        {'field_specs': contract.field_specs[:-1] + (bad_ell,)},
        {'antifield_specs': contract.antifield_specs[:-1]},
        {'upstream_hashes': (('E70-F', '0' * 64),)},
        {'determinant_polynomial_constructed': False},
        {'adjugate_identity_computed': False},
        {'determinant_weight_two_covariance_computed': False},
        {'compatibility_ideal_brst_stable': False},
        {'weight_minus_one_multiplier_constructed': False},
        {'conditional_positive_rho_metric_reconstruction_computed': False},
        {'bounded_local_bv_quotient_constructed': False},
        {'explicit_afn0_and_afn1_currents_constructed': False},
        {'compatibility_cme_mod_dh_computed': False},
        {'live_negative_controls_computed': False},
        {'silent_terminal_truncation_allowed': True},
        {'rho_zero_patch_allowed': True},
        {'negative_rho_orientation_branch_admitted': True},
        {'time_orientation_selected': True},
        {'global_metric_reconstruction_proved': True},
        {'curvature_tensor_constructed': True},
        {'einstein_hilbert_action_used': True},
        {'ghy_boundary_term_used': True},
        {'full_m1_functional_constructed': True},
        {'global_boundary_completion_proved': True},
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


def test_29_pairs_and_2030_bounded_jets_are_complete(receipt) -> None:
    fields = compatibility_field_specs()
    antifields = compatibility_antifield_specs()
    assert len(fields) == len(antifields) == 29
    assert fields[-1].name == 'ell'
    assert fields[-1].density_weight == -1
    assert antifields[-1].density_weight == 2
    for field, antifield in zip(fields, antifields, strict=True):
        assert antifield.name == f'{field.name}_star'
        assert antifield.parity == (field.parity + 1) % 2
        assert antifield.ghost_number == -field.ghost_number - 1
        assert antifield.density_weight == 1 - field.density_weight
    assert len(ALL_VARIABLE_SPECS) == 58
    assert len(MULTIINDICES) == 35
    assert len(JET_LOOKUP) == 2030
    assert receipt.bounded_even_jet_generator_count == 1015
    assert receipt.bounded_odd_jet_generator_count == 1015
    with pytest.raises(CompatibilityJetOrderExceeded):
        horizontal_derivative(
            generator('ell', (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0)),
            0,
        )


def test_determinant_adjugate_and_density_weights_close_exactly(receipt) -> None:
    model = metric_density_compatibility_bv_model()
    assert model.determinant_density.term_count == 17
    assert component_factor_weight_sums(model.determinant_density) == frozenset((4,))
    assert all(residual.is_zero for residual in adjugate_identity_residuals())
    determinant_variation = apply_compatibility_brst(
        model.determinant_density,
        model.transformations,
    )
    assert determinant_variation == density_transformation_target(
        model.determinant_density,
        2,
    )
    constraint_variation = apply_compatibility_brst(
        model.compatibility_constraint,
        model.transformations,
    )
    assert constraint_variation == density_transformation_target(
        model.compatibility_constraint,
        2,
    )
    compatibility_variation = apply_compatibility_brst(
        model.compatibility_density,
        model.transformations,
    )
    assert compatibility_variation == density_transformation_target(
        model.compatibility_density,
        1,
    )
    assert receipt.determinant_weight_two_mismatch_term_count == 0
    assert receipt.wrong_determinant_weight_four_mismatch_term_count == 68


def test_classical_density_and_29_left_brst_maps_close(receipt) -> None:
    model = metric_density_compatibility_bv_model()
    variation = apply_compatibility_brst(
        model.classical_density,
        model.transformations,
    )
    assert variation == divergence(model.classical_boundary_current)
    assert all(
        apply_compatibility_brst(image, model.transformations).is_zero
        for image in model.transformations.values()
    )
    assert receipt.scalar_density_term_count == 52
    assert receipt.compatibility_density_term_count == 18
    assert receipt.classical_density_term_count == 70
    assert receipt.antifield_density_term_count == 172
    assert receipt.extended_density_term_count == 242
    assert receipt.classical_variation_term_count == 1116
    assert receipt.classical_current_term_count == 280
    assert receipt.classical_current_divergence_term_count == 1116
    assert receipt.classical_identity_mismatch_term_count == 0
    assert receipt.base_nilpotency_component_count == 29
    assert receipt.base_nilpotency_nonzero_component_count == 0
    assert receipt.derived_transformation_mismatch_term_count == 0


def test_standard_left_bv_signs_and_master_currents_are_live(receipt) -> None:
    ell = generator('ell')
    ell_one = generator('ell', unit_multiindex(0))
    ell_star = generator('ell_star')
    ghost = generator('c0')
    ghost_one = generator('c0', unit_multiindex(0))
    ghost_star = generator('c0_star')
    one = SparseSuperPolynomial.scalar(1)
    assert local_bv_antibracket_density(ell, ell_star) == one
    assert local_bv_antibracket_density(ell_star, ell) == -one
    two_odd = ghost * ell_star
    assert jet_partial_derivative(
        two_odd, 'c0', ZERO_MULTIINDEX, side='left'
    ) == -jet_partial_derivative(
        two_odd, 'c0', ZERO_MULTIINDEX, side='right'
    )
    first = ell_star * ghost * ell_one
    second = -(ghost_star * ghost * ghost_one)
    assert graded_antisymmetry_residual(first, second).is_zero
    assert graded_jacobi_residual(first, second, ell).is_zero
    assert receipt.jacobi_nonzero_nested_bracket_count == 2
    assert receipt.master_density_term_count == 3036
    assert receipt.master_density_maximum_total_jet_order == 2
    assert receipt.master_density_ghost_numbers == (1,)
    assert receipt.master_afn0_term_count == 1356
    assert receipt.master_afn1_term_count == 1680
    assert receipt.analytic_afn0_current_term_count == 460
    assert receipt.compatibility_afn0_current_increment_term_count == 72
    assert receipt.analytic_afn0_mismatch_term_count == 0
    assert receipt.homotopy_afn1_current_term_count == 1800
    assert receipt.homotopy_afn1_direct_mismatch_term_count == 0
    assert receipt.full_master_current_term_count == 2260
    assert receipt.full_master_current_divergence_term_count == 3036
    assert receipt.full_master_current_mismatch_term_count == 0
    assert receipt.master_euler_audit_count == 58
    assert receipt.master_euler_nonzero_count == 0


def test_positive_rho_metric_patch_reconstructs_exactly_and_bad_patches_fail(receipt) -> None:
    zero = Fraction(0)
    h = (
        (Fraction(-4), zero, zero, zero),
        (zero, Fraction(4), zero, zero),
        (zero, zero, Fraction(1), zero),
        (zero, zero, zero, Fraction(1)),
    )
    reconstructed = reconstruct_metric_patch(h, Fraction(4))
    assert reconstructed.h_determinant == -16
    assert reconstructed.compatibility_residual == 0
    assert reconstructed.inverse_product_maximum_residual == 0
    assert reconstructed.g_covariant_determinant == -16
    assert reconstructed.g_contravariant_determinant == Fraction(-1, 16)
    assert reconstructed.real_symmetric_nondegenerate_lorentzian_inertia
    assert not reconstructed.time_orientation_selected
    assert not reconstructed.global_patch_reconstruction_proved
    assert receipt.correct_constraint_numeric_residual == 0
    assert receipt.wrong_sign_constraint_numeric_residual == -32
    assert receipt.rho_zero_patch_rejected
    assert receipt.negative_rho_patch_rejected
    assert receipt.incompatible_determinant_patch_rejected
    assert receipt.nonsymmetric_h_patch_rejected


def test_bridge_specific_negative_controls_fail(receipt) -> None:
    assert receipt.missing_h_weight_covariance_mismatch_term_count == 68
    assert receipt.ell_weight_zero_density_mismatch_term_count == 72
    assert receipt.ell_weight_zero_classical_mismatch_term_count == 72
    assert receipt.ell_weight_zero_euler_ell_term_count == 72
    assert receipt.omitted_ell_antifield_transformation_mismatch_term_count == 8
    assert receipt.omitted_ell_antifield_master_ell_euler_term_count == 300
    assert receipt.wrong_antibracket_canonical_residual_term_count == 1
    assert receipt.wrong_antibracket_antisymmetry_residual_term_count == 1
    assert receipt.wrong_antibracket_jacobi_residual_term_count == 1
    assert receipt.terminal_jet_derivative_rejected


def test_scope_is_local_metric_density_compatibility_not_curvature_eh_or_quantum(receipt) -> None:
    assert receipt.upstream_e70_f_verified
    assert 'two-dimensional metric-density precedent' in receipt.source_boundary
    assert 'neither literally supplies' in receipt.source_boundary
    assert 'weight two' in receipt.compatibility_relation
    assert receipt.determinant_polynomial_constructed
    assert receipt.adjugate_identity_computed
    assert receipt.determinant_weight_two_covariance_computed
    assert receipt.compatibility_ideal_brst_stable
    assert receipt.weight_minus_one_multiplier_constructed
    assert receipt.conditional_positive_rho_metric_reconstruction_computed
    assert receipt.compatibility_cme_mod_dh_computed
    assert not receipt.silent_terminal_truncation_allowed
    assert not receipt.rho_zero_patch_allowed
    assert not receipt.negative_rho_orientation_branch_admitted
    assert not receipt.time_orientation_selected
    assert not receipt.global_metric_reconstruction_proved
    assert not receipt.curvature_tensor_constructed
    assert not receipt.einstein_hilbert_action_used
    assert not receipt.ghy_boundary_term_used
    assert not receipt.full_m1_functional_constructed
    assert not receipt.global_boundary_completion_proved
    assert not receipt.functional_measure_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.continuum_loop_st_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.declared_m1_metric_density_compatibility_bv_gate_passed
