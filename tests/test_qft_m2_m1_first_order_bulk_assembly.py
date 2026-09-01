from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_m2_m1_first_order_bulk_assembly import (
    COEFFICIENT_FIXTURE,
    CONNECTION_KEYS,
    CONTRACT_SHA256,
    FIELD_SPECS,
    JET_LOOKUP,
    MULTIINDICES,
    FirstOrderBulkJetOrderExceeded,
    NormalizedCoefficientFixture,
    _gamma_name,
    apply_bulk_brst,
    canonical_connection_euler_component,
    canonical_contract_payload,
    contract_payload_sha256,
    divergence,
    even_euler_derivative,
    evaluate_m1_first_order_bulk_assembly_gate,
    first_order_bulk_assembly_model,
    generator,
    horizontal_derivative,
    lift_external_polynomial,
    m1_first_order_bulk_assembly_contract,
    validate_coefficient_fixture,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_first_order_bulk_assembly_gate()


def test_contract_hash_sources_basis_coefficients_and_ceiling_fail_closed() -> None:
    contract = m1_first_order_bulk_assembly_contract()
    validate_contract(contract)
    validate_coefficient_fixture(contract.coefficient_fixture)
    assert canonical_contract_payload(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    with pytest.raises(ValueError):
        generator('missing_first_order_bulk_variable')
    bad_coefficients = replace(contract.coefficient_fixture, gravity=Fraction(0))
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'density_brst_source': 'unlocked'},
        {'first_order_loop_caveat_source': 'literal independent-Gamma BV'},
        {'m1_action_edition': 'unknown action'},
        {'source_boundary': 'all metric-affine matter is minimal'},
        {'normalization': 'prime values are physical predictions'},
        {'physical_parameter_dictionary': 'untracked'},
        {'dimension': 3},
        {'maximum_total_jet_order': 2},
        {'field_specs': contract.field_specs[:-1]},
        {'coefficient_fixture': bad_coefficients},
        {'upstream_hashes': (('E70-I', '0' * 64),)},
        {'m1_six_bulk_structures_assembled': False},
        {'compatibility_multiplier_assembled': False},
        {'shared_brst_maps_matched': False},
        {'all_69_base_nilpotency_computed': False},
        {'seven_unit_density_identities_computed': False},
        {'mixed_density_total_divergence_computed': False},
        {'nonzero_boundary_current_retained': False},
        {'gamma_euler_factorization_computed': False},
        {'ell_constraint_euler_computed': False},
        {'live_negative_controls_computed': False},
        {'physical_coefficients_inferred_from_fixture': True},
        {'arbitrary_metric_affine_matter_generalized': True},
        {'all_field_euler_equations_computed': True},
        {'palatini_boundary_term_constructed': True},
        {'ghy_boundary_term_used': True},
        {'global_first_second_order_equivalence_proved': True},
        {'full_m1_bv_functional_constructed': True},
        {'classical_master_equation_computed': True},
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


def test_69_field_catalog_2415_jets_and_shared_maps_are_exact(receipt) -> None:
    model = first_order_bulk_assembly_model()
    assert len(FIELD_SPECS) == 69
    assert sum(spec.parity == 0 for spec in FIELD_SPECS) == 61
    assert sum(spec.parity == 1 for spec in FIELD_SPECS) == 8
    assert len(MULTIINDICES) == 35
    assert len(JET_LOOKUP) == 2415
    assert receipt.bounded_even_jet_generator_count == 2135
    assert receipt.bounded_odd_jet_generator_count == 280
    assert len(model.transformations) == 69
    assert len(model.shared_transformation_mismatches) == 14
    assert all(value.is_zero for value in model.shared_transformation_mismatches)
    with pytest.raises(FirstOrderBulkJetOrderExceeded):
        horizontal_derivative(generator('Gamma0_00', (3, 0, 0, 0)), 0)


def test_all_69_merged_brst_maps_are_nilpotent(receipt) -> None:
    model = first_order_bulk_assembly_model()
    for name in ('phi_chi', 'rho', 'h00', 'Gamma0_00', 'c0', 'barc0', 'B0', 'ell'):
        assert apply_bulk_brst(
            model.transformations[name],
            model.transformations,
        ).is_zero
    assert receipt.base_nilpotency_component_count == 69
    assert receipt.base_nilpotency_nonzero_component_count == 0
    assert receipt.base_nilpotency_maximum_residual_term_count == 0


def test_seven_unit_densities_close_independently(receipt) -> None:
    expected = (
        ('einstein_hilbert_palatini', 276, 4032, 1104, 4032),
        ('cosmological_constant', 1, 8, 4, 8),
        ('chi_kinetic', 10, 144, 40, 144),
        ('chi_mass', 1, 12, 4, 12),
        ('chi_quartic', 1, 12, 4, 12),
        ('reference_kinetic', 40, 576, 160, 576),
        ('metric_density_compatibility', 18, 372, 72, 372),
    )
    observed = tuple(
        (
            value.name,
            value.density_term_count,
            value.variation_term_count,
            value.current_term_count,
            value.current_divergence_term_count,
        )
        for value in receipt.unit_density_receipts
    )
    assert observed == expected
    assert all(
        value.identity_mismatch_term_count == 0
        for value in receipt.unit_density_receipts
    )


def test_mixed_prime_density_closes_with_retained_current(receipt) -> None:
    model = first_order_bulk_assembly_model()
    variation = apply_bulk_brst(model.classical_density, model.transformations)
    assert variation == divergence(model.boundary_current)
    assert receipt.coefficient_fixture == COEFFICIENT_FIXTURE
    assert receipt.classical_density_term_count == 347
    assert receipt.classical_variation_term_count == 5156
    assert receipt.boundary_current_term_count == 1388
    assert receipt.boundary_current_divergence_term_count == 5156
    assert receipt.classical_identity_mismatch_term_count == 0


def test_gamma_and_ell_euler_equations_factorize_exactly(receipt) -> None:
    model = first_order_bulk_assembly_model()
    for key in ((0, 0, 0), (0, 0, 1), (3, 3, 3)):
        direct = even_euler_derivative(
            model.classical_density,
            _gamma_name(*key),
        )
        target = (
            COEFFICIENT_FIXTURE.gravity
            * lift_external_polynomial(canonical_connection_euler_component(key))
        )
        assert direct == target
    ell_euler = even_euler_derivative(model.classical_density, 'ell')
    assert ell_euler == (
        COEFFICIENT_FIXTURE.compatibility
        * model.compatibility_constraint
    )
    assert receipt.gamma_euler_component_count == len(CONNECTION_KEYS) == 40
    assert receipt.gamma_euler_total_term_count == 456
    assert receipt.gamma_euler_nonzero_factorization_mismatch_count == 0
    assert receipt.ell_euler_term_count == 18
    assert receipt.ell_constraint_mismatch_term_count == 0


def test_assembly_specific_negative_controls_are_live(receipt) -> None:
    assert receipt.wrong_gravity_factor_nonzero_gamma_component_count == 40
    assert receipt.wrong_gravity_factor_total_mismatch_term_count == 456
    assert receipt.wrong_gravity_factor_maximum_mismatch_term_count == 17
    assert receipt.direct_gamma_matter_contamination_nonzero_component_count == 1
    assert receipt.direct_gamma_matter_contamination_total_term_count == 1
    assert receipt.omitted_ell_map_compatibility_mismatch_term_count == 144
    assert receipt.wrong_cosmological_scalar_mismatch_term_count == 4
    assert receipt.perturbed_shared_map_mismatch_term_count == 1
    assert receipt.omitted_boundary_current_residual_term_count == 5156
    assert receipt.invalid_coefficient_fixture_rejected
    assert receipt.terminal_jet_derivative_rejected
    with pytest.raises(ValueError):
        validate_coefficient_fixture(
            NormalizedCoefficientFixture(
                Fraction(2),
                Fraction(2),
                Fraction(5),
                Fraction(7),
                Fraction(11),
                Fraction(13),
                Fraction(17),
            )
        )


def test_scope_is_normalized_classical_bulk_not_boundary_bv_or_quantum(receipt) -> None:
    assert receipt.upstream_e70_c_verified
    assert receipt.upstream_e70_i_verified
    assert 'convention-adapted assembly' in receipt.source_boundary
    assert 'not physical parameter estimates' in receipt.normalization
    assert receipt.m1_six_bulk_structures_assembled
    assert receipt.compatibility_multiplier_assembled
    assert receipt.shared_brst_maps_matched
    assert receipt.all_69_base_nilpotency_computed
    assert receipt.seven_unit_density_identities_computed
    assert receipt.mixed_density_total_divergence_computed
    assert receipt.nonzero_boundary_current_retained
    assert receipt.gamma_euler_factorization_computed
    assert receipt.ell_constraint_euler_computed
    assert receipt.live_negative_controls_computed
    assert not receipt.physical_coefficients_inferred_from_fixture
    assert not receipt.arbitrary_metric_affine_matter_generalized
    assert not receipt.all_field_euler_equations_computed
    assert not receipt.palatini_boundary_term_constructed
    assert not receipt.ghy_boundary_term_used
    assert not receipt.global_first_second_order_equivalence_proved
    assert not receipt.full_m1_bv_functional_constructed
    assert not receipt.classical_master_equation_computed
    assert not receipt.functional_measure_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.continuum_loop_st_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.declared_m1_first_order_bulk_assembly_gate_passed
