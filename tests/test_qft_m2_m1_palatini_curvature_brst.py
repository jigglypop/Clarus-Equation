from dataclasses import replace

import pytest

from examples.physics.qft_m2_m1_palatini_curvature_brst import (
    CONTRACT_SHA256,
    JET_LOOKUP,
    MAXIMUM_TOTAL_JET_ORDER,
    MULTIINDICES,
    PalatiniJetOrderExceeded,
    VARIABLE_SPECS,
    apply_palatini_brst,
    canonical_contract_payload,
    contract_payload_sha256,
    divergence,
    evaluate_m1_palatini_curvature_brst_gate,
    generator,
    horizontal_derivative,
    m1_palatini_curvature_brst_contract,
    palatini_curvature_brst_model,
    ricci_covariance_target,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_palatini_curvature_brst_gate()


def test_contract_hash_sources_basis_and_claim_ceiling_fail_closed() -> None:
    contract = m1_palatini_curvature_brst_contract()
    validate_contract(contract)
    assert canonical_contract_payload(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    bad_h = replace(
        contract.variable_specs[0],
        geometric_density_weight=0,
    )
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'affine_source': 'unlocked'},
        {'two_dimensional_precedent': 'literal 4D proof'},
        {'source_boundary': 'all canonical maps equal diffeomorphisms'},
        {'palatini_density': 'R alone'},
        {'normalization': 'dimensionful unspecified variables'},
        {'affine_brst_convention': 'connection transforms as a tensor'},
        {'dimension': 3},
        {'maximum_total_jet_order': 2},
        {'variable_specs': (bad_h,) + contract.variable_specs[1:]},
        {'variable_specs': contract.variable_specs[:-1]},
        {'upstream_hashes': (('E70-G', '0' * 64),)},
        {'torsion_free_connection_basis_constructed': False},
        {'affine_second_ghost_derivative_included': False},
        {'affine_brst_nilpotency_computed': False},
        {'ricci_tensor_covariance_computed': False},
        {'nonsymmetric_ricci_fixture_retained': False},
        {'palatini_bulk_density_constructed': False},
        {'palatini_density_total_divergence_computed': False},
        {'nonzero_boundary_current_retained': False},
        {'live_negative_controls_computed': False},
        {'silent_terminal_truncation_allowed': True},
        {'metric_density_constraint_included': True},
        {'connection_equation_derived': True},
        {'metric_compatibility_derived': True},
        {'levi_civita_connection_derived': True},
        {'first_second_order_equivalence_proved': True},
        {'canonical_gauge_generator_equated_to_diffeomorphism': True},
        {'ghy_boundary_term_used': True},
        {'global_boundary_completion_proved': True},
        {'scalar_full_m1_action_assembled': True},
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


def test_torsion_free_54_component_basis_and_1890_jets_are_locked(receipt) -> None:
    h_specs = tuple(spec for spec in VARIABLE_SPECS if spec.name.startswith('h'))
    gamma_specs = tuple(
        spec for spec in VARIABLE_SPECS if spec.name.startswith('Gamma')
    )
    ghost_specs = tuple(spec for spec in VARIABLE_SPECS if spec.name.startswith('c'))
    assert len(h_specs) == 10
    assert len(gamma_specs) == 40
    assert len(ghost_specs) == 4
    assert len(VARIABLE_SPECS) == 54
    assert len(MULTIINDICES) == 35
    assert len(JET_LOOKUP) == 1890
    assert receipt.bounded_even_jet_generator_count == 1750
    assert receipt.bounded_odd_jet_generator_count == 140
    assert receipt.torsion_free_name_symmetry_locked
    with pytest.raises(KeyError):
        generator('Gamma0_10')
    with pytest.raises(PalatiniJetOrderExceeded):
        horizontal_derivative(
            generator(
                'Gamma0_00',
                (MAXIMUM_TOTAL_JET_ORDER, 0, 0, 0),
            ),
            0,
        )


def test_affine_maps_include_second_ghost_jets_and_are_nilpotent(receipt) -> None:
    model = palatini_curvature_brst_model()
    assert len(model.transformations) == 54
    assert receipt.affine_inhomogeneous_second_ghost_component_count == 40
    for name in ('h00', 'Gamma0_00', 'Gamma3_33', 'c0'):
        assert apply_palatini_brst(
            model.transformations[name],
            model.transformations,
        ).is_zero
    assert receipt.base_nilpotency_component_count == 54
    assert receipt.base_nilpotency_nonzero_component_count == 0
    assert receipt.base_nilpotency_maximum_residual_term_count == 0


def test_all_ricci_components_transform_covariantly_without_symmetrizing(receipt) -> None:
    model = palatini_curvature_brst_model()
    for mu, nu in ((0, 0), (0, 1), (2, 3), (3, 3)):
        variation = apply_palatini_brst(
            model.ricci_components[(mu, nu)],
            model.transformations,
        )
        assert variation == ricci_covariance_target(
            model.ricci_components,
            mu,
            nu,
        )
    assert receipt.ricci_component_count == 16
    assert receipt.ricci_total_term_count == 396
    assert receipt.ricci_minimum_component_term_count == 24
    assert receipt.ricci_maximum_component_term_count == 27
    assert receipt.ricci_covariance_nonzero_component_count == 0
    assert receipt.nonsymmetric_ricci_component_count == 6
    assert receipt.nonsymmetric_ricci_total_term_count == 48


def test_palatini_density_closes_to_the_retained_local_current(receipt) -> None:
    model = palatini_curvature_brst_model()
    variation = apply_palatini_brst(
        model.palatini_density,
        model.transformations,
    )
    assert variation == divergence(model.boundary_current)
    assert receipt.palatini_density_term_count == 276
    assert receipt.palatini_variation_term_count == 4032
    assert receipt.boundary_current_term_count == 1104
    assert receipt.boundary_current_divergence_term_count == 4032
    assert receipt.palatini_density_identity_mismatch_term_count == 0


def test_affine_and_density_negative_controls_are_live(receipt) -> None:
    assert receipt.omitted_inhomogeneous_nonzero_nilpotency_component_count == 0
    assert receipt.omitted_inhomogeneous_ricci_nonzero_component_count == 16
    assert receipt.omitted_inhomogeneous_ricci_maximum_residual_term_count == 39
    assert receipt.omitted_inhomogeneous_density_mismatch_term_count == 372
    assert receipt.omitted_lower_transport_nonzero_nilpotency_component_count == 40
    assert receipt.omitted_lower_transport_ricci_nonzero_component_count == 16
    assert receipt.omitted_lower_transport_density_mismatch_term_count == 1786
    assert receipt.wrong_upper_transport_nonzero_nilpotency_component_count == 40
    assert receipt.wrong_upper_transport_ricci_nonzero_component_count == 16
    assert receipt.wrong_upper_transport_density_mismatch_term_count == 2040
    assert receipt.missing_h_weight_density_mismatch_term_count == 1104
    assert receipt.wrong_ghost_sign_nonzero_nilpotency_component_count == 50
    assert receipt.wrong_ghost_sign_maximum_nilpotency_residual_term_count == 121


def test_ricci_sign_controls_are_live(receipt) -> None:
    assert receipt.wrong_ricci_derivative_nonzero_covariance_component_count == 16
    assert receipt.wrong_ricci_derivative_density_mismatch_term_count == 200
    assert receipt.wrong_ricci_product_nonzero_covariance_component_count == 16
    assert receipt.wrong_ricci_product_density_mismatch_term_count == 256
    assert receipt.terminal_jet_derivative_rejected


def test_scope_is_bounded_palatini_covariance_not_eom_bv_or_quantum(receipt) -> None:
    assert receipt.upstream_e70_g_verified
    assert 'finite-jet adaptation' in receipt.source_boundary
    assert 'not identified with every canonical' in receipt.source_boundary
    assert receipt.torsion_free_connection_basis_constructed
    assert receipt.affine_second_ghost_derivative_included
    assert receipt.affine_brst_nilpotency_computed
    assert receipt.ricci_tensor_covariance_computed
    assert receipt.nonsymmetric_ricci_fixture_retained
    assert receipt.palatini_bulk_density_constructed
    assert receipt.palatini_density_total_divergence_computed
    assert receipt.nonzero_boundary_current_retained
    assert receipt.live_negative_controls_computed
    assert not receipt.silent_terminal_truncation_allowed
    assert not receipt.metric_density_constraint_included
    assert not receipt.connection_equation_derived
    assert not receipt.metric_compatibility_derived
    assert not receipt.levi_civita_connection_derived
    assert not receipt.first_second_order_equivalence_proved
    assert not receipt.canonical_gauge_generator_equated_to_diffeomorphism
    assert not receipt.ghy_boundary_term_used
    assert not receipt.global_boundary_completion_proved
    assert not receipt.scalar_full_m1_action_assembled
    assert not receipt.full_m1_bv_functional_constructed
    assert not receipt.classical_master_equation_computed
    assert not receipt.functional_measure_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.continuum_loop_st_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.declared_m1_palatini_curvature_brst_gate_passed
