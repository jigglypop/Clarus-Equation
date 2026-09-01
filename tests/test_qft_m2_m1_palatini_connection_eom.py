from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_m2_m1_palatini_connection_eom import (
    CONNECTION_KEYS,
    CONTRACT_SHA256,
    analytic_connection_euler_component,
    canonical_contract_payload,
    connection_euler_derivative,
    connection_euler_mismatches,
    contract_payload_sha256,
    covariant_derivative_h,
    evaluate_m1_palatini_connection_eom_gate,
    evaluate_metric_jet_fixture,
    m1_palatini_connection_eom_contract,
    metric_density_jet,
    metric_jet_fixtures,
    palatini_first_variation,
    projective_shift_lower_symmetry_violation_count,
    trace_reduction_residuals,
    validate_contract,
)
from examples.physics.qft_m2_m1_palatini_curvature_brst import (
    palatini_curvature_brst_model,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m1_palatini_connection_eom_gate()


def test_contract_hash_sources_variation_and_ceiling_fail_closed() -> None:
    contract = m1_palatini_connection_eom_contract()
    validate_contract(contract)
    assert canonical_contract_payload(contract)
    assert contract_payload_sha256(contract) == CONTRACT_SHA256
    changes = (
        {'contract_sha256': '0' * 64},
        {'primary_source': 'unsourced'},
        {'projective_source': 'unlocked'},
        {'projective_invariance_source': 'unlocked'},
        {'first_order_source': 'unlocked'},
        {'source_boundary': 'torsion order never matters'},
        {'normalization': 'dimensionful unspecified variables'},
        {'variation_convention': 'unrestricted connection'},
        {'dimension': 2},
        {'maximum_total_jet_order': 2},
        {'connection_keys': contract.connection_keys[:-1]},
        {'upstream_hashes': (('E70-H', '0' * 64),)},
        {'torsion_free_variation_preregistered': False},
        {'direct_connection_euler_computed': False},
        {'retained_boundary_current_computed': False},
        {'analytic_euler_match_computed': False},
        {'traced_equation_reduction_computed': False},
        {'positive_rho_metric_jet_fixtures_constructed': False},
        {'compatibility_linear_system_full_rank': False},
        {'unique_levi_civita_reconstruction_computed': False},
        {'projective_scope_controlled': False},
        {'live_negative_controls_computed': False},
        {'unrestricted_connection_variation_used': True},
        {'unrestricted_projective_family_classified': True},
        {'palatini_boundary_term_constructed': True},
        {'ghy_boundary_term_used': True},
        {'global_first_second_order_equivalence_proved': True},
        {'full_m1_action_assembled': True},
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


def test_all_40_direct_connection_euler_derivatives_match_analytic_formula(receipt) -> None:
    density = palatini_curvature_brst_model().palatini_density
    mismatches = connection_euler_mismatches(density)
    assert len(CONNECTION_KEYS) == 40
    assert len(mismatches) == 40
    assert all(value.is_zero for value in mismatches)
    diagonal = (0, 0, 0)
    off_diagonal = (0, 0, 1)
    assert connection_euler_derivative(density, diagonal) == (
        analytic_connection_euler_component(*diagonal)
    )
    assert connection_euler_derivative(density, off_diagonal) == (
        2 * analytic_connection_euler_component(*off_diagonal)
    )
    assert receipt.palatini_density_term_count == 276
    assert receipt.direct_analytic_euler_nonzero_mismatch_count == 0


def test_first_variation_retains_and_reconstructs_boundary_current(receipt) -> None:
    variation = palatini_first_variation()
    assert variation.raw_variation == (
        variation.bulk_variation + variation.boundary_divergence
    )
    assert variation.analytic_boundary_mismatch.is_zero
    assert variation.raw_variation.term_count == 456
    assert variation.bulk_variation.term_count == 456
    assert sum(value.term_count for value in variation.boundary_current) == 84
    assert variation.boundary_divergence.term_count == 168
    assert receipt.omitted_boundary_current_residual_term_count == 168


def test_trace_reduction_gives_three_halves_density_divergence(receipt) -> None:
    residuals = trace_reduction_residuals()
    assert len(residuals) == 4
    assert all(value.is_zero for value in residuals)
    assert receipt.trace_factor_numerator == 3
    assert receipt.trace_factor_denominator == 2
    assert receipt.trace_reduction_nonzero_residual_count == 0


def test_four_exact_metric_jets_solve_uniquely_to_levi_civita(receipt) -> None:
    direct = tuple(
        evaluate_metric_jet_fixture(fixture)
        for fixture in metric_jet_fixtures()
    )
    assert len(direct) == 4
    assert tuple(value.rho for value in direct) == (
        Fraction(1),
        Fraction(1),
        Fraction(1),
        Fraction(4),
    )
    assert all(value.connection_system_rank == 40 for value in direct)
    assert all(value.determinant_compatibility_residual == 0 for value in direct)
    assert all(value.solve_nonzero_residual_count == 0 for value in direct)
    assert all(value.levi_civita_nonzero_mismatch_count == 0 for value in direct)
    assert receipt.minimum_connection_system_rank == 40
    assert receipt.maximum_levi_civita_absolute_mismatch == 0


def test_euler_and_density_trace_negative_controls_are_live(receipt) -> None:
    density = palatini_curvature_brst_model().palatini_density
    wrong = tuple(
        connection_euler_derivative(density, key)
        - (1 if key[1] == key[2] else 2)
        * (-covariant_derivative_h(*key))
        for key in CONNECTION_KEYS
    )
    assert sum(not value.is_zero for value in wrong) == 16
    assert sum(value.term_count for value in wrong) == 224
    assert max(value.term_count for value in wrong) == 14
    assert receipt.wrong_density_trace_connection_system_rank == 40
    assert receipt.wrong_density_trace_levi_civita_nonzero_mismatch_count == 28
    assert (
        receipt.wrong_density_trace_correct_compatibility_nonzero_residual_count
        == 16
    )
    assert receipt.wrong_density_trace_maximum_absolute_mismatch == Fraction(5, 3)


def test_degenerate_nonsymmetric_projective_and_terminal_controls_fail(receipt) -> None:
    fixture = metric_jet_fixtures()[0]
    metric = fixture.metric_covariant
    bad_metric = (
        (metric[0][0], Fraction(1), metric[0][2], metric[0][3]),
        metric[1],
        metric[2],
        metric[3],
    )
    with pytest.raises(ValueError):
        metric_density_jet(bad_metric, fixture.metric_derivatives)
    assert receipt.singular_h_connection_system_rank == 33
    assert projective_shift_lower_symmetry_violation_count() == 24
    assert receipt.degenerate_metric_jet_rejected
    assert receipt.nonsymmetric_metric_jet_rejected
    assert receipt.terminal_jet_derivative_rejected


def test_scope_is_local_torsion_free_bulk_reduction_not_global_or_quantum(receipt) -> None:
    assert receipt.upstream_e70_h_verified
    assert 'imposes a torsion-free' in receipt.source_boundary
    assert 'unrestricted metric-affine/projective' in receipt.source_boundary
    assert 'vanishes on the boundary' in receipt.variation_convention
    assert receipt.torsion_free_variation_preregistered
    assert receipt.direct_connection_euler_computed
    assert receipt.retained_boundary_current_computed
    assert receipt.analytic_euler_match_computed
    assert receipt.traced_equation_reduction_computed
    assert receipt.positive_rho_metric_jet_fixtures_constructed
    assert receipt.compatibility_linear_system_full_rank
    assert receipt.unique_levi_civita_reconstruction_computed
    assert receipt.projective_scope_controlled
    assert receipt.live_negative_controls_computed
    assert not receipt.unrestricted_connection_variation_used
    assert not receipt.unrestricted_projective_family_classified
    assert not receipt.palatini_boundary_term_constructed
    assert not receipt.ghy_boundary_term_used
    assert not receipt.global_first_second_order_equivalence_proved
    assert not receipt.full_m1_action_assembled
    assert not receipt.full_m1_bv_functional_constructed
    assert not receipt.classical_master_equation_computed
    assert not receipt.functional_measure_computed
    assert not receipt.quantum_master_equation_computed
    assert not receipt.continuum_loop_st_computed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
    assert not receipt.m3_relational_observables_unlocked
    assert receipt.declared_m1_palatini_connection_eom_gate_passed
