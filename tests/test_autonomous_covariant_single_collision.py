import math

import pytest

from examples.physics.causal.autonomous_covariant_single_collision import (
    certify_autonomous_covariant_single_collision,
    smooth_compact_bump,
    smooth_compact_bump_derivative,
)


def _certificate(**overrides):
    arguments = dict(
        head_mass=2.0,
        detector_battery_gap=1.25,
        clock_scale=2.0,
        clock_rate=0.25,
        exchange_coupling=0.2,
        quartic_coupling=0.8,
        mode_overlap_rate=0.75,
        trigger_momentum=0.7,
    )
    arguments.update(overrides)
    return certify_autonomous_covariant_single_collision(**arguments)


def test_action_core_is_dimensionally_closed_hermitian_and_stable():
    certificate = _certificate()

    assert certificate.action_terms_have_mass_dimension_four
    assert certificate.dimensionless_core_arguments == (
        ("T / M_T", "dimensionless clock-bump argument"),
        ("mu_P^2 Delta tau / (2 omega_H)", "dimensionless prep area"),
        ("g_eff Delta tau", "dimensionless exchange area theta"),
        ("sin(theta)^2", "dimensionless probability"),
    )
    assert certificate.compact_smooth_clock_bumps
    assert certificate.clock_windows_disjoint_and_ordered
    assert certificate.potential_hermitian
    assert certificate.potential_reality_residual < 1.0e-12
    assert certificate.analytic_stability_bound_pass
    assert certificate.sampled_stability_bound_pass
    assert certificate.extremal_quartic_potential == pytest.approx(
        certificate.quartic_analytic_lower_bound_coefficient,
        abs=1.0e-12,
    )
    assert certificate.minimum_sampled_quartic_potential >= (
        certificate.quartic_analytic_lower_bound_coefficient - 1.0e-10
    )
    assert certificate.head_mass_matrix_positive
    assert certificate.minimum_head_mass_squared_eigenvalue > 0.0


def test_clock_is_internal_but_has_nonzero_cost_and_backreaction():
    certificate = _certificate()

    assert not certificate.explicit_coordinate_switching_present
    assert certificate.clock_energy_density == pytest.approx(0.5 * 0.25**2)
    assert certificate.dynamic_clock_backreaction_retained
    assert certificate.clock_backreaction_source_norm > 0.0
    assert certificate.potential_gradient_bookkeeping_within_tolerance
    assert certificate.maximum_relative_chain_rule_residual < 1.0e-8
    assert certificate.potential_gradient_bookkeeping_residual < 1.0e-12
    assert not certificate.metric_variation_machine_verified

    assert certificate.maximum_allocation_total_current_residual < 1.0e-12
    assert certificate.allocation_current_difference > 0.0
    assert not certificate.unique_sector_exchange_current_derived


def test_head_vacuum_is_an_exact_no_trigger_counterexample():
    certificate = _certificate()

    assert certificate.head_number_conserved
    assert certificate.head_phase_symmetry_residual < 1.0e-12
    assert certificate.vacuum_head_sector_invariant
    assert certificate.vacuum_head_force_residual < 1.0e-12
    assert not certificate.spontaneous_trigger_from_vacuum_derived
    assert not certificate.initial_trigger_wavepacket_derived
    assert not certificate.initial_clock_state_derived


def test_projected_one_cell_coin_is_cptp_and_matches_sine_probability():
    certificate = _certificate()

    assert certificate.prep_angle == pytest.approx(math.pi / 2.0, abs=1.0e-12)
    assert math.sin(certificate.prep_angle) ** 2 == pytest.approx(1.0)
    assert certificate.trigger_probability == pytest.approx(
        math.sin(certificate.exchange_angle) ** 2
    )
    assert certificate.projected_detector_activation_probability == pytest.approx(
        certificate.trigger_probability, abs=1.0e-10
    )
    assert certificate.one_cell_activation_formula_residual < 1.0e-10

    assert certificate.projected_channel_cptp_within_tolerance
    assert certificate.projected_unitary_residual < 1.0e-10
    assert certificate.projected_kraus_completeness_residual < 1.0e-10
    assert certificate.projected_minimum_choi_eigenvalue > -1.0e-10
    assert certificate.projected_output_trace_residual < 1.0e-10
    assert certificate.projected_minimum_output_eigenvalue > -1.0e-10
    assert certificate.projected_one_cell_e12_channel_match
    assert certificate.projected_one_cell_channel_residual < 1.0e-10
    assert certificate.projected_standard_limit_superoperator_residual < 1.0e-10


def test_projected_battery_receipt_closes_without_double_counting():
    certificate = _certificate()

    assert certificate.projected_energy_receipt_within_tolerance
    assert certificate.relative_projected_energy_commutator_residual < 1.0e-10
    assert certificate.projected_total_energy_balance_residual < 1.0e-10
    assert certificate.projected_reverse_transfer_identity_residual < 1.0e-10
    assert certificate.projected_maximum_branch_energy_residual < 1.0e-10
    assert certificate.projected_expected_battery_energy_paid == pytest.approx(
        certificate.projected_final_detector_energy
        - certificate.projected_initial_detector_energy,
        abs=1.0e-10,
    )
    assert sum(
        outcome.probability for outcome in certificate.projected_battery_outcomes
    ) == pytest.approx(1.0, abs=1.0e-10)


def test_causal_and_gr_claim_ceiling_remains_explicit():
    certificate = _certificate()

    assert certificate.trigger_group_velocity < 1.0
    assert certificate.canonical_matter_principal_symbol
    assert certificate.fixed_background_causal_domain_of_dependence
    assert not certificate.interacting_qft_microcausality_derived
    assert not certificate.operational_no_signalling_instrument_derived
    assert certificate.diffeomorphism_invariant_action_by_construction
    assert certificate.einstein_hilbert_term_present
    assert certificate.pure_einstein_limit_when_matter_vanishes
    assert not certificate.coupled_einstein_hyperbolicity_derived

    assert certificate.projected_single_mode_assumptions_declared
    assert not certificate.continuum_action_cptp_instrument_derived
    assert not certificate.exact_full_qft_to_projected_mode_limit_derived
    assert not certificate.full_e12_domino_equivalence_derived
    assert not certificate.durable_detector_pointer_derived
    assert not certificate.gr_source_matching_derived
    assert not certificate.cross_dataset_parameter_fixing_derived
    assert not certificate.independent_holdout_prediction_derived


def test_compact_bump_and_invalid_action_parameters_fail_closed():
    assert smooth_compact_bump(-1.0, center=-1.0, half_width=0.45) == pytest.approx(1.0)
    assert smooth_compact_bump(-2.0, center=-1.0, half_width=0.45) == 0.0
    assert smooth_compact_bump_derivative(-1.0, center=-1.0, half_width=0.45) == 0.0
    assert smooth_compact_bump_derivative(-0.8, center=-1.0, half_width=0.45) < 0.0

    with pytest.raises(ValueError, match="quartic stability"):
        _certificate(quartic_coupling=0.05)
    with pytest.raises(ValueError, match="disjoint and ordered"):
        _certificate(coin_center=-0.8)
    with pytest.raises(ValueError, match="head mass matrix"):
        _certificate(clock_rate=5.0)
