import math

import pytest

from examples.physics.time_homogeneous_pointer_qca import (
    certify_continuous_hamiltonian_front,
    certify_time_homogeneous_pointer_qca,
)


def _certificate(theta: float = 0.61):
    return certify_time_homogeneous_pointer_qca(
        site_count=4,
        audited_depth=2,
        dead_state_count=4,
        theta=theta,
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
        energy_gap=2.5,
    )


def test_propagating_chain_refutes_open_interval_delay_but_not_all_local_hamiltonians():
    audit = certify_continuous_hamiltonian_front(
        hop_count=3,
        coupling_rate=1.0,
        elapsed_time=0.1,
        lattice_spacing=1.0,
        causal_speed=1.0,
    )

    assert audit.dimensionless_coupling_time == pytest.approx(0.1)
    assert audit.path_product == pytest.approx(1.0)
    assert audit.first_nonzero_power == 3
    assert audit.unique_minimal_path
    assert audit.minimal_path_coefficient_nonzero
    assert audit.open_interval_exact_delay_impossible
    assert audit.sampled_before_causal_arrival
    assert audit.sampled_early_tail_nonzero
    assert audit.exact_endpoint_probability > 0.0
    assert audit.relative_leading_residual < 0.01

    assert audit.commuting_ising_negative_control_pass
    assert audit.ising_distant_commutator_norm < 1.0e-10
    assert audit.broad_all_local_hamiltonians_spread_claim_refuted
    assert not audit.every_positive_time_nonzero_claimed
    assert not audit.exact_relativistic_dynamics_derived


def test_fixed_local_update_is_unitary_energy_preserving_and_multihead_safe():
    certificate = _certificate()

    assert certificate.local_coin_unitarity_residual < 1.0e-10
    assert certificate.relative_local_energy_commutator_residual < 1.0e-10
    assert certificate.head_shift_bijection
    assert (
        certificate.head_shift_unique_image_count
        == certificate.head_shift_configuration_count
    )
    assert certificate.arbitrary_multihead_configurations_covered
    assert certificate.full_tensor_qca_unitary_by_composition
    assert certificate.time_homogeneous_discrete_update
    assert not certificate.external_per_edge_schedule_required


def test_reduced_channel_is_cptp_causal_and_has_identity_standard_limit():
    certificate = _certificate()

    assert certificate.cptp_within_tolerance
    assert certificate.kraus_completeness_residual < 1.0e-10
    assert certificate.minimum_choi_eigenvalue > -1.0e-10
    assert certificate.output_trace_residual < 1.0e-10
    assert certificate.minimum_output_eigenvalue > -1.0e-10
    assert certificate.born_probability_sum_residual < 1.0e-10
    assert certificate.minimum_born_probability > -1.0e-10

    assert certificate.structural_causal_support_exact
    assert certificate.structural_influence_cone == (0, 1, 2)
    assert certificate.spacelike_sites == (3,)
    assert certificate.front_speed_bound <= certificate.causal_speed
    assert certificate.maximum_seed_variation_spacelike_trace_distance < 1.0e-10
    assert certificate.quantum_identity_limit_at_zero_coupling
    assert certificate.standard_limit_superoperator_residual < 1.0e-10


def test_seed_domino_statistics_and_energy_receipts_close_without_double_counting():
    certificate = _certificate()
    probability = math.sin(certificate.theta) ** 2

    assert certificate.activation_probabilities == pytest.approx(
        (1.0, probability, probability**2, 0.0), abs=1.0e-10
    )
    assert certificate.expected_activation_probabilities == pytest.approx(
        certificate.activation_probabilities, abs=1.0e-10
    )
    assert certificate.paid_energy_probabilities == pytest.approx(
        (1.0 - probability, probability * (1.0 - probability), probability**2),
        abs=1.0e-10,
    )
    assert certificate.expected_paid_energy_probabilities == pytest.approx(
        certificate.paid_energy_probabilities, abs=1.0e-10
    )
    assert certificate.pointer_seed_statistics_match_prior_domino

    assert certificate.energy_conserved_within_tolerance
    assert certificate.energy_resolved_instrument_within_tolerance
    assert certificate.relative_total_energy_balance_residual < 1.0e-10
    assert certificate.relative_reverse_transfer_identity_residual < 1.0e-10
    assert certificate.maximum_relative_branch_energy_residual < 1.0e-10
    assert certificate.expected_battery_energy_paid == pytest.approx(
        certificate.final_system_energy - certificate.initial_system_energy,
        abs=1.0e-10,
    )
    assert sum(
        outcome.probability for outcome in certificate.battery_outcomes
    ) == pytest.approx(1.0, abs=1.0e-10)


def test_certificate_keeps_finite_horizon_and_open_bridges_explicit():
    certificate = _certificate()

    assert certificate.audited_depth_less_than_dead_count
    assert certificate.finite_horizon_only
    assert not certificate.trigger_head_preparation_derived
    assert not certificate.continuous_physical_clock_derived
    assert not certificate.permanent_absorbing_dead_state_derived
    assert not certificate.covariant_action_derived
    assert not certificate.gr_limit_derived
    assert not certificate.record_to_gravity_source_derived
    assert not certificate.full_prior_domino_channel_equivalence_derived
    assert not certificate.cross_dataset_parameter_fixing_derived
    assert not certificate.independent_holdout_prediction_derived


def test_invalid_horizon_and_superluminal_tick_are_rejected():
    common = dict(
        site_count=4,
        audited_depth=2,
        dead_state_count=4,
        theta=0.4,
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
    )

    with pytest.raises(ValueError, match="dead_state_count must exceed"):
        certify_time_homogeneous_pointer_qca(
            **{**common, "dead_state_count": 2}
        )
    with pytest.raises(ValueError, match="causal timing"):
        certify_time_homogeneous_pointer_qca(
            **{**common, "clock_step": 0.5}
        )
    with pytest.raises(ValueError, match="leave an unvisited site"):
        certify_time_homogeneous_pointer_qca(
            **{**common, "audited_depth": 3}
        )
