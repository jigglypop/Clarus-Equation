from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal.causal_domino import (
    _single_site_reduced,
    certify_autonomous_repeated_domino_obstruction,
    certify_causal_probability_deformation_lattice,
    certify_causal_quantum_domino,
    homogeneous_continuous_time_early_arrival_probability,
)


def test_partial_domino_is_cptp_causal_and_energy_conserving() -> None:
    trigger_probability = 0.4
    certificate = certify_causal_quantum_domino(
        site_count=4,
        depth=2,
        theta=math.asin(math.sqrt(trigger_probability)),
        lattice_spacing=2.0,
        clock_step=0.5,
        causal_speed=4.0,
        energy_gap=3.0,
    )

    assert certificate.trigger_probability == pytest.approx(trigger_probability)
    assert certificate.activation_probabilities == pytest.approx((1.0, 0.4, 0.16, 0.0))
    assert certificate.structural_influence_cone == (0, 1, 2)
    assert certificate.spacelike_sites == (3,)
    assert certificate.causal_ratio == pytest.approx(1.0)
    assert certificate.front_speed_bound == pytest.approx(certificate.causal_speed)
    assert certificate.maximum_sampled_spacelike_trace_distance < 1.0e-12
    assert certificate.structural_causal_support_exact
    assert certificate.sampled_spacelike_marginals_pass

    assert certificate.unitary_residual < 1.0e-12
    assert certificate.kraus_completeness_residual < 1.0e-12
    assert certificate.minimum_choi_eigenvalue > -1.0e-12
    assert certificate.output_trace_residual < 1.0e-12
    assert certificate.minimum_output_eigenvalue > -1.0e-12
    assert certificate.born_probability_sum_residual < 1.0e-12
    assert certificate.minimum_born_probability > -1.0e-12
    assert certificate.kraus_vs_direct_partial_trace_residual < 1.0e-12
    assert certificate.cptp_within_tolerance

    expected_system_gain = certificate.energy_gap * (0.4 + 0.16)
    assert certificate.final_system_energy - certificate.initial_system_energy == pytest.approx(
        expected_system_gain
    )
    assert certificate.final_battery_energy - certificate.initial_battery_energy == pytest.approx(
        -expected_system_gain
    )
    assert certificate.relative_energy_commutator_residual < 1.0e-12
    assert certificate.relative_total_energy_balance_residual < 1.0e-12
    assert certificate.energy_conserved_within_tolerance
    assert certificate.relative_reverse_transfer_identity_residual < 1.0e-12
    assert certificate.maximum_relative_branch_energy_residual < 1.0e-12
    assert certificate.expected_battery_energy_paid == pytest.approx(expected_system_gain)
    assert certificate.energy_resolved_instrument_within_tolerance
    assert not certificate.durable_physical_pointer_derived
    assert not certificate.covariant_matching_current_derived
    assert not certificate.record_to_gravity_source_derived


def test_battery_instrument_resolves_exclusive_transfer_receipts() -> None:
    certificate = certify_causal_quantum_domino(
        site_count=4,
        depth=2,
        theta=math.asin(math.sqrt(0.4)),
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
        energy_gap=3.0,
    )
    outcomes = {outcome.basis_label: outcome for outcome in certificate.battery_outcomes}

    assert outcomes["00"].probability == pytest.approx(0.16)
    assert outcomes["00"].energy_paid_to_system == pytest.approx(6.0)
    assert outcomes["00"].conditional_system_energy == pytest.approx(9.0)
    assert outcomes["01"].probability == pytest.approx(0.24)
    assert outcomes["01"].energy_paid_to_system == pytest.approx(3.0)
    assert outcomes["01"].conditional_system_energy == pytest.approx(6.0)
    assert outcomes["10"].probability == pytest.approx(0.0)
    assert outcomes["10"].conditional_system_energy is None
    assert outcomes["11"].probability == pytest.approx(0.6)
    assert outcomes["11"].energy_paid_to_system == pytest.approx(0.0)
    assert outcomes["11"].conditional_system_energy == pytest.approx(3.0)
    assert sum(outcome.probability for outcome in outcomes.values()) == pytest.approx(1.0)


def test_zero_angle_recovers_the_identity_channel() -> None:
    certificate = certify_causal_quantum_domino(
        site_count=4,
        depth=2,
        theta=0.0,
        lattice_spacing=1.0,
        clock_step=2.0,
        causal_speed=1.0,
    )

    assert certificate.trigger_probability == 0.0
    assert certificate.activation_probabilities == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert certificate.standard_limit_superoperator_residual < 1.0e-12
    assert certificate.final_system_energy == pytest.approx(certificate.initial_system_energy)
    assert certificate.final_battery_energy == pytest.approx(certificate.initial_battery_energy)
    assert certificate.expected_battery_energy_paid == pytest.approx(0.0)
    assert certificate.energy_resolved_instrument_within_tolerance


def test_continuous_time_domino_has_a_spacelike_early_arrival_tail() -> None:
    # a=c=1 이면 두 도약에는 이산 인과 시간 두 단위 이상이 필요하다.
    # 그럼에도 연속 시간 얼랑 사슬은 t=0.5 에 양의 가중치를 주므로,
    # 최근접 이웃 비율만으로는 엄격한 빛원뿔이 되지 않는다.
    probability = homogeneous_continuous_time_early_arrival_probability(
        rate_per_time=1.0,
        hops=2,
        elapsed_time=0.5,
    )

    assert probability == pytest.approx(1.0 - 1.5 * math.exp(-0.5))
    assert probability > 0.0


def test_continuous_time_tail_avoids_small_argument_cancellation() -> None:
    assert homogeneous_continuous_time_early_arrival_probability(1.0e-18, 1, 1.0) == pytest.approx(
        1.0e-18
    )
    assert homogeneous_continuous_time_early_arrival_probability(1.0e-9, 10, 1.0) > 0.0
    assert homogeneous_continuous_time_early_arrival_probability(5.0e-324, 1, 0.5) == 0.0


def test_single_site_partial_trace_handles_entanglement() -> None:
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    density = np.outer(bell, bell.conj())

    assert np.allclose(_single_site_reduced(density, 2, 0), 0.5 * np.eye(2))
    assert np.allclose(_single_site_reduced(density, 2, 1), 0.5 * np.eye(2))


def test_causal_timing_fails_closed() -> None:
    with pytest.raises(ValueError, match="causal timing"):
        certify_causal_quantum_domino(
            site_count=3,
            depth=1,
            theta=0.5,
            lattice_spacing=2.0,
            clock_step=0.49,
            causal_speed=4.0,
        )


def test_certificate_size_and_tolerance_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite certificate limit"):
        certify_causal_quantum_domino(
            site_count=6,
            depth=1,
            theta=0.5,
            lattice_spacing=1.0,
            clock_step=1.0,
            causal_speed=1.0,
        )
    with pytest.raises(ValueError, match="tolerance"):
        certify_causal_quantum_domino(
            site_count=3,
            depth=1,
            theta=0.5,
            lattice_spacing=1.0,
            clock_step=1.0,
            causal_speed=1.0,
            tolerance=1.0e-3,
        )


@pytest.mark.parametrize("theta", [-0.1, math.pi / 2.0 + 0.1, math.inf])
def test_trigger_angle_must_be_a_finite_nonredundant_probability_chart(theta: float) -> None:
    with pytest.raises(ValueError, match="theta"):
        certify_causal_quantum_domino(
            site_count=3,
            depth=1,
            theta=theta,
            lattice_spacing=1.0,
            clock_step=1.0,
            causal_speed=1.0,
        )


def _certificate(**overrides):
    arguments = dict(
        n_links=3,
        couplings=(0.8, -1.1, 0.6),
        field_mass=1.5,
        clock_scale=2.0,
        prep_mass_squared=0.7,
        battery_energy_per_cell=1.25,
        exchange_coupling=0.3,
        quartic_coupling=0.6,
    )
    arguments.update(overrides)
    return certify_autonomous_repeated_domino_obstruction(**arguments)


def test_open_chain_bound_is_tied_to_the_declared_normalization():
    certificate = _certificate()

    assert certificate.stability_bound_pass
    assert certificate.quartic_lower_bound_coefficient == pytest.approx(0.0)
    assert certificate.quartic_coupling == pytest.approx(
        2.0 * abs(certificate.exchange_coupling)
    )
    assert certificate.dimensions_closed
    assert not certificate.explicit_coordinate_time_switching_present
    assert certificate.local_coordinate_species_field_candidate_by_construction
    assert certificate.carrier_prep_stability_pass
    assert certificate.carrier_quadratic_minimum_mass_squared == pytest.approx(
        1.5**2 - 0.7
    )

    above_bound = _certificate(quartic_coupling=0.8)
    assert above_bound.quartic_lower_bound_coefficient > 0.0

    dimensionless_arguments = dict(certificate.dimensionless_core_arguments)
    assert "J_j Delta_tau" in dimensionless_arguments
    assert "mass dimension" in dimensionless_arguments["J_j Delta_tau"]
    assert "g Delta_tau" not in dimensionless_arguments


def test_below_bound_is_rejected_fail_closed():
    with pytest.raises(ValueError, match="lambda >= 2 \\|g\\|"):
        _certificate(quartic_coupling=0.599)
    with pytest.raises(ValueError, match="carrier preparation stability"):
        _certificate(prep_mass_squared=1.5**2)
    with pytest.raises(ValueError, match="carrier preparation stability"):
        _certificate(prep_mass_squared=1.5**2 + 0.1)


def test_endpoint_powers_and_taylor_coefficient_are_exact_for_the_path():
    certificate = _certificate()
    product = math.prod(certificate.couplings)

    assert certificate.finite_hamiltonian_hermitian
    assert certificate.hamiltonian_hermiticity_residual < 1.0e-14
    assert certificate.lower_order_endpoint_power_residual < 1.0e-14
    assert certificate.endpoint_order_n_value == pytest.approx(product, abs=1.0e-14)
    assert certificate.endpoint_taylor_coefficient == pytest.approx(
        (-1j) ** certificate.n_links * product / math.factorial(certificate.n_links),
        abs=1.0e-14,
    )
    assert certificate.analytic_coefficient_conditions_pass
    assert certificate.small_time_remainder_magnitude > 0.0
    # t=1e-3, N=3 에서 다음 허용 경로 기여는 O(t^5)이다.
    # 이는 점근적 수치 증인이지 정확한 전파자 주장이 아니다.
    assert abs(
        certificate.small_time_endpoint_amplitude - certificate.small_time_leading_term
    ) < 1.0e-12


def test_all_success_receipt_scales_as_capacity_not_expected_energy():
    certificate = _certificate(n_links=4, couplings=(0.2, 0.3, 0.4, 0.5))

    assert certificate.finite_all_success_resource_receipt
    assert certificate.all_success_initially_clean_battery_count == 4
    assert certificate.all_success_initially_clean_record_count == 4
    assert certificate.all_success_battery_energy == pytest.approx(4 * 1.25)


def test_claim_ceilings_stay_false():
    certificate = _certificate()

    assert not certificate.species_index_is_physical_spatial_distance
    assert not certificate.physical_lattice_or_worldtube_completion
    assert not certificate.coupled_clock_global_monotonicity_one_pass
    assert not certificate.exact_delayed_front_derived
    assert not certificate.projected_link_rates_from_action_derived
    assert not certificate.iterated_fresh_ancilla_cptp_instrument_derived
    assert not certificate.continuum_qft_microcausality_derived
    assert not certificate.operational_no_signalling_derived
    assert not certificate.gr_source_stress_matching_derived
    assert not certificate.unbounded_front_from_finite_resources_derived
    assert not certificate.durable_records_derived
    assert not certificate.cross_dataset_parameter_fixing_derived
    assert not certificate.independent_holdout_prediction_derived


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_links": 0}, "n_links"),
        ({"couplings": (0.2, 0.0, 0.4)}, "nonzero"),
        ({"field_mass": 0.0}, "field_mass"),
        ({"clock_scale": -1.0}, "clock_scale"),
        ({"prep_mass_squared": 0.0}, "prep_mass_squared"),
        ({"battery_energy_per_cell": 0.0}, "battery_energy_per_cell"),
        ({"couplings": (0.2, 0.3)}, "exactly n_links"),
    ],
)
def test_invalid_parameters_fail_closed(override, message):
    with pytest.raises(ValueError, match=message):
        _certificate(**override)


def test_cfl_one_support_and_first_detector_arrival_are_causal() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.support_violation == 0.0
    assert certificate.first_nonzero_detector_sample == certificate.expected_first_detector_sample == 4
    assert certificate.source_detector_chi[:4] == (0.0, 0.0, 0.0, 0.0)
    assert certificate.source_detector_chi[4] == pytest.approx(0.8)
    assert certificate.finite_lattice_causal_front_witness
    assert certificate.front_speed == pytest.approx(certificate.light_speed)


def test_probability_response_waits_for_arrival_then_differs() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.prearrival_probability_difference == 0.0
    assert certificate.source_probabilities[:4] == certificate.control_probabilities[:4]
    assert certificate.postarrival_probability_difference > 1.0e-6


def test_source_and_coupling_negative_controls_are_inert() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.source_off_probability_difference == 0.0
    assert certificate.coupling_off_probability_difference == 0.0
    source_off = certify_causal_probability_deformation_lattice(source_amplitude=0.0)
    coupling_off = certify_causal_probability_deformation_lattice(coupling_energy=0.0)
    assert source_off.source_probabilities == source_off.control_probabilities
    assert coupling_off.source_probabilities == coupling_off.control_probabilities


def test_local_channel_is_unitary_cptp_and_trace_preserving() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert certificate.local_unitary_residual < 1.0e-14
    assert certificate.local_trace_residual < 1.0e-14
    assert certificate.local_choi_minimum_eigenvalue >= -1.0e-13


def test_rn_separation_dimensions_and_false_claim_ceiling_are_explicit() -> None:
    certificate = certify_causal_probability_deformation_lattice()
    assert not certificate.rn_reweighting_used
    assert certificate.chi_dimensionless
    assert certificate.source_q_dimensionless
    assert certificate.continuum_source_s_length_power == -2
    assert certificate.coupling_g_is_energy
    assert certificate.dt_hamiltonian_over_hbar_dimensionless
    assert certificate.dimensions_pass
    assert not any((
        certificate.mass_to_q_derived,
        certificate.energy_current_or_backreaction_derived,
        certificate.probability_deformation_equals_attraction_derived,
        certificate.continuous_qft_microcausality_derived,
        certificate.gr_or_lensing_derived,
        certificate.repeated_measurement_or_physical_selection_derived,
        certificate.observational_holdout_derived,
        certificate.gates_5_to_8_closed,
        certificate.two_residuals_reduced,
        certificate.complexity_success,
    ))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lattice_spacing": 0.0}, "lattice_spacing"),
        ({"light_speed": 0.0}, "light_speed"),
        ({"omega": 0.0}, "omega"),
        ({"hbar": 0.0}, "hbar"),
        ({"detector_distance_cells": -1}, "detector_distance_cells"),
        ({"time_steps": 3}, "time_steps"),
        ({"grid_radius_cells": 5}, "grid_radius_cells"),
    ],
)
def test_horizon_and_numeric_contract_fails_closed(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        certify_causal_probability_deformation_lattice(**kwargs)
