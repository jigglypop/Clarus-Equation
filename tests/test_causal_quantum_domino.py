from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal_quantum_domino import (
    _single_site_reduced,
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
    # With a=c=1, two hops need at least two units of discrete causal time.
    # The continuous-time Erlang chain nevertheless assigns positive weight at
    # t=0.5, so nearest-neighbour rates alone do not make a strict light cone.
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
