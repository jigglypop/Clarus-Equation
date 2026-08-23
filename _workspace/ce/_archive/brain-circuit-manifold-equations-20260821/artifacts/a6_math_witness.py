"""Deterministic math witnesses for the A6-P/A6-C equation family.

This script opens no empirical response data.  It checks only analytic
identities and counterexamples frozen in the research contract.
"""

from __future__ import annotations

import json
import math

import numpy as np


def sech_sq(value: float) -> float:
    return 1.0 / math.cosh(value) ** 2


def tanh_second(value: float) -> float:
    return -2.0 * math.tanh(value) * sech_sq(value)


def scalar_two_step_metric(weight: float, initial: float) -> float:
    first_drive = weight * initial
    first_state = math.tanh(first_drive)
    second_drive = weight * first_state
    first_jacobian = sech_sq(first_drive) * weight
    second_jacobian = sech_sq(second_drive) * weight
    endpoint_jacobian = second_jacobian * first_jacobian
    return endpoint_jacobian**2


def scalar_two_step_metric_response(
    weight: float, circuit_strength: float, initial: float
) -> tuple[float, float]:
    first_drive = weight * initial
    first_state = math.tanh(first_drive)
    first_gain = sech_sq(first_drive)
    first_jacobian = first_gain * weight

    first_drive_dot = circuit_strength * initial
    first_state_dot = first_gain * first_drive_dot
    first_jacobian_dot = (
        tanh_second(first_drive) * first_drive_dot * weight
        + first_gain * circuit_strength
    )

    second_drive = weight * first_state
    second_gain = sech_sq(second_drive)
    second_jacobian = second_gain * weight

    second_drive_dot = circuit_strength * first_state + weight * first_state_dot
    second_jacobian_dot = (
        tanh_second(second_drive) * second_drive_dot * weight
        + second_gain * circuit_strength
    )

    endpoint_jacobian = second_jacobian * first_jacobian
    endpoint_jacobian_dot = (
        second_jacobian_dot * first_jacobian
        + second_jacobian * first_jacobian_dot
    )
    metric = endpoint_jacobian**2
    metric_dot = 2.0 * endpoint_jacobian * endpoint_jacobian_dot
    return metric, metric_dot


def main() -> None:
    # P0 witness: a state-dependent efficacy needs its derivative.
    activity = 0.5
    efficacy = (activity + 1.0) / 2.0
    drive = efficacy * activity
    gain = sech_sq(drive)
    frozen_efficacy_jacobian = gain * efficacy
    true_jacobian = gain * (efficacy + activity * 0.5)
    assert math.isclose(frozen_efficacy_jacobian, 0.6536849813, abs_tol=1e-10)
    assert math.isclose(true_jacobian, 0.8715799750, abs_tol=1e-10)
    assert not math.isclose(frozen_efficacy_jacobian, true_jacobian)

    # Passive pullback: anisotropic stretch can preserve metric volume.
    endpoint_jacobian = np.diag([2.0, 0.5])
    reference_metric = np.eye(2)
    pullback = endpoint_jacobian.T @ reference_metric @ endpoint_jacobian
    np.testing.assert_allclose(pullback, np.diag([4.0, 0.25]))
    stretches = np.sqrt(np.linalg.eigvalsh(pullback))
    np.testing.assert_allclose(stretches, [0.5, 2.0])
    log_volume_change = 0.5 * math.log(float(np.linalg.det(pullback)))
    assert math.isclose(log_volume_change, 0.0, abs_tol=1e-12)

    rank_loss_jacobian = np.diag([1.0, 0.0])
    rank_loss_metric = rank_loss_jacobian.T @ rank_loss_jacobian
    assert np.linalg.matrix_rank(rank_loss_metric) == 1
    assert np.linalg.eigvalsh(rank_loss_metric)[0] == 0.0

    # Nonnormal two-tick reachability witness.
    transition = np.array([[0.9, 10.0], [0.0, 0.9]])
    actuator = np.array([[0.0], [1.0]])
    gramian = transition @ actuator @ actuator.T @ transition.T + actuator @ actuator.T
    expected_gramian = np.array([[100.0, 9.0], [9.0, 1.81]])
    np.testing.assert_allclose(gramian, expected_gramian, atol=1e-12)
    target = np.array([1.0, 0.0])
    target_energy = float(target @ np.linalg.inv(gramian) @ target)
    assert math.isclose(target_energy, 0.0181, abs_tol=1e-12)

    # Total circuit-response chain rule versus central finite difference.
    weight = 0.7
    circuit_strength = 0.3
    initial = 0.4
    metric, analytic_metric_dot = scalar_two_step_metric_response(
        weight, circuit_strength, initial
    )
    step = 1e-6
    finite_difference_metric_dot = (
        scalar_two_step_metric(weight + step * circuit_strength, initial)
        - scalar_two_step_metric(weight - step * circuit_strength, initial)
    ) / (2.0 * step)
    assert math.isclose(metric, scalar_two_step_metric(weight, initial), abs_tol=1e-14)
    assert math.isclose(
        analytic_metric_dot,
        finite_difference_metric_dot,
        rel_tol=2e-9,
        abs_tol=1e-10,
    )

    # Full-rank scalar reachability-energy derivative.
    scalar_transition = 0.8
    scalar_transition_dot = 0.25
    scalar_gramian = 1.0 + scalar_transition**2
    scalar_gramian_dot = 2.0 * scalar_transition * scalar_transition_dot
    analytic_energy_dot = -scalar_gramian_dot / scalar_gramian**2
    finite_difference_energy_dot = (
        1.0 / (1.0 + (scalar_transition + step * scalar_transition_dot) ** 2)
        - 1.0 / (1.0 + (scalar_transition - step * scalar_transition_dot) ** 2)
    ) / (2.0 * step)
    assert math.isclose(
        analytic_energy_dot,
        finite_difference_energy_dot,
        rel_tol=2e-9,
        abs_tol=1e-10,
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "activity_dependent_efficacy": {
                    "frozen_jacobian": frozen_efficacy_jacobian,
                    "true_jacobian": true_jacobian,
                },
                "passive_pullback": {
                    "principal_stretches": stretches.tolist(),
                    "log_volume_change": log_volume_change,
                    "rank_loss_rank": int(np.linalg.matrix_rank(rank_loss_metric)),
                },
                "nonnormal_reachability": {
                    "gramian": gramian.tolist(),
                    "energy_e1": target_energy,
                },
                "circuit_response": {
                    "analytic_metric_derivative": analytic_metric_dot,
                    "finite_difference_metric_derivative": finite_difference_metric_dot,
                    "analytic_energy_derivative": analytic_energy_dot,
                    "finite_difference_energy_derivative": finite_difference_energy_dot,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
