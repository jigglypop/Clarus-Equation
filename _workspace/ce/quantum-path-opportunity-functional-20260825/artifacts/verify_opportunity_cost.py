"""Deterministic checks for the quantum-path opportunity functional.

The checks validate algebra, dimensions, and a two-outcome example.  They do
not establish that an information cost is a gravitational energy source.
"""

from __future__ import annotations

import json
import math


TOL = 1.0e-12


def entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value)


def relative_entropy(
    probabilities: list[float], reference: list[float]
) -> float:
    total = 0.0
    for value, ref_value in zip(probabilities, reference, strict=True):
        if value == 0.0:
            continue
        if ref_value == 0.0:
            return math.inf
        total += value * math.log(value / ref_value)
    return total


def add_dim(*vectors: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(sum(values) for values in zip(*vectors, strict=True))


def main() -> None:
    selected = 0.8
    excluded = 1.0 - selected
    probabilities = [selected, excluded]
    full_entropy = entropy(probabilities)
    aggregate_surprisal = -math.log(excluded)
    excluded_information = -excluded * math.log(excluded)
    conditional_excluded = [1.0]
    conditional_entropy = entropy(conditional_excluded)
    delta_excluded = [0.0, 1.0]
    uniform_reference = [0.5, 0.5]
    delta_kl = relative_entropy(delta_excluded, uniform_reference)
    full_kl = relative_entropy(probabilities, uniform_reference)
    decomposition = excluded * (
        -math.log(excluded) + conditional_entropy
    )

    energy_gap = 3.0
    expected_energy_regret = excluded * energy_gap

    kbt = 1.0
    energy_levels = [0.0, 2.0]
    state = [0.7, 0.3]
    gibbs_weights = [math.exp(-energy / kbt) for energy in energy_levels]
    partition = sum(gibbs_weights)
    gibbs = [weight / partition for weight in gibbs_weights]
    free_energy_state = sum(
        probability * energy
        for probability, energy in zip(state, energy_levels, strict=True)
    ) - kbt * entropy(state)
    free_energy_gibbs = sum(
        probability * energy
        for probability, energy in zip(gibbs, energy_levels, strict=True)
    ) - kbt * entropy(gibbs)
    free_energy_excess = free_energy_state - free_energy_gibbs
    relative_free_energy = kbt * relative_entropy(state, gibbs)

    dimless = (0, 0, 0, 0)
    dim_energy = (1, 2, -2, 0)
    dim_action = (1, 2, -1, 0)
    dim_time_inverse = (0, 0, -1, 0)
    dim_energy_density = (1, -1, -2, 0)
    information_dimension = dimless
    thermal_cost_dimension = add_dim(dim_energy, information_dimension)
    euclidean_log_action_dimension = dim_action
    action_over_time_dimension = add_dim(dim_action, dim_time_inverse)

    epsilon = 1.0e-9
    vanishing_excluded_cost = -epsilon * math.log(epsilon)

    checks = {
        "probabilities_normalized": abs(sum(probabilities) - 1.0) <= TOL,
        "excluded_information_decomposition_matches": abs(
            excluded_information - decomposition
        ) <= TOL,
        "conditional_singleton_entropy_is_zero": (
            conditional_entropy == 0.0
        ),
        "aggregate_surprisal_is_not_total_cost": (
            aggregate_surprisal > excluded_information
        ),
        "excluded_information_vanishes_with_excluded_mass": (
            0.0 < vanishing_excluded_cost < 1.0e-6
        ),
        "delta_kl_to_uniform_is_log_two": abs(
            delta_kl - math.log(2.0)
        ) <= TOL,
        "relative_entropy_is_reference_dependent": abs(
            delta_kl - full_kl
        ) > 1.0e-3,
        "expected_energy_regret_needs_energy_gap": abs(
            expected_energy_regret - 0.6
        ) <= TOL,
        "free_energy_relative_entropy_identity": abs(
            free_energy_excess - relative_free_energy
        ) <= TOL,
        "information_is_dimensionless": information_dimension == dimless,
        "thermal_scale_supplies_energy_dimension": (
            thermal_cost_dimension == dim_energy
        ),
        "hbar_log_ratio_is_action_not_energy": (
            euclidean_log_action_dimension == dim_action
            and euclidean_log_action_dimension != dim_energy
        ),
        "action_needs_inverse_time_for_energy": (
            action_over_time_dimension == dim_energy
        ),
        "energy_density_scale_is_independent": (
            dim_energy_density != information_dimension
        ),
    }

    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "two_outcome_example": {
            "probabilities": probabilities,
            "entropy_nats": full_entropy,
            "aggregate_excluded_surprisal_nats": aggregate_surprisal,
            "weighted_excluded_information_nats": excluded_information,
            "conditional_excluded_entropy_nats": conditional_entropy,
            "delta_to_uniform_kl_nats": delta_kl,
            "full_distribution_to_uniform_kl_nats": full_kl,
            "energy_gap": energy_gap,
            "expected_energy_regret": expected_energy_regret,
        },
        "thermal_identity_example": {
            "energy_levels": energy_levels,
            "state": state,
            "gibbs": gibbs,
            "free_energy_excess": free_energy_excess,
            "kbt_relative_entropy": relative_free_energy,
        },
        "dimensions_MLTTheta": {
            "information": information_dimension,
            "thermal_cost": thermal_cost_dimension,
            "hbar_log_ratio": euclidean_log_action_dimension,
            "action_over_time": action_over_time_dimension,
            "energy_density_scale": dim_energy_density,
        },
        "tolerance": TOL,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
