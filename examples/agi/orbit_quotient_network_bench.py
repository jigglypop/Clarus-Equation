from __future__ import annotations

import json

import numpy as np

from reality_stone.clarus.orbit_quotient_network import (
    DelayedEdge,
    DelayedOrbitNetwork,
    lift_orbit_trajectory,
    simulate_full,
    simulate_quotient,
    simulate_sparse_initial_deviation,
    translate_cells,
)


def main() -> None:
    network = DelayedOrbitNetwork(
        3,
        (0.05, -0.03, 0.02),
        (
            DelayedEdge(0, 0, -1, 1, 0.16),
            DelayedEdge(0, 1, 0, 2, -0.11),
            DelayedEdge(1, 0, 1, 1, 0.13),
            DelayedEdge(1, 2, 0, 3, 0.12),
            DelayedEdge(2, 1, -1, 2, 0.17),
            DelayedEdge(2, 2, 1, 1, 0.14),
        ),
    )
    initial = np.asarray((0.2, -0.1, 0.3))
    inputs = np.asarray(tuple((0.01 * t, -0.02, 0.015) for t in range(8)))
    quotient = simulate_quotient(network, initial, inputs)
    errors = {}
    for size in (32, 64, 128, 256):
        lifted = lift_orbit_trajectory(quotient, size)
        full = simulate_full(network, lifted[0], lift_orbit_trajectory(inputs, size))
        errors[str(size)] = float(np.max(np.abs(full - lifted)))
    sparse = simulate_sparse_initial_deviation(
        network, 64, initial, inputs, {(31, 0): 0.4}, active_budget=64
    )
    full_initial = lift_orbit_trajectory(initial, 64)
    full_initial[31, 0] += 0.4
    full = simulate_full(network, full_initial, lift_orbit_trajectory(inputs, 64))
    rng = np.random.default_rng(19)
    random_initial = rng.normal(0.0, 0.2, size=(32, 3))
    random_input = rng.normal(0.0, 0.03, size=(1, 32, 3))
    translated = simulate_full(
        network, translate_cells(random_initial, 5), translate_cells(random_input, 5)
    )
    translated_reference = translate_cells(
        simulate_full(network, random_initial, random_input), 5
    )
    metrics = {
        "schema": "clarus.dynamic-delayed-orbit-quotient.validation.v1",
        "cover_errors": errors,
        "maximum_cover_error": max(errors.values()),
        "sparse_reconstruction_error": float(
            np.max(np.abs(sparse.reconstructed - full))
        ),
        "translation_error": float(np.max(np.abs(translated - translated_reference))),
        "maximum_active_nodes": max(map(len, sparse.active_by_time)),
        "registered_cone_bound": network.orbit_count
        * (2 * network.maximum_shift * inputs.shape[0] + 1),
        "quotient_work_per_step": network.quotient_work,
        "small_gain": network.small_gain,
    }
    metrics["verdict"] = "GO" if (
        metrics["maximum_cover_error"] <= 1e-10
        and metrics["sparse_reconstruction_error"] <= 1e-10
        and metrics["translation_error"] <= 1e-10
        and metrics["maximum_active_nodes"] <= metrics["registered_cone_bound"]
        and metrics["small_gain"] < 1.0
    ) else "STOP"
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
