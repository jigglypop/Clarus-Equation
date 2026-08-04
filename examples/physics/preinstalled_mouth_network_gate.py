from __future__ import annotations

import math

import numpy as np

from reality_stone.clarus.preinstalled_mouth_network import (
    clock_synchronization_audit,
    endpoint_coverage_audit,
    network_chronology_audit,
    preinstalled_route_audit,
    realtime_chronology_interlock,
)


def main() -> None:
    light_year_m = 9.4607304725808e15
    positions = np.array([[0.0, 0.0, 0.0], [light_year_m, 0.0, 0.0]])
    route = preinstalled_route_audit(
        positions,
        np.array([[math.inf, 10.0], [10.0, math.inf]]),
        source=0,
        target=1,
        local_speed_fraction_c=0.1,
    )
    coverage = endpoint_coverage_audit(
        positions,
        np.array([light_year_m / 2.0, 0.0, 0.0]),
        tolerance_m=1.0,
    )
    chronology = network_chronology_audit(
        np.array([[math.inf, -2.0], [1.0, math.inf]])
    )
    synchronizable = clock_synchronization_audit(
        np.array([[math.inf, -2.0], [3.0, math.inf]]),
        future_margin_s=0.4,
    )
    impossible_sync = clock_synchronization_audit(
        np.array([[math.inf, -2.0], [1.0, math.inf]]),
        future_margin_s=0.0,
    )
    zero_margin_strict = clock_synchronization_audit(
        np.array([[math.inf, 0.0], [1.0, math.inf]]),
        future_margin_s=0.0,
    )
    interlock = realtime_chronology_interlock(
        np.array([[[math.inf, -2.0], [1.0, math.inf]]]),
        measurement_uncertainty_s=0.0,
        maximum_edge_drift_s_per_s=0.0,
        sample_interval_s=0.1,
        future_margin_s=0.01,
    )

    print("CE PREINSTALLED MOUTH NETWORK LOOP")
    print(" route", route.path)
    print(" travel s", route.network_travel_time_s)
    print(" effective shortcut", route.beats_exterior_light)
    print(" instantaneous", route.exactly_instantaneous)
    print(" midpoint covered", coverage.target_covered)
    print(" negative time cycle", chronology.negative_time_cycle_exists)
    print(" positive-cycle sync", synchronizable.synchronization_exists)
    print(" synchronized minimum edge s", synchronizable.minimum_synchronized_edge_s)
    print(" negative-cycle sync", impossible_sync.synchronization_exists)
    print(" zero-margin strict exists", zero_margin_strict.strict_graph_time_function_exists)
    print(" strict witness margin s", zero_margin_strict.strict_witness_margin_s)
    print(" interlock enabled edges", interlock.enabled_edge_counts)
    print(" interlock disabled edges", interlock.disabled_edge_counts)
    print(" network physics derived", route.preinstalled_network_physics_derived)


if __name__ == "__main__":
    main()
