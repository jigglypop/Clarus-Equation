from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.preinstalled_mouth_network import (
    clock_synchronization_audit,
    endpoint_coverage_audit,
    network_chronology_audit,
    preinstalled_route_audit,
    realtime_chronology_interlock,
)


LIGHT_YEAR_M = 9.4607304725808e15


def test_preinstalled_route_is_fast_but_not_instantaneous() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [LIGHT_YEAR_M, 0.0, 0.0]])
    lengths = np.array([[math.inf, 10.0], [10.0, math.inf]])
    audit = preinstalled_route_audit(
        positions,
        lengths,
        source=0,
        target=1,
        local_speed_fraction_c=0.1,
    )

    assert audit.path == (0, 1)
    assert audit.reachable
    assert audit.beats_exterior_light
    assert not audit.exactly_instantaneous
    assert audit.locally_subluminal
    assert not audit.remote_stress_creation_required
    assert not audit.preinstalled_network_physics_derived


def test_route_selector_chooses_lower_total_throat_time() -> None:
    positions = np.array([[0.0], [10.0], [20.0]])
    lengths = np.array(
        [
            [math.inf, 1.0, 10.0],
            [math.inf, math.inf, 1.0],
            [math.inf, math.inf, math.inf],
        ]
    )
    audit = preinstalled_route_audit(
        positions,
        lengths,
        source=0,
        target=2,
        local_speed_fraction_c=0.5,
    )

    assert audit.path == (0, 1, 2)


def test_preinstalled_mouths_do_not_cover_arbitrary_positions() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    audit = endpoint_coverage_audit(
        positions,
        np.array([5.0, 0.0, 0.0]),
        tolerance_m=1.0,
    )

    assert audit.nearest_distance_m == 5.0
    assert not audit.target_covered
    assert not audit.arbitrary_position_reachable


def test_negative_coordinate_time_cycle_is_detected() -> None:
    edges = np.array([[math.inf, -2.0], [1.0, math.inf]])
    audit = network_chronology_audit(edges)

    assert audit.negative_time_cycle_exists
    assert not audit.chronology_safe_in_control_graph
    assert not audit.global_time_function_derived


def test_nonnegative_cycle_passes_only_the_finite_graph_gate() -> None:
    edges = np.array([[math.inf, 2.0], [1.0, math.inf]])
    audit = network_chronology_audit(edges)

    assert not audit.negative_time_cycle_exists
    assert audit.chronology_safe_in_control_graph
    assert not audit.global_time_function_derived


def test_route_rejects_luminal_or_superluminal_local_speed() -> None:
    with pytest.raises(ValueError):
        preinstalled_route_audit(
            np.array([[0.0], [1.0]]),
            np.array([[math.inf, 1.0], [1.0, math.inf]]),
            source=0,
            target=1,
            local_speed_fraction_c=1.0,
        )


def test_clock_offsets_make_edges_strictly_future_directed_when_cycle_allows() -> None:
    edges = np.array([[math.inf, -2.0], [3.0, math.inf]])
    audit = clock_synchronization_audit(edges, future_margin_s=0.4)

    assert audit.synchronization_exists
    assert audit.minimum_synchronized_edge_s >= 0.4 - 1e-12
    assert audit.strict_graph_time_function_exists
    assert audit.cycle_sums_are_gauge_invariant
    assert not audit.spacetime_chronology_protection_derived


def test_requested_margin_cannot_exceed_cycle_budget() -> None:
    edges = np.array([[math.inf, -2.0], [3.0, math.inf]])
    audit = clock_synchronization_audit(edges, future_margin_s=0.6)

    assert not audit.synchronization_exists
    assert np.all(np.isnan(audit.clock_offsets_s))
    assert not audit.strict_graph_time_function_exists


def test_clock_relabelling_cannot_remove_a_negative_time_cycle() -> None:
    edges = np.array([[math.inf, -2.0], [1.0, math.inf]])
    audit = clock_synchronization_audit(edges, future_margin_s=0.0)

    assert not audit.synchronization_exists
    assert audit.cycle_sums_are_gauge_invariant


def test_zero_time_cycle_fails_strict_future_margin() -> None:
    edges = np.array([[math.inf, -1.0], [1.0, math.inf]])
    nonstrict = clock_synchronization_audit(edges, future_margin_s=0.0)
    strict = clock_synchronization_audit(edges, future_margin_s=1e-3)

    assert nonstrict.synchronization_exists
    assert not nonstrict.strict_graph_time_function_exists
    assert not strict.synchronization_exists


def test_realtime_interlock_cuts_an_edge_that_would_close_a_negative_cycle() -> None:
    frames = np.array([[[math.inf, -2.0], [1.0, math.inf]]])
    audit = realtime_chronology_interlock(
        frames,
        measurement_uncertainty_s=0.0,
        maximum_edge_drift_s_per_s=0.0,
        sample_interval_s=0.1,
        future_margin_s=0.01,
    )

    assert audit.enabled_edge_counts == (1,)
    assert audit.disabled_edge_counts == (1,)
    assert audit.every_enabled_frame_synchronizable
    assert audit.enabled_edges[0, 0, 1]
    assert not audit.enabled_edges[0, 1, 0]


def test_uncertainty_and_drift_can_force_conservative_edge_shutdown() -> None:
    frames = np.array(
        [
            [[math.inf, 0.7], [0.7, math.inf]],
            [[math.inf, 0.4], [0.4, math.inf]],
        ]
    )
    audit = realtime_chronology_interlock(
        frames,
        measurement_uncertainty_s=0.2,
        maximum_edge_drift_s_per_s=0.3,
        sample_interval_s=1.0,
        future_margin_s=0.1,
    )

    assert math.isclose(audit.robust_lower_time_edges_s[0, 0, 1], 0.2)
    assert math.isclose(audit.robust_lower_time_edges_s[1, 0, 1], -0.1)
    assert audit.enabled_edge_counts == (2, 1)
    assert audit.disabled_edge_counts == (0, 1)


def test_sensor_fault_fails_closed() -> None:
    frames = np.array(
        [
            [[math.inf, 1.0], [1.0, math.inf]],
            [[math.inf, math.nan], [1.0, math.inf]],
        ]
    )
    audit = realtime_chronology_interlock(
        frames,
        measurement_uncertainty_s=0.0,
        maximum_edge_drift_s_per_s=0.0,
        sample_interval_s=0.1,
        future_margin_s=0.1,
    )

    assert audit.sensor_fault_frames == (1,)
    assert audit.enabled_edge_counts == (2, 0)
    assert audit.fail_closed_on_sensor_fault
    assert not audit.continuous_spacetime_protection_derived


def test_interlock_rejects_zero_sample_interval() -> None:
    with pytest.raises(ValueError):
        realtime_chronology_interlock(
            np.array([[[math.inf]]]),
            measurement_uncertainty_s=0.0,
            maximum_edge_drift_s_per_s=0.0,
            sample_interval_s=0.0,
            future_margin_s=0.0,
        )
