"""Finite gates for routing through a preinstalled wormhole-mouth network.

Preinstallation removes the need to create stress energy at a newly selected
remote point.  It does not provide arbitrary destinations, zero traversal time,
or chronology protection.  The functions below audit these three boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np

from .spatial_folding import SPEED_OF_LIGHT_M_S


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class MouthRouteAudit:
    path: tuple[int, ...]
    reachable: bool
    exterior_distance_m: float
    exterior_light_time_s: float
    network_travel_time_s: float
    beats_exterior_light: bool
    exactly_instantaneous: bool
    locally_subluminal: bool
    remote_stress_creation_required: bool
    preinstalled_network_physics_derived: bool


@dataclass(frozen=True)
class EndpointCoverageAudit:
    target_position_m: np.ndarray
    nearest_mouth: int
    nearest_distance_m: float
    tolerance_m: float
    target_covered: bool
    arbitrary_position_reachable: bool


@dataclass(frozen=True)
class NetworkChronologyAudit:
    node_count: int
    edge_count: int
    negative_time_cycle_exists: bool
    chronology_safe_in_control_graph: bool
    global_time_function_derived: bool


@dataclass(frozen=True)
class ClockSynchronizationAudit:
    requested_future_margin_s: float
    clock_offsets_s: np.ndarray
    synchronized_time_edges_s: np.ndarray
    synchronization_exists: bool
    minimum_synchronized_edge_s: float
    cycle_sums_are_gauge_invariant: bool
    strict_graph_time_function_exists: bool
    spacetime_chronology_protection_derived: bool


@dataclass(frozen=True)
class RealtimeInterlockAudit:
    robust_lower_time_edges_s: np.ndarray
    enabled_edges: np.ndarray
    enabled_edge_counts: tuple[int, ...]
    disabled_edge_counts: tuple[int, ...]
    sensor_fault_frames: tuple[int, ...]
    every_enabled_frame_synchronizable: bool
    fail_closed_on_sensor_fault: bool
    uncertainty_and_drift_bounded: bool
    continuous_spacetime_protection_derived: bool


def _positions(value: ArrayLike) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] < 1:
        raise ValueError("mouth_positions_m must have shape (mouths, dimensions)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("mouth_positions_m must be finite")
    return positions


def preinstalled_route_audit(
    mouth_positions_m: ArrayLike,
    throat_lengths_m: ArrayLike,
    *,
    source: int,
    target: int,
    local_speed_fraction_c: float,
    switch_latency_s: float = 0.0,
) -> MouthRouteAudit:
    """Find the minimum-time route through fixed mouths using Dijkstra's rule."""

    positions = _positions(mouth_positions_m)
    lengths = np.asarray(throat_lengths_m, dtype=float)
    count = positions.shape[0]
    if lengths.shape != (count, count):
        raise ValueError("throat_lengths_m must be a square mouth-by-mouth matrix")
    if np.any(lengths < 0.0) or np.any(np.isnan(lengths)):
        raise ValueError("throat lengths must be non-negative or infinity")
    if not 0 <= source < count or not 0 <= target < count:
        raise ValueError("source and target must index a mouth")
    beta = float(local_speed_fraction_c)
    latency = float(switch_latency_s)
    if not math.isfinite(beta) or not 0.0 < beta < 1.0:
        raise ValueError("local_speed_fraction_c must lie strictly between zero and one")
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("switch_latency_s must be finite and non-negative")

    distances = [math.inf] * count
    predecessors: list[int | None] = [None] * count
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        elapsed, node = heapq.heappop(queue)
        if elapsed != distances[node]:
            continue
        for neighbor in range(count):
            length = float(lengths[node, neighbor])
            if neighbor == node or not math.isfinite(length):
                continue
            candidate = elapsed + length / (beta * SPEED_OF_LIGHT_M_S) + latency
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    reachable = math.isfinite(distances[target])
    path: tuple[int, ...] = ()
    if reachable:
        reversed_path = [target]
        while reversed_path[-1] != source:
            previous = predecessors[reversed_path[-1]]
            if previous is None:
                raise RuntimeError("route reconstruction failed")
            reversed_path.append(previous)
        path = tuple(reversed(reversed_path))

    exterior = float(np.linalg.norm(positions[target] - positions[source]))
    exterior_time = exterior / SPEED_OF_LIGHT_M_S
    network_time = distances[target]
    return MouthRouteAudit(
        path=path,
        reachable=reachable,
        exterior_distance_m=exterior,
        exterior_light_time_s=exterior_time,
        network_travel_time_s=network_time,
        beats_exterior_light=reachable and network_time < exterior_time,
        exactly_instantaneous=reachable and network_time == 0.0,
        locally_subluminal=beta < 1.0,
        remote_stress_creation_required=False,
        preinstalled_network_physics_derived=False,
    )


def endpoint_coverage_audit(
    mouth_positions_m: ArrayLike,
    target_position_m: ArrayLike,
    *,
    tolerance_m: float,
) -> EndpointCoverageAudit:
    """Check whether an arbitrary requested point lies near an installed mouth."""

    positions = _positions(mouth_positions_m)
    target = np.asarray(target_position_m, dtype=float)
    if target.shape != (positions.shape[1],) or not np.all(np.isfinite(target)):
        raise ValueError("target_position_m must match the mouth coordinate dimension")
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_m must be finite and non-negative")

    distances = np.linalg.norm(positions - target, axis=1)
    nearest = int(np.argmin(distances))
    nearest_distance = float(distances[nearest])
    return EndpointCoverageAudit(
        target_position_m=target,
        nearest_mouth=nearest,
        nearest_distance_m=nearest_distance,
        tolerance_m=tolerance,
        target_covered=nearest_distance <= tolerance,
        arbitrary_position_reachable=False,
    )


def network_chronology_audit(coordinate_time_edges_s: ArrayLike) -> NetworkChronologyAudit:
    """Detect a negative elapsed-coordinate-time cycle with Bellman--Ford.

    Finite matrix entries are directed-edge elapsed coordinate times, including
    throat traversal and any mouth clock offset.  ``inf`` denotes no edge.
    A negative cycle is a discrete control-model analogue of a chronology loop.
    """

    edges = np.asarray(coordinate_time_edges_s, dtype=float)
    if edges.ndim != 2 or edges.shape[0] < 1 or edges.shape[0] != edges.shape[1]:
        raise ValueError("coordinate_time_edges_s must be a non-empty square matrix")
    if np.any(np.isnan(edges)):
        raise ValueError("coordinate_time_edges_s must not contain NaN")

    count = edges.shape[0]
    directed = [
        (source, target, float(edges[source, target]))
        for source in range(count)
        for target in range(count)
        if source != target and math.isfinite(float(edges[source, target]))
    ]
    distance = [0.0] * count
    negative_cycle = False
    for iteration in range(count):
        changed = False
        for source, target, elapsed in directed:
            if distance[source] + elapsed < distance[target]:
                distance[target] = distance[source] + elapsed
                changed = True
                if iteration == count - 1:
                    negative_cycle = True
        if not changed:
            break

    return NetworkChronologyAudit(
        node_count=count,
        edge_count=len(directed),
        negative_time_cycle_exists=negative_cycle,
        chronology_safe_in_control_graph=not negative_cycle,
        global_time_function_derived=False,
    )


def clock_synchronization_audit(
    coordinate_time_edges_s: ArrayLike,
    *,
    future_margin_s: float,
) -> ClockSynchronizationAudit:
    """Find mouth clock offsets making every enabled edge future-directed.

    For original edge time ``w_ij`` and mouth offsets ``s_i``, the relabelled
    time is ``w'_ij = w_ij + s_j - s_i``.  Requiring ``w'_ij >= epsilon`` is a
    system of difference constraints.  A Bellman--Ford feasibility pass either
    returns offsets or proves that the requested margin conflicts with a cycle.
    Clock relabelling telescopes around cycles and therefore cannot change their
    total elapsed time.
    """

    edges = np.asarray(coordinate_time_edges_s, dtype=float)
    if edges.ndim != 2 or edges.shape[0] < 1 or edges.shape[0] != edges.shape[1]:
        raise ValueError("coordinate_time_edges_s must be a non-empty square matrix")
    if np.any(np.isnan(edges)):
        raise ValueError("coordinate_time_edges_s must not contain NaN")
    margin = float(future_margin_s)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("future_margin_s must be finite and non-negative")

    count = edges.shape[0]
    original = [
        (source, target, float(edges[source, target]))
        for source in range(count)
        for target in range(count)
        if source != target and math.isfinite(float(edges[source, target]))
    ]
    # s_i <= s_j + w_ij - margin, represented as j -> i constraints.
    constraints = [
        (target, source, elapsed - margin) for source, target, elapsed in original
    ]
    offsets = np.zeros(count, dtype=float)
    feasible = True
    for iteration in range(count):
        changed = False
        for source, target, bound in constraints:
            candidate = offsets[source] + bound
            if candidate < offsets[target]:
                offsets[target] = candidate
                changed = True
                if iteration == count - 1:
                    feasible = False
        if not changed:
            break

    synchronized = np.full_like(edges, math.inf)
    minimum = math.inf
    if feasible:
        for source, target, elapsed in original:
            adjusted = elapsed + offsets[target] - offsets[source]
            synchronized[source, target] = adjusted
            minimum = min(minimum, adjusted)
    else:
        offsets = np.full(count, math.nan)

    strict = feasible and bool(original) and minimum > 0.0
    return ClockSynchronizationAudit(
        requested_future_margin_s=margin,
        clock_offsets_s=offsets,
        synchronized_time_edges_s=synchronized,
        synchronization_exists=feasible,
        minimum_synchronized_edge_s=minimum,
        cycle_sums_are_gauge_invariant=True,
        strict_graph_time_function_exists=strict,
        spacetime_chronology_protection_derived=False,
    )


def realtime_chronology_interlock(
    measured_time_edges_s: ArrayLike,
    *,
    measurement_uncertainty_s: float,
    maximum_edge_drift_s_per_s: float,
    sample_interval_s: float,
    future_margin_s: float,
) -> RealtimeInterlockAudit:
    """Greedily enable only edges that preserve robust clock feasibility.

    The input shape is ``(frames, mouths, mouths)``.  At each frame the lower
    bound used by the interlock is

    ``measured - uncertainty - maximum_drift_rate * sample_interval``.

    Candidate edges are considered in deterministic row-major order.  An edge
    is rejected if adding it would make the clock difference constraints
    infeasible.  This policy is safety-oriented and not a maximum-throughput or
    minimum-cut optimizer.  A NaN sensor frame disables every edge in that frame.
    """

    measured = np.asarray(measured_time_edges_s, dtype=float)
    if (
        measured.ndim != 3
        or measured.shape[0] < 1
        or measured.shape[1] < 1
        or measured.shape[1] != measured.shape[2]
    ):
        raise ValueError("measured_time_edges_s must have shape (frames, mouths, mouths)")
    uncertainty = float(measurement_uncertainty_s)
    drift_rate = float(maximum_edge_drift_s_per_s)
    interval = float(sample_interval_s)
    margin = float(future_margin_s)
    for value, name in (
        (uncertainty, "measurement_uncertainty_s"),
        (drift_rate, "maximum_edge_drift_s_per_s"),
        (interval, "sample_interval_s"),
        (margin, "future_margin_s"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if interval == 0.0:
        raise ValueError("sample_interval_s must be positive")

    robust = measured - uncertainty - drift_rate * interval
    enabled = np.zeros(measured.shape, dtype=bool)
    enabled_counts: list[int] = []
    disabled_counts: list[int] = []
    fault_frames: list[int] = []
    all_feasible = True
    mouth_count = measured.shape[1]

    for frame in range(measured.shape[0]):
        frame_values = measured[frame]
        if np.any(np.isnan(frame_values)):
            fault_frames.append(frame)
            enabled_counts.append(0)
            disabled_counts.append(
                int(np.count_nonzero(np.isfinite(frame_values) & ~np.eye(mouth_count, dtype=bool)))
            )
            continue

        candidate_edges = [
            (source, target)
            for source in range(mouth_count)
            for target in range(mouth_count)
            if source != target and math.isfinite(float(frame_values[source, target]))
        ]
        active = np.full((mouth_count, mouth_count), math.inf)
        for source, target in candidate_edges:
            active[source, target] = robust[frame, source, target]
            trial = clock_synchronization_audit(active, future_margin_s=margin)
            if trial.synchronization_exists:
                enabled[frame, source, target] = True
            else:
                active[source, target] = math.inf

        final = clock_synchronization_audit(active, future_margin_s=margin)
        all_feasible = all_feasible and final.synchronization_exists
        count = int(np.count_nonzero(enabled[frame]))
        enabled_counts.append(count)
        disabled_counts.append(len(candidate_edges) - count)

    return RealtimeInterlockAudit(
        robust_lower_time_edges_s=robust,
        enabled_edges=enabled,
        enabled_edge_counts=tuple(enabled_counts),
        disabled_edge_counts=tuple(disabled_counts),
        sensor_fault_frames=tuple(fault_frames),
        every_enabled_frame_synchronizable=all_feasible,
        fail_closed_on_sensor_fault=all(
            not np.any(enabled[frame]) for frame in fault_frames
        ),
        uncertainty_and_drift_bounded=True,
        continuous_spacetime_protection_derived=False,
    )
