"""Independent killing fixtures for the V15 scored proof claims F1--F5."""

from __future__ import annotations

import json
import math

import numpy as np

from reality_stone.clarus.unified_metric import (
    UnifiedMetricCore,
    affine_chart_change,
)


TOLERANCE = 1.0e-10


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def chain_fixture() -> tuple[UnifiedMetricCore, object, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [2.0, 0.7, 0.3],
            [2.8, 1.2, -0.2],
            [4.0, 1.1, 0.5],
        ],
        dtype=np.float64,
    )
    adjacency = np.zeros((5, 5), dtype=np.float64)
    for node in range(4):
        adjacency[node, node + 1] = 1.0
        adjacency[node + 1, node] = 1.0
    factors = np.array(
        [
            [[1.0, 0.2, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.8]],
            [[1.2, -0.1, 0.2], [0.1, 0.8, 0.0], [0.0, 0.2, 1.0]],
            [[0.8, 0.3, 0.1], [0.0, 1.1, -0.2], [0.2, 0.0, 0.9]],
            [[1.1, 0.0, -0.1], [0.2, 0.9, 0.3], [0.0, 0.1, 1.2]],
            [[0.9, -0.2, 0.0], [0.1, 1.0, 0.2], [0.3, 0.0, 1.1]],
        ],
        dtype=np.float64,
    )
    metric = np.array(
        [factor.T @ factor + 0.4 * np.eye(3) for factor in factors],
        dtype=np.float64,
    )
    core = UnifiedMetricCore(points, adjacency)
    return core, core.make_state(metric), metric


def verify_f1() -> dict[str, float | bool]:
    core, state, metric = chain_fixture()
    jacobian = np.array(
        [[1.7, 0.4, -0.2], [-0.3, 1.2, 0.5], [0.2, -0.1, 0.9]],
        dtype=np.float64,
    )
    offset = np.array([2.0, -3.0, 0.75], dtype=np.float64)
    points_y, metric_y = affine_chart_change(
        core.points,
        metric,
        jacobian,
        offset,
    )
    core_y = UnifiedMetricCore(points_y, core.adjacency_mask.astype(np.float64))
    state_y = core_y.make_state(metric_y)

    vector = np.array([0.31, -0.72, 0.18], dtype=np.float64)
    local_x = core.local_length_squared(state, 2, vector)
    local_y = core_y.local_length_squared(state_y, 2, jacobian @ vector)
    edges_x = core.edge_lengths(state)
    edges_y = core_y.edge_lengths(state_y)
    edge_mask = np.isfinite(edges_x)
    edge_error = max(
        relative_error(float(left), float(right))
        for left, right in zip(edges_x[edge_mask], edges_y[edge_mask], strict=True)
    )
    path_x = core.shortest_path(state, 0, 4)
    path_y = core_y.shortest_path(state_y, 0, 4)
    local_error = relative_error(local_x, local_y)
    path_error = relative_error(path_x.cost, path_y.cost)
    maximum = max(local_error, edge_error, path_error)
    assert maximum <= TOLERANCE
    return {
        "local_relative_error": local_error,
        "edge_max_relative_error": edge_error,
        "path_relative_error": path_error,
        "maximum_relative_error": maximum,
        "pass": True,
    }


def verify_f2() -> dict[str, float | bool]:
    points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    raw_metric = np.repeat(np.diag([9.0, 1.0])[None, :, :], 2, axis=0)
    jacobian = np.diag([3.0, 1.0])

    core_x = UnifiedMetricCore(points, adjacency)
    projected_x = core_x.project_metric(raw_metric)
    points_y, transported_raw = affine_chart_change(points, raw_metric, jacobian)
    core_y = UnifiedMetricCore(points_y, adjacency)
    project_after_transport = np.asarray(
        core_y.project_metric(transported_raw).metric,
        dtype=np.float64,
    )
    _, transport_after_project = affine_chart_change(
        points,
        np.asarray(projected_x.metric, dtype=np.float64),
        jacobian,
    )
    defect = float(np.max(np.abs(project_after_transport - transport_after_project)))
    assert defect > 1.0e-3
    return {
        "max_absolute_covariance_defect": defect,
        "expected_exact_defect": 5.0 / 9.0,
        "pass": True,
    }


def verify_f3() -> dict[str, float | bool]:
    core, state, _ = chain_fixture()
    forward = core.shortest_path(state, 0, 4)
    backward = core.shortest_path(state, 4, 0)
    error = relative_error(forward.cost, backward.cost)
    assert backward.nodes == tuple(reversed(forward.nodes))
    assert error <= 1.0e-12
    return {
        "forward_cost": forward.cost,
        "backward_cost": backward.cost,
        "relative_error": error,
        "pass": True,
    }


def verify_f4() -> dict[str, object]:
    points = np.array(
        [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    adjacency = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    core = UnifiedMetricCore(points, adjacency)
    readout = core.minimum_cost_targets(core.identity_state(), 0, [2, 1])
    assert readout.minimizers == (1, 2)
    assert not readout.unique
    return {
        "costs": [[node, cost] for node, cost in readout.costs],
        "minimizers": list(readout.minimizers),
        "unique": readout.unique,
        "pass": True,
    }


def verify_f5() -> dict[str, float | bool]:
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    endpoint_metric = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    core = UnifiedMetricCore(points, adjacency)
    finite_edge = float(core.edge_lengths(core.make_state(endpoint_metric))[0, 1])

    sample_count = 200_000
    midpoint_x = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    length_flat = 1.0
    length_bulged = float(np.mean(1.0 + np.sin(math.pi * midpoint_x) ** 2))
    difference = abs(length_bulged - length_flat)
    assert finite_edge == 1.0
    assert difference > 1.0e-2
    return {
        "finite_endpoint_edge_cost_for_both": finite_edge,
        "flat_continuum_length": length_flat,
        "bulged_continuum_length_midpoint": length_bulged,
        "exact_continuum_difference": 0.5,
        "reproduced_difference": difference,
        "pass": True,
    }


def main() -> None:
    results = {
        "F1": verify_f1(),
        "F2": verify_f2(),
        "F3": verify_f3(),
        "F4": verify_f4(),
        "F5": verify_f5(),
    }
    results["proof_fixture_score"] = f"{sum(bool(result['pass']) for result in results.values())}/5"
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
