"""Adversarial finite-input probes for the frozen V15 implementation."""

from __future__ import annotations

import json
import math
import queue
import threading
import warnings

import numpy as np

from reality_stone.clarus.unified_metric import (
    UnifiedMetricCore,
    affine_chart_change,
)


def exception_name(callback: object) -> str | None:
    try:
        callback()  # type: ignore[operator]
    except BaseException as error:  # pragma: no cover - result is reported
        return type(error).__name__
    return None


def unique_chain_scale_probe() -> dict[str, object]:
    results: queue.Queue[object] = queue.Queue()

    def worker() -> None:
        points = np.array([[0.0, 0.0], [1.0e-16, 0.0], [2.0e-16, 0.0]])
        adjacency = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        core = UnifiedMetricCore(points, adjacency)
        path = core.shortest_path(core.identity_state(), 2, 0)
        results.put({"nodes": list(path.nodes), "cost": path.cost})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=1.0)
    if thread.is_alive():
        return {
            "terminated_within_one_second": False,
            "expected_unique_path": [2, 1, 0],
            "expected_cost": 2.0e-16,
            "pass": False,
        }
    outcome = results.get_nowait()
    passed = outcome == {"nodes": [2, 1, 0], "cost": 2.0e-16}
    return {"terminated_within_one_second": True, "outcome": outcome, "pass": passed}


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    with np.errstate(all="ignore"):
        tiny_points = np.array([[0.0, 0.0], [1.0e-16, 0.0], [2.0e-16, 0.0]])
        complete = np.ones((3, 3), dtype=np.float64) - np.eye(3)
        tiny = UnifiedMetricCore(tiny_points, complete)
        tiny_state = tiny.identity_state()
        tiny_goal = tiny.minimum_cost_targets(tiny_state, 2, [0, 1])
        source_self = tiny.shortest_path(tiny_state, 2, 2)

        ordinary_points = np.array([[0.0, 0.0], [1.0, 0.0]])
        pair = np.array([[0.0, 1.0], [1.0, 0.0]])
        ordinary = UnifiedMetricCore(ordinary_points, pair)
        ordinary_state = ordinary.identity_state()
        small_scale_error = exception_name(
            lambda: ordinary.surprise_gate(
                ordinary_state,
                0,
                [1.0, 0.0],
                [0.0, 0.0],
                reference_scale=1.0e-200,
                threshold=0.0,
            )
        )
        large_scale = ordinary.surprise_gate(
            ordinary_state,
            0,
            [1.0, 0.0],
            [0.0, 0.0],
            reference_scale=1.0e200,
            threshold=0.0,
        )

        huge_points = np.array([[-1.0e308, 0.0], [1.0e308, 0.0]])
        huge_core = UnifiedMetricCore(huge_points, pair)
        huge_edge = float(huge_core.edge_lengths(huge_core.identity_state())[0, 1])

        huge_metric = np.repeat((1.0e308 * np.eye(2))[None, :, :], 2, axis=0)
        projected = ordinary.project_metric(huge_metric)
        projected_finite = bool(np.all(np.isfinite(np.asarray(projected.metric))))

        _, underflow_metric = affine_chart_change(
            ordinary_points,
            np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            np.diag([1.0e308, 1.0]),
        )
        _, overflow_metric = affine_chart_change(
            ordinary_points,
            np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            np.diag([1.0e-308, 1.0]),
        )

    results = {
        "unique_chain_positive_scale": unique_chain_scale_probe(),
        "tiny_goal": {
            "reported_minimizers": list(tiny_goal.minimizers),
            "reference_minimizers": [1],
            "reported_costs": [[node, cost] for node, cost in tiny_goal.costs],
            "pass": tiny_goal.minimizers == (1,),
        },
        "source_to_self_uniqueness": {
            "reported_unique": source_self.unique,
            "reference_unique": True,
            "pass": source_self.unique,
        },
        "small_reference_scale": {
            "exception": small_scale_error,
            "pass": small_scale_error is None,
        },
        "large_reference_scale": {
            "reported_normalized": large_scale.normalized_squared_length,
            "reported_gate": large_scale.hard_gate,
            "exact_ratio_is_strictly_positive": True,
            "threshold": 0.0,
            "pass": large_scale.hard_gate == 1,
        },
        "huge_finite_points": {
            "reported_edge": "nan" if math.isnan(huge_edge) else huge_edge,
            "finite": math.isfinite(huge_edge),
            "pass": math.isfinite(huge_edge),
        },
        "huge_finite_metric_projection": {
            "reported_metric_finite": projected_finite,
            "pass": projected_finite,
        },
        "extreme_affine_transport": {
            "large_jacobian_metric_positive_definite": bool(
                np.min(np.linalg.eigvalsh(underflow_metric)) > 0.0
            ),
            "small_jacobian_metric_finite": bool(np.all(np.isfinite(overflow_metric))),
            "pass": bool(
                np.min(np.linalg.eigvalsh(underflow_metric)) > 0.0
                and np.all(np.isfinite(overflow_metric))
            ),
        },
    }
    results["failure_count"] = sum(
        not bool(result.get("pass", False))
        for result in results.values()
        if isinstance(result, dict)
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
