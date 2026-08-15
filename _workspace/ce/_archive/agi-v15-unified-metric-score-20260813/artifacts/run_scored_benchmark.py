"""Preregistered independent score for the frozen V15 unified-metric core."""

from __future__ import annotations

import hashlib
import heapq
import inspect
import json
import math
from pathlib import Path
import queue
import threading

import numpy as np

from reality_stone.clarus.unified_metric import (
    UnifiedMetricCore,
    affine_chart_change,
)


EXPECTED_SHA256 = "0599fc3b212f924424de0675266881f8f1a6611d880382533708cd55f2529be4"
CORRECTNESS_SEEDS = range(915_000, 915_256)
UTILITY_SEEDS = range(916_000, 916_256)
RELATIVE_TOLERANCE = 1.0e-10


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(1.0e-300, abs(left), abs(right))


def reference_edges(
    points: np.ndarray,
    adjacency: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    """Independent endpoint-average metric edge implementation."""

    node_count = len(points)
    result = np.full((node_count, node_count), np.inf, dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    for source in range(node_count):
        for target in range(source + 1, node_count):
            if adjacency[source, target] <= 0.0:
                continue
            delta = points[target] - points[source]
            average_metric = (metric[source] + metric[target]) / 2.0
            result[source, target] = math.sqrt(float(delta @ average_metric @ delta))
            result[target, source] = result[source, target]
    return result


def reference_all_pairs(edges: np.ndarray) -> np.ndarray:
    """Independent Floyd--Warshall shortest-cost implementation."""

    distances = edges.copy()
    for pivot in range(len(edges)):
        distances = np.minimum(
            distances,
            distances[:, pivot, None] + distances[None, pivot, :],
        )
    return distances


def reference_path(edges: np.ndarray, source: int, target: int) -> tuple[int, ...]:
    """Strict-comparison Dijkstra used only for path validity cross-checks."""

    distances = [math.inf] * len(edges)
    predecessor = [-1] * len(edges)
    distances[source] = 0.0
    pending = [(0.0, source)]
    while pending:
        distance, node = heapq.heappop(pending)
        if distance != distances[node]:
            continue
        for neighbor, edge in enumerate(edges[node]):
            if neighbor == node or not math.isfinite(float(edge)):
                continue
            candidate = distance + float(edge)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessor[neighbor] = node
                heapq.heappush(pending, (candidate, neighbor))
    path = [target]
    cursor = target
    while cursor != source:
        cursor = predecessor[cursor]
        if cursor < 0:
            raise AssertionError("reference graph unexpectedly disconnected")
        path.append(cursor)
    return tuple(reversed(path))


def random_spd(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    metrics = []
    for _ in range(count):
        basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        eigenvalues = np.exp(rng.uniform(math.log(0.25), math.log(4.0), dimension))
        metrics.append(basis @ np.diag(eigenvalues) @ basis.T)
    return np.asarray(metrics, dtype=np.float64)


def random_fixture(
    seed: int,
) -> tuple[np.random.Generator, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimension = int(rng.integers(2, 5))
    node_count = int(rng.integers(5, 10))
    points = rng.normal(size=(node_count, dimension))
    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    order = rng.permutation(node_count)
    for index in range(1, node_count):
        source = int(order[index])
        target = int(order[int(rng.integers(0, index))])
        adjacency[source, target] = adjacency[target, source] = 1.0
    for source in range(node_count):
        for target in range(source + 1, node_count):
            if rng.random() < 0.35:
                adjacency[source, target] = adjacency[target, source] = 1.0
    return rng, points, adjacency, random_spd(rng, node_count, dimension)


def valid_path(
    nodes: tuple[int, ...],
    source: int,
    target: int,
    edges: np.ndarray,
    reported_cost: float,
) -> bool:
    if not nodes or nodes[0] != source or nodes[-1] != target:
        return False
    if len(set(nodes)) != len(nodes):
        return False
    cost = 0.0
    for left, right in zip(nodes, nodes[1:]):
        edge = float(edges[left, right])
        if not math.isfinite(edge):
            return False
        cost += edge
    return relative_error(cost, reported_cost) <= RELATIVE_TOLERANCE


def held_out_correctness() -> tuple[dict[str, object], dict[str, object]]:
    scalar_errors: list[float] = []
    path_matches = 0
    path_total = 0
    valid_paths = 0
    goal_matches = 0
    finite_trials = 0
    affine_errors: list[float] = []
    affine_goal_matches = 0
    affine_path_matches = 0
    permutation_goal_matches = 0
    permutation_path_matches = 0

    for seed in CORRECTNESS_SEEDS:
        rng, points, adjacency, metric = random_fixture(seed)
        core = UnifiedMetricCore(points, adjacency)
        state = core.make_state(metric)
        edges = reference_edges(points, adjacency, metric)
        all_pairs = reference_all_pairs(edges)
        trial_finite = np.all(np.isfinite(all_pairs))

        for source in range(len(points)):
            for target in range(len(points)):
                path = core.shortest_path(state, source, target)
                error = relative_error(path.cost, float(all_pairs[source, target]))
                scalar_errors.append(error)
                path_total += 1
                path_matches += int(error <= RELATIVE_TOLERANCE)
                valid_paths += int(
                    valid_path(path.nodes, source, target, edges, path.cost)
                )
                trial_finite = trial_finite and math.isfinite(path.cost)

        source = int(rng.integers(0, len(points)))
        candidates = [node for node in range(len(points)) if node != source]
        rng.shuffle(candidates)
        candidates = candidates[: int(rng.integers(2, len(candidates) + 1))]
        goal = core.minimum_cost_targets(state, source, candidates)
        optimum = min(float(all_pairs[source, node]) for node in candidates)
        goal_tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, abs(optimum))
        reference_goals = tuple(
            sorted(
                node
                for node in candidates
                if abs(float(all_pairs[source, node]) - optimum) <= goal_tolerance
            )
        )
        goal_matches += int(goal.minimizers == reference_goals)

        observed = rng.normal(size=points.shape[1])
        predicted = rng.normal(size=points.shape[1])
        reference_scale = float(np.exp(rng.uniform(math.log(0.2), math.log(3.0))))
        threshold = float(rng.uniform(0.0, 8.0))
        node = int(rng.integers(0, len(points)))
        delta = observed - predicted
        reference_squared = float(delta @ metric[node] @ delta)
        reference_normalized = reference_squared / reference_scale**2
        surprise = core.surprise_gate(
            state,
            node,
            observed,
            predicted,
            reference_scale=reference_scale,
            threshold=threshold,
        )
        scalar_errors.extend(
            [
                relative_error(surprise.squared_length, reference_squared),
                relative_error(
                    surprise.normalized_squared_length,
                    reference_normalized,
                ),
            ]
        )
        trial_finite = trial_finite and all(
            math.isfinite(value)
            for value in (
                surprise.squared_length,
                surprise.normalized_squared_length,
            )
        )
        trial_finite = trial_finite and surprise.hard_gate == int(
            reference_normalized > threshold
        )
        finite_trials += int(trial_finite)

        left, _ = np.linalg.qr(rng.normal(size=(points.shape[1], points.shape[1])))
        right, _ = np.linalg.qr(rng.normal(size=(points.shape[1], points.shape[1])))
        scales = np.exp(
            rng.uniform(math.log(0.25), math.log(4.0), points.shape[1])
        )
        jacobian = left @ np.diag(scales) @ right
        offset = rng.normal(size=points.shape[1])
        points_y, metric_y = affine_chart_change(points, metric, jacobian, offset)
        core_y = UnifiedMetricCore(points_y, adjacency)
        state_y = core_y.make_state(metric_y)
        edges_y = core_y.edge_lengths(state_y)
        mask = np.isfinite(edges)
        affine_errors.extend(
            relative_error(float(before), float(after))
            for before, after in zip(edges[mask], edges_y[mask], strict=True)
        )
        affine_path = core_y.shortest_path(state_y, source, candidates[0])
        affine_path_matches += int(
            relative_error(
                affine_path.cost,
                float(all_pairs[source, candidates[0]]),
            )
            <= RELATIVE_TOLERANCE
        )
        affine_goal = core_y.minimum_cost_targets(state_y, source, candidates)
        affine_goal_matches += int(affine_goal.minimizers == goal.minimizers)

        order = rng.permutation(len(points))
        old_to_new = np.argsort(order)
        permuted = UnifiedMetricCore(
            points[order],
            adjacency[np.ix_(order, order)],
        )
        permuted_state = permuted.make_state(metric[order])
        new_source = int(old_to_new[source])
        new_candidates = [int(old_to_new[node]) for node in candidates]
        permuted_goal = permuted.minimum_cost_targets(
            permuted_state,
            new_source,
            new_candidates,
        )
        mapped_goals = tuple(sorted(int(order[node]) for node in permuted_goal.minimizers))
        permutation_goal_matches += int(mapped_goals == goal.minimizers)
        new_target = int(old_to_new[candidates[0]])
        permuted_path = permuted.shortest_path(permuted_state, new_source, new_target)
        permutation_path_matches += int(
            relative_error(
                permuted_path.cost,
                float(all_pairs[source, candidates[0]]),
            )
            <= RELATIVE_TOLERANCE
        )

    trial_count = len(CORRECTNESS_SEEDS)
    correctness = {
        "seeds": trial_count,
        "finite_trial_rate": finite_trials / trial_count,
        "path_cost_agreement": path_matches / path_total,
        "valid_path_rate": valid_paths / path_total,
        "goal_exact_agreement": goal_matches / trial_count,
        "maximum_relative_scalar_error": max(scalar_errors),
    }
    correctness["pass"] = bool(
        correctness["finite_trial_rate"] == 1.0
        and correctness["path_cost_agreement"] >= 0.999
        and correctness["valid_path_rate"] >= 0.999
        and correctness["goal_exact_agreement"] >= 0.999
        and correctness["maximum_relative_scalar_error"] <= RELATIVE_TOLERANCE
    )
    ood = {
        "seeds": trial_count,
        "affine_max_relative_edge_error": max(affine_errors),
        "affine_path_cost_agreement": affine_path_matches / trial_count,
        "affine_goal_exact_agreement": affine_goal_matches / trial_count,
        "permutation_path_cost_agreement": permutation_path_matches / trial_count,
        "permutation_goal_exact_agreement": permutation_goal_matches / trial_count,
    }
    ood["pass"] = bool(
        ood["affine_max_relative_edge_error"] <= RELATIVE_TOLERANCE
        and ood["affine_path_cost_agreement"] >= 0.999
        and ood["affine_goal_exact_agreement"] >= 0.999
        and ood["permutation_path_cost_agreement"] >= 0.999
        and ood["permutation_goal_exact_agreement"] >= 0.999
    )
    return correctness, ood


def _scale_path_worker(results: queue.Queue[object]) -> None:
    points = np.array([[0.0, 0.0], [1.0e-16, 0.0], [2.0e-16, 0.0]])
    adjacency = np.ones((3, 3), dtype=np.float64) - np.eye(3)
    core = UnifiedMetricCore(points, adjacency)
    try:
        path = core.shortest_path(core.identity_state(), 2, 0)
        results.put({"nodes": list(path.nodes), "cost": path.cost})
    except BaseException as error:  # pragma: no cover - defensive scoring boundary
        results.put({"error": f"{type(error).__name__}: {error}"})


def positive_scale_killing_fixture() -> dict[str, object]:
    results: queue.Queue[object] = queue.Queue()
    worker = threading.Thread(target=_scale_path_worker, args=(results,), daemon=True)
    worker.start()
    worker.join(timeout=1.0)
    if worker.is_alive():
        return {
            "pass": False,
            "terminated_within_one_second": False,
            "expected_nodes": [2, 0],
            "expected_cost": 2.0e-16,
            "failure": "shortest_path predecessor cycle did not terminate",
        }
    outcome = results.get_nowait()
    if not isinstance(outcome, dict) or "error" in outcome:
        return {
            "pass": False,
            "terminated_within_one_second": True,
            "outcome": outcome,
        }
    nodes = outcome["nodes"]
    cost = float(outcome["cost"])
    passed = nodes == [2, 0] and relative_error(cost, 2.0e-16) <= RELATIVE_TOLERANCE
    return {
        "pass": passed,
        "terminated_within_one_second": True,
        "nodes": nodes,
        "cost": cost,
        "expected_nodes": [2, 0],
        "expected_cost": 2.0e-16,
    }


def branch_cost(
    points: np.ndarray,
    metric: np.ndarray,
    middle: int,
) -> float:
    adjacency = np.zeros((4, 4), dtype=np.float64)
    adjacency[0, middle] = adjacency[middle, 0] = 1.0
    adjacency[middle, 3] = adjacency[3, middle] = 1.0
    edges = reference_edges(points, adjacency, metric)
    return float(edges[0, middle] + edges[middle, 3])


def oracle_navigation_utility() -> dict[str, object]:
    v15_correct = 0
    baseline_correct = 0
    v15_regrets: list[float] = []
    baseline_regrets: list[float] = []
    v15_cost_errors: list[float] = []
    for seed in UTILITY_SEEDS:
        rng = np.random.default_rng(seed)
        height_one, height_two = rng.uniform(0.05, 1.2, size=2)
        points = np.array(
            [[0.0, 0.0], [1.0, height_one], [1.0, -height_two], [2.0, 0.0]],
            dtype=np.float64,
        )
        multipliers = np.ones(4, dtype=np.float64)
        multipliers[1:3] = np.exp(
            rng.uniform(math.log(0.25), math.log(16.0), size=2)
        )
        metric = np.asarray([value * np.eye(2) for value in multipliers])
        adjacency = np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )
        costs = {middle: branch_cost(points, metric, middle) for middle in (1, 2)}
        optimum_middle = min(costs, key=costs.get)
        optimum_cost = costs[optimum_middle]

        v15 = UnifiedMetricCore(points, adjacency)
        v15_path = v15.shortest_path(v15.make_state(metric), 0, 3)
        v15_middle = v15_path.nodes[1]
        v15_cost = costs[v15_middle]
        v15_correct += int(v15_middle == optimum_middle)
        v15_regrets.append((v15_cost - optimum_cost) / max(optimum_cost, 1.0e-12))
        v15_cost_errors.append(relative_error(v15_path.cost, optimum_cost))

        baseline = UnifiedMetricCore(points, adjacency)
        baseline_path = baseline.shortest_path(baseline.identity_state(), 0, 3)
        baseline_middle = baseline_path.nodes[1]
        baseline_cost = costs[baseline_middle]
        baseline_correct += int(baseline_middle == optimum_middle)
        baseline_regrets.append(
            (baseline_cost - optimum_cost) / max(optimum_cost, 1.0e-12)
        )

    trial_count = len(UTILITY_SEEDS)
    v15_mean_regret = float(np.mean(v15_regrets))
    baseline_mean_regret = float(np.mean(baseline_regrets))
    improvement = baseline_mean_regret - v15_mean_regret
    result = {
        "seeds": trial_count,
        "v15_exact_choice_accuracy": v15_correct / trial_count,
        "v15_mean_normalized_regret": v15_mean_regret,
        "v15_max_relative_cost_error": max(v15_cost_errors),
        "identity_exact_choice_accuracy": baseline_correct / trial_count,
        "identity_mean_normalized_regret": baseline_mean_regret,
        "paired_mean_regret_improvement": improvement,
        "same_search_algorithm_and_call_count": True,
        "metric_is_oracle_supplied": True,
    }
    result["pass"] = bool(
        result["v15_exact_choice_accuracy"] >= 0.99
        and result["v15_mean_normalized_regret"] <= 1.0e-10
        and improvement >= 0.05
    )
    return result


def main() -> None:
    module_path = Path(inspect.getfile(UnifiedMetricCore)).resolve()
    frozen_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    correctness, ood = held_out_correctness()
    utility = oracle_navigation_utility()
    scale = positive_scale_killing_fixture()
    finite_core_go = bool(correctness["pass"] and ood["pass"] and scale["pass"])
    autonomous_gates = {
        "A1_raw_observation_metric_learning": 0,
        "A2_closed_perception_action_loop": 0,
        "A3_delayed_credit_assignment": 0,
        "A4_compute_matched_learned_compositional_ood": 0,
    }
    results = {
        "frozen_sha256": frozen_sha256,
        "frozen_sha256_matches_contract": frozen_sha256 == EXPECTED_SHA256,
        "held_out_correctness": correctness,
        "affine_permutation_ood": ood,
        "positive_scale_killing_fixture": scale,
        "finite_core_go": finite_core_go,
        "oracle_navigation_utility": utility,
        "oracle_utility_go": bool(utility["pass"]),
        "autonomous_agent_gates": autonomous_gates,
        "autonomous_agent_score": f"{sum(autonomous_gates.values())}/4",
        "internal_agi_qualification_percent": 0.0,
        "agi_verdict": "STOP",
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
