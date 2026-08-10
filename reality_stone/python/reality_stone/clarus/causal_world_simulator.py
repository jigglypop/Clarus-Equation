"""Finite causal-world simulator with exact linear certificates.

The module implements the smallest falsifiable version of an embodied world
model: multiple partial sensor charts reconstruct a latent state, a controlled
transition law predicts interventions, and a planner compares counterfactual
actions.  Passing the synthetic gate proves only the declared finite model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


Array = np.ndarray


def _as_matrix(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def reconstruct_latent(observation_operator: Array, observation: Array) -> Array:
    """Minimum-norm latent reconstruction C^dagger y for one or many samples."""
    operator = _as_matrix(observation_operator, "observation_operator")
    measured = np.asarray(observation, dtype=float)
    if measured.shape[-1] != operator.shape[0]:
        raise ValueError("observation trailing dimension must match operator rows")
    return measured @ np.linalg.pinv(operator).T


@dataclass(frozen=True)
class LinearWorldModel:
    transition: Array
    control: Array

    def predict(self, state: Array, action: Array) -> Array:
        state_value = np.asarray(state, dtype=float)
        action_value = np.asarray(action, dtype=float)
        return state_value @ self.transition.T + action_value @ self.control.T

    def rollout(self, state: Array, actions: Array) -> Array:
        current = np.asarray(state, dtype=float)
        action_sequence = _as_matrix(actions, "actions")
        result = np.empty((len(action_sequence) + 1, self.transition.shape[0]))
        result[0] = current
        for index, action in enumerate(action_sequence):
            current = self.predict(current, action)
            result[index + 1] = current
        return result

    def counterfactual_effect(self, action_a: Array, action_b: Array) -> Array:
        """One-step structural intervention effect B(a-b)."""
        return self.control @ (np.asarray(action_a, dtype=float) - np.asarray(action_b, dtype=float))

    def optimal_one_step_action(
        self,
        state: Array,
        target: Array,
        state_cost: Array,
        action_cost: Array,
    ) -> Array:
        """Exact minimizer of ||Az+Ba-target||_Q^2 + ||a||_R^2."""
        q_matrix = _as_matrix(state_cost, "state_cost")
        r_matrix = _as_matrix(action_cost, "action_cost")
        predicted_without_action = self.transition @ np.asarray(state, dtype=float)
        left = self.control.T @ q_matrix @ self.control + r_matrix
        right = -self.control.T @ q_matrix @ (
            predicted_without_action - np.asarray(target, dtype=float)
        )
        return np.linalg.solve(left, right)


def fit_controlled_linear_model(states: Array, actions: Array, ridge: float = 0.0) -> LinearWorldModel:
    """Fit z[t+1] = A z[t] + B a[t] by chronological least squares."""
    state_sequence = _as_matrix(states, "states")
    action_sequence = _as_matrix(actions, "actions")
    if len(state_sequence) != len(action_sequence) + 1:
        raise ValueError("states must contain one more row than actions")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    design = np.column_stack((state_sequence[:-1], action_sequence))
    target = state_sequence[1:]
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ target) if ridge > 0.0 else np.linalg.pinv(design) @ target
    state_dimension = state_sequence.shape[1]
    return LinearWorldModel(
        transition=coefficients[:state_dimension].T,
        control=coefficients[state_dimension:].T,
    )


def multistep_error_bound(
    transition_norm: float,
    initial_error: float,
    per_step_defect: float,
    steps: int,
) -> float:
    """Geometric rollout bound for e[t+1] <= rho e[t] + defect."""
    if min(transition_norm, initial_error, per_step_defect) < 0.0 or steps < 0:
        raise ValueError("bounds and steps must be nonnegative")
    if math.isclose(transition_norm, 1.0):
        return initial_error + steps * per_step_defect
    return transition_norm**steps * initial_error + per_step_defect * (
        1.0 - transition_norm**steps
    ) / (1.0 - transition_norm)


def harmonic_anchor_extension(
    laplacian: Array,
    anchor_indices: Sequence[int],
    anchor_values: Array,
) -> Array:
    """Unique graph-harmonic extension with fixed Dirichlet anchors."""
    operator = _as_matrix(laplacian, "laplacian")
    if operator.shape[0] != operator.shape[1] or not np.allclose(operator, operator.T):
        raise ValueError("laplacian must be square and symmetric")
    anchors = np.asarray(anchor_indices, dtype=int)
    values = np.asarray(anchor_values, dtype=float)
    if anchors.ndim != 1 or len(np.unique(anchors)) != len(anchors):
        raise ValueError("anchor indices must be unique")
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != len(anchors):
        raise ValueError("anchor_values must have one row per anchor")
    free = np.setdiff1d(np.arange(operator.shape[0]), anchors)
    result = np.empty((operator.shape[0], values.shape[1]))
    result[anchors] = values
    if len(free):
        free_block = operator[np.ix_(free, free)]
        coupling = operator[np.ix_(free, anchors)]
        try:
            result[free] = np.linalg.solve(free_block, -coupling @ values)
        except np.linalg.LinAlgError as error:
            raise ValueError("every connected component must contain an anchor") from error
    return result[:, 0] if result.shape[1] == 1 else result


def chart_transition(chart_from: Array, chart_to: Array) -> Array:
    """Coordinate transition Q_to Q_from^{-1} for invertible linear charts."""
    source = _as_matrix(chart_from, "chart_from")
    target = _as_matrix(chart_to, "chart_to")
    if source.shape[0] != source.shape[1] or target.shape != source.shape:
        raise ValueError("charts must be invertible square matrices of equal shape")
    return target @ np.linalg.inv(source)


def cycle_holonomy(transitions: Mapping[tuple[str, str], Array], cycle: Sequence[str]) -> Array:
    """Compose directed chart transitions around a closed cycle."""
    if len(cycle) < 3 or cycle[0] != cycle[-1]:
        raise ValueError("cycle must contain at least two edges and return to its start")
    first = np.asarray(transitions[(cycle[0], cycle[1])], dtype=float)
    product = np.eye(first.shape[0])
    for source, target in zip(cycle[:-1], cycle[1:]):
        product = np.asarray(transitions[(source, target)], dtype=float) @ product
    return product


def holonomy_frustration(holonomy: Array) -> float:
    value = _as_matrix(holonomy, "holonomy")
    if value.shape[0] != value.shape[1]:
        raise ValueError("holonomy must be square")
    return 0.5 * float(np.linalg.norm(value - np.eye(value.shape[0]), ord="fro") ** 2)


def quadratic_cost(state: Array, action: Array, target: Array, q: Array, r: Array) -> float:
    difference = np.asarray(state, dtype=float) - np.asarray(target, dtype=float)
    action_value = np.asarray(action, dtype=float)
    return float(difference @ q @ difference + action_value @ r @ action_value)


def r_squared(target: Array, prediction: Array) -> float:
    target_value = np.asarray(target, dtype=float)
    prediction_value = np.asarray(prediction, dtype=float)
    denominator = float(np.sum((target_value - np.mean(target_value, axis=0)) ** 2))
    if denominator <= 1e-15:
        return 0.0
    return 1.0 - float(np.sum((target_value - prediction_value) ** 2)) / denominator


@dataclass(frozen=True)
class SyntheticWorldData:
    states: Array
    actions: Array
    observations: Array
    observation_operator: Array
    transition: Array
    control: Array


def generate_synthetic_world(steps: int = 1600, seed: int = 20260810) -> SyntheticWorldData:
    """Generate a stable, partially observed, persistently excited controlled world."""
    if steps < 100:
        raise ValueError("steps must be at least 100")
    transition = np.array(
        [
            [0.84, 0.12, 0.03, 0.00],
            [-0.10, 0.86, 0.00, 0.04],
            [0.05, 0.00, 0.78, 0.16],
            [0.00, -0.03, -0.14, 0.80],
        ]
    )
    control = np.array([[0.28, 0.00], [0.08, 0.12], [0.00, 0.25], [-0.10, 0.08]])
    visual = np.array([[1.0, 0.0, 0.20, 0.0], [0.0, 1.0, 0.0, 0.10]])
    body = np.array([[0.0, 0.10, 1.0, 0.0], [0.15, 0.0, 0.0, 1.0]])
    observation_operator = np.vstack((visual, body))
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1.0, 1.0, size=(steps, 2))
    states = np.empty((steps + 1, 4))
    states[0] = rng.normal(scale=0.4, size=4)
    for index in range(steps):
        states[index + 1] = transition @ states[index] + control @ actions[index]
        states[index + 1] += rng.normal(scale=0.002, size=4)
    observations = states @ observation_operator.T
    observations += rng.normal(scale=0.002, size=observations.shape)
    return SyntheticWorldData(states, actions, observations, observation_operator, transition, control)


@dataclass(frozen=True)
class WorldSimulatorReport:
    seed: int
    train_steps: int
    test_steps: int
    stacked_observation_rank: int
    visual_only_rank: int
    reconstruction_rmse: float
    visual_only_reconstruction_rmse: float
    transition_error: float
    control_error: float
    test_r2_model: float
    test_r2_persistence: float
    counterfactual_effect_error: float
    planned_mean_cost: float
    zero_action_mean_cost: float
    random_action_mean_cost: float
    harmonic_residual: float
    harmonic_energy_advantage: float
    exact_holonomy_frustration: float
    corrupted_holonomy_frustration: float
    passed: bool


def run_synthetic_gate(seed: int = 20260810) -> tuple[WorldSimulatorReport, SyntheticWorldData]:
    data = generate_synthetic_world(seed=seed)
    reconstructed = reconstruct_latent(data.observation_operator, data.observations)
    visual_operator = data.observation_operator[:2]
    visual_reconstructed = reconstruct_latent(visual_operator, data.observations[:, :2])
    reconstruction_rmse = float(np.sqrt(np.mean((reconstructed - data.states) ** 2)))
    visual_rmse = float(np.sqrt(np.mean((visual_reconstructed - data.states) ** 2)))

    train_steps = 1000
    fitted = fit_controlled_linear_model(
        reconstructed[: train_steps + 1],
        data.actions[:train_steps],
        ridge=1e-8,
    )
    test_state = reconstructed[train_steps:-1]
    test_action = data.actions[train_steps:]
    target = data.states[train_steps + 1 :]
    prediction = fitted.predict(test_state, test_action)
    persistence = test_state

    action_a = np.array([0.7, -0.4])
    action_b = np.array([-0.2, 0.5])
    effect_error = float(
        np.linalg.norm(
            fitted.counterfactual_effect(action_a, action_b)
            - data.control @ (action_a - action_b)
        )
    )

    q_matrix = np.diag([1.0, 0.7, 1.2, 0.8])
    r_matrix = np.eye(2) * 0.08
    target_state = np.zeros(4)
    planned_costs: list[float] = []
    zero_costs: list[float] = []
    random_costs: list[float] = []
    rng = np.random.default_rng(seed + 11)
    for state in data.states[train_steps : train_steps + 300]:
        planned_action = fitted.optimal_one_step_action(
            state, target_state, q_matrix, r_matrix
        )
        random_action = rng.normal(scale=0.5, size=2)
        planned_next = data.transition @ state + data.control @ planned_action
        zero_next = data.transition @ state
        random_next = data.transition @ state + data.control @ random_action
        planned_costs.append(
            quadratic_cost(planned_next, planned_action, target_state, q_matrix, r_matrix)
        )
        zero_costs.append(quadratic_cost(zero_next, np.zeros(2), target_state, q_matrix, r_matrix))
        random_costs.append(
            quadratic_cost(random_next, random_action, target_state, q_matrix, r_matrix)
        )

    adjacency = np.zeros((7, 7))
    for index in range(6):
        adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    harmonic = harmonic_anchor_extension(laplacian, (0, 6), np.array([0.0, 1.0]))
    harmonic_residual = float(np.linalg.norm((laplacian @ harmonic)[1:-1]))
    perturbation = harmonic.copy()
    perturbation[3] += 0.2
    harmonic_energy = float(harmonic @ laplacian @ harmonic)
    perturbation_energy = float(perturbation @ laplacian @ perturbation)

    charts = {
        "vision": np.eye(4),
        "body": np.array(
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.0, 1.0]]
        ),
        "action": np.array(
            [[1.0, 0.0, 0.0, 0.1], [0.0, 0.8, 0.2, 0.0], [0.0, -0.1, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    }
    transitions = {
        (source, target_name): chart_transition(charts[source], charts[target_name])
        for source, target_name in (("vision", "body"), ("body", "action"), ("action", "vision"))
    }
    exact_holonomy = cycle_holonomy(transitions, ("vision", "body", "action", "vision"))
    corrupted = dict(transitions)
    corrupted[("body", "action")] = corrupted[("body", "action")].copy()
    corrupted[("body", "action")][0, 2] += 0.12
    corrupted_holonomy = cycle_holonomy(corrupted, ("vision", "body", "action", "vision"))

    report = WorldSimulatorReport(
        seed=seed,
        train_steps=train_steps,
        test_steps=len(test_action),
        stacked_observation_rank=int(np.linalg.matrix_rank(data.observation_operator)),
        visual_only_rank=int(np.linalg.matrix_rank(visual_operator)),
        reconstruction_rmse=reconstruction_rmse,
        visual_only_reconstruction_rmse=visual_rmse,
        transition_error=float(np.linalg.norm(fitted.transition - data.transition, ord="fro")),
        control_error=float(np.linalg.norm(fitted.control - data.control, ord="fro")),
        test_r2_model=r_squared(target, prediction),
        test_r2_persistence=r_squared(target, persistence),
        counterfactual_effect_error=effect_error,
        planned_mean_cost=float(np.mean(planned_costs)),
        zero_action_mean_cost=float(np.mean(zero_costs)),
        random_action_mean_cost=float(np.mean(random_costs)),
        harmonic_residual=harmonic_residual,
        harmonic_energy_advantage=perturbation_energy - harmonic_energy,
        exact_holonomy_frustration=holonomy_frustration(exact_holonomy),
        corrupted_holonomy_frustration=holonomy_frustration(corrupted_holonomy),
        passed=bool(
            reconstruction_rmse < visual_rmse * 0.1
            and np.linalg.matrix_rank(data.observation_operator) == 4
            and r_squared(target, prediction) > 0.99
            and r_squared(target, prediction) > r_squared(target, persistence) + 0.1
            and effect_error < 0.01
            and np.mean(planned_costs) < np.mean(zero_costs)
            and np.mean(planned_costs) < np.mean(random_costs)
            and harmonic_residual < 1e-12
            and perturbation_energy > harmonic_energy
            and holonomy_frustration(exact_holonomy) < 1e-24
            and holonomy_frustration(corrupted_holonomy) > 1e-5
        ),
    )
    return report, data


def _write_data(path: Path, data: SyntheticWorldData) -> None:
    action_padding = np.vstack((data.actions, np.full((1, data.actions.shape[1]), np.nan)))
    rows = np.column_stack((np.arange(len(data.states)), data.states, action_padding, data.observations))
    header = ",".join(
        ["time"]
        + [f"state_{index}" for index in range(data.states.shape[1])]
        + [f"action_{index}" for index in range(data.actions.shape[1])]
        + [f"observation_{index}" for index in range(data.observations.shape[1])]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, default=Path("artifacts/agi/causal_world_report.json"))
    parser.add_argument("--data-output", type=Path, default=Path("artifacts/agi/causal_world_data.csv"))
    args = parser.parse_args(argv)
    report, data = run_synthetic_gate(args.seed)
    rendered = json.dumps(asdict(report), ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    _write_data(args.data_output, data)
    print(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
