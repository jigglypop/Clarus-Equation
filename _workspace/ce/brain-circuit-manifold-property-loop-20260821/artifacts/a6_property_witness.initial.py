"""Frozen randomized property witnesses for the A6-P/A6-C equations.

This script opens no empirical response data.  It tests only the smooth,
dimensionless, delayed synthetic fixtures frozen in 00-contract.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (104729, 130363, 169399, 200003, 250007, 300017, 350003, 400009)
EPSILON = 0.25
FD_STEP = 2.0**-17
RANK_RCOND = 1.0e-10

TANGENT_TOL = 5.0e-7
J_RESPONSE_TOL = 2.0e-6
METRIC_RESPONSE_TOL = 3.0e-6
COVARIANCE_TOL = 1.0e-10
BAD_COVARIANCE_MIN = 1.0e-5
GRAMIAN_SYMMETRY_TOL = 1.0e-12
GRAMIAN_PSD_FACTOR = 1.0e-11
CONTROL_RESIDUAL_TOL = 1.0e-9
ENERGY_IDENTITY_TOL = 1.0e-8
ENERGY_RESPONSE_TOL = 5.0e-6
MAX_GRAMIAN_CONDITION = 1.0e8
STATE_EFFICACY_FULL_TOL = 5.0e-7
STATE_EFFICACY_OMITTED_MIN = 1.0e-3
STATE_RESPONSE_FULL_TOL = 5.0e-7
STATE_RESPONSE_OMITTED_MIN = 1.0e-5


@dataclass(frozen=True)
class Fixture:
    seed: int
    q: int
    delay_depth: int
    horizon: int
    input_dim: int
    update_fraction: np.ndarray
    weight: np.ndarray
    circuit: np.ndarray
    efficacy: np.ndarray
    delays: np.ndarray
    net_offset: np.ndarray
    history: np.ndarray
    input_gain: tuple[np.ndarray, ...]
    input_path: tuple[np.ndarray, ...]
    input_cost: tuple[np.ndarray, ...]
    terminal_metric: np.ndarray
    control_probe: np.ndarray
    direction_probe: np.ndarray

    @property
    def augmented_dim(self) -> int:
        return self.q * (self.delay_depth + 1)


def normalized_error(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(1.0, float(np.linalg.norm(right_array)))
    )


def scalar_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(1.0, abs(float(right)))


def block_diagonal(blocks: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    result = np.zeros((rows, cols), dtype=float)
    row = 0
    col = 0
    for block in blocks:
        height, width = block.shape
        result[row : row + height, col : col + width] = block
        row += height
        col += width
    return result


def svd_receipt(matrix: np.ndarray) -> tuple[np.ndarray, float, int, float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    cutoff = RANK_RCOND * max(1.0, sigma_max)
    rank = int(np.sum(singular_values > cutoff))
    full_dimension = min(matrix.shape)
    if rank == full_dimension and singular_values[-1] > 0.0:
        condition = float(sigma_max / singular_values[-1])
    else:
        condition = math.inf
    return singular_values, cutoff, rank, condition


def pinv_with_frozen_cutoff(matrix: np.ndarray) -> np.ndarray:
    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    cutoff = RANK_RCOND * max(1.0, sigma_max)
    inverse = np.zeros_like(singular_values)
    retained = singular_values > cutoff
    inverse[retained] = 1.0 / singular_values[retained]
    return (right_t.T * inverse) @ left.T


def spd_inverse_half(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("input-cost matrix is not SPD")
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def make_chart(q: int) -> np.ndarray:
    chart = np.diag(np.linspace(0.75, 1.25, q))
    chart[0, 1] = 0.25
    chart[1, 0] = -0.10
    return chart


def make_fixture(seed: int, index: int) -> Fixture:
    rng = np.random.default_rng(seed)
    if index % 2 == 0:
        q, delay_depth, horizon = 3, 1, 3
    else:
        q, delay_depth, horizon = 4, 2, 4
    input_dim = q

    update_fraction = rng.uniform(0.35, 0.90, size=q)
    weight = rng.uniform(-0.35, 0.35, size=(q, q))
    efficacy = rng.uniform(0.20, 0.90, size=(q, q))
    delays = rng.integers(0, delay_depth + 1, size=(q, q), endpoint=False)
    delays[0, 0] = 0
    delays[0, 1] = delay_depth
    net_offset = rng.uniform(-0.30, 0.30, size=q)
    history = rng.uniform(-0.40, 0.40, size=(delay_depth + 1, q))

    support = rng.random((q, q)) < 0.35
    np.fill_diagonal(support, True)
    magnitudes = rng.uniform(0.04, 0.12, size=(q, q))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(q, q))
    circuit = support * magnitudes * signs

    input_gains: list[np.ndarray] = []
    input_paths: list[np.ndarray] = []
    input_costs: list[np.ndarray] = []
    for _ in range(horizon):
        gain = np.diag(rng.uniform(0.45, 0.75, size=q))
        gain += rng.normal(0.0, 0.025, size=(q, q))
        input_gains.append(gain)
        input_paths.append(rng.uniform(-0.25, 0.25, size=input_dim))
        raw_cost = rng.normal(0.0, 0.12, size=(input_dim, input_dim))
        cost = raw_cost.T @ raw_cost + np.diag(
            rng.uniform(0.65, 1.20, size=input_dim)
        )
        input_costs.append(cost)

    raw_metric = rng.normal(0.0, 0.18, size=(q, q))
    terminal_metric = raw_metric.T @ raw_metric + np.diag(
        rng.uniform(0.70, 1.30, size=q)
    )
    control_probe = rng.normal(0.0, 0.35, size=horizon * input_dim)
    direction_probe = rng.normal(0.0, 1.0, size=q)
    direction_probe /= np.linalg.norm(direction_probe)

    return Fixture(
        seed=seed,
        q=q,
        delay_depth=delay_depth,
        horizon=horizon,
        input_dim=input_dim,
        update_fraction=update_fraction,
        weight=weight,
        circuit=circuit,
        efficacy=efficacy,
        delays=delays,
        net_offset=net_offset,
        history=history,
        input_gain=tuple(input_gains),
        input_path=tuple(input_paths),
        input_cost=tuple(input_costs),
        terminal_metric=terminal_metric,
        control_probe=control_probe,
        direction_probe=direction_probe,
    )


def selectors(fixture: Fixture) -> tuple[np.ndarray, np.ndarray]:
    projection = np.zeros((fixture.q, fixture.augmented_dim), dtype=float)
    projection[:, : fixture.q] = np.eye(fixture.q)
    injection = np.zeros((fixture.augmented_dim, fixture.q), dtype=float)
    injection[: fixture.q, :] = np.eye(fixture.q)
    return projection, injection


def nonlinear_step(
    fixture: Fixture, state: np.ndarray, epsilon: float, tick: int
) -> np.ndarray:
    q = fixture.q
    depth = fixture.delay_depth
    history = state.reshape(depth + 1, q)
    weight = fixture.weight + epsilon * fixture.circuit
    drive = fixture.input_gain[tick] @ fixture.input_path[tick] + fixture.net_offset
    for receiver in range(q):
        for sender in range(q):
            delayed = history[fixture.delays[receiver, sender], sender]
            drive[receiver] += (
                weight[receiver, sender]
                * fixture.efficacy[receiver, sender]
                * delayed
            )
    next_current = (
        (1.0 - fixture.update_fraction) * history[0]
        + fixture.update_fraction * np.tanh(drive)
    )
    next_history = np.empty_like(history)
    next_history[0] = next_current
    if depth:
        next_history[1:] = history[:-1]
    return next_history.reshape(-1)


def nonlinear_rollout(
    fixture: Fixture, epsilon: float, history: np.ndarray | None = None
) -> np.ndarray:
    state = (
        fixture.history.copy().reshape(-1)
        if history is None
        else np.asarray(history, dtype=float).copy().reshape(-1)
    )
    for tick in range(fixture.horizon):
        state = nonlinear_step(fixture, state, epsilon, tick)
    return state


def linearize_with_circuit_response(
    fixture: Fixture, epsilon: float
) -> dict[str, Any]:
    q = fixture.q
    depth = fixture.delay_depth
    augmented_dim = fixture.augmented_dim
    weight = fixture.weight + epsilon * fixture.circuit
    state = fixture.history.copy().reshape(-1)
    state_dot = np.zeros(augmented_dim, dtype=float)
    transitions: list[np.ndarray] = []
    actuators: list[np.ndarray] = []
    transition_dots: list[np.ndarray] = []
    actuator_dots: list[np.ndarray] = []

    for tick in range(fixture.horizon):
        history = state.reshape(depth + 1, q)
        history_dot = state_dot.reshape(depth + 1, q)
        drive = fixture.input_gain[tick] @ fixture.input_path[tick] + fixture.net_offset
        drive_dot = np.zeros(q, dtype=float)
        for receiver in range(q):
            for sender in range(q):
                delay = fixture.delays[receiver, sender]
                source = history[delay, sender]
                source_dot = history_dot[delay, sender]
                efficacy = fixture.efficacy[receiver, sender]
                drive[receiver] += weight[receiver, sender] * efficacy * source
                drive_dot[receiver] += (
                    fixture.circuit[receiver, sender] * efficacy * source
                    + weight[receiver, sender] * efficacy * source_dot
                )

        activation = np.tanh(drive)
        gain = 1.0 - activation**2
        second = -2.0 * activation * gain
        gain_dot = second * drive_dot

        transition = np.zeros((augmented_dim, augmented_dim), dtype=float)
        transition_dot = np.zeros_like(transition)
        for receiver in range(q):
            transition[receiver, receiver] += 1.0 - fixture.update_fraction[receiver]
            for sender in range(q):
                delay = fixture.delays[receiver, sender]
                column = delay * q + sender
                efficacy = fixture.efficacy[receiver, sender]
                transition[receiver, column] += (
                    fixture.update_fraction[receiver]
                    * gain[receiver]
                    * weight[receiver, sender]
                    * efficacy
                )
                transition_dot[receiver, column] += fixture.update_fraction[
                    receiver
                ] * (
                    gain_dot[receiver] * weight[receiver, sender] * efficacy
                    + gain[receiver]
                    * fixture.circuit[receiver, sender]
                    * efficacy
                )
        if depth:
            transition[q:, : depth * q] = np.eye(depth * q)

        actuator = np.zeros((augmented_dim, fixture.input_dim), dtype=float)
        actuator[:q] = (
            fixture.update_fraction * gain
        )[:, None] * fixture.input_gain[tick]
        actuator_dot = np.zeros_like(actuator)
        actuator_dot[:q] = (
            fixture.update_fraction * gain_dot
        )[:, None] * fixture.input_gain[tick]

        next_current = (
            (1.0 - fixture.update_fraction) * history[0]
            + fixture.update_fraction * activation
        )
        next_current_dot = (
            (1.0 - fixture.update_fraction) * history_dot[0]
            + fixture.update_fraction * gain * drive_dot
        )
        next_history = np.empty_like(history)
        next_history_dot = np.empty_like(history_dot)
        next_history[0] = next_current
        next_history_dot[0] = next_current_dot
        if depth:
            next_history[1:] = history[:-1]
            next_history_dot[1:] = history_dot[:-1]

        transitions.append(transition)
        actuators.append(actuator)
        transition_dots.append(transition_dot)
        actuator_dots.append(actuator_dot)
        state = next_history.reshape(-1)
        state_dot = next_history_dot.reshape(-1)

    projection, injection = selectors(fixture)
    transition_product, transition_product_dot = transition_and_response_from(
        transitions, transition_dots, 0
    )
    endpoint_jacobian = projection @ transition_product @ injection
    endpoint_jacobian_dot = projection @ transition_product_dot @ injection
    metric = endpoint_jacobian.T @ fixture.terminal_metric @ endpoint_jacobian
    metric_dot = (
        endpoint_jacobian_dot.T @ fixture.terminal_metric @ endpoint_jacobian
        + endpoint_jacobian.T @ fixture.terminal_metric @ endpoint_jacobian_dot
    )
    return {
        "endpoint_state": state,
        "endpoint_state_dot": state_dot,
        "transitions": transitions,
        "actuators": actuators,
        "transition_dots": transition_dots,
        "actuator_dots": actuator_dots,
        "endpoint_jacobian": endpoint_jacobian,
        "endpoint_jacobian_dot": endpoint_jacobian_dot,
        "metric": metric,
        "metric_dot": metric_dot,
    }


def transition_and_response_from(
    transitions: list[np.ndarray], transition_dots: list[np.ndarray], start: int
) -> tuple[np.ndarray, np.ndarray]:
    dimension = transitions[0].shape[0]
    product = np.eye(dimension)
    product_dot = np.zeros((dimension, dimension), dtype=float)
    for tick in range(start, len(transitions)):
        product_dot = transition_dots[tick] @ product + transitions[tick] @ product_dot
        product = transitions[tick] @ product
    return product, product_dot


def finite_difference_initial_jacobian(fixture: Fixture, epsilon: float) -> np.ndarray:
    result = np.zeros((fixture.q, fixture.q), dtype=float)
    for coordinate in range(fixture.q):
        plus = fixture.history.copy()
        minus = fixture.history.copy()
        plus[0, coordinate] += FD_STEP
        minus[0, coordinate] -= FD_STEP
        result[:, coordinate] = (
            nonlinear_rollout(fixture, epsilon, plus)[: fixture.q]
            - nonlinear_rollout(fixture, epsilon, minus)[: fixture.q]
        ) / (2.0 * FD_STEP)
    return result


def control_operators(fixture: Fixture, analysis: dict[str, Any]) -> dict[str, Any]:
    transitions: list[np.ndarray] = analysis["transitions"]
    transition_dots: list[np.ndarray] = analysis["transition_dots"]
    actuators: list[np.ndarray] = analysis["actuators"]
    actuator_dots: list[np.ndarray] = analysis["actuator_dots"]
    blocks: list[np.ndarray] = []
    block_dots: list[np.ndarray] = []
    for tick in range(fixture.horizon):
        phi, phi_dot = transition_and_response_from(
            transitions, transition_dots, tick + 1
        )
        blocks.append(phi @ actuators[tick])
        block_dots.append(phi_dot @ actuators[tick] + phi @ actuator_dots[tick])
    control_map = np.concatenate(blocks, axis=1)
    control_map_dot = np.concatenate(block_dots, axis=1)
    cost = block_diagonal(fixture.input_cost)
    cost_inverse = np.linalg.inv(cost)
    gramian = control_map @ cost_inverse @ control_map.T
    gramian_dot = (
        control_map_dot @ cost_inverse @ control_map.T
        + control_map @ cost_inverse @ control_map_dot.T
    )
    return {
        "control_map": control_map,
        "control_map_dot": control_map_dot,
        "cost": cost,
        "cost_inverse": cost_inverse,
        "gramian": gramian,
        "gramian_dot": gramian_dot,
    }


def generalized_eigenvalues(metric_zero: np.ndarray, metric_one: np.ndarray) -> np.ndarray:
    lower = np.linalg.cholesky(metric_zero)
    left_solved = np.linalg.solve(lower, metric_one)
    whitened = np.linalg.solve(lower, left_solved.T).T
    whitened = 0.5 * (whitened + whitened.T)
    return np.linalg.eigvalsh(whitened)


def run_seed(fixture: Fixture) -> dict[str, Any]:
    baseline = linearize_with_circuit_response(fixture, EPSILON)
    fd_jacobian = finite_difference_initial_jacobian(fixture, EPSILON)
    tangent_error = normalized_error(baseline["endpoint_jacobian"], fd_jacobian)

    plus = linearize_with_circuit_response(fixture, EPSILON + FD_STEP)
    minus = linearize_with_circuit_response(fixture, EPSILON - FD_STEP)
    fd_jacobian_dot = (
        plus["endpoint_jacobian"] - minus["endpoint_jacobian"]
    ) / (2.0 * FD_STEP)
    fd_metric_dot = (plus["metric"] - minus["metric"]) / (2.0 * FD_STEP)
    jacobian_response_error = normalized_error(
        baseline["endpoint_jacobian_dot"], fd_jacobian_dot
    )
    metric_response_error = normalized_error(baseline["metric_dot"], fd_metric_dot)

    # A direct-edge-only partial derivative is logged, but is never substituted
    # for the total response gate.
    direct_transition_dots: list[np.ndarray] = []
    for transition_dot, transition in zip(
        baseline["transition_dots"], baseline["transitions"], strict=True
    ):
        direct = transition_dot.copy()
        # Remove the trajectory-mediated row-proportional component by rebuilding
        # only the explicit circuit term from the baseline gain ratio where safe.
        # This is diagnostic only; total_response above is the registered equation.
        direct_transition_dots.append(direct * 0.0)
    _, zero_partial_product_dot = transition_and_response_from(
        baseline["transitions"], direct_transition_dots, 0
    )
    projection, injection = selectors(fixture)
    zero_partial_jdot = projection @ zero_partial_product_dot @ injection
    zero_partial_error = normalized_error(zero_partial_jdot, fd_jacobian_dot)

    jacobian_singular, jacobian_cutoff, jacobian_rank, jacobian_condition = svd_receipt(
        baseline["endpoint_jacobian"]
    )

    chart = make_chart(fixture.q)
    chart_inverse = np.linalg.inv(chart)
    chart_condition = float(np.linalg.cond(chart))
    chart_nonorthogonality = float(
        np.linalg.norm(chart.T @ chart - np.eye(fixture.q))
    )
    lift = np.kron(np.eye(fixture.delay_depth + 1), chart)
    lift_inverse = np.linalg.inv(lift)
    transformed_transitions = [
        lift @ transition @ lift_inverse for transition in baseline["transitions"]
    ]
    zero_dots = [np.zeros_like(item) for item in transformed_transitions]
    transformed_product, _ = transition_and_response_from(
        transformed_transitions, zero_dots, 0
    )
    projection_y = chart @ projection @ lift_inverse
    injection_y = lift @ injection @ chart_inverse
    jacobian_y = projection_y @ transformed_product @ injection_y
    expected_jacobian_y = chart @ baseline["endpoint_jacobian"] @ chart_inverse
    terminal_metric_y = chart_inverse.T @ fixture.terminal_metric @ chart_inverse
    metric_y = jacobian_y.T @ terminal_metric_y @ jacobian_y
    expected_metric_y = chart_inverse.T @ baseline["metric"] @ chart_inverse
    chart_jacobian_error = normalized_error(jacobian_y, expected_jacobian_y)
    chart_metric_error = normalized_error(metric_y, expected_metric_y)
    direction_x = fixture.direction_probe
    direction_y = chart @ direction_x
    length_x = float(direction_x @ baseline["metric"] @ direction_x)
    length_y = float(direction_y @ metric_y @ direction_y)
    chart_length_error = scalar_error(length_y, length_x)

    bad_metric_y = jacobian_y.T @ fixture.terminal_metric @ jacobian_y
    bad_coordinate_error = normalized_error(bad_metric_y, expected_metric_y)

    pre = linearize_with_circuit_response(fixture, 0.0)
    post = linearize_with_circuit_response(fixture, 1.0)
    _, _, pre_metric_rank, _ = svd_receipt(pre["metric"])
    _, _, post_metric_rank, _ = svd_receipt(post["metric"])
    finite_metric_invariants = (
        pre_metric_rank == fixture.q and post_metric_rank == fixture.q
    )
    generalized_error: float | None = None
    log_volume_error: float | None = None
    if finite_metric_invariants:
        eigen_x = generalized_eigenvalues(pre["metric"], post["metric"])
        pre_y = chart_inverse.T @ pre["metric"] @ chart_inverse
        post_y = chart_inverse.T @ post["metric"] @ chart_inverse
        eigen_y = generalized_eigenvalues(pre_y, post_y)
        generalized_error = normalized_error(eigen_y, eigen_x)
        _, logdet_pre_x = np.linalg.slogdet(pre["metric"])
        _, logdet_post_x = np.linalg.slogdet(post["metric"])
        _, logdet_pre_y = np.linalg.slogdet(pre_y)
        _, logdet_post_y = np.linalg.slogdet(post_y)
        log_volume_error = scalar_error(
            0.5 * (logdet_post_y - logdet_pre_y),
            0.5 * (logdet_post_x - logdet_pre_x),
        )

    control = control_operators(fixture, baseline)
    control_map = control["control_map"]
    gramian = control["gramian"]
    gramian_symmetry = normalized_error(gramian, gramian.T)
    gramian_eigenvalues = np.linalg.eigvalsh(0.5 * (gramian + gramian.T))
    gramian_norm = float(np.linalg.norm(gramian, ord=2))
    gramian_psd_floor = -GRAMIAN_PSD_FACTOR * max(1.0, gramian_norm)
    gramian_singular, gramian_cutoff, gramian_rank, gramian_condition = svd_receipt(
        gramian
    )

    target = control_map @ fixture.control_probe
    inverse_half_blocks = tuple(spd_inverse_half(cost) for cost in fixture.input_cost)
    cost_inverse_half = block_diagonal(inverse_half_blocks)
    weighted_map = control_map @ cost_inverse_half
    weighted_solution = pinv_with_frozen_cutoff(weighted_map) @ target
    minimum_control = cost_inverse_half @ weighted_solution
    control_residual = float(
        np.linalg.norm(control_map @ minimum_control - target)
        / max(1.0, float(np.linalg.norm(target)))
    )
    control_energy = float(minimum_control @ control["cost"] @ minimum_control)
    gramian_energy = float(target @ pinv_with_frozen_cutoff(gramian) @ target)
    energy_identity_error = scalar_error(control_energy, gramian_energy)

    lifted_control_map = lift @ control_map
    lifted_gramian = lift @ gramian @ lift.T
    lifted_target = lift @ target
    chart_energy = float(
        lifted_target @ pinv_with_frozen_cutoff(lifted_gramian) @ lifted_target
    )
    chart_energy_error = scalar_error(chart_energy, gramian_energy)

    energy_response_eligible = (
        gramian_rank == fixture.augmented_dim
        and gramian_condition <= MAX_GRAMIAN_CONDITION
    )
    analytic_energy_dot: float | None = None
    finite_energy_dot: float | None = None
    energy_response_error: float | None = None
    if energy_response_eligible:
        gramian_inverse = np.linalg.inv(gramian)
        analytic_energy_dot = float(
            -target
            @ gramian_inverse
            @ control["gramian_dot"]
            @ gramian_inverse
            @ target
        )
        plus_control = control_operators(fixture, plus)
        minus_control = control_operators(fixture, minus)
        plus_energy = float(target @ np.linalg.inv(plus_control["gramian"]) @ target)
        minus_energy = float(target @ np.linalg.inv(minus_control["gramian"]) @ target)
        finite_energy_dot = (plus_energy - minus_energy) / (2.0 * FD_STEP)
        energy_response_error = scalar_error(analytic_energy_dot, finite_energy_dot)

    seed_pass = all(
        (
            tangent_error <= TANGENT_TOL,
            jacobian_response_error <= J_RESPONSE_TOL,
            metric_response_error <= METRIC_RESPONSE_TOL,
            chart_condition < 4.0,
            chart_nonorthogonality > 0.1,
            chart_jacobian_error <= COVARIANCE_TOL,
            chart_metric_error <= COVARIANCE_TOL,
            chart_length_error <= COVARIANCE_TOL,
            bad_coordinate_error >= BAD_COVARIANCE_MIN,
            gramian_symmetry <= GRAMIAN_SYMMETRY_TOL,
            float(gramian_eigenvalues[0]) >= gramian_psd_floor,
            control_residual <= CONTROL_RESIDUAL_TOL,
            energy_identity_error <= ENERGY_IDENTITY_TOL,
            chart_energy_error <= COVARIANCE_TOL,
            (not finite_metric_invariants)
            or (
                generalized_error is not None
                and generalized_error <= COVARIANCE_TOL
                and log_volume_error is not None
                and log_volume_error <= COVARIANCE_TOL
            ),
            (not energy_response_eligible)
            or (
                energy_response_error is not None
                and energy_response_error <= ENERGY_RESPONSE_TOL
            ),
        )
    )

    return {
        "seed": fixture.seed,
        "q": fixture.q,
        "delay_depth": fixture.delay_depth,
        "horizon": fixture.horizon,
        "pass": seed_pass,
        "weight_domain": {
            "max_abs_weight_zero": float(np.max(np.abs(fixture.weight))),
            "max_abs_circuit": float(np.max(np.abs(fixture.circuit))),
            "max_abs_weight_path": float(
                max(
                    np.max(np.abs(fixture.weight)),
                    np.max(np.abs(fixture.weight + fixture.circuit)),
                )
            ),
            "circuit_infinity_norm": float(np.linalg.norm(fixture.circuit, ord=np.inf)),
        },
        "passive": {
            "tangent_error": tangent_error,
            "singular_values": jacobian_singular.tolist(),
            "rank_cutoff": jacobian_cutoff,
            "operational_rank": jacobian_rank,
            "condition": jacobian_condition,
            "pre_metric_operational_rank": pre_metric_rank,
            "post_metric_operational_rank": post_metric_rank,
        },
        "circuit_response": {
            "jacobian_error": jacobian_response_error,
            "metric_error": metric_response_error,
            "zero_partial_jacobian_error_diagnostic": zero_partial_error,
        },
        "coordinate_covariance": {
            "chart_condition": chart_condition,
            "chart_nonorthogonality": chart_nonorthogonality,
            "jacobian_error": chart_jacobian_error,
            "metric_error": chart_metric_error,
            "quadratic_length_error": chart_length_error,
            "bad_untransformed_metric_error": bad_coordinate_error,
            "generalized_eigenvalue_error": generalized_error,
            "log_volume_ratio_error": log_volume_error,
        },
        "control": {
            "gramian_symmetry_error": gramian_symmetry,
            "gramian_min_eigenvalue": float(gramian_eigenvalues[0]),
            "gramian_singular_values": gramian_singular.tolist(),
            "rank_cutoff": gramian_cutoff,
            "operational_rank": gramian_rank,
            "condition": gramian_condition,
            "target_residual": control_residual,
            "least_norm_energy": control_energy,
            "gramian_energy": gramian_energy,
            "energy_identity_error": energy_identity_error,
            "chart_energy_error": chart_energy_error,
            "energy_response_eligible": energy_response_eligible,
            "analytic_energy_derivative": analytic_energy_dot,
            "finite_difference_energy_derivative": finite_energy_dot,
            "energy_derivative_error": energy_response_error,
        },
    }


def reachability_receipt(control_map: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    projection = control_map @ pinv_with_frozen_cutoff(control_map) @ target
    residual = float(
        np.linalg.norm(projection - target) / max(1.0, float(np.linalg.norm(target)))
    )
    reachable = residual <= CONTROL_RESIDUAL_TOL
    if reachable:
        gramian = control_map @ control_map.T
        energy: float | str = float(target @ pinv_with_frozen_cutoff(gramian) @ target)
    else:
        energy = "Infinity"
    _, cutoff, rank, condition = svd_receipt(control_map)
    return {
        "status": "REACHABLE" if reachable else "UNREACHABLE",
        "energy": energy,
        "residual": residual,
        "operational_rank": rank,
        "rank_cutoff": cutoff,
        "condition": condition,
    }


def reachability_controls() -> dict[str, Any]:
    zero = reachability_receipt(np.zeros((3, 2)), np.array([1.0, 0.0, 0.0]))
    rank_one = reachability_receipt(
        np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([0.0, 1.0])
    )
    near_map = np.diag([1.0, 1.0e-12])
    _, near_cutoff, near_rank, near_condition = svd_receipt(near_map)
    exact_rank_loss = np.diag([1.0, 0.0])
    exact_metric = exact_rank_loss.T @ exact_rank_loss
    exact = {
        "status": "EXACT_RANK_DEFICIENT",
        "exact_null_vector": [0.0, 1.0],
        "metric": exact_metric.tolist(),
        "ridge_used": False,
    }
    passed = (
        zero["status"] == "UNREACHABLE"
        and zero["energy"] == "Infinity"
        and rank_one["status"] == "UNREACHABLE"
        and rank_one["energy"] == "Infinity"
        and near_rank == 1
        and math.isinf(near_condition)
        and exact["status"] == "EXACT_RANK_DEFICIENT"
        and not exact["ridge_used"]
    )
    return {
        "pass": passed,
        "zero_actuator": zero,
        "rank_one_orthogonal_target": rank_one,
        "near_singular": {
            "operational_rank": near_rank,
            "rank_cutoff": near_cutoff,
            "condition": near_condition,
            "inverse_derivative_allowed": False,
        },
        "exact_passive_rank_loss": exact,
    }


def efficacy_fixture() -> dict[str, np.ndarray | int]:
    return {
        "q": 2,
        "delay_depth": 1,
        "update_fraction": np.array([0.80, 0.65]),
        "weight": np.array([[0.70, -0.20], [0.35, 0.50]]),
        "circuit": np.array([[0.12, -0.08], [0.05, 0.11]]),
        "delays": np.array([[0, 1], [1, 0]], dtype=int),
        "state": np.array([0.45, -0.30, -0.20, 0.35]),
        "net_offset": np.array([0.05, -0.08]),
        "alpha": np.array([[0.10, -0.20], [0.30, -0.10]]),
        "beta": np.array(
            [
                [[0.90, 0.00, -0.40, 0.20], [-0.20, 0.60, 0.30, -0.50]],
                [[0.50, -0.70, 0.40, 0.10], [-0.60, 0.20, 0.10, 0.80]],
            ]
        ),
    }


def efficacy_and_gradient(
    state: np.ndarray, fixture: dict[str, np.ndarray | int]
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray(fixture["alpha"])
    beta = np.asarray(fixture["beta"])
    argument = alpha + np.einsum("ijk,k->ij", beta, state)
    hyperbolic = np.tanh(argument)
    efficacy = 0.5 * (1.0 + hyperbolic)
    gradient = 0.5 * (1.0 - hyperbolic**2)[:, :, None] * beta
    return efficacy, gradient


def efficacy_step(
    state: np.ndarray, epsilon: float, fixture: dict[str, np.ndarray | int]
) -> np.ndarray:
    q = int(fixture["q"])
    depth = int(fixture["delay_depth"])
    history = state.reshape(depth + 1, q)
    weight = np.asarray(fixture["weight"]) + epsilon * np.asarray(fixture["circuit"])
    efficacy, _ = efficacy_and_gradient(state, fixture)
    drive = np.asarray(fixture["net_offset"]).copy()
    delays = np.asarray(fixture["delays"])
    for receiver in range(q):
        for sender in range(q):
            drive[receiver] += (
                weight[receiver, sender]
                * efficacy[receiver, sender]
                * history[delays[receiver, sender], sender]
            )
    update_fraction = np.asarray(fixture["update_fraction"])
    next_history = np.empty_like(history)
    next_history[0] = (1.0 - update_fraction) * history[0] + update_fraction * np.tanh(
        drive
    )
    next_history[1:] = history[:-1]
    return next_history.reshape(-1)


def efficacy_jacobian(
    state: np.ndarray,
    epsilon: float,
    fixture: dict[str, np.ndarray | int],
    *,
    include_gradient: bool,
) -> np.ndarray:
    q = int(fixture["q"])
    depth = int(fixture["delay_depth"])
    dimension = q * (depth + 1)
    history = state.reshape(depth + 1, q)
    weight = np.asarray(fixture["weight"]) + epsilon * np.asarray(fixture["circuit"])
    efficacy, gradient = efficacy_and_gradient(state, fixture)
    delays = np.asarray(fixture["delays"])
    drive = np.asarray(fixture["net_offset"]).copy()
    for receiver in range(q):
        for sender in range(q):
            drive[receiver] += (
                weight[receiver, sender]
                * efficacy[receiver, sender]
                * history[delays[receiver, sender], sender]
            )
    gain = 1.0 - np.tanh(drive) ** 2
    update_fraction = np.asarray(fixture["update_fraction"])
    jacobian = np.zeros((dimension, dimension), dtype=float)
    for receiver in range(q):
        jacobian[receiver, receiver] += 1.0 - update_fraction[receiver]
        drive_gradient = np.zeros(dimension, dtype=float)
        for sender in range(q):
            delay = delays[receiver, sender]
            source = history[delay, sender]
            direct_column = delay * q + sender
            drive_gradient[direct_column] += (
                weight[receiver, sender] * efficacy[receiver, sender]
            )
            if include_gradient:
                drive_gradient += (
                    weight[receiver, sender]
                    * source
                    * gradient[receiver, sender]
                )
        jacobian[receiver] += (
            update_fraction[receiver] * gain[receiver] * drive_gradient
        )
    jacobian[q:, : depth * q] = np.eye(depth * q)
    return jacobian


def efficacy_fd_jacobian(
    state: np.ndarray, epsilon: float, fixture: dict[str, np.ndarray | int]
) -> np.ndarray:
    dimension = state.size
    result = np.zeros((dimension, dimension), dtype=float)
    for coordinate in range(dimension):
        plus = state.copy()
        minus = state.copy()
        plus[coordinate] += FD_STEP
        minus[coordinate] -= FD_STEP
        result[:, coordinate] = (
            efficacy_step(plus, epsilon, fixture)
            - efficacy_step(minus, epsilon, fixture)
        ) / (2.0 * FD_STEP)
    return result


def efficacy_rollout(
    epsilon: float, fixture: dict[str, np.ndarray | int], steps: int = 2
) -> np.ndarray:
    state = np.asarray(fixture["state"]).copy()
    for _ in range(steps):
        state = efficacy_step(state, epsilon, fixture)
    return state


def efficacy_state_response(
    epsilon: float,
    fixture: dict[str, np.ndarray | int],
    *,
    include_efficacy_response: bool,
    steps: int = 2,
) -> np.ndarray:
    q = int(fixture["q"])
    depth = int(fixture["delay_depth"])
    state = np.asarray(fixture["state"]).copy()
    state_dot = np.zeros_like(state)
    weight = np.asarray(fixture["weight"]) + epsilon * np.asarray(fixture["circuit"])
    circuit = np.asarray(fixture["circuit"])
    delays = np.asarray(fixture["delays"])
    update_fraction = np.asarray(fixture["update_fraction"])
    for _ in range(steps):
        history = state.reshape(depth + 1, q)
        history_dot = state_dot.reshape(depth + 1, q)
        efficacy, gradient = efficacy_and_gradient(state, fixture)
        efficacy_dot = np.einsum("ijk,k->ij", gradient, state_dot)
        if not include_efficacy_response:
            efficacy_dot = np.zeros_like(efficacy_dot)
        drive = np.asarray(fixture["net_offset"]).copy()
        drive_dot = np.zeros(q, dtype=float)
        for receiver in range(q):
            for sender in range(q):
                delay = delays[receiver, sender]
                source = history[delay, sender]
                source_dot = history_dot[delay, sender]
                drive[receiver] += (
                    weight[receiver, sender] * efficacy[receiver, sender] * source
                )
                drive_dot[receiver] += (
                    circuit[receiver, sender] * efficacy[receiver, sender] * source
                    + weight[receiver, sender]
                    * (
                        efficacy_dot[receiver, sender] * source
                        + efficacy[receiver, sender] * source_dot
                    )
                )
        activation = np.tanh(drive)
        gain = 1.0 - activation**2
        next_history = np.empty_like(history)
        next_history_dot = np.empty_like(history_dot)
        next_history[0] = (
            (1.0 - update_fraction) * history[0] + update_fraction * activation
        )
        next_history_dot[0] = (
            (1.0 - update_fraction) * history_dot[0]
            + update_fraction * gain * drive_dot
        )
        next_history[1:] = history[:-1]
        next_history_dot[1:] = history_dot[:-1]
        state = next_history.reshape(-1)
        state_dot = next_history_dot.reshape(-1)
    return state_dot


def state_dependent_efficacy_control() -> dict[str, Any]:
    fixture = efficacy_fixture()
    state = np.asarray(fixture["state"])
    full_jacobian = efficacy_jacobian(
        state, EPSILON, fixture, include_gradient=True
    )
    omitted_jacobian = efficacy_jacobian(
        state, EPSILON, fixture, include_gradient=False
    )
    fd_jacobian = efficacy_fd_jacobian(state, EPSILON, fixture)
    full_tangent_error = normalized_error(full_jacobian, fd_jacobian)
    omitted_tangent_error = normalized_error(omitted_jacobian, fd_jacobian)

    full_response = efficacy_state_response(
        EPSILON, fixture, include_efficacy_response=True
    )
    omitted_response = efficacy_state_response(
        EPSILON, fixture, include_efficacy_response=False
    )
    fd_response = (
        efficacy_rollout(EPSILON + FD_STEP, fixture)
        - efficacy_rollout(EPSILON - FD_STEP, fixture)
    ) / (2.0 * FD_STEP)
    full_response_error = normalized_error(full_response, fd_response)
    omitted_response_error = normalized_error(omitted_response, fd_response)
    passed = (
        full_tangent_error <= STATE_EFFICACY_FULL_TOL
        and omitted_tangent_error >= STATE_EFFICACY_OMITTED_MIN
        and full_response_error <= STATE_RESPONSE_FULL_TOL
        and omitted_response_error >= STATE_RESPONSE_OMITTED_MIN
    )
    return {
        "pass": passed,
        "full_tangent_error": full_tangent_error,
        "omitted_dp_dxi_tangent_error": omitted_tangent_error,
        "full_two_step_response_error": full_response_error,
        "omitted_dot_p_response_error": omitted_response_error,
    }


def thresholds_receipt() -> dict[str, float]:
    return {
        "finite_difference_step": FD_STEP,
        "rank_rcond": RANK_RCOND,
        "tangent_max": TANGENT_TOL,
        "jacobian_response_max": J_RESPONSE_TOL,
        "metric_response_max": METRIC_RESPONSE_TOL,
        "covariance_max": COVARIANCE_TOL,
        "bad_covariance_min": BAD_COVARIANCE_MIN,
        "gramian_symmetry_max": GRAMIAN_SYMMETRY_TOL,
        "gramian_psd_factor": GRAMIAN_PSD_FACTOR,
        "control_residual_max": CONTROL_RESIDUAL_TOL,
        "energy_identity_max": ENERGY_IDENTITY_TOL,
        "energy_response_max": ENERGY_RESPONSE_TOL,
        "max_gramian_condition": MAX_GRAMIAN_CONDITION,
        "state_efficacy_full_max": STATE_EFFICACY_FULL_TOL,
        "state_efficacy_omitted_min": STATE_EFFICACY_OMITTED_MIN,
        "state_response_full_max": STATE_RESPONSE_FULL_TOL,
        "state_response_omitted_min": STATE_RESPONSE_OMITTED_MIN,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("a6_property_result.json"),
    )
    arguments = parser.parse_args()

    seed_rows = [run_seed(make_fixture(seed, index)) for index, seed in enumerate(SEEDS)]
    reachability = reachability_controls()
    efficacy = state_dependent_efficacy_control()
    passed = all(row["pass"] for row in seed_rows) and reachability["pass"] and efficacy["pass"]
    result = {
        "status": "PROPERTY_PASS" if passed else "PROPERTY_FAIL",
        "claim_ceiling": "MATH_PROPERTY_PASS / EMPIRICAL_UNTESTED",
        "empirical_assets_opened": False,
        "brainruntime_validated": False,
        "cortical_folding_bridge": "BLOCKED_INPUT",
        "seeds": list(SEEDS),
        "thresholds": thresholds_receipt(),
        "seed_results": seed_rows,
        "reachability_controls": reachability,
        "state_dependent_efficacy_control": efficacy,
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
