"""Independent numerical fixtures for the Phase A shared-B context model.

This file is deliberately isolated from product/runtime code.  It checks only
the algebra recorded in 11-math.md and emits a deterministic JSON receipt.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def namespace_seed(master: int, role: str, index: int) -> int:
    payload = f"phase-a-v1|{master}|{role}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rng_for(master: int, role: str, index: int = 0) -> np.random.Generator:
    return np.random.default_rng(namespace_seed(master, role, index))


def stacked_design(
    states: list[np.ndarray], inputs: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts = len(states)
    n = states[0].shape[0]
    columns = sum(block.shape[1] for block in states)
    block_state = np.zeros((contexts * n, columns), dtype=float)
    input_rows = []
    start = 0
    for context, (state, control) in enumerate(zip(states, inputs, strict=True)):
        stop = start + state.shape[1]
        block_state[context * n : (context + 1) * n, start:stop] = state
        input_rows.append(control)
        start = stop
    stacked_input = np.concatenate(input_rows, axis=1)
    return np.vstack((block_state, stacked_input)), block_state, stacked_input


def full_rank_recovery() -> dict[str, float | int]:
    n, m, contexts, rows_per_context = 2, 1, 3, 9
    states: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    for context in range(contexts):
        states.append(rng_for(7001, "trajectory", context).normal(size=(n, rows_per_context)))
        inputs.append(rng_for(7001, "intervention", context).normal(size=(m, rows_per_context)))
    design, block_state, stacked_input = stacked_design(states, inputs)
    transitions = [
        np.array([[0.50, 0.10], [-0.20, 0.35]]),
        np.array([[0.20, -0.30], [0.15, 0.45]]),
        np.array([[-0.10, 0.25], [0.30, 0.20]]),
    ]
    shared_input = np.array([[0.70], [-0.40]])
    truth = np.hstack((*transitions, shared_input))
    outcomes = truth @ design
    estimate = np.linalg.lstsq(design.T, outcomes.T, rcond=None)[0].T

    projector = block_state.T @ np.linalg.inv(block_state @ block_state.T) @ block_state
    residual_input = stacked_input @ (np.eye(design.shape[1]) - projector)
    schur = residual_input @ residual_input.T
    error = float(np.max(np.abs(estimate - truth)))
    assert np.linalg.matrix_rank(design) == contexts * n + m
    assert np.min(np.linalg.eigvalsh(schur)) > 1e-10
    assert error < 1e-10
    return {
        "design_rank": int(np.linalg.matrix_rank(design)),
        "required_rank": contexts * n + m,
        "minimum_input_schur_eigenvalue": float(np.min(np.linalg.eigvalsh(schur))),
        "maximum_coefficient_error": error,
    }


def rank_deficiency_fixture() -> dict[str, float | int]:
    n, m, contexts, rows_per_context = 2, 1, 3, 7
    states: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    context_rows = [
        np.array([[1.0, -0.5]]),
        np.array([[0.25, 0.75]]),
        np.array([[-0.4, 0.6]]),
    ]
    for context in range(contexts):
        state = rng_for(7101, "trajectory", context).normal(size=(n, rows_per_context))
        states.append(state)
        inputs.append(context_rows[context] @ state)
    design, block_state, stacked_input = stacked_design(states, inputs)
    transitions = [
        np.array([[0.5, 0.1], [-0.2, 0.35]]),
        np.array([[0.2, -0.3], [0.15, 0.45]]),
        np.array([[-0.1, 0.25], [0.3, 0.2]]),
    ]
    shared_input = np.array([[0.7], [-0.4]])
    truth = np.hstack((*transitions, shared_input))
    direction = np.array([[0.37], [-0.21]])
    altered_transitions = [
        transition - direction @ row
        for transition, row in zip(transitions, context_rows, strict=True)
    ]
    alternative = np.hstack((*altered_transitions, shared_input + direction))
    outcome_gap = float(np.max(np.abs(truth @ design - alternative @ design)))
    coefficient_gap = float(np.max(np.abs(truth - alternative)))

    projector = block_state.T @ np.linalg.inv(block_state @ block_state.T) @ block_state
    residual_input = stacked_input @ (np.eye(design.shape[1]) - projector)
    assert all(np.linalg.matrix_rank(state) == n for state in states)
    assert np.linalg.matrix_rank(design) < contexts * n + m
    assert coefficient_gap > 0.1
    assert outcome_gap < 1e-10
    return {
        "each_context_state_rank": n,
        "design_rank": int(np.linalg.matrix_rank(design)),
        "required_rank": contexts * n + m,
        "residual_input_norm": float(np.linalg.norm(residual_input)),
        "alternative_coefficient_gap": coefficient_gap,
        "alternative_outcome_gap": outcome_gap,
    }


def similarity_no_go() -> dict[str, float | bool]:
    transition = np.diag([0.5, 1.0 / 3.0])
    mix = np.array([[1.0, 1.0], [0.0, 1.0]])
    mixed_transition = mix @ transition @ np.linalg.inv(mix)
    decoder = np.array([[1.0, 2.0]])
    mixed_decoder = decoder @ np.linalg.inv(mix)
    state = np.array([[2.0], [3.0]])
    mixed_state = mix @ state
    first = []
    second = []
    for _ in range(12):
        first.append(float((decoder @ state)[0, 0]))
        second.append(float((mixed_decoder @ mixed_state)[0, 0]))
        state = transition @ state
        mixed_state = mixed_transition @ mixed_state
    gap = float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
    original_off_diagonal = int(np.count_nonzero(transition - np.diag(np.diag(transition))))
    mixed_off_diagonal = int(
        np.count_nonzero(
            np.abs(mixed_transition - np.diag(np.diag(mixed_transition))) > 1e-14
        )
    )
    assert gap < 1e-12
    assert original_off_diagonal != mixed_off_diagonal
    return {
        "maximum_observation_gap": gap,
        "original_off_diagonal_support": original_off_diagonal,
        "mixed_off_diagonal_support": mixed_off_diagonal,
        "exact_edge_certificate_unknown_mix": False,
    }


def degrees_of_freedom_fixture() -> dict[str, int]:
    n, m, contexts, context_basis = 4, 2, 3, 2
    pooled = n * (n + m)
    shared = n * (contexts * n + m)
    separated = n * contexts * (n + m)
    bilinear = n * ((context_basis + 1) * n + m)
    assert shared - pooled == (contexts - 1) * n * n
    assert separated - shared == (contexts - 1) * n * m
    assert bilinear == shared  # q=K-1 is only a reparameterization.
    return {
        "pooled_A_shared_B": pooled,
        "context_A_shared_B": shared,
        "context_A_context_B": separated,
        "bilinear_full_contrast": bilinear,
        "shared_minus_pooled": shared - pooled,
        "separated_minus_shared": separated - shared,
    }


def fixed_scale_nll_fixture() -> dict[str, float]:
    sigma = 0.2
    pooled_residual = np.array([[0.4, -0.2], [0.3, 0.1], [-0.1, 0.2]])
    context_residual = np.array([[0.1, -0.1], [0.2, 0.0], [-0.1, 0.1]])

    def nll(residual: np.ndarray) -> float:
        scalar_count = residual.size
        return float(
            0.5 * scalar_count * math.log(2.0 * math.pi * sigma * sigma)
            + np.sum(residual * residual) / (2.0 * sigma * sigma)
        )

    difference = nll(pooled_residual) - nll(context_residual)
    sse_identity = (
        np.sum(pooled_residual * pooled_residual)
        - np.sum(context_residual * context_residual)
    ) / (2.0 * sigma * sigma)
    assert abs(difference - sse_identity) < 1e-12
    return {
        "common_manifest_sigma": sigma,
        "pooled_minus_context_nll": float(difference),
        "sse_identity": float(sse_identity),
    }


def seed_namespace_fixture() -> dict[str, object]:
    master = 9001
    roles = ("graph", "trajectory", "intervention")
    seeds = {role: namespace_seed(master, role, 0) for role in roles}
    draws = {role: rng_for(master, role).normal(size=4).tolist() for role in roles}
    replay = {role: rng_for(master, role).normal(size=4).tolist() for role in roles}
    assert draws == replay
    assert len(set(seeds.values())) == len(roles)
    assert all(draws[left] != draws[right] for left in roles for right in roles if left < right)
    return {
        "namespace_seeds": seeds,
        "deterministic_replay": draws == replay,
        "distinct_streams": True,
        "claim": "domain-separated pseudorandom streams; statistical independence assumes the PRNG model",
    }


def main() -> None:
    report = {
        "status": "PASS",
        "full_rank_recovery": full_rank_recovery(),
        "rank_deficiency": rank_deficiency_fixture(),
        "similarity_no_go": similarity_no_go(),
        "degrees_of_freedom": degrees_of_freedom_fixture(),
        "fixed_scale_nll": fixed_scale_nll_fixture(),
        "seed_namespaces": seed_namespace_fixture(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
