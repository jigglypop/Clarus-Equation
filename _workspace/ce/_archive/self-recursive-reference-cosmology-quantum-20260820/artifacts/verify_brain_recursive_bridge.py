"""Focused verification for the typed brain/branching/quantum bridge.

This script checks mathematical witnesses only.  It does not certify a
biological mapping, a quantum-to-genealogy bridge, or a cosmological readout.
"""

from __future__ import annotations

import json
import math
from typing import Callable


TOL = 1.0e-12


def poisson_map(x: float, mean: float) -> float:
    return math.exp(-mean * (1.0 - x))


def iterate_map(
    fn: Callable[[float], float], x0: float, steps: int
) -> list[float]:
    values = [x0]
    for _ in range(steps):
        values.append(fn(values[-1]))
    return values


def bisection_small_root(mean: float) -> float:
    if mean <= 1.0:
        return 1.0
    lo = 0.0
    hi = 1.0 / mean
    for _ in range(200):
        mid = (lo + hi) / 2.0
        residual = mid - poisson_map(mid, mean)
        if residual > 0.0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def clip_unit(value: float) -> float:
    return min(1.0, max(-1.0, value))


def activation_step(
    activation: float,
    drive: float,
    gamma: float,
    kappa: float,
    *,
    projected: bool,
) -> float:
    candidate = (1.0 - gamma) * activation + kappa * math.tanh(drive)
    return clip_unit(candidate) if projected else candidate


def dale_sender_columns(weight: list[list[float]], signs: list[int]) -> bool:
    for row in weight:
        for sender, value in enumerate(row):
            if value != 0.0 and math.copysign(1.0, value) != signs[sender]:
                return False
    return True


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(value * component for value, component in zip(row, vector))
        for row in matrix
    ]


def spectral_radius_2x2(matrix: list[list[float]]) -> float:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("closed-form verifier accepts exactly a 2x2 matrix")
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace * trace - 4.0 * determinant
    if discriminant < -TOL:
        real = trace / 2.0
        imaginary = math.sqrt(-discriminant) / 2.0
        return math.hypot(real, imaginary)
    root = math.sqrt(max(0.0, discriminant))
    eigenvalues = ((trace + root) / 2.0, (trace - root) / 2.0)
    return max(abs(value) for value in eigenvalues)


def stp_step(
    resource: float,
    utilization: float,
    event: int,
    dt_over_rec: float,
    dt_over_fac: float,
    baseline_utilization: float,
) -> tuple[float, float]:
    next_resource = clip_unit(
        resource
        + dt_over_rec * (1.0 - resource)
        - utilization * resource * event
    )
    next_resource = min(1.0, max(0.0, next_resource))
    next_utilization = (
        utilization
        + dt_over_fac * (baseline_utilization - utilization)
        + baseline_utilization * (1.0 - utilization) * event
    )
    next_utilization = min(1.0, max(0.0, next_utilization))
    return next_resource, next_utilization


def causal_eligibility_once(
    previous_pre: list[float],
    previous_post: list[float],
    current_event: list[float],
    ltp: float,
    ltd: float,
) -> list[list[float]]:
    return [
        [
            ltp * current_event[post] * previous_pre[pre]
            - ltd * previous_post[post] * current_event[pre]
            for pre in range(len(current_event))
        ]
        for post in range(len(current_event))
    ]


def branching_map(q: list[float], matrix: list[list[float]]) -> list[float]:
    # matrix[parent][child]
    return [
        math.exp(
            -sum(
                matrix[parent][child] * (1.0 - q[child])
                for child in range(len(q))
            )
        )
        for parent in range(len(q))
    ]


def branching_iterates(
    matrix: list[list[float]], generations: int
) -> list[list[float]]:
    q = [0.0] * len(matrix)
    values = [q]
    for _ in range(generations):
        q = branching_map(q, matrix)
        values.append(q)
    return values


def matrix_close(
    left: list[list[float]], right: list[list[float]], tol: float = TOL
) -> bool:
    return all(
        abs(a - b) <= tol
        for row_a, row_b in zip(left, right)
        for a, b in zip(row_a, row_b)
    )


def heterogeneous_event(
    activation: float,
    baseline: float,
    scale: float,
    threshold: float,
    saturation: float,
    scale_floor: float = 1.0e-6,
) -> tuple[int, float]:
    standardized = (abs(activation) - baseline) / max(scale, scale_floor)
    event = int(standardized >= threshold)
    excess = max(0.0, standardized - threshold)
    strength = event * min(1.0, excess / saturation)
    return event, strength


def circuit_strength(
    edge_lengths: list[float], edge_strengths: list[float]
) -> tuple[float, float]:
    if len(edge_lengths) != len(edge_strengths) or not edge_lengths:
        raise ValueError("circuit receipts must be nonempty and aligned")
    total_length = sum(edge_lengths)
    if total_length <= 0.0:
        raise ValueError("circuit length must be positive")
    mean = sum(
        length * strength
        for length, strength in zip(edge_lengths, edge_strengths)
    ) / total_length
    return mean, min(edge_strengths)


def expm_symmetric_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b - c) > TOL:
        raise ValueError("matrix exponential witness requires symmetry")
    mean = (a + d) / 2.0
    x = (a - d) / 2.0
    radius = math.hypot(x, b)
    scale = math.exp(mean)
    if radius <= TOL:
        return [[scale, 0.0], [0.0, scale]]
    diagonal = math.cosh(radius)
    off_scale = math.sinh(radius) / radius
    return [
        [scale * (diagonal + off_scale * x), scale * off_scale * b],
        [scale * off_scale * b, scale * (diagonal - off_scale * x)],
    ]


def x_conjugation(rho: list[list[float]]) -> list[list[float]]:
    return [[rho[1][1], rho[1][0]], [rho[0][1], rho[0][0]]]


def dephase(rho: list[list[float]]) -> list[list[float]]:
    return [[rho[0][0], 0.0], [0.0, rho[1][1]]]


def main() -> None:
    report: dict[str, object] = {}

    gamma = 0.18
    kappa = 0.82
    drive = 40.0
    unprojected = 0.0
    projected = 0.0
    for _ in range(2):
        unprojected = activation_step(
            unprojected, drive, gamma, kappa, projected=False
        )
        projected = activation_step(
            projected, drive, gamma, kappa, projected=True
        )
    assert unprojected > 1.0
    assert -1.0 <= projected <= 1.0
    report["activation_bound_counterexample"] = {
        "unprojected_after_two_ticks": unprojected,
        "projected_after_two_ticks": projected,
    }

    sender_signs = [1, -1, 1]
    sender_signed = [
        [0.2, -0.1, 0.3],
        [0.4, -0.5, 0.1],
    ]
    receiver_signed = [
        [0.2, 0.1, 0.3],
        [-0.4, -0.5, -0.1],
    ]
    assert dale_sender_columns(sender_signed, sender_signs)
    assert not dale_sender_columns(receiver_signed, sender_signs)
    report["dale_orientation"] = {
        "sender_column_pass": True,
        "receiver_row_counterexample": True,
    }

    subcritical_mean = 0.8
    subcritical = iterate_map(
        lambda value: poisson_map(value, subcritical_mean), 0.0, 1000
    )
    assert abs(subcritical[-1] - 1.0) < 1.0e-12
    finite_horizon_survival = 1.0 - subcritical[8]
    assert finite_horizon_survival > 0.0

    supercritical_mean = 1.2
    supercritical_root = bisection_small_root(supercritical_mean)
    assert 0.0 < supercritical_root < 1.0 / supercritical_mean
    assert abs(
        supercritical_root - poisson_map(supercritical_root, supercritical_mean)
    ) < TOL
    report["scalar_branching"] = {
        "subcritical_mean": subcritical_mean,
        "q_infinity": subcritical[-1],
        "finite_horizon_H8_survival": finite_horizon_survival,
        "supercritical_mean": supercritical_mean,
        "supercritical_small_root": supercritical_root,
    }

    offspring = [[0.35, 0.15], [0.10, 0.25]]
    radius = spectral_radius_2x2(offspring)
    assert radius < 1.0
    vector_history = branching_iterates(offspring, 1000)
    q_vector = vector_history[-1]
    assert max(abs(value - 1.0) for value in q_vector) < TOL
    report["multitype_branching"] = {
        "spectral_radius": radius,
        "q_infinity": q_vector,
        "H8_survival": [
            1.0 - value for value in branching_iterates(offspring, 8)[-1]
        ],
    }

    child_responsibility = [
        [0.6, 0.4],
        [0.0, 1.0],
        [0.3, 0.0],
    ]
    assert all(sum(row) <= 1.0 + TOL for row in child_responsibility)
    parent_counts = [2.0, 3.0]
    child_totals = [
        sum(row[parent] for row in child_responsibility)
        for parent in range(2)
    ]
    estimated_offspring = [
        child_totals[parent] / parent_counts[parent] for parent in range(2)
    ]
    assert all(value >= 0.0 for value in estimated_offspring)
    report["genealogy_receipt"] = {
        "no_double_count": True,
        "estimated_parent_offspring": estimated_offspring,
    }

    event_a = heterogeneous_event(
        activation=0.7,
        baseline=0.1,
        scale=0.2,
        threshold=2.0,
        saturation=2.0,
    )
    event_b = heterogeneous_event(
        activation=0.7,
        baseline=0.5,
        scale=0.1,
        threshold=2.5,
        saturation=2.0,
    )
    assert event_a[0] == 1
    assert abs(event_a[1] - 0.5) <= TOL
    assert event_b[0] == 0
    assert abs(event_b[1]) <= TOL

    heterogeneous_mean, heterogeneous_bottleneck = circuit_strength(
        [1.0, 1.0], [0.2, 0.8]
    )
    uniform_mean, uniform_bottleneck = circuit_strength(
        [1.0, 1.0], [0.5, 0.5]
    )
    assert abs(heterogeneous_mean - uniform_mean) <= TOL
    assert heterogeneous_bottleneck < uniform_bottleneck
    report["heterogeneous_event_and_circuit"] = {
        "neuron_a_event_strength": event_a,
        "neuron_b_event_strength": event_b,
        "equal_circuit_mean": heterogeneous_mean,
        "heterogeneous_bottleneck": heterogeneous_bottleneck,
        "uniform_bottleneck": uniform_bottleneck,
    }

    tangent_deformation = [[-0.2, 0.1], [0.1, 0.3]]
    exponential = expm_symmetric_2x2(tangent_deformation)
    # Congruence by sqrt(diag(4, 1)) gives the exponential metric witness.
    effective_metric = [
        [4.0 * exponential[0][0], 2.0 * exponential[0][1]],
        [2.0 * exponential[1][0], exponential[1][1]],
    ]
    determinant = (
        effective_metric[0][0] * effective_metric[1][1]
        - effective_metric[0][1] * effective_metric[1][0]
    )
    assert matrix_close(
        effective_metric,
        [
            [effective_metric[0][0], effective_metric[1][0]],
            [effective_metric[0][1], effective_metric[1][1]],
        ],
    )
    assert effective_metric[0][0] > 0.0
    assert determinant > 0.0
    report["functional_metric_spd"] = {
        "leading_minor": effective_metric[0][0],
        "determinant": determinant,
        "symmetric": True,
    }

    resource, utilization = stp_step(
        resource=0.05,
        utilization=0.95,
        event=1,
        dt_over_rec=0.5,
        dt_over_fac=0.5,
        baseline_utilization=0.8,
    )
    assert 0.0 <= resource <= 1.0
    assert 0.0 <= utilization <= 1.0
    report["stp_projection"] = {
        "resource": resource,
        "utilization": utilization,
    }

    eligibility = causal_eligibility_once(
        previous_pre=[1.0, 0.0],
        previous_post=[0.0, 0.0],
        current_event=[0.0, 1.0],
        ltp=1.0,
        ltd=1.0,
    )
    assert eligibility[1][0] == 1.0
    assert eligibility[0][1] == 0.0
    report["causal_eligibility_orientation"] = {
        "post_1_pre_0": eligibility[1][0],
        "post_0_pre_1": eligibility[0][1],
    }

    rho0 = [[1.0, 0.0], [0.0, 0.0]]
    rho1 = x_conjugation(rho0)
    rho2 = x_conjugation(rho1)
    assert not matrix_close(rho0, rho1)
    assert matrix_close(rho0, rho2)

    diagonal_a = [[0.8, 0.0], [0.0, 0.2]]
    diagonal_b = [[0.3, 0.0], [0.0, 0.7]]
    assert matrix_close(dephase(diagonal_a), diagonal_a)
    assert matrix_close(dephase(diagonal_b), diagonal_b)
    assert not matrix_close(diagonal_a, diagonal_b)
    report["quantum_channel_counterexamples"] = {
        "x_unitary_period_two": True,
        "dephasing_fixed_states_nonunique": True,
    }

    report["status"] = "PASS_MATH_WITNESSES_ONLY"
    report["implementation_parity"] = "BLOCKED_NOT_TESTED"
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
