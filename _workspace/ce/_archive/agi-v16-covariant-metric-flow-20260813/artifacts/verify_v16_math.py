"""Deterministic killing tests for the V16 metric-flow proof lane."""

from __future__ import annotations

import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


def update(g: np.ndarray, x: np.ndarray, cost: float, eta: float) -> np.ndarray:
    p = float(x @ g @ x)
    a = math.exp(-eta * math.log(p / cost)) - 1.0
    gx = g @ x
    return g + (a / p) * np.outer(gx, gx)


def airm_exponential_step(
    g: np.ndarray, x: np.ndarray, cost: float, eta: float
) -> np.ndarray:
    p = float(x @ g @ x)
    r = math.log(p / cost)
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    unit = root @ x / math.sqrt(p)
    projector = np.outer(unit, unit)
    # exp(-eta*r*projector) = I + (exp(-eta*r)-1)*projector.
    middle = np.eye(g.shape[0]) + math.expm1(-eta * r) * projector
    return root @ middle @ root


def burg_divergence(target: np.ndarray, metric: np.ndarray) -> float:
    relative = np.linalg.solve(metric, target)
    sign, log_determinant = np.linalg.slogdet(relative)
    if sign <= 0:
        raise ValueError("relative SPD determinant must be positive")
    return float(np.trace(relative) - log_determinant - metric.shape[0])


def main() -> None:
    rng = np.random.default_rng(160013)
    maxima = {
        "airm_relative_error": 0.0,
        "affine_relative_error": 0.0,
        "contraction_absolute_error": 0.0,
        "determinant_relative_error": 0.0,
        "burg_decrement_absolute_error": 0.0,
    }
    minimum_eigenvalue = math.inf

    for dimension in (2, 3, 4):
        for _ in range(256):
            factor = rng.normal(size=(dimension, dimension))
            g = factor @ factor.T + 0.2 * np.eye(dimension)
            x = rng.normal(size=dimension)
            cost = float(math.exp(rng.uniform(-2.0, 2.0)))
            eta = float(rng.uniform(1e-3, 1.0))
            p = float(x @ g @ x)
            r = math.log(p / cost)
            updated = update(g, x, cost, eta)

            expected = airm_exponential_step(g, x, cost, eta)
            maxima["airm_relative_error"] = max(
                maxima["airm_relative_error"],
                float(np.linalg.norm(updated - expected) / np.linalg.norm(expected)),
            )

            jacobian = rng.normal(size=(dimension, dimension))
            while abs(np.linalg.det(jacobian)) < 0.1:
                jacobian = rng.normal(size=(dimension, dimension))
            inverse = np.linalg.inv(jacobian)
            transported = inverse.T @ g @ inverse
            affine_updated = update(transported, jacobian @ x, cost, eta)
            expected_affine = inverse.T @ updated @ inverse
            maxima["affine_relative_error"] = max(
                maxima["affine_relative_error"],
                float(
                    np.linalg.norm(affine_updated - expected_affine)
                    / np.linalg.norm(expected_affine)
                ),
            )

            next_prediction = float(x @ updated @ x)
            exact_prediction = math.exp((1.0 - eta) * math.log(p) + eta * math.log(cost))
            maxima["contraction_absolute_error"] = max(
                maxima["contraction_absolute_error"],
                abs(math.log(next_prediction / exact_prediction)),
            )

            exact_determinant = np.linalg.det(g) * math.exp(-eta * r)
            maxima["determinant_relative_error"] = max(
                maxima["determinant_relative_error"],
                abs(float(np.linalg.det(updated) / exact_determinant) - 1.0),
            )
            minimum_eigenvalue = min(
                minimum_eigenvalue, float(np.linalg.eigvalsh(updated)[0])
            )

            target_factor = rng.normal(size=(dimension, dimension))
            target = target_factor @ target_factor.T + 0.2 * np.eye(dimension)
            noiseless_cost = float(x @ target @ x)
            noiseless_prediction = float(x @ g @ x)
            z = noiseless_cost / noiseless_prediction
            noiseless_updated = update(g, x, noiseless_cost, eta)
            measured_decrement = burg_divergence(
                target, noiseless_updated
            ) - burg_divergence(target, g)
            exact_decrement = z ** (1.0 - eta) - z + eta * math.log(z)
            maxima["burg_decrement_absolute_error"] = max(
                maxima["burg_decrement_absolute_error"],
                abs(measured_decrement - exact_decrement),
            )

    # Identifiability: a nonspanning diagonal design cannot observe H.
    design = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    hidden_direction = np.array([[0.0, 1.0], [1.0, 0.0]])
    g_plus = np.eye(2) + 0.5 * hidden_direction
    g_minus = np.eye(2) - 0.5 * hidden_direction
    measurement_defect = max(
        abs(float(x @ g_plus @ x - x @ g_minus @ x)) for x in design
    )

    # One noiseless observation can increase invariant error to g_*=I.
    q = np.array([[1.0, 10.0], [10.0, -1.0]]) / math.sqrt(101.0)
    bad_g = q @ np.diag([400.0, 1.0 / 400.0]) @ q.T
    bad_x = np.array([9.5, -1.0])
    bad_cost = float(bad_x @ bad_x)
    bad_updated = update(bad_g, bad_x, bad_cost, 1.0)

    # Finite spanning family, cyclic schedule (bounded gap), noiseless target.
    dimension = 3
    spanning_directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
    ]
    convergence_target = np.array(
        [[2.0, 0.3, -0.2], [0.3, 0.8, 0.1], [-0.2, 0.1, 1.4]]
    )
    convergence_metric = np.eye(dimension)
    previous_divergence = burg_divergence(convergence_target, convergence_metric)
    maximum_divergence_increase = 0.0
    for step in range(12_000):
        direction = spanning_directions[step % len(spanning_directions)]
        cost = float(direction @ convergence_target @ direction)
        convergence_metric = update(convergence_metric, direction, cost, 0.4)
        next_divergence = burg_divergence(convergence_target, convergence_metric)
        maximum_divergence_increase = max(
            maximum_divergence_increase, next_divergence - previous_divergence
        )
        previous_divergence = next_divergence

    def invariant_rms(metric: np.ndarray) -> float:
        logs = np.log(np.linalg.eigvalsh(metric))
        return float(np.sqrt(np.mean(logs * logs)))

    output = {
        "random_trials": 3 * 256,
        **maxima,
        "minimum_updated_eigenvalue": minimum_eigenvalue,
        "nonspanning_measurement_defect": measurement_defect,
        "nonspanning_distinct_metric_norm": float(np.linalg.norm(g_plus - g_minus)),
        "one_step_counterexample": {
            "initial_invariant_rms": invariant_rms(bad_g),
            "updated_invariant_rms": invariant_rms(bad_updated),
            "initial_eigenvalues": np.linalg.eigvalsh(bad_g).tolist(),
            "updated_eigenvalues": np.linalg.eigvalsh(bad_updated).tolist(),
        },
        "noiseless_bounded_gap_fixture": {
            "steps": 12_000,
            "bounded_gap": len(spanning_directions),
            "final_burg_divergence": previous_divergence,
            "maximum_burg_increase": maximum_divergence_increase,
            "final_frobenius_error": float(
                np.linalg.norm(convergence_metric - convergence_target)
            ),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    log_path = Path(__file__).with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
        main()
    print(log_path.read_text(encoding="utf-8"), end="")
