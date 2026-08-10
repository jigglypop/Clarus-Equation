#!/usr/bin/env python
"""Deterministic math certificates for the exploratory Riemannian gear model.

This is not biological validation. It checks only pair locking, cycle
frustration, discrete contraction/ISS, and spectral truncation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Certificate:
    pair_exact_error: float
    frustration_projection_error: float
    iss_bound_margin: float
    spectral_bound_margin: float


def pair_residual_exact(delta_0: float, coupling_rate: float, time: float) -> float:
    """Solve delta_dot = -coupling_rate * sin(delta) on its principal branch."""
    return 2.0 * math.atan(math.tan(delta_0 / 2.0) * math.exp(-coupling_rate * time))


def minimum_frustration(
    incidence: np.ndarray,
    stiffness: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return F_k, one least-squares minimizer, and the cycle-space residual."""
    sqrt_stiffness = np.sqrt(stiffness)
    operator = sqrt_stiffness @ incidence
    weighted_target = sqrt_stiffness @ target
    theta_star = np.linalg.pinv(operator) @ weighted_target
    residual = weighted_target - operator @ theta_star
    return 0.5 * float(residual @ residual), theta_star, residual


def run_certificate() -> Certificate:
    # Pair theorem: compare the exact solution with RK4 integration.
    delta_0, coupling_rate, horizon = 1.1, 3.2, 2.0
    steps = 20_000
    step = horizon / steps
    delta = delta_0

    def field(value: float) -> float:
        return -coupling_rate * math.sin(value)

    for _ in range(steps):
        k1 = field(delta)
        k2 = field(delta + 0.5 * step * k1)
        k3 = field(delta + 0.5 * step * k2)
        k4 = field(delta + step * k3)
        delta += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    pair_error = abs(delta - pair_residual_exact(delta_0, coupling_rate, horizon))

    # An inconsistent triangle has a nonzero projection onto ker(A^T).
    incidence = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]])
    stiffness = np.diag([1.0, 2.0, 1.5])
    target = np.array([0.3, -0.1, 0.7])
    frustration, theta_star, residual = minimum_frustration(incidence, stiffness, target)
    quadratic = 0.5 * np.linalg.norm(
        np.sqrt(stiffness) @ (incidence @ theta_star - target)
    ) ** 2
    projection_error = abs(frustration - quadratic) + float(
        np.linalg.norm((np.sqrt(stiffness) @ incidence).T @ residual)
    )

    # Discrete contraction under bounded forcing.
    hessian = np.diag([1.0, 3.0])
    eta = 0.4
    transition = np.eye(2) - eta * hessian
    contraction = float(np.linalg.norm(transition, ord=2))
    forcing = np.array([0.01, -0.02])
    forcing_bound = float(np.linalg.norm(forcing))
    state = np.array([0.8, -0.4])
    initial_norm = float(np.linalg.norm(state))
    for _ in range(50):
        state = transition @ state + forcing
    iss_bound = contraction**50 * initial_norm + (1.0 - contraction**50) * forcing_bound / (
        1.0 - contraction
    )
    iss_margin = iss_bound - float(np.linalg.norm(state))

    # Discard modes 2 and 3 and compare with the spectral tail bound.
    eigenvalues = np.array([0.5, 1.5, 4.0])
    initial_modes = np.array([1.0, -0.7, 0.2])
    time = 1.3
    evolved = np.exp(-eigenvalues * time) * initial_modes
    truncation_error = float(np.linalg.norm(evolved[1:]))
    spectral_bound = math.exp(-eigenvalues[1] * time) * float(np.linalg.norm(initial_modes))
    spectral_margin = spectral_bound - truncation_error

    return Certificate(pair_error, projection_error, iss_margin, spectral_margin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()
    result = run_certificate()
    payload = asdict(result)
    payload["pass"] = (
        result.pair_exact_error <= args.tol
        and result.frustration_projection_error <= args.tol
        and result.iss_bound_margin >= -args.tol
        and result.spectral_bound_margin >= -args.tol
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
