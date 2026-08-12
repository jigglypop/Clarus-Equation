"""Independent checks for AGI V15 unified-metric math claims.

This script checks only finite algebraic/counterexample obligations.  It does
not implement the proposed agent and does not constitute an AGI benchmark.
"""

from __future__ import annotations

import math

import numpy as np


TOL = 1.0e-12


def spectral_clip(g: np.ndarray, lo: float, hi: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(g)
    return (vectors * np.clip(values, lo, hi)) @ vectors.T


def quadratic(v: np.ndarray, g: np.ndarray) -> float:
    return float(v @ g @ v)


def main() -> None:
    rng = np.random.default_rng(20260813)

    print("== UM-1: affine tensor covariance ==")
    max_relative_error = 0.0
    for _ in range(100):
        l = rng.normal(size=(3, 3))
        g_x = l @ l.T + 0.3 * np.eye(3)
        a = rng.normal(size=(3, 3))
        while abs(np.linalg.det(a)) < 0.2:
            a = rng.normal(size=(3, 3))
        a_inv = np.linalg.inv(a)
        g_y = a_inv.T @ g_x @ a_inv
        v_x = rng.normal(size=3)
        v_y = a @ v_x
        q_x = quadratic(v_x, g_x)
        q_y = quadratic(v_y, g_y)
        relative_error = abs(q_x - q_y) / max(abs(q_x), 1.0)
        max_relative_error = max(max_relative_error, relative_error)
    print(f"max relative quadratic-length error = {max_relative_error:.3e}")
    assert max_relative_error < 1.0e-12

    print("\n== UM-1/UM-2 boundary: spectral clipping is not affine-covariant ==")
    lo, hi = 0.1, 10.0
    g_x = np.eye(2)
    a = np.diag([10.0, 1.0])
    a_inv = np.linalg.inv(a)
    g_y = a_inv.T @ g_x @ a_inv
    g_y_clipped = spectral_clip(g_y, lo, hi)
    v_x = np.array([1.0, 0.0])
    v_y = a @ v_x
    q_x = quadratic(v_x, g_x)
    q_y_before = quadratic(v_y, g_y)
    q_y_after = quadratic(v_y, g_y_clipped)
    covariance_defect = abs(q_y_after - q_x)
    print(f"q_x={q_x:.6f}, q_y(before clip)={q_y_before:.6f}, q_y(after clip)={q_y_after:.6f}")
    print(f"clipping covariance defect = {covariance_defect:.6f}")
    assert abs(q_y_before - q_x) < TOL
    assert covariance_defect > 1.0

    print("\n== UM-2: LL^T + epsilon I and fixed-chart spectral certificate ==")
    l = rng.normal(size=(5, 3))
    epsilon = 0.07
    g = l @ l.T + epsilon * np.eye(5)
    eig = np.linalg.eigvalsh(g)
    clipped = spectral_clip(g, 0.1, 2.0)
    clipped_eig = np.linalg.eigvalsh(clipped)
    condition = float(clipped_eig[-1] / clipped_eig[0])
    print(f"lambda_min(LL^T+eps I)={eig[0]:.12f} >= eps={epsilon}")
    print(f"clipped eigen range=[{clipped_eig[0]:.12f}, {clipped_eig[-1]:.12f}], cond={condition:.6f}")
    assert eig[0] >= epsilon - TOL
    assert clipped_eig[0] >= 0.1 - TOL
    assert clipped_eig[-1] <= 2.0 + TOL
    assert condition <= 20.0 + TOL

    print("\n== UM-3 complete-branch counterexample: bounded source need not be L2 ==")
    print("M=R^d, r=1, phi0=0 gives phi(t)=(1-exp(-lambda*t))/lambda, a nonzero constant")
    print("hence ||phi(t)||_L2(R^d)=infinity for every t>0 although r is bounded and nonnegative")

    print("\n== UM-3 time-varying-metric counterexample ==")
    dimension = 2
    metric_scale_rate = 2.0
    lam = 1.0
    t = 1.0
    # g_t = exp(2*c*t) g_0, so dmu_t/dmu_0 = exp(c*d*t).
    trace_rate = 2.0 * metric_scale_rate * dimension
    energy_ratio = math.exp((metric_scale_rate * dimension - 2.0 * lam) * t)
    print(f"tr_g(dot g)={trace_rate:.6f}, 4*lambda={4.0 * lam:.6f}")
    print(f"constant-mode E(t)/E(0)={energy_ratio:.12f}")
    assert trace_rate > 4.0 * lam
    assert energy_ratio > 1.0

    print("\n== UM-4: reflection-isometry selector no-go ==")
    reflection = np.diag([-1.0, 1.0])
    candidates = (np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    fixed_candidates = [p for p in candidates if np.array_equal(reflection @ p, p)]
    print(f"candidate fixed points under swap isometry = {len(fixed_candidates)}")
    assert not fixed_candidates

    print("\n== UM-5/UM-6: finite point metrics do not identify continuum distance ==")
    # Both metrics equal I at the only stored points (-1,0),(1,0).  The second
    # smooth conformal metric has u=0 at the endpoints but a low-cost corridor
    # between them.  Its straight-path length is already < the Euclidean
    # distance, so the geodesic distances cannot agree.
    x = np.linspace(-1.0, 1.0, 200_001)
    amplitude = 1.5
    u = -amplitude * (1.0 - x * x) ** 2
    corridor_path_length = float(np.trapezoid(np.exp(u), x))
    euclidean_distance = 2.0
    endpoint_u = (float(u[0]), float(u[-1]))
    print(f"endpoint conformal factors u={endpoint_u}")
    print(f"Euclidean distance={euclidean_distance:.12f}, deformed straight-path length={corridor_path_length:.12f}")
    assert endpoint_u == (0.0, 0.0)
    assert corridor_path_length < euclidean_distance - 0.1

    print("\nALL UNIFIED-METRIC MATH CHECKS PASSED")


if __name__ == "__main__":
    main()
