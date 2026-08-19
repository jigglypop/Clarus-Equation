"""Small, synthetic-only checks for the statistical claims in 11-math.md.

This never estimates an empirical effect.  It demonstrates two failure modes:
binomial unit subsampling attenuates a branching-regression slope, and treating
overlapping windows as independent greatly understates uncertainty.
"""
from __future__ import annotations

import numpy as np


def branching_series(m: float, immigration: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(n, dtype=np.int64)
    x[0] = 20
    for t in range(n - 1):
        x[t + 1] = rng.poisson(m * x[t] + immigration)
    return x


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def cluster_se(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """OLS slope, naive iid SE, and one-way cluster-robust SE."""
    X = np.column_stack((np.ones(len(x)), x))
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for g in np.unique(groups):
        score = X[groups == g].T @ resid[groups == g]
        meat += np.outer(score, score)
    naive = np.sqrt(np.sum(resid**2) / (len(x) - 2) * bread[1, 1])
    robust = np.sqrt((bread @ meat @ bread)[1, 1])
    return float(beta[1]), float(naive), float(robust)


def main() -> None:
    rng = np.random.default_rng(20260818)
    x = branching_series(m=0.82, immigration=4.0, n=300_000, rng=rng)[10_000:]
    true_slope = slope(x[:-1], x[1:])
    print(f"full-observation lag slope: {true_slope:.4f}")
    for p in (0.8, 0.4, 0.2):
        y = rng.binomial(x, p)
        print(f"observed fraction={p:.1f}, lag slope={slope(y[:-1], y[1:]):.4f}")

    # Twenty animals have a shared animal intercept; 80 windows per animal.
    n_animals, n_windows = 20, 80
    animal = np.repeat(np.arange(n_animals), n_windows)
    # Shared animal components in both predictor and residual mimic repeated
    # windows from the same preparation; iid window SE then overstates n.
    animal_b = rng.normal(0, 1.0, n_animals)
    u = rng.normal(0, 1.0, n_animals)
    b = animal_b[animal] + rng.normal(0, 0.18, n_animals * n_windows)
    q = -0.35 * b + u[animal] + rng.normal(0, 0.35, len(b))
    est, iid, clustered = cluster_se(b, q, animal)
    print(f"window regression slope={est:.4f}, iid_se={iid:.4f}, animal_cluster_se={clustered:.4f}")


if __name__ == "__main__":
    main()
