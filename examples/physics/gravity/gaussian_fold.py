"""양의 가우스 측도의 접힘 제약과 유한 분해능 갱신을 계산한다.

All coordinates, covariance entries and resolution scales are dimensionless.
The caller supplies the physical measure; these identities do not select it.
"""

from __future__ import annotations

import math

import numpy as np


def _positive_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be finite and symmetric")
    matrix = (matrix + matrix.T) / 2
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return matrix


def _logdet(matrix: np.ndarray) -> float:
    return float(2 * np.log(np.diag(np.linalg.cholesky(matrix))).sum())


def log_density_at_zero(covariance: np.ndarray) -> float:
    """Centered Gaussian density at zero, not the probability of an exact event."""
    covariance = _positive_matrix(covariance, "covariance")
    return -0.5 * (len(covariance) * math.log(2 * math.pi) + _logdet(covariance))


def split_log_density(covariance: np.ndarray, first: int) -> tuple[float, float]:
    """Return log p(y1=0), log p(y2=0 | y1=0) with transported covariance."""
    covariance = _positive_matrix(covariance, "covariance")
    if not isinstance(first, int) or isinstance(first, bool) or not 0 < first < len(covariance):
        raise ValueError("first must split the constraints into two nonempty groups")
    a = covariance[:first, :first]
    b = covariance[:first, first:]
    d = covariance[first:, first:]
    conditional = d - b.T @ np.linalg.solve(a, b)
    return log_density_at_zero(a), log_density_at_zero(conditional)


def soft_fold(covariance: np.ndarray, constraints: np.ndarray, resolution: float) -> tuple[float, np.ndarray]:
    """Return log E exp(-||Lx||²/(2 eps²)) and the normalized posterior covariance.

    eps is supplied resolution, not a predicted constant. The returned weight
    lies in (0, 1]; unlike a delta density it is dimensionless and bounded.
    """
    covariance = _positive_matrix(covariance, "covariance")
    constraints = np.asarray(constraints, dtype=float)
    if constraints.ndim != 2 or constraints.shape[1] != len(covariance) or not np.isfinite(constraints).all():
        raise ValueError("constraints must be a finite matrix with matching columns")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    variance = resolution * resolution
    if not math.isfinite(variance) or variance == 0:
        raise ValueError("resolution squared must be finite and positive")
    if len(constraints) == 0:
        return 0.0, covariance.copy()
    root = np.linalg.cholesky(covariance)
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = (constraints @ root) / resolution
    if not np.isfinite(whitened).all():
        raise ValueError("scaled constraints exceed floating-point range")
    # Whiten first; scale each singular direction without subtracting C from C.
    _, singular, vectors = np.linalg.svd(whitened, full_matrices=True)
    scales = np.ones(len(covariance))
    scales[:len(singular)] = 1 / np.hypot(1, singular)
    log_weight = -float(np.log(np.hypot(1, singular)).sum())
    posterior_root = (root @ vectors.T) * scales
    posterior = posterior_root @ posterior_root.T
    return log_weight, (posterior + posterior.T) / 2


def stationary_scale(eigenvalues: np.ndarray, action: float, resolution: float) -> float:
    """Unique t=ell²/ell_P² minimum for a supplied soft-fold spectrum and cost.

    gamma(t)=action*t/(8*pi)+sum(log(1+8*pi*lambda/(eps²*t)))/2.
    The positive input action and resolution are model choices.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("eigenvalues must be finite and positive")
    if not math.isfinite(action) or action <= 0 or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("action and resolution must be finite and positive")
    log_coefficients = math.log(8 * math.pi) + np.log(values) - 2 * math.log(resolution)
    log_cost = math.log(4 * math.pi) - math.log(action)
    lower = math.log(float(np.nextafter(0.0, 1.0)))
    upper = math.log(np.finfo(float).max)

    def residual(log_t: float) -> float:
        log_terms = -np.logaddexp(0.0, log_t - log_coefficients)
        return log_t - log_cost - float(np.logaddexp.reduce(log_terms))

    if residual(lower) > 0 or residual(upper) < 0:
        raise ValueError("stationary scale exceeds floating-point range")
    # Monotone log-domain equation also resolves roots far below the old bound.
    for _ in range(100):
        middle = (lower + upper) / 2
        if residual(middle) < 0:
            lower = middle
        else:
            upper = middle
    return math.exp((lower + upper) / 2)
