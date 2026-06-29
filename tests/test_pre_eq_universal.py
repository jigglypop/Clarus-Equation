from __future__ import annotations

import numpy as np

"""Numeric regression for docs/9_등호이전/04c theorem 2.1.

The Gibbs posterior of the linear equality ``Ax = b`` with squared-residual
defect and standard Gaussian prior is Gaussian with closed-form moments.
Its mean must converge to the Moore-Penrose solution, and the prior variance
must survive untouched along ``ker A``.
"""


def _gibbs_moments(a: np.ndarray, b: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    precision = 2.0 * beta * a.T @ a + np.eye(a.shape[1])
    covariance = np.linalg.inv(precision)
    mean = covariance @ (2.0 * beta * a.T @ b)
    return mean, covariance


def test_linear_manifest_is_moore_penrose() -> None:
    a = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]])
    b = np.array([1.0, 3.0])

    mean, _ = _gibbs_moments(a, b, beta=1e8)

    assert np.allclose(mean, np.linalg.pinv(a) @ b, atol=1e-6)


def test_kernel_direction_keeps_prior_variance() -> None:
    a = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]])
    b = np.array([1.0, 3.0])
    kernel_direction = np.array([0.0, 0.0, 1.0])

    _, covariance = _gibbs_moments(a, b, beta=1e8)

    kernel_variance = float(kernel_direction @ covariance @ kernel_direction)
    assert abs(kernel_variance - 1.0) < 1e-12

    row_space_direction = np.array([1.0, 0.0, 0.0])
    row_variance = float(row_space_direction @ covariance @ row_space_direction)
    assert row_variance < 1e-6


def test_inconsistent_system_manifests_least_squares() -> None:
    a = np.array([[1.0], [1.0]])
    b = np.array([0.0, 1.0])

    mean, _ = _gibbs_moments(a, b, beta=1e8)

    assert np.allclose(mean, [0.5], atol=1e-6)
