"""Cross-space recursion for the strengthened Clarus bootstrap core.

The scalar equation

    x = exp(-D * (1-x))

is the one-type member of a larger family.  For non-negative coupling matrix
``A``, the multitype zero-trigger update is

    x_i = exp(-sum_j A_ij * (1-x_j)).

``A_ii`` measures self-recursion and ``A_ij`` for ``i != j`` measures directed
cross-space recursion.  The same equation is the extinction-probability
equation of a multitype Poisson branching process.  This gives the smallest
fixed point a meaning that does not depend on matching an observed number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatVector: TypeAlias = NDArray[np.float64]
FloatMatrix: TypeAlias = NDArray[np.float64]


def _as_coupling_matrix(coupling: ArrayLike) -> FloatMatrix:
    matrix = np.asarray(coupling, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("coupling must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("coupling entries must be finite")
    if np.any(matrix < 0.0):
        raise ValueError("coupling entries must be non-negative")
    return matrix


def _as_survival_vector(survival: ArrayLike, size: int) -> FloatVector:
    vector = np.asarray(survival, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != size:
        raise ValueError(f"survival must be a vector of length {size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("survival entries must be finite")
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError("survival entries must lie in [0, 1]")
    return vector


@dataclass(frozen=True)
class MultispaceFixedPoint:
    """Result of monotone iteration from the zero vector."""

    survival: tuple[float, ...]
    iterations: int
    residual: float
    stability_radius: float

    def as_array(self) -> FloatVector:
        return np.asarray(self.survival, dtype=np.float64)


def multispace_trigger_mean(survival: ArrayLike, coupling: ArrayLike) -> FloatVector:
    """Expected trigger counts for every target row."""
    matrix = _as_coupling_matrix(coupling)
    vector = _as_survival_vector(survival, matrix.shape[0])
    return matrix @ (1.0 - vector)


def multispace_bootstrap_map(survival: ArrayLike, coupling: ArrayLike) -> FloatVector:
    """Apply one independent-Poisson, zero-trigger survival update."""
    return np.exp(-multispace_trigger_mean(survival, coupling))


def multispace_residual(survival: ArrayLike, coupling: ArrayLike) -> FloatVector:
    """Vector residual ``x-F_A(x)``."""
    matrix = _as_coupling_matrix(coupling)
    vector = _as_survival_vector(survival, matrix.shape[0])
    return vector - multispace_bootstrap_map(vector, matrix)


def multispace_jacobian(survival: ArrayLike, coupling: ArrayLike) -> FloatMatrix:
    """Jacobian of the update at a fixed point: ``diag(x) @ A``."""
    matrix = _as_coupling_matrix(coupling)
    vector = _as_survival_vector(survival, matrix.shape[0])
    return np.diag(vector) @ matrix


def spectral_radius(matrix: ArrayLike) -> float:
    """Largest absolute eigenvalue of a square matrix."""
    square = np.asarray(matrix, dtype=np.float64)
    if square.ndim != 2 or square.shape[0] != square.shape[1] or square.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not np.all(np.isfinite(square)):
        raise ValueError("matrix entries must be finite")
    return float(np.max(np.abs(np.linalg.eigvals(square))))


def fixed_point_stability_radius(survival: ArrayLike, coupling: ArrayLike) -> float:
    """Spectral stability radius of a multispace fixed point."""
    return spectral_radius(multispace_jacobian(survival, coupling))


def identity_branch_radius(coupling: ArrayLike) -> float:
    """Stability radius of the always-present identity branch ``x=1``."""
    return spectral_radius(_as_coupling_matrix(coupling))


def branching_regime(coupling: ArrayLike, *, tolerance: float = 1e-12) -> str:
    """Classify the identity branch by the Perron threshold of ``A``."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    radius = identity_branch_radius(coupling)
    if radius < 1.0 - tolerance:
        return "subcritical"
    if radius > 1.0 + tolerance:
        return "supercritical"
    return "critical"


def minimal_multispace_fixed_point(
    coupling: ArrayLike,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 100_000,
) -> MultispaceFixedPoint:
    """Compute the minimal fixed point by monotone iteration from ``x=0``.

    For a multitype probability-generating map, iteration from zero converges
    to the componentwise minimal fixed point.  In the branching interpretation
    this is the vector of eventual extinction probabilities.
    """
    matrix = _as_coupling_matrix(coupling)
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid solver controls")

    survival = np.zeros(matrix.shape[0], dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        updated = multispace_bootstrap_map(survival, matrix)
        if np.any(updated + tolerance < survival):
            raise RuntimeError("probability-generating iteration lost monotonicity")
        step = float(np.max(np.abs(updated - survival)))
        survival = updated
        if step <= tolerance:
            residual = float(np.max(np.abs(multispace_residual(survival, matrix))))
            return MultispaceFixedPoint(
                survival=tuple(float(value) for value in survival),
                iterations=iteration,
                residual=residual,
                stability_radius=fixed_point_stability_radius(survival, matrix),
            )
    raise RuntimeError("multispace fixed-point iteration did not converge")


def symmetric_reduction_depth(
    coupling: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Return scalar depth when the diagonal subspace ``x_i=x`` is invariant.

    The scalar reduction is exact precisely when every row has the same total
    coupling.  It does not require the matrix itself to be symmetric.
    """
    matrix = _as_coupling_matrix(coupling)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, row_sums[0], rtol=0.0, atol=tolerance):
        raise ValueError("row sums differ; no exact one-scalar diagonal reduction")
    return float(row_sums[0])


def is_irreducible(coupling: ArrayLike) -> bool:
    """Whether the directed positive-coupling graph is strongly connected."""
    matrix = _as_coupling_matrix(coupling)
    size = matrix.shape[0]
    if size == 1:
        return True

    adjacency = matrix > 0.0

    def reachable(graph: NDArray[np.bool_]) -> set[int]:
        seen = {0}
        frontier = [0]
        while frontier:
            source = frontier.pop()
            for target in np.flatnonzero(graph[source]):
                node = int(target)
                if node not in seen:
                    seen.add(node)
                    frontier.append(node)
        return seen

    return len(reachable(adjacency)) == size and len(reachable(adjacency.T)) == size


def nearest_neighbor_coupling(
    size: int,
    *,
    self_depth: float,
    neighbor_depth: float,
    periodic: bool,
) -> FloatMatrix:
    """Build a one-dimensional nearest-neighbor recursion lattice.

    A periodic ring has constant row sum ``self_depth + 2*neighbor_depth`` and
    therefore an exact homogeneous scalar sector.  An open chain has boundary
    rows with fewer neighbors, so its fixed point is generally a spatial
    profile rather than one number.
    """
    if size < 3:
        raise ValueError("size must be at least three")
    if self_depth < 0.0 or neighbor_depth < 0.0:
        raise ValueError("depths must be non-negative")

    coupling = np.eye(size, dtype=np.float64) * self_depth
    for index in range(size - 1):
        coupling[index, index + 1] += neighbor_depth
        coupling[index + 1, index] += neighbor_depth
    if periodic:
        coupling[0, size - 1] += neighbor_depth
        coupling[size - 1, 0] += neighbor_depth
    return coupling


def normalized_transfer_coupling(
    self_depth: float,
    cross_depth: float,
    transfer: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> FloatMatrix:
    """Construct ``A = self_depth*I + cross_depth*B`` for row-stochastic ``B``.

    The normalization ``B @ 1 = 1`` makes the homogeneous mode exact and fixes
    its effective scalar depth to ``self_depth + cross_depth``.  This provides
    a spectral route to an additive effective depth; deriving the physical
    transfer operator remains a separate bridge.
    """
    if self_depth < 0.0 or cross_depth < 0.0:
        raise ValueError("depths must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    transition = _as_coupling_matrix(transfer)
    row_sums = transition.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("transfer must be row-stochastic")
    return (
        self_depth * np.eye(transition.shape[0], dtype=np.float64)
        + cross_depth * transition
    )
