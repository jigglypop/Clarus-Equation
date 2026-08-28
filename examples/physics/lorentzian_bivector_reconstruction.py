"""Constructive finite Lorentzian linear-simplicity reconstruction.

Conventions use signature (-,+,+,+), contravariant vectors, and
epsilon_0123=+1.  For a unit future timelike normal n and an edge E orthogonal
to it, define the bivector

    B^{IJ} = n^I E^J - n^J E^I.

The linear-simplicity condition is n_I (*B)^{IJ}=0.  On this declared sector,

    E^J = -n_I B^{IJ},
    G_ab = E_a . E_b = -(1/2) B_{a IJ} B_b^{IJ}.

Thus three independent bivectors reconstruct one labelled spacelike face
triad and its intrinsic Gram matrix.  This is a finite inverse lemma only.  It
does not select a full Plebanski branch, prove four-simplex closure, choose a
spin-foam amplitude, or imply a continuum GR limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MINKOWSKI_METRIC = np.diag((-1.0, 1.0, 1.0, 1.0))


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _stable_frobenius_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.linalg.norm(array / maximum))


def _permutation_sign(indices: tuple[int, int, int, int]) -> int:
    if len(set(indices)) < 4:
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


# Raising all four indices changes the sign because det(eta)=-1.
_EPSILON_UPPER = np.empty((4, 4, 4, 4), dtype=float)
for _i in range(4):
    for _j in range(4):
        for _k in range(4):
            for _l in range(4):
                _EPSILON_UPPER[_i, _j, _k, _l] = -_permutation_sign(
                    (_i, _j, _k, _l)
                )


def minkowski_inner(first: np.ndarray, second: np.ndarray) -> float:
    first = _require_shape("first", first, (4,))
    second = _require_shape("second", second, (4,))
    return float(first @ MINKOWSKI_METRIC @ second)


def bivector_from_normal_edge(normal: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Return B=n wedge E after validating the declared geometric sector."""

    normal = _require_shape("normal", normal, (4,))
    edge = _require_shape("edge", edge, (4,))
    if abs(minkowski_inner(normal, normal) + 1.0) > 1.0e-10 or normal[0] <= 0.0:
        raise ValueError("normal must be unit future timelike")
    edge_scale = _stable_frobenius_norm(edge)
    if edge_scale == 0.0:
        raise ValueError("edge must be nonzero")
    if abs(minkowski_inner(normal, edge)) / edge_scale > 1.0e-10:
        raise ValueError("edge must be orthogonal to normal")
    return np.outer(normal, edge) - np.outer(edge, normal)


def hodge_dual(bivector: np.ndarray) -> np.ndarray:
    """Return the Lorentzian Hodge dual; on two-forms star squared is -1."""

    bivector = _require_shape("bivector", bivector, (4, 4))
    lower = MINKOWSKI_METRIC @ bivector @ MINKOWSKI_METRIC
    return 0.5 * np.einsum("ijkl,kl->ij", _EPSILON_UPPER, lower)


def bivector_inner(first: np.ndarray, second: np.ndarray) -> float:
    """Return B_IJ C^IJ, including both antisymmetric matrix triangles."""

    first = _require_shape("first", first, (4, 4))
    second = _require_shape("second", second, (4, 4))
    first_lower = MINKOWSKI_METRIC @ first @ MINKOWSKI_METRIC
    return float(np.sum(first_lower * second))


def common_linear_simplicity_nullity(
    bivectors: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> int:
    """Return the dimension of normals satisfying n_I (*B_a)^IJ=0."""

    bivectors = np.asarray(bivectors, dtype=float)
    if bivectors.ndim != 3 or bivectors.shape[1:] != (4, 4):
        raise ValueError("bivectors must have shape (count, 4, 4)")
    if not np.all(np.isfinite(bivectors)):
        raise ValueError("bivectors must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    constraint = np.vstack([hodge_dual(item).T for item in bivectors])
    constraint_scale = _stable_frobenius_norm(constraint)
    if constraint_scale > 0.0:
        constraint = constraint / constraint_scale
    singular_values = np.linalg.svd(constraint, compute_uv=False)
    maximum = float(np.max(singular_values)) if singular_values.size else 0.0
    rank = (
        int(np.count_nonzero(singular_values > tolerance * maximum))
        if maximum > 0.0
        else 0
    )
    return 4 - rank


@dataclass(frozen=True)
class LorentzianBivectorFaceAudit:
    reconstructed_edges: np.ndarray
    edge_gram: np.ndarray
    bivector_gram: np.ndarray
    normal_residual: float
    antisymmetry_residual: float
    linear_simplicity_residual: float
    reconstruction_residual: float
    gram_identity_residual: float
    oriented_face_volume: float
    common_normal_nullity: int
    hard_reconstruction: bool
    status: str
    plebanski_branch: str = "NOT_TESTED_BY_LINEAR_FACE_DATA"
    claim_ceiling: str = "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTION_ONLY"


def bivector_face_reconstruction_audit(
    normal: np.ndarray,
    bivectors: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> LorentzianBivectorFaceAudit:
    """Reconstruct a labelled spacelike triad from three simple bivectors."""

    normal = _require_shape("normal", normal, (4,))
    bivectors = _require_shape("bivectors", bivectors, (3, 4, 4))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    normal_residual = abs(minkowski_inner(normal, normal) + 1.0)
    bivector_scale = _stable_frobenius_norm(bivectors)
    antisymmetry_residual = (
        _stable_frobenius_norm(bivectors + np.swapaxes(bivectors, 1, 2))
        / bivector_scale
        if bivector_scale > 0.0
        else math.inf
    )
    normal_lower = MINKOWSKI_METRIC @ normal
    stars = np.asarray([hodge_dual(item) for item in bivectors])
    simplicity_numerators = np.asarray(
        [normal_lower @ star for star in stars]
    )
    star_scale = _stable_frobenius_norm(stars)
    linear_simplicity_residual = (
        _stable_frobenius_norm(simplicity_numerators) / star_scale
        if star_scale > 0.0
        else math.inf
    )

    reconstructed_edges = np.asarray(
        [-normal_lower @ bivector for bivector in bivectors]
    )
    reconstructed_bivectors = np.asarray(
        [
            np.outer(normal, edge) - np.outer(edge, normal)
            for edge in reconstructed_edges
        ]
    )
    reconstruction_residual = (
        _stable_frobenius_norm(bivectors - reconstructed_bivectors) / bivector_scale
        if bivector_scale > 0.0
        else math.inf
    )
    edge_scale = _stable_frobenius_norm(reconstructed_edges)
    normalized_edges = (
        reconstructed_edges / edge_scale
        if edge_scale > 0.0
        else reconstructed_edges
    )
    normalized_edge_gram = (
        normalized_edges @ MINKOWSKI_METRIC @ normalized_edges.T
    )
    normalized_bivectors = (
        bivectors / bivector_scale if bivector_scale > 0.0 else bivectors
    )
    normalized_bivector_gram = np.asarray(
        [
            [
                -0.5
                * bivector_inner(
                    normalized_bivectors[left], normalized_bivectors[right]
                )
                for right in range(3)
            ]
            for left in range(3)
        ]
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        edge_scale_squared = np.float64(edge_scale) * np.float64(edge_scale)
        bivector_scale_squared = (
            np.float64(bivector_scale) * np.float64(bivector_scale)
        )
        edge_gram = edge_scale_squared * normalized_edge_gram
        bivector_gram = bivector_scale_squared * normalized_bivector_gram
    scale_ratio_squared = (
        (bivector_scale / edge_scale) ** 2
        if edge_scale > 0.0 and bivector_scale > 0.0
        else 0.0
    )
    comparable_bivector_gram = scale_ratio_squared * normalized_bivector_gram
    gram_scale = max(
        _stable_frobenius_norm(normalized_edge_gram),
        _stable_frobenius_norm(comparable_bivector_gram),
    )
    gram_identity_residual = (
        _stable_frobenius_norm(
            normalized_edge_gram - comparable_bivector_gram
        )
        / gram_scale
        if gram_scale > 0.0
        else math.inf
    )
    eigenvalues = np.linalg.eigvalsh(normalized_edge_gram)
    maximum_eigenvalue = float(np.max(eigenvalues))
    spacelike_rank_three = (
        maximum_eigenvalue > 0.0
        and float(np.min(eigenvalues)) / maximum_eigenvalue > tolerance
    )
    normalized_oriented_volume = float(
        np.linalg.det(np.vstack((normal, normalized_edges)))
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        edge_scale_cubed = (
            np.float64(edge_scale)
            * np.float64(edge_scale)
            * np.float64(edge_scale)
        )
        oriented_face_volume = float(normalized_oriented_volume * edge_scale_cubed)
    common_normal_nullity = common_linear_simplicity_nullity(
        bivectors,
        tolerance=tolerance,
    )

    if normal_residual > tolerance or normal[0] <= tolerance:
        status = "INVALID_UNIT_FUTURE_NORMAL"
    elif antisymmetry_residual > tolerance:
        status = "NON_ANTISYMMETRIC_BIVECTOR_DATA"
    elif linear_simplicity_residual > tolerance:
        status = "LINEAR_SIMPLICITY_FAILED"
    elif reconstruction_residual > tolerance:
        status = "BIVECTOR_INVERSE_RECONSTRUCTION_FAILED"
    elif not spacelike_rank_three:
        status = "NONSPACELIKE_OR_RANK_DEFICIENT_FACE"
    elif gram_identity_residual > tolerance:
        status = "BIVECTOR_GRAM_IDENTITY_FAILED"
    else:
        status = "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTED"

    return LorentzianBivectorFaceAudit(
        reconstructed_edges=reconstructed_edges,
        edge_gram=edge_gram,
        bivector_gram=bivector_gram,
        normal_residual=normal_residual,
        antisymmetry_residual=antisymmetry_residual,
        linear_simplicity_residual=linear_simplicity_residual,
        reconstruction_residual=reconstruction_residual,
        gram_identity_residual=gram_identity_residual,
        oriented_face_volume=oriented_face_volume,
        common_normal_nullity=common_normal_nullity,
        hard_reconstruction=(status == "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTED"),
        status=status,
    )
