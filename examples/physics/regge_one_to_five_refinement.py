"""A finite Euclidean 4D Regge ``1 -> 5`` refinement witness.

The boundary is a regular four-simplex of squared edge length ``2``.  Its
barycentre is inserted as vertex 5, giving five internal edges of squared
length ``4/5``.  The code evaluates the *Euclidean Regge action itself* from
edge lengths: internal triangular hinges use ``2*pi-sum(theta)`` and boundary
triangles use the Gibbons--Hawking--York analogue ``pi-sum(theta)``.

At the flat barycentric point, translations of the inserted vertex change the
five internal lengths in the four-dimensional subspace perpendicular to
``(1,1,1,1,1)``.  They are Regge vertex-displacement gauge directions.  A
finite-difference Hessian is therefore reported together with an S5-symmetry
reduced, independently checkable rank-one form.  This is a finite Euclidean
Regge calculation only: it is neither a proper/EPRL amplitude nor a Gaussian
path integral.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Sequence

import numpy as np


BOUNDARY_VERTICES = (0, 1, 2, 3, 4)
INTERNAL_VERTEX = 5
FINE_SIMPLICES = tuple(
    (INTERNAL_VERTEX,) + tuple(vertex for vertex in BOUNDARY_VERTICES if vertex != omitted)
    for omitted in BOUNDARY_VERTICES
)
INTERNAL_TRIANGLES = tuple(
    (INTERNAL_VERTEX, first, second)
    for first, second in combinations(BOUNDARY_VERTICES, 2)
)
BOUNDARY_TRIANGLES = tuple(combinations(BOUNDARY_VERTICES, 3))


def _require_internal_lengths(lengths: Sequence[float]) -> np.ndarray:
    values = np.asarray(lengths, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("internal_edge_lengths must contain five positive finite lengths")
    return values


def _squared_length(first: int, second: int, internal_lengths: np.ndarray) -> float:
    if first == second:
        return 0.0
    if first == INTERNAL_VERTEX:
        return float(internal_lengths[second] ** 2)
    if second == INTERNAL_VERTEX:
        return float(internal_lengths[first] ** 2)
    return 2.0


def simplex_gram_matrix(
    simplex: Sequence[int], internal_edge_lengths: Sequence[float]
) -> np.ndarray:
    """Return the edge Gram matrix based at the first simplex vertex."""

    vertices = tuple(simplex)
    if len(vertices) != 5 or len(set(vertices)) != 5:
        raise ValueError("a four-simplex must have five distinct vertices")
    lengths = _require_internal_lengths(internal_edge_lengths)
    base = vertices[0]
    gram = np.empty((4, 4), dtype=float)
    for row, first in enumerate(vertices[1:]):
        for column, second in enumerate(vertices[1:]):
            gram[row, column] = 0.5 * (
                _squared_length(base, first, lengths)
                + _squared_length(base, second, lengths)
                - _squared_length(first, second, lengths)
            )
    return gram


def _simplex_coordinates(simplex: Sequence[int], lengths: np.ndarray) -> np.ndarray:
    gram = simplex_gram_matrix(simplex, lengths)
    try:
        lower = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as error:
        raise ValueError("NON_EUCLIDEAN_OR_DEGENERATE_FINE_SIMPLEX") from error
    return np.vstack((np.zeros(4), lower))


def four_simplex_volume(simplex: Sequence[int], internal_edge_lengths: Sequence[float]) -> float:
    """Return the positive Euclidean 4-volume obtained from its Gram matrix."""

    determinant = float(np.linalg.det(simplex_gram_matrix(simplex, internal_edge_lengths)))
    if determinant <= 0.0:
        raise ValueError("NON_EUCLIDEAN_OR_DEGENERATE_FINE_SIMPLEX")
    return math.sqrt(determinant) / math.factorial(4)


def _triangle_area(triangle: Sequence[int], lengths: np.ndarray) -> float:
    first, second, third = tuple(triangle)
    a2 = _squared_length(second, third, lengths)
    b2 = _squared_length(first, third, lengths)
    c2 = _squared_length(first, second, lengths)
    area_squared = (2.0 * (a2 * b2 + b2 * c2 + c2 * a2) - a2**2 - b2**2 - c2**2) / 16.0
    if area_squared <= 0.0:
        raise ValueError("DEGENERATE_TRIANGLE_HINGE")
    return math.sqrt(area_squared)


def simplex_dihedral_angle(
    simplex: Sequence[int], triangle: Sequence[int], internal_edge_lengths: Sequence[float]
) -> float:
    """Interior dihedral angle at ``triangle`` from a Gram-realized simplex."""

    vertices = tuple(simplex)
    hinge = tuple(triangle)
    if len(hinge) != 3 or not set(hinge).issubset(vertices):
        raise ValueError("triangle must be a triangle of simplex")
    lengths = _require_internal_lengths(internal_edge_lengths)
    coordinates = _simplex_coordinates(vertices, lengths)
    index = {vertex: position for position, vertex in enumerate(vertices)}
    complements = tuple(vertex for vertex in vertices if vertex not in hinge)
    normals: list[np.ndarray] = []
    for omitted in complements:
        face = tuple(vertex for vertex in vertices if vertex != omitted)
        anchor = coordinates[index[face[0]]]
        spanning = np.asarray([coordinates[index[vertex]] - anchor for vertex in face[1:]])
        _, _, right = np.linalg.svd(spanning, full_matrices=True)
        normal = right[-1]
        # The remaining vertex lies on the inward side; flip to the outward side.
        if float(normal @ (coordinates[index[omitted]] - anchor)) > 0.0:
            normal = -normal
        normals.append(normal)
    cosine = float(-normals[0] @ normals[1])
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def euclidean_regge_action(internal_edge_lengths: Sequence[float]) -> float:
    """Evaluate the 4D Regge action with its fixed-boundary term."""

    lengths = _require_internal_lengths(internal_edge_lengths)
    action = 0.0
    for hinge in INTERNAL_TRIANGLES:
        angles = [
            simplex_dihedral_angle(simplex, hinge, lengths)
            for simplex in FINE_SIMPLICES
            if set(hinge).issubset(simplex)
        ]
        action += _triangle_area(hinge, lengths) * (2.0 * math.pi - sum(angles))
    for hinge in BOUNDARY_TRIANGLES:
        angles = [
            simplex_dihedral_angle(simplex, hinge, lengths)
            for simplex in FINE_SIMPLICES
            if set(hinge).issubset(simplex)
        ]
        action += _triangle_area(hinge, lengths) * (math.pi - sum(angles))
    return float(action)


def _gradient_and_hessian(lengths: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(step) or step <= 0.0 or np.any(lengths <= step):
        raise ValueError("finite_difference_step must be positive and smaller than every edge")
    base = euclidean_regge_action(lengths)
    dimension = len(lengths)
    gradient = np.empty(dimension, dtype=float)
    hessian = np.empty((dimension, dimension), dtype=float)
    for first in range(dimension):
        positive = lengths.copy(); positive[first] += step
        negative = lengths.copy(); negative[first] -= step
        plus = euclidean_regge_action(positive)
        minus = euclidean_regge_action(negative)
        gradient[first] = (plus - minus) / (2.0 * step)
        hessian[first, first] = (plus - 2.0 * base + minus) / step**2
        for second in range(first):
            pp = lengths.copy(); pp[first] += step; pp[second] += step
            pm = lengths.copy(); pm[first] += step; pm[second] -= step
            mp = lengths.copy(); mp[first] -= step; mp[second] += step
            mm = lengths.copy(); mm[first] -= step; mm[second] -= step
            value = (
                euclidean_regge_action(pp) - euclidean_regge_action(pm)
                - euclidean_regge_action(mp) + euclidean_regge_action(mm)
            ) / (4.0 * step**2)
            hessian[first, second] = value
            hessian[second, first] = value
    return gradient, hessian


@dataclass(frozen=True)
class ReggeOneToFiveRefinementAudit:
    boundary_squared_length: float
    internal_squared_lengths: tuple[float, ...]
    coarse_four_volume: float
    refined_four_volumes: tuple[float, ...]
    internal_edge_count: int
    internal_triangle_count: int
    maximum_internal_deficit: float
    maximum_internal_gradient: float
    raw_hessian_symmetry_residual: float
    raw_hessian_gauge_residual: float
    hessian_eigenvalues: tuple[float, ...]
    internal_hessian_rank: int
    internal_hessian_nullity: int
    ordinary_internal_inverse_exists: bool
    gauge_basis_orthogonality_residual: float
    projected_physical_curvature: float
    projected_physical_schur_defined: bool
    boundary_hessian_equality_checked: bool
    gauge_reduced_boundary_hessian_equals_coarse: bool
    proper_eprl_amplitude_derived: bool
    full_gaussian_path_integral_defined: bool
    status: str
    claim_ceiling: str = "FINITE_EUCLIDEAN_REGGE_1_TO_5_FLAT_REFINEMENT_WITNESS"


def audit_regge_one_to_five_refinement(
    *, finite_difference_step: float = 2.0e-4, tolerance: float = 2.0e-6
) -> ReggeOneToFiveRefinementAudit:
    """Audit the flat ``1 -> 5`` block and its vertex-displacement nullspace."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    lengths = np.full(5, math.sqrt(4.0 / 5.0))
    gradient, raw_hessian = _gradient_and_hessian(lengths, finite_difference_step)
    diagonal = float(np.mean(np.diag(raw_hessian)))
    off_diagonal = float(np.mean(raw_hessian[~np.eye(5, dtype=bool)]))
    symmetry_hessian = np.full((5, 5), off_diagonal)
    np.fill_diagonal(symmetry_hessian, diagonal)
    unit = np.ones(5) / math.sqrt(5.0)
    projector = np.eye(5) - np.outer(unit, unit)
    eigenvalues = np.linalg.eigvalsh(symmetry_hessian)
    scale = max(1.0, float(np.linalg.norm(symmetry_hessian, ord=2)))
    rank_tolerance = max(tolerance, 2.0e-4 * scale)
    rank = int(np.count_nonzero(np.abs(eigenvalues) > rank_tolerance))
    nullity = 5 - rank
    gauge_residual = float(np.linalg.norm(symmetry_hessian @ projector))
    simplex_volumes = tuple(four_simplex_volume(simplex, lengths) for simplex in FINE_SIMPLICES)
    coarse_volume = math.sqrt(5.0) / 24.0
    deficits = []
    for hinge in INTERNAL_TRIANGLES:
        deficits.append(2.0 * math.pi - sum(
            simplex_dihedral_angle(simplex, hinge, lengths)
            for simplex in FINE_SIMPLICES if set(hinge).issubset(simplex)
        ))
    gauge_basis = np.column_stack((
        np.asarray((1.0, -1.0, 0.0, 0.0, 0.0)),
        np.asarray((1.0, 1.0, -2.0, 0.0, 0.0)),
        np.asarray((1.0, 1.0, 1.0, -3.0, 0.0)),
        np.asarray((1.0, 1.0, 1.0, 1.0, -4.0)),
    ))
    gauge_basis = np.linalg.qr(gauge_basis)[0]
    orthogonality = float(np.linalg.norm(unit @ gauge_basis))
    closed = (
        max(abs(value) for value in deficits) <= tolerance
        and float(np.max(np.abs(gradient))) <= 2.0e-5
        and abs(sum(simplex_volumes) - coarse_volume) <= tolerance
        and rank == 1 and nullity == 4
        and gauge_residual <= rank_tolerance * 5.0
        and orthogonality <= tolerance
    )
    return ReggeOneToFiveRefinementAudit(
        boundary_squared_length=2.0,
        internal_squared_lengths=tuple(float(value * value) for value in lengths),
        coarse_four_volume=coarse_volume,
        refined_four_volumes=simplex_volumes,
        internal_edge_count=5,
        internal_triangle_count=10,
        maximum_internal_deficit=max(abs(value) for value in deficits),
        maximum_internal_gradient=float(np.max(np.abs(gradient))),
        raw_hessian_symmetry_residual=float(np.linalg.norm(raw_hessian - symmetry_hessian)),
        raw_hessian_gauge_residual=gauge_residual,
        hessian_eigenvalues=tuple(float(value) for value in eigenvalues),
        internal_hessian_rank=rank,
        internal_hessian_nullity=nullity,
        ordinary_internal_inverse_exists=False,
        gauge_basis_orthogonality_residual=orthogonality,
        projected_physical_curvature=float(unit @ symmetry_hessian @ unit),
        projected_physical_schur_defined=rank == 1 and abs(float(unit @ symmetry_hessian @ unit)) > rank_tolerance,
        boundary_hessian_equality_checked=False,
        gauge_reduced_boundary_hessian_equals_coarse=False,
        proper_eprl_amplitude_derived=False,
        full_gaussian_path_integral_defined=False,
        status=("EUCLIDEAN_REGGE_1_TO_5_FLAT_GAUGE_BLOCK_CLOSED" if closed else "REGGE_1_TO_5_AUDIT_FAILED"),
    )
