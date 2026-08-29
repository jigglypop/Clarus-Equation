"""A finite Euclidean 4D Regge ``1 -> 5`` refinement witness.

The boundary is a regular four-simplex of squared edge length ``2``.  Its
barycentre is inserted as vertex 5, giving five internal edges of squared
length ``4/5``.  Edge lengths are ratios to a common ``L_ref`` and the code
evaluates the reduced geometric action ``S_geometric/L_ref^2``: internal
triangular hinges use ``A_hat*(2*pi-sum(theta))`` and boundary triangles use
the Gibbons--Hawking--York analogue ``A_hat*(pi-sum(theta))``.  The physical
gravitational prefactor ``L_ref^2/(8*pi*G)`` is not included.

At the flat barycentric point, translations of the inserted vertex change the
five internal lengths in the four-dimensional subspace perpendicular to
``(1,1,1,1,1)``.  They are exact Regge vertex-displacement gauge directions.
The fixed-boundary Schlaefli identity makes the Regge gradient vanish along
the local family of flat subdivisions, so differentiating that gradient gives
four exact Hessian null vectors.  S5 symmetry then reduces the remaining
direction to the collective radial mode, whose exact eigenvalue is
``40*sqrt(5)``.  The raw finite-difference Hessian is nevertheless full rank
because truncation error lifts the four zero modes, so it is retained only as
a convergence check.  This is a finite Euclidean Regge calculation only: it
is neither a boundary Schur complement, a proper/EPRL amplitude, nor a
Gaussian path integral.
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
    '''Evaluate reduced S_geometric/L_ref^2 with its boundary term.'''

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


def equal_radius_regge_action(radius: float) -> float:
    '''Return the exact reduced action when all five internal edges equal r.

    The nondegenerate Euclidean domain is r^2 > 3/4.  There are ten internal
    and ten boundary triangle hinges, with three and two incident fine
    four-simplices respectively.
    '''

    if not math.isfinite(radius) or radius * radius <= 3.0 / 4.0:
        raise ValueError('equal internal radius must satisfy r^2 > 3/4')
    squared = radius * radius
    internal_area = 0.5 * math.sqrt(2.0 * squared - 1.0)
    boundary_area = math.sqrt(3.0) / 2.0
    internal_angle = math.acos((squared - 1.0) / (3.0 * squared - 2.0))
    boundary_angle = math.acos(1.0 / (2.0 * math.sqrt(3.0 * squared - 2.0)))
    return float(
        10.0 * internal_area * (2.0 * math.pi - 3.0 * internal_angle)
        + 10.0 * boundary_area * (math.pi - 2.0 * boundary_angle)
    )


def barycentric_internal_length_jacobian() -> np.ndarray:
    '''Return the exact length-map Jacobian in the five-dimensional embedding.

    Boundary vertices are v_i=e_i-1/5*1 and r0=sqrt(4/5).  For a displacement
    x in the sum-zero hyperplane, d l_i=-v_i dot x/r0.  Extending the map by
    zero on the normal direction gives J=-(I-11^T/5)/r0, of rank four and
    image exactly 1-perp.
    '''

    projector = np.eye(5) - np.ones((5, 5)) / 5.0
    return -projector / math.sqrt(4.0 / 5.0)


def analytic_barycentric_internal_hessian() -> np.ndarray:
    '''Return the exact fixed-boundary Regge internal Hessian.

    Flat 1-to-5 subdivisions obey grad_l S(l(x))=0 by the fixed-boundary
    Schlaefli identity.  Differentiation gives H J=0 for the rank-four length
    Jacobian J.  Regular-boundary S5 symmetry leaves only the collective
    direction.  Along l_i=r, the hinge formula gives

        S''(sqrt(4/5)) = 200*sqrt(5).

    Since the normalized collective coordinate changes every r by 1/sqrt(5),
    its eigenvalue is 40*sqrt(5), hence H=8*sqrt(5)*11^T.
    '''

    return np.full((5, 5), 8.0 * math.sqrt(5.0))


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
    finite_difference_step: float
    raw_hessian_transpose_residual: float
    s5_symmetry_reduction_residual: float
    raw_hessian_gauge_residual: float
    half_step_raw_hessian_gauge_residual: float
    raw_gauge_residual_decreases_with_step: bool
    raw_hessian_eigenvalues: tuple[float, ...]
    raw_hessian_condition_number: float
    finite_difference_raw_inverse_exists: bool
    s5_averaged_finite_difference_gauge_residual: float
    s5_averaged_finite_difference_eigenvalues: tuple[float, ...]
    s5_averaged_tolerance_rank: int
    s5_averaged_tolerance_nullity: int
    barycentric_length_jacobian_rank: int
    barycentric_length_jacobian_gram_residual: float
    analytic_hessian_gauge_residual: float
    analytic_hessian_eigenvalues: tuple[float, ...]
    analytic_internal_hessian_rank: int
    analytic_internal_hessian_nullity: int
    analytic_radial_curvature: float
    raw_hessian_to_analytic_residual: float
    half_step_hessian_to_analytic_residual: float
    finite_difference_converges_to_analytic_hessian: bool
    exact_gauge_unfixed_inverse_defined: bool
    gauge_basis_orthogonality_residual: float
    projected_physical_curvature: float
    projected_physical_internal_inverse_defined: bool
    boundary_schur_complement_computed: bool
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
    _, half_step_raw_hessian = _gradient_and_hessian(lengths, finite_difference_step / 2.0)
    diagonal = float(np.mean(np.diag(raw_hessian)))
    off_diagonal = float(np.mean(raw_hessian[~np.eye(5, dtype=bool)]))
    symmetry_hessian = np.full((5, 5), off_diagonal)
    np.fill_diagonal(symmetry_hessian, diagonal)
    unit = np.ones(5) / math.sqrt(5.0)
    projector = np.eye(5) - np.outer(unit, unit)
    length_jacobian = barycentric_internal_length_jacobian()
    analytic_hessian = analytic_barycentric_internal_hessian()
    raw_eigenvalues = np.linalg.eigvalsh(raw_hessian)
    symmetry_eigenvalues = np.linalg.eigvalsh(symmetry_hessian)
    analytic_eigenvalues = np.linalg.eigvalsh(analytic_hessian)
    scale = max(1.0, float(np.linalg.norm(symmetry_hessian, ord=2)))
    rank_tolerance = max(tolerance, 2.0e-4 * scale)
    s5_rank = int(np.count_nonzero(np.abs(symmetry_eigenvalues) > rank_tolerance))
    s5_nullity = 5 - s5_rank
    raw_gauge_residual = float(np.linalg.norm(raw_hessian @ projector))
    half_step_raw_gauge_residual = float(np.linalg.norm(half_step_raw_hessian @ projector))
    symmetry_gauge_residual = float(np.linalg.norm(symmetry_hessian @ projector))
    analytic_gauge_residual = float(np.linalg.norm(analytic_hessian @ projector))
    raw_to_analytic = float(np.linalg.norm(raw_hessian - analytic_hessian))
    half_step_to_analytic = float(np.linalg.norm(half_step_raw_hessian - analytic_hessian))
    jacobian_gram_residual = float(
        np.linalg.norm(length_jacobian @ length_jacobian.T - 1.25 * projector)
    )
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
        and int(np.linalg.matrix_rank(length_jacobian)) == 4
        and jacobian_gram_residual <= tolerance
        and analytic_gauge_residual <= tolerance
        and max(abs(value) for value in analytic_eigenvalues[:4]) <= tolerance
        and abs(analytic_eigenvalues[-1] - 40.0 * math.sqrt(5.0)) <= tolerance
        and half_step_to_analytic < raw_to_analytic
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
        finite_difference_step=finite_difference_step,
        raw_hessian_transpose_residual=float(np.linalg.norm(raw_hessian - raw_hessian.T)),
        s5_symmetry_reduction_residual=float(np.linalg.norm(raw_hessian - symmetry_hessian)),
        raw_hessian_gauge_residual=raw_gauge_residual,
        half_step_raw_hessian_gauge_residual=half_step_raw_gauge_residual,
        raw_gauge_residual_decreases_with_step=half_step_raw_gauge_residual < raw_gauge_residual,
        raw_hessian_eigenvalues=tuple(float(value) for value in raw_eigenvalues),
        raw_hessian_condition_number=float(np.linalg.cond(raw_hessian)),
        finite_difference_raw_inverse_exists=int(np.linalg.matrix_rank(raw_hessian)) == 5,
        s5_averaged_finite_difference_gauge_residual=symmetry_gauge_residual,
        s5_averaged_finite_difference_eigenvalues=tuple(
            float(value) for value in symmetry_eigenvalues
        ),
        s5_averaged_tolerance_rank=s5_rank,
        s5_averaged_tolerance_nullity=s5_nullity,
        barycentric_length_jacobian_rank=int(np.linalg.matrix_rank(length_jacobian)),
        barycentric_length_jacobian_gram_residual=jacobian_gram_residual,
        analytic_hessian_gauge_residual=analytic_gauge_residual,
        analytic_hessian_eigenvalues=tuple(float(value) for value in analytic_eigenvalues),
        analytic_internal_hessian_rank=1,
        analytic_internal_hessian_nullity=4,
        analytic_radial_curvature=40.0 * math.sqrt(5.0),
        raw_hessian_to_analytic_residual=raw_to_analytic,
        half_step_hessian_to_analytic_residual=half_step_to_analytic,
        finite_difference_converges_to_analytic_hessian=half_step_to_analytic < raw_to_analytic,
        exact_gauge_unfixed_inverse_defined=False,
        gauge_basis_orthogonality_residual=orthogonality,
        projected_physical_curvature=float(unit @ analytic_hessian @ unit),
        projected_physical_internal_inverse_defined=(
            abs(float(unit @ analytic_hessian @ unit)) > tolerance
        ),
        boundary_schur_complement_computed=False,
        boundary_hessian_equality_checked=False,
        gauge_reduced_boundary_hessian_equals_coarse=False,
        proper_eprl_amplitude_derived=False,
        full_gaussian_path_integral_defined=False,
        status=(
            'EUCLIDEAN_REGGE_1_TO_5_ANALYTIC_INTERNAL_HESSIAN_CLOSED'
            if closed else 'REGGE_1_TO_5_AUDIT_FAILED'
        ),
    )
