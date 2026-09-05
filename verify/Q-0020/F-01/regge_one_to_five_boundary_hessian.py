'''Gauge-reduced boundary Hessian for a Euclidean Regge 1-to-5 refinement.

For every nondegenerate Euclidean boundary four-simplex b and every interior
point q, its five-simplex subdivision is flat and

    S_f(b, y(b, q)) = S_c(b).

The internal Regge equations vanish on this family.  Differentiating those
equations along the interior-point orbit gives C Q=0.  The boundary gradient
equals the coarse boundary gradient all along the same orbit, and
differentiating that identity gives B Q=0.  Here Q is the four-dimensional
vertex-displacement projector.  The exact internal block is

    C = 40*sqrt(5) u u.T,  u=(1,1,1,1,1)/sqrt(5).

Consequently the classical on-shell quotient Hessian is

    H_eff = A - B C^+ B.T = H_coarse.

Writing physical lengths as l=L_ref*l_hat, the functions evaluate the reduced
geometric action

    S_hat = S_geometric/L_ref^2 = sum_h A_hat_h*angle_h.

Thus the reported Hessians with respect to l_hat are dimensionless.  The code
does not include the physical gravitational prefactor L_ref^2/(8*pi*G).  This
is a fixed-boundary Euclidean Regge identity, not an ordinary raw
finite-difference inverse, a convergent Gaussian integral, a proper/EPRL
multi-vertex amplitude, or a continuum-limit result.
'''

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
from typing import Callable, Sequence

import numpy as np

from regge_one_to_five_refinement import (
    BOUNDARY_TRIANGLES,
    BOUNDARY_VERTICES,
    FINE_SIMPLICES,
    INTERNAL_TRIANGLES,
    INTERNAL_VERTEX,
    analytic_barycentric_internal_hessian,
)


BOUNDARY_EDGES = tuple(combinations(BOUNDARY_VERTICES, 2))
_BOUNDARY_EDGE_INDEX = {edge: index for index, edge in enumerate(BOUNDARY_EDGES)}


def _require_boundary_lengths(lengths: Sequence[float]) -> np.ndarray:
    values = np.asarray(lengths, dtype=float)
    if values.shape != (10,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError('boundary_edge_lengths must contain ten positive finite lengths')
    return values


def _require_internal_lengths(lengths: Sequence[float]) -> np.ndarray:
    values = np.asarray(lengths, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError('internal_edge_lengths must contain five positive finite lengths')
    return values


def _squared_length(
    first: int,
    second: int,
    boundary_lengths: np.ndarray,
    internal_lengths: np.ndarray | None,
) -> float:
    if first == second:
        return 0.0
    if first == INTERNAL_VERTEX or second == INTERNAL_VERTEX:
        if internal_lengths is None:
            raise ValueError('coarse simplex has no internal vertex')
        boundary_vertex = second if first == INTERNAL_VERTEX else first
        return float(internal_lengths[boundary_vertex] ** 2)
    edge = tuple(sorted((first, second)))
    return float(boundary_lengths[_BOUNDARY_EDGE_INDEX[edge]] ** 2)


def _simplex_gram_matrix(
    simplex: Sequence[int],
    boundary_lengths: np.ndarray,
    internal_lengths: np.ndarray | None,
) -> np.ndarray:
    vertices = tuple(simplex)
    if len(vertices) != 5 or len(set(vertices)) != 5:
        raise ValueError('a four-simplex must have five distinct vertices')
    base = vertices[0]
    gram = np.empty((4, 4), dtype=float)
    for row, first in enumerate(vertices[1:]):
        for column, second in enumerate(vertices[1:]):
            gram[row, column] = 0.5 * (
                _squared_length(base, first, boundary_lengths, internal_lengths)
                + _squared_length(base, second, boundary_lengths, internal_lengths)
                - _squared_length(first, second, boundary_lengths, internal_lengths)
            )
    return gram


def _simplex_coordinates(
    simplex: Sequence[int],
    boundary_lengths: np.ndarray,
    internal_lengths: np.ndarray | None,
) -> np.ndarray:
    gram = _simplex_gram_matrix(simplex, boundary_lengths, internal_lengths)
    try:
        lower = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as error:
        raise ValueError('NON_EUCLIDEAN_OR_DEGENERATE_SIMPLEX') from error
    return np.vstack((np.zeros(4), lower))


def _triangle_area(
    triangle: Sequence[int],
    boundary_lengths: np.ndarray,
    internal_lengths: np.ndarray | None,
) -> float:
    first, second, third = tuple(triangle)
    a2 = _squared_length(second, third, boundary_lengths, internal_lengths)
    b2 = _squared_length(first, third, boundary_lengths, internal_lengths)
    c2 = _squared_length(first, second, boundary_lengths, internal_lengths)
    area_squared = (2.0 * (a2 * b2 + b2 * c2 + c2 * a2) - a2**2 - b2**2 - c2**2) / 16.0
    if area_squared <= 0.0:
        raise ValueError('DEGENERATE_TRIANGLE_HINGE')
    return math.sqrt(area_squared)


def _dihedral_angle(
    simplex: Sequence[int],
    triangle: Sequence[int],
    boundary_lengths: np.ndarray,
    internal_lengths: np.ndarray | None,
) -> float:
    vertices = tuple(simplex)
    hinge = tuple(triangle)
    if len(hinge) != 3 or not set(hinge).issubset(vertices):
        raise ValueError('triangle must be a triangle of simplex')
    coordinates = _simplex_coordinates(vertices, boundary_lengths, internal_lengths)
    index = {vertex: position for position, vertex in enumerate(vertices)}
    complements = tuple(vertex for vertex in vertices if vertex not in hinge)
    normals: list[np.ndarray] = []
    for omitted in complements:
        face = tuple(vertex for vertex in vertices if vertex != omitted)
        anchor = coordinates[index[face[0]]]
        spanning = np.asarray([coordinates[index[vertex]] - anchor for vertex in face[1:]])
        _, _, right = np.linalg.svd(spanning, full_matrices=True)
        normal = right[-1]
        if float(normal @ (coordinates[index[omitted]] - anchor)) > 0.0:
            normal = -normal
        normals.append(normal)
    cosine = float(-normals[0] @ normals[1])
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def coarse_euclidean_regge_boundary_action(
    boundary_edge_lengths: Sequence[float],
) -> float:
    '''Return S_geometric/L_ref^2 for one Euclidean four-simplex boundary.'''

    boundary = _require_boundary_lengths(boundary_edge_lengths)
    action = 0.0
    for hinge in BOUNDARY_TRIANGLES:
        angle = _dihedral_angle(BOUNDARY_VERTICES, hinge, boundary, None)
        action += _triangle_area(hinge, boundary, None) * (math.pi - angle)
    return float(action)


def euclidean_regge_one_to_five_action(
    boundary_edge_lengths: Sequence[float],
    internal_edge_lengths: Sequence[float],
) -> float:
    '''Return fine S_geometric/L_ref^2 with the fixed-boundary term.'''

    boundary = _require_boundary_lengths(boundary_edge_lengths)
    internal = _require_internal_lengths(internal_edge_lengths)
    action = 0.0
    for hinge in INTERNAL_TRIANGLES:
        angles = [
            _dihedral_angle(simplex, hinge, boundary, internal)
            for simplex in FINE_SIMPLICES
            if set(hinge).issubset(simplex)
        ]
        action += _triangle_area(hinge, boundary, internal) * (
            2.0 * math.pi - sum(angles)
        )
    for hinge in BOUNDARY_TRIANGLES:
        angles = [
            _dihedral_angle(simplex, hinge, boundary, internal)
            for simplex in FINE_SIMPLICES
            if set(hinge).issubset(simplex)
        ]
        action += _triangle_area(hinge, boundary, internal) * (
            math.pi - sum(angles)
        )
    return float(action)


def interior_point_internal_lengths(
    boundary_edge_lengths: Sequence[float],
    barycentric_weights: Sequence[float],
) -> np.ndarray:
    '''Return distances from an interior affine point to all boundary vertices.'''

    boundary = _require_boundary_lengths(boundary_edge_lengths)
    weights = np.asarray(barycentric_weights, dtype=float)
    if (
        weights.shape != (5,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not math.isclose(float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError('barycentric_weights must be five positive values summing to one')
    coordinates = _simplex_coordinates(BOUNDARY_VERTICES, boundary, None)
    point = weights @ coordinates
    return np.asarray(
        [np.linalg.norm(point - coordinates[index]) for index in range(5)],
        dtype=float,
    )


def barycentric_internal_lengths(
    boundary_edge_lengths: Sequence[float],
) -> np.ndarray:
    '''Return the exact centroid-to-vertex lengths from boundary edge lengths.'''

    boundary = _require_boundary_lengths(boundary_edge_lengths)
    squared = boundary * boundary
    total = float(np.sum(squared))
    result = np.empty(5, dtype=float)
    for vertex in BOUNDARY_VERTICES:
        incident_sum = sum(
            squared[index]
            for index, edge in enumerate(BOUNDARY_EDGES)
            if vertex in edge
        )
        radius_squared = incident_sum / 5.0 - total / 25.0
        if radius_squared <= 0.0:
            raise ValueError('NON_EUCLIDEAN_BARYCENTRIC_RADIUS')
        result[vertex] = math.sqrt(radius_squared)
    return result


def barycentric_section_jacobian(
    boundary_edge_lengths: Sequence[float],
) -> np.ndarray:
    '''Return d(internal centroid lengths)/d(boundary lengths), shape 5 by 10.'''

    boundary = _require_boundary_lengths(boundary_edge_lengths)
    internal = barycentric_internal_lengths(boundary)
    jacobian = np.empty((5, 10), dtype=float)
    for vertex in BOUNDARY_VERTICES:
        for index, edge in enumerate(BOUNDARY_EDGES):
            incidence = 1.0 if vertex in edge else 0.0
            jacobian[vertex, index] = (
                boundary[index] / internal[vertex] * (incidence / 5.0 - 1.0 / 25.0)
            )
    return jacobian


def _gradient_and_hessian(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(step) or step <= 0.0 or np.any(point <= step):
        raise ValueError('finite_difference_step must be positive and smaller than every edge')
    base = function(point)
    dimension = len(point)
    gradient = np.empty(dimension, dtype=float)
    hessian = np.empty((dimension, dimension), dtype=float)
    for first in range(dimension):
        positive = point.copy()
        negative = point.copy()
        positive[first] += step
        negative[first] -= step
        plus = function(positive)
        minus = function(negative)
        gradient[first] = (plus - minus) / (2.0 * step)
        hessian[first, first] = (plus - 2.0 * base + minus) / step**2
        for second in range(first):
            pp = point.copy()
            pm = point.copy()
            mp = point.copy()
            mm = point.copy()
            pp[first] += step
            pp[second] += step
            pm[first] += step
            pm[second] -= step
            mp[first] -= step
            mp[second] += step
            mm[first] -= step
            mm[second] -= step
            value = (
                function(pp) - function(pm) - function(mp) + function(mm)
            ) / (4.0 * step**2)
            hessian[first, second] = value
            hessian[second, first] = value
    return gradient, hessian


@dataclass(frozen=True)
class ReggeOneToFiveBoundaryHessianAudit:
    finite_difference_step: float
    numerical_tolerance: float
    boundary_dimension: int
    internal_dimension: int
    full_dimension: int
    maximum_flat_section_action_residual: float
    barycentric_formula_coordinate_residual: float
    raw_full_hessian_transpose_residual: float
    half_step_full_hessian_transpose_residual: float
    raw_internal_gradient_residual: float
    half_step_internal_gradient_residual: float
    raw_internal_gauge_residual: float
    half_step_internal_gauge_residual: float
    raw_mixing_gauge_residual: float
    half_step_mixing_gauge_residual: float
    raw_internal_to_analytic_residual: float
    half_step_internal_to_analytic_residual: float
    raw_section_stationarity_derivative_residual: float
    half_step_section_stationarity_derivative_residual: float
    raw_on_shell_pullback_residual: float
    half_step_on_shell_pullback_residual: float
    raw_gauge_reduced_schur_coarse_residual: float
    half_step_gauge_reduced_schur_coarse_residual: float
    half_step_relative_schur_coarse_residual: float
    raw_finite_difference_internal_rank: int
    analytic_internal_rank: int
    analytic_internal_nullity: int
    analytic_internal_radial_curvature: float
    reduced_geometric_action_normalization_used: bool
    physical_gravitational_prefactor_included: bool
    finite_difference_residuals_decrease: bool
    gauge_reduced_internal_pseudoinverse_used: bool
    raw_finite_difference_pseudoinverse_used: bool
    classical_on_shell_boundary_hessian_identity_closed: bool
    conditional_gaussian_integral_defined: bool
    proper_eprl_multicell_hessian_computed: bool
    spin_foam_measure_and_contour_derived: bool
    curved_refinement_identity_derived: bool
    continuum_einstein_hilbert_dominance_derived: bool
    status: str
    claim_ceiling: str = (
        'FLAT_EUCLIDEAN_REGGE_1_TO_5_CLASSICAL_QUOTIENT_SCHUR_NOT_SPINFOAM_GAUSSIAN'
    )


@lru_cache(maxsize=8)
def audit_regge_one_to_five_boundary_hessian(
    *,
    finite_difference_step: float = 2.0e-4,
    tolerance: float = 2.0e-6,
) -> ReggeOneToFiveBoundaryHessianAudit:
    '''Audit the exact flat-section theorem with two-step numerical Hessians.'''

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    boundary = np.full(10, math.sqrt(2.0))
    internal = barycentric_internal_lengths(boundary)
    point = np.concatenate((boundary, internal))
    uniform_weights = np.full(5, 0.2)
    coordinate_internal = interior_point_internal_lengths(boundary, uniform_weights)
    barycentric_coordinate_residual = float(np.linalg.norm(internal - coordinate_internal))

    perturbed_boundary = boundary * (
        1.0
        + np.asarray((0.006, -0.004, 0.003, -0.002, 0.005, -0.003, 0.002, 0.004, -0.005, 0.001))
    )
    weights = (
        uniform_weights,
        np.asarray((0.24, 0.18, 0.19, 0.17, 0.22)),
    )
    action_residuals = []
    for sample_boundary in (boundary, perturbed_boundary):
        coarse_action = coarse_euclidean_regge_boundary_action(sample_boundary)
        for sample_weights in weights:
            sample_internal = interior_point_internal_lengths(
                sample_boundary, sample_weights
            )
            fine_action = euclidean_regge_one_to_five_action(
                sample_boundary, sample_internal
            )
            action_residuals.append(abs(fine_action - coarse_action))

    def fine_action(values: np.ndarray) -> float:
        return euclidean_regge_one_to_five_action(values[:10], values[10:])

    def coarse_action(values: np.ndarray) -> float:
        return coarse_euclidean_regge_boundary_action(values)

    raw_gradient, raw_hessian = _gradient_and_hessian(
        fine_action, point, finite_difference_step
    )
    half_gradient, half_hessian = _gradient_and_hessian(
        fine_action, point, finite_difference_step / 2.0
    )
    _, raw_coarse_hessian = _gradient_and_hessian(
        coarse_action, boundary, finite_difference_step
    )
    _, half_coarse_hessian = _gradient_and_hessian(
        coarse_action, boundary, finite_difference_step / 2.0
    )

    raw_boundary = raw_hessian[:10, :10]
    raw_mixing = raw_hessian[:10, 10:]
    raw_internal = raw_hessian[10:, 10:]
    half_boundary = half_hessian[:10, :10]
    half_mixing = half_hessian[:10, 10:]
    half_internal = half_hessian[10:, 10:]
    unit = np.ones(5) / math.sqrt(5.0)
    radial_projector = np.outer(unit, unit)
    gauge_projector = np.eye(5) - radial_projector
    analytic_internal = analytic_barycentric_internal_hessian()
    analytic_radial_curvature = 40.0 * math.sqrt(5.0)
    analytic_pseudoinverse = radial_projector / analytic_radial_curvature
    section_jacobian = barycentric_section_jacobian(boundary)

    raw_effective = raw_boundary - raw_mixing @ analytic_pseudoinverse @ raw_mixing.T
    half_effective = (
        half_boundary - half_mixing @ analytic_pseudoinverse @ half_mixing.T
    )
    raw_schur_residual = float(np.linalg.norm(raw_effective - raw_coarse_hessian))
    half_schur_residual = float(np.linalg.norm(half_effective - half_coarse_hessian))
    half_schur_scale = max(1.0, float(np.linalg.norm(half_coarse_hessian)))
    raw_pullback_residual = float(
        np.linalg.norm(raw_boundary + raw_mixing @ section_jacobian - raw_coarse_hessian)
    )
    half_pullback_residual = float(
        np.linalg.norm(
            half_boundary + half_mixing @ section_jacobian - half_coarse_hessian
        )
    )
    raw_stationarity_derivative = float(
        np.linalg.norm(raw_mixing.T + raw_internal @ section_jacobian)
    )
    half_stationarity_derivative = float(
        np.linalg.norm(half_mixing.T + half_internal @ section_jacobian)
    )

    raw_internal_gauge = float(np.linalg.norm(raw_internal @ gauge_projector))
    half_internal_gauge = float(np.linalg.norm(half_internal @ gauge_projector))
    raw_mixing_gauge = float(np.linalg.norm(raw_mixing @ gauge_projector))
    half_mixing_gauge = float(np.linalg.norm(half_mixing @ gauge_projector))
    raw_internal_to_analytic = float(np.linalg.norm(raw_internal - analytic_internal))
    half_internal_to_analytic = float(np.linalg.norm(half_internal - analytic_internal))
    residuals_decrease = (
        half_internal_gauge < raw_internal_gauge
        and half_mixing_gauge < raw_mixing_gauge
        and half_internal_to_analytic < raw_internal_to_analytic
        and half_stationarity_derivative < raw_stationarity_derivative
        and half_pullback_residual < raw_pullback_residual
        and half_schur_residual < raw_schur_residual
    )
    closed = (
        max(action_residuals) <= 1.0e-10
        and barycentric_coordinate_residual <= tolerance
        and residuals_decrease
        and float(np.linalg.norm(half_gradient[10:])) <= tolerance
        and half_internal_gauge <= 10.0 * tolerance
        and half_mixing_gauge <= 10.0 * tolerance
        and half_stationarity_derivative <= 10.0 * tolerance
        and half_pullback_residual <= 10.0 * tolerance
        and half_schur_residual / half_schur_scale <= tolerance
    )

    return ReggeOneToFiveBoundaryHessianAudit(
        finite_difference_step=finite_difference_step,
        numerical_tolerance=tolerance,
        boundary_dimension=10,
        internal_dimension=5,
        full_dimension=15,
        maximum_flat_section_action_residual=max(action_residuals),
        barycentric_formula_coordinate_residual=barycentric_coordinate_residual,
        raw_full_hessian_transpose_residual=float(np.linalg.norm(raw_hessian - raw_hessian.T)),
        half_step_full_hessian_transpose_residual=float(
            np.linalg.norm(half_hessian - half_hessian.T)
        ),
        raw_internal_gradient_residual=float(np.linalg.norm(raw_gradient[10:])),
        half_step_internal_gradient_residual=float(np.linalg.norm(half_gradient[10:])),
        raw_internal_gauge_residual=raw_internal_gauge,
        half_step_internal_gauge_residual=half_internal_gauge,
        raw_mixing_gauge_residual=raw_mixing_gauge,
        half_step_mixing_gauge_residual=half_mixing_gauge,
        raw_internal_to_analytic_residual=raw_internal_to_analytic,
        half_step_internal_to_analytic_residual=half_internal_to_analytic,
        raw_section_stationarity_derivative_residual=raw_stationarity_derivative,
        half_step_section_stationarity_derivative_residual=half_stationarity_derivative,
        raw_on_shell_pullback_residual=raw_pullback_residual,
        half_step_on_shell_pullback_residual=half_pullback_residual,
        raw_gauge_reduced_schur_coarse_residual=raw_schur_residual,
        half_step_gauge_reduced_schur_coarse_residual=half_schur_residual,
        half_step_relative_schur_coarse_residual=half_schur_residual / half_schur_scale,
        raw_finite_difference_internal_rank=int(np.linalg.matrix_rank(raw_internal)),
        analytic_internal_rank=1,
        analytic_internal_nullity=4,
        analytic_internal_radial_curvature=analytic_radial_curvature,
        reduced_geometric_action_normalization_used=True,
        physical_gravitational_prefactor_included=False,
        finite_difference_residuals_decrease=residuals_decrease,
        gauge_reduced_internal_pseudoinverse_used=True,
        raw_finite_difference_pseudoinverse_used=False,
        classical_on_shell_boundary_hessian_identity_closed=closed,
        conditional_gaussian_integral_defined=False,
        proper_eprl_multicell_hessian_computed=False,
        spin_foam_measure_and_contour_derived=False,
        curved_refinement_identity_derived=False,
        continuum_einstein_hilbert_dominance_derived=False,
        status=(
            'FLAT_REGGE_1_TO_5_GAUGE_REDUCED_BOUNDARY_HESSIAN_IDENTITY_CLOSED'
            if closed else 'REGGE_1_TO_5_BOUNDARY_HESSIAN_AUDIT_FAILED'
        ),
    )
