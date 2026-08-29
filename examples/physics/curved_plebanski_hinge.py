"""Exact finite non-flat Plebanski witness on a closed labelled hinge.

The earlier two-simplex CE witness has an open link around every shared
triangle, so it cannot by itself support a closed Regge dual face.  This
module preserves both original 4-simplices and, on the fixed six-vertex set,
adds one minimal missing cell

    (0, 1, 2, 3, 5)

which closes the link of ``f=(1,2,3)`` into the three-cycle
``0--4--5--0``.  On the same labelled vertices it installs a stereographic
de Sitter tetrad with dimensionless curvature ``kappa = K L_ref^2`` and
checks

    T^I = 0,
    R^{IJ} = K e^I wedge e^J,
    D_A Sigma^i = 0,
    F^i = (Lambda/3) Sigma^i,
    Lambda L_ref^2 = 3 kappa.

Both a closed primal triangle holonomy and a six-segment projective
barycentric realization of the abstract dual loop are computed from the same
constant-curvature connection.  The latter is not identified with a Regge
deficit angle.  This is a finite non-flat conditional existence certificate.
The all-points field identities are analytic; the runtime point evaluations
are numerical regressions only.  The certificate does not derive
``kappa`` from bare 0D data, choose a proper spin-foam amplitude or measure,
prove refinement convergence, or derive the two graviton degrees of freedom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from examples.physics.causal_face_simplicity import (
    CompositionFace,
    proper_orthochronous_residual,
)
from examples.physics.zerod_plebanski_closure import (
    ConstantCurvatureEinsteinAudit,
    constant_curvature_einstein_audit,
    typed_rank_four_event_trace,
)


VertexId = int
TriangleId = tuple[VertexId, VertexId, VertexId]
SimplexId = tuple[VertexId, VertexId, VertexId, VertexId, VertexId]

MINKOWSKI_4 = np.diag((-1.0, 1.0, 1.0, 1.0))
MINKOWSKI_5 = np.diag((-1.0, 1.0, 1.0, 1.0, 1.0))


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _stable_norm(value: np.ndarray) -> float:
    array = np.asarray(value)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.linalg.norm(array / maximum))


def _permutation_parity(indices: tuple[int, ...]) -> int:
    if len(set(indices)) < len(indices):
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    )
    return -1 if inversions % 2 else 1


EPSILON_4 = np.empty((4, 4, 4, 4), dtype=float)
for _a in range(4):
    for _b in range(4):
        for _c in range(4):
            for _d in range(4):
                EPSILON_4[_a, _b, _c, _d] = _permutation_parity(
                    (_a, _b, _c, _d)
                )

EPSILON_3 = np.empty((3, 3, 3), dtype=float)
for _i in range(3):
    for _j in range(3):
        for _k in range(3):
            EPSILON_3[_i, _j, _k] = _permutation_parity((_i, _j, _k))


def reference_vertex_coordinates() -> dict[VertexId, np.ndarray]:
    """Return the common dimensionless coordinates ``y=x/L_ref``."""

    return {
        0: np.asarray((1.0, 0.2, 0.2, 0.2)),
        1: np.asarray((0.0, 0.0, 0.0, 0.0)),
        2: np.asarray((0.0, 1.0, 0.0, 0.0)),
        3: np.asarray((0.0, 0.0, 1.0, 0.0)),
        4: np.asarray((0.0, 0.0, 0.0, 1.0)),
        5: np.asarray((-1.0, 0.2, 0.2, 0.2)),
    }


def _simplex_oriented_coordinate_volume(
    simplex: SimplexId,
    coordinates: Mapping[VertexId, np.ndarray],
) -> float:
    base = coordinates[simplex[0]]
    edges = np.asarray(
        [coordinates[vertex] - base for vertex in simplex[1:]], dtype=float
    )
    return float(np.linalg.det(edges))


def _hinge_link_edges(
    simplices: Sequence[SimplexId],
    hinge: TriangleId,
) -> tuple[tuple[VertexId, VertexId], ...]:
    hinge_set = set(hinge)
    edges = {
        tuple(sorted(set(simplex) - hinge_set))
        for simplex in simplices
        if hinge_set.issubset(simplex)
    }
    if any(len(edge) != 2 for edge in edges):
        raise ValueError("every 4-simplex containing the hinge must add two link vertices")
    return tuple(sorted(edges))


def _connected_graph(
    edges: Sequence[tuple[VertexId, VertexId]],
) -> bool:
    if not edges:
        return False
    adjacency: dict[VertexId, set[VertexId]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    pending = [next(iter(adjacency))]
    visited: set[VertexId] = set()
    while pending:
        vertex = pending.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        pending.extend(adjacency[vertex] - visited)
    return visited == set(adjacency)


@dataclass(frozen=True)
class CurvedHingeTraceAudit:
    history_id: str
    original_simplices: tuple[SimplexId, ...]
    added_simplex: SimplexId
    extended_simplices: tuple[SimplexId, ...]
    hinge: TriangleId
    mapped_causal_face: CompositionFace
    original_link_edges: tuple[tuple[VertexId, VertexId], ...]
    extended_link_edges: tuple[tuple[VertexId, VertexId], ...]
    extended_link_degrees: tuple[tuple[VertexId, int], ...]
    simplex_oriented_coordinate_volumes: tuple[float, ...]
    original_two_cell_link_open: bool
    closed_three_cycle: bool
    minimal_one_simplex_closure: bool
    original_cells_preserved: bool
    causal_label_matches_hinge: bool
    status: str


def curved_hinge_trace_audit(
    *,
    include_closing_simplex: bool = True,
    history_id: str = "CE-C4-H0-CURVED",
) -> CurvedHingeTraceAudit:
    """Close the original two-cell link around ``f=(1,2,3)`` minimally."""

    if not isinstance(include_closing_simplex, bool):
        raise ValueError("include_closing_simplex must be boolean")
    if not history_id:
        raise ValueError("history_id must be nonempty")
    original_trace = typed_rank_four_event_trace(history_id=history_id)
    original = original_trace.simplex_cells
    added: SimplexId = (0, 1, 2, 3, 5)
    extended = original + ((added,) if include_closing_simplex else ())
    hinge: TriangleId = (1, 2, 3)
    original_link = _hinge_link_edges(original, hinge)
    extended_link = _hinge_link_edges(extended, hinge)
    degrees: dict[VertexId, int] = {}
    for left, right in extended_link:
        degrees[left] = degrees.get(left, 0) + 1
        degrees[right] = degrees.get(right, 0) + 1
    closed_cycle = (
        len(extended_link) == 3
        and len(degrees) == 3
        and all(degree == 2 for degree in degrees.values())
        and _connected_graph(extended_link)
    )
    original_degrees: dict[VertexId, int] = {}
    for left, right in original_link:
        original_degrees[left] = original_degrees.get(left, 0) + 1
        original_degrees[right] = original_degrees.get(right, 0) + 1
    original_open = (
        len(original_link) == 2
        and sorted(original_degrees.values()) == [1, 1, 2]
        and _connected_graph(original_link)
    )
    mapped = tuple(
        (face, triangle)
        for face, triangle in original_trace.causal_to_shared_triangle
        if triangle == hinge
    )
    if len(mapped) != 1:
        raise ValueError("the original causal map must label the hinge exactly once")
    coordinates = reference_vertex_coordinates()
    volumes = tuple(
        _simplex_oriented_coordinate_volume(simplex, coordinates)
        for simplex in extended
    )
    minimal = (
        original_open
        and include_closing_simplex
        and set(added) - set(hinge) == {0, 5}
        and (0, 5) in extended_link
        and closed_cycle
    )
    closed = (
        include_closing_simplex
        and closed_cycle
        and minimal
        and all(abs(volume) > 1.0e-12 for volume in volumes)
    )
    return CurvedHingeTraceAudit(
        history_id=history_id,
        original_simplices=original,
        added_simplex=added,
        extended_simplices=extended,
        hinge=hinge,
        mapped_causal_face=mapped[0][0],
        original_link_edges=original_link,
        extended_link_edges=extended_link,
        extended_link_degrees=tuple(sorted(degrees.items())),
        simplex_oriented_coordinate_volumes=volumes,
        original_two_cell_link_open=original_open,
        closed_three_cycle=closed_cycle,
        minimal_one_simplex_closure=minimal,
        original_cells_preserved=extended[: len(original)] == original,
        causal_label_matches_hinge=mapped[0][1] == hinge,
        status=(
            "MINIMAL_THREE_SIMPLEX_INTERNAL_HINGE_CLOSED"
            if closed
            else "NO_CLOSED_DUAL_FACE"
        ),
    )


def _two_form(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.outer(first, second) - np.outer(second, first)


def _wedge_one_two(one_form: np.ndarray, two_form: np.ndarray) -> np.ndarray:
    return (
        np.einsum("a,bc->abc", one_form, two_form)
        + np.einsum("b,ca->abc", one_form, two_form)
        + np.einsum("c,ab->abc", one_form, two_form)
    )


def _hodge_two_form_conformal_lorentz(two_form: np.ndarray) -> np.ndarray:
    # In four dimensions the Hodge star on 2-forms is conformally invariant.
    raised = MINKOWSKI_4 @ two_form @ MINKOWSKI_4
    return 0.5 * np.einsum("mnrs,rs->mn", EPSILON_4, raised)


def _wedge_four_volume(first: np.ndarray, second: np.ndarray) -> complex:
    return complex(0.25 * np.einsum("mnrs,mn,rs->", EPSILON_4, first, second))


@dataclass(frozen=True)
class DeSitterPlebanskiPointAudit:
    coordinate_over_reference_length: tuple[float, float, float, float]
    conformal_factor: float
    patch_denominator: float
    torsion_residual: float
    riemann_constant_curvature_residual: float
    self_duality_residual: float
    simplicity_tracefree_residual: float
    covariant_constancy_residual: float
    plebanski_curvature_residual: float


def de_sitter_plebanski_point_audit(
    coordinate_over_reference_length: Sequence[float],
    *,
    curvature_times_reference_length_squared: float,
    cosmological_constant_times_reference_length_squared: float | None = None,
) -> DeSitterPlebanskiPointAudit:
    """Evaluate the exact stereographic de Sitter identities at one point."""

    y = np.asarray(coordinate_over_reference_length, dtype=float)
    if y.shape != (4,) or not np.all(np.isfinite(y)):
        raise ValueError("coordinate_over_reference_length must be a finite four-vector")
    kappa = _require_finite(
        "curvature_times_reference_length_squared",
        curvature_times_reference_length_squared,
    )
    if kappa < 0.0:
        raise ValueError("this de Sitter certificate requires non-negative curvature")
    lambda_bar = (
        3.0 * kappa
        if cosmological_constant_times_reference_length_squared is None
        else _require_finite(
            "cosmological_constant_times_reference_length_squared",
            cosmological_constant_times_reference_length_squared,
        )
    )
    y_lower = MINKOWSKI_4 @ y
    interval = float(y @ y_lower)
    denominator = 1.0 + 0.25 * kappa * interval
    if denominator <= 0.0:
        raise ValueError("point lies outside the stereographic de Sitter patch")
    omega_factor = 1.0 / denominator
    derivative_omega_factor = -0.5 * kappa * omega_factor**2 * y_lower

    coframe = omega_factor * np.eye(4)
    derivative_coframe = np.einsum(
        "r,Im->rIm", derivative_omega_factor, np.eye(4)
    )
    spin = np.zeros((4, 4, 4), dtype=float)
    derivative_spin = np.zeros((4, 4, 4, 4), dtype=float)
    for internal_left in range(4):
        for internal_right in range(4):
            for coordinate in range(4):
                algebraic = (
                    y[internal_left] * (internal_right == coordinate)
                    - y[internal_right] * (internal_left == coordinate)
                )
                spin[internal_left, internal_right, coordinate] = (
                    0.5 * kappa * omega_factor * algebraic
                )
                for derivative in range(4):
                    derivative_algebraic = (
                        (internal_left == derivative) * (internal_right == coordinate)
                        - (internal_right == derivative) * (internal_left == coordinate)
                    )
                    derivative_spin[
                        derivative, internal_left, internal_right, coordinate
                    ] = 0.5 * kappa * (
                        derivative_omega_factor[derivative] * algebraic
                        + omega_factor * derivative_algebraic
                    )

    mixed_spin = np.einsum("IKm,KJ->IJm", spin, MINKOWSKI_4)
    torsion = np.zeros((4, 4, 4), dtype=float)
    for internal in range(4):
        for first in range(4):
            for second in range(4):
                torsion[internal, first, second] = (
                    derivative_coframe[first, internal, second]
                    - derivative_coframe[second, internal, first]
                    + np.dot(mixed_spin[internal, :, first], coframe[:, second])
                    - np.dot(mixed_spin[internal, :, second], coframe[:, first])
                )

    riemann = np.zeros((4, 4, 4, 4), dtype=float)
    expected_riemann = np.zeros_like(riemann)
    for left in range(4):
        for right in range(4):
            for first in range(4):
                for second in range(4):
                    riemann[left, right, first, second] = (
                        derivative_spin[first, left, right, second]
                        - derivative_spin[second, left, right, first]
                        + np.dot(
                            mixed_spin[left, :, first], spin[:, right, second]
                        )
                        - np.dot(
                            mixed_spin[left, :, second], spin[:, right, first]
                        )
                    )
                    expected_riemann[left, right, first, second] = kappa * (
                        coframe[left, first] * coframe[right, second]
                        - coframe[left, second] * coframe[right, first]
                    )

    sigma = np.asarray(
        (
            1j * _two_form(coframe[0], coframe[1])
            - _two_form(coframe[2], coframe[3]),
            1j * _two_form(coframe[0], coframe[2])
            - _two_form(coframe[3], coframe[1]),
            1j * _two_form(coframe[0], coframe[3])
            - _two_form(coframe[1], coframe[2]),
        )
    )
    duals = np.asarray(
        [_hodge_two_form_conformal_lorentz(item) for item in sigma]
    )
    self_duality = _stable_norm(duals - 1j * sigma) / max(
        1.0, _stable_norm(sigma)
    )
    wedge_matrix = np.asarray(
        [[_wedge_four_volume(left, right) for right in sigma] for left in sigma]
    )
    trace_average = np.trace(wedge_matrix) / 3.0
    tracefree = wedge_matrix - trace_average * np.eye(3)
    simplicity = _stable_norm(tracefree) / max(1.0, _stable_norm(wedge_matrix))

    chiral_connection = np.zeros((3, 4), dtype=complex)
    derivative_chiral_connection = np.zeros((4, 3, 4), dtype=complex)
    for internal in range(3):
        spatial = internal + 1
        chiral_connection[internal] = 1j * spin[0, spatial]
        derivative_chiral_connection[:, internal] = (
            1j * derivative_spin[:, 0, spatial]
        )
        for left in range(3):
            for right in range(3):
                chiral_connection[internal] -= (
                    0.5
                    * EPSILON_3[internal, left, right]
                    * spin[left + 1, right + 1]
                )
                derivative_chiral_connection[:, internal] -= (
                    0.5
                    * EPSILON_3[internal, left, right]
                    * derivative_spin[:, left + 1, right + 1]
                )

    curvature = np.zeros((3, 4, 4), dtype=complex)
    for internal in range(3):
        for first in range(4):
            for second in range(4):
                curvature[internal, first, second] = (
                    derivative_chiral_connection[first, internal, second]
                    - derivative_chiral_connection[second, internal, first]
                    + 0.5
                    * sum(
                        EPSILON_3[internal, left, right]
                        * (
                            chiral_connection[left, first]
                            * chiral_connection[right, second]
                            - chiral_connection[left, second]
                            * chiral_connection[right, first]
                        )
                        for left in range(3)
                        for right in range(3)
                    )
                )

    derivative_sigma = np.zeros((4, 3, 4, 4), dtype=complex)
    for derivative in range(4):
        derivative_basis = derivative_coframe[derivative]
        derivative_sigma[derivative] = np.asarray(
            (
                1j
                * (
                    _two_form(derivative_basis[0], coframe[1])
                    + _two_form(coframe[0], derivative_basis[1])
                )
                - _two_form(derivative_basis[2], coframe[3])
                - _two_form(coframe[2], derivative_basis[3]),
                1j
                * (
                    _two_form(derivative_basis[0], coframe[2])
                    + _two_form(coframe[0], derivative_basis[2])
                )
                - _two_form(derivative_basis[3], coframe[1])
                - _two_form(coframe[3], derivative_basis[1]),
                1j
                * (
                    _two_form(derivative_basis[0], coframe[3])
                    + _two_form(coframe[0], derivative_basis[3])
                )
                - _two_form(derivative_basis[1], coframe[2])
                - _two_form(coframe[1], derivative_basis[2]),
            )
        )

    covariant_derivative = np.zeros((3, 4, 4, 4), dtype=complex)
    for internal in range(3):
        exterior_derivative = (
            derivative_sigma[:, internal]
            + np.transpose(derivative_sigma[:, internal], (1, 2, 0))
            + np.transpose(derivative_sigma[:, internal], (2, 0, 1))
        )
        connection_term = sum(
            EPSILON_3[internal, left, right]
            * _wedge_one_two(chiral_connection[left], sigma[right])
            for left in range(3)
            for right in range(3)
        )
        covariant_derivative[internal] = exterior_derivative + connection_term

    torsion_residual = _stable_norm(torsion) / max(1.0, _stable_norm(coframe))
    riemann_residual = _stable_norm(riemann - expected_riemann) / max(
        1.0, _stable_norm(expected_riemann)
    )
    covariant_residual = _stable_norm(covariant_derivative) / max(
        1.0, _stable_norm(sigma)
    )
    expected_chiral_curvature = (lambda_bar / 3.0) * sigma
    plebanski_residual = _stable_norm(curvature - expected_chiral_curvature) / max(
        1.0,
        _stable_norm(curvature),
        _stable_norm(expected_chiral_curvature),
    )
    return DeSitterPlebanskiPointAudit(
        coordinate_over_reference_length=tuple(float(value) for value in y),
        conformal_factor=omega_factor,
        patch_denominator=denominator,
        torsion_residual=torsion_residual,
        riemann_constant_curvature_residual=riemann_residual,
        self_duality_residual=self_duality,
        simplicity_tracefree_residual=simplicity,
        covariant_constancy_residual=covariant_residual,
        plebanski_curvature_residual=plebanski_residual,
    )


@dataclass(frozen=True)
class ExactPrimalHolonomyAudit:
    face_id: TriangleId
    curvature_times_reference_length_squared: float
    coordinate_triangle_area_over_reference_length_squared: float
    transport_convention: str
    oriented_boundary: tuple[VertexId, VertexId, VertexId, VertexId]
    rotation_angle: float
    holonomy: np.ndarray
    lorentz_residual: float
    flatness_residual: float
    nontrivial_curvature_holonomy: bool
    status: str


def exact_primal_triangle_holonomy(
    *,
    curvature_times_reference_length_squared: float,
    face_id: TriangleId = (1, 2, 3),
    tolerance: float = 1.0e-12,
) -> ExactPrimalHolonomyAudit:
    """Return the exact stereographic de Sitter holonomy of the labelled face."""

    kappa = _require_finite(
        "curvature_times_reference_length_squared",
        curvature_times_reference_length_squared,
    )
    tolerance = _require_finite("tolerance", tolerance)
    if kappa < 0.0 or tolerance <= 0.0:
        raise ValueError("curvature must be non-negative and tolerance positive")
    if tuple(face_id) != (1, 2, 3):
        raise ValueError("face_id must be the labelled internal hinge (1,2,3)")
    coordinates = reference_vertex_coordinates()
    first_leg = coordinates[2] - coordinates[1]
    second_leg = coordinates[3] - coordinates[1]
    first_leg_squared = float(first_leg @ MINKOWSKI_4 @ first_leg)
    second_leg_squared = float(second_leg @ MINKOWSKI_4 @ second_leg)
    mutual_product = float(first_leg @ MINKOWSKI_4 @ second_leg)
    if not (
        abs(first_leg_squared - 1.0) <= tolerance
        and abs(second_leg_squared - 1.0) <= tolerance
        and abs(mutual_product) <= tolerance
    ):
        raise ValueError("the labelled reference hinge must be the fixed unit right triangle")
    side = math.sqrt(first_leg_squared)
    if kappa == 0.0:
        angle = 0.0
    else:
        scaled = 0.25 * kappa * side * side
        half_scaled = 0.5 * scaled
        angle = (
            (0.5 * kappa * side * side)
            / math.sqrt((1.0 + half_scaled) * half_scaled)
            * math.atan(math.sqrt(half_scaled / (1.0 + half_scaled)))
        )
    holonomy = np.eye(4)
    holonomy[1:3, 1:3] = np.asarray(
        (
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        )
    )
    lorentz_residual = proper_orthochronous_residual(holonomy)
    flatness = _stable_norm(holonomy - np.eye(4)) / max(1.0, _stable_norm(holonomy))
    curved = kappa > 0.0 and flatness > tolerance and lorentz_residual <= tolerance
    return ExactPrimalHolonomyAudit(
        face_id=tuple(face_id),
        curvature_times_reference_length_squared=kappa,
        coordinate_triangle_area_over_reference_length_squared=0.5 * side * side,
        transport_convention="dV + omega V = 0",
        oriented_boundary=(1, 2, 3, 1),
        rotation_angle=angle,
        holonomy=holonomy,
        lorentz_residual=lorentz_residual,
        flatness_residual=flatness,
        nontrivial_curvature_holonomy=curved,
        status=(
            "EXACT_NONTRIVIAL_DE_SITTER_PRIMAL_HOLONOMY"
            if curved
            else "FLAT_PRIMAL_HOLONOMY"
        ),
    )


def _de_sitter_embedding(
    coordinate: np.ndarray,
    kappa: float,
) -> np.ndarray:
    interval = float(coordinate @ MINKOWSKI_4 @ coordinate)
    denominator = 1.0 + 0.25 * kappa * interval
    if denominator <= 0.0:
        raise ValueError("vertex lies outside the stereographic de Sitter patch")
    conformal = 1.0 / denominator
    embedded = np.empty(5, dtype=float)
    embedded[:4] = conformal * coordinate
    embedded[4] = (
        (1.0 - 0.25 * kappa * interval)
        * conformal
        / math.sqrt(kappa)
    )
    return embedded


def _de_sitter_centre(points: Sequence[np.ndarray], kappa: float) -> np.ndarray:
    total = np.sum(np.asarray(points, dtype=float), axis=0)
    norm_squared = float(total @ MINKOWSKI_5 @ total)
    if norm_squared <= 0.0:
        raise ValueError("dual centre sum must be spacelike in the ambient metric")
    return total / math.sqrt(kappa * norm_squared)


def _parallel_transport_matrix(
    source: np.ndarray,
    target: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, float]:
    denominator = 1.0 + kappa * float(source @ MINKOWSKI_5 @ target)
    if denominator <= 0.0:
        raise ValueError("dual centres are not joined by the selected geodesic branch")
    target_covector = MINKOWSKI_5 @ target
    operator = np.eye(5) - (
        kappa / denominator * np.outer(source + target, target_covector)
    )
    return operator, denominator


def _orthonormal_tangent_frame(point: np.ndarray, kappa: float) -> np.ndarray:
    tangent_projector = np.eye(5) - kappa * np.outer(
        point, MINKOWSKI_5 @ point
    )
    left_singular_vectors, singular_values, _ = np.linalg.svd(tangent_projector)
    rank = int(np.count_nonzero(singular_values > 1.0e-10))
    if rank != 4:
        raise ValueError("de Sitter tangent projector must have rank four")
    spanning = left_singular_vectors[:, :4]
    induced = spanning.T @ MINKOWSKI_5 @ spanning
    eigenvalues, eigenvectors = np.linalg.eigh(induced)
    if not (np.count_nonzero(eigenvalues < 0.0) == 1 and np.count_nonzero(eigenvalues > 0.0) == 3):
        raise ValueError("de Sitter tangent space must have signature (-,+,+,+)")
    order = tuple(np.where(eigenvalues < 0.0)[0]) + tuple(
        np.where(eigenvalues > 0.0)[0]
    )
    ordered_values = eigenvalues[list(order)]
    ordered_vectors = eigenvectors[:, list(order)]
    return spanning @ ordered_vectors @ np.diag(1.0 / np.sqrt(np.abs(ordered_values)))


@dataclass(frozen=True)
class ExactDualHolonomyAudit:
    face_id: TriangleId
    path_labels: tuple[tuple[str, tuple[int, ...]], ...]
    minimum_geodesic_denominator: float
    maximum_hyperboloid_residual: float
    maximum_segment_tangency_residual: float
    lorentz_residual: float
    flatness_residual: float
    holonomy: np.ndarray
    projective_barycentric_realization: bool
    positive_cone_segment_inclusion: bool
    regge_deficit_angle_derived: bool
    closed_dual_curvature_holonomy: bool
    status: str


def exact_dual_hinge_holonomy(
    trace: CurvedHingeTraceAudit,
    *,
    curvature_times_reference_length_squared: float,
    tolerance: float = 1.0e-9,
) -> ExactDualHolonomyAudit:
    """Transport around a projective realization of the abstract dual loop.

    For a cell ``S`` the chosen centre is the normalized positive ambient sum
    of its embedded vertices.  For every flag ``T subset S`` the connecting
    geodesic lies in the normalized positive cone of ``S``: a positive linear
    combination of the two centres expands into positive vertex weights.
    Thus the six flags below realize the barycentric dual cycle of the closed
    link.  This definition is not a circumcentric Regge dual and its holonomy
    is not asserted to equal a Regge deficit angle.
    """

    if not trace.closed_three_cycle:
        raise ValueError("NO_CLOSED_DUAL_FACE")
    kappa = _require_finite(
        "curvature_times_reference_length_squared",
        curvature_times_reference_length_squared,
    )
    tolerance = _require_finite("tolerance", tolerance)
    if kappa <= 0.0 or tolerance <= 0.0:
        raise ValueError("dual de Sitter holonomy requires positive curvature/tolerance")
    coordinates = reference_vertex_coordinates()
    embedded = {
        vertex: _de_sitter_embedding(coordinate, kappa)
        for vertex, coordinate in coordinates.items()
    }
    simplex_left, simplex_right, simplex_closing = trace.extended_simplices
    tetra_14 = (1, 2, 3, 4)
    tetra_15 = (1, 2, 3, 5)
    tetra_01 = (0, 1, 2, 3)
    labels: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("simplex", simplex_left),
        ("tetrahedron", tetra_14),
        ("simplex", simplex_right),
        ("tetrahedron", tetra_15),
        ("simplex", simplex_closing),
        ("tetrahedron", tetra_01),
    )
    nested_flags = all(
        set(labels[(index + 1) % len(labels)][1]).issubset(set(vertices))
        or set(vertices).issubset(set(labels[(index + 1) % len(labels)][1]))
        for index, (_, vertices) in enumerate(labels)
    )
    centres = tuple(
        _de_sitter_centre(tuple(embedded[vertex] for vertex in vertices), kappa)
        for _, vertices in labels
    )
    hyperboloid_residuals = tuple(
        abs(float(centre @ MINKOWSKI_5 @ centre) - 1.0 / kappa)
        for centre in centres
    )
    total_transport = np.eye(5)
    denominators: list[float] = []
    tangency_residuals: list[float] = []
    for index, source in enumerate(centres):
        target = centres[(index + 1) % len(centres)]
        operator, denominator = _parallel_transport_matrix(source, target, kappa)
        denominators.append(denominator)
        tangent_projector = np.eye(5) - kappa * np.outer(
            source, MINKOWSKI_5 @ source
        )
        test_vectors = tangent_projector[:, :4]
        transported = operator @ test_vectors
        tangency_residuals.append(
            _stable_norm(target @ MINKOWSKI_5 @ transported)
            / max(1.0, _stable_norm(transported))
        )
        total_transport = operator @ total_transport

    base = centres[0]
    frame = _orthonormal_tangent_frame(base, kappa)
    tangent_metric = np.diag((-1.0, 1.0, 1.0, 1.0))
    holonomy = (
        tangent_metric @ frame.T @ MINKOWSKI_5 @ total_transport @ frame
    )
    lorentz_residual = proper_orthochronous_residual(holonomy)
    flatness = _stable_norm(holonomy - np.eye(4)) / max(1.0, _stable_norm(holonomy))
    positive_cone_inclusion = nested_flags and min(denominators) > 0.0
    closed = (
        positive_cone_inclusion
        and max(hyperboloid_residuals) <= tolerance
        and max(tangency_residuals) <= tolerance
        and lorentz_residual <= tolerance
        and flatness > tolerance
    )
    return ExactDualHolonomyAudit(
        face_id=trace.hinge,
        path_labels=labels,
        minimum_geodesic_denominator=min(denominators),
        maximum_hyperboloid_residual=max(hyperboloid_residuals),
        maximum_segment_tangency_residual=max(tangency_residuals),
        lorentz_residual=lorentz_residual,
        flatness_residual=flatness,
        holonomy=holonomy,
        projective_barycentric_realization=nested_flags,
        positive_cone_segment_inclusion=positive_cone_inclusion,
        regge_deficit_angle_derived=False,
        closed_dual_curvature_holonomy=closed,
        status=(
            "EXACT_NONTRIVIAL_DE_SITTER_PROJECTIVE_DUAL_LOOP_HOLONOMY"
            if closed
            else "DUAL_HINGE_HOLONOMY_AUDIT_FAILED"
        ),
    )


@dataclass(frozen=True)
class CurvedPlebanskiHingeCertificate:
    trace: CurvedHingeTraceAudit
    curvature_times_reference_length_squared: float
    cosmological_constant_times_reference_length_squared: float
    convex_hull_patch_denominator_lower_bound: float
    sampled_point_audits: tuple[DeSitterPlebanskiPointAudit, ...]
    maximum_sampled_field_residual: float
    field_evidence_scope: str
    primal_holonomy: ExactPrimalHolonomyAudit
    dual_holonomy: ExactDualHolonomyAudit
    einstein_endpoint: ConstantCurvatureEinsteinAudit
    same_history_nonflat_plebanski_witness_closed: bool
    proper_vertex_amplitude_derived: bool
    continuum_refinement_derived: bool
    two_dof_ir_spectrum_derived: bool
    status: str
    claim_ceiling: str = (
        "FINITE_NONFLAT_CONSTANT_CURVATURE_CONDITIONAL_WITNESS_NOT_CONTINUUM_QG"
    )


def constructive_curved_plebanski_hinge_witness(
    *,
    curvature_times_reference_length_squared: float = 1.0,
    cosmological_constant_times_reference_length_squared: float | None = None,
    history_id: str = "CE-C4-H0-CURVED",
) -> CurvedPlebanskiHingeCertificate:
    """Build the minimum same-history non-flat Plebanski/Einstein witness."""

    kappa = _require_finite(
        "curvature_times_reference_length_squared",
        curvature_times_reference_length_squared,
    )
    if not 0.0 < kappa < 4.0:
        raise ValueError("curvature must lie in (0,4) for the certified convex patch")
    lambda_bar = (
        3.0 * kappa
        if cosmological_constant_times_reference_length_squared is None
        else _require_finite(
            "cosmological_constant_times_reference_length_squared",
            cosmological_constant_times_reference_length_squared,
        )
    )
    trace = curved_hinge_trace_audit(history_id=history_id)
    coordinates = reference_vertex_coordinates()
    sample_points = tuple(coordinates.values()) + tuple(
        np.mean(np.asarray([coordinates[vertex] for vertex in simplex]), axis=0)
        for simplex in trace.extended_simplices
    )
    points = tuple(
        de_sitter_plebanski_point_audit(
            point,
            curvature_times_reference_length_squared=kappa,
            cosmological_constant_times_reference_length_squared=lambda_bar,
        )
        for point in sample_points
    )
    primal = exact_primal_triangle_holonomy(
        curvature_times_reference_length_squared=kappa,
        face_id=trace.hinge,
    )
    dual = exact_dual_hinge_holonomy(
        trace,
        curvature_times_reference_length_squared=kappa,
    )
    endpoint = constant_curvature_einstein_audit(kappa)
    maximum_field_residual = max(
        max(
            point.torsion_residual,
            point.riemann_constant_curvature_residual,
            point.self_duality_residual,
            point.simplicity_tracefree_residual,
            point.covariant_constancy_residual,
            point.plebanski_curvature_residual,
        )
        for point in points
    )
    patch_lower_bound = 1.0 - 0.25 * kappa
    closed = all(
        (
            trace.closed_three_cycle,
            trace.minimal_one_simplex_closure,
            trace.original_cells_preserved,
            trace.causal_label_matches_hinge,
            patch_lower_bound > 0.0,
            maximum_field_residual <= 1.0e-9,
            primal.nontrivial_curvature_holonomy,
            dual.closed_dual_curvature_holonomy,
            endpoint.lorentzian_einstein_geometry,
            abs(lambda_bar - 3.0 * kappa) <= 1.0e-12,
        )
    )
    return CurvedPlebanskiHingeCertificate(
        trace=trace,
        curvature_times_reference_length_squared=kappa,
        cosmological_constant_times_reference_length_squared=lambda_bar,
        convex_hull_patch_denominator_lower_bound=patch_lower_bound,
        sampled_point_audits=points,
        maximum_sampled_field_residual=maximum_field_residual,
        field_evidence_scope=(
            "ANALYTIC_ALL_POINTS_IDENTITIES_WITH_NINE_POINT_NUMERICAL_REGRESSION"
        ),
        primal_holonomy=primal,
        dual_holonomy=dual,
        einstein_endpoint=endpoint,
        same_history_nonflat_plebanski_witness_closed=closed,
        proper_vertex_amplitude_derived=False,
        continuum_refinement_derived=False,
        two_dof_ir_spectrum_derived=False,
        status=(
            "SAME_HISTORY_NONFLAT_PLEBANSKI_HINGE_WITNESS_CLOSED"
            if closed
            else "NONFLAT_PLEBANSKI_HINGE_WITNESS_FAILED"
        ),
    )
