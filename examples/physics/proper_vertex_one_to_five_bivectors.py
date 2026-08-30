'''Classical oriented triangle bivectors on the Lorentzian one-to-five witness.

For each of the fifty cell/triangle wedges this module keeps an exact rational
tangent wedge, applies an explicit Lorentzian Hodge convention, and constructs
the corresponding cell-oriented normal-plane bivector.  Independent formulas
using the two tetrahedron outward normals and the two tetrahedron rest frames
must agree with that exact-first construction.  The already certified global
flat coframe connection transports the canonical bivectors between cells.

The earlier edge-aligned SO(3) section is retained as a negative control: 26
of its 50 links do not map the second labelled physical triangle edge.  Its
overlap phase therefore remains a local U(1) convention, not a Regge-state or
Regge-action phase.  No Y_gamma map, proper projector, amplitude, or Hessian is
constructed here.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np

from examples.physics.lorentzian_bivector_reconstruction import (
    MINKOWSKI_METRIC,
    bivector_inner,
    hodge_dual,
)
from examples.physics.proper_vertex_boundary import (
    RationalVector,
    SimplexId,
    TetrahedronId,
    VertexId,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    FINE_SIMPLICES,
    INTERNAL_TRIANGLES,
    certify_lorentzian_one_to_five_classical_gluing,
    lorentzian_one_to_five_coordinates,
    triangle_area_squared,
)
from examples.physics.proper_vertex_one_to_five_bra_ket_gluing import (
    cell_local_bra_ket_gluing,
    local_triangle_face_frame,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    certify_lorentzian_one_to_five_frame_lifts,
    sl2c_lorentz_matrix,
)
from examples.physics.proper_vertex_one_to_five_global_connection import (
    certify_lorentzian_one_to_five_global_connection,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    oriented_tetrahedron_tangent_frame,
)


TriangleId = tuple[VertexId, VertexId, VertexId]
ExactBivector = tuple[
    tuple[Fraction, Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction, Fraction],
]
_METRIC_SIGNS = (Fraction(-1), Fraction(1), Fraction(1), Fraction(1))


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _permutation_sign(indices: tuple[int, int, int, int]) -> int:
    if len(set(indices)) < 4:
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


def _exact_wedge(first: Sequence[Fraction], second: Sequence[Fraction]) -> ExactBivector:
    return tuple(
        tuple(first[i] * second[j] - second[i] * first[j] for j in range(4))
        for i in range(4)
    )  # type: ignore[return-value]


def _exact_hodge_dual(bivector: ExactBivector) -> ExactBivector:
    # Signature (-,+,+,+), epsilon_0123=+1.  Raising all epsilon indices
    # contributes det(eta)=-1, matching hodge_dual in the shared runtime.
    result: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
    for i in range(4):
        row: list[Fraction] = []
        for j in range(4):
            value = Fraction(0)
            for k in range(4):
                for l in range(4):
                    epsilon_upper = -_permutation_sign((i, j, k, l))
                    value += (
                        Fraction(epsilon_upper, 2)
                        * _METRIC_SIGNS[k]
                        * _METRIC_SIGNS[l]
                        * bivector[k][l]
                    )
            row.append(value)
        result.append(tuple(row))
    return tuple(result)  # type: ignore[return-value]


def _exact_bivector_inner(first: ExactBivector, second: ExactBivector) -> Fraction:
    return sum(
        _METRIC_SIGNS[i] * _METRIC_SIGNS[j] * first[i][j] * second[i][j]
        for i in range(4)
        for j in range(4)
    )


def _determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError('orientation matrix must be four by four')
    work = [list(row) for row in matrix]
    sign = Fraction(1)
    determinant = Fraction(1)
    for column in range(4):
        pivot = next(
            (row for row in range(column, 4) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, 4):
            factor = work[row][column] / pivot_value
            for item in range(column + 1, 4):
                work[row][item] -= factor * work[column][item]
    return sign * determinant


def _exact_scale_free_bivector(bivector: ExactBivector) -> np.ndarray:
    scale = max(abs(value) for row in bivector for value in row)
    if scale <= 0:
        raise ValueError('bivector must be nonzero')
    numeric = np.asarray(
        [[float(value / scale) for value in row] for row in bivector]
    )
    norm_squared = -0.5 * bivector_inner(numeric, numeric)
    if norm_squared <= 0.0:
        raise ValueError('normal-plane bivector must have timelike norm')
    return numeric / math.sqrt(norm_squared)


def _numeric_wedge(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.outer(first, second) - np.outer(second, first)


def _unit_normal_plane_bivector(bivector: np.ndarray) -> np.ndarray:
    norm_squared = -0.5 * bivector_inner(bivector, bivector)
    if norm_squared <= 0.0:
        raise ValueError('normal-plane bivector must have timelike norm')
    return bivector / math.sqrt(norm_squared)


def _local_triangle_edge_directions(
    tetrahedron: TetrahedronId,
    triangle: TriangleId,
    coordinates: Mapping[VertexId, RationalVector],
) -> tuple[np.ndarray, np.ndarray]:
    frame = oriented_tetrahedron_tangent_frame(tetrahedron, coordinates)
    first = coordinates[triangle[0]]
    exact_edges = tuple(
        _subtract(coordinates[vertex], first) for vertex in triangle[1:]
    )
    scale = max(abs(value) for edge in exact_edges for value in edge)
    if scale <= 0:
        raise ValueError('triangle edges must be nonzero')
    directions: list[np.ndarray] = []
    for exact_edge in exact_edges:
        edge = np.asarray([float(value / scale) for value in exact_edge])
        local = np.linalg.solve(frame.full_lorentz_frame, edge)
        if abs(float(local[0])) > 4.0e-12:
            raise ValueError('triangle edge must lie in tetrahedron rest space')
        spatial = local[1:]
        directions.append(spatial / np.linalg.norm(spatial))
    return directions[0], directions[1]


@dataclass(frozen=True)
class CellTriangleBivector:
    cell: SimplexId
    omitted_left: VertexId
    omitted_right: VertexId
    left_tetrahedron: TetrahedronId
    right_tetrahedron: TetrahedronId
    triangle: TriangleId
    sigma_exact: ExactBivector
    sigma_area_squared_exact: Fraction
    canonical_b0_exact: ExactBivector
    cell_orientation_sign: int
    cell_oriented_unit_bivector: np.ndarray
    normal_plane_unit_bivector: np.ndarray
    left_rest_simple_unit_bivector: np.ndarray
    right_rest_simple_unit_bivector: np.ndarray
    exact_area_identity_holds: bool
    sigma_matrix_antisymmetry_residual: float
    reverse_label_antisymmetry_holds: bool
    normal_route_residual: float
    left_rest_route_residual: float
    right_rest_route_residual: float
    linear_simplicity_residual: float
    signed_orientation_residual: float
    first_shared_edge_shape_mismatch_residual: float
    second_shared_edge_shape_mismatch_residual: float


@dataclass(frozen=True)
class CrossCellBivectorTransport:
    source_cell: SimplexId
    target_cell: SimplexId
    triangle: TriangleId
    source_orientation_sign: int
    target_orientation_sign: int
    exact_canonical_b0_agreement: bool
    canonical_lorentz_transport_residual: float
    canonical_sl2c_adjoint_transport_residual: float
    signed_cell_bivector_transport_residual: float


@dataclass(frozen=True)
class InternalTriangleBivectorLoop:
    triangle: TriangleId
    ordered_cells: tuple[SimplexId, SimplexId, SimplexId]
    canonical_bivector_loop_residual: float


@dataclass(frozen=True)
class LorentzianOneToFiveBivectorCertificate:
    cell_wedge_count: int
    cross_cell_transport_count: int
    internal_triangle_loop_count: int
    wedge_data: tuple[CellTriangleBivector, ...]
    cross_cell_transports: tuple[CrossCellBivectorTransport, ...]
    internal_triangle_loops: tuple[InternalTriangleBivectorLoop, ...]
    hodge_convention: str
    spacetime_orientation_identifier: str
    all_exact_area_identities_hold: bool
    all_sigma_bivectors_antisymmetric: bool
    all_reverse_label_antisymmetries_verified: bool
    all_cross_cell_exact_canonical_bivectors_agree: bool
    all_three_bivector_routes_agree: bool
    all_classical_signed_orientation_equations_verified: bool
    all_cross_cell_bivector_transports_verified: bool
    all_internal_triangle_bivector_loops_verified: bool
    full_labeled_triangle_shape_gluing_failures: int
    full_labeled_triangle_shape_gluing_successes: int
    minimum_failed_second_edge_mismatch: float
    maximum_second_edge_mismatch: float
    max_wedge_residual: float
    max_cross_cell_transport_residual: float
    max_internal_triangle_loop_residual: float
    classical_oriented_bivector_geometry_constructed: bool
    old_local_phase_is_regge_phase: bool
    global_regge_spinor_phase_constructed: bool
    full_eprl_critical_orientation_equation_verified: bool
    global_eprl_state_constructed: bool
    eprl_y_gamma_map_materialized: bool
    proper_projectors_materialized: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = 'FIXED_FLAT_CLASSICAL_ORIENTED_BIVECTORS_ONLY'


def certify_lorentzian_one_to_five_bivectors(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 5.0e-12,
) -> LorentzianOneToFiveBivectorCertificate:
    '''Certify exact-first oriented bivectors and their flat transport.'''

    if coordinates is not None and scale != 1:
        raise ValueError('scale cannot be combined with explicit coordinates')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    placement = (
        lorentzian_one_to_five_coordinates(scale=scale)
        if coordinates is None
        else dict(coordinates)
    )
    skeleton = certify_lorentzian_one_to_five_classical_gluing(placement)
    if not skeleton.classical_lorentzian_gluing_prerequisite_closed:
        raise ValueError('classical Lorentzian gluing skeleton must be closed')
    frame_lifts = certify_lorentzian_one_to_five_frame_lifts(placement)
    incidences = {
        (record.cell, record.tetrahedron): record
        for record in frame_lifts.incidence_data
    }
    global_connection = certify_lorentzian_one_to_five_global_connection(
        placement
    )
    coframes = {record.cell: record for record in global_connection.cell_coframes}

    wedges: list[CellTriangleBivector] = []
    for cell in FINE_SIMPLICES:
        for omitted_left, omitted_right in combinations(sorted(cell), 2):
            left_tetrahedron = tuple(
                sorted(vertex for vertex in cell if vertex != omitted_left)
            )
            right_tetrahedron = tuple(
                sorted(vertex for vertex in cell if vertex != omitted_right)
            )
            triangle = tuple(
                sorted(set(left_tetrahedron).intersection(right_tetrahedron))
            )
            triangle_base = placement[triangle[0]]
            exact_first_edge = _subtract(placement[triangle[1]], triangle_base)
            exact_second_edge = _subtract(placement[triangle[2]], triangle_base)
            sigma = _exact_wedge(exact_first_edge, exact_second_edge)
            sigma_area_squared = _exact_bivector_inner(sigma, sigma) / 8
            expected_area_squared = triangle_area_squared(triangle, placement)
            canonical_b0 = tuple(
                tuple(value / 2 for value in row)
                for row in _exact_hodge_dual(sigma)
            )
            exact_b0_area_squared = -_exact_bivector_inner(
                canonical_b0, canonical_b0
            ) / 2

            omitted_vectors = (
                _subtract(placement[omitted_left], triangle_base),
                _subtract(placement[omitted_right], triangle_base),
            )
            orientation_columns = (
                exact_first_edge,
                exact_second_edge,
                omitted_vectors[0],
                omitted_vectors[1],
            )
            orientation_determinant = _determinant(
                tuple(
                    tuple(column[row] for column in orientation_columns)
                    for row in range(4)
                )
            )
            if orientation_determinant == 0:
                raise ValueError('cell triangle orientation must be nondegenerate')
            orientation_sign = -1 if orientation_determinant > 0 else 1
            reversed_orientation_determinant = _determinant(
                tuple(
                    tuple(
                        column[row]
                        for column in (
                            exact_first_edge,
                            exact_second_edge,
                            omitted_vectors[1],
                            omitted_vectors[0],
                        )
                    )
                    for row in range(4)
                )
            )
            reversed_orientation_sign = (
                -1 if reversed_orientation_determinant > 0 else 1
            )
            canonical_unit = _exact_scale_free_bivector(canonical_b0)
            cell_unit = orientation_sign * canonical_unit

            left_incidence = incidences[(cell, left_tetrahedron)]
            right_incidence = incidences[(cell, right_tetrahedron)]
            outward_wedge = _numeric_wedge(
                left_incidence.outward_unit_normal,
                right_incidence.outward_unit_normal,
            )
            normal_plane_unit = -_unit_normal_plane_bivector(outward_wedge)

            left_face = local_triangle_face_frame(
                left_tetrahedron, omitted_right, placement
            )
            right_face = local_triangle_face_frame(
                right_tetrahedron, omitted_left, placement
            )
            time_axis = np.asarray((1.0, 0.0, 0.0, 0.0))
            left_spatial_normal = np.concatenate(
                ((0.0,), left_incidence.outward_side_sign * left_face.outward_unit_normal)
            )
            right_spatial_normal = np.concatenate(
                ((0.0,), right_incidence.outward_side_sign * right_face.outward_unit_normal)
            )
            left_rest = -_numeric_wedge(time_axis, left_spatial_normal)
            right_rest = -_numeric_wedge(time_axis, right_spatial_normal)
            left_frame = oriented_tetrahedron_tangent_frame(
                left_tetrahedron, placement
            ).full_lorentz_frame
            right_frame = oriented_tetrahedron_tangent_frame(
                right_tetrahedron, placement
            ).full_lorentz_frame
            left_global = left_frame @ left_rest @ left_frame.T
            right_global = right_frame @ right_rest @ right_frame.T

            left_edges = _local_triangle_edge_directions(
                left_tetrahedron, triangle, placement
            )
            right_edges = _local_triangle_edge_directions(
                right_tetrahedron, triangle, placement
            )
            local_section = cell_local_bra_ket_gluing(
                cell, omitted_left, omitted_right, placement
            )
            first_edge_mismatch = float(
                np.linalg.norm(
                    local_section.relative_rotation @ left_edges[0]
                    - right_edges[0]
                )
            )
            second_edge_mismatch = float(
                np.linalg.norm(
                    local_section.relative_rotation @ left_edges[1]
                    - right_edges[1]
                )
            )

            simplicity_residuals: list[float] = []
            for normal in (
                left_incidence.future_unit_normal,
                right_incidence.future_unit_normal,
            ):
                normal_lower = MINKOWSKI_METRIC @ normal
                simplicity_residuals.append(
                    float(np.linalg.norm(normal_lower @ hodge_dual(cell_unit)))
                )
            wedges.append(
                CellTriangleBivector(
                    cell=cell,
                    omitted_left=omitted_left,
                    omitted_right=omitted_right,
                    left_tetrahedron=left_tetrahedron,  # type: ignore[arg-type]
                    right_tetrahedron=right_tetrahedron,  # type: ignore[arg-type]
                    triangle=triangle,  # type: ignore[arg-type]
                    sigma_exact=sigma,
                    sigma_area_squared_exact=sigma_area_squared,
                    canonical_b0_exact=canonical_b0,  # type: ignore[arg-type]
                    cell_orientation_sign=orientation_sign,
                    cell_oriented_unit_bivector=cell_unit,
                    normal_plane_unit_bivector=normal_plane_unit,
                    left_rest_simple_unit_bivector=left_rest,
                    right_rest_simple_unit_bivector=right_rest,
                    exact_area_identity_holds=(
                        sigma_area_squared
                        == expected_area_squared
                        == exact_b0_area_squared
                    ),
                    sigma_matrix_antisymmetry_residual=(
                        0.0
                        if all(
                            sigma[i][j] == -sigma[j][i]
                            for i in range(4)
                            for j in range(4)
                        )
                        else math.inf
                    ),
                    reverse_label_antisymmetry_holds=(
                        reversed_orientation_sign == -orientation_sign
                    ),
                    normal_route_residual=float(
                        np.linalg.norm(normal_plane_unit - cell_unit)
                    ),
                    left_rest_route_residual=float(
                        np.linalg.norm(left_global - cell_unit)
                    ),
                    right_rest_route_residual=float(
                        np.linalg.norm(right_global + cell_unit)
                    ),
                    linear_simplicity_residual=max(simplicity_residuals),
                    signed_orientation_residual=float(
                        np.linalg.norm(left_global + right_global)
                    ),
                    first_shared_edge_shape_mismatch_residual=first_edge_mismatch,
                    second_shared_edge_shape_mismatch_residual=second_edge_mismatch,
                )
            )

    by_triangle: dict[TriangleId, list[CellTriangleBivector]] = {}
    for wedge in wedges:
        by_triangle.setdefault(wedge.triangle, []).append(wedge)
    transports: list[CrossCellBivectorTransport] = []
    for triangle, records in by_triangle.items():
        for source, target in combinations(records, 2):
            source_coframe = coframes[source.cell]
            target_coframe = coframes[target.cell]
            lorentz = np.linalg.solve(
                target_coframe.lorentz_frame, source_coframe.lorentz_frame
            )
            sl2c = np.linalg.solve(
                target_coframe.sl2c_frame, source_coframe.sl2c_frame
            )
            sl2c_lorentz = sl2c_lorentz_matrix(sl2c)
            canonical_global = _exact_scale_free_bivector(
                source.canonical_b0_exact
            )
            source_inverse = np.linalg.inv(source_coframe.lorentz_frame)
            target_inverse = np.linalg.inv(target_coframe.lorentz_frame)
            source_local = source_inverse @ canonical_global @ source_inverse.T
            target_local = target_inverse @ canonical_global @ target_inverse.T
            expected_signed_factor = (
                source.cell_orientation_sign * target.cell_orientation_sign
            )
            source_signed = source.cell_orientation_sign * source_local
            target_signed = target.cell_orientation_sign * target_local
            transports.append(
                CrossCellBivectorTransport(
                    source_cell=source.cell,
                    target_cell=target.cell,
                    triangle=triangle,
                    source_orientation_sign=source.cell_orientation_sign,
                    target_orientation_sign=target.cell_orientation_sign,
                    exact_canonical_b0_agreement=(
                        source.canonical_b0_exact == target.canonical_b0_exact
                    ),
                    canonical_lorentz_transport_residual=float(
                        np.linalg.norm(
                            lorentz @ source_local @ lorentz.T - target_local
                        )
                    ),
                    canonical_sl2c_adjoint_transport_residual=float(
                        np.linalg.norm(
                            sl2c_lorentz @ source_local @ sl2c_lorentz.T
                            - target_local
                        )
                    ),
                    signed_cell_bivector_transport_residual=float(
                        np.linalg.norm(
                            lorentz @ source_signed @ lorentz.T
                            - expected_signed_factor * target_signed
                        )
                    ),
                )
            )

    loops: list[InternalTriangleBivectorLoop] = []
    for triangle in INTERNAL_TRIANGLES:
        incident = tuple(
            cell for cell in FINE_SIMPLICES if set(triangle).issubset(cell)
        )
        start = coframes[incident[0]]
        canonical_global = _exact_scale_free_bivector(
            by_triangle[triangle][0].canonical_b0_exact
        )
        start_inverse = np.linalg.inv(start.lorentz_frame)
        bivector = start_inverse @ canonical_global @ start_inverse.T
        transported = bivector.copy()
        for source_cell, target_cell in zip(
            incident, incident[1:] + incident[:1]
        ):
            transition = np.linalg.solve(
                coframes[target_cell].lorentz_frame,
                coframes[source_cell].lorentz_frame,
            )
            transported = transition @ transported @ transition.T
        loops.append(
            InternalTriangleBivectorLoop(
                triangle=triangle,
                ordered_cells=incident,  # type: ignore[arg-type]
                canonical_bivector_loop_residual=float(
                    np.linalg.norm(transported - bivector)
                ),
            )
        )

    wedge_residuals = tuple(
        max(
            record.sigma_matrix_antisymmetry_residual,
            record.normal_route_residual,
            record.left_rest_route_residual,
            record.right_rest_route_residual,
            record.linear_simplicity_residual,
            record.signed_orientation_residual,
            record.first_shared_edge_shape_mismatch_residual,
        )
        for record in wedges
    )
    transport_residuals = tuple(
        max(
            record.canonical_lorentz_transport_residual,
            record.canonical_sl2c_adjoint_transport_residual,
            record.signed_cell_bivector_transport_residual,
        )
        for record in transports
    )
    loop_residuals = tuple(
        record.canonical_bivector_loop_residual for record in loops
    )
    failed_shape_records = tuple(
        record
        for record in wedges
        if record.second_shared_edge_shape_mismatch_residual > tolerance
    )
    successful_shape_records = len(wedges) - len(failed_shape_records)
    closed = (
        len(wedges) == 50
        and len(transports) == 40
        and len(loops) == 10
        and all(record.exact_area_identity_holds for record in wedges)
        and all(record.reverse_label_antisymmetry_holds for record in wedges)
        and all(record.exact_canonical_b0_agreement for record in transports)
        and max(wedge_residuals) <= tolerance
        and max(transport_residuals) <= tolerance
        and max(loop_residuals) <= tolerance
        and len(failed_shape_records) == 26
        and successful_shape_records == 24
    )
    return LorentzianOneToFiveBivectorCertificate(
        cell_wedge_count=len(wedges),
        cross_cell_transport_count=len(transports),
        internal_triangle_loop_count=len(loops),
        wedge_data=tuple(wedges),
        cross_cell_transports=tuple(transports),
        internal_triangle_loops=tuple(loops),
        hodge_convention=(
            'signature(-,+,+,+); epsilon_0123=+1; '
            'B0=(1/2) star(Sigma); A^2=(1/8)Sigma_IJ Sigma^IJ'
        ),
        spacetime_orientation_identifier='GLOBAL_VERTEX_COORDINATE_ORDER_0_1_2_3',
        all_exact_area_identities_hold=all(
            record.exact_area_identity_holds for record in wedges
        ),
        all_sigma_bivectors_antisymmetric=all(
            record.sigma_matrix_antisymmetry_residual <= tolerance
            for record in wedges
        ),
        all_reverse_label_antisymmetries_verified=all(
            record.reverse_label_antisymmetry_holds for record in wedges
        ),
        all_cross_cell_exact_canonical_bivectors_agree=all(
            record.exact_canonical_b0_agreement for record in transports
        ),
        all_three_bivector_routes_agree=all(
            max(
                record.normal_route_residual,
                record.left_rest_route_residual,
                record.right_rest_route_residual,
                record.linear_simplicity_residual,
            )
            <= tolerance
            for record in wedges
        ),
        all_classical_signed_orientation_equations_verified=all(
            record.signed_orientation_residual <= tolerance for record in wedges
        ),
        all_cross_cell_bivector_transports_verified=(
            max(transport_residuals) <= tolerance
        ),
        all_internal_triangle_bivector_loops_verified=(
            max(loop_residuals) <= tolerance
        ),
        full_labeled_triangle_shape_gluing_failures=len(failed_shape_records),
        full_labeled_triangle_shape_gluing_successes=successful_shape_records,
        minimum_failed_second_edge_mismatch=(
            min(
                record.second_shared_edge_shape_mismatch_residual
                for record in failed_shape_records
            )
            if failed_shape_records
            else 0.0
        ),
        maximum_second_edge_mismatch=max(
            record.second_shared_edge_shape_mismatch_residual for record in wedges
        ),
        max_wedge_residual=max(wedge_residuals),
        max_cross_cell_transport_residual=max(transport_residuals),
        max_internal_triangle_loop_residual=max(loop_residuals),
        classical_oriented_bivector_geometry_constructed=closed,
        old_local_phase_is_regge_phase=False,
        global_regge_spinor_phase_constructed=False,
        full_eprl_critical_orientation_equation_verified=False,
        global_eprl_state_constructed=False,
        eprl_y_gamma_map_materialized=False,
        proper_projectors_materialized=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_CLASSICAL_ORIENTED_BIVECTORS_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_BIVECTOR_CONSTRUCTION_FAILED'
        ),
    )
