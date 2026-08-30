'''Full-shape face transports and their parity obstruction.

For every cell/triangle wedge of the fixed Lorentzian one-to-five witness,
there is a unique orthogonal map that transports both sorted labelled
triangle edges and sends the source outward face normal to the antipode of
the target outward normal.  Exact four-dimensional orientation determinants
show that only twenty-four of those fifty maps are proper rotations.  The
remaining twenty-six are reflections and therefore have no SU(2) lift.

Changing one tetrahedron face convention can repair one link, but the link
parities contain an explicit negative cycle.  Hence no global per-tetrahedron
parity assignment repairs all fifty maps while retaining pointwise labels.

The genuine four-dimensional Lorentz transitions remain proper and form a
flat cocycle.  Their canonical-boost Wigner factors are local SO(3) sections,
not a functorial projection of the Lorentz connection.  This module does not
construct a Regge spinor phase, Y_gamma, a proper projector, or an amplitude.
'''

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np

from examples.physics.proper_vertex_boundary import (
    RationalVector,
    SimplexId,
    TetrahedronId,
    VertexId,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    FINE_SIMPLICES,
    certify_lorentzian_one_to_five_classical_gluing,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_bra_ket_gluing import (
    cell_local_bra_ket_gluing,
    local_triangle_face_frame,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    MINKOWSKI_METRIC,
    canonical_pure_boost,
    exact_tetrahedron_future_normal,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    oriented_tetrahedron_tangent_frame,
    su2_lift_of_rotation,
)


TriangleId = tuple[VertexId, VertexId, VertexId]


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _determinant_four(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError('matrix must be four by four')
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


def _scale_free_float(vector: Sequence[Fraction]) -> np.ndarray:
    scale = max(abs(value) for value in vector)
    if scale <= 0:
        raise ValueError('vector must be nonzero')
    return np.asarray([float(value / scale) for value in vector])


def _tetrahedron(cell: SimplexId, omitted: VertexId) -> TetrahedronId:
    vertices = tuple(sorted(vertex for vertex in cell if vertex != omitted))
    if len(vertices) != 4:
        raise ValueError('omitted vertex must belong to the four-simplex')
    return vertices  # type: ignore[return-value]


def _exact_face_orientation(
    tetrahedron: TetrahedronId,
    opposite_vertex: VertexId,
    coordinates: Mapping[VertexId, RationalVector],
) -> tuple[Fraction, int]:
    '''Return the exact determinant and outward face-frame orientation sign.'''

    triangle = tuple(vertex for vertex in tetrahedron if vertex != opposite_vertex)
    if len(triangle) != 3:
        raise ValueError('opposite_vertex must belong to tetrahedron')
    base = coordinates[triangle[0]]
    first_edge = _subtract(coordinates[triangle[1]], base)
    second_edge = _subtract(coordinates[triangle[2]], base)
    inward = _subtract(coordinates[opposite_vertex], base)
    future = exact_tetrahedron_future_normal(
        tetrahedron, coordinates
    ).exact_future_contravariant_vector
    columns = (future, first_edge, second_edge, inward)
    determinant = _determinant_four(
        tuple(tuple(column[row] for column in columns) for row in range(4))
    )
    if determinant == 0:
        raise ValueError('face orientation determinant must be nonzero')
    # In the proper Lorentz rest frame det(e1,e2,inward) has the same sign as
    # this four-determinant.  The outward normal points opposite to inward.
    return determinant, (-1 if determinant > 0 else 1)


def _local_labelled_edges(
    tetrahedron: TetrahedronId,
    triangle: TriangleId,
    coordinates: Mapping[VertexId, RationalVector],
) -> tuple[np.ndarray, np.ndarray]:
    frame = oriented_tetrahedron_tangent_frame(tetrahedron, coordinates)
    base = coordinates[triangle[0]]
    exact_edges = (
        _subtract(coordinates[triangle[1]], base),
        _subtract(coordinates[triangle[2]], base),
    )
    result: list[np.ndarray] = []
    for exact_edge in exact_edges:
        local = np.linalg.solve(frame.full_lorentz_frame, _scale_free_float(exact_edge))
        if abs(float(local[0])) > 5.0e-12:
            raise ValueError('triangle edge must lie in tetrahedron rest space')
        result.append(local[1:])
    return result[0], result[1]


def _wigner_factor(lorentz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    image_time_axis = lorentz @ np.asarray((1.0, 0.0, 0.0, 0.0))
    boost = canonical_pure_boost(image_time_axis)
    little_group = np.linalg.solve(boost, lorentz)
    off_block = little_group.copy()
    off_block[0, 0] -= 1.0
    off_block[1:, 1:] = 0.0
    return little_group[1:, 1:], little_group, float(np.linalg.norm(off_block))


@dataclass(frozen=True)
class FullShapeFaceTransport:
    cell: SimplexId
    triangle: TriangleId
    source_tetrahedron: TetrahedronId
    target_tetrahedron: TetrahedronId
    source_opposite_vertex: VertexId
    target_opposite_vertex: VertexId
    source_exact_orientation_determinant: Fraction
    target_exact_orientation_determinant: Fraction
    source_face_orientation_sign: int
    target_face_orientation_sign: int
    exact_map_determinant_sign: int
    full_shape_map: np.ndarray
    su2_lift: np.ndarray | None
    orthogonality_residual: float
    determinant_residual: float
    first_labelled_edge_residual: float
    second_labelled_edge_residual: float
    normal_antipode_residual: float
    reverse_transpose_residual: float
    improper_su2_rejection_verified: bool


@dataclass(frozen=True)
class TetrahedronParityConstraint:
    source_tetrahedron: TetrahedronId
    target_tetrahedron: TetrahedronId
    required_parity_product: int


@dataclass(frozen=True)
class NegativeParityCycle:
    tetrahedra: tuple[TetrahedronId, TetrahedronId, TetrahedronId]
    edge_products: tuple[int, int, int]
    cycle_product: int


@dataclass(frozen=True)
class LorentzWignerFaceTransport:
    cell: SimplexId
    triangle: TriangleId
    source_tetrahedron: TetrahedronId
    target_tetrahedron: TetrahedronId
    lorentz_transition: np.ndarray
    wigner_rotation: np.ndarray
    existing_local_rotation: np.ndarray
    lorentz_residual: float
    lorentz_determinant_residual: float
    lorentz_future_margin: float
    first_four_edge_residual: float
    second_four_edge_residual: float
    little_group_block_residual: float
    wigner_rotation_residual: float
    wigner_to_existing_local_residual: float


@dataclass(frozen=True)
class LorentzianOneToFiveReggeFaceCertificate:
    face_transport_count: int
    proper_full_shape_count: int
    improper_full_shape_count: int
    full_shape_transports: tuple[FullShapeFaceTransport, ...]
    parity_constraints: tuple[TetrahedronParityConstraint, ...]
    negative_parity_cycle: NegativeParityCycle
    parity_constraint_violation_count: int
    lorentz_wigner_transports: tuple[LorentzWignerFaceTransport, ...]
    all_exact_face_orientation_signs_match_numeric_frames: bool
    all_full_shape_maps_orthogonal: bool
    all_full_shape_maps_transport_both_labelled_edges: bool
    all_full_shape_maps_reverse_by_transpose: bool
    all_and_only_proper_full_shape_maps_admit_su2_lifts: bool
    global_tetrahedron_parity_assignment_exists: bool
    explicit_negative_parity_cycle_verified: bool
    all_lorentz_transitions_proper_orthochronous: bool
    all_lorentz_transitions_transport_both_four_edges: bool
    all_wigner_factors_proper_rotations: bool
    wigner_existing_local_agreement_count: int
    wigner_existing_local_maximal_mismatch_count: int
    max_lorentz_cocycle_residual: float
    min_wigner_loop_residual: float
    max_wigner_loop_residual: float
    max_wigner_loop_cell: SimplexId
    max_wigner_loop_omitted_vertices: tuple[VertexId, VertexId, VertexId]
    full_lorentz_transitions_form_flat_cocycle: bool
    wigner_factors_form_global_cocycle: bool
    global_pointwise_labelled_su2_face_transport_constructed: bool
    global_regge_spinor_phase_constructed: bool
    global_eprl_boundary_state_constructed: bool
    eprl_y_gamma_map_materialized: bool
    proper_projectors_materialized: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    max_residual: float
    status: str
    claim_ceiling: str = 'FULL_SHAPE_PARITY_NO_GO_AND_SPLIT_LORENTZ_TRANSPORT_ONLY'


def certify_lorentzian_one_to_five_regge_faces(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 8.0e-12,
) -> LorentzianOneToFiveReggeFaceCertificate:
    '''Certify the 24/26 face-map split and the global parity obstruction.'''

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

    frames: dict[TetrahedronId, np.ndarray] = {}
    full_shape_records: list[FullShapeFaceTransport] = []
    wigner_records: list[LorentzWignerFaceTransport] = []
    numeric_orientation_matches: list[bool] = []

    for cell in FINE_SIMPLICES:
        for omitted_source, omitted_target in combinations(sorted(cell), 2):
            source_tetrahedron = _tetrahedron(cell, omitted_source)
            target_tetrahedron = _tetrahedron(cell, omitted_target)
            source_face = local_triangle_face_frame(
                source_tetrahedron, omitted_target, placement
            )
            target_face = local_triangle_face_frame(
                target_tetrahedron, omitted_source, placement
            )
            if source_face.triangle != target_face.triangle:
                raise ValueError('tetrahedra must share the same labelled triangle')
            triangle = source_face.triangle
            source_exact, source_sign = _exact_face_orientation(
                source_tetrahedron, omitted_target, placement
            )
            target_exact, target_sign = _exact_face_orientation(
                target_tetrahedron, omitted_source, placement
            )
            numeric_orientation_matches.extend(
                (
                    source_sign == source_face.orientation_sign,
                    target_sign == target_face.orientation_sign,
                )
            )
            exact_map_sign = -source_sign * target_sign
            source_face_matrix = np.column_stack(
                (
                    source_face.first_edge_axis,
                    source_face.second_tangent_axis,
                    source_face.outward_unit_normal,
                )
            )
            target_face_matrix = np.column_stack(
                (
                    target_face.first_edge_axis,
                    target_face.second_tangent_axis,
                    -target_face.outward_unit_normal,
                )
            )
            full_shape = target_face_matrix @ source_face_matrix.T
            reverse_shape = source_face_matrix @ target_face_matrix.T
            source_edges = _local_labelled_edges(
                source_tetrahedron, triangle, placement
            )
            target_edges = _local_labelled_edges(
                target_tetrahedron, triangle, placement
            )
            lift: np.ndarray | None = None
            rejected = False
            if exact_map_sign == 1:
                lift = su2_lift_of_rotation(full_shape)
            else:
                try:
                    su2_lift_of_rotation(full_shape)
                except ValueError:
                    rejected = True
            full_shape_records.append(
                FullShapeFaceTransport(
                    cell=cell,
                    triangle=triangle,
                    source_tetrahedron=source_tetrahedron,
                    target_tetrahedron=target_tetrahedron,
                    source_opposite_vertex=omitted_target,
                    target_opposite_vertex=omitted_source,
                    source_exact_orientation_determinant=source_exact,
                    target_exact_orientation_determinant=target_exact,
                    source_face_orientation_sign=source_sign,
                    target_face_orientation_sign=target_sign,
                    exact_map_determinant_sign=exact_map_sign,
                    full_shape_map=full_shape,
                    su2_lift=lift,
                    orthogonality_residual=float(
                        np.linalg.norm(full_shape.T @ full_shape - np.eye(3))
                    ),
                    determinant_residual=abs(
                        float(np.linalg.det(full_shape)) - exact_map_sign
                    ),
                    first_labelled_edge_residual=float(
                        np.linalg.norm(full_shape @ source_edges[0] - target_edges[0])
                    ),
                    second_labelled_edge_residual=float(
                        np.linalg.norm(full_shape @ source_edges[1] - target_edges[1])
                    ),
                    normal_antipode_residual=float(
                        np.linalg.norm(
                            full_shape @ source_face.outward_unit_normal
                            + target_face.outward_unit_normal
                        )
                    ),
                    reverse_transpose_residual=float(
                        np.linalg.norm(reverse_shape - full_shape.T)
                    ),
                    improper_su2_rejection_verified=rejected,
                )
            )

            source_frame = frames.setdefault(
                source_tetrahedron,
                oriented_tetrahedron_tangent_frame(
                    source_tetrahedron, placement
                ).full_lorentz_frame,
            )
            target_frame = frames.setdefault(
                target_tetrahedron,
                oriented_tetrahedron_tangent_frame(
                    target_tetrahedron, placement
                ).full_lorentz_frame,
            )
            lorentz = np.linalg.solve(target_frame, source_frame)
            wigner, little_group, block_residual = _wigner_factor(lorentz)
            local = cell_local_bra_ket_gluing(
                cell, omitted_source, omitted_target, placement
            ).relative_rotation
            base = placement[triangle[0]]
            four_edge_residuals: list[float] = []
            for vertex in triangle[1:]:
                edge = _scale_free_float(_subtract(placement[vertex], base))
                source_edge = np.linalg.solve(source_frame, edge)
                target_edge = np.linalg.solve(target_frame, edge)
                four_edge_residuals.append(
                    float(np.linalg.norm(lorentz @ source_edge - target_edge))
                )
            wigner_records.append(
                LorentzWignerFaceTransport(
                    cell=cell,
                    triangle=triangle,
                    source_tetrahedron=source_tetrahedron,
                    target_tetrahedron=target_tetrahedron,
                    lorentz_transition=lorentz,
                    wigner_rotation=wigner,
                    existing_local_rotation=local,
                    lorentz_residual=float(
                        np.linalg.norm(
                            lorentz.T @ MINKOWSKI_METRIC @ lorentz
                            - MINKOWSKI_METRIC
                        )
                    ),
                    lorentz_determinant_residual=abs(
                        float(np.linalg.det(lorentz)) - 1.0
                    ),
                    lorentz_future_margin=float(lorentz[0, 0] - 1.0),
                    first_four_edge_residual=four_edge_residuals[0],
                    second_four_edge_residual=four_edge_residuals[1],
                    little_group_block_residual=block_residual,
                    wigner_rotation_residual=float(
                        np.linalg.norm(wigner.T @ wigner - np.eye(3))
                        + abs(float(np.linalg.det(wigner)) - 1.0)
                    ),
                    wigner_to_existing_local_residual=float(
                        np.linalg.norm(wigner - local)
                    ),
                )
            )

    constraints = tuple(
        TetrahedronParityConstraint(
            source_tetrahedron=record.source_tetrahedron,
            target_tetrahedron=record.target_tetrahedron,
            required_parity_product=record.exact_map_determinant_sign,
        )
        for record in full_shape_records
    )
    adjacency: dict[TetrahedronId, list[tuple[TetrahedronId, int]]] = {}
    for constraint in constraints:
        adjacency.setdefault(constraint.source_tetrahedron, []).append(
            (constraint.target_tetrahedron, constraint.required_parity_product)
        )
        adjacency.setdefault(constraint.target_tetrahedron, []).append(
            (constraint.source_tetrahedron, constraint.required_parity_product)
        )
    assigned: dict[TetrahedronId, int] = {}
    for start in sorted(adjacency):
        if start in assigned:
            continue
        assigned[start] = 1
        queue = deque((start,))
        while queue:
            source = queue.popleft()
            for target, required in adjacency[source]:
                proposed = assigned[source] * required
                if target not in assigned:
                    assigned[target] = proposed
                    queue.append(target)
    violations = tuple(
        constraint
        for constraint in constraints
        if assigned[constraint.source_tetrahedron]
        * assigned[constraint.target_tetrahedron]
        != constraint.required_parity_product
    )

    tetrahedron_a: TetrahedronId = (1, 3, 4, 5)
    tetrahedron_b: TetrahedronId = (2, 3, 4, 5)
    tetrahedron_c: TetrahedronId = (0, 3, 4, 5)
    constraint_lookup = {
        frozenset((item.source_tetrahedron, item.target_tetrahedron)):
        item.required_parity_product
        for item in constraints
    }
    cycle_edges = (
        constraint_lookup[frozenset((tetrahedron_a, tetrahedron_b))],
        constraint_lookup[frozenset((tetrahedron_b, tetrahedron_c))],
        constraint_lookup[frozenset((tetrahedron_a, tetrahedron_c))],
    )
    cycle = NegativeParityCycle(
        tetrahedra=(tetrahedron_a, tetrahedron_b, tetrahedron_c),
        edge_products=cycle_edges,
        cycle_product=math.prod(cycle_edges),
    )

    def directed_lorentz_and_wigner(
        cell: SimplexId,
        omitted_source: VertexId,
        omitted_target: VertexId,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = frames[_tetrahedron(cell, omitted_source)]
        target = frames[_tetrahedron(cell, omitted_target)]
        lorentz = np.linalg.solve(target, source)
        wigner, _, _ = _wigner_factor(lorentz)
        return lorentz, wigner

    lorentz_loop_residuals: list[float] = []
    wigner_loop_residuals: list[float] = []
    wigner_loop_witnesses: list[
        tuple[SimplexId, tuple[VertexId, VertexId, VertexId]]
    ] = []
    for cell in FINE_SIMPLICES:
        for first, second, third in combinations(sorted(cell), 3):
            first_second = directed_lorentz_and_wigner(cell, first, second)
            second_third = directed_lorentz_and_wigner(cell, second, third)
            third_first = directed_lorentz_and_wigner(cell, third, first)
            lorentz_loop_residuals.append(
                float(
                    np.linalg.norm(
                        third_first[0] @ second_third[0] @ first_second[0]
                        - np.eye(4)
                    )
                )
            )
            wigner_loop_residuals.append(
                float(
                    np.linalg.norm(
                        third_first[1] @ second_third[1] @ first_second[1]
                        - np.eye(3)
                    )
                )
            )
            wigner_loop_witnesses.append((cell, (first, second, third)))

    proper = tuple(
        record for record in full_shape_records
        if record.exact_map_determinant_sign == 1
    )
    improper = tuple(
        record for record in full_shape_records
        if record.exact_map_determinant_sign == -1
    )
    shape_residuals = tuple(
        max(
            record.orthogonality_residual,
            record.determinant_residual,
            record.first_labelled_edge_residual,
            record.second_labelled_edge_residual,
            record.normal_antipode_residual,
            record.reverse_transpose_residual,
        )
        for record in full_shape_records
    )
    lorentz_residuals = tuple(
        max(
            record.lorentz_residual,
            record.lorentz_determinant_residual,
            max(0.0, -record.lorentz_future_margin),
            record.first_four_edge_residual,
            record.second_four_edge_residual,
            record.little_group_block_residual,
            record.wigner_rotation_residual,
        )
        for record in wigner_records
    )
    agreement_count = sum(
        record.wigner_to_existing_local_residual <= tolerance
        for record in wigner_records
    )
    maximal_mismatch_count = sum(
        abs(record.wigner_to_existing_local_residual - math.sqrt(8.0))
        <= tolerance
        for record in wigner_records
    )
    cycle_verified = cycle.edge_products == (1, 1, -1) and cycle.cycle_product == -1
    max_wigner_loop_index = int(np.argmax(wigner_loop_residuals))
    max_wigner_loop_witness = wigner_loop_witnesses[max_wigner_loop_index]
    closed = (
        len(full_shape_records) == 50
        and len(proper) == 24
        and len(improper) == 26
        and all(numeric_orientation_matches)
        and max(shape_residuals) <= tolerance
        and all(record.su2_lift is not None for record in proper)
        and all(record.improper_su2_rejection_verified for record in improper)
        and len(violations) > 0
        and cycle_verified
        and max(lorentz_residuals) <= tolerance
        and agreement_count == 24
        and maximal_mismatch_count == 26
        and max(lorentz_loop_residuals) <= tolerance
        and max(wigner_loop_residuals) > 1.0e-5
    )
    return LorentzianOneToFiveReggeFaceCertificate(
        face_transport_count=len(full_shape_records),
        proper_full_shape_count=len(proper),
        improper_full_shape_count=len(improper),
        full_shape_transports=tuple(full_shape_records),
        parity_constraints=constraints,
        negative_parity_cycle=cycle,
        parity_constraint_violation_count=len(violations),
        lorentz_wigner_transports=tuple(wigner_records),
        all_exact_face_orientation_signs_match_numeric_frames=all(
            numeric_orientation_matches
        ),
        all_full_shape_maps_orthogonal=(max(shape_residuals) <= tolerance),
        all_full_shape_maps_transport_both_labelled_edges=all(
            max(
                record.first_labelled_edge_residual,
                record.second_labelled_edge_residual,
                record.normal_antipode_residual,
            )
            <= tolerance
            for record in full_shape_records
        ),
        all_full_shape_maps_reverse_by_transpose=all(
            record.reverse_transpose_residual <= tolerance
            for record in full_shape_records
        ),
        all_and_only_proper_full_shape_maps_admit_su2_lifts=(
            all(record.su2_lift is not None for record in proper)
            and all(record.improper_su2_rejection_verified for record in improper)
        ),
        global_tetrahedron_parity_assignment_exists=(len(violations) == 0),
        explicit_negative_parity_cycle_verified=cycle_verified,
        all_lorentz_transitions_proper_orthochronous=all(
            max(
                record.lorentz_residual,
                record.lorentz_determinant_residual,
                max(0.0, -record.lorentz_future_margin),
            )
            <= tolerance
            for record in wigner_records
        ),
        all_lorentz_transitions_transport_both_four_edges=all(
            max(record.first_four_edge_residual, record.second_four_edge_residual)
            <= tolerance
            for record in wigner_records
        ),
        all_wigner_factors_proper_rotations=all(
            record.wigner_rotation_residual <= tolerance for record in wigner_records
        ),
        wigner_existing_local_agreement_count=agreement_count,
        wigner_existing_local_maximal_mismatch_count=maximal_mismatch_count,
        max_lorentz_cocycle_residual=max(lorentz_loop_residuals),
        min_wigner_loop_residual=min(wigner_loop_residuals),
        max_wigner_loop_residual=max(wigner_loop_residuals),
        max_wigner_loop_cell=max_wigner_loop_witness[0],
        max_wigner_loop_omitted_vertices=max_wigner_loop_witness[1],
        full_lorentz_transitions_form_flat_cocycle=(
            max(lorentz_loop_residuals) <= tolerance
        ),
        wigner_factors_form_global_cocycle=(
            max(wigner_loop_residuals) <= tolerance
        ),
        global_pointwise_labelled_su2_face_transport_constructed=False,
        global_regge_spinor_phase_constructed=False,
        global_eprl_boundary_state_constructed=False,
        eprl_y_gamma_map_materialized=False,
        proper_projectors_materialized=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        max_residual=max(max(shape_residuals), max(lorentz_residuals)),
        status=(
            'LORENTZIAN_1_TO_5_FULL_SHAPE_PARITY_NO_GO_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_REGGE_FACE_AUDIT_FAILED'
        ),
    )
