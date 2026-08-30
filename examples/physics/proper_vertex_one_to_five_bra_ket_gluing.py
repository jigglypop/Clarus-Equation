'''Cell-local face transports and J-dualized spinor gluing.

For each of the ten tetrahedron pairs in each fine four-simplex, the common
triangle carries two outward normals in two oriented tangent frames.  A
proper edge-aligned rotation is chosen to send the left outward normal to the
negative right outward normal.  Its SU(2) lift then fixes the remaining U(1)
phase so that U_ab xi_ab = J xi_ba exactly up to floating-point residual.

This is a finite boundary-data section.  It does not insert the EPRL
Y_gamma map, prove a global spin-connection holonomy, choose a proper-sector
projector, evaluate an SL(2,C) integral, or construct the five-vertex sum.
'''

from __future__ import annotations

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
from examples.physics.proper_vertex_one_to_five_coherent_boundary import (
    direction_spinor,
    spinor_pauli_direction,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    oriented_tetrahedron_tangent_frame,
    su2_lift_of_rotation,
    su2_rotation_matrix,
)


TriangleId = tuple[VertexId, VertexId, VertexId]


def j_dual_spinor(spinor: Sequence[complex]) -> np.ndarray:
    '''Return J(z0,z1)=(-conj(z1),conj(z0)).'''

    values = np.asarray(spinor, dtype=complex)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError('spinor must contain two finite complex components')
    return np.asarray((-np.conjugate(values[1]), np.conjugate(values[0])))


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


@dataclass(frozen=True)
class LocalTriangleFaceFrame:
    tetrahedron: TetrahedronId
    triangle: TriangleId
    opposite_vertex: VertexId
    first_edge_axis: np.ndarray
    second_tangent_axis: np.ndarray
    outward_unit_normal: np.ndarray
    orientation_sign: int
    direction_spinor: np.ndarray
    frame_orthogonality_residual: float


def local_triangle_face_frame(
    tetrahedron: TetrahedronId,
    opposite_vertex: VertexId,
    coordinates: Mapping[VertexId, RationalVector],
) -> LocalTriangleFaceFrame:
    '''Construct an oriented triangle frame in a tetrahedron tangent chart.'''

    vertices = tuple(sorted(tetrahedron))
    if opposite_vertex not in vertices:
        raise ValueError('opposite_vertex must belong to tetrahedron')
    triangle = tuple(vertex for vertex in vertices if vertex != opposite_vertex)
    tangent_frame = oriented_tetrahedron_tangent_frame(vertices, coordinates)
    base = coordinates[vertices[0]]
    exact_edges = {
        vertex: _subtract(coordinates[vertex], base)
        for vertex in vertices[1:]
    }
    exact_edge_scale = max(
        abs(value) for edge in exact_edges.values() for value in edge
    )
    if exact_edge_scale <= 0:
        raise ValueError('tetrahedron edges must have positive exact scale')
    local_coordinates: dict[VertexId, np.ndarray] = {
        vertices[0]: np.zeros(3)
    }
    for vertex in vertices[1:]:
        edge = np.asarray(
            [float(value / exact_edge_scale) for value in exact_edges[vertex]]
        )
        local = np.linalg.solve(tangent_frame.full_lorentz_frame, edge)
        if abs(float(local[0])) > 2.0e-12:
            raise ValueError('tetrahedron edge must lie in tangent rest space')
        local_coordinates[vertex] = local[1:]

    first, second, third = (
        local_coordinates[vertex] for vertex in triangle
    )
    first_edge = second - first
    first_axis = first_edge / np.linalg.norm(first_edge)
    second_edge = third - first
    second_remainder = second_edge - first_axis * float(first_axis @ second_edge)
    second_axis = second_remainder / np.linalg.norm(second_remainder)
    normal = np.cross(first_edge, second_edge)
    face_centroid = (first + second + third) / 3.0
    inward = local_coordinates[opposite_vertex] - face_centroid
    if float(normal @ inward) > 0.0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    face_frame = np.column_stack((first_axis, second_axis, normal))
    determinant = float(np.linalg.det(face_frame))
    if abs(abs(determinant) - 1.0) > 1.0e-10:
        raise ValueError('triangle face frame must be orthonormal')
    return LocalTriangleFaceFrame(
        tetrahedron=vertices,
        triangle=triangle,  # type: ignore[arg-type]
        opposite_vertex=opposite_vertex,
        first_edge_axis=first_axis,
        second_tangent_axis=second_axis,
        outward_unit_normal=normal,
        orientation_sign=1 if determinant > 0.0 else -1,
        direction_spinor=np.asarray(direction_spinor(normal), dtype=complex),
        frame_orthogonality_residual=float(
            np.linalg.norm(face_frame.T @ face_frame - np.eye(3))
        ),
    )


@dataclass(frozen=True)
class CellLocalBraKetGluing:
    cell: SimplexId
    triangle: TriangleId
    left_tetrahedron: TetrahedronId
    right_tetrahedron: TetrahedronId
    left_face_orientation_sign: int
    right_face_orientation_sign: int
    second_tangent_transport_sign: int
    relative_rotation: np.ndarray
    relative_su2_lift: np.ndarray
    source_phase_correction: complex
    rotation_residual: float
    normal_antipodal_residual: float
    su2_rotation_residual: float
    j_dual_direction_residual: float
    phase_gluing_residual: float


def cell_local_bra_ket_gluing(
    cell: SimplexId,
    omitted_left: VertexId,
    omitted_right: VertexId,
    coordinates: Mapping[VertexId, RationalVector],
) -> CellLocalBraKetGluing:
    '''Build one oriented shared-triangle transport and phase equation.'''

    if omitted_left == omitted_right or not {
        omitted_left,
        omitted_right,
    }.issubset(cell):
        raise ValueError('omitted vertices must be distinct members of cell')
    left_tetrahedron = tuple(
        sorted(vertex for vertex in cell if vertex != omitted_left)
    )
    right_tetrahedron = tuple(
        sorted(vertex for vertex in cell if vertex != omitted_right)
    )
    left = local_triangle_face_frame(
        left_tetrahedron,
        omitted_right,
        coordinates,
    )
    right = local_triangle_face_frame(
        right_tetrahedron,
        omitted_left,
        coordinates,
    )
    if left.triangle != right.triangle:
        raise ValueError('tetrahedra must share the same triangle')

    tangent_sign = -left.orientation_sign * right.orientation_sign
    source_frame = np.column_stack(
        (left.first_edge_axis, left.second_tangent_axis, left.outward_unit_normal)
    )
    target_frame = np.column_stack(
        (
            right.first_edge_axis,
            tangent_sign * right.second_tangent_axis,
            -right.outward_unit_normal,
        )
    )
    rotation = target_frame @ source_frame.T
    lift = su2_lift_of_rotation(rotation)
    transported = lift @ left.direction_spinor
    target_spinor = j_dual_spinor(right.direction_spinor)
    overlap = complex(np.vdot(target_spinor, transported))
    if abs(overlap) <= 1.0e-14:
        raise ValueError('transported and J-dual spinors must overlap')
    phase_correction = np.conjugate(overlap / abs(overlap))
    corrected = phase_correction * transported
    return CellLocalBraKetGluing(
        cell=tuple(cell),
        triangle=left.triangle,
        left_tetrahedron=left_tetrahedron,  # type: ignore[arg-type]
        right_tetrahedron=right_tetrahedron,  # type: ignore[arg-type]
        left_face_orientation_sign=left.orientation_sign,
        right_face_orientation_sign=right.orientation_sign,
        second_tangent_transport_sign=tangent_sign,
        relative_rotation=rotation,
        relative_su2_lift=lift,
        source_phase_correction=phase_correction,
        rotation_residual=float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3))
            + abs(float(np.linalg.det(rotation)) - 1.0)
        ),
        normal_antipodal_residual=float(
            np.linalg.norm(
                rotation @ left.outward_unit_normal
                + right.outward_unit_normal
            )
        ),
        su2_rotation_residual=float(
            np.linalg.norm(su2_rotation_matrix(lift) - rotation)
        ),
        j_dual_direction_residual=float(
            np.linalg.norm(
                spinor_pauli_direction(target_spinor)
                + right.outward_unit_normal
            )
        ),
        phase_gluing_residual=float(np.linalg.norm(corrected - target_spinor)),
    )


@dataclass(frozen=True)
class LorentzianOneToFiveBraKetCertificate:
    fine_cell_count: int
    gluing_count: int
    gluing_data: tuple[CellLocalBraKetGluing, ...]
    second_tangent_preserving_count: int
    second_tangent_reversing_count: int
    all_relative_rotations_proper: bool
    all_outward_normals_mapped_antipodally: bool
    all_su2_lifts_verified: bool
    all_j_dualized_phase_equations_verified: bool
    max_residual: float
    edge_aligned_face_transport_sections_constructed: bool
    cell_local_j_dualized_matching_constructed: bool
    face_spinor_phase_gluing_verified: bool
    global_regge_levi_civita_holonomy_derived: bool
    eprl_y_gamma_map_materialized: bool
    proper_projectors_materialized: bool
    lorentzian_sl2c_group_integrals_evaluated: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = 'CELL_LOCAL_EDGE_ALIGNED_J_DUALIZED_GLUING_ONLY'


def certify_lorentzian_one_to_five_bra_ket_gluing(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 3.0e-12,
) -> LorentzianOneToFiveBraKetCertificate:
    '''Certify all fifty cell-local shared-triangle phase equations.'''

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
    records = tuple(
        cell_local_bra_ket_gluing(
            cell,
            omitted_left,
            omitted_right,
            placement,
        )
        for cell in FINE_SIMPLICES
        for omitted_left, omitted_right in combinations(sorted(cell), 2)
    )
    residuals = tuple(
        max(
            record.rotation_residual,
            record.normal_antipodal_residual,
            record.su2_rotation_residual,
            record.j_dual_direction_residual,
            record.phase_gluing_residual,
        )
        for record in records
    )
    closed = len(records) == 50 and max(residuals) <= tolerance
    preserving = sum(
        record.second_tangent_transport_sign == 1 for record in records
    )
    return LorentzianOneToFiveBraKetCertificate(
        fine_cell_count=len(FINE_SIMPLICES),
        gluing_count=len(records),
        gluing_data=records,
        second_tangent_preserving_count=preserving,
        second_tangent_reversing_count=len(records) - preserving,
        all_relative_rotations_proper=closed,
        all_outward_normals_mapped_antipodally=closed,
        all_su2_lifts_verified=closed,
        all_j_dualized_phase_equations_verified=closed,
        max_residual=max(residuals),
        edge_aligned_face_transport_sections_constructed=closed,
        cell_local_j_dualized_matching_constructed=closed,
        face_spinor_phase_gluing_verified=closed,
        global_regge_levi_civita_holonomy_derived=False,
        eprl_y_gamma_map_materialized=False,
        proper_projectors_materialized=False,
        lorentzian_sl2c_group_integrals_evaluated=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_CELL_LOCAL_BRA_KET_GLUING_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_BRA_KET_GLUING_FAILED'
        ),
    )
