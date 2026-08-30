'''Oriented tangent frames on the Lorentzian one-to-five witness.

The globally sorted Cholesky charts used for intrinsic closure are not all
orientation compatible with the future-normal rest spaces: seven of the
fifteen comparison maps have determinant minus one and therefore admit no
SU(2) lift.  This module records that finite counterexample and takes the
surviving route.  In each tetrahedron rest space it builds a right-handed
triad directly from the first two sorted edge vectors, lifts the resulting
SO(3) rotation to SU(2), and combines it with the positive-Hermitian boost.

The construction fixes a coordinate section, not a physical gauge.  It does
not yet construct triangle bivectors, Regge spinor phases, shared bra/ket
dualization, an EPRL Y_gamma map, proper projectors, or a five-vertex
amplitude.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from examples.physics.proper_vertex_boundary import (
    RationalVector,
    TetrahedronId,
    VertexId,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    BOUNDARY_TETRAHEDRA,
    INTERNAL_TETRAHEDRA,
    certify_lorentzian_one_to_five_classical_gluing,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    IDENTITY_TWO,
    MINKOWSKI_METRIC,
    PAULI_MATRICES,
    canonical_pure_boost,
    exact_tetrahedron_future_normal,
    hermitian_sl2c_boost_lift,
    sl2c_lorentz_matrix,
)


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def su2_rotation_matrix(su2_element: np.ndarray) -> np.ndarray:
    '''Return the SO(3) action U sigma_j U-dagger = R_ij sigma_i.'''

    element = np.asarray(su2_element, dtype=complex)
    if element.shape != (2, 2) or not np.all(np.isfinite(element)):
        raise ValueError('su2_element must be a finite two by two matrix')
    rotation = np.empty((3, 3), dtype=float)
    for column, source in enumerate(PAULI_MATRICES):
        image = element @ source @ np.conjugate(element.T)
        for row, target in enumerate(PAULI_MATRICES):
            rotation[row, column] = 0.5 * float(
                np.trace(target @ image).real
            )
    return rotation


def _canonicalize_quaternion_sign(quaternion: np.ndarray) -> np.ndarray:
    for value in quaternion:
        if abs(float(value)) > 1.0e-15:
            return quaternion if value > 0.0 else -quaternion
    raise ValueError('rotation quaternion must be nonzero')


def su2_lift_of_rotation(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    '''Return one canonical-sign SU(2) lift of a proper rotation.'''

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError('rotation must be a finite three by three matrix')
    if (
        np.linalg.norm(matrix.T @ matrix - np.eye(3)) > 1.0e-10
        or abs(float(np.linalg.det(matrix)) - 1.0) > 1.0e-10
    ):
        raise ValueError('rotation must lie in SO(3)')

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(1.0 + trace)
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * math.sqrt(
                max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            )
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(
                max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            )
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * math.sqrt(
                max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            )
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion = _canonicalize_quaternion_sign(
        quaternion / np.linalg.norm(quaternion)
    )
    scalar, x_value, y_value, z_value = quaternion
    generator = (
        x_value * PAULI_MATRICES[0]
        + y_value * PAULI_MATRICES[1]
        + z_value * PAULI_MATRICES[2]
    )
    return scalar * IDENTITY_TWO - 1.0j * generator


@dataclass(frozen=True)
class OrientedTetrahedronTangentFrame:
    tetrahedron: TetrahedronId
    future_unit_normal: np.ndarray
    sorted_edge_orientation_sign_in_rest_space: int
    sorted_cholesky_to_rest_comparison: np.ndarray
    sorted_cholesky_comparison_determinant: float
    sorted_cholesky_comparison_residual: float
    right_handed_tangent_rotation: np.ndarray
    tangent_su2_lift: np.ndarray
    full_lorentz_frame: np.ndarray
    full_sl2c_frame: np.ndarray
    max_rest_edge_time_component: float
    tangent_rotation_residual: float
    tangent_su2_unitarity_residual: float
    tangent_su2_determinant_residual: float
    tangent_su2_rotation_residual: float
    full_frame_lorentz_residual: float
    full_frame_normal_residual: float
    full_sl2c_to_lorentz_residual: float


def oriented_tetrahedron_tangent_frame(
    tetrahedron: TetrahedronId,
    coordinates: Mapping[VertexId, RationalVector],
) -> OrientedTetrahedronTangentFrame:
    '''Build a right-handed tangent triad and its SU(2) lift.'''

    vertices = tuple(sorted(tetrahedron))
    if len(vertices) != 4 or len(set(vertices)) != 4:
        raise ValueError('tetrahedron must contain four distinct vertices')
    normal_data = exact_tetrahedron_future_normal(vertices, coordinates)
    future = normal_data.future_unit_normal
    boost = canonical_pure_boost(future)
    boost_lift = hermitian_sl2c_boost_lift(future)
    base = coordinates[vertices[0]]
    exact_edges = tuple(
        _subtract(coordinates[vertex], base) for vertex in vertices[1:]
    )
    exact_edge_scale = max(
        abs(value) for edge in exact_edges for value in edge
    )
    if exact_edge_scale <= 0:
        raise ValueError('tetrahedron edges must have positive exact scale')
    rest_edges: list[np.ndarray] = []
    rest_time_components: list[float] = []
    for exact_edge in exact_edges:
        edge = np.asarray(
            [float(value / exact_edge_scale) for value in exact_edge]
        )
        rest_edge = np.linalg.solve(boost, edge)
        rest_time_components.append(abs(float(rest_edge[0])))
        rest_edges.append(rest_edge[1:])
    common_scale = max(abs(float(value)) for edge in rest_edges for value in edge)
    if common_scale <= 0.0:
        raise ValueError('tetrahedron rest edges must be nonzero')
    first_edge, second_edge, third_edge = (
        edge / common_scale for edge in rest_edges
    )
    rest_edge_rows = np.vstack((first_edge, second_edge, third_edge))
    normalized_gram = rest_edge_rows @ rest_edge_rows.T
    cholesky_rows = np.linalg.cholesky(normalized_gram)
    cholesky_comparison = (
        rest_edge_rows.T @ np.linalg.inv(cholesky_rows.T)
    )
    first_axis = first_edge / np.linalg.norm(first_edge)
    second_remainder = second_edge - first_axis * float(first_axis @ second_edge)
    second_norm = float(np.linalg.norm(second_remainder))
    if second_norm <= 1.0e-14:
        raise ValueError('first two sorted rest edges must be independent')
    second_axis = second_remainder / second_norm
    third_axis = np.cross(first_axis, second_axis)
    tangent_rotation = np.column_stack(
        (first_axis, second_axis, third_axis)
    )
    orientation_value = float(np.linalg.det(cholesky_comparison))
    if abs(orientation_value) <= 1.0e-14:
        raise ValueError('sorted rest edges must have nonzero orientation')
    orientation_sign = 1 if orientation_value > 0.0 else -1

    tangent_lift = su2_lift_of_rotation(tangent_rotation)
    rest_rotation = np.eye(4)
    rest_rotation[1:, 1:] = tangent_rotation
    full_lorentz = boost @ rest_rotation
    full_sl2c = boost_lift @ tangent_lift
    return OrientedTetrahedronTangentFrame(
        tetrahedron=vertices,
        future_unit_normal=future,
        sorted_edge_orientation_sign_in_rest_space=orientation_sign,
        sorted_cholesky_to_rest_comparison=cholesky_comparison,
        sorted_cholesky_comparison_determinant=orientation_value,
        sorted_cholesky_comparison_residual=float(
            np.linalg.norm(
                cholesky_comparison.T @ cholesky_comparison - np.eye(3)
            )
        ),
        right_handed_tangent_rotation=tangent_rotation,
        tangent_su2_lift=tangent_lift,
        full_lorentz_frame=full_lorentz,
        full_sl2c_frame=full_sl2c,
        max_rest_edge_time_component=max(rest_time_components),
        tangent_rotation_residual=float(
            np.linalg.norm(tangent_rotation.T @ tangent_rotation - np.eye(3))
            + abs(float(np.linalg.det(tangent_rotation)) - 1.0)
        ),
        tangent_su2_unitarity_residual=float(
            np.linalg.norm(
                np.conjugate(tangent_lift.T) @ tangent_lift - IDENTITY_TWO
            )
        ),
        tangent_su2_determinant_residual=abs(
            complex(np.linalg.det(tangent_lift)) - 1.0
        ),
        tangent_su2_rotation_residual=float(
            np.linalg.norm(su2_rotation_matrix(tangent_lift) - tangent_rotation)
        ),
        full_frame_lorentz_residual=float(
            np.linalg.norm(
                full_lorentz.T @ MINKOWSKI_METRIC @ full_lorentz
                - MINKOWSKI_METRIC
            )
            + abs(float(np.linalg.det(full_lorentz)) - 1.0)
        ),
        full_frame_normal_residual=float(
            np.linalg.norm(
                full_lorentz @ np.asarray((1.0, 0.0, 0.0, 0.0)) - future
            )
        ),
        full_sl2c_to_lorentz_residual=float(
            np.linalg.norm(sl2c_lorentz_matrix(full_sl2c) - full_lorentz)
        ),
    )


@dataclass(frozen=True)
class LorentzianOneToFiveTangentFrameCertificate:
    tetrahedron_count: int
    frame_data: tuple[OrientedTetrahedronTangentFrame, ...]
    orientation_preserving_sorted_charts: int
    orientation_reversing_sorted_charts: int
    naive_all_cholesky_charts_admit_su2_lifts: bool
    all_right_handed_tangent_frames_constructed: bool
    all_tangent_su2_lifts_verified: bool
    all_full_sl2c_frame_sections_verified: bool
    max_residual: float
    local_tangent_su2_frames_constructed: bool
    face_bivectors_constructed: bool
    relative_regge_transports_constructed: bool
    face_spinor_phase_gluing_verified: bool
    shared_bra_ket_dualization_constructed: bool
    eprl_y_gamma_map_materialized: bool
    proper_projectors_materialized: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = 'ORIENTED_LOCAL_TANGENT_SU2_FRAME_SECTIONS_ONLY'


def certify_lorentzian_one_to_five_tangent_frames(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 2.0e-12,
) -> LorentzianOneToFiveTangentFrameCertificate:
    '''Certify fifteen coordinate-section tangent frames and SU(2) lifts.'''

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
    frames = tuple(
        oriented_tetrahedron_tangent_frame(tetrahedron, placement)
        for tetrahedron in BOUNDARY_TETRAHEDRA + INTERNAL_TETRAHEDRA
    )
    positive = sum(
        frame.sorted_edge_orientation_sign_in_rest_space == 1
        for frame in frames
    )
    negative = len(frames) - positive
    residuals = tuple(
        max(
            frame.max_rest_edge_time_component,
            frame.sorted_cholesky_comparison_residual,
            frame.tangent_rotation_residual,
            frame.tangent_su2_unitarity_residual,
            frame.tangent_su2_determinant_residual,
            frame.tangent_su2_rotation_residual,
            frame.full_frame_lorentz_residual,
            frame.full_frame_normal_residual,
            frame.full_sl2c_to_lorentz_residual,
        )
        for frame in frames
    )
    closed = len(frames) == 15 and max(residuals) <= tolerance
    return LorentzianOneToFiveTangentFrameCertificate(
        tetrahedron_count=len(frames),
        frame_data=frames,
        orientation_preserving_sorted_charts=positive,
        orientation_reversing_sorted_charts=negative,
        naive_all_cholesky_charts_admit_su2_lifts=(negative == 0),
        all_right_handed_tangent_frames_constructed=closed,
        all_tangent_su2_lifts_verified=closed,
        all_full_sl2c_frame_sections_verified=closed,
        max_residual=max(residuals),
        local_tangent_su2_frames_constructed=closed,
        face_bivectors_constructed=False,
        relative_regge_transports_constructed=False,
        face_spinor_phase_gluing_verified=False,
        shared_bra_ket_dualization_constructed=False,
        eprl_y_gamma_map_materialized=False,
        proper_projectors_materialized=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_ORIENTED_TANGENT_SU2_FRAMES_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_TANGENT_FRAME_CONSTRUCTION_FAILED'
        ),
    )
