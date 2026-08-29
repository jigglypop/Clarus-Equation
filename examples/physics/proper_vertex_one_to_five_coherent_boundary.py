'''Intrinsic tetrahedron closure and direction spinors on Lorentzian 1-to-5.

Every tetrahedron in the classical Lorentzian gluing skeleton is spacelike.
Its induced positive Gram matrix therefore defines a deterministic intrinsic
Euclidean coordinate representative after sorting global vertex labels and
applying a Cholesky convention.  The four outward area vectors close, and each
unit face normal admits a fixed Hopf-section spinor whose Pauli expectation is
that normal.

These are geometric direction spinors only.  No half-integer spins, area
spectrum scale, Livine-Speziale group average, time orientation, bra-ket
dualization, SU(2)/SL(2,C) frame lift, proper projector, or amplitude is built.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np

from examples.physics.proper_vertex_boundary import (
    MINKOWSKI_DIAGONAL,
    RationalVector,
    TetrahedronId,
    VertexId,
    spacelike_tetrahedron_audit,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    BOUNDARY_TETRAHEDRA,
    FINE_SIMPLICES,
    INTERNAL_TETRAHEDRA,
    certify_lorentzian_one_to_five_classical_gluing,
    lorentzian_one_to_five_coordinates,
    triangle_area_squared,
)


TriangleId = tuple[VertexId, VertexId, VertexId]


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _minkowski_product(left: RationalVector, right: RationalVector) -> Fraction:
    return sum(
        metric * a * b
        for metric, a, b in zip(MINKOWSKI_DIAGONAL, left, right)
    )


def direction_spinor(
    unit_normal: Sequence[float],
) -> tuple[complex, complex]:
    '''Return a stable two-chart Hopf representative of a unit normal.'''

    normal = np.asarray(unit_normal, dtype=float)
    if (
        normal.shape != (3,)
        or not np.all(np.isfinite(normal))
        or not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1.0e-12)
    ):
        raise ValueError('unit_normal must be a finite unit three-vector')
    x_value, y_value, z_value = (float(value) for value in normal)
    if z_value >= 0.0:
        first = math.sqrt((1.0 + z_value) / 2.0)
        second = complex(x_value, y_value) / math.sqrt(
            2.0 * (1.0 + z_value)
        )
        return (complex(first), second)
    first = complex(x_value, -y_value) / math.sqrt(
        2.0 * (1.0 - z_value)
    )
    second = math.sqrt((1.0 - z_value) / 2.0)
    return (first, complex(second))


def spinor_pauli_direction(spinor: Sequence[complex]) -> np.ndarray:
    '''Return xi-dagger sigma xi for a two-component spinor.'''

    values = np.asarray(spinor, dtype=complex)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError('spinor must contain two finite complex components')
    first, second = values
    overlap = np.conjugate(first) * second
    return np.asarray(
        (
            2.0 * float(np.real(overlap)),
            2.0 * float(np.imag(overlap)),
            float(abs(first) ** 2 - abs(second) ** 2),
        )
    )


def relative_closure_residual(area_vectors: Sequence[Sequence[float]]) -> float:
    '''Return ||sum A_f||/sum ||A_f|| for nonzero face-area vectors.'''

    vectors = np.asarray(area_vectors, dtype=float)
    if (
        vectors.ndim != 2
        or vectors.shape[1:] != (3,)
        or len(vectors) == 0
        or not np.all(np.isfinite(vectors))
    ):
        raise ValueError('area_vectors must be a finite nonempty N by 3 array')
    scale = float(np.sum(np.linalg.norm(vectors, axis=1)))
    if scale <= 0.0:
        raise ValueError('area_vectors must have positive total area')
    return float(np.linalg.norm(np.sum(vectors, axis=0)) / scale)


@dataclass(frozen=True)
class IntrinsicFaceDirectionData:
    face_vertices: TriangleId
    opposite_vertex: VertexId
    area_squared_exact: Fraction
    normalized_area_vector: tuple[float, float, float]
    unit_normal: tuple[float, float, float]
    direction_spinor: tuple[complex, complex]
    spinor_norm_residual: float
    pauli_map_residual: float


@dataclass(frozen=True)
class IntrinsicTetrahedronDirectionData:
    tetrahedron: TetrahedronId
    intrinsic_gram_exact: tuple[tuple[Fraction, Fraction, Fraction], ...]
    intrinsic_gram_scale_exact: Fraction
    face_data: tuple[IntrinsicFaceDirectionData, ...]
    normalized_closure_vector: tuple[float, float, float]
    relative_closure_residual: float
    nondegenerate_spacelike: bool
    all_face_areas_positive: bool


def intrinsic_tetrahedron_direction_data(
    tetrahedron: TetrahedronId,
    coordinates: Mapping[VertexId, RationalVector],
    *,
    tolerance: float = 1.0e-12,
) -> IntrinsicTetrahedronDirectionData:
    '''Construct deterministic intrinsic normals and Hopf representatives.'''

    vertices = tuple(sorted(tetrahedron))
    if len(vertices) != 4 or len(set(vertices)) != 4:
        raise ValueError('tetrahedron must contain four distinct vertices')
    if any(vertex not in coordinates for vertex in vertices):
        raise ValueError('every tetrahedron vertex must have coordinates')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    audit = spacelike_tetrahedron_audit(vertices, coordinates)
    if not audit.nondegenerate_spacelike:
        raise ValueError('tetrahedron must be nondegenerate and spacelike')

    base = coordinates[vertices[0]]
    edges = tuple(_subtract(coordinates[vertex], base) for vertex in vertices[1:])
    gram_exact = tuple(
        tuple(_minkowski_product(left, right) for right in edges)
        for left in edges
    )
    gram_scale_exact = max(abs(value) for row in gram_exact for value in row)
    if gram_scale_exact <= 0:
        raise ValueError('tetrahedron intrinsic Gram scale must be positive')
    gram = np.asarray(
        [
            [float(value / gram_scale_exact) for value in row]
            for row in gram_exact
        ],
        dtype=float,
    )
    try:
        lower = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as error:
        raise ValueError('tetrahedron intrinsic Gram matrix must be positive') from error
    intrinsic_coordinates = np.vstack((np.zeros(3), lower))
    face_records: list[IntrinsicFaceDirectionData] = []
    area_vectors: list[np.ndarray] = []
    for opposite_index, opposite_vertex in enumerate(vertices):
        face_indices = tuple(index for index in range(4) if index != opposite_index)
        face_vertices = tuple(vertices[index] for index in face_indices)
        first, second, third = (
            intrinsic_coordinates[index] for index in face_indices
        )
        normalized_area_vector = 0.5 * np.cross(
            second - first,
            third - first,
        )
        face_centroid = (first + second + third) / 3.0
        inward_displacement = (
            intrinsic_coordinates[opposite_index] - face_centroid
        )
        if float(normalized_area_vector @ inward_displacement) > 0.0:
            normalized_area_vector = -normalized_area_vector
        normalized_area = float(np.linalg.norm(normalized_area_vector))
        if normalized_area <= 0.0:
            raise ValueError('tetrahedron face must have positive area')
        area_squared_exact = triangle_area_squared(face_vertices, coordinates)
        if area_squared_exact <= 0:
            raise ValueError('tetrahedron face must be spacelike')
        normalized_area_squared_exact = (
            area_squared_exact / gram_scale_exact**2
        )
        if not math.isclose(
            normalized_area * normalized_area,
            float(normalized_area_squared_exact),
            rel_tol=1.0e-10,
            abs_tol=tolerance,
        ):
            raise ValueError('intrinsic and Lorentzian triangle areas disagree')
        unit_normal = normalized_area_vector / normalized_area
        spinor = direction_spinor(unit_normal)
        spinor_values = np.asarray(spinor, dtype=complex)
        spinor_norm_residual = abs(float(np.vdot(spinor_values, spinor_values).real) - 1.0)
        pauli_residual = float(
            np.linalg.norm(spinor_pauli_direction(spinor) - unit_normal)
        )
        area_vectors.append(normalized_area_vector)
        face_records.append(
            IntrinsicFaceDirectionData(
                face_vertices=face_vertices,
                opposite_vertex=opposite_vertex,
                area_squared_exact=area_squared_exact,
                normalized_area_vector=tuple(
                    float(value) for value in normalized_area_vector
                ),
                unit_normal=tuple(float(value) for value in unit_normal),
                direction_spinor=spinor,
                spinor_norm_residual=spinor_norm_residual,
                pauli_map_residual=pauli_residual,
            )
        )
    closure_vector = np.sum(np.asarray(area_vectors), axis=0)
    return IntrinsicTetrahedronDirectionData(
        tetrahedron=vertices,
        intrinsic_gram_exact=gram_exact,  # type: ignore[arg-type]
        intrinsic_gram_scale_exact=gram_scale_exact,
        face_data=tuple(face_records),
        normalized_closure_vector=tuple(
            float(value) for value in closure_vector
        ),
        relative_closure_residual=relative_closure_residual(area_vectors),
        nondegenerate_spacelike=audit.nondegenerate_spacelike,
        all_face_areas_positive=all(
            face.area_squared_exact > 0 for face in face_records
        ),
    )


@dataclass(frozen=True)
class LorentzianOneToFiveIntrinsicDirectionSpinorCertificate:
    tetrahedron_count: int
    boundary_tetrahedron_count: int
    internal_tetrahedron_count: int
    tetrahedron_data: tuple[IntrinsicTetrahedronDirectionData, ...]
    internal_tetrahedron_incidence_counts: tuple[int, ...]
    max_relative_closure_residual: float
    max_spinor_norm_residual: float
    max_pauli_map_residual: float
    max_repeated_canonical_unit_normal_residual: float
    max_repeated_canonical_direction_spinor_residual: float
    classical_gluing_skeleton_closed: bool
    all_geometric_face_closures_verified: bool
    all_normalized_direction_spinors_materialized: bool
    repeated_area_squared_labels_match_exactly: bool
    canonical_shared_tetrahedron_reconstruction_is_repeatable: bool
    half_integer_spin_assignment_constructed: bool
    area_spectrum_scale_and_immirzi_parameter_selected: bool
    livine_speziale_coherent_intertwiners_constructed: bool
    tetrahedron_time_orientations_assigned: bool
    shared_bra_ket_dualization_constructed: bool
    independent_frame_su2_lifts_constructed: bool
    lorentzian_sl2c_lifts_constructed: bool
    proper_projectors_materialized: bool
    proper_single_vertex_integrals_evaluated: bool
    standard_proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'LORENTZIAN_1_TO_5_INTRINSIC_CLOSURE_AND_DIRECTION_SPINORS_ONLY'
    )


def certify_lorentzian_one_to_five_intrinsic_direction_spinors(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 1.0e-12,
) -> LorentzianOneToFiveIntrinsicDirectionSpinorCertificate:
    '''Certify all intrinsic tetrahedron closures and direction spinors.'''

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

    tetrahedra = BOUNDARY_TETRAHEDRA + INTERNAL_TETRAHEDRA
    tetrahedron_data = tuple(
        intrinsic_tetrahedron_direction_data(
            tetrahedron,
            placement,
            tolerance=tolerance,
        )
        for tetrahedron in tetrahedra
    )
    data_by_tetrahedron = {
        item.tetrahedron: item for item in tetrahedron_data
    }

    incidence_counts: list[int] = []
    shared_area_labels_exact = True
    max_shared_normal_residual = 0.0
    max_shared_spinor_residual = 0.0
    for tetrahedron in INTERNAL_TETRAHEDRA:
        canonical = tuple(sorted(tetrahedron))
        reference = data_by_tetrahedron[canonical]
        incident_data: list[IntrinsicTetrahedronDirectionData] = []
        for simplex in FINE_SIMPLICES:
            local_tetrahedra = tuple(
                tuple(face) for face in combinations(simplex, 4)
            )
            if any(set(face) == set(canonical) for face in local_tetrahedra):
                # Both incidences deliberately use the same globally sorted
                # Cholesky chart.  Equality below is convention-level
                # reproducibility, not independently framed simplex data.
                incident_data.append(
                    intrinsic_tetrahedron_direction_data(
                        canonical,
                        placement,
                        tolerance=tolerance,
                    )
                )
        incidence_counts.append(len(incident_data))
        reference_faces = {
            face.face_vertices: face for face in reference.face_data
        }
        for incidence in incident_data:
            incidence_faces = {
                face.face_vertices: face for face in incidence.face_data
            }
            shared_area_labels_exact = shared_area_labels_exact and all(
                reference_faces[face].area_squared_exact
                == incidence_faces[face].area_squared_exact
                for face in reference_faces
            )
            max_shared_normal_residual = max(
                max_shared_normal_residual,
                *(
                    float(
                        np.linalg.norm(
                            np.asarray(reference_faces[face].unit_normal)
                            - np.asarray(incidence_faces[face].unit_normal)
                        )
                    )
                    for face in reference_faces
                ),
            )
            max_shared_spinor_residual = max(
                max_shared_spinor_residual,
                *(
                    float(
                        np.linalg.norm(
                            np.asarray(
                                reference_faces[face].direction_spinor,
                                dtype=complex,
                            )
                            - np.asarray(
                                incidence_faces[face].direction_spinor,
                                dtype=complex,
                            )
                        )
                    )
                    for face in reference_faces
                ),
            )

    face_data = tuple(
        face for tetrahedron in tetrahedron_data for face in tetrahedron.face_data
    )
    max_closure = max(
        tetrahedron.relative_closure_residual
        for tetrahedron in tetrahedron_data
    )
    max_spinor_norm = max(face.spinor_norm_residual for face in face_data)
    max_pauli = max(face.pauli_map_residual for face in face_data)
    all_closures = all(
        tetrahedron.nondegenerate_spacelike
        and tetrahedron.all_face_areas_positive
        and tetrahedron.relative_closure_residual <= tolerance
        for tetrahedron in tetrahedron_data
    )
    all_spinors = all(
        face.spinor_norm_residual <= tolerance
        and face.pauli_map_residual <= tolerance
        for face in face_data
    )
    canonical_reconstruction_repeatable = (
        all(count == 2 for count in incidence_counts)
        and shared_area_labels_exact
        and max_shared_normal_residual <= tolerance
        and max_shared_spinor_residual <= tolerance
    )
    closed = (
        skeleton.classical_lorentzian_gluing_prerequisite_closed
        and len(tetrahedron_data) == 15
        and all_closures
        and all_spinors
        and canonical_reconstruction_repeatable
    )

    return LorentzianOneToFiveIntrinsicDirectionSpinorCertificate(
        tetrahedron_count=len(tetrahedron_data),
        boundary_tetrahedron_count=len(BOUNDARY_TETRAHEDRA),
        internal_tetrahedron_count=len(INTERNAL_TETRAHEDRA),
        tetrahedron_data=tetrahedron_data,
        internal_tetrahedron_incidence_counts=tuple(incidence_counts),
        max_relative_closure_residual=max_closure,
        max_spinor_norm_residual=max_spinor_norm,
        max_pauli_map_residual=max_pauli,
        max_repeated_canonical_unit_normal_residual=(
            max_shared_normal_residual
        ),
        max_repeated_canonical_direction_spinor_residual=(
            max_shared_spinor_residual
        ),
        classical_gluing_skeleton_closed=(
            skeleton.classical_lorentzian_gluing_prerequisite_closed
        ),
        all_geometric_face_closures_verified=all_closures,
        all_normalized_direction_spinors_materialized=all_spinors,
        repeated_area_squared_labels_match_exactly=shared_area_labels_exact,
        canonical_shared_tetrahedron_reconstruction_is_repeatable=(
            canonical_reconstruction_repeatable
        ),
        half_integer_spin_assignment_constructed=False,
        area_spectrum_scale_and_immirzi_parameter_selected=False,
        livine_speziale_coherent_intertwiners_constructed=False,
        tetrahedron_time_orientations_assigned=False,
        shared_bra_ket_dualization_constructed=False,
        independent_frame_su2_lifts_constructed=False,
        lorentzian_sl2c_lifts_constructed=False,
        proper_projectors_materialized=False,
        proper_single_vertex_integrals_evaluated=False,
        standard_proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_INTRINSIC_CLOSURE_AND_DIRECTION_SPINORS_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_INTRINSIC_DIRECTION_DATA_FAILED'
        ),
    )
