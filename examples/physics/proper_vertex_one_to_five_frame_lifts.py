'''Future/outward normals and rotation-free lifts on Lorentzian 1-to-5.

For every one of the 25 fine-cell/tetrahedron incidences, an exact cofactor
covector determines the spacelike tetrahedron's timelike normal line.  The
future unit representative and the outward side sign are stored separately:
an outward normal need not be future directed.  A canonical pure boost and
its positive Hermitian SL(2,C) lift materialize only a rotation-free coset
representative for that future normal.

These data prepare the normal part of ``N_a = +/- X_a T``.  They do not choose
the missing SU(2) tangent rotation, face bivectors, Regge spinor phases,
shared bra/ket dualization, EPRL Y_gamma map, or a proper vertex amplitude.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from examples.physics.causal_face_simplicity import (
    proper_orthochronous_residual,
)
from examples.physics.proper_vertex_boundary import (
    RationalVector,
    SimplexId,
    TetrahedronId,
    VertexId,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    BOUNDARY_TETRAHEDRA,
    FINE_SIMPLICES,
    INTERNAL_TETRAHEDRA,
    certify_lorentzian_one_to_five_classical_gluing,
    lorentzian_one_to_five_coordinates,
)


MINKOWSKI_METRIC = np.diag((-1.0, 1.0, 1.0, 1.0))
IDENTITY_TWO = np.eye(2, dtype=complex)
PAULI_MATRICES = (
    np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=complex),
    np.asarray(((0.0, -1.0j), (1.0j, 0.0)), dtype=complex),
    np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=complex),
)
HERMITEAN_VECTOR_BASIS = (IDENTITY_TWO,) + PAULI_MATRICES


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _determinant_three(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError('matrix must be three by three')
    first, second, third = matrix
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _covector_evaluation(
    covector: Sequence[Fraction],
    vector: Sequence[Fraction],
) -> Fraction:
    return sum(component * value for component, value in zip(covector, vector))


def _exact_minkowski_squared(vector: Sequence[Fraction]) -> Fraction:
    return -vector[0] ** 2 + sum(component**2 for component in vector[1:])


@dataclass(frozen=True)
class ExactTetrahedronFutureNormal:
    tetrahedron: TetrahedronId
    exact_future_covector: tuple[Fraction, Fraction, Fraction, Fraction]
    exact_future_contravariant_vector: tuple[
        Fraction,
        Fraction,
        Fraction,
        Fraction,
    ]
    exact_vector_squared: Fraction
    exact_tangent_annihilations: tuple[Fraction, Fraction, Fraction]
    future_unit_normal: np.ndarray
    unit_timelike_residual: float


def exact_tetrahedron_future_normal(
    tetrahedron: TetrahedronId,
    coordinates: Mapping[VertexId, RationalVector],
) -> ExactTetrahedronFutureNormal:
    '''Construct the exact normal line and a scale-safe future unit vector.'''

    vertices = tuple(sorted(tetrahedron))
    if len(vertices) != 4 or len(set(vertices)) != 4:
        raise ValueError('tetrahedron must contain four distinct vertices')
    if any(vertex not in coordinates for vertex in vertices):
        raise ValueError('all tetrahedron vertices need coordinates')
    base = coordinates[vertices[0]]
    tangents = tuple(
        _subtract(coordinates[vertex], base) for vertex in vertices[1:]
    )
    covector = tuple(
        (-1 if component % 2 else 1)
        * _determinant_three(
            tuple(
                tuple(
                    tangent[index]
                    for index in range(4)
                    if index != component
                )
                for tangent in tangents
            )
        )
        for component in range(4)
    )
    contravariant = (-covector[0], covector[1], covector[2], covector[3])
    if contravariant[0] < 0:
        covector = tuple(-value for value in covector)
        contravariant = tuple(-value for value in contravariant)
    squared = _exact_minkowski_squared(contravariant)
    annihilations = tuple(
        _covector_evaluation(covector, tangent) for tangent in tangents
    )
    if any(value != 0 for value in annihilations):
        raise ValueError('cofactor covector failed exact tangent annihilation')
    if squared >= 0 or contravariant[0] <= 0:
        raise ValueError('tetrahedron normal line must be future timelike')
    component_scale = max(abs(value) for value in contravariant)
    scaled = np.asarray(
        [float(value / component_scale) for value in contravariant]
    )
    scaled_squared = float(scaled @ MINKOWSKI_METRIC @ scaled)
    if scaled_squared >= 0.0:
        raise ValueError('scaled normal must remain timelike')
    future = scaled / math.sqrt(-scaled_squared)
    unit_residual = abs(float(future @ MINKOWSKI_METRIC @ future) + 1.0)
    return ExactTetrahedronFutureNormal(
        tetrahedron=vertices,
        exact_future_covector=covector,  # type: ignore[arg-type]
        exact_future_contravariant_vector=contravariant,  # type: ignore[arg-type]
        exact_vector_squared=squared,
        exact_tangent_annihilations=annihilations,
        future_unit_normal=future,
        unit_timelike_residual=unit_residual,
    )


def canonical_pure_boost(future_unit_normal: Sequence[float]) -> np.ndarray:
    '''Return the rotation-free proper boost sending e0 to the future normal.'''

    normal = np.asarray(future_unit_normal, dtype=float)
    if normal.shape != (4,) or not np.all(np.isfinite(normal)):
        raise ValueError('future_unit_normal must be a finite four-vector')
    if (
        normal[0] <= 0.0
        or abs(float(normal @ MINKOWSKI_METRIC @ normal) + 1.0) > 1.0e-10
    ):
        raise ValueError('future_unit_normal must be future unit timelike')
    spatial = normal[1:]
    boost = np.eye(4)
    boost[0, 0] = normal[0]
    boost[0, 1:] = spatial
    boost[1:, 0] = spatial
    boost[1:, 1:] += np.outer(spatial, spatial) / (1.0 + normal[0])
    return boost


def hermitian_sl2c_boost_lift(
    future_unit_normal: Sequence[float],
) -> np.ndarray:
    '''Return the positive Hermitian SL(2,C) lift of the pure boost.'''

    normal = np.asarray(future_unit_normal, dtype=float)
    canonical_pure_boost(normal)
    numerator = (1.0 + normal[0]) * IDENTITY_TWO
    for component, pauli in zip(normal[1:], PAULI_MATRICES):
        numerator = numerator + component * pauli
    return numerator / math.sqrt(2.0 * (1.0 + normal[0]))


def sl2c_lorentz_matrix(sl2c_element: np.ndarray) -> np.ndarray:
    '''Return the Lorentz matrix induced on x0 I + x.sigma.'''

    element = np.asarray(sl2c_element, dtype=complex)
    if element.shape != (2, 2) or not np.all(np.isfinite(element)):
        raise ValueError('sl2c_element must be a finite two by two matrix')
    lorentz = np.empty((4, 4), dtype=float)
    for column, basis in enumerate(HERMITEAN_VECTOR_BASIS):
        image = element @ basis @ np.conjugate(element.T)
        lorentz[0, column] = 0.5 * float(np.trace(image).real)
        for row, pauli in enumerate(PAULI_MATRICES, start=1):
            lorentz[row, column] = 0.5 * float(np.trace(pauli @ image).real)
    return lorentz


@dataclass(frozen=True)
class CellTetrahedronFrameIncidence:
    cell: SimplexId
    tetrahedron: TetrahedronId
    opposite_vertex: VertexId
    exact_future_covector: tuple[Fraction, Fraction, Fraction, Fraction]
    exact_normal_vector_squared: Fraction
    exact_face_evaluation: Fraction
    outward_side_sign: int
    future_unit_normal: np.ndarray
    outward_unit_normal: np.ndarray
    outward_is_future: bool
    pure_boost: np.ndarray
    hermitian_sl2c_lift: np.ndarray
    tangent_annihilation_residual: float
    unit_timelike_residual: float
    boost_e0_residual: float
    boost_lorentz_residual: float
    lift_normal_residual: float
    lift_determinant_residual: float
    lift_to_boost_residual: float


@dataclass(frozen=True)
class LorentzianOneToFiveFrameLiftCertificate:
    incidence_count: int
    boundary_tetrahedron_incidence_count: int
    internal_tetrahedron_incidence_count: int
    unique_tetrahedron_count: int
    incidence_data: tuple[CellTetrahedronFrameIncidence, ...]
    shared_internal_tetrahedron_incidence_counts: tuple[int, ...]
    shared_internal_absolute_face_evaluations: tuple[Fraction, ...]
    all_exact_normal_covectors_annihilate_tangents: bool
    all_normal_lines_timelike: bool
    all_future_unit_normal_representatives: bool
    all_outward_side_evaluations_negative: bool
    all_pure_boosts_proper_orthochronous: bool
    all_hermitian_sl2c_normal_lifts_verified: bool
    shared_future_normal_representatives_agree: bool
    shared_outward_normals_are_opposite: bool
    shared_exact_face_evaluations_are_opposite: bool
    max_tangent_annihilation_residual: float
    max_unit_timelike_residual: float
    max_boost_e0_residual: float
    max_boost_lorentz_residual: float
    max_lift_normal_residual: float
    max_lift_determinant_residual: float
    max_lift_to_boost_residual: float
    rotation_free_future_normal_coset_representatives_materialized: bool
    full_engle_zipfel_boundary_frames_constructed: bool
    local_su2_tangent_frames_constructed: bool
    face_bivectors_constructed: bool
    eprl_orientation_equation_verified: bool
    relative_regge_transports_constructed: bool
    face_spinor_phase_gluing_verified: bool
    local_ls_intertwiners_integrated_with_frames: bool
    shared_bra_ket_dualization_constructed: bool
    eprl_y_gamma_map_materialized: bool
    proper_projectors_materialized: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'FUTURE_OUTWARD_NORMALS_AND_ROTATION_FREE_COSET_LIFTS_ONLY'
    )


def certify_lorentzian_one_to_five_frame_lifts(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 2.0e-12,
) -> LorentzianOneToFiveFrameLiftCertificate:
    '''Certify all 25 normal incidences and their rotation-free lifts.'''

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

    records: list[CellTetrahedronFrameIncidence] = []
    for cell in FINE_SIMPLICES:
        for opposite_vertex in cell:
            tetrahedron = tuple(
                sorted(vertex for vertex in cell if vertex != opposite_vertex)
            )
            normal_data = exact_tetrahedron_future_normal(
                tetrahedron,
                placement,
            )
            base = placement[tetrahedron[0]]
            opposite_displacement = _subtract(
                placement[opposite_vertex],
                base,
            )
            evaluation = _covector_evaluation(
                normal_data.exact_future_covector,
                opposite_displacement,
            )
            if evaluation == 0:
                raise ValueError('opposite cell vertex must not lie in the face')
            outward_sign = -1 if evaluation > 0 else 1
            future = normal_data.future_unit_normal
            outward = outward_sign * future
            boost = canonical_pure_boost(future)
            lift = hermitian_sl2c_boost_lift(future)
            target_hermitian = future[0] * IDENTITY_TWO
            for component, pauli in zip(future[1:], PAULI_MATRICES):
                target_hermitian = target_hermitian + component * pauli
            records.append(
                CellTetrahedronFrameIncidence(
                    cell=cell,
                    tetrahedron=tetrahedron,
                    opposite_vertex=opposite_vertex,
                    exact_future_covector=normal_data.exact_future_covector,
                    exact_normal_vector_squared=(
                        normal_data.exact_vector_squared
                    ),
                    exact_face_evaluation=evaluation,
                    outward_side_sign=outward_sign,
                    future_unit_normal=future,
                    outward_unit_normal=outward,
                    outward_is_future=(outward_sign == 1),
                    pure_boost=boost,
                    hermitian_sl2c_lift=lift,
                    tangent_annihilation_residual=float(
                        max(
                            abs(value)
                            for value in normal_data.exact_tangent_annihilations
                        )
                    ),
                    unit_timelike_residual=(
                        normal_data.unit_timelike_residual
                    ),
                    boost_e0_residual=float(
                        np.linalg.norm(
                            boost @ np.asarray((1.0, 0.0, 0.0, 0.0))
                            - future
                        )
                    ),
                    boost_lorentz_residual=proper_orthochronous_residual(
                        boost
                    ),
                    lift_normal_residual=float(
                        np.linalg.norm(
                            lift @ np.conjugate(lift.T) - target_hermitian
                        )
                    ),
                    lift_determinant_residual=abs(
                        complex(np.linalg.det(lift)) - 1.0
                    ),
                    lift_to_boost_residual=float(
                        np.linalg.norm(sl2c_lorentz_matrix(lift) - boost)
                    ),
                )
            )

    shared_counts: list[int] = []
    shared_absolute_evaluations: list[Fraction] = []
    shared_future_agree = True
    shared_outward_opposite = True
    shared_evaluations_opposite = True
    for tetrahedron in INTERNAL_TETRAHEDRA:
        canonical = tuple(sorted(tetrahedron))
        incidences = tuple(
            record for record in records if record.tetrahedron == canonical
        )
        shared_counts.append(len(incidences))
        if len(incidences) != 2:
            shared_future_agree = False
            shared_outward_opposite = False
            shared_evaluations_opposite = False
            continue
        left, right = incidences
        shared_absolute_evaluations.append(abs(left.exact_face_evaluation))
        shared_future_agree = shared_future_agree and np.allclose(
            left.future_unit_normal,
            right.future_unit_normal,
            rtol=0.0,
            atol=tolerance,
        )
        shared_outward_opposite = shared_outward_opposite and np.allclose(
            left.outward_unit_normal,
            -right.outward_unit_normal,
            rtol=0.0,
            atol=tolerance,
        )
        shared_evaluations_opposite = shared_evaluations_opposite and (
            left.exact_face_evaluation == -right.exact_face_evaluation
        )

    tangent_exact = all(
        record.tangent_annihilation_residual == 0.0 for record in records
    )
    all_timelike = all(
        record.exact_normal_vector_squared < 0 for record in records
    )
    all_future = all(
        record.future_unit_normal[0] > 0.0
        and record.unit_timelike_residual <= tolerance
        for record in records
    )
    all_outward = all(
        record.outward_side_sign * record.exact_face_evaluation < 0
        for record in records
    )
    boosts_verified = all(
        record.boost_e0_residual <= tolerance
        and record.boost_lorentz_residual <= tolerance
        for record in records
    )
    lifts_verified = all(
        record.lift_normal_residual <= tolerance
        and record.lift_determinant_residual <= tolerance
        and record.lift_to_boost_residual <= tolerance
        for record in records
    )
    closed = (
        len(records) == 25
        and tangent_exact
        and all_timelike
        and all_future
        and all_outward
        and boosts_verified
        and lifts_verified
        and all(count == 2 for count in shared_counts)
        and shared_future_agree
        and shared_outward_opposite
        and shared_evaluations_opposite
    )
    return LorentzianOneToFiveFrameLiftCertificate(
        incidence_count=len(records),
        boundary_tetrahedron_incidence_count=sum(
            record.tetrahedron in BOUNDARY_TETRAHEDRA for record in records
        ),
        internal_tetrahedron_incidence_count=sum(
            record.tetrahedron in INTERNAL_TETRAHEDRA for record in records
        ),
        unique_tetrahedron_count=len(
            {record.tetrahedron for record in records}
        ),
        incidence_data=tuple(records),
        shared_internal_tetrahedron_incidence_counts=tuple(shared_counts),
        shared_internal_absolute_face_evaluations=tuple(
            shared_absolute_evaluations
        ),
        all_exact_normal_covectors_annihilate_tangents=tangent_exact,
        all_normal_lines_timelike=all_timelike,
        all_future_unit_normal_representatives=all_future,
        all_outward_side_evaluations_negative=all_outward,
        all_pure_boosts_proper_orthochronous=boosts_verified,
        all_hermitian_sl2c_normal_lifts_verified=lifts_verified,
        shared_future_normal_representatives_agree=shared_future_agree,
        shared_outward_normals_are_opposite=shared_outward_opposite,
        shared_exact_face_evaluations_are_opposite=(
            shared_evaluations_opposite
        ),
        max_tangent_annihilation_residual=max(
            record.tangent_annihilation_residual for record in records
        ),
        max_unit_timelike_residual=max(
            record.unit_timelike_residual for record in records
        ),
        max_boost_e0_residual=max(
            record.boost_e0_residual for record in records
        ),
        max_boost_lorentz_residual=max(
            record.boost_lorentz_residual for record in records
        ),
        max_lift_normal_residual=max(
            record.lift_normal_residual for record in records
        ),
        max_lift_determinant_residual=max(
            record.lift_determinant_residual for record in records
        ),
        max_lift_to_boost_residual=max(
            record.lift_to_boost_residual for record in records
        ),
        rotation_free_future_normal_coset_representatives_materialized=closed,
        full_engle_zipfel_boundary_frames_constructed=False,
        local_su2_tangent_frames_constructed=False,
        face_bivectors_constructed=False,
        eprl_orientation_equation_verified=False,
        relative_regge_transports_constructed=False,
        face_spinor_phase_gluing_verified=False,
        local_ls_intertwiners_integrated_with_frames=False,
        shared_bra_ket_dualization_constructed=False,
        eprl_y_gamma_map_materialized=False,
        proper_projectors_materialized=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_FUTURE_NORMAL_COSET_LIFTS_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_FRAME_LIFT_PREREQUISITE_FAILED'
        ),
    )
