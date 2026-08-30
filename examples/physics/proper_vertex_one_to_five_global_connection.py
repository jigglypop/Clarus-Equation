'''Global flat Lorentz coframe connection on the one-to-five witness.

The five fine four-simplices already live in one affine Minkowski space.  A
deterministic proper-orthochronous frame is attached to each cell using its
unique boundary tetrahedron.  Relative cell frames then represent the same
global flat connection in five local gauges.  Their products telescope around
all ten internal-triangle dual loops.

This is a gauge-dependent section of the discrete Levi-Civita connection for
this globally embedded *flat* witness.  It is not the earlier cell-local
edge-aligned bra/ket section, and it does not construct a Regge boundary-state
phase, the EPRL Y_gamma map, proper projectors, or a five-vertex amplitude.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
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
    INTERNAL_TRIANGLES,
    certify_lorentzian_one_to_five_classical_gluing,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_bra_ket_gluing import (
    cell_local_bra_ket_gluing,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    MINKOWSKI_METRIC,
    certify_lorentzian_one_to_five_frame_lifts,
    sl2c_lorentz_matrix,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    oriented_tetrahedron_tangent_frame,
)


TriangleId = tuple[VertexId, VertexId, VertexId]


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _scale_free_float(vector: Sequence[Fraction]) -> np.ndarray:
    exact_scale = max(abs(component) for component in vector)
    if exact_scale <= 0:
        raise ValueError('a shared tangent vector must be nonzero')
    return np.asarray([float(component / exact_scale) for component in vector])


def _anchor_tetrahedron(cell: SimplexId) -> TetrahedronId:
    anchor = tuple(sorted(vertex for vertex in cell if vertex != 5))
    if len(anchor) != 4:
        raise ValueError('each fine cell must have one unique boundary tetrahedron')
    return anchor  # type: ignore[return-value]


@dataclass(frozen=True)
class CellLorentzCoframe:
    cell: SimplexId
    anchor_tetrahedron: TetrahedronId
    lorentz_frame: np.ndarray
    sl2c_frame: np.ndarray
    lorentz_residual: float
    determinant_residual: float
    future_orientation_margin: float
    sl2c_determinant_residual: float
    sl2c_to_lorentz_residual: float


@dataclass(frozen=True)
class SharedTetrahedronTransition:
    source_cell: SimplexId
    target_cell: SimplexId
    tetrahedron: TetrahedronId
    source_outward_sign: int
    target_outward_sign: int
    lorentz_transition: np.ndarray
    sl2c_transition: np.ndarray
    lorentz_residual: float
    determinant_residual: float
    future_orientation_margin: float
    inverse_residual: float
    sl2c_inverse_residual: float
    sl2c_determinant_residual: float
    sl2c_to_lorentz_residual: float
    shared_tangent_residual: float
    global_future_normal_agreement_residual: float
    shared_future_normal_residual: float
    outward_antipode_residual: float


@dataclass(frozen=True)
class InternalTriangleHolonomy:
    triangle: TriangleId
    ordered_cells: tuple[SimplexId, SimplexId, SimplexId]
    lorentz_holonomy: np.ndarray
    sl2c_holonomy: np.ndarray
    lorentz_identity_residual: float
    sl2c_identity_residual: float
    hinge_tangent_residual: float
    boost_trace_domain_residual: float
    boost_deficit: float


@dataclass(frozen=True)
class LorentzianOneToFiveGlobalConnectionCertificate:
    cell_count: int
    shared_tetrahedron_transition_count: int
    internal_triangle_holonomy_count: int
    cell_coframes: tuple[CellLorentzCoframe, ...]
    shared_transitions: tuple[SharedTetrahedronTransition, ...]
    internal_triangle_holonomies: tuple[InternalTriangleHolonomy, ...]
    all_shared_shapes_match_exactly: bool
    common_global_affine_frame_declared: bool
    all_cell_frames_proper_orthochronous: bool
    all_sl2c_cell_lifts_verified: bool
    all_shared_transitions_proper_orthochronous: bool
    all_transition_inverse_relations_verified: bool
    all_transition_cocycles_verified: bool
    all_shared_tangents_and_future_normals_preserved: bool
    all_shared_outward_normals_mapped_antipodally: bool
    all_internal_hinge_tangent_planes_fixed: bool
    all_internal_triangle_holonomies_identity: bool
    all_internal_regge_boost_deficits_zero: bool
    max_cell_frame_residual: float
    max_transition_residual: float
    max_cocycle_residual: float
    max_holonomy_residual: float
    max_cell_local_pairwise_loop_residual: float
    cell_local_pairwise_links_form_global_connection: bool
    global_flat_affine_levi_civita_connection_constructed: bool
    global_regge_spinor_phase_constructed: bool
    global_eprl_boundary_state_constructed: bool
    eprl_y_gamma_map_materialized: bool
    proper_projectors_materialized: bool
    lorentzian_sl2c_group_integrals_evaluated: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'FIXED_FLAT_EMBEDDED_GLOBAL_COFRAME_CONNECTION_ONLY'
    )


def _cell_coframe(
    cell: SimplexId,
    coordinates: Mapping[VertexId, RationalVector],
) -> CellLorentzCoframe:
    anchor = _anchor_tetrahedron(cell)
    frame = oriented_tetrahedron_tangent_frame(anchor, coordinates)
    lorentz = frame.full_lorentz_frame
    sl2c = frame.full_sl2c_frame
    return CellLorentzCoframe(
        cell=cell,
        anchor_tetrahedron=anchor,
        lorentz_frame=lorentz,
        sl2c_frame=sl2c,
        lorentz_residual=float(
            np.linalg.norm(
                lorentz.T @ MINKOWSKI_METRIC @ lorentz - MINKOWSKI_METRIC
            )
        ),
        determinant_residual=abs(float(np.linalg.det(lorentz)) - 1.0),
        future_orientation_margin=float(lorentz[0, 0] - 1.0),
        sl2c_determinant_residual=abs(complex(np.linalg.det(sl2c)) - 1.0),
        sl2c_to_lorentz_residual=float(
            np.linalg.norm(sl2c_lorentz_matrix(sl2c) - lorentz)
        ),
    )


def _transition_matrix(
    source: CellLorentzCoframe,
    target: CellLorentzCoframe,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linalg.solve(target.lorentz_frame, source.lorentz_frame),
        np.linalg.solve(target.sl2c_frame, source.sl2c_frame),
    )


def _cell_local_loop_negative_control(
    coordinates: Mapping[VertexId, RationalVector],
) -> float:
    residuals: list[float] = []
    for cell in FINE_SIMPLICES:
        for first, second, third in combinations(sorted(cell), 3):
            first_to_second = cell_local_bra_ket_gluing(
                cell, first, second, coordinates
            ).relative_rotation
            second_to_third = cell_local_bra_ket_gluing(
                cell, second, third, coordinates
            ).relative_rotation
            third_to_first = cell_local_bra_ket_gluing(
                cell, third, first, coordinates
            ).relative_rotation
            residuals.append(
                float(
                    np.linalg.norm(
                        third_to_first
                        @ second_to_third
                        @ first_to_second
                        - np.eye(3)
                    )
                )
            )
    return max(residuals)


def certify_lorentzian_one_to_five_global_connection(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 4.0e-12,
) -> LorentzianOneToFiveGlobalConnectionCertificate:
    '''Certify the flat affine coframe connection on the fixed witness.'''

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
    coframes = tuple(_cell_coframe(cell, placement) for cell in FINE_SIMPLICES)
    by_cell = {frame.cell: frame for frame in coframes}

    transitions: list[SharedTetrahedronTransition] = []
    for source_cell, target_cell in combinations(FINE_SIMPLICES, 2):
        shared = tuple(sorted(set(source_cell).intersection(target_cell)))
        if len(shared) != 4:
            raise ValueError('each pair of fine cells must share one tetrahedron')
        tetrahedron: TetrahedronId = shared  # type: ignore[assignment]
        source = by_cell[source_cell]
        target = by_cell[target_cell]
        lorentz, sl2c = _transition_matrix(source, target)
        reverse_lorentz, reverse_sl2c = _transition_matrix(target, source)
        source_incidence = incidences[(source_cell, tetrahedron)]
        target_incidence = incidences[(target_cell, tetrahedron)]

        shared_tangent_residual = 0.0
        base = placement[tetrahedron[0]]
        for vertex in tetrahedron[1:]:
            global_edge = _scale_free_float(
                _subtract(placement[vertex], base)
            )
            source_edge = np.linalg.solve(source.lorentz_frame, global_edge)
            target_edge = np.linalg.solve(target.lorentz_frame, global_edge)
            shared_tangent_residual = max(
                shared_tangent_residual,
                float(np.linalg.norm(lorentz @ source_edge - target_edge)),
            )

        source_global_future = source_incidence.future_unit_normal
        target_global_future = target_incidence.future_unit_normal
        source_future = np.linalg.solve(
            source.lorentz_frame, source_global_future
        )
        target_future = np.linalg.solve(
            target.lorentz_frame, target_global_future
        )
        source_outward = np.linalg.solve(
            source.lorentz_frame, source_incidence.outward_unit_normal
        )
        target_outward = np.linalg.solve(
            target.lorentz_frame, target_incidence.outward_unit_normal
        )
        transitions.append(
            SharedTetrahedronTransition(
                source_cell=source_cell,
                target_cell=target_cell,
                tetrahedron=tetrahedron,
                source_outward_sign=source_incidence.outward_side_sign,
                target_outward_sign=target_incidence.outward_side_sign,
                lorentz_transition=lorentz,
                sl2c_transition=sl2c,
                lorentz_residual=float(
                    np.linalg.norm(
                        lorentz.T @ MINKOWSKI_METRIC @ lorentz
                        - MINKOWSKI_METRIC
                    )
                ),
                determinant_residual=abs(float(np.linalg.det(lorentz)) - 1.0),
                future_orientation_margin=float(lorentz[0, 0] - 1.0),
                inverse_residual=float(
                    np.linalg.norm(reverse_lorentz @ lorentz - np.eye(4))
                ),
                sl2c_inverse_residual=float(
                    np.linalg.norm(reverse_sl2c @ sl2c - np.eye(2))
                ),
                sl2c_determinant_residual=abs(
                    complex(np.linalg.det(sl2c)) - 1.0
                ),
                sl2c_to_lorentz_residual=float(
                    np.linalg.norm(sl2c_lorentz_matrix(sl2c) - lorentz)
                ),
                shared_tangent_residual=shared_tangent_residual,
                global_future_normal_agreement_residual=float(
                    np.linalg.norm(source_global_future - target_global_future)
                ),
                shared_future_normal_residual=float(
                    np.linalg.norm(lorentz @ source_future - target_future)
                ),
                outward_antipode_residual=float(
                    np.linalg.norm(lorentz @ source_outward + target_outward)
                ),
            )
        )

    def directed_transition(
        source_cell: SimplexId,
        target_cell: SimplexId,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _transition_matrix(by_cell[source_cell], by_cell[target_cell])

    cocycle_residuals: list[float] = []
    for source_cell, middle_cell, target_cell in permutations(FINE_SIMPLICES, 3):
        source_to_middle, source_to_middle_sl2c = directed_transition(
            source_cell, middle_cell
        )
        middle_to_target, middle_to_target_sl2c = directed_transition(
            middle_cell, target_cell
        )
        source_to_target, source_to_target_sl2c = directed_transition(
            source_cell, target_cell
        )
        cocycle_residuals.append(
            max(
                float(
                    np.linalg.norm(
                        middle_to_target @ source_to_middle - source_to_target
                    )
                ),
                float(
                    np.linalg.norm(
                        middle_to_target_sl2c @ source_to_middle_sl2c
                        - source_to_target_sl2c
                    )
                ),
            )
        )

    holonomies: list[InternalTriangleHolonomy] = []
    for triangle in INTERNAL_TRIANGLES:
        incident = tuple(
            cell for cell in FINE_SIMPLICES if set(triangle).issubset(cell)
        )
        if len(incident) != 3:
            raise ValueError('each internal triangle must have three incident cells')
        lorentz_holonomy = np.eye(4)
        sl2c_holonomy = np.eye(2, dtype=complex)
        for source_cell, target_cell in zip(
            incident, incident[1:] + incident[:1]
        ):
            lorentz, sl2c = directed_transition(source_cell, target_cell)
            lorentz_holonomy = lorentz @ lorentz_holonomy
            sl2c_holonomy = sl2c @ sl2c_holonomy
        starting_frame = by_cell[incident[0]].lorentz_frame
        triangle_base = placement[triangle[0]]
        local_hinge_edges = tuple(
            np.linalg.solve(
                starting_frame,
                _scale_free_float(_subtract(placement[vertex], triangle_base)),
            )
            for vertex in triangle[1:]
        )
        hinge_tangent_residual = max(
            float(np.linalg.norm(lorentz_holonomy @ edge - edge))
            for edge in local_hinge_edges
        )
        # For a spacelike Regge hinge the orthogonal-plane holonomy is a
        # boost with tr(H)=2+2 cosh(delta).  The exact affine telescope has
        # H=I; this trace extraction independently returns |delta|=0.
        raw_boost_cosh = (float(np.trace(lorentz_holonomy)) - 2.0) / 2.0
        boost_trace_domain_residual = max(0.0, 1.0 - raw_boost_cosh)
        # Clamp only after recording the amount by which roundoff or a
        # nonboost holonomy left the real arcosh domain.  The recorded amount
        # is part of the closed gate below, so an invalid trace is not hidden.
        boost_cosh = max(1.0, raw_boost_cosh)
        holonomies.append(
            InternalTriangleHolonomy(
                triangle=triangle,
                ordered_cells=incident,  # type: ignore[arg-type]
                lorentz_holonomy=lorentz_holonomy,
                sl2c_holonomy=sl2c_holonomy,
                lorentz_identity_residual=float(
                    np.linalg.norm(lorentz_holonomy - np.eye(4))
                ),
                sl2c_identity_residual=float(
                    np.linalg.norm(sl2c_holonomy - np.eye(2))
                ),
                hinge_tangent_residual=hinge_tangent_residual,
                boost_trace_domain_residual=boost_trace_domain_residual,
                boost_deficit=math.acosh(boost_cosh),
            )
        )

    cell_residuals = tuple(
        max(
            frame.lorentz_residual,
            frame.determinant_residual,
            frame.sl2c_determinant_residual,
            frame.sl2c_to_lorentz_residual,
            max(0.0, -frame.future_orientation_margin),
        )
        for frame in coframes
    )
    transition_residuals = tuple(
        max(
            transition.lorentz_residual,
            transition.determinant_residual,
            transition.inverse_residual,
            transition.sl2c_inverse_residual,
            transition.sl2c_determinant_residual,
            transition.sl2c_to_lorentz_residual,
            transition.shared_tangent_residual,
            transition.global_future_normal_agreement_residual,
            transition.shared_future_normal_residual,
            transition.outward_antipode_residual,
            max(0.0, -transition.future_orientation_margin),
        )
        for transition in transitions
    )
    holonomy_residuals = tuple(
        max(
            item.lorentz_identity_residual,
            item.sl2c_identity_residual,
            item.hinge_tangent_residual,
            item.boost_trace_domain_residual,
            item.boost_deficit,
        )
        for item in holonomies
    )
    old_loop_residual = _cell_local_loop_negative_control(placement)
    closed = (
        len(coframes) == 5
        and len(transitions) == 10
        and len(holonomies) == 10
        and max(cell_residuals) <= tolerance
        and max(transition_residuals) <= tolerance
        and max(cocycle_residuals) <= tolerance
        and max(holonomy_residuals) <= tolerance
        and old_loop_residual > 1.0
        and skeleton.globally_embedded_shared_tetrahedron_intrinsic_shapes_match
        and skeleton.globally_embedded_triangle_area_squared_labels_consistent
        and all(
            transition.source_outward_sign == -transition.target_outward_sign
            for transition in transitions
        )
    )
    return LorentzianOneToFiveGlobalConnectionCertificate(
        cell_count=len(coframes),
        shared_tetrahedron_transition_count=len(transitions),
        internal_triangle_holonomy_count=len(holonomies),
        cell_coframes=coframes,
        shared_transitions=tuple(transitions),
        internal_triangle_holonomies=tuple(holonomies),
        all_shared_shapes_match_exactly=(
            skeleton.globally_embedded_shared_tetrahedron_intrinsic_shapes_match
            and skeleton.globally_embedded_triangle_area_squared_labels_consistent
        ),
        common_global_affine_frame_declared=True,
        all_cell_frames_proper_orthochronous=(max(cell_residuals) <= tolerance),
        all_sl2c_cell_lifts_verified=(max(cell_residuals) <= tolerance),
        all_shared_transitions_proper_orthochronous=(
            max(transition_residuals) <= tolerance
        ),
        all_transition_inverse_relations_verified=all(
            max(item.inverse_residual, item.sl2c_inverse_residual) <= tolerance
            for item in transitions
        ),
        all_transition_cocycles_verified=(max(cocycle_residuals) <= tolerance),
        all_shared_tangents_and_future_normals_preserved=all(
            max(
                item.shared_tangent_residual,
                item.global_future_normal_agreement_residual,
                item.shared_future_normal_residual,
            )
            <= tolerance
            for item in transitions
        ),
        all_shared_outward_normals_mapped_antipodally=all(
            item.source_outward_sign == -item.target_outward_sign
            and item.outward_antipode_residual <= tolerance
            for item in transitions
        ),
        all_internal_hinge_tangent_planes_fixed=all(
            item.hinge_tangent_residual <= tolerance for item in holonomies
        ),
        all_internal_triangle_holonomies_identity=(
            max(holonomy_residuals) <= tolerance
        ),
        all_internal_regge_boost_deficits_zero=(
            max(holonomy_residuals) <= tolerance
            and all(item.boost_deficit <= tolerance for item in holonomies)
        ),
        max_cell_frame_residual=max(cell_residuals),
        max_transition_residual=max(transition_residuals),
        max_cocycle_residual=max(cocycle_residuals),
        max_holonomy_residual=max(holonomy_residuals),
        max_cell_local_pairwise_loop_residual=old_loop_residual,
        cell_local_pairwise_links_form_global_connection=False,
        global_flat_affine_levi_civita_connection_constructed=closed,
        global_regge_spinor_phase_constructed=False,
        global_eprl_boundary_state_constructed=False,
        eprl_y_gamma_map_materialized=False,
        proper_projectors_materialized=False,
        lorentzian_sl2c_group_integrals_evaluated=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_GLOBAL_FLAT_COFRAME_CONNECTION_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_GLOBAL_CONNECTION_CONSTRUCTION_FAILED'
        ),
    )
