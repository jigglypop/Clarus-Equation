'''Incidence-oriented spinor-line gluing on the Lorentzian 1-to-5 witness.

The pointwise-labelled face normals used by the full-shape parity no-go omit
the four-simplex incidence orientation.  At a node ``(cell,tetrahedron,face)``
the oriented normal is instead ``epsilon_ct * n_tf``, where ``epsilon_ct`` is
the outward sign of the tetrahedron in the cell.  The exact signs then make
all fifty within-cell full-shape maps proper without changing either labelled
triangle edge.

Forty further links compare the two cells incident on each internal
tetrahedron in its common deterministic rest frame.  The ten internal
triangles form alternating six-cycles whose SO(3) and consistently signed
SU(2) products are identity.  Link phases solve a chosen Hopf-spinor equation
but remain a U(1) convention, not a Regge action phase.
'''

from __future__ import annotations

from collections.abc import Mapping
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
    INTERNAL_TETRAHEDRA,
    INTERNAL_TRIANGLES,
    certify_lorentzian_one_to_five_classical_gluing,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_bra_ket_gluing import (
    j_dual_spinor,
    local_triangle_face_frame,
)
from examples.physics.proper_vertex_one_to_five_coherent_boundary import (
    direction_spinor,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    certify_lorentzian_one_to_five_frame_lifts,
)
from examples.physics.proper_vertex_one_to_five_regge_faces import (
    _exact_face_orientation,
    _local_labelled_edges,
    _tetrahedron,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    su2_lift_of_rotation,
    su2_rotation_matrix,
)


TriangleId = tuple[VertexId, VertexId, VertexId]
NodeId = tuple[SimplexId, TetrahedronId, TriangleId]


@dataclass(frozen=True)
class OrientedSpinorIncidence:
    cell: SimplexId
    tetrahedron: TetrahedronId
    triangle: TriangleId
    opposite_vertex: VertexId
    outward_tetrahedron_sign: int
    exact_unoriented_face_sign: int
    exact_oriented_face_sign: int
    unoriented_face_normal: np.ndarray
    oriented_face_normal: np.ndarray
    canonical_spinor: np.ndarray
    normal_unit_residual: float


@dataclass(frozen=True)
class OrientedSpinorLink:
    source_node: NodeId
    target_node: NodeId
    link_kind: str
    rotation: np.ndarray
    su2_lift: np.ndarray
    phase_correction: complex
    exact_incidence_sign_relation_holds: bool
    rotation_residual: float
    su2_rotation_residual: float
    first_labelled_edge_residual: float
    second_labelled_edge_residual: float
    normal_antipode_residual: float
    spinor_equation_residual: float
    reverse_rotation_residual: float
    reverse_su2_residual: float


@dataclass(frozen=True)
class InternalTriangleSpinorCycle:
    triangle: TriangleId
    ordered_nodes: tuple[NodeId, NodeId, NodeId, NodeId, NodeId, NodeId]
    so3_holonomy: np.ndarray
    su2_holonomy: np.ndarray
    phase_corrected_spinor_holonomy: np.ndarray
    so3_identity_residual: float
    su2_identity_residual: float
    phase_corrected_spinor_identity_residual: float


@dataclass(frozen=True)
class LorentzianOneToFiveIncidenceSpinorCertificate:
    incidence_count: int
    within_cell_link_count: int
    cross_cell_link_count: int
    internal_triangle_cycle_count: int
    boundary_triangle_path_component_count: int
    incidences: tuple[OrientedSpinorIncidence, ...]
    within_cell_links: tuple[OrientedSpinorLink, ...]
    cross_cell_links: tuple[OrientedSpinorLink, ...]
    internal_triangle_cycles: tuple[InternalTriangleSpinorCycle, ...]
    all_incidence_normals_unit: bool
    all_exact_incidence_sign_relations_hold: bool
    all_within_cell_maps_proper_and_full_shape: bool
    all_cross_cell_links_use_common_tetrahedron_rest_gauge: bool
    all_link_spinor_equations_verified: bool
    all_reverse_links_verified: bool
    all_internal_so3_cycles_identity: bool
    all_internal_su2_lift_signs_globally_consistent: bool
    all_internal_phase_corrected_spinor_cycles_identity: bool
    max_link_residual: float
    max_so3_cycle_residual: float
    max_su2_cycle_residual: float
    max_phase_corrected_spinor_cycle_residual: float
    incidence_oriented_spinor_line_section_constructed: bool
    linkwise_u1_phase_convention_constructed: bool
    physical_regge_state_phase_constructed: bool
    regge_action_phase_derived: bool
    global_eprl_boundary_state_constructed: bool
    eprl_y_gamma_map_materialized: bool
    proper_projectors_materialized: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = 'INCIDENCE_ORIENTED_SPINOR_LINE_SECTION_ONLY'


def _node_id(record: OrientedSpinorIncidence) -> NodeId:
    return record.cell, record.tetrahedron, record.triangle


def _phase_corrected_link(
    source: OrientedSpinorIncidence,
    target: OrientedSpinorIncidence,
    rotation: np.ndarray,
    *,
    link_kind: str,
    exact_sign_relation: bool,
    edge_residuals: tuple[float, float],
) -> OrientedSpinorLink:
    lift = su2_lift_of_rotation(rotation)
    transported = lift @ source.canonical_spinor
    target_dual = j_dual_spinor(target.canonical_spinor)
    overlap = complex(np.vdot(target_dual, transported))
    if abs(overlap) <= 1.0e-14:
        raise ValueError('transported incidence spinors must overlap')
    phase = np.conjugate(overlap / abs(overlap))
    reverse_rotation = rotation.T
    reverse_lift = np.conjugate(lift.T)
    return OrientedSpinorLink(
        source_node=_node_id(source),
        target_node=_node_id(target),
        link_kind=link_kind,
        rotation=rotation,
        su2_lift=lift,
        phase_correction=phase,
        exact_incidence_sign_relation_holds=exact_sign_relation,
        rotation_residual=float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3))
            + abs(float(np.linalg.det(rotation)) - 1.0)
        ),
        su2_rotation_residual=float(
            np.linalg.norm(su2_rotation_matrix(lift) - rotation)
        ),
        first_labelled_edge_residual=edge_residuals[0],
        second_labelled_edge_residual=edge_residuals[1],
        normal_antipode_residual=float(
            np.linalg.norm(rotation @ source.oriented_face_normal + target.oriented_face_normal)
        ),
        spinor_equation_residual=float(
            np.linalg.norm(phase * transported - target_dual)
        ),
        reverse_rotation_residual=float(
            np.linalg.norm(reverse_rotation @ rotation - np.eye(3))
        ),
        reverse_su2_residual=float(
            np.linalg.norm(reverse_lift @ lift - np.eye(2))
        ),
    )


def certify_lorentzian_one_to_five_incidence_spinors(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
    tolerance: float = 8.0e-12,
) -> LorentzianOneToFiveIncidenceSpinorCertificate:
    '''Certify all incidence-oriented spinor links and internal six-cycles.'''

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
    outward_signs = {
        (record.cell, record.tetrahedron): record.outward_side_sign
        for record in frame_lifts.incidence_data
    }

    incidence_records: list[OrientedSpinorIncidence] = []
    by_node: dict[NodeId, OrientedSpinorIncidence] = {}
    for cell in FINE_SIMPLICES:
        for omitted_tetrahedron in sorted(cell):
            tetrahedron = _tetrahedron(cell, omitted_tetrahedron)
            epsilon = outward_signs[(cell, tetrahedron)]
            for opposite_vertex in tetrahedron:
                face = local_triangle_face_frame(
                    tetrahedron, opposite_vertex, placement
                )
                _, exact_face_sign = _exact_face_orientation(
                    tetrahedron, opposite_vertex, placement
                )
                oriented_normal = epsilon * face.outward_unit_normal
                record = OrientedSpinorIncidence(
                    cell=cell,
                    tetrahedron=tetrahedron,
                    triangle=face.triangle,
                    opposite_vertex=opposite_vertex,
                    outward_tetrahedron_sign=epsilon,
                    exact_unoriented_face_sign=exact_face_sign,
                    exact_oriented_face_sign=epsilon * exact_face_sign,
                    unoriented_face_normal=face.outward_unit_normal,
                    oriented_face_normal=oriented_normal,
                    canonical_spinor=np.asarray(
                        direction_spinor(oriented_normal), dtype=complex
                    ),
                    normal_unit_residual=abs(
                        float(oriented_normal @ oriented_normal) - 1.0
                    ),
                )
                node = _node_id(record)
                if node in by_node:
                    raise ValueError('incidence node must be unique')
                by_node[node] = record
                incidence_records.append(record)

    within_links: list[OrientedSpinorLink] = []
    for cell in FINE_SIMPLICES:
        for omitted_source, omitted_target in combinations(sorted(cell), 2):
            source_tetrahedron = _tetrahedron(cell, omitted_source)
            target_tetrahedron = _tetrahedron(cell, omitted_target)
            triangle = tuple(
                sorted(set(source_tetrahedron).intersection(target_tetrahedron))
            )
            source_node: NodeId = (
                cell,
                source_tetrahedron,
                triangle,  # type: ignore[arg-type]
            )
            target_node: NodeId = (
                cell,
                target_tetrahedron,
                triangle,  # type: ignore[arg-type]
            )
            source = by_node[source_node]
            target = by_node[target_node]
            source_face = local_triangle_face_frame(
                source_tetrahedron, omitted_target, placement
            )
            target_face = local_triangle_face_frame(
                target_tetrahedron, omitted_source, placement
            )
            source_matrix = np.column_stack(
                (
                    source_face.first_edge_axis,
                    source_face.second_tangent_axis,
                    source.oriented_face_normal,
                )
            )
            target_matrix = np.column_stack(
                (
                    target_face.first_edge_axis,
                    target_face.second_tangent_axis,
                    -target.oriented_face_normal,
                )
            )
            rotation = target_matrix @ source_matrix.T
            source_edges = _local_labelled_edges(
                source_tetrahedron, source.triangle, placement
            )
            target_edges = _local_labelled_edges(
                target_tetrahedron, target.triangle, placement
            )
            within_links.append(
                _phase_corrected_link(
                    source,
                    target,
                    rotation,
                    link_kind='within_cell',
                    exact_sign_relation=(
                        source.exact_oriented_face_sign
                        == -target.exact_oriented_face_sign
                    ),
                    edge_residuals=(
                        float(np.linalg.norm(rotation @ source_edges[0] - target_edges[0])),
                        float(np.linalg.norm(rotation @ source_edges[1] - target_edges[1])),
                    ),
                )
            )

    cross_links: list[OrientedSpinorLink] = []
    for tetrahedron in INTERNAL_TETRAHEDRA:
        incident_cells = tuple(
            cell for cell in FINE_SIMPLICES if set(tetrahedron).issubset(cell)
        )
        if len(incident_cells) != 2:
            raise ValueError('internal tetrahedron must have two incident cells')
        source_cell, target_cell = sorted(incident_cells)
        for opposite_vertex in tetrahedron:
            triangle = tuple(vertex for vertex in tetrahedron if vertex != opposite_vertex)
            source = by_node[(source_cell, tetrahedron, triangle)]  # type: ignore[index]
            target = by_node[(target_cell, tetrahedron, triangle)]  # type: ignore[index]
            cross_links.append(
                _phase_corrected_link(
                    source,
                    target,
                    np.eye(3),
                    link_kind='shared_tetrahedron_rest_gauge',
                    exact_sign_relation=(
                        source.outward_tetrahedron_sign
                        == -target.outward_tetrahedron_sign
                        and source.exact_oriented_face_sign
                        == -target.exact_oriented_face_sign
                    ),
                    edge_residuals=(0.0, 0.0),
                )
            )

    all_links = tuple(within_links + cross_links)
    link_by_nodes = {
        frozenset((link.source_node, link.target_node)): link for link in all_links
    }

    def directed_link(source: NodeId, target: NodeId) -> tuple[np.ndarray, np.ndarray]:
        link = link_by_nodes[frozenset((source, target))]
        if source == link.source_node:
            return link.rotation, link.su2_lift
        return link.rotation.T, np.conjugate(link.su2_lift.T)

    cycles: list[InternalTriangleSpinorCycle] = []
    j_matrix = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=complex)
    for triangle in INTERNAL_TRIANGLES:
        nodes = tuple(node for node in by_node if node[2] == triangle)
        if len(nodes) != 6:
            raise ValueError('internal triangle must have six oriented incidences')
        adjacency: dict[NodeId, list[NodeId]] = {node: [] for node in nodes}
        for link in all_links:
            if link.source_node in adjacency and link.target_node in adjacency:
                adjacency[link.source_node].append(link.target_node)
                adjacency[link.target_node].append(link.source_node)
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise ValueError('internal incidence component must be a six-cycle')
        start = min(nodes)
        ordered: list[NodeId] = [start]
        previous: NodeId | None = None
        current = start
        while True:
            candidates = sorted(
                neighbor for neighbor in adjacency[current] if neighbor != previous
            )
            following = candidates[0]
            if following == start:
                break
            ordered.append(following)
            previous, current = current, following
        if len(ordered) != 6:
            raise ValueError('incidence traversal must visit six distinct nodes')
        so3 = np.eye(3)
        su2 = np.eye(2, dtype=complex)
        phase_corrected = np.eye(2, dtype=complex)
        for source, target in zip(ordered, ordered[1:] + ordered[:1]):
            rotation, lift = directed_link(source, target)
            so3 = rotation @ so3
            su2 = lift @ su2
            source_spinor = by_node[source].canonical_spinor
            target_spinor = by_node[target].canonical_spinor
            overlap = complex(
                np.vdot(j_dual_spinor(target_spinor), lift @ source_spinor)
            )
            if abs(overlap) <= 1.0e-14:
                raise ValueError('directed cycle spinors must overlap')
            phase = np.conjugate(overlap / abs(overlap))
            # -J maps the phase-corrected J-dual target back to the target
            # spinor.  It is anti-linear, K conjugate(z), so six links compose
            # to the linear matrix updated below.
            anti_linear_matrix = (
                -np.conjugate(phase) * j_matrix @ np.conjugate(lift)
            )
            phase_corrected = anti_linear_matrix @ np.conjugate(phase_corrected)
        cycles.append(
            InternalTriangleSpinorCycle(
                triangle=triangle,
                ordered_nodes=tuple(ordered),  # type: ignore[arg-type]
                so3_holonomy=so3,
                su2_holonomy=su2,
                phase_corrected_spinor_holonomy=phase_corrected,
                so3_identity_residual=float(np.linalg.norm(so3 - np.eye(3))),
                su2_identity_residual=float(np.linalg.norm(su2 - np.eye(2))),
                phase_corrected_spinor_identity_residual=float(
                    np.linalg.norm(phase_corrected - np.eye(2))
                ),
            )
        )

    link_residuals = tuple(
        max(
            link.rotation_residual,
            link.su2_rotation_residual,
            link.first_labelled_edge_residual,
            link.second_labelled_edge_residual,
            link.normal_antipode_residual,
            link.spinor_equation_residual,
            link.reverse_rotation_residual,
            link.reverse_su2_residual,
        )
        for link in all_links
    )
    so3_cycle_residual = max(item.so3_identity_residual for item in cycles)
    su2_cycle_residual = max(item.su2_identity_residual for item in cycles)
    phase_cycle_residual = max(
        item.phase_corrected_spinor_identity_residual for item in cycles
    )
    closed = (
        len(incidence_records) == 100
        and len(within_links) == 50
        and len(cross_links) == 40
        and len(cycles) == 10
        and all(record.normal_unit_residual <= tolerance for record in incidence_records)
        and all(link.exact_incidence_sign_relation_holds for link in all_links)
        and max(link_residuals) <= tolerance
        and so3_cycle_residual <= tolerance
        and su2_cycle_residual <= tolerance
        and phase_cycle_residual <= tolerance
    )
    return LorentzianOneToFiveIncidenceSpinorCertificate(
        incidence_count=len(incidence_records),
        within_cell_link_count=len(within_links),
        cross_cell_link_count=len(cross_links),
        internal_triangle_cycle_count=len(cycles),
        boundary_triangle_path_component_count=10,
        incidences=tuple(incidence_records),
        within_cell_links=tuple(within_links),
        cross_cell_links=tuple(cross_links),
        internal_triangle_cycles=tuple(cycles),
        all_incidence_normals_unit=all(
            record.normal_unit_residual <= tolerance for record in incidence_records
        ),
        all_exact_incidence_sign_relations_hold=all(
            link.exact_incidence_sign_relation_holds for link in all_links
        ),
        all_within_cell_maps_proper_and_full_shape=all(
            link.rotation_residual <= tolerance
            and link.first_labelled_edge_residual <= tolerance
            and link.second_labelled_edge_residual <= tolerance
            and link.normal_antipode_residual <= tolerance
            for link in within_links
        ),
        all_cross_cell_links_use_common_tetrahedron_rest_gauge=all(
            link.link_kind == 'shared_tetrahedron_rest_gauge'
            and np.linalg.norm(link.rotation - np.eye(3)) <= tolerance
            for link in cross_links
        ),
        all_link_spinor_equations_verified=all(
            link.spinor_equation_residual <= tolerance for link in all_links
        ),
        all_reverse_links_verified=all(
            max(link.reverse_rotation_residual, link.reverse_su2_residual) <= tolerance
            for link in all_links
        ),
        all_internal_so3_cycles_identity=(so3_cycle_residual <= tolerance),
        all_internal_su2_lift_signs_globally_consistent=(
            su2_cycle_residual <= tolerance
        ),
        all_internal_phase_corrected_spinor_cycles_identity=(
            phase_cycle_residual <= tolerance
        ),
        max_link_residual=max(link_residuals),
        max_so3_cycle_residual=so3_cycle_residual,
        max_su2_cycle_residual=su2_cycle_residual,
        max_phase_corrected_spinor_cycle_residual=phase_cycle_residual,
        incidence_oriented_spinor_line_section_constructed=closed,
        linkwise_u1_phase_convention_constructed=closed,
        physical_regge_state_phase_constructed=False,
        regge_action_phase_derived=False,
        global_eprl_boundary_state_constructed=False,
        eprl_y_gamma_map_materialized=False,
        proper_projectors_materialized=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_INCIDENCE_SPINOR_LINE_SECTION_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_INCIDENCE_SPINOR_CONSTRUCTION_FAILED'
        ),
    )
