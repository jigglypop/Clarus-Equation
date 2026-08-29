'''Exact Lorentzian 1-to-5 boundary skeleton for proper-vertex work.

The module constructs one rational six-vertex Minkowski placement.  Vertices
0,...,4 form a nondegenerate Lorentzian four-simplex with five spacelike
boundary tetrahedra; vertex 5 is its affine barycentre.  Replacing each coarse
vertex in turn by vertex 5 gives five consistently oriented Lorentzian fine
four-simplices.  All ten internal and five boundary tetrahedra are spacelike,
and shared induced edge/triangle geometry is globally identical.

This closes only a classical gluing prerequisite.  No coherent spinors,
half-integer area-spectrum assignment, proper projector, SL(2,C) integral,
internal spin/intertwiner sum, or five-vertex amplitude is constructed.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from examples.physics.proper_vertex_boundary import (
    LorentzianFourSimplexAudit,
    MINKOWSKI_DIAGONAL,
    RationalVector,
    SimplexId,
    SpacelikeTetrahedronAudit,
    TetrahedronId,
    VertexId,
    lorentzian_four_simplex_audit,
    spacelike_tetrahedron_audit,
)


BOUNDARY_VERTICES = (0, 1, 2, 3, 4)
INTERNAL_VERTEX = 5
COARSE_SIMPLEX: SimplexId = BOUNDARY_VERTICES
FINE_SIMPLICES: tuple[SimplexId, ...] = tuple(
    tuple(
        INTERNAL_VERTEX if vertex == omitted else vertex
        for vertex in BOUNDARY_VERTICES
    )
    for omitted in BOUNDARY_VERTICES
)
BOUNDARY_TETRAHEDRA: tuple[TetrahedronId, ...] = tuple(
    combinations(BOUNDARY_VERTICES, 4)
)
INTERNAL_TETRAHEDRA: tuple[TetrahedronId, ...] = tuple(
    (INTERNAL_VERTEX,) + face
    for face in combinations(BOUNDARY_VERTICES, 3)
)
BOUNDARY_TRIANGLES = tuple(combinations(BOUNDARY_VERTICES, 3))
INTERNAL_TRIANGLES = tuple(
    (INTERNAL_VERTEX,) + edge
    for edge in combinations(BOUNDARY_VERTICES, 2)
)
BOUNDARY_EDGES = tuple(combinations(BOUNDARY_VERTICES, 2))
INTERNAL_EDGES = tuple((INTERNAL_VERTEX, vertex) for vertex in BOUNDARY_VERTICES)


def _fraction(value: int, denominator: int = 1) -> Fraction:
    return Fraction(value, denominator)


def lorentzian_one_to_five_coordinates(
    *,
    scale: Fraction = Fraction(1),
) -> dict[VertexId, RationalVector]:
    '''Return an exact proper-compatible Lorentzian barycentric placement.'''

    if not isinstance(scale, Fraction) or scale <= 0:
        raise ValueError('scale must be a positive Fraction')
    raw: dict[VertexId, RationalVector] = {
        0: (_fraction(0), _fraction(0), _fraction(0), _fraction(0)),
        1: (_fraction(1, 1000), _fraction(1), _fraction(0), _fraction(0)),
        2: (_fraction(1, 250), _fraction(0), _fraction(1), _fraction(0)),
        3: (_fraction(9, 1000), _fraction(0), _fraction(0), _fraction(1)),
        4: (_fraction(2, 125), _fraction(-3), _fraction(-2), _fraction(1)),
        5: (_fraction(3, 500), _fraction(-2, 5), _fraction(-1, 5), _fraction(2, 5)),
    }
    return {
        vertex: tuple(scale * component for component in coordinate)  # type: ignore[return-value]
        for vertex, coordinate in raw.items()
    }


def _subtract(left: RationalVector, right: RationalVector) -> RationalVector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _minkowski_product(left: RationalVector, right: RationalVector) -> Fraction:
    return sum(
        metric * a * b
        for metric, a, b in zip(MINKOWSKI_DIAGONAL, left, right)
    )


def _determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError('matrix must be nonempty and square')
    work = [list(row) for row in matrix]
    sign = Fraction(1)
    determinant = Fraction(1)
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, size):
            factor = work[row][column] / pivot_value
            for item in range(column + 1, size):
                work[row][item] -= factor * work[column][item]
    return sign * determinant


def _coordinate_determinant(
    simplex: SimplexId,
    coordinates: Mapping[VertexId, RationalVector],
) -> Fraction:
    base = coordinates[simplex[0]]
    edges = tuple(_subtract(coordinates[vertex], base) for vertex in simplex[1:])
    component_matrix = tuple(
        tuple(edge[component] for edge in edges)
        for component in range(4)
    )
    return _determinant(component_matrix)


def _squared_interval(
    first: VertexId,
    second: VertexId,
    coordinates: Mapping[VertexId, RationalVector],
) -> Fraction:
    displacement = _subtract(coordinates[second], coordinates[first])
    return _minkowski_product(displacement, displacement)


def triangle_area_squared(
    triangle: tuple[VertexId, VertexId, VertexId],
    coordinates: Mapping[VertexId, RationalVector],
) -> Fraction:
    '''Return the exact induced spacelike triangle area squared.'''

    first, second, third = triangle
    edge_a = _subtract(coordinates[second], coordinates[first])
    edge_b = _subtract(coordinates[third], coordinates[first])
    determinant = (
        _minkowski_product(edge_a, edge_a)
        * _minkowski_product(edge_b, edge_b)
        - _minkowski_product(edge_a, edge_b) ** 2
    )
    return determinant / 4


def _tetrahedron_edge_data(
    tetrahedron: TetrahedronId,
    coordinates: Mapping[VertexId, RationalVector],
) -> tuple[tuple[tuple[VertexId, VertexId], Fraction], ...]:
    return tuple(
        (tuple(sorted(edge)), _squared_interval(edge[0], edge[1], coordinates))
        for edge in combinations(tetrahedron, 2)
    )


@dataclass(frozen=True)
class LorentzianOneToFiveProperBoundaryCertificate:
    vertex_count: int
    boundary_edge_count: int
    internal_edge_count: int
    boundary_triangle_count: int
    internal_triangle_count: int
    boundary_tetrahedron_count: int
    internal_tetrahedron_count: int
    fine_four_simplex_count: int
    coarse_simplex_audit: LorentzianFourSimplexAudit
    fine_simplex_audits: tuple[LorentzianFourSimplexAudit, ...]
    boundary_tetrahedron_audits: tuple[SpacelikeTetrahedronAudit, ...]
    internal_tetrahedron_audits: tuple[SpacelikeTetrahedronAudit, ...]
    coarse_coordinate_determinant: Fraction
    fine_to_coarse_coordinate_determinant_ratios: tuple[Fraction, ...]
    fine_to_coarse_gram_determinant_ratios: tuple[Fraction, ...]
    internal_tetrahedron_incidence_counts: tuple[int, ...]
    internal_triangle_incidence_counts: tuple[int, ...]
    boundary_triangle_incidence_counts: tuple[int, ...]
    inserted_vertex_is_exact_barycentre: bool
    all_five_cells_nondegenerate_lorentzian: bool
    all_fifteen_unique_tetrahedra_spacelike: bool
    all_fine_cells_share_coarse_orientation: bool
    all_fine_four_volumes_are_one_fifth_of_coarse: bool
    internal_tetrahedra_have_two_incident_cells: bool
    internal_triangles_have_three_incident_cells: bool
    boundary_triangles_have_two_incident_cells: bool
    shared_tetrahedron_intrinsic_shape_matching: bool
    global_triangle_area_squared_labels_consistent: bool
    all_triangle_area_squared_positive: bool
    classical_proper_boundary_geometry_prerequisite_closed: bool
    regge_coherent_spinors_materialized: bool
    half_integer_spin_assignment_constructed: bool
    shared_bra_ket_orientation_data_constructed: bool
    proper_projectors_materialized: bool
    proper_single_vertex_integrals_evaluated: bool
    internal_spins_summed: bool
    internal_intertwiners_integrated: bool
    standard_proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'LORENTZIAN_1_TO_5_CLASSICAL_BOUNDARY_SKELETON_NOT_PROPER_EPRL_AMPLITUDE'
    )


def certify_lorentzian_one_to_five_proper_boundary(
    coordinates: Mapping[VertexId, RationalVector] | None = None,
    *,
    scale: Fraction = Fraction(1),
) -> LorentzianOneToFiveProperBoundaryCertificate:
    '''Certify the exact classical 1-to-5 Lorentzian gluing prerequisite.'''

    if coordinates is not None and scale != 1:
        raise ValueError('scale cannot be combined with explicit coordinates')
    placement = (
        lorentzian_one_to_five_coordinates(scale=scale)
        if coordinates is None
        else dict(coordinates)
    )
    if set(placement) != set(range(6)):
        raise ValueError('coordinates must contain exactly vertex labels 0 through 5')
    if any(
        len(coordinate) != 4
        or any(not isinstance(component, Fraction) for component in coordinate)
        for coordinate in placement.values()
    ):
        raise ValueError('every coordinate must contain four Fraction components')

    coarse_audit = lorentzian_four_simplex_audit(COARSE_SIMPLEX, placement)
    fine_audits = tuple(
        lorentzian_four_simplex_audit(simplex, placement)
        for simplex in FINE_SIMPLICES
    )
    boundary_tetrahedron_audits = tuple(
        spacelike_tetrahedron_audit(tetrahedron, placement)
        for tetrahedron in BOUNDARY_TETRAHEDRA
    )
    internal_tetrahedron_audits = tuple(
        spacelike_tetrahedron_audit(tetrahedron, placement)
        for tetrahedron in INTERNAL_TETRAHEDRA
    )

    coarse_determinant = _coordinate_determinant(COARSE_SIMPLEX, placement)
    fine_coordinate_determinants = tuple(
        _coordinate_determinant(simplex, placement)
        for simplex in FINE_SIMPLICES
    )
    coordinate_ratios = tuple(
        determinant / coarse_determinant
        if coarse_determinant != 0 else Fraction(0)
        for determinant in fine_coordinate_determinants
    )
    gram_ratios = tuple(
        audit.gram_determinant / coarse_audit.gram_determinant
        if coarse_audit.gram_determinant != 0 else Fraction(0)
        for audit in fine_audits
    )
    internal_tetrahedron_incidence = tuple(
        sum(set(tetrahedron).issubset(simplex) for simplex in FINE_SIMPLICES)
        for tetrahedron in INTERNAL_TETRAHEDRA
    )
    internal_triangle_incidence = tuple(
        sum(set(triangle).issubset(simplex) for simplex in FINE_SIMPLICES)
        for triangle in INTERNAL_TRIANGLES
    )
    boundary_triangle_incidence = tuple(
        sum(set(triangle).issubset(simplex) for simplex in FINE_SIMPLICES)
        for triangle in BOUNDARY_TRIANGLES
    )
    expected_barycentre = tuple(
        sum(placement[vertex][component] for vertex in BOUNDARY_VERTICES) / 5
        for component in range(4)
    )
    barycentric = placement[INTERNAL_VERTEX] == expected_barycentre

    shared_shape_matching = all(
        len(
            {
                _tetrahedron_edge_data(tetrahedron, placement)
                for simplex in FINE_SIMPLICES
                if set(tetrahedron).issubset(simplex)
            }
        )
        == 1
        for tetrahedron in INTERNAL_TETRAHEDRA
    )
    triangles = BOUNDARY_TRIANGLES + INTERNAL_TRIANGLES
    area_labels = {
        tuple(sorted(triangle)): triangle_area_squared(triangle, placement)
        for triangle in triangles
    }
    triangle_labels_consistent = all(
        len(
            {
                area_labels[tuple(sorted(triangle))]
                for simplex in FINE_SIMPLICES
                if set(triangle).issubset(simplex)
            }
        )
        == 1
        for triangle in triangles
    )
    all_triangle_areas_positive = all(value > 0 for value in area_labels.values())
    all_fine_lorentzian = all(
        audit.nondegenerate_lorentzian for audit in fine_audits
    )
    all_unique_tetrahedra_spacelike = (
        all(audit.nondegenerate_spacelike for audit in boundary_tetrahedron_audits)
        and all(audit.nondegenerate_spacelike for audit in internal_tetrahedron_audits)
    )
    same_orientation = all(ratio > 0 for ratio in coordinate_ratios)
    one_fifth = all(ratio == Fraction(1, 5) for ratio in coordinate_ratios)
    gram_one_twenty_fifth = all(
        ratio == Fraction(1, 25) for ratio in gram_ratios
    )
    closed = (
        barycentric
        and coarse_audit.nondegenerate_lorentzian
        and coarse_audit.all_boundary_tetrahedra_spacelike
        and all_fine_lorentzian
        and all_unique_tetrahedra_spacelike
        and same_orientation
        and one_fifth
        and gram_one_twenty_fifth
        and all(count == 2 for count in internal_tetrahedron_incidence)
        and all(count == 3 for count in internal_triangle_incidence)
        and all(count == 2 for count in boundary_triangle_incidence)
        and shared_shape_matching
        and triangle_labels_consistent
        and all_triangle_areas_positive
    )

    return LorentzianOneToFiveProperBoundaryCertificate(
        vertex_count=6,
        boundary_edge_count=len(BOUNDARY_EDGES),
        internal_edge_count=len(INTERNAL_EDGES),
        boundary_triangle_count=len(BOUNDARY_TRIANGLES),
        internal_triangle_count=len(INTERNAL_TRIANGLES),
        boundary_tetrahedron_count=len(BOUNDARY_TETRAHEDRA),
        internal_tetrahedron_count=len(INTERNAL_TETRAHEDRA),
        fine_four_simplex_count=len(FINE_SIMPLICES),
        coarse_simplex_audit=coarse_audit,
        fine_simplex_audits=fine_audits,
        boundary_tetrahedron_audits=boundary_tetrahedron_audits,
        internal_tetrahedron_audits=internal_tetrahedron_audits,
        coarse_coordinate_determinant=coarse_determinant,
        fine_to_coarse_coordinate_determinant_ratios=coordinate_ratios,
        fine_to_coarse_gram_determinant_ratios=gram_ratios,
        internal_tetrahedron_incidence_counts=internal_tetrahedron_incidence,
        internal_triangle_incidence_counts=internal_triangle_incidence,
        boundary_triangle_incidence_counts=boundary_triangle_incidence,
        inserted_vertex_is_exact_barycentre=barycentric,
        all_five_cells_nondegenerate_lorentzian=all_fine_lorentzian,
        all_fifteen_unique_tetrahedra_spacelike=all_unique_tetrahedra_spacelike,
        all_fine_cells_share_coarse_orientation=same_orientation,
        all_fine_four_volumes_are_one_fifth_of_coarse=(
            one_fifth and gram_one_twenty_fifth
        ),
        internal_tetrahedra_have_two_incident_cells=all(
            count == 2 for count in internal_tetrahedron_incidence
        ),
        internal_triangles_have_three_incident_cells=all(
            count == 3 for count in internal_triangle_incidence
        ),
        boundary_triangles_have_two_incident_cells=all(
            count == 2 for count in boundary_triangle_incidence
        ),
        shared_tetrahedron_intrinsic_shape_matching=shared_shape_matching,
        global_triangle_area_squared_labels_consistent=triangle_labels_consistent,
        all_triangle_area_squared_positive=all_triangle_areas_positive,
        classical_proper_boundary_geometry_prerequisite_closed=closed,
        regge_coherent_spinors_materialized=False,
        half_integer_spin_assignment_constructed=False,
        shared_bra_ket_orientation_data_constructed=False,
        proper_projectors_materialized=False,
        proper_single_vertex_integrals_evaluated=False,
        internal_spins_summed=False,
        internal_intertwiners_integrated=False,
        standard_proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_PROPER_COMPATIBLE_1_TO_5_CLASSICAL_BOUNDARY_SKELETON_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_BOUNDARY_SKELETON_FAILED'
        ),
    )
