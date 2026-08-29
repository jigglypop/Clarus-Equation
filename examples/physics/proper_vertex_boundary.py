"""Exact spacelike boundary geometry for the Lorentzian proper vertex.

The standard Lorentzian EPRL/proper-vertex asymptotic theorem is a theorem
about one *flat* Lorentzian 4-simplex whose five boundary tetrahedra are
nondegenerate and spacelike.  The earlier CE coordinates do not satisfy that
boundary hypothesis.  This module supplies a rational six-vertex placement
for the same three-cell incidence complex and checks the relevant signatures
with exact rational arithmetic.

The first certificate closes a classical boundary-geometry prerequisite.  A
later certificate also supplies an exact spin-quantized one-vertex family and
defines a three-cell *conditioned rank-one contraction*.  It still does not
materialize the coherent spinors, numerically evaluate a proper-vertex
integral, construct the standard internal state sum, or turn the flat Regge
cells into the de Sitter curved-cell witness of
``curved_plebanski_hinge.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np


VertexId = int
RationalVector = tuple[Fraction, Fraction, Fraction, Fraction]
TetrahedronId = tuple[VertexId, VertexId, VertexId, VertexId]
SimplexId = tuple[VertexId, VertexId, VertexId, VertexId, VertexId]

MINKOWSKI_DIAGONAL = (
    Fraction(-1),
    Fraction(1),
    Fraction(1),
    Fraction(1),
)
THREE_CELLS: tuple[SimplexId, ...] = (
    (0, 1, 2, 3, 4),
    (1, 2, 3, 4, 5),
    (0, 1, 2, 3, 5),
)
SHARED_TETRAHEDRA: tuple[TetrahedronId, ...] = (
    (1, 2, 3, 4),
    (1, 2, 3, 5),
    (0, 1, 2, 3),
)


def _fraction(value: int, denominator: int = 1) -> Fraction:
    return Fraction(value, denominator)


def proper_compatible_vertex_coordinates(
    *,
    scale: Fraction = Fraction(1),
) -> dict[VertexId, RationalVector]:
    """Return one rational placement satisfying all spacelike-face tests."""

    if not isinstance(scale, Fraction) or scale <= 0:
        raise ValueError("scale must be a positive Fraction")
    raw: dict[VertexId, RationalVector] = {
        0: (_fraction(0), _fraction(0), _fraction(0), _fraction(0)),
        1: (_fraction(1, 100), _fraction(1), _fraction(0), _fraction(0)),
        2: (_fraction(1, 25), _fraction(0), _fraction(1), _fraction(0)),
        3: (_fraction(9, 100), _fraction(0), _fraction(0), _fraction(1)),
        4: (_fraction(4, 25), _fraction(1), _fraction(1), _fraction(1)),
        5: (_fraction(1, 4), _fraction(4), _fraction(-3), _fraction(-3)),
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


def _gram_matrix(edges: Sequence[RationalVector]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(_minkowski_product(left, right) for right in edges)
        for left in edges
    )


def _determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be nonempty and square")
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


def _leading_principal_minors(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[Fraction, ...]:
    return tuple(
        _determinant(tuple(tuple(row[:size]) for row in matrix[:size]))
        for size in range(1, len(matrix) + 1)
    )


def _float_eigenvalues(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[float, ...]:
    values = np.linalg.eigvalsh(
        np.asarray([[float(item) for item in row] for row in matrix])
    )
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class SpacelikeTetrahedronAudit:
    vertices: TetrahedronId
    leading_principal_minors: tuple[Fraction, Fraction, Fraction]
    gram_determinant: Fraction
    minimum_gram_eigenvalue: float
    nondegenerate_spacelike: bool


def spacelike_tetrahedron_audit(
    vertices: TetrahedronId,
    coordinates: Mapping[VertexId, RationalVector],
) -> SpacelikeTetrahedronAudit:
    """Apply Sylvester's criterion to one induced tetrahedron metric."""

    if len(vertices) != 4 or len(set(vertices)) != 4:
        raise ValueError("vertices must contain four distinct labels")
    if any(vertex not in coordinates for vertex in vertices):
        raise ValueError("every tetrahedron vertex must have a coordinate")
    base = coordinates[vertices[0]]
    edges = tuple(_subtract(coordinates[vertex], base) for vertex in vertices[1:])
    gram = _gram_matrix(edges)
    minors = _leading_principal_minors(gram)
    eigenvalues = _float_eigenvalues(gram)
    spacelike = all(value > 0 for value in minors)
    return SpacelikeTetrahedronAudit(
        vertices=tuple(vertices),
        leading_principal_minors=minors,  # type: ignore[arg-type]
        gram_determinant=minors[-1],
        minimum_gram_eigenvalue=min(eigenvalues),
        nondegenerate_spacelike=spacelike,
    )


@dataclass(frozen=True)
class LorentzianFourSimplexAudit:
    vertices: SimplexId
    boundary_tetrahedra: tuple[SpacelikeTetrahedronAudit, ...]
    gram_determinant: Fraction
    gram_eigenvalues: tuple[float, float, float, float]
    negative_eigenvalue_count: int
    positive_eigenvalue_count: int
    all_boundary_tetrahedra_spacelike: bool
    nondegenerate_lorentzian: bool


def lorentzian_four_simplex_audit(
    vertices: SimplexId,
    coordinates: Mapping[VertexId, RationalVector],
    *,
    tolerance: float = 1.0e-12,
) -> LorentzianFourSimplexAudit:
    """Check one 4-simplex and each of its five boundary tetrahedra."""

    if len(vertices) != 5 or len(set(vertices)) != 5:
        raise ValueError("vertices must contain five distinct labels")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if any(vertex not in coordinates for vertex in vertices):
        raise ValueError("every simplex vertex must have a coordinate")
    base = coordinates[vertices[0]]
    edges = tuple(_subtract(coordinates[vertex], base) for vertex in vertices[1:])
    gram = _gram_matrix(edges)
    determinant = _determinant(gram)
    eigenvalues = _float_eigenvalues(gram)
    negative = sum(value < -tolerance for value in eigenvalues)
    positive = sum(value > tolerance for value in eigenvalues)
    boundary = tuple(
        spacelike_tetrahedron_audit(tuple(face), coordinates)
        for face in combinations(vertices, 4)
    )
    all_spacelike = all(face.nondegenerate_spacelike for face in boundary)
    lorentzian = determinant < 0 and negative == 1 and positive == 3
    return LorentzianFourSimplexAudit(
        vertices=tuple(vertices),
        boundary_tetrahedra=boundary,
        gram_determinant=determinant,
        gram_eigenvalues=eigenvalues,  # type: ignore[arg-type]
        negative_eigenvalue_count=negative,
        positive_eigenvalue_count=positive,
        all_boundary_tetrahedra_spacelike=all_spacelike,
        nondegenerate_lorentzian=lorentzian,
    )


@dataclass(frozen=True)
class ProperVertexBoundaryCertificate:
    cells: tuple[SimplexId, ...]
    shared_tetrahedra: tuple[TetrahedronId, ...]
    simplex_audits: tuple[LorentzianFourSimplexAudit, ...]
    unique_tetrahedron_count: int
    shared_tetrahedra_have_two_incident_cells_in_global_coordinates: bool
    all_twelve_tetrahedra_spacelike: bool
    all_three_cells_lorentzian: bool
    classical_boundary_geometry_ready_for_standard_proper_vertex: bool
    quantized_regge_coherent_states_constructed: bool
    three_vertex_glued_amplitude_derived: bool
    curved_de_sitter_proper_amplitude_derived: bool
    status: str
    claim_ceiling: str = (
        "FLAT_SINGLE_VERTEX_PROPER_BOUNDARY_GEOMETRY_NOT_CURVED_OR_GLUED_AMPLITUDE"
    )


def constructive_proper_vertex_boundary_certificate(
    *,
    scale: Fraction = Fraction(1),
) -> ProperVertexBoundaryCertificate:
    """Construct the exact classical boundary prerequisite on all three cells."""

    coordinates = proper_compatible_vertex_coordinates(scale=scale)
    audits = tuple(
        lorentzian_four_simplex_audit(cell, coordinates) for cell in THREE_CELLS
    )
    unique_faces = {
        tuple(face)
        for cell in THREE_CELLS
        for face in combinations(cell, 4)
    }
    shared_identical = all(
        sum(set(face).issubset(cell) for cell in THREE_CELLS) == 2
        for face in SHARED_TETRAHEDRA
    )
    all_spacelike = all(
        face.nondegenerate_spacelike
        for audit in audits
        for face in audit.boundary_tetrahedra
    )
    all_lorentzian = all(audit.nondegenerate_lorentzian for audit in audits)
    ready = (
        len(unique_faces) == 12
        and shared_identical
        and all_spacelike
        and all_lorentzian
    )
    return ProperVertexBoundaryCertificate(
        cells=THREE_CELLS,
        shared_tetrahedra=SHARED_TETRAHEDRA,
        simplex_audits=audits,
        unique_tetrahedron_count=len(unique_faces),
        shared_tetrahedra_have_two_incident_cells_in_global_coordinates=(
            shared_identical
        ),
        all_twelve_tetrahedra_spacelike=all_spacelike,
        all_three_cells_lorentzian=all_lorentzian,
        classical_boundary_geometry_ready_for_standard_proper_vertex=ready,
        quantized_regge_coherent_states_constructed=False,
        three_vertex_glued_amplitude_derived=False,
        curved_de_sitter_proper_amplitude_derived=False,
        status=(
            "STANDARD_PROPER_VERTEX_CLASSICAL_BOUNDARY_GEOMETRY_CLOSED"
            if ready
            else "PROPER_VERTEX_CLASSICAL_BOUNDARY_GEOMETRY_FAILED"
        ),
    )


def _isosceles_squared_interval(left: VertexId, right: VertexId) -> Fraction:
    """Squared interval in units where the regular base edge has length one."""

    if left == right:
        return Fraction(0)
    if not 0 <= left <= 4 or not 0 <= right <= 4:
        raise ValueError("isosceles proper-vertex labels must lie in {0,1,2,3,4}")
    return Fraction(91, 256) if 4 in (left, right) else Fraction(1)


def _gram_from_squared_intervals(
    vertices: Sequence[VertexId],
) -> tuple[tuple[Fraction, ...], ...]:
    base = vertices[0]
    others = vertices[1:]
    return tuple(
        tuple(
            (
                _isosceles_squared_interval(base, left)
                + _isosceles_squared_interval(base, right)
                - _isosceles_squared_interval(left, right)
            )
            / 2
            for right in others
        )
        for left in others
    )


def four_spin_invariant_exists(spins: Sequence[Fraction]) -> bool:
    """Return the polygon/parity criterion for an SU(2) four-valent invariant."""

    if len(spins) != 4 or any(
        spin < 0 or spin.denominator not in (1, 2) for spin in spins
    ):
        return False
    total = sum(spins, Fraction(0))
    return total.denominator == 1 and 2 * max(spins) <= total


@dataclass(frozen=True)
class SpinQuantizedProperVertexCertificate:
    spin_multiplier: int
    time_height_squared_over_base_edge_squared: Fraction
    apex_edge_squared_over_base_edge_squared: Fraction
    base_triangle_area_squared_over_base_edge_fourth: Fraction
    apex_triangle_area_squared_over_base_edge_fourth: Fraction
    base_triangle_spin: Fraction
    apex_triangle_spin: Fraction
    base_tetrahedron_leading_minors: tuple[Fraction, Fraction, Fraction]
    side_tetrahedron_leading_minors: tuple[Fraction, Fraction, Fraction]
    normalized_four_simplex_gram_determinant: Fraction
    base_tetrahedron_intertwiner_admissible: bool
    side_tetrahedron_intertwiner_admissible: bool
    all_five_boundary_tetrahedra_spacelike: bool
    nondegenerate_lorentzian_four_simplex: bool
    regge_coherent_boundary_state_exists: bool
    base_triangle_boost_angle: float
    apex_triangle_boost_angle: float
    dimensionless_regge_phase_coefficient: float
    published_proper_vertex_definition_applicable_to_scaling_family: bool
    published_single_term_asymptotic_theorem_applies: bool
    large_spin_limit_is_multiplier_to_infinity: bool
    explicit_boundary_spinors_and_regge_phases_materialized: bool
    proper_vertex_amplitude_numerically_evaluated: bool
    hessian_prefactor_evaluated: bool
    large_spin_power: int
    three_vertex_glued_amplitude_derived: bool
    curved_de_sitter_proper_amplitude_derived: bool
    amplitude_definition_source: str
    asymptotic_theorem_source: str
    status: str
    claim_ceiling: str = (
        "ONE_FLAT_SPIN_QUANTIZED_LORENTZIAN_PROPER_VERTEX_NOT_CURVED_OR_GLUED"
    )


def spin_quantized_proper_vertex_certificate(
    *,
    spin_multiplier: int = 1,
    barbero_immirzi_parameter: float = 1.0,
) -> SpinQuantizedProperVertexCertificate:
    """Instantiate one exact Regge-like boundary state for the proper vertex.

    Four vertices form a regular Euclidean tetrahedron in ``t=0`` and the
    fifth lies above its centre with ``T^2/L^2=5/256``.  In normalized units
    its squared apex-to-base edge is ``91/256``.  Scaling the physical base
    edge so that ``L^2=32 m/sqrt(3)`` gives four base-triangle spins ``8m``
    and six apex-triangle spins ``3m``.  These are exact SU(2) spins and each
    of the five tetrahedra admits an invariant coherent intertwiner.

    The final asymptotic flag invokes the published one-vertex theorem of
    Engle--Vilensky--Zipfel; the code verifies its geometric/admissibility
    hypotheses but does not re-prove that external stationary-phase theorem.
    """

    if not isinstance(spin_multiplier, int) or spin_multiplier <= 0:
        raise ValueError("spin_multiplier must be a positive integer")
    gamma = float(barbero_immirzi_parameter)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("barbero_immirzi_parameter must be finite and positive")

    base_tetrahedron_gram = _gram_from_squared_intervals((0, 1, 2, 3))
    side_tetrahedron_gram = _gram_from_squared_intervals((0, 1, 2, 4))
    simplex_gram = _gram_from_squared_intervals((0, 1, 2, 3, 4))
    base_minors = _leading_principal_minors(base_tetrahedron_gram)
    side_minors = _leading_principal_minors(side_tetrahedron_gram)
    simplex_determinant = _determinant(simplex_gram)
    simplex_eigenvalues = _float_eigenvalues(simplex_gram)

    base_spin = Fraction(8 * spin_multiplier)
    apex_spin = Fraction(3 * spin_multiplier)
    base_admissible = four_spin_invariant_exists((base_spin,) * 4)
    side_admissible = four_spin_invariant_exists(
        (base_spin, apex_spin, apex_spin, apex_spin)
    )
    all_spacelike = all(value > 0 for value in base_minors + side_minors)
    lorentzian = (
        simplex_determinant < 0
        and sum(value < -1.0e-12 for value in simplex_eigenvalues) == 1
        and sum(value > 1.0e-12 for value in simplex_eigenvalues) == 3
    )
    regge_state = all_spacelike and lorentzian and base_admissible and side_admissible

    base_angle = math.acosh(4.0 * math.sqrt(2.0 / 17.0))
    apex_angle = math.acosh(37.0 / 17.0)
    phase = gamma * spin_multiplier * (
        32.0 * base_angle + 18.0 * apex_angle
    )
    theorem_applies = regge_state
    return SpinQuantizedProperVertexCertificate(
        spin_multiplier=spin_multiplier,
        time_height_squared_over_base_edge_squared=Fraction(5, 256),
        apex_edge_squared_over_base_edge_squared=Fraction(91, 256),
        base_triangle_area_squared_over_base_edge_fourth=Fraction(3, 16),
        apex_triangle_area_squared_over_base_edge_fourth=Fraction(27, 1024),
        base_triangle_spin=base_spin,
        apex_triangle_spin=apex_spin,
        base_tetrahedron_leading_minors=base_minors,  # type: ignore[arg-type]
        side_tetrahedron_leading_minors=side_minors,  # type: ignore[arg-type]
        normalized_four_simplex_gram_determinant=simplex_determinant,
        base_tetrahedron_intertwiner_admissible=base_admissible,
        side_tetrahedron_intertwiner_admissible=side_admissible,
        all_five_boundary_tetrahedra_spacelike=all_spacelike,
        nondegenerate_lorentzian_four_simplex=lorentzian,
        regge_coherent_boundary_state_exists=regge_state,
        base_triangle_boost_angle=base_angle,
        apex_triangle_boost_angle=apex_angle,
        dimensionless_regge_phase_coefficient=phase,
        published_proper_vertex_definition_applicable_to_scaling_family=(
            regge_state
        ),
        published_single_term_asymptotic_theorem_applies=theorem_applies,
        large_spin_limit_is_multiplier_to_infinity=True,
        explicit_boundary_spinors_and_regge_phases_materialized=False,
        proper_vertex_amplitude_numerically_evaluated=False,
        hessian_prefactor_evaluated=False,
        large_spin_power=-12,
        three_vertex_glued_amplitude_derived=False,
        curved_de_sitter_proper_amplitude_derived=False,
        amplitude_definition_source="Engle--Zipfel arXiv:1502.04640 Eq. (53)",
        asymptotic_theorem_source=(
            "Engle--Vilensky--Zipfel arXiv:1505.06683 Theorem 3"
        ),
        status=(
            "ONE_SPIN_QUANTIZED_PROPER_VERTEX_SCALING_FAMILY_CLOSED"
            if theorem_applies
            else "SPIN_QUANTIZED_PROPER_VERTEX_HYPOTHESES_FAILED"
        ),
    )


@dataclass(frozen=True)
class ThreeCellRankOneProperAmplitudeCertificate:
    cells: tuple[SimplexId, ...]
    cell_apices: tuple[VertexId, VertexId, VertexId]
    shared_tetrahedra: tuple[TetrahedronId, ...]
    unique_triangle_count: int
    triangle_spin_assignments: tuple[
        tuple[tuple[VertexId, VertexId, VertexId], tuple[Fraction, ...]], ...
    ]
    global_edge_shape_assignment_consistent: bool
    global_triangle_spin_assignment_consistent: bool
    shared_side_tetrahedra_shape_matched: bool
    every_cell_in_single_vertex_proper_scaling_family: bool
    normalized_rank_one_internal_intertwiner_projectors_declared: bool
    internal_projector_definition: str
    shared_bra_ket_dualization_declared: bool
    compatible_local_time_orientations_declared: bool
    compatible_positive_regge_phase_branches_declared: bool
    projector_coherent_spinors_materialized: bool
    fixed_internal_face_amplitudes: bool
    product_haar_measure_and_per_vertex_gauge_fixing_declared: bool
    independent_vertex_gauge_fixings_declared: bool
    conditioned_rank_one_contraction_asymptotic_derived_under_declared_model: bool
    conditioned_large_spin_power: int
    conditioned_regge_phase_coefficient: float
    contraction_numerically_evaluated: bool
    internal_spins_summed: bool
    internal_intertwiners_integrated: bool
    standard_eprl_multi_vertex_state_sum_derived: bool
    curved_de_sitter_proper_amplitude_derived: bool
    status: str
    claim_ceiling: str = (
        "THREE_FLAT_PROPER_VERTICES_WITH_FIXED_RANK_ONE_INTERNAL_DATA_NOT_STANDARD_STATE_SUM"
    )


def three_cell_rank_one_proper_amplitude_certificate(
    *,
    spin_multiplier: int = 1,
    barbero_immirzi_parameter: float = 1.0,
    cell_apices: tuple[VertexId, VertexId, VertexId] = (1, 1, 1),
) -> ThreeCellRankOneProperAmplitudeCertificate:
    """Glue three proper vertices only through declared rank-one edge data.

    With common apex label 1, every abstract cell is the same isosceles
    Lorentzian simplex.  Edges incident to 1 have the common apex-edge length;
    all other edges have the common base length.  Hence the three shared
    tetrahedra are congruent side tetrahedra carrying spins ``(8m,3m,3m,3m)``.

    The amplitude certified here inserts one normalized rank-one projector on
    each shared tetrahedron, fixes all internal spins, and sets the internal
    face amplitudes to one.  Under those explicit model choices it factorizes
    into three published proper vertices, so their one-vertex asymptotics
    multiply to ``m^-36 exp(i m S)``.  It is deliberately not the standard
    EPRL state sum, which would sum spins and contract/integrate a complete
    internal intertwiner basis.
    """

    if len(cell_apices) != len(THREE_CELLS):
        raise ValueError("cell_apices must provide one apex for each cell")
    if any(apex not in cell for apex, cell in zip(cell_apices, THREE_CELLS)):
        raise ValueError("every apex must belong to its corresponding cell")
    single = spin_quantized_proper_vertex_certificate(
        spin_multiplier=spin_multiplier,
        barbero_immirzi_parameter=barbero_immirzi_parameter,
    )

    edge_classes: dict[tuple[VertexId, VertexId], set[str]] = {}
    triangle_spins: dict[
        tuple[VertexId, VertexId, VertexId], set[Fraction]
    ] = {}
    for cell, apex in zip(THREE_CELLS, cell_apices):
        for edge in combinations(cell, 2):
            edge_classes.setdefault(tuple(edge), set()).add(
                "APEX_EDGE" if apex in edge else "BASE_EDGE"
            )
        for triangle in combinations(cell, 3):
            triangle_spins.setdefault(tuple(triangle), set()).add(
                single.apex_triangle_spin
                if apex in triangle
                else single.base_triangle_spin
            )
    edge_consistent = all(len(values) == 1 for values in edge_classes.values())
    spin_consistent = all(len(values) == 1 for values in triangle_spins.values())
    shared_side_matched = (
        edge_consistent
        and spin_consistent
        and all(
            all(
                cell_apices[index] in tetrahedron
                for index, cell in enumerate(THREE_CELLS)
                if set(tetrahedron).issubset(cell)
            )
            and four_spin_invariant_exists(
                (
                    single.base_triangle_spin,
                    single.apex_triangle_spin,
                    single.apex_triangle_spin,
                    single.apex_triangle_spin,
                )
            )
            for tetrahedron in SHARED_TETRAHEDRA
        )
    )
    every_cell = single.published_single_term_asymptotic_theorem_applies
    rank_one_model = edge_consistent and spin_consistent and shared_side_matched and every_cell
    return ThreeCellRankOneProperAmplitudeCertificate(
        cells=THREE_CELLS,
        cell_apices=cell_apices,
        shared_tetrahedra=SHARED_TETRAHEDRA,
        unique_triangle_count=len(triangle_spins),
        triangle_spin_assignments=tuple(
            (triangle, tuple(sorted(values)))
            for triangle, values in sorted(triangle_spins.items())
        ),
        global_edge_shape_assignment_consistent=edge_consistent,
        global_triangle_spin_assignment_consistent=spin_consistent,
        shared_side_tetrahedra_shape_matched=shared_side_matched,
        every_cell_in_single_vertex_proper_scaling_family=every_cell,
        normalized_rank_one_internal_intertwiner_projectors_declared=(
            rank_one_model
        ),
        internal_projector_definition=(
            "P_e=|iota_e^Regge/LS><iota_e^Regge/LS| with <iota_e|iota_e>=1"
        ),
        shared_bra_ket_dualization_declared=rank_one_model,
        compatible_local_time_orientations_declared=rank_one_model,
        compatible_positive_regge_phase_branches_declared=rank_one_model,
        projector_coherent_spinors_materialized=False,
        fixed_internal_face_amplitudes=rank_one_model,
        product_haar_measure_and_per_vertex_gauge_fixing_declared=(
            rank_one_model
        ),
        independent_vertex_gauge_fixings_declared=rank_one_model,
        conditioned_rank_one_contraction_asymptotic_derived_under_declared_model=(
            rank_one_model
        ),
        conditioned_large_spin_power=-36 if rank_one_model else 0,
        conditioned_regge_phase_coefficient=(
            3.0 * single.dimensionless_regge_phase_coefficient
            if rank_one_model
            else math.nan
        ),
        contraction_numerically_evaluated=False,
        internal_spins_summed=False,
        internal_intertwiners_integrated=False,
        standard_eprl_multi_vertex_state_sum_derived=False,
        curved_de_sitter_proper_amplitude_derived=False,
        status=(
            "THREE_CELL_RANK_ONE_PROPER_CONTRACTION_ASYMPTOTIC_CLOSED"
            if rank_one_model
            else "THREE_CELL_PROPER_BOUNDARY_MATCHING_FAILED"
        ),
    )
