'''Asymptotic integer-spin assignment for the Lorentzian 1-to-5 geometry.

The rational Minkowski coordinates are interpreted in units of one reference
length, so ``triangle_area_squared`` returns the dimensionless quantity
``(A_f / L_ref**2)**2``.  Two exact area ratios prove that no single linear
area unit can turn all twenty geometric areas into half-integer spins exactly.

The constructive replacement deliberately uses the linear proxy
``j_f ~= N * A_f / L_ref**2`` and rounds to the nearest integer.  It is not an
identification with the standard LQG area spectrum proportional to
``sqrt(j_f * (j_f + 1))``.  Integer-valued ``j`` labels are allowed SU(2)
spins.  Exact rational square-root intervals give a uniform level above which
every four-valent invariant space is nonzero, while the inherited geometric
closure defect is bounded by ``2/N``.  This does not materialize a nonzero
Livine-Speziale group average, EPRL map, proper projector, or five-vertex
amplitude.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from examples.physics.proper_vertex_boundary import four_spin_invariant_exists
from examples.physics.proper_vertex_one_to_five_boundary import (
    BOUNDARY_TRIANGLES,
    INTERNAL_TRIANGLES,
    triangle_area_squared,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_coherent_boundary import (
    certify_lorentzian_one_to_five_intrinsic_direction_spinors,
)


TriangleId = tuple[int, int, int]


OBSTRUCTION_TRIANGLES: tuple[TriangleId, TriangleId] = (
    (0, 1, 2),
    (0, 1, 3),
)
SQRT_INTERVAL_DECIMAL_DIGITS = 24


def _is_rational_square(value: Fraction) -> bool:
    if value < 0:
        return False
    return (
        math.isqrt(value.numerator) ** 2 == value.numerator
        and math.isqrt(value.denominator) ** 2 == value.denominator
    )


def _prime_valuation(integer: int, prime: int) -> int:
    if integer <= 0 or prime <= 1:
        raise ValueError('integer must be positive and prime must exceed one')
    valuation = 0
    while integer % prime == 0:
        integer //= prime
        valuation += 1
    return valuation


def _fraction_prime_valuation(value: Fraction, prime: int) -> int:
    if value <= 0:
        raise ValueError('value must be positive')
    return _prime_valuation(value.numerator, prime) - _prime_valuation(
        value.denominator,
        prime,
    )


def rational_sqrt_interval(
    value: Fraction,
    *,
    decimal_digits: int = SQRT_INTERVAL_DECIMAL_DIGITS,
) -> tuple[Fraction, Fraction]:
    '''Return exact rational lower/upper bounds for a positive square root.'''

    if value <= 0:
        raise ValueError('value must be positive')
    if not isinstance(decimal_digits, int) or decimal_digits <= 0:
        raise ValueError('decimal_digits must be a positive integer')
    denominator = 10**decimal_digits
    scaled_square_floor = (
        value.numerator * denominator**2 // value.denominator
    )
    lower_numerator = math.isqrt(scaled_square_floor)
    lower = Fraction(lower_numerator, denominator)
    if lower * lower == value:
        return lower, lower
    return lower, Fraction(lower_numerator + 1, denominator)


def nearest_integer_to_scaled_sqrt(
    value: Fraction,
    level: int,
) -> int:
    '''Return nearestInteger(level * sqrt(value)) using exact comparisons.'''

    if value <= 0:
        raise ValueError('value must be positive')
    if type(level) is not int or level <= 0:
        raise ValueError('level must be a positive integer')
    scaled_square = value * level**2
    lower = math.isqrt(scaled_square.numerator // scaled_square.denominator)
    midpoint_squared = Fraction((2 * lower + 1) ** 2, 4)
    return lower + 1 if scaled_square >= midpoint_squared else lower


def _nearest_integer_error_bound_is_exact(
    value: Fraction,
    level: int,
    nearest_integer: int,
) -> bool:
    scaled_square = value * level**2
    lower = max(Fraction(0), Fraction(2 * nearest_integer - 1, 2))
    upper = Fraction(2 * nearest_integer + 1, 2)
    return lower >= 0 and lower**2 <= scaled_square <= upper**2


@dataclass(frozen=True)
class IntegerSpinFaceData:
    triangle: TriangleId
    dimensionless_area_squared_exact: Fraction
    dimensionless_area: float
    rounded_su2_spin_j: int
    rescaled_spin_area: float
    absolute_area_error: float


@dataclass(frozen=True)
class IntegerSpinTetrahedronData:
    tetrahedron: tuple[int, int, int, int]
    face_spin_j_labels: tuple[int, int, int, int]
    dimensionless_polygon_margin: float
    exact_polygon_margin_lower_bound: Fraction
    invariant_intertwiner_space_nonzero: bool
    rescaled_closure_defect_vector: tuple[float, float, float]
    rescaled_closure_defect_norm: float


@dataclass(frozen=True)
class LorentzianOneToFiveIntegerSpinCertificate:
    level: int
    triangle_count: int
    tetrahedron_count: int
    obstruction_triangles: tuple[TriangleId, TriangleId]
    obstruction_area_squared_ratio: Fraction
    obstruction_ratio_three_adic_valuation: int
    obstruction_ratio_is_rational_square: bool
    exact_global_linear_half_integer_area_scale_ruled_out: bool
    face_data: tuple[IntegerSpinFaceData, ...]
    tetrahedron_data: tuple[IntegerSpinTetrahedronData, ...]
    exact_uniform_polygon_margin_lower_bound: Fraction
    uniform_admissibility_sufficient_level: int
    level_meets_uniform_admissibility_bound: bool
    uniform_area_error_bound: Fraction
    max_observed_dimensionless_area_error: float
    uniform_rescaled_closure_defect_bound: Fraction
    max_observed_rescaled_closure_defect: float
    all_rounding_bounds_exactly_certified: bool
    all_twenty_spins_are_positive_integers: bool
    all_fifteen_invariant_intertwiner_spaces_nonzero: bool
    closure_bound_derived_from_exact_tetrahedron_boundary_identity: bool
    numerical_direction_reconstruction_respects_closure_bound: bool
    asymptotic_linear_proxy_integer_spin_family_constructed: bool
    finite_level_exact_geometric_area_matching: bool
    finite_level_exact_geometric_closure_preserved: bool
    finite_level_spin_weighted_ls_closure_verified: bool
    finite_level_exact_regge_boundary_state_constructed: bool
    lqg_area_spectrum_scale_and_gamma_selected: bool
    livine_speziale_group_averages_materialized: bool
    eprl_y_gamma_map_materialized: bool
    shared_bra_ket_orientation_data_constructed: bool
    proper_projectors_materialized: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'LORENTZIAN_1_TO_5_LINEAR_PROXY_ADMISSIBILITY_AND_O1_OVER_N_ONLY'
    )


def certify_lorentzian_one_to_five_integer_spin_assignment(
    *,
    level: int = 118,
) -> LorentzianOneToFiveIntegerSpinCertificate:
    '''Certify exact SU(2) admissibility and asymptotic geometry bounds.'''

    if type(level) is not int or level <= 0:
        raise ValueError('level must be a positive integer')
    coordinates = lorentzian_one_to_five_coordinates()
    direction_certificate = (
        certify_lorentzian_one_to_five_intrinsic_direction_spinors()
    )
    if not (
        direction_certificate.all_geometric_face_closures_verified
        and direction_certificate.all_normalized_direction_spinors_materialized
    ):
        raise ValueError('intrinsic closure and direction spinors must be closed')

    triangles = BOUNDARY_TRIANGLES + INTERNAL_TRIANGLES
    area_squared = {
        tuple(sorted(triangle)): triangle_area_squared(triangle, coordinates)
        for triangle in triangles
    }
    first_obstruction, second_obstruction = OBSTRUCTION_TRIANGLES
    obstruction_ratio = (
        area_squared[first_obstruction] / area_squared[second_obstruction]
    )
    obstruction_is_square = _is_rational_square(obstruction_ratio)

    sqrt_intervals = {
        triangle: rational_sqrt_interval(value)
        for triangle, value in area_squared.items()
    }
    face_records: list[IntegerSpinFaceData] = []
    spin_by_triangle: dict[TriangleId, int] = {}
    for triangle in triangles:
        canonical = tuple(sorted(triangle))
        squared_area = area_squared[canonical]
        spin = nearest_integer_to_scaled_sqrt(squared_area, level)
        area = math.sqrt(float(squared_area))
        spin_by_triangle[canonical] = spin
        face_records.append(
            IntegerSpinFaceData(
                triangle=canonical,
                dimensionless_area_squared_exact=squared_area,
                dimensionless_area=area,
                rounded_su2_spin_j=spin,
                rescaled_spin_area=spin / level,
                absolute_area_error=abs(spin / level - area),
            )
        )

    tetrahedron_records: list[IntegerSpinTetrahedronData] = []
    exact_margin_lower_bounds: list[Fraction] = []
    for tetrahedron in direction_certificate.tetrahedron_data:
        faces = tuple(face.face_vertices for face in tetrahedron.face_data)
        spins = tuple(spin_by_triangle[face] for face in faces)
        areas = tuple(math.sqrt(float(area_squared[face])) for face in faces)
        margin = sum(areas) - 2.0 * max(areas)
        per_face_margin_lower_bounds = tuple(
            sum(
                sqrt_intervals[other][0]
                for other in faces
                if other != face
            )
            - sqrt_intervals[face][1]
            for face in faces
        )
        margin_lower_bound = min(per_face_margin_lower_bounds)
        exact_margin_lower_bounds.append(margin_lower_bound)
        closure_vector = np.sum(
            np.asarray(
                [
                    (spin_by_triangle[face.face_vertices] / level)
                    * np.asarray(face.unit_normal)
                    for face in tetrahedron.face_data
                ]
            ),
            axis=0,
        )
        tetrahedron_records.append(
            IntegerSpinTetrahedronData(
                tetrahedron=tetrahedron.tetrahedron,
                face_spin_j_labels=spins,  # type: ignore[arg-type]
                dimensionless_polygon_margin=margin,
                exact_polygon_margin_lower_bound=margin_lower_bound,
                invariant_intertwiner_space_nonzero=(
                    four_spin_invariant_exists(
                        tuple(Fraction(spin) for spin in spins)
                    )
                ),
                rescaled_closure_defect_vector=tuple(
                    float(value) for value in closure_vector
                ),
                rescaled_closure_defect_norm=float(
                    np.linalg.norm(closure_vector)
                ),
            )
        )

    uniform_margin_lower_bound = min(exact_margin_lower_bounds)
    if uniform_margin_lower_bound <= 0:
        raise ValueError('all tetrahedra need a positive polygon margin')
    threshold_ratio = Fraction(2, 1) / uniform_margin_lower_bound
    sufficient_level = (
        threshold_ratio.numerator // threshold_ratio.denominator + 1
    )
    area_error_bound = Fraction(1, 2 * level)
    closure_bound = Fraction(2, level)
    max_area_error = max(record.absolute_area_error for record in face_records)
    max_closure_defect = max(
        record.rescaled_closure_defect_norm
        for record in tetrahedron_records
    )
    rounding_bounds_certified = all(
        _nearest_integer_error_bound_is_exact(
            record.dimensionless_area_squared_exact,
            level,
            record.rounded_su2_spin_j,
        )
        for record in face_records
    )
    all_positive_integer = all(
        isinstance(record.rounded_su2_spin_j, int)
        and record.rounded_su2_spin_j > 0
        for record in face_records
    )
    all_admissible = all(
        record.invariant_intertwiner_space_nonzero
        for record in tetrahedron_records
    )
    level_meets_bound = level >= sufficient_level
    exact_boundary_identity_applies = all(
        tetrahedron.nondegenerate_spacelike
        and tetrahedron.all_face_areas_positive
        for tetrahedron in direction_certificate.tetrahedron_data
    )
    numerical_closure_bound_respected = (
        max_closure_defect <= float(closure_bound) + 1.0e-12
    )
    constructed = (
        len(face_records) == 20
        and len(tetrahedron_records) == 15
        and rounding_bounds_certified
        and all_positive_integer
        and all_admissible
        and level_meets_bound
        and exact_boundary_identity_applies
        and numerical_closure_bound_respected
    )

    return LorentzianOneToFiveIntegerSpinCertificate(
        level=level,
        triangle_count=len(face_records),
        tetrahedron_count=len(tetrahedron_records),
        obstruction_triangles=OBSTRUCTION_TRIANGLES,
        obstruction_area_squared_ratio=obstruction_ratio,
        obstruction_ratio_three_adic_valuation=(
            _fraction_prime_valuation(obstruction_ratio, 3)
        ),
        obstruction_ratio_is_rational_square=obstruction_is_square,
        exact_global_linear_half_integer_area_scale_ruled_out=(
            not obstruction_is_square
        ),
        face_data=tuple(face_records),
        tetrahedron_data=tuple(tetrahedron_records),
        exact_uniform_polygon_margin_lower_bound=uniform_margin_lower_bound,
        uniform_admissibility_sufficient_level=sufficient_level,
        level_meets_uniform_admissibility_bound=level_meets_bound,
        uniform_area_error_bound=area_error_bound,
        max_observed_dimensionless_area_error=max_area_error,
        uniform_rescaled_closure_defect_bound=closure_bound,
        max_observed_rescaled_closure_defect=max_closure_defect,
        all_rounding_bounds_exactly_certified=rounding_bounds_certified,
        all_twenty_spins_are_positive_integers=all_positive_integer,
        all_fifteen_invariant_intertwiner_spaces_nonzero=all_admissible,
        closure_bound_derived_from_exact_tetrahedron_boundary_identity=(
            exact_boundary_identity_applies
        ),
        numerical_direction_reconstruction_respects_closure_bound=(
            numerical_closure_bound_respected
        ),
        asymptotic_linear_proxy_integer_spin_family_constructed=constructed,
        finite_level_exact_geometric_area_matching=False,
        finite_level_exact_geometric_closure_preserved=False,
        finite_level_spin_weighted_ls_closure_verified=False,
        finite_level_exact_regge_boundary_state_constructed=False,
        lqg_area_spectrum_scale_and_gamma_selected=False,
        livine_speziale_group_averages_materialized=False,
        eprl_y_gamma_map_materialized=False,
        shared_bra_ket_orientation_data_constructed=False,
        proper_projectors_materialized=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_LINEAR_PROXY_SPIN_ADMISSIBILITY_CLOSED'
            if constructed
            else 'LORENTZIAN_1_TO_5_LINEAR_PROXY_UNIFORM_BOUND_NOT_CLOSED'
        ),
    )
