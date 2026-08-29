'''Explicit local Livine-Speziale intertwiners on one 1-to-5 spin witness.

At the computationally small global spin level ``N=3``, all fifteen
four-valent SU(2) invariant spaces are nonzero.  For each tetrahedron this
module expands the product of four direction-coherent states in an
orthonormal recoupling basis and stores

    P_Inv tensor_f |j_f, xi_tf> = sum_k c_k |k>.

The invariant projector ``P_Inv`` is the normalized SU(2) Haar group average,
so the stored nonzero coefficient vector is an explicit numerical
Livine-Speziale intertwiner witness.  The calculation is deliberately local:
shared bra/ket orientation, independent tetrahedron frames, the EPRL
``Y_gamma`` map, Lorentzian SL(2,C) integration, and proper projectors remain
open.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from examples.physics.proper_vertex_one_to_five_coherent_boundary import (
    certify_lorentzian_one_to_five_intrinsic_direction_spinors,
)
from examples.physics.proper_vertex_one_to_five_spin_assignment import (
    certify_lorentzian_one_to_five_integer_spin_assignment,
)


TriangleId = tuple[int, int, int]
TetrahedronId = tuple[int, int, int, int]


def spin_coherent_state_coefficients(
    spin_j: int,
    direction_spinor: Sequence[complex],
) -> tuple[complex, ...]:
    '''Return |j,xi> coefficients ordered by m=-j,...,j.'''

    if type(spin_j) is not int or spin_j < 0:
        raise ValueError('spin_j must be a nonnegative integer')
    spinor = np.asarray(direction_spinor, dtype=complex)
    if spinor.shape != (2,) or not np.all(np.isfinite(spinor)):
        raise ValueError('direction_spinor must have two finite components')
    norm = float(np.vdot(spinor, spinor).real)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError('direction_spinor must be normalized')
    first, second = (complex(value) for value in spinor)
    return tuple(
        math.sqrt(math.comb(2 * spin_j, spin_j + magnetic_m))
        * first ** (spin_j + magnetic_m)
        * second ** (spin_j - magnetic_m)
        for magnetic_m in range(-spin_j, spin_j + 1)
    )


@lru_cache(maxsize=None)
def integer_clebsch_gordan(
    first_spin: int,
    first_magnetic: int,
    second_spin: int,
    second_magnetic: int,
    coupled_spin: int,
    coupled_magnetic: int,
) -> float:
    '''Return a Condon-Shortley CG coefficient for integer SU(2) spins.'''

    labels = (
        first_spin,
        first_magnetic,
        second_spin,
        second_magnetic,
        coupled_spin,
        coupled_magnetic,
    )
    if any(type(label) is not int for label in labels):
        raise ValueError('all Clebsch-Gordan labels must be integers')
    if (
        first_spin < 0
        or second_spin < 0
        or coupled_spin < 0
        or abs(first_magnetic) > first_spin
        or abs(second_magnetic) > second_spin
        or abs(coupled_magnetic) > coupled_spin
        or coupled_magnetic != first_magnetic + second_magnetic
        or coupled_spin < abs(first_spin - second_spin)
        or coupled_spin > first_spin + second_spin
    ):
        return 0.0

    factorial = math.factorial
    triangle_factor = Fraction(
        (2 * coupled_spin + 1)
        * factorial(coupled_spin + first_spin - second_spin)
        * factorial(coupled_spin - first_spin + second_spin)
        * factorial(first_spin + second_spin - coupled_spin),
        factorial(first_spin + second_spin + coupled_spin + 1),
    )
    magnetic_factor = (
        factorial(coupled_spin + coupled_magnetic)
        * factorial(coupled_spin - coupled_magnetic)
        * factorial(first_spin - first_magnetic)
        * factorial(first_spin + first_magnetic)
        * factorial(second_spin - second_magnetic)
        * factorial(second_spin + second_magnetic)
    )

    lower_sum_index = max(
        0,
        second_spin - coupled_spin - first_magnetic,
        first_spin + second_magnetic - coupled_spin,
    )
    upper_sum_index = min(
        first_spin + second_spin - coupled_spin,
        first_spin - first_magnetic,
        second_spin + second_magnetic,
    )
    series = Fraction(0)
    for summation_index in range(lower_sum_index, upper_sum_index + 1):
        denominator = (
            factorial(summation_index)
            * factorial(first_spin + second_spin - coupled_spin - summation_index)
            * factorial(first_spin - first_magnetic - summation_index)
            * factorial(second_spin + second_magnetic - summation_index)
            * factorial(
                coupled_spin - second_spin + first_magnetic + summation_index
            )
            * factorial(
                coupled_spin - first_spin - second_magnetic + summation_index
            )
        )
        series += Fraction((-1) ** summation_index, denominator)
    return (
        math.sqrt(float(triangle_factor * magnetic_factor))
        * float(series)
    )


def allowed_four_valent_recoupling_channels(
    spins_j: Sequence[int],
) -> tuple[int, ...]:
    '''Return k values shared by the (12) and (34) coupling channels.'''

    if len(spins_j) != 4 or any(
        type(spin) is not int or spin < 0 for spin in spins_j
    ):
        raise ValueError('spins_j must contain four nonnegative integers')
    first, second, third, fourth = spins_j
    lower = max(abs(first - second), abs(third - fourth))
    upper = min(first + second, third + fourth)
    return tuple(range(lower, upper + 1)) if lower <= upper else ()


def _pair_coupled_amplitudes(
    first_spin: int,
    first_state: Sequence[complex],
    second_spin: int,
    second_state: Sequence[complex],
    coupled_spin: int,
) -> dict[int, complex]:
    amplitudes: dict[int, complex] = {}
    for coupled_magnetic in range(-coupled_spin, coupled_spin + 1):
        value = 0.0j
        for first_magnetic in range(-first_spin, first_spin + 1):
            second_magnetic = coupled_magnetic - first_magnetic
            if -second_spin <= second_magnetic <= second_spin:
                value += integer_clebsch_gordan(
                    first_spin,
                    first_magnetic,
                    second_spin,
                    second_magnetic,
                    coupled_spin,
                    coupled_magnetic,
                ) * (
                    first_state[first_magnetic + first_spin]
                    * second_state[second_magnetic + second_spin]
                )
        amplitudes[coupled_magnetic] = value
    return amplitudes


def four_valent_ls_recoupling_coefficients(
    spins_j: Sequence[int],
    direction_spinors: Sequence[Sequence[complex]],
) -> tuple[tuple[int, ...], tuple[complex, ...]]:
    '''Project four coherent states into the invariant recoupling basis.'''

    channels = allowed_four_valent_recoupling_channels(spins_j)
    if len(direction_spinors) != 4:
        raise ValueError('direction_spinors must contain four spinors')
    coherent_states = tuple(
        spin_coherent_state_coefficients(spin, spinor)
        for spin, spinor in zip(spins_j, direction_spinors)
    )
    coefficients: list[complex] = []
    first_spin, second_spin, third_spin, fourth_spin = spins_j
    for channel in channels:
        first_pair = _pair_coupled_amplitudes(
            first_spin,
            coherent_states[0],
            second_spin,
            coherent_states[1],
            channel,
        )
        second_pair = _pair_coupled_amplitudes(
            third_spin,
            coherent_states[2],
            fourth_spin,
            coherent_states[3],
            channel,
        )
        coefficient = sum(
            ((-1) ** (channel - magnetic))
            * first_pair[magnetic]
            * second_pair[-magnetic]
            / math.sqrt(2 * channel + 1)
            for magnetic in range(-channel, channel + 1)
        )
        coefficients.append(coefficient)
    return channels, tuple(coefficients)


@dataclass(frozen=True)
class LocalLivineSpezialeIntertwinerData:
    tetrahedron: TetrahedronId
    face_triangles: tuple[TriangleId, TriangleId, TriangleId, TriangleId]
    face_spin_j_labels: tuple[int, int, int, int]
    recoupling_channel_k_values: tuple[int, ...]
    invariant_space_dimension: int
    unnormalized_recoupling_coefficients: tuple[complex, ...]
    unnormalized_group_average_norm: float
    normalized_recoupling_coefficients: tuple[complex, ...]
    normalized_coefficient_norm_residual: float
    product_coherent_state_norm_residual: float
    nonzero_group_averaged_intertwiner_materialized: bool


@dataclass(frozen=True)
class LorentzianOneToFiveLocalLSCertificate:
    spin_level: int
    tetrahedron_count: int
    global_triangle_spin_count: int
    max_spin_j: int
    tetrahedron_data: tuple[LocalLivineSpezialeIntertwinerData, ...]
    min_unnormalized_group_average_norm: float
    max_normalized_coefficient_norm_residual: float
    max_product_coherent_state_norm_residual: float
    all_fifteen_invariant_spaces_nonzero: bool
    all_fifteen_local_ls_group_averages_materialized: bool
    normalized_haar_projector_identified_with_invariant_projector: bool
    spin_weighted_geometric_closure_exact: bool
    independent_tetrahedron_su2_frames_constructed: bool
    tetrahedron_time_orientations_assigned: bool
    shared_bra_ket_dualization_constructed: bool
    eprl_y_gamma_map_materialized: bool
    lorentzian_sl2c_group_integrals_evaluated: bool
    proper_projectors_materialized: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = 'LOCAL_NUMERICAL_LS_INTERTWINERS_ONLY'


def certify_lorentzian_one_to_five_local_ls_intertwiners(
    *,
    spin_level: int = 3,
    computational_spin_cap: int = 12,
    nonzero_tolerance: float = 1.0e-14,
) -> LorentzianOneToFiveLocalLSCertificate:
    '''Materialize all fifteen local LS vectors in recoupling bases.'''

    if type(spin_level) is not int or spin_level <= 0:
        raise ValueError('spin_level must be a positive integer')
    if type(computational_spin_cap) is not int or computational_spin_cap < 0:
        raise ValueError('computational_spin_cap must be a nonnegative integer')
    if not math.isfinite(nonzero_tolerance) or nonzero_tolerance <= 0.0:
        raise ValueError('nonzero_tolerance must be finite and positive')

    spin_certificate = certify_lorentzian_one_to_five_integer_spin_assignment(
        level=spin_level
    )
    direction_certificate = (
        certify_lorentzian_one_to_five_intrinsic_direction_spinors()
    )
    if not spin_certificate.all_twenty_spins_are_positive_integers:
        raise ValueError('all global face spins must be positive integers')
    if not spin_certificate.all_fifteen_invariant_intertwiner_spaces_nonzero:
        raise ValueError('all four-valent invariant spaces must be nonzero')
    spin_by_triangle = {
        face.triangle: face.rounded_su2_spin_j
        for face in spin_certificate.face_data
    }
    largest_spin = max(spin_by_triangle.values())
    if largest_spin > computational_spin_cap:
        raise ValueError(
            'spin level exceeds the explicit recoupling computational cap'
        )

    tetrahedron_records: list[LocalLivineSpezialeIntertwinerData] = []
    for tetrahedron in direction_certificate.tetrahedron_data:
        faces = tuple(face.face_vertices for face in tetrahedron.face_data)
        spins = tuple(spin_by_triangle[face] for face in faces)
        spinors = tuple(face.direction_spinor for face in tetrahedron.face_data)
        coherent_states = tuple(
            spin_coherent_state_coefficients(spin, spinor)
            for spin, spinor in zip(spins, spinors)
        )
        product_norm = math.prod(
            float(np.vdot(state, state).real) for state in coherent_states
        )
        channels, coefficients = four_valent_ls_recoupling_coefficients(
            spins,
            spinors,
        )
        group_average_norm = math.sqrt(
            sum(abs(coefficient) ** 2 for coefficient in coefficients)
        )
        nonzero = group_average_norm > nonzero_tolerance
        normalized = (
            tuple(coefficient / group_average_norm for coefficient in coefficients)
            if nonzero
            else tuple(0.0j for _ in coefficients)
        )
        normalized_norm_residual = (
            abs(sum(abs(value) ** 2 for value in normalized) - 1.0)
            if nonzero
            else 1.0
        )
        tetrahedron_records.append(
            LocalLivineSpezialeIntertwinerData(
                tetrahedron=tetrahedron.tetrahedron,
                face_triangles=faces,  # type: ignore[arg-type]
                face_spin_j_labels=spins,  # type: ignore[arg-type]
                recoupling_channel_k_values=channels,
                invariant_space_dimension=len(channels),
                unnormalized_recoupling_coefficients=coefficients,
                unnormalized_group_average_norm=group_average_norm,
                normalized_recoupling_coefficients=normalized,
                normalized_coefficient_norm_residual=normalized_norm_residual,
                product_coherent_state_norm_residual=abs(product_norm - 1.0),
                nonzero_group_averaged_intertwiner_materialized=nonzero,
            )
        )

    min_group_norm = min(
        record.unnormalized_group_average_norm
        for record in tetrahedron_records
    )
    max_normalized_residual = max(
        record.normalized_coefficient_norm_residual
        for record in tetrahedron_records
    )
    max_product_residual = max(
        record.product_coherent_state_norm_residual
        for record in tetrahedron_records
    )
    all_nonzero = all(
        record.nonzero_group_averaged_intertwiner_materialized
        for record in tetrahedron_records
    )
    closed = (
        len(tetrahedron_records) == 15
        and all_nonzero
        and max_normalized_residual <= 1.0e-12
        and max_product_residual <= 1.0e-12
    )
    return LorentzianOneToFiveLocalLSCertificate(
        spin_level=spin_level,
        tetrahedron_count=len(tetrahedron_records),
        global_triangle_spin_count=len(spin_by_triangle),
        max_spin_j=largest_spin,
        tetrahedron_data=tuple(tetrahedron_records),
        min_unnormalized_group_average_norm=min_group_norm,
        max_normalized_coefficient_norm_residual=max_normalized_residual,
        max_product_coherent_state_norm_residual=max_product_residual,
        all_fifteen_invariant_spaces_nonzero=(
            spin_certificate.all_fifteen_invariant_intertwiner_spaces_nonzero
        ),
        all_fifteen_local_ls_group_averages_materialized=all_nonzero,
        normalized_haar_projector_identified_with_invariant_projector=True,
        spin_weighted_geometric_closure_exact=False,
        independent_tetrahedron_su2_frames_constructed=False,
        tetrahedron_time_orientations_assigned=False,
        shared_bra_ket_dualization_constructed=False,
        eprl_y_gamma_map_materialized=False,
        lorentzian_sl2c_group_integrals_evaluated=False,
        proper_projectors_materialized=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_LOCAL_NUMERICAL_LS_INTERTWINERS_MATERIALIZED'
            if closed
            else 'LORENTZIAN_1_TO_5_LOCAL_LS_INTERTWINERS_FAILED'
        ),
    )
