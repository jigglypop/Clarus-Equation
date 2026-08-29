from __future__ import annotations

from fractions import Fraction

import pytest

from examples.physics.proper_vertex_boundary import four_spin_invariant_exists
from examples.physics.proper_vertex_one_to_five_spin_assignment import (
    certify_lorentzian_one_to_five_integer_spin_assignment,
    nearest_integer_to_scaled_sqrt,
    rational_sqrt_interval,
)


def test_two_exact_areas_obstruct_one_global_linear_half_integer_scale() -> None:
    certificate = certify_lorentzian_one_to_five_integer_spin_assignment()

    assert certificate.obstruction_triangles == ((0, 1, 2), (0, 1, 3))
    assert certificate.obstruction_area_squared_ratio == Fraction(
        999983,
        999918,
    )
    assert certificate.obstruction_ratio_three_adic_valuation == -3
    assert not certificate.obstruction_ratio_is_rational_square
    assert not certificate.exact_global_linear_half_integer_area_scale_exists


def test_rational_square_root_intervals_are_exact_enclosures() -> None:
    values = (
        Fraction(999983, 4000000),
        Fraction(499959, 2000000),
        Fraction(9, 16),
    )

    for value in values:
        lower, upper = rational_sqrt_interval(value)
        assert lower**2 <= value <= upper**2
        assert upper - lower <= Fraction(1, 10**24)
    assert rational_sqrt_interval(Fraction(9, 16)) == (
        Fraction(3, 4),
        Fraction(3, 4),
    )


def test_nearest_integer_spin_rounding_uses_exact_midpoint_comparisons() -> None:
    assert nearest_integer_to_scaled_sqrt(Fraction(1, 1), 118) == 118
    assert nearest_integer_to_scaled_sqrt(Fraction(1, 4), 3) == 2
    assert nearest_integer_to_scaled_sqrt(Fraction(1, 4), 2) == 1


def test_level_118_certifies_all_twenty_spins_and_fifteen_intertwiners() -> None:
    certificate = certify_lorentzian_one_to_five_integer_spin_assignment(
        level=118
    )

    assert certificate.triangle_count == 20
    assert certificate.tetrahedron_count == 15
    assert certificate.uniform_admissibility_sufficient_level == 118
    margin = certificate.exact_uniform_polygon_margin_lower_bound
    assert 117 * margin <= 2 < 118 * margin
    assert certificate.level_meets_uniform_admissibility_bound
    assert certificate.all_rounding_bounds_exactly_certified
    assert certificate.all_twenty_spins_are_positive_integers
    assert certificate.all_fifteen_invariant_intertwiner_spaces_nonzero
    assert certificate.asymptotic_integer_spin_family_constructed
    assert certificate.status == (
        'LORENTZIAN_1_TO_5_INTEGER_SPIN_ADMISSIBILITY_CLOSED'
    )
    assert min(face.integer_spin for face in certificate.face_data) == 26
    assert max(face.integer_spin for face in certificate.face_data) == 364
    assert all(
        four_spin_invariant_exists(
            tuple(Fraction(spin) for spin in tetrahedron.face_spins)
        )
        for tetrahedron in certificate.tetrahedron_data
    )


def test_area_and_inherited_closure_defects_obey_uniform_one_over_n_bounds() -> None:
    certificate = certify_lorentzian_one_to_five_integer_spin_assignment(
        level=118
    )

    assert certificate.uniform_area_error_bound == Fraction(1, 236)
    assert certificate.max_observed_dimensionless_area_error <= float(
        certificate.uniform_area_error_bound
    )
    assert certificate.uniform_rescaled_closure_defect_bound == Fraction(
        1,
        59,
    )
    assert certificate.max_observed_rescaled_closure_defect <= float(
        certificate.uniform_rescaled_closure_defect_bound
    )
    assert certificate.max_observed_rescaled_closure_defect > 0.0

    finer = certify_lorentzian_one_to_five_integer_spin_assignment(level=1180)
    assert finer.asymptotic_integer_spin_family_constructed
    assert finer.uniform_area_error_bound == Fraction(1, 2360)
    assert finer.uniform_rescaled_closure_defect_bound == Fraction(1, 590)


def test_spin_labels_are_global_even_though_normals_are_tetrahedron_local() -> None:
    certificate = certify_lorentzian_one_to_five_integer_spin_assignment()
    spin_by_triangle = {
        face.triangle: face.integer_spin for face in certificate.face_data
    }

    for tetrahedron in certificate.tetrahedron_data:
        expected = tuple(
            spin_by_triangle[tuple(sorted(set(tetrahedron.tetrahedron) - {vertex}))]
            for vertex in tetrahedron.tetrahedron
        )
        assert sorted(tetrahedron.face_spins) == sorted(expected)


def test_claim_ceiling_keeps_exact_geometry_ls_and_proper_steps_open() -> None:
    certificate = certify_lorentzian_one_to_five_integer_spin_assignment()

    assert not certificate.finite_level_exact_geometric_area_matching
    assert not certificate.finite_level_exact_geometric_closure_preserved
    assert not certificate.livine_speziale_group_averages_materialized
    assert not certificate.eprl_y_gamma_map_materialized
    assert not certificate.shared_bra_ket_orientation_data_constructed
    assert not certificate.proper_projectors_materialized
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.claim_ceiling.endswith('ADMISSIBILITY_AND_O1_OVER_N_ONLY')


@pytest.mark.parametrize('level', (0, -1, 1.5))
def test_invalid_levels_are_rejected(level: object) -> None:
    with pytest.raises(ValueError, match='level must be a positive integer'):
        certify_lorentzian_one_to_five_integer_spin_assignment(  # type: ignore[arg-type]
            level=level
        )
