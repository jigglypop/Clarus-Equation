from __future__ import annotations

import math

import pytest

from reality_stone.clarus.clarus_negative_source_search import (
    averaged_nonminimal_null_audit,
    casimir_plate_scale_audit,
    clarus_negative_source_funnel,
    effective_planck_amplification_audit,
    nonminimal_scalar_null_audit,
    physical_averaged_nonminimal_null_audit,
)
from reality_stone.clarus.spatial_folding import (
    casimir_cell_conversion_audit,
    wormhole_throat_audit,
)


def test_minimally_coupled_canonical_scalar_stays_nec_nonnegative() -> None:
    audit = nonminimal_scalar_null_audit(
        nonminimal_coupling=0.0,
        field_value_planck=0.5,
        affine_first_derivative=0.1,
        affine_second_derivative=1.0,
    )

    assert math.isclose(audit.effective_null_source, 0.01)
    assert not audit.locally_violates_effective_nec
    assert audit.canonical_kinetic_sign_retained


def test_ce_nonminimal_coupling_has_a_local_nec_violating_region_without_phantom_sign() -> None:
    audit = nonminimal_scalar_null_audit(
        nonminimal_coupling=0.49,
        field_value_planck=0.5,
        affine_first_derivative=0.1,
        affine_second_derivative=1.0,
    )

    assert audit.null_numerator < 0.0
    assert audit.effective_planck_factor > 0.0
    assert audit.locally_violates_effective_nec
    assert audit.canonical_kinetic_sign_retained
    assert audit.local_candidate_survives
    assert not audit.global_solution_derived


def test_effective_planck_sign_flip_rejects_local_candidate() -> None:
    audit = nonminimal_scalar_null_audit(
        nonminimal_coupling=0.49,
        field_value_planck=2.0,
        affine_first_derivative=0.1,
        affine_second_derivative=1.0,
    )

    assert not audit.positive_effective_planck_mass
    assert math.isnan(audit.effective_null_source)
    assert not audit.local_candidate_survives


def test_localized_complete_profile_loses_nonminimal_boundary_advantage() -> None:
    audit = averaged_nonminimal_null_audit(
        nonminimal_coupling=0.49,
        gradient_squared_integral=2.0,
        endpoint_field_squared_derivative_jump=0.0,
    )

    assert audit.averaged_null_numerator == 2.0
    assert not audit.averaged_null_numerator_negative
    assert not audit.physical_effective_anec_computed
    assert audit.localized_vacuum_boundary_conditions
    assert audit.boundary_or_topology_support_required


def test_nonzero_boundary_term_can_make_the_unrearranged_average_negative() -> None:
    audit = averaged_nonminimal_null_audit(
        nonminimal_coupling=0.49,
        gradient_squared_integral=2.0,
        endpoint_field_squared_derivative_jump=5.0,
    )

    assert math.isclose(audit.averaged_null_numerator, -0.45)
    assert audit.averaged_null_numerator_negative
    assert not audit.localized_vacuum_boundary_conditions


def test_physical_effective_anec_keeps_local_negative_pocket_but_positive_average() -> None:
    sample_count = 4_001
    coordinates = [
        -8.0 + 16.0 * index / (sample_count - 1)
        for index in range(sample_count)
    ]
    field = [0.5 * math.exp(-value * value / 2.0) for value in coordinates]
    first = [-value * profile for value, profile in zip(coordinates, field, strict=True)]
    second = [
        (value * value - 1.0) * profile
        for value, profile in zip(coordinates, field, strict=True)
    ]
    audit = physical_averaged_nonminimal_null_audit(
        nonminimal_coupling=0.49,
        affine_parameter=coordinates,
        field_value_planck=field,
        affine_first_derivative=first,
        affine_second_derivative=second,
    )

    assert audit.local_effective_nec_violation_sampled
    assert math.isclose(audit.minimum_effective_planck_factor, 0.8775)
    assert math.isclose(audit.direct_effective_anec, 0.2535027873638524)
    assert audit.kinetic_over_planck_integral > 0.0
    assert audit.log_planck_gradient_squared_integral > 0.0
    assert abs(audit.endpoint_log_planck_derivative_jump) < 1.0e-20
    assert abs(audit.identity_residual) < 1.0e-12
    assert audit.identity_numerically_verified
    assert audit.localized_log_planck_boundary_conditions
    assert not audit.physical_effective_anec_violated
    assert audit.healthy_localized_profile_anec_nonnegative


def test_unrearranged_numerator_can_false_positive_against_physical_anec() -> None:
    xi = 0.2796948642528183
    coefficients = (-0.3830363518, 0.6322527182, -0.6385472402, 0.1632003273)
    sample_count = 20_001
    coordinates = [
        -1.0 + 2.0 * index / (sample_count - 1)
        for index in range(sample_count)
    ]
    field = [
        coefficients[0]
        + coefficients[1] * value
        + coefficients[2] * value**2
        + coefficients[3] * value**3
        for value in coordinates
    ]
    first = [
        coefficients[1]
        + 2.0 * coefficients[2] * value
        + 3.0 * coefficients[3] * value**2
        for value in coordinates
    ]
    second = [
        2.0 * coefficients[2] + 6.0 * coefficients[3] * value
        for value in coordinates
    ]
    gradient_integral = math.fsum(
        0.5
        * (first[index] ** 2 + first[index + 1] ** 2)
        * (coordinates[index + 1] - coordinates[index])
        for index in range(sample_count - 1)
    )
    endpoint_jump = 2.0 * (
        field[-1] * first[-1] - field[0] * first[0]
    )

    numerator_only = averaged_nonminimal_null_audit(
        nonminimal_coupling=xi,
        gradient_squared_integral=gradient_integral,
        endpoint_field_squared_derivative_jump=endpoint_jump,
    )
    physical = physical_averaged_nonminimal_null_audit(
        nonminimal_coupling=xi,
        affine_parameter=coordinates,
        field_value_planck=field,
        affine_first_derivative=first,
        affine_second_derivative=second,
    )

    assert numerator_only.averaged_null_numerator < -0.06
    assert numerator_only.averaged_null_numerator_negative
    assert physical.minimum_effective_planck_factor > 0.07
    assert physical.direct_effective_anec > 0.07
    assert not physical.physical_effective_anec_violated
    assert not physical.localized_log_planck_boundary_conditions


def test_physical_effective_anec_rejects_planck_factor_zero_crossing() -> None:
    with pytest.raises(ValueError, match="Planck factor"):
        physical_averaged_nonminimal_null_audit(
            nonminimal_coupling=0.49,
            affine_parameter=[-1.0, 0.0, 1.0],
            field_value_planck=[0.0, 2.0, 0.0],
            affine_first_derivative=[2.0, 0.0, -2.0],
            affine_second_derivative=[0.0, -4.0, 0.0],
        )


@pytest.mark.parametrize(
    "coordinates",
    [[0.0, 1.0], [0.0, 1.0, 1.0]],
)
def test_physical_effective_anec_rejects_invalid_affine_grid(
    coordinates: list[float],
) -> None:
    values = [0.0] * len(coordinates)
    with pytest.raises(ValueError):
        physical_averaged_nonminimal_null_audit(
            nonminimal_coupling=0.49,
            affine_parameter=coordinates,
            field_value_planck=values,
            affine_first_derivative=values,
            affine_second_derivative=values,
        )


def test_candidate_funnel_demotes_boundary_route_after_physical_scale_failure() -> None:
    funnel = clarus_negative_source_funnel()
    frontier_a = [candidate for candidate in funnel if candidate.frontier == "FRONTIER_A"]
    deferred_boundary = [
        candidate
        for candidate in funnel
        if candidate.frontier == "DEFERRED_PHYSICAL_BOUNDARY"
    ]
    rejected = [candidate for candidate in funnel if candidate.frontier == "REJECTED"]

    assert len(funnel) == 9
    assert len(frontier_a) == 1
    assert len(deferred_boundary) == 1
    assert "Casimir" in deferred_boundary[0].name
    assert rejected[0].name == "phantom Clarus scalar"


def test_ideal_plate_control_requires_subnuclear_separation_for_one_meter_throat() -> None:
    density = casimir_cell_conversion_audit().energy_density_j_m3
    throat = wormhole_throat_audit(
        throat_radius_m=1.0,
        candidate_negative_density_j_m3=density,
    )
    audit = casimir_plate_scale_audit(
        required_null_magnitude_j_m3=abs(throat.nec_energy_density_j_m3),
        ce_correlation_length_m=6.65e-15,
    )

    assert 3.6e-18 < audit.plate_separation_m < 3.8e-18
    assert audit.separation_to_ce_correlation_ratio < 6e-4
    assert not audit.macroscopic_throat_source_established


def test_nonminimal_denominator_closes_gap_only_near_zero_effective_planck_factor() -> None:
    audit = effective_planck_amplification_audit(
        required_amplification=2.8504437240960828e16,
        nonminimal_coupling=0.49,
    )

    assert audit.required_effective_planck_factor < 3.6e-17
    assert 1.42 < audit.critical_field_planck < 1.43
    assert audit.relative_distance_below_critical < 1e-15
    assert audit.algebraically_closes_density_gap
    assert not audit.regular_effective_gravity_limit_established
