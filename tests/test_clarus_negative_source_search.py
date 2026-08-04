from __future__ import annotations

import math

from reality_stone.clarus.clarus_negative_source_search import (
    averaged_nonminimal_null_audit,
    casimir_plate_scale_audit,
    clarus_negative_source_funnel,
    effective_planck_amplification_audit,
    nonminimal_scalar_null_audit,
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
    assert not audit.averaged_nec_violated
    assert audit.localized_vacuum_boundary_conditions
    assert audit.boundary_or_topology_support_required


def test_nonzero_boundary_term_can_make_the_unrearranged_average_negative() -> None:
    audit = averaged_nonminimal_null_audit(
        nonminimal_coupling=0.49,
        gradient_squared_integral=2.0,
        endpoint_field_squared_derivative_jump=5.0,
    )

    assert math.isclose(audit.averaged_null_numerator, -0.45)
    assert audit.averaged_nec_violated
    assert not audit.localized_vacuum_boundary_conditions


def test_candidate_funnel_keeps_two_ce_native_frontier_a_options() -> None:
    funnel = clarus_negative_source_funnel()
    frontier_a = [candidate for candidate in funnel if candidate.frontier == "FRONTIER_A"]
    rejected = [candidate for candidate in funnel if candidate.frontier == "REJECTED"]

    assert len(funnel) == 8
    assert len(frontier_a) == 2
    assert "Casimir" in frontier_a[0].name
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
