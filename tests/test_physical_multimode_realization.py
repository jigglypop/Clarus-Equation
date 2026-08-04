from __future__ import annotations

import math

import pytest

from reality_stone.clarus.physical_multimode_realization import (
    physical_multimode_realization_audit,
)


def test_one_metre_target_requires_subnuclear_boundary_and_153_gev_mode() -> None:
    audit = physical_multimode_realization_audit()

    assert 4.0e-18 < audit.ideal_casimir_separation_m < 4.1e-18
    assert 1.52e11 < audit.fundamental_energy_ev < 1.54e11
    assert audit.separation_to_boundary_resolution_ratio < 0.005
    assert not audit.boundary_resolved_at_target_radius
    assert not audit.physical_reflector_model_derived


def test_boundary_resolution_tradeoff_moves_cost_to_large_gravitating_mass() -> None:
    audit = physical_multimode_realization_audit()

    assert 4.2e4 < audit.radius_for_boundary_resolution_m < 4.4e4
    assert 9.0 < audit.resolved_radius_coordinate_mass_equivalent_solar < 11.0
    assert 18.0 < audit.resolved_radius_proper_mass_equivalent_solar < 19.0
    assert 70.0 < audit.two_sided_coordinate_mass_equivalent_earths < 80.0
    assert 140.0 < audit.two_sided_proper_mass_equivalent_earths < 145.0


def test_coordinate_and_proper_energy_integrals_are_not_conflated() -> None:
    audit = physical_multimode_realization_audit()

    assert math.isclose(
        audit.two_sided_coordinate_density_integral_magnitude_j,
        4.034185214e43,
        rel_tol=1e-9,
    )
    assert math.isclose(
        audit.proper_volume_integral_dimensionless,
        0.6314661793178,
        rel_tol=1e-12,
    )
    assert math.isclose(
        audit.two_sided_proper_matter_energy_magnitude_j,
        7.642354572e43,
        rel_tol=1e-9,
    )
    assert math.isclose(
        audit.proper_to_coordinate_energy_ratio,
        1.8943985379534,
        rel_tol=1e-12,
    )
    assert audit.proper_volume_tail_bound_dimensionless < 1.0e-27
    assert audit.proper_volume_quadrature_delta_dimensionless < 1.0e-12


def test_multimodes_do_not_reduce_fixed_geometry_energy_or_qi_control() -> None:
    audit = physical_multimode_realization_audit()

    assert audit.crossing_to_qi_duration_ratio > 1.0e17
    assert not audit.multimode_superposition_reduces_geometric_total
    assert not audit.renormalized_multimode_stress_derived
    assert not audit.current_physical_realization_pass


@pytest.mark.parametrize("radius,boundary", [(0.0, 1.0), (1.0, -1.0)])
def test_nonpositive_physical_scales_are_rejected(radius: float, boundary: float) -> None:
    with pytest.raises(ValueError):
        physical_multimode_realization_audit(
            throat_radius_m=radius,
            boundary_resolution_m=boundary,
        )


@pytest.mark.parametrize("steps", [996, 1_002, 1_000.0, True])
def test_invalid_proper_volume_quadrature_is_rejected(steps: object) -> None:
    with pytest.raises(ValueError):
        physical_multimode_realization_audit(
            proper_volume_integration_steps=steps,  # type: ignore[arg-type]
        )
