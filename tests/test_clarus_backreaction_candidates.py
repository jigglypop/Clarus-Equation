from __future__ import annotations

import math

import pytest

from reality_stone.clarus.clarus_backreaction_candidates import (
    ideal_casimir_zero_redshift_match_audit,
    vacuum_polarization_scale_audit,
)


def test_ideal_casimir_matches_rho_and_radial_pressure_but_not_tangential() -> None:
    audit = ideal_casimir_zero_redshift_match_audit()

    assert math.isclose(audit.shape_derivative_from_radial_match, -1.0 / 3.0)
    assert math.isclose(audit.residual_rho_over_scale, 0.0)
    assert math.isclose(audit.residual_radial_pressure_over_scale, 0.0)
    assert math.isclose(audit.residual_tangential_pressure_over_scale, 1.0 / 3.0)
    assert not audit.exact_zero_redshift_tensor_match
    assert audit.auxiliary_pressure_source_required
    assert not audit.conserved_global_solution_derived


def test_ce_scale_massive_vacuum_polarization_is_tiny_at_one_meter() -> None:
    audit = vacuum_polarization_scale_audit(
        throat_radius_m=1.0,
        field_correlation_length_m=6.65e-15,
    )

    assert 1.50e14 < audit.large_mass_expansion_parameter < 1.51e14
    assert audit.large_mass_control_applicable
    assert audit.backreaction_ratio < 1e-96
    assert audit.multiplicity_required > 1e96
    assert not audit.order_one_backreaction_reached
    assert not audit.exact_renormalized_stress_derived


def test_large_mass_control_is_not_applicable_below_correlation_length() -> None:
    audit = vacuum_polarization_scale_audit(
        throat_radius_m=1e-15,
        field_correlation_length_m=6.65e-15,
    )

    assert not audit.large_mass_control_applicable


@pytest.mark.parametrize(
    ("radius", "correlation"),
    [(0.0, 1.0), (1.0, 0.0)],
)
def test_vacuum_scale_rejects_nonpositive_lengths(
    radius: float,
    correlation: float,
) -> None:
    with pytest.raises(ValueError):
        vacuum_polarization_scale_audit(
            throat_radius_m=radius,
            field_correlation_length_m=correlation,
        )
