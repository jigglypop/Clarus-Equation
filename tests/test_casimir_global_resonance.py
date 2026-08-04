from __future__ import annotations

import math

import pytest

from reality_stone.clarus.casimir_global_resonance import (
    engineered_eight_thirds_tail_audit,
    fixed_casimir_eos_asymptotic_audit,
    wavelength_resonance_audit,
)


def test_finite_redshift_tail_has_infinite_total_exotic_energy() -> None:
    audit = fixed_casimir_eos_asymptotic_audit(density_tail_power=8.0 / 3.0)

    assert audit.spatial_metric_asymptotically_flat
    assert audit.finite_redshift_at_infinity
    assert not audit.finite_total_source_energy
    assert not audit.finite_adm_mass_falloff
    assert not audit.all_global_conditions_met
    assert audit.fixed_casimir_eos_global_no_go


def test_finite_energy_tail_forces_logarithmically_divergent_redshift() -> None:
    audit = fixed_casimir_eos_asymptotic_audit(density_tail_power=4.0)

    assert audit.finite_total_source_energy
    assert audit.finite_adm_mass_falloff
    assert not audit.finite_redshift_at_infinity
    assert math.isclose(audit.redshift_log_coefficient, 1.0)
    assert not audit.all_global_conditions_met


def test_engineered_tail_closes_metric_but_not_finite_mass_gate() -> None:
    audit = engineered_eight_thirds_tail_audit()

    assert audit.throat_log_density_slope == audit.required_throat_log_density_slope
    assert audit.asymptotic_redshift_finite
    assert audit.shape_over_radius_tends_to_zero
    assert not audit.total_source_energy_finite
    assert not audit.standard_finite_mass_asymptotics


def test_required_casimir_wavelength_is_electroweak_not_ce_light_pole() -> None:
    audit = wavelength_resonance_audit(cavity_separation_m=3.662808556063564e-18)

    assert 7.3e-18 < audit.fundamental_wavelength_m < 7.4e-18
    assert 1.68e11 < audit.fundamental_quantum_energy_ev < 1.70e11
    assert 5.6e3 < audit.required_harmonic_ratio < 5.8e3
    assert not audit.same_as_ce_light_pole
    assert not audit.quality_factor_changes_carrier_frequency
    assert not audit.negative_vacuum_stress_from_driven_resonance_derived


@pytest.mark.parametrize("separation", [0.0, -1.0])
def test_wavelength_gate_rejects_nonpositive_separation(separation: float) -> None:
    with pytest.raises(ValueError):
        wavelength_resonance_audit(cavity_separation_m=separation)
