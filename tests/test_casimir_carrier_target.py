from __future__ import annotations

import math

import pytest

from reality_stone.clarus.casimir_carrier_target import (
    CURRENT_TARGET_LABEL,
    LEGACY_CONTROL_LABEL,
    exact_casimir_carrier_target,
    legacy_bprime_minus_one_null_control,
)
from reality_stone.clarus.physical_multimode_realization import (
    physical_multimode_realization_audit,
)


def test_current_full_tensor_target_is_the_152_93_gev_branch() -> None:
    target = exact_casimir_carrier_target()

    assert target.target_definition == CURRENT_TARGET_LABEL
    assert target.is_current_full_tensor_target
    assert target.shape_derivative == pytest.approx(-1.0 / 3.0)
    assert target.target_rho_over_curvature_scale == pytest.approx(-1.0 / 3.0)
    assert target.target_radial_pressure_over_curvature_scale == -1.0
    assert target.target_tangential_pressure_over_curvature_scale == pytest.approx(1.0 / 3.0)
    assert target.separation_m == pytest.approx(4.053564004319189e-18)
    assert target.wavelength_m == pytest.approx(8.107128008638377e-18)
    assert target.carrier_energy_ev == pytest.approx(1.5293233093781677e11)
    assert target.carrier_to_ce_pole_ratio == pytest.approx(5158.34285703067)
    assert target.nearest_integer_harmonic == 5158
    assert abs(target.nearest_harmonic_detuning_ev) > 1.0e6
    assert target.wavelength_equals_twice_separation_is_planar_mode_choice
    assert not target.single_mode_determines_casimir_stress
    assert not target.throat_boundary_eigenmode_derived
    assert not target.quality_factor_changes_carrier_frequency
    assert not target.harmonic_vertex_derived


def test_physical_multimode_audit_uses_the_canonical_target() -> None:
    target = exact_casimir_carrier_target()
    physical = physical_multimode_realization_audit()

    assert physical.ideal_casimir_separation_m == target.separation_m
    assert physical.fundamental_wavelength_m == target.wavelength_m
    assert physical.fundamental_energy_ev == target.carrier_energy_ev


def test_carrier_scaling_with_throat_radius_is_explicit() -> None:
    one_metre = exact_casimir_carrier_target(throat_radius_m=1.0)
    four_metres = exact_casimir_carrier_target(throat_radius_m=4.0)

    assert four_metres.separation_m == pytest.approx(2.0 * one_metre.separation_m)
    assert four_metres.wavelength_m == pytest.approx(2.0 * one_metre.wavelength_m)
    assert four_metres.carrier_energy_ev == pytest.approx(one_metre.carrier_energy_ev / 2.0)


def test_legacy_169_gev_control_cannot_masquerade_as_current_target() -> None:
    current = exact_casimir_carrier_target()
    legacy = legacy_bprime_minus_one_null_control()

    assert legacy.target_definition == LEGACY_CONTROL_LABEL
    assert not legacy.is_current_full_tensor_target
    assert legacy.shape_derivative == -1.0
    assert legacy.target_rho_over_curvature_scale is None
    assert 1.68e11 < legacy.carrier_energy_ev < 1.70e11
    assert legacy.nearest_integer_harmonic == 5709
    assert legacy.carrier_energy_ev != pytest.approx(current.carrier_energy_ev)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"throat_radius_m": True},
        {"throat_radius_m": 0.0},
        {"throat_radius_m": -1.0},
        {"throat_radius_m": math.inf},
        {"ce_pole_energy_mev": math.nan},
    ],
)
def test_target_rejects_adversarial_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        exact_casimir_carrier_target(**kwargs)
