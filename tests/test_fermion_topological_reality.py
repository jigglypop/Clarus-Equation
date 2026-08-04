from __future__ import annotations

import pytest

from reality_stone.clarus.fermion_topological_reality import (
    charged_fermion_topology_audit,
    flux_multiplicity_control,
)


def test_one_metre_channel_has_no_effectively_massless_sm_charged_fermion() -> None:
    audit = charged_fermion_topology_audit()

    assert 1.9e-7 < audit.effectively_massless_energy_bound_ev < 2.0e-7
    assert 2.5e12 < audit.electron_to_massless_bound_ratio < 2.7e12
    assert not audit.electron_effectively_massless
    assert not audit.standard_model_macroscopic_charged_mode_available
    assert not audit.human_scale_ce_mapping_pass


def test_external_control_does_not_become_a_ce_shortcut() -> None:
    audit = charged_fermion_topology_audit()

    assert audit.negative_casimir_stress_derived_in_external_control
    assert audit.external_control_is_long_wormhole
    assert not audit.ambient_space_shortcut
    assert not audit.ce_charged_massless_fermion_specified
    assert not audit.ce_quantized_magnetic_flux_sector_specified


def test_flux_multiplicity_is_a_physical_count_not_free_quality_factor() -> None:
    failed = flux_multiplicity_control(
        wormhole_length_m=1000.0,
        magnetic_radius_m=1.0,
        flux_zero_mode_count=999,
    )
    passed = flux_multiplicity_control(
        wormhole_length_m=1000.0,
        magnetic_radius_m=1.0,
        flux_zero_mode_count=1000,
    )

    assert not failed.scale_bound_satisfied
    assert passed.scale_bound_satisfied
    assert passed.dsnec_scale_lower_bound == 1000.0
    assert not passed.exact_integer_flux_action_specified


@pytest.mark.parametrize("length", [0.0, -1.0])
def test_invalid_channel_length_is_rejected(length: float) -> None:
    with pytest.raises(ValueError):
        charged_fermion_topology_audit(channel_length_m=length)
