from __future__ import annotations

import pytest

from reality_stone.clarus.fusion_direct_scattering_loop import (
    audit_direct_nuclear_scattering,
)


def test_direct_scalar_produces_resolvable_free_born_control() -> None:
    audit = audit_direct_nuclear_scattering()

    assert audit.required_nucleon_coupling == pytest.approx(0.0174265, rel=3.0e-6)
    assert audit.scalar_range_fm == pytest.approx(6.65576, rel=2.0e-6)
    assert audit.free_born_scattering_length_shift_fm < 0.0
    assert abs(audit.free_born_scattering_length_shift_fm) == pytest.approx(
        0.00509035,
        rel=4.0e-4,
    )
    assert audit.born_shift_to_triplet_uncertainty > 3.0
    assert audit.born_shift_to_singlet_uncertainty > 1.0
    assert audit.free_born_shift_resolved_by_reported_precision


def test_hulthen_control_has_kev_scale_binding_shift_but_is_not_an_exclusion() -> None:
    audit = audit_direct_nuclear_scattering()

    assert audit.deuteron_binding_momentum_fm_inv == pytest.approx(0.2316, rel=5.0e-4)
    assert audit.hulthen_normalization_fm_inv_sqrt > 0.8
    assert audit.hulthen_yukawa_expectation_mev < 0.0
    assert audit.absolute_deuteron_shift_kev == pytest.approx(2.077, rel=3.0e-3)
    assert audit.deuteron_shift_to_binding_fraction > 9.0e-4
    assert not audit.strong_potential_refit_performed
    assert not audit.distorted_wave_born_calculation_performed
    assert not audit.few_body_binding_refit_performed
    assert not audit.experimental_exclusion_derived
    assert not audit.physical_direct_operator_gate_pass


def test_direct_scattering_audit_is_deterministic() -> None:
    assert audit_direct_nuclear_scattering() == audit_direct_nuclear_scattering()
