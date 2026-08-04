from __future__ import annotations

import math

import pytest

from reality_stone.clarus.multimode_global_throat import (
    global_anisotropic_target_audit,
    multimode_target_fit_audit,
)


def test_global_target_exactly_matches_casimir_at_throat() -> None:
    audit = global_anisotropic_target_audit()

    assert math.isclose(audit.throat_density, -1.0 / 3.0)
    assert math.isclose(audit.throat_radial_pressure, -1.0)
    assert math.isclose(audit.throat_tangential_pressure, 1.0 / 3.0)
    assert audit.throat_matches_ideal_casimir
    assert audit.flare_out_satisfied


def test_variable_anisotropy_closes_global_geometry_control_gate() -> None:
    audit = global_anisotropic_target_audit()

    assert audit.horizon_free
    assert audit.minimum_shape_gap > 0.0
    assert audit.maximum_conservation_residual < 1.0e-10
    assert audit.shape_over_radius_at_cutoff < 1.0e-3
    assert abs(audit.redshift_at_cutoff) < 1.0e-10
    assert audit.finite_adm_mass
    assert math.isclose(audit.adm_mass_in_throat_radii, 1.0 / 3.0)
    assert audit.asymptotically_flat
    assert audit.two_sided_geometric_extension_available
    assert audit.global_geometry_control_pass


def test_geometry_target_does_not_claim_the_missing_physics() -> None:
    audit = global_anisotropic_target_audit()

    assert not audit.fixed_casimir_eos_preserved_globally
    assert not audit.ce_multimode_stress_derived
    assert not audit.perturbative_stability_derived


def test_shared_spectral_modes_converge_to_the_target_tensor() -> None:
    audit = multimode_target_fit_audit()
    errors = [level.maximum_normalized_error for level in audit.levels]

    assert audit.error_decreases
    assert errors[-1] < 1.0e-6
    assert audit.finite_mode_target_approximation_pass
    assert not audit.basis_is_physical_resonator_spectrum
    assert not audit.carrier_envelope_bridge_derived
    assert not audit.quantized_negative_stress_derived


@pytest.mark.parametrize("cutoff", [1.0, 0.0, float("inf")])
def test_invalid_radial_cutoff_is_rejected(cutoff: float) -> None:
    with pytest.raises(ValueError):
        global_anisotropic_target_audit(radial_cutoff=cutoff)


def test_invalid_mode_sequence_is_rejected() -> None:
    with pytest.raises(ValueError):
        multimode_target_fit_audit(mode_counts=(8, 4))
