from __future__ import annotations

import math

import pytest

from reality_stone.clarus.spatial_folding import (
    SPEED_OF_LIGHT_M_S,
    casimir_cell_conversion_audit,
    spatial_shortcut_audit,
    wormhole_throat_audit,
)


def test_spatial_shortcut_is_effectively_superluminal_but_locally_causal() -> None:
    light_year_m = 9.4607304725808e15
    audit = spatial_shortcut_audit(light_year_m, 10.0, 0.1)

    assert audit.locally_subluminal
    assert audit.positive_proper_time
    assert audit.beats_exterior_light
    assert not audit.exactly_instantaneous
    assert math.isclose(audit.apparent_speed_over_c, 0.1 * light_year_m / 10.0)
    assert math.isclose(audit.exterior_light_time_s, light_year_m / SPEED_OF_LIGHT_M_S)


def test_ce_casimir_cell_conversion_exposes_mass_density_unit_error() -> None:
    audit = casimir_cell_conversion_audit()

    assert math.isclose(audit.energy_density_j_m3, 3.378739840408162e26, rel_tol=1e-12)
    assert math.isclose(
        audit.mass_equivalent_density_kg_m3,
        3.7593550728207355e9,
        rel_tol=1e-12,
    )
    assert audit.reported_to_correct_ratio > 9e19


def test_one_meter_control_throat_requires_nec_violation_far_above_ce_cell() -> None:
    candidate = casimir_cell_conversion_audit().energy_density_j_m3
    audit = wormhole_throat_audit(
        throat_radius_m=1.0,
        candidate_negative_density_j_m3=candidate,
    )

    assert audit.flare_out_satisfied
    assert audit.nec_violated
    assert audit.nec_energy_density_j_m3 < 0.0
    assert 2.8e16 < audit.local_density_gap < 2.9e16
    assert 1.5e14 < audit.coherence_q_required < 1.51e14
    assert not audit.candidate_meets_local_density
    assert not audit.candidate_coherent_over_throat
    assert not audit.ce_stress_tensor_bridge_derived
    assert not audit.physical_wormhole_established


def test_non_flare_out_profile_does_not_require_negative_throat_nec() -> None:
    audit = wormhole_throat_audit(
        throat_radius_m=1.0,
        shape_derivative=1.0,
        candidate_negative_density_j_m3=1.0,
    )

    assert not audit.flare_out_satisfied
    assert not audit.nec_violated
    assert audit.nec_energy_density_j_m3 == 0.0
    assert not audit.candidate_meets_local_density


@pytest.mark.parametrize(
    ("exterior", "throat", "beta"),
    [(0.0, 1.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 1.0)],
)
def test_shortcut_rejects_nonphysical_inputs(
    exterior: float,
    throat: float,
    beta: float,
) -> None:
    with pytest.raises(ValueError):
        spatial_shortcut_audit(exterior, throat, beta)
