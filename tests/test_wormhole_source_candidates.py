from __future__ import annotations

import math

import pytest

from reality_stone.clarus.spatial_folding import casimir_cell_conversion_audit
from reality_stone.clarus.wormhole_source_candidates import (
    resonance_source_audit,
    scalar_null_energy_audit,
    source_candidate_catalog,
)


def test_linear_q_ansatz_needs_density_gap_for_one_meter_throat() -> None:
    density = casimir_cell_conversion_audit().energy_density_j_m3
    audit = resonance_source_audit(
        throat_radius_m=1.0,
        base_negative_density_j_m3=density,
        base_correlation_length_m=6.65e-15,
        density_gain_exponent=1.0,
    )

    assert 2.8e16 < audit.density_gap < 2.9e16
    assert 1.50e14 < audit.coherence_q_required < 1.51e14
    assert math.isclose(audit.combined_q_required, audit.density_q_required)
    assert audit.numerical_gates_pass_under_ansatz
    assert not audit.density_scaling_law_derived_from_ce
    assert not audit.renormalized_stress_tensor_derived


def test_quadratic_q_ansatz_is_coherence_limited_but_still_not_derived() -> None:
    density = casimir_cell_conversion_audit().energy_density_j_m3
    audit = resonance_source_audit(
        throat_radius_m=1.0,
        base_negative_density_j_m3=density,
        base_correlation_length_m=6.65e-15,
        density_gain_exponent=2.0,
    )

    assert 1.68e8 < audit.density_q_required < 1.70e8
    assert math.isclose(audit.combined_q_required, audit.coherence_q_required)
    assert audit.numerical_gates_pass_under_ansatz
    assert not audit.density_scaling_law_derived_from_ce


def test_canonical_scalar_cannot_violate_nec() -> None:
    audit = scalar_null_energy_audit(null_directional_derivative=3.0)

    assert audit.null_projection == 9.0
    assert not audit.violates_nec
    assert audit.ghost_free_kinetic_term
    assert not audit.supports_throat_and_is_ghost_free


def test_phantom_scalar_violates_nec_but_fails_ghost_gate() -> None:
    audit = scalar_null_energy_audit(
        null_directional_derivative=3.0,
        kinetic_sign=-1.0,
    )

    assert audit.null_projection == -9.0
    assert audit.violates_nec
    assert not audit.ghost_free_kinetic_term
    assert not audit.supports_throat_and_is_ghost_free


def test_catalog_keeps_literature_controls_separate_from_ce_derivation() -> None:
    catalog = source_candidate_catalog()
    by_name = {candidate.name: candidate for candidate in catalog}

    assert len(catalog) == 7
    assert (
        by_name["CE minimally coupled canonical scalar channel"].nec_or_anec_violation
        == "FAIL"
    )
    assert by_name["Gao-Jafferis-Wall control"].derived_from_ce == "NO"
    assert by_name["Maldacena-Milekhin-Popov control"].derived_from_ce == "NO"


@pytest.mark.parametrize("exponent", [0.0, -1.0])
def test_resonance_screen_rejects_nonpositive_gain_exponents(exponent: float) -> None:
    with pytest.raises(ValueError):
        resonance_source_audit(
            throat_radius_m=1.0,
            base_negative_density_j_m3=1.0,
            base_correlation_length_m=1.0,
            density_gain_exponent=exponent,
        )
