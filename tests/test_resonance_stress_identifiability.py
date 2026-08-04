from __future__ import annotations

import math

import pytest

from reality_stone.clarus.resonance_stress_identifiability import (
    ce_resonance_bridge_audit,
    pole_family_countermodel_audit,
)


def test_same_correlation_length_allows_incompatible_stress_scalings() -> None:
    audit = pole_family_countermodel_audit(
        resonance_q=10.0,
        requested_stress_exponents=(0.0, 1.0, 2.0),
    )

    assert audit.all_countermodels_have_same_correlation_length
    assert not audit.stress_scaling_unique_from_correlation_length
    assert [model.residue_gain for model in audit.countermodels] == [1e4, 1e5, 1e6]
    assert [model.dimensional_stress_proxy_gain for model in audit.countermodels] == [
        1.0,
        10.0,
        100.0,
    ]
    assert not audit.physical_null_stress_derived


def test_ce_required_coherence_q_still_leaves_p_underdetermined() -> None:
    q_value = 1.0 / 6.65e-15
    audit = pole_family_countermodel_audit(resonance_q=q_value)

    assert math.isclose(audit.resonance_q, 1.5037593984962406e14)
    assert audit.pole_residue_required
    assert audit.spectral_density_required
    assert audit.renormalization_required
    assert not audit.stress_scaling_unique_from_correlation_length


def test_requesting_one_exponent_does_not_make_stress_scaling_unique() -> None:
    audit = pole_family_countermodel_audit(
        resonance_q=10.0,
        requested_stress_exponents=(0.0,),
    )

    assert len(audit.countermodels) == 1
    assert not audit.stress_scaling_unique_from_correlation_length
    assert not audit.physical_null_stress_derived


def test_documented_ce_bridge_stops_before_physical_pole_and_stress() -> None:
    audit = ce_resonance_bridge_audit()

    assert audit.correlation_length_ansatz_documented
    assert not audit.correlation_length_scaling_derived
    assert not audit.isolated_positive_pole_derived
    assert not audit.pole_residue_scaling_derived
    assert not audit.spectral_density_derived
    assert not audit.renormalized_stress_tensor_derived
    assert not audit.metric_backreaction_solved
    assert audit.maximum_supported_stage == "KINEMATIC_CORRELATION_ANSATZ"


@pytest.mark.parametrize("q_value", [1.0, 0.0, math.inf])
def test_countermodel_rejects_invalid_resonance_q(q_value: float) -> None:
    with pytest.raises(ValueError):
        pole_family_countermodel_audit(resonance_q=q_value)
