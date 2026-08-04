from __future__ import annotations

import math

import pytest

from reality_stone.clarus.portal_one_loop_control import (
    CONTROL_SCOPE,
    CONTROL_STATUS,
    a0_msbar_finite,
    audit_portal_one_loop_scalar_control,
    b0_derivative_below_threshold,
    b0_msbar_finite_below_threshold,
)
from reality_stone.clarus.q0_manifest_gate import (
    q0_control_action_definition_sha256,
)


def test_a0_finite_convention_and_massless_dimreg_limit() -> None:
    assert a0_msbar_finite(mass_squared_gev2=4.0, scale_squared_gev2=4.0) == 4.0
    assert a0_msbar_finite(mass_squared_gev2=0.0, scale_squared_gev2=4.0) == 0.0


def test_b0_is_symmetric_and_has_positive_derivative_below_cut() -> None:
    arguments = {
        "invariant_gev2": 0.5,
        "first_mass_squared_gev2": 4.0,
        "second_mass_squared_gev2": 1.0,
        "scale_squared_gev2": 4.0,
    }
    forward = b0_msbar_finite_below_threshold(**arguments)
    reverse = b0_msbar_finite_below_threshold(
        invariant_gev2=0.5,
        first_mass_squared_gev2=1.0,
        second_mass_squared_gev2=4.0,
        scale_squared_gev2=4.0,
    )
    derivative = b0_derivative_below_threshold(
        invariant_gev2=0.5,
        first_mass_squared_gev2=4.0,
        second_mass_squared_gev2=1.0,
    )

    assert forward == pytest.approx(reverse, abs=2.0e-11)
    assert derivative > 0.0


def test_light_target_has_large_scale_dependent_finite_mass_piece() -> None:
    audit = audit_portal_one_loop_scalar_control()

    assert audit.scope == CONTROL_SCOPE
    assert audit.status == CONTROL_STATUS
    assert audit.action_definition_sha256 == q0_control_action_definition_sha256()
    assert audit.cubic_h_phi_phi_coupling_gev == pytest.approx(15.5601, rel=2.0e-5)
    assert audit.higgs_tadpole_finite_gev2 == pytest.approx(3.13903, rel=2.0e-5)
    assert audit.mixed_bubble_finite_gev2 == pytest.approx(1.53322, rel=2.0e-5)
    assert audit.portal_only_finite_sum_gev2 == pytest.approx(4.67225, rel=2.0e-5)
    assert audit.singlet_tadpole_finite_gev2 == pytest.approx(2.955e-5, rel=2.0e-3)
    assert audit.finite_sum_to_target_mass_squared_ratio > 5300.0
    assert audit.target_mass_squared_to_finite_sum_ratio < 1.9e-4
    assert not audit.raw_finite_piece_small_against_light_target
    assert audit.half_scale_finite_sum_gev2 < 0.0
    assert audit.double_scale_finite_sum_gev2 > audit.scalar_truncation_finite_sum_gev2
    assert audit.finite_sum_changes_sign_over_scale_holdout


def test_light_target_is_below_mixed_cut_with_small_residue_correction() -> None:
    audit = audit_portal_one_loop_scalar_control()

    assert audit.first_mixed_cut_energy_gev == pytest.approx(125.27964757)
    assert audit.below_first_mixed_cut
    assert audit.mixed_bubble_imaginary_part_gev2 == 0.0
    assert audit.bubble_self_energy_derivative == pytest.approx(4.8868e-5, rel=3.0e-4)
    assert 0.9999 < audit.linearized_residue_control < 1.0
    assert audit.perturbative_coupling_control_pass
    assert audit.scalar_loop_expansion_parameter < 2.1e-4


def test_finite_control_never_self_certifies_a_physical_ce_pole() -> None:
    audit = audit_portal_one_loop_scalar_control()
    payload = audit.to_dict()

    assert not audit.counterterm_basis_included
    assert not audit.renormalization_condition_supplied
    assert not audit.full_gauge_goldstone_sector_included
    assert not audit.rg_improvement_included
    assert not audit.renormalized_pole_mass_predicted
    assert not audit.spectral_density_derived
    assert not audit.lsz_particle_derived
    assert not audit.ce_field_identity_derived
    assert payload["renormalized_pole_mass_predicted"] is False
    assert "radiative tuning" in audit.conclusion


def test_b0_rejects_a_point_on_or_above_the_branch_cut() -> None:
    threshold = (2.0 + 1.0) ** 2
    with pytest.raises(ValueError, match="below the two-particle threshold"):
        b0_msbar_finite_below_threshold(
            invariant_gev2=threshold,
            first_mass_squared_gev2=4.0,
            second_mass_squared_gev2=1.0,
            scale_squared_gev2=4.0,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("target_mass_gev", 0.0, "positive"),
        ("lambda_hp", -0.1, "nonnegative"),
        ("lambda_phi", math.inf, "finite"),
        ("higgs_vev_gev", True, "real scalar"),
        ("renormalization_scale_gev", -1.0, "positive"),
    ],
)
def test_one_loop_control_rejects_invalid_inputs(
    keyword: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {}
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        audit_portal_one_loop_scalar_control(**arguments)
