from __future__ import annotations

import math

import pytest

from reality_stone.clarus.a1_q0_action_bridge import (
    A1_Q0_SCOPE,
    CONDITIONAL_PASS,
    a1_q0_action_report,
    audit_higgs_invisible_width,
    audit_nonlinear_hessian_transform,
    audit_z2_higgs_portal,
)


def test_ordinary_hessian_has_non_tensor_term_away_from_stationarity() -> None:
    audit = audit_nonlinear_hessian_transform(
        action_gradient_x=3.0,
        action_hessian_x=5.0,
        dx_dy=2.0,
        d2x_dy2=4.0,
    )

    assert audit.tensor_pullback_hessian_y == pytest.approx(20.0)
    assert audit.non_tensor_extra_term == pytest.approx(12.0)
    assert audit.ordinary_hessian_y == pytest.approx(32.0)
    assert not audit.stationary
    assert not audit.ordinary_tensorial


def test_induced_connection_cancels_the_non_tensor_term() -> None:
    audit = audit_nonlinear_hessian_transform(
        action_gradient_x=3.0,
        action_hessian_x=5.0,
        dx_dy=2.0,
        d2x_dy2=4.0,
    )

    assert audit.induced_connection_y == pytest.approx(2.0)
    assert audit.action_gradient_y == pytest.approx(6.0)
    assert audit.connection_correction == pytest.approx(12.0)
    assert audit.covariant_hessian_y == pytest.approx(20.0)
    assert audit.covariance_residual < 1.0e-12
    assert audit.covariant_tensorial
    assert audit.structural_pass


def test_stationary_point_removes_the_ordinary_hessian_extra_term() -> None:
    audit = audit_nonlinear_hessian_transform(
        action_gradient_x=0.0,
        action_hessian_x=5.0,
        dx_dy=2.0,
        d2x_dy2=4.0,
    )

    assert audit.stationary
    assert audit.non_tensor_extra_term == 0.0
    assert audit.ordinary_hessian_y == pytest.approx(
        audit.tensor_pullback_hessian_y
    )
    assert audit.ordinary_tensorial
    assert audit.covariant_tensorial


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("dx_dy", 0.0, "dx_dy must be nonzero"),
        ("d2x_dy2", 0.0, "locally nonlinear"),
        ("action_gradient_x", math.inf, "must be finite"),
        ("action_hessian_x", 1.0 + 2.0j, "must be real"),
    ],
)
def test_hessian_audit_rejects_invalid_local_coordinate_data(
    keyword: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "action_gradient_x": 1.0,
        "action_hessian_x": 2.0,
        "dx_dy": 1.0,
        "d2x_dy2": 1.0,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        audit_nonlinear_hessian_transform(**arguments)  # type: ignore[arg-type]


def test_z2_portal_vacuum_derivatives_match_exact_identities() -> None:
    coupling = 0.13
    vev = 246.0

    audit = audit_z2_higgs_portal(lambda_hp=coupling, higgs_vev=vev)

    assert audit.h_phi_cross_hessian == 0.0
    assert audit.phi_mass_shift == pytest.approx(coupling * vev**2)
    assert audit.h_phi_phi_cubic == pytest.approx(2.0 * coupling * vev)
    assert audit.h_h_phi_phi_quartic == pytest.approx(2.0 * coupling)
    assert audit.cross_hessian_zero
    assert audit.algebraic_pass


def test_zero_cross_hessian_does_not_imply_zero_portal_interactions() -> None:
    audit = audit_z2_higgs_portal(lambda_hp=0.2, higgs_vev=10.0)

    assert audit.h_phi_cross_hessian == 0.0
    assert audit.h_phi_phi_cubic == pytest.approx(4.0)
    assert audit.h_h_phi_phi_quartic == pytest.approx(0.4)


def test_legacy_portal_benchmark_fails_supplied_invisible_width_limit() -> None:
    audit = audit_higgs_invisible_width(
        lambda_hp=0.0316,
        higgs_vev=246.22,
        higgs_mass=125.25,
        scalar_mass=43.77,
        sm_higgs_width=0.00407,
        branching_fraction_upper_limit=0.11,
    )

    assert audit.kinematically_open
    assert audit.partial_width == pytest.approx(0.013754011, rel=1.0e-7)
    assert audit.branching_fraction == pytest.approx(0.771656, rel=1.0e-6)
    assert not audit.benchmark_allowed


def test_closed_portal_channel_has_zero_invisible_width() -> None:
    audit = audit_higgs_invisible_width(
        lambda_hp=0.0316,
        higgs_vev=246.22,
        higgs_mass=125.25,
        scalar_mass=70.0,
        sm_higgs_width=0.00407,
        branching_fraction_upper_limit=0.11,
    )

    assert not audit.kinematically_open
    assert audit.phase_space_factor == 0.0
    assert audit.partial_width == 0.0
    assert audit.branching_fraction == 0.0
    assert audit.benchmark_allowed


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("lambda_hp", math.nan, "must be finite"),
        ("higgs_vev", 1.0j, "must be real"),
        ("tolerance", 0.0, "must be positive"),
    ],
)
def test_portal_audit_rejects_invalid_inputs(
    keyword: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "lambda_hp": 0.1,
        "higgs_vev": 246.0,
        "tolerance": 1.0e-12,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        audit_z2_higgs_portal(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("higgs_mass", 0.0, "higgs_mass must be positive"),
        ("scalar_mass", -1.0, "scalar_mass must be nonnegative"),
        ("sm_higgs_width", 0.0, "sm_higgs_width must be positive"),
        (
            "branching_fraction_upper_limit",
            1.1,
            "must be between 0 and 1",
        ),
    ],
)
def test_invisible_width_audit_rejects_invalid_inputs(
    keyword: str,
    value: float,
    message: str,
) -> None:
    arguments = {
        "lambda_hp": 0.0316,
        "higgs_vev": 246.22,
        "higgs_mass": 125.25,
        "scalar_mass": 43.77,
        "sm_higgs_width": 0.00407,
        "branching_fraction_upper_limit": 0.11,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        audit_higgs_invisible_width(**arguments)


def test_report_is_conditional_and_locks_physical_completion_flags() -> None:
    report = a1_q0_action_report(
        action_gradient_x=3.0,
        action_hessian_x=5.0,
        dx_dy=2.0,
        d2x_dy2=4.0,
        lambda_hp=0.13,
        higgs_vev=246.0,
    )
    payload = report.to_dict()

    assert report.scope == A1_Q0_SCOPE
    assert report.conditional_status == CONDITIONAL_PASS
    assert not report.covariant_action_complete
    assert not report.stress_tensor_derived
    assert not report.spectral_density_derived
    assert payload["covariant_action_complete"] is False
    assert "not a covariant CE+SM action" in report.conclusion
    assert any(
        "gauge fixing" in assumption
        for assumption in report.assumptions_not_audited
    )
