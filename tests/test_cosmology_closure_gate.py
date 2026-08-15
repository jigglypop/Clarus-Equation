import json
import math

from examples.physics.ce_residual_forward_model import CEForwardParams, parameter_provenance
from examples.physics.cosmological_constant_holographic_gate import (
    OMEGA_LAMBDA,
    h_lambda_over_h0,
    horizon_scale_definitions,
    main as holographic_main,
)
from examples.physics.cosmology_closure_gate import build_audit, main
from examples.physics.hubble_tension import (
    OMEGA_R0,
    exact_flrw_ricci_over_h2,
    exact_static_flrw_ricci_over_h2,
    historical_h0_toy_input_activity,
    historical_matter_lambda_ricci_over_h2,
    historical_omega_m_of_a,
    m_eff_over_h,
    omega_m_of_a,
    omega_r_of_a,
    static_flrw_e2,
)
from examples.physics.primordial_spectrum_readout_gate import (
    main as primordial_main,
    readouts,
)


def test_default_mode_reports_audit_success_without_claiming_closure(capsys) -> None:
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "audit"
    assert payload["audit_execution"] == "PASS"
    assert payload["formal_status_gate"] == "PASS"
    assert payload["physical_closure"]["status"] == "INCOMPLETE"
    assert not payload["physical_closure"]["ready"]
    assert payload["release_gate"] == "NOT_READY"
    assert payload["observational_prediction_confirmation"] == "NONE"
    assert payload["historical_reproduction"]["status"] == "EXCLUDED_BY_DEFAULT"


def test_physical_closure_requirement_fails_closed_with_exit_2(capsys) -> None:
    assert main(["--require-physical-closure"]) == 2
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "require_physical_closure"
    blocker_ids = {item["id"] for item in payload["physical_closure"]["blockers"]}
    assert "U5_FULL_LIKELIHOOD_MISSING" in blocker_ids
    assert "U6_PRIMORDIAL_ACTION_SCALE_REHEATING_MISSING" in blocker_ids
    assert "U6_VACUUM_BRANCH_EPOCH_BRIDGE_MISSING" in blocker_ids
    assert "U7_INDEPENDENT_HOLDOUT_MISSING" in blocker_ids


def test_historical_reproduction_is_explicit_and_nonconfirmatory(capsys) -> None:
    assert main(["--historical-reproduction"]) == 0
    payload = json.loads(capsys.readouterr().out)
    historical = payload["historical_reproduction"]

    assert historical["requested"]
    assert historical["status"] == "HISTORICAL_REPRODUCTION_ONLY"
    assert not historical["counts_as_physical_closure"]
    assert not historical["counts_as_blind_confirmation"]
    assert not historical["h0_theta_toy"]["input_activity"]["omega_b_h2"]
    assert historical["holographic_scale"]["target_aware"]
    assert historical["primordial_projector"]["target_aware"]


def test_exact_ricci_helper_has_radiation_and_legacy_formula_is_named() -> None:
    assert exact_flrw_ricci_over_h2(0.0, 1.0) == 0.0
    assert exact_flrw_ricci_over_h2(1.0, 0.0) == 3.0
    assert exact_flrw_ricci_over_h2(0.0, 0.0) == 12.0
    assert historical_matter_lambda_ricci_over_h2(0.0) == 12.0
    assert historical_h0_toy_input_activity()["omega_b_h2"] is False


def test_static_fractions_and_ricci_share_radiation_inclusive_e2() -> None:
    a = 0.01
    omega_m0 = 0.31
    omega_lambda0 = 1.0 - omega_m0 - OMEGA_R0
    expected_e2 = (
        omega_m0 * a**-3 + OMEGA_R0 * a**-4 + omega_lambda0
    )

    e2 = static_flrw_e2(a, omega_m0, omega_lambda0)
    omega_m = omega_m_of_a(a, omega_m0, omega_lambda0)
    omega_r = omega_r_of_a(a, omega_m0, omega_lambda0)
    omega_lambda = omega_lambda0 / e2
    exact_ricci = exact_static_flrw_ricci_over_h2(a, omega_m0, omega_lambda0)
    trace_form = (3.0 * omega_m0 * a**-3 + 12.0 * omega_lambda0) / e2

    assert math.isclose(e2, expected_e2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(omega_m + omega_r + omega_lambda, 1.0, abs_tol=1e-15)
    assert math.isclose(exact_ricci, trace_form, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(
        historical_matter_lambda_ricci_over_h2(omega_m) - exact_ricci,
        12.0 * omega_r,
        rel_tol=0.0,
        abs_tol=1e-14,
    )
    historical_omega_m = historical_omega_m_of_a(a, omega_m0, omega_lambda0)
    assert math.isclose(
        m_eff_over_h(a, xi=2.0, alpha=0.5, om_m0=omega_m0, om_l0=omega_lambda0),
        math.sqrt(8.0 * historical_matter_lambda_ricci_over_h2(historical_omega_m)),
        rel_tol=0.0,
        abs_tol=1e-14,
    )


def test_horizon_scales_are_distinct_definitions() -> None:
    definitions = horizon_scale_definitions()
    assert set(definitions) == {"H_L", "H_*", "H0"}
    assert len({entry["epoch"] for entry in definitions.values()}) == 3
    assert definitions["H_*"]["value_status"] == "[미완성]"
    assert math.isclose(h_lambda_over_h0(OMEGA_LAMBDA), math.sqrt(OMEGA_LAMBDA))
    assert not math.isclose(h_lambda_over_h0(OMEGA_LAMBDA), 1.0)


def test_legacy_fit_and_provenance_strings_do_not_promote_to_predictions() -> None:
    legacy_passes = [item for item in readouts() if item.status == "pass"]
    assert legacy_passes
    assert all(item.closure_status == "target_aware_candidate" for item in legacy_passes)
    assert all(not item.qualifies_as_physical_prediction for item in legacy_passes)

    provenance = {item.name: item for item in parameter_provenance(CEForwardParams())}
    for name in ("omega_b0", "omega_dm0", "omega_lambda0"):
        assert provenance[name].role == "ce_prediction"  # historical serialized field
        assert provenance[name].closure_role == "legacy_model_boundary"
        assert not provenance[name].qualifies_as_physical_prediction


def test_target_hypotheses_remain_incomplete_not_deleted() -> None:
    report = build_audit()
    assert report["target_hypotheses"]
    assert set(report["target_hypotheses"].values()) == {"[미완성]"}


def test_legacy_standalone_scripts_require_explicit_opt_in(capsys) -> None:
    assert holographic_main([]) == 2
    assert "--historical-reproduction" in capsys.readouterr().out
    assert primordial_main([]) == 2
    assert "--historical-reproduction" in capsys.readouterr().out
