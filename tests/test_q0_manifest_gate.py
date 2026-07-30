from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from reality_stone.clarus.q0_manifest_gate import (
    ACTION_CONVENTION,
    COVARIANT_DERIVATIVE_CONVENTION,
    GAUGE_TRANSFORMATION_CONVENTION,
    GHOST_CONVENTION,
    R_XI_CONVENTION,
    SIGNATURE_CONVENTION,
    TOY_CONDITIONAL_PASS,
    TOY_SCOPE,
    Q0ControlInputs,
    Q0StructuralManifest,
    audit_abelian_higgs_r_xi,
    audit_background_tadpole,
    audit_field_space_local_jet,
    audit_q0_manifest,
    load_q0_control_benchmark,
    q0_manifest_gate_report,
)


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "q0_minimal_abelian_higgs_v1.json"
)


@pytest.fixture
def benchmark():
    return load_q0_control_benchmark(BENCHMARK)


def test_fixed_benchmark_loads_and_manifest_is_complete(benchmark) -> None:
    audit = audit_q0_manifest(benchmark.manifest)

    assert benchmark.manifest.scope_id == TOY_SCOPE
    assert benchmark.manifest.spacetime_signature == SIGNATURE_CONVENTION
    assert benchmark.manifest.action_convention == ACTION_CONVENTION
    assert (
        benchmark.manifest.covariant_derivative
        == COVARIANT_DERIVATIVE_CONVENTION
    )
    assert (
        benchmark.manifest.gauge_transformation
        == GAUGE_TRANSFORMATION_CONVENTION
    )
    assert benchmark.manifest.gauge_fixing == R_XI_CONVENTION
    assert benchmark.manifest.ghost_action == GHOST_CONVENTION
    assert not benchmark.manifest.full_ce_sm_complete
    assert audit.complete
    assert audit.excluded_sectors_explicit
    assert audit.full_claim_locked_false


def test_manifest_missing_ghost_is_incomplete(benchmark) -> None:
    manifest = replace(benchmark.manifest, ghost_action="")

    audit = audit_q0_manifest(manifest)

    assert "ghost_action" in audit.missing_sections
    assert not audit.complete


def test_manifest_rejects_scope_spoof_and_full_theory_claim(benchmark) -> None:
    manifest = replace(
        benchmark.manifest,
        scope_id="full_ce_sm",
        full_ce_sm_complete=True,
        excluded_sectors=("physical_spectral_density",),
    )

    audit = audit_q0_manifest(manifest)

    assert not audit.scope_locked
    assert not audit.excluded_sectors_explicit
    assert not audit.full_claim_locked_false
    assert {"scope_id", "excluded_sectors", "full_ce_sm_complete"}.issubset(
        audit.convention_issues
    )
    assert not audit.complete


def test_manifest_duplicate_entries_are_invalid(benchmark) -> None:
    manifest = replace(
        benchmark.manifest,
        counterterms=("delta_Z_A", "delta_Z_A"),
    )

    audit = audit_q0_manifest(manifest)

    assert "counterterms" in audit.invalid_sections
    assert not audit.complete


def test_nonlinear_coordinate_has_extra_ordinary_hessian_term() -> None:
    audit = audit_field_space_local_jet(
        action_gradient_x=3.0,
        action_hessian_x=5.0,
        dx_dy=2.0,
        d2x_dy2=4.0,
        field_metric_x=1.5,
    )

    assert audit.tensor_pullback_hessian_y == pytest.approx(20.0)
    assert audit.non_tensor_extra_term == pytest.approx(12.0)
    assert audit.ordinary_hessian_y == pytest.approx(32.0)
    assert not audit.ordinary_tensorial
    assert audit.levi_civita_connection_y == pytest.approx(2.0)
    assert audit.covariant_hessian_y == pytest.approx(20.0)
    assert audit.covariant_tensorial
    assert audit.structural_pass


def test_stationary_point_is_not_a_counterexample_to_ordinary_hessian() -> None:
    audit = audit_field_space_local_jet(
        action_gradient_x=0.0,
        action_hessian_x=5.0,
        dx_dy=2.0,
        d2x_dy2=4.0,
        field_metric_x=1.5,
    )

    assert audit.stationary
    assert audit.non_tensor_extra_term == 0.0
    assert audit.ordinary_tensorial
    assert audit.covariant_tensorial


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("dx_dy", 0.0, "dx_dy must be nonzero"),
        ("d2x_dy2", 0.0, "nonlinear local jet"),
        ("field_metric_x", 0.0, "field_metric_x must be positive"),
        ("action_gradient_x", math.inf, "must be finite"),
    ],
)
def test_field_space_audit_rejects_invalid_inputs(
    keyword: str,
    value: float,
    message: str,
) -> None:
    arguments = {
        "action_gradient_x": 3.0,
        "action_hessian_x": 5.0,
        "dx_dy": 2.0,
        "d2x_dy2": 4.0,
        "field_metric_x": 1.5,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        audit_field_space_local_jet(**arguments)


def test_background_tadpole_passes_only_on_supplied_stationary_point() -> None:
    on_shell = audit_background_tadpole(
        mu_squared=2.0,
        scalar_self_coupling=0.5,
        scalar_vev=2.0,
    )
    off_shell = audit_background_tadpole(
        mu_squared=2.1,
        scalar_self_coupling=0.5,
        scalar_vev=2.0,
    )

    assert on_shell.tadpole == pytest.approx(0.0)
    assert on_shell.goldstone_curvature == pytest.approx(0.0)
    assert on_shell.radial_curvature == pytest.approx(4.0)
    assert on_shell.on_shell_background
    assert off_shell.tadpole == pytest.approx(-0.2)
    assert not off_shell.on_shell_background


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("mu_squared", 0.0, "mu_squared must be positive"),
        ("scalar_self_coupling", -0.1, "must be positive"),
        ("scalar_vev", 0.0, "scalar_vev must be positive"),
        ("tolerance", math.nan, "must be finite"),
    ],
)
def test_background_audit_rejects_invalid_inputs(
    keyword: str,
    value: float,
    message: str,
) -> None:
    arguments = {
        "mu_squared": 2.0,
        "scalar_self_coupling": 0.5,
        "scalar_vev": 2.0,
        "tolerance": 1.0e-12,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        audit_background_tadpole(**arguments)


def test_r_xi_signs_cancel_mixing_and_match_fp_ghost_mass() -> None:
    audit = audit_abelian_higgs_r_xi(
        gauge_coupling=0.4,
        scalar_vev=2.0,
        xi=2.0,
        gauge_fixing_goldstone_coefficient=1.6,
        declared_ghost_mass_squared=1.28,
    )

    assert audit.vector_mass == pytest.approx(0.8)
    assert audit.kinetic_a_dot_dchi_coefficient == pytest.approx(0.8)
    assert audit.gauge_fixing_a_dot_dchi_coefficient == pytest.approx(-0.8)
    assert audit.net_a_dot_dchi_coefficient == pytest.approx(0.0)
    assert audit.gauge_fixing_goldstone_mass_squared == pytest.approx(1.28)
    assert audit.fp_operator_ghost_mass_squared == pytest.approx(1.28)
    assert audit.expected_r_xi_mass_squared == pytest.approx(1.28)
    assert audit.mixing_cancelled
    assert audit.fp_operator_consistent
    assert audit.goldstone_ghost_masses_match
    assert audit.structural_pass


def test_wrong_gauge_fixing_sign_is_a_structural_counterexample() -> None:
    audit = audit_abelian_higgs_r_xi(
        gauge_coupling=0.4,
        scalar_vev=2.0,
        xi=2.0,
        gauge_fixing_goldstone_coefficient=-1.6,
        declared_ghost_mass_squared=1.28,
    )

    assert audit.net_a_dot_dchi_coefficient == pytest.approx(1.6)
    assert not audit.mixing_cancelled
    assert not audit.fp_operator_consistent
    assert not audit.structural_pass


def test_wrong_declared_ghost_mass_fails_fp_identity() -> None:
    audit = audit_abelian_higgs_r_xi(
        gauge_coupling=0.4,
        scalar_vev=2.0,
        xi=2.0,
        gauge_fixing_goldstone_coefficient=1.6,
        declared_ghost_mass_squared=1.0,
    )

    assert audit.mixing_cancelled
    assert not audit.fp_operator_consistent
    assert not audit.goldstone_ghost_masses_match
    assert not audit.structural_pass


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("gauge_coupling", 0.0, "gauge_coupling must be positive"),
        ("scalar_vev", -1.0, "scalar_vev must be positive"),
        ("xi", 0.0, "xi must be positive"),
        (
            "declared_ghost_mass_squared",
            math.inf,
            "must be finite",
        ),
    ],
)
def test_r_xi_audit_rejects_invalid_inputs(
    keyword: str,
    value: float,
    message: str,
) -> None:
    arguments = {
        "gauge_coupling": 0.4,
        "scalar_vev": 2.0,
        "xi": 2.0,
        "gauge_fixing_goldstone_coefficient": 1.6,
        "declared_ghost_mass_squared": 1.28,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        audit_abelian_higgs_r_xi(**arguments)


def test_report_passes_only_control_slice_and_locks_full_claims(
    benchmark,
) -> None:
    report = q0_manifest_gate_report(
        benchmark.manifest,
        benchmark.control_inputs,
    )
    payload = report.to_dict()

    assert report.structural_status == TOY_CONDITIONAL_PASS
    assert report.control_scope == TOY_SCOPE
    assert report.control_q0_0_manifest_pass
    assert report.control_q0_1_field_space_pass
    assert report.control_q0_2_background_pass
    assert report.control_q0_3_gauge_pass
    assert report.abelian_control_slice_pass
    assert not report.q0_0_scope_complete
    assert not report.q0_1_field_space_complete
    assert not report.q0_2_background_complete
    assert not report.q0_3_gauge_complete
    assert not report.full_q0_pass
    assert not report.full_ce_sm_complete
    assert not report.stress_tensor_derived
    assert not report.spectral_density_derived
    assert payload["q0_0_scope_complete"] is False
    assert payload["q0_1_field_space_complete"] is False
    assert payload["q0_2_background_complete"] is False
    assert payload["q0_3_gauge_complete"] is False
    assert payload["full_q0_pass"] is False
    assert payload["full_ce_sm_complete"] is False
    assert report.excluded_sectors
    assert "full Q0" in report.conclusion


def test_report_flags_are_cumulative_after_background_failure(
    benchmark,
) -> None:
    inputs = replace(benchmark.control_inputs, mu_squared=2.1)

    report = q0_manifest_gate_report(benchmark.manifest, inputs)

    assert report.control_q0_0_manifest_pass
    assert report.control_q0_1_field_space_pass
    assert not report.control_q0_2_background_pass
    assert not report.control_q0_3_gauge_pass
    assert report.gauge_audit.structural_pass
    assert not report.abelian_control_slice_pass
    assert not report.full_q0_pass


def test_report_flags_are_cumulative_after_manifest_failure(
    benchmark,
) -> None:
    manifest = replace(benchmark.manifest, regularization="")

    report = q0_manifest_gate_report(manifest, benchmark.control_inputs)

    assert not report.control_q0_0_manifest_pass
    assert not report.control_q0_1_field_space_pass
    assert not report.control_q0_2_background_pass
    assert not report.control_q0_3_gauge_pass
    assert report.field_space_audit.structural_pass
    assert report.background_audit.on_shell_background
    assert report.gauge_audit.structural_pass


def test_loader_rejects_missing_required_manifest_key(tmp_path: Path) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    del payload["manifest"]["ghost_action"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.ghost_action is required"):
        load_q0_control_benchmark(path)


def test_report_rejects_wrong_input_container(benchmark) -> None:
    with pytest.raises(TypeError, match="Q0ControlInputs"):
        q0_manifest_gate_report(
            benchmark.manifest,
            {"xi": 2.0},  # type: ignore[arg-type]
        )


def test_manifest_audit_rejects_wrong_container() -> None:
    with pytest.raises(TypeError, match="Q0StructuralManifest"):
        audit_q0_manifest({})  # type: ignore[arg-type]


def test_manifest_dataclass_requires_explicit_full_completion_flag() -> None:
    fields = Q0StructuralManifest.__dataclass_fields__
    inputs = Q0ControlInputs.__dataclass_fields__

    assert "excluded_sectors" in fields
    assert "full_ce_sm_complete" in fields
    assert "declared_ghost_mass_squared" in inputs
