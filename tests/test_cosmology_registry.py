from __future__ import annotations

import json
import math
from pathlib import Path
import struct

from examples.physics import cosmology_ratio_audit
from reality_stone.clarus.bootstrap_solver import BootstrapSolver
from reality_stone.clarus.constants import (
    ACTIVE_RATIO,
    BACKGROUND_RATIO,
    BOOTSTRAP_CONTRACTION,
    EPSILON_SQUARED_LEGACY,
    STRUCT_RATIO,
)
from reality_stone.clarus.cosmology_registry import (
    CE_CORE_EXACT_V1,
    CE_DENSITY_LO_V1,
    CE_DENSITY_NLO_CANDIDATE_V1,
    CE_DENSITY_THREE_LAYER_APPROX_V1,
    CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1,
    CE_RESIDUAL_FLAT_LCDM_GR_V1,
    LEGACY_DELTA_5DP_V1,
    LEGACY_DIRECT_READOUT_V1,
    LEGACY_ROUNDED_RUNTIME_V1,
    ROUTE_CLAIMS,
    SCIENTIFIC_DENSITY_DEFAULT,
    FormalStatus,
    ModelStatus,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "cosmology" / "observations_v1.json"


def _binary64(value: float) -> bytes:
    return struct.pack(">d", value)


def test_exact_and_rounded_delta_chains_are_distinct_and_certified() -> None:
    exact = CE_CORE_EXACT_V1
    legacy = LEGACY_DELTA_5DP_V1

    assert exact.model_id == "CE_CORE_EXACT_V1"
    assert exact.delta == 0.17775842340997383
    assert exact.d_eff == 3.1777584234099736
    assert exact.q_ext == 0.048646719644028225
    assert exact.survival == 1.0 - exact.q_ext
    assert exact.contraction == 0.15458752312007412
    assert exact.residual <= 8.0 * math.ulp(exact.q_ext)
    assert exact.q_ext < 1.0 / exact.d_eff

    assert legacy.model_id == "LEGACY_DELTA_5DP_V1"
    assert legacy.delta == 0.17776
    assert legacy.d_eff == 3.17776
    assert math.isclose(legacy.q_ext, 0.04864663333721408, rel_tol=0.0, abs_tol=1e-16)
    assert legacy.q_ext != exact.q_ext
    assert "rounded" in legacy.fixed_point.precision

    serialized = json.loads(json.dumps(exact.fixed_point.to_dict(), sort_keys=True))
    assert serialized["model_id"] == exact.model_id
    assert serialized["absolute_residual"] == exact.residual
    assert serialized["precision"] == exact.fixed_point.precision


def test_runtime_aliases_preserve_literal_bits_and_raw_sum() -> None:
    runtime = LEGACY_ROUNDED_RUNTIME_V1

    assert _binary64(ACTIVE_RATIO) == _binary64(0.0487) == _binary64(runtime.active_ratio)
    assert _binary64(STRUCT_RATIO) == _binary64(0.2623) == _binary64(runtime.struct_ratio)
    assert _binary64(BACKGROUND_RATIO) == _binary64(0.6891) == _binary64(
        runtime.background_ratio
    )
    assert _binary64(BOOTSTRAP_CONTRACTION) == _binary64(0.155) == _binary64(
        runtime.contraction_display
    )
    assert _binary64(EPSILON_SQUARED_LEGACY) == _binary64(LEGACY_DELTA_5DP_V1.q_ext)
    assert runtime.raw_sum == 1.0001
    assert runtime.status is ModelStatus.COMPATIBILITY_ONLY


def test_raw_runtime_and_flat_normalized_background_are_not_silently_mixed() -> None:
    background = CE_RESIDUAL_FLAT_LCDM_GR_V1

    assert background.source_model_id == LEGACY_ROUNDED_RUNTIME_V1.model_id
    assert background.raw_omega_m == 0.311
    assert background.raw_omega_lambda == 0.6891
    assert background.raw_sum == 1.0001
    assert background.omega_m == 0.31096890310968905
    assert background.omega_lambda == 0.6890310968903111
    assert background.omega_m + background.omega_lambda == 1.0


def test_named_density_models_coexist_without_a_scientific_default() -> None:
    assert SCIENTIFIC_DENSITY_DEFAULT is None
    assert CE_DENSITY_LO_V1.status is ModelStatus.CONDITIONAL
    assert CE_DENSITY_LO_V1.omega_c == 0.25927170943410105
    assert CE_DENSITY_LO_V1.omega_lambda == 0.6920815709218708
    assert CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1.status is ModelStatus.HISTORICAL
    assert CE_DENSITY_THREE_LAYER_APPROX_V1.status is ModelStatus.HISTORICAL
    assert CE_DENSITY_NLO_CANDIDATE_V1.status is ModelStatus.CANDIDATE
    assert all(
        math.isclose(model.total, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for model in (
            CE_DENSITY_LO_V1,
            CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1,
            CE_DENSITY_THREE_LAYER_APPROX_V1,
            CE_DENSITY_NLO_CANDIDATE_V1,
        )
    )


def test_failed_direct_readout_route_does_not_remove_the_target() -> None:
    target = ROUTE_CLAIMS["T-U2-ABS"]

    assert target.kind == "TARGET-HYPOTHESIS"
    assert target.formal_status is FormalStatus.INCOMPLETE
    assert target.active_scientific_claim
    assert LEGACY_DIRECT_READOUT_V1.formal_status is FormalStatus.AXIOM
    assert not LEGACY_DIRECT_READOUT_V1.active_scientific_claim
    assert LEGACY_DIRECT_READOUT_V1.priority == "P0-CLOSED"


def test_bootstrap_solver_defaults_to_legacy_and_exposes_exact_route() -> None:
    legacy = BootstrapSolver()
    exact = BootstrapSolver.exact()

    q_legacy = legacy.solve(method="newton")
    q_exact = exact.solve(method="newton")
    assert math.isclose(q_legacy, LEGACY_DELTA_5DP_V1.q_ext, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(q_exact, CE_CORE_EXACT_V1.q_ext, rel_tol=0.0, abs_tol=1e-15)

    report = exact.verify_fixed_point(q_exact)
    assert report["model_id"] == "CE_CORE_EXACT_V1"
    assert report["quantity_semantics"] == "branching_extinction_probability"
    assert report["survival"] == 1.0 - report["q_ext"]
    assert report["omega_b_sigma_offset"] is None
    assert report["observation_comparison_status"] == "historical_display_only_no_covariance"


def test_observation_manifest_is_complete_and_blind_gate_fails_closed() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    required = set(manifest["provenance_policy"]["required_entry_fields"])

    assert manifest["manifest_id"] == "CE_COSMOLOGY_OBSERVATIONS_V1"
    assert manifest["blind_holdout"]["qualifying_independent_holdout_count"] == 0
    assert manifest["blind_holdout"]["gate_status"] == "NOT_READY"
    assert manifest["blind_holdout"]["fail_closed"] is True
    assert all(required <= set(entry) for entry in manifest["observations"])

    hybrid = next(
        entry
        for entry in manifest["observations"]
        if entry["observation_id"] == "Planck_ACT_SPT_combined"
    )
    assert hybrid["validity"]["status"] == "EXCLUDED_HISTORICAL"
    assert hybrid["validity"]["scientific_score_eligible"] is False
    assert hybrid["covariance"]["status"] == "NONEXISTENT_FOR_HYBRID"


def test_ratio_audit_uses_registry_and_manifest_with_legacy_numeric_parity() -> None:
    assert cosmology_ratio_audit.CE_RATIOS == {
        "omega_b": 0.0487,
        "omega_c": 0.2623,
        "omega_lambda": 0.6891,
    }
    assert tuple(item.name for item in cosmology_ratio_audit.RECENT_BASELINES) == (
        "Planck2018_base",
        "Planck_ACT_SPT_combined",
        "ACT_DR6_DESI_reported",
        "SPT3G_CMBSPA",
    )
    hybrid = cosmology_ratio_audit.RECENT_BASELINES[1]
    assert hybrid.validity_status == "EXCLUDED_HISTORICAL"
    assert not hybrid.scientific_score_eligible
    verdict = cosmology_ratio_audit.coverage_verdict()
    assert verdict.density_ratios_close
    assert not verdict.scientific_score_eligible
    assert verdict.closure_role == "exploratory_ratio_diagnostic"
