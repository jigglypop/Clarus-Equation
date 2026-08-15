"""Machine-readable, fail-closed gate for CE cosmology closure.

The default command audits status and therefore exits zero when the audit ran
successfully. It does *not* mean that the cosmology is physically closed. Use
``--require-physical-closure`` for a scientific/release gate; that mode exits
2 while any required bridge or independent holdout remains incomplete.

Historical numerical reproductions are excluded by default and appear only
with ``--historical-reproduction``. Their values are compatibility evidence,
never blind confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Sequence

try:
    from examples.physics.ce_residual_forward_model import (
        CEForwardParams,
        parameter_provenance,
    )
    from examples.physics.cosmological_constant_holographic_gate import (
        HISTORICAL_HOLOGRAPHIC_MODEL_ID,
        OMEGA_LAMBDA,
        RHO_LAMBDA_OBS_MEV,
        derive_entropy,
        h_lambda_over_h0,
        horizon_scale_definitions,
        rel_error,
        rho_lambda_quarter_mev,
        true_de_sitter_vacuum_quarter_mev,
    )
    from examples.physics.hubble_tension import (
        HISTORICAL_H0_TOY_MODEL_ID,
        HISTORICAL_H0_TOY_PHYSICAL_CLOSURE,
        exact_flrw_ricci_over_h2,
        historical_h0_toy_input_activity,
        historical_matter_lambda_ricci_over_h2,
    )
    from examples.physics.primordial_spectrum_readout_gate import (
        HISTORICAL_PRIMORDIAL_PROJECTOR_MODEL_ID,
        readouts,
    )
except ModuleNotFoundError:  # Direct execution from examples/physics.
    from ce_residual_forward_model import CEForwardParams, parameter_provenance
    from cosmological_constant_holographic_gate import (
        HISTORICAL_HOLOGRAPHIC_MODEL_ID,
        OMEGA_LAMBDA,
        RHO_LAMBDA_OBS_MEV,
        derive_entropy,
        h_lambda_over_h0,
        horizon_scale_definitions,
        rel_error,
        rho_lambda_quarter_mev,
        true_de_sitter_vacuum_quarter_mev,
    )
    from hubble_tension import (
        HISTORICAL_H0_TOY_MODEL_ID,
        HISTORICAL_H0_TOY_PHYSICAL_CLOSURE,
        exact_flrw_ricci_over_h2,
        historical_h0_toy_input_activity,
        historical_matter_lambda_ricci_over_h2,
    )
    from primordial_spectrum_readout_gate import (
        HISTORICAL_PRIMORDIAL_PROJECTOR_MODEL_ID,
        readouts,
    )


SCHEMA_VERSION = "ce.cosmology.closure-gate.v1"
GATE_ID = "CE_COSMOLOGY_FAIL_CLOSED_V1"
INCOMPLETE_EXIT_CODE = 2
TARGET_INCOMPLETE = "[미완성]"
INDEPENDENT_CONFIRMATORY_HOLDOUTS = 0


def _check(check_id: str, status: str, claim: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status,
        "claim": claim,
        "evidence": evidence,
    }


def _historical_reproductions() -> dict[str, Any]:
    entropy = derive_entropy()
    legacy_lambda_mev = rho_lambda_quarter_mev(entropy["log_s"], OMEGA_LAMBDA)
    true_ds_mev = true_de_sitter_vacuum_quarter_mev(entropy["log_s"])
    return {
        "status": "HISTORICAL_REPRODUCTION_ONLY",
        "counts_as_physical_closure": False,
        "counts_as_blind_confirmation": False,
        "h0_theta_toy": {
            "model_id": HISTORICAL_H0_TOY_MODEL_ID,
            "input_activity": historical_h0_toy_input_activity(),
            "physical_closure": HISTORICAL_H0_TOY_PHYSICAL_CLOSURE,
            "note": "numeric route retained; om_b_h2 is inactive",
        },
        "holographic_scale": {
            "model_id": HISTORICAL_HOLOGRAPHIC_MODEL_ID,
            "legacy_mixed_readout_mev": legacy_lambda_mev,
            "true_de_sitter_readout_mev": true_ds_mev,
            "observational_comparison_mev": RHO_LAMBDA_OBS_MEV,
            "legacy_relative_error_percent": rel_error(
                legacy_lambda_mev, RHO_LAMBDA_OBS_MEV
            ),
            "target_aware": True,
            "physical_closure": False,
        },
        "primordial_projector": {
            "model_id": HISTORICAL_PRIMORDIAL_PROJECTOR_MODEL_ID,
            "target_aware": True,
            "physical_closure": False,
            "readouts": [
                {
                    "name": item.name,
                    "legacy_fit_status": item.status,
                    "closure_status": item.closure_status,
                    "as_1e9": item.as_1e9,
                    "qualifies_as_physical_prediction": (
                        item.qualifies_as_physical_prediction
                    ),
                }
                for item in readouts()
            ],
        },
    }


def build_audit(*, include_historical_reproduction: bool = False) -> dict[str, Any]:
    """Build the complete audit record without deciding a process exit code."""
    ricci_limits = {
        "radiation": exact_flrw_ricci_over_h2(0.0, 1.0),
        "matter": exact_flrw_ricci_over_h2(1.0, 0.0),
        "de_sitter": exact_flrw_ricci_over_h2(0.0, 0.0),
    }
    ricci_limits_ok = ricci_limits == {"radiation": 0.0, "matter": 3.0, "de_sitter": 12.0}
    legacy_radiation_witness = historical_matter_lambda_ricci_over_h2(0.0)
    legacy_scoped = legacy_radiation_witness == 12.0 and ricci_limits["radiation"] == 0.0

    h0_activity = historical_h0_toy_input_activity()
    h0_toy_closed = HISTORICAL_H0_TOY_PHYSICAL_CLOSURE and all(h0_activity.values())

    scale_definitions = horizon_scale_definitions()
    scale_names_ok = set(scale_definitions) == {"H_L", "H_*", "H0"}
    epochs = {entry["epoch"] for entry in scale_definitions.values()}
    scale_epochs_distinct = len(epochs) == 3
    h_l_ratio = h_lambda_over_h0(OMEGA_LAMBDA)
    mixed_epoch_not_identity = not math.isclose(h_l_ratio, 1.0, rel_tol=0.0, abs_tol=1e-15)

    primordial = readouts()
    primordial_fail_closed = all(
        not item.qualifies_as_physical_prediction for item in primordial
    ) and all(item.closure_status != "physical_prediction" for item in primordial)

    provenance = parameter_provenance(CEForwardParams())
    density_names = {"omega_b0", "omega_dm0", "omega_lambda0"}
    density_provenance = [entry for entry in provenance if entry.name in density_names]
    provenance_fail_closed = len(density_provenance) == 3 and all(
        entry.closure_role == "legacy_model_boundary"
        and not entry.qualifies_as_physical_prediction
        for entry in density_provenance
    )

    target_hypotheses = {
        "T-U5-H0": TARGET_INCOMPLETE,
        "T-U6-PRIM": TARGET_INCOMPLETE,
        "T-U6-LAMBDA": TARGET_INCOMPLETE,
        "T-U7-PROV": TARGET_INCOMPLETE,
        "T-U8-INTEGRATE": TARGET_INCOMPLETE,
    }
    targets_preserved = all(status == TARGET_INCOMPLETE for status in target_hypotheses.values())

    checks = [
        _check(
            "U4_EXACT_RICCI_RADIATION",
            "PASS" if ricci_limits_ok else "ERROR",
            "Exact static FLRW Ricci helper includes radiation.",
            {"R_over_H2_limits": ricci_limits, "dimension": "dimensionless"},
        ),
        _check(
            "U4_LEGACY_RICCI_SCOPED",
            "CLOSED_NARROWED" if legacy_scoped else "ERROR",
            "Radiation-free Ricci formula is named historical only.",
            {
                "legacy_pure_radiation_witness": legacy_radiation_witness,
                "model_scope": HISTORICAL_H0_TOY_MODEL_ID,
            },
        ),
        _check(
            "U5_H0_TOY_INPUT_ACTIVITY",
            "CLOSED_NARROWED" if not h0_toy_closed else "ERROR",
            "Legacy theta toy cannot qualify as baryon-aware physical H0 inference.",
            {"input_activity": h0_activity, "physical_closure": False},
        ),
        _check(
            "U5_PHYSICAL_FORWARD_LIKELIHOOD",
            "INCOMPLETE",
            "A released full CLASS/CAMB likelihood route has not been executed.",
            {"target_status": TARGET_INCOMPLETE, "posterior_available": False},
        ),
        _check(
            "U6_HORIZON_EPOCH_SEPARATION",
            "PASS" if scale_names_ok and scale_epochs_distinct and mixed_epoch_not_identity else "ERROR",
            "H_L, H_*, and H0 are distinct definitions and epochs.",
            {
                "definitions": scale_definitions,
                "H_L_over_H0_for_historical_OmegaLambda": h_l_ratio,
            },
        ),
        _check(
            "U6_TRUE_DS_EQUALS_H0",
            "CLOSED_EXCLUDED" if mixed_epoch_not_identity else "ERROR",
            "True-dS H_L may not be silently identified with present H0.",
            {"Omega_Lambda": OMEGA_LAMBDA, "H_L_over_H0": h_l_ratio},
        ),
        _check(
            "U6_PRIMORDIAL_PROJECTOR",
            "CLOSED_NARROWED" if primordial_fail_closed else "ERROR",
            "Numerically close projected readouts remain target-aware candidates.",
            {
                "legacy_fit_statuses": [item.status for item in primordial],
                "closure_statuses": [item.closure_status for item in primordial],
                "blind_prediction": False,
            },
        ),
        _check(
            "U7_CONFIRMATORY_HOLDOUT",
            "INCOMPLETE",
            "No qualifying independent confirmatory holdout is frozen.",
            {"qualifying_holdout_count": INDEPENDENT_CONFIRMATORY_HOLDOUTS},
        ),
        _check(
            "U8_PARAMETER_PROVENANCE",
            "PASS" if provenance_fail_closed else "ERROR",
            "Legacy ce_prediction strings are compatibility fields, not closure roles.",
            {
                "density_entries": [
                    {
                        "name": entry.name,
                        "legacy_role": entry.role,
                        "closure_role": entry.closure_role,
                        "qualifies_as_physical_prediction": (
                            entry.qualifies_as_physical_prediction
                        ),
                    }
                    for entry in density_provenance
                ]
            },
        ),
        _check(
            "TARGET_HYPOTHESES_PRESERVED",
            "PASS" if targets_preserved else "ERROR",
            "Failed routes do not delete or lower the CE target hypotheses.",
            {"targets": target_hypotheses},
        ),
    ]

    audit_ok = all(check["status"] != "ERROR" for check in checks)
    blockers = [
        {
            "id": "U5_FULL_LIKELIHOOD_MISSING",
            "status": TARGET_INCOMPLETE,
            "reopen_when": "all physical inputs active in a released likelihood with covariance",
        },
        {
            "id": "U6_PRIMORDIAL_ACTION_SCALE_REHEATING_MISSING",
            "status": TARGET_INCOMPLETE,
            "reopen_when": "one action jointly derives A_s, n_s, r without target selection",
        },
        {
            "id": "U6_VACUUM_BRANCH_EPOCH_BRIDGE_MISSING",
            "status": TARGET_INCOMPLETE,
            "reopen_when": "unique branch and horizon epoch bridge derive the absolute scale",
        },
        {
            "id": "U7_INDEPENDENT_HOLDOUT_MISSING",
            "status": TARGET_INCOMPLETE,
            "reopen_when": "a preregistered independent holdout is frozen before release",
        },
    ]
    physical_ready = audit_ok and not blockers and INDEPENDENT_CONFIRMATORY_HOLDOUTS > 0

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "audit_execution": "PASS" if audit_ok else "ERROR",
        "formal_status_gate": "PASS" if audit_ok else "REVISE",
        "physical_closure": {
            "status": "COMPLETE" if physical_ready else "INCOMPLETE",
            "ready": physical_ready,
            "blockers": blockers,
        },
        "observational_prediction_confirmation": "NONE",
        "release_gate": "READY" if physical_ready else "NOT_READY",
        "target_hypotheses": target_hypotheses,
        "checks": checks,
        "exit_semantics": {
            "default_exit_0": "audit executed; not a claim of physical closure",
            "require_physical_closure_incomplete": INCOMPLETE_EXIT_CODE,
        },
        "historical_reproduction": {
            "requested": include_historical_reproduction,
            "status": "EXCLUDED_BY_DEFAULT",
        },
    }
    if include_historical_reproduction:
        report["historical_reproduction"] = {
            "requested": True,
            **_historical_reproductions(),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cosmology_closure_gate")
    parser.add_argument(
        "--require-physical-closure",
        action="store_true",
        help="exit 2 unless every physical bridge and blind holdout is complete",
    )
    parser.add_argument(
        "--historical-reproduction",
        action="store_true",
        help="include explicitly non-confirmatory historical numerical snapshots",
    )
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    args = parser.parse_args(argv)

    report = build_audit(
        include_historical_reproduction=args.historical_reproduction,
    )
    report["mode"] = (
        "require_physical_closure" if args.require_physical_closure else "audit"
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )

    if report["audit_execution"] != "PASS":
        return 1
    if args.require_physical_closure and not report["physical_closure"]["ready"]:
        return INCOMPLETE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
