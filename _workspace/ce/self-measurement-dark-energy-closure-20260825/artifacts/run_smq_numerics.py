from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time


STAGING = Path(__file__).resolve().parents[4] / ".tmp" / "ce-cosmo-dso-20260825"
sys.path.insert(0, str(STAGING / "src"))

from ce_cosmo.gates.self_measurement_quintessence import (  # noqa: E402
    B_DESI,
    B_PLANCK,
    SMQConfig,
    exponential_fixed_point,
    scan_desi_lambda,
    solve_background,
)


def profile_record(profile: object) -> dict[str, float | int]:
    return {
        "lambda": profile.lambda_,
        "scale_c_over_h0_rd": profile.scale,
        "chi2": profile.chi2,
        "dof": profile.dof,
        "p_value": profile.p_value,
        "aic": profile.aic,
        "bic": profile.bic,
        "profile_derivative": profile.profile_derivative,
    }


def boundary_record(boundary: object) -> dict[str, object]:
    started = time.perf_counter()
    scan = scan_desi_lambda(boundary, grid_step=0.01, steps=800)
    best_trajectory = solve_background(
        SMQConfig(boundary, scan.best.lambda_, steps=1600)
    )
    fixed = exponential_fixed_point(scan.best.lambda_)
    neighbors = sorted(scan.profiles, key=lambda item: item.chi2)[:5]
    return {
        "boundary_label": boundary.label,
        "omega_m0": boundary.omega_m0,
        "omega_r0": boundary.omega_r0,
        "omega_phi0_external_flat_target": boundary.omega_phi0,
        "grid_step": scan.grid_step,
        "grid_count_finite_branch": len(scan.profiles),
        "baseline": profile_record(scan.baseline),
        "best": profile_record(scan.best),
        "delta_best_minus_baseline": {
            "chi2": scan.best.chi2 - scan.baseline.chi2,
            "aic": scan.best.aic - scan.baseline.aic,
            "bic": scan.best.bic - scan.baseline.bic,
        },
        "nearest_five": [profile_record(item) for item in neighbors],
        "best_background": {
            "dimensionless_amplitude_A": best_trajectory.amplitude,
            "theta0": best_trajectory.theta_at_z(0.0),
            "w_phi0": best_trajectory.w_phi_at_z(0.0),
            "w_phi_z1": best_trajectory.w_phi_at_z(1.0),
            "omega_phi0": best_trajectory.omega_phi_at_z(0.0),
            "growth_D_z0_5": best_trajectory.growth_at_z(0.5),
            "shooting_residual": best_trajectory.shooting_residual,
            "friedmann_residual": best_trajectory.friedmann_residual,
            "scalar_continuity_residual": best_trajectory.scalar_continuity_residual,
            "matter_continuity_residual": best_trajectory.matter_continuity_residual,
            "radiation_continuity_residual": best_trajectory.radiation_continuity_residual,
            "continuity_diagnostic_kind": best_trajectory.continuity_diagnostic_kind,
        },
        "fixed_point_asymptote": {
            "w_phi": fixed.w_phi,
            "theta_prime": fixed.theta_prime,
            "eigenvalues": fixed.eigenvalues,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


source = STAGING / "src" / "ce_cosmo" / "gates" / "self_measurement_quintessence.py"
test = STAGING / "tests" / "test_self_measurement_quintessence.py"
result = {
    "status": "POST_HOC_EMPIRICAL_CALIBRATION_NOT_PREDICTION",
    "lambda_grid": "0.00 closure-limit baseline plus finite 0.01..1.40",
    "steps_scan": 800,
    "steps_reported_background": 1600,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
    "boundaries": [boundary_record(B_DESI), boundary_record(B_PLANCK)],
}
print(json.dumps(result, indent=2, sort_keys=True))
