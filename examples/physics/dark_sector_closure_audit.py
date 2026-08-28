"""Fail-open research diagnostic for the remaining dark-sector closure gap.

The calculation deliberately does not predict H0.  It injects a versioned
external Hubble target, profiles the DESI DR2 BAO scale of the already
boundary-calibrated kinetic background, and asks which drag and recombination
sound horizons would be required.  It then exposes the separate physical-
density condition that a compressed acoustic-angle match does not test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from examples.physics.ce_residual_forward_model import (
    CEForwardParams,
    early_universe_sound_horizon,
)
from examples.physics.kinetic_dark_sector_gate import (
    KineticClockConfig,
    OMEGA_B0,
    OMEGA_K0,
    OMEGA_V0,
    PLANCK_100_THETA_STAR,
    REFERENCE_RD_MPC,
    SPEED_OF_LIGHT_KM_S,
    compressed_cmb_acoustic_diagnostic,
    profile_desi_bao,
    solve_background,
)
from examples.physics.theater_seat_ledger import (
    uniform_opening_energy_requirement,
)


PLANCK_COMPRESSED_OMEGA_M_H2 = 0.0224 + 0.1200


@dataclass(frozen=True)
class ExternalHubbleTarget:
    target_id: str
    h0_km_s_mpc: float
    sigma_km_s_mpc: float
    units: str
    source: str
    release: str
    role: str = "external_target"

    def __post_init__(self) -> None:
        if self.h0_km_s_mpc <= 0.0 or self.sigma_km_s_mpc <= 0.0:
            raise ValueError("Hubble target value and uncertainty must be positive")


SH0ES_2022_COMPATIBILITY_TARGET = ExternalHubbleTarget(
    target_id="SH0ES_RIESS_2022_BASELINE",
    h0_km_s_mpc=73.04,
    sigma_km_s_mpc=1.04,
    units="km s^-1 Mpc^-1",
    source="https://doi.org/10.3847/2041-8213/ac5c5b",
    release="Riess et al., ApJ Letters 934 L7 (2022)",
    role="historical_compatibility_external_target",
)


SH0ES_2024_SMC_TARGET = ExternalHubbleTarget(
    target_id="SH0ES_BREUVAL_SMC_2024",
    h0_km_s_mpc=73.17,
    sigma_km_s_mpc=0.86,
    units="km s^-1 Mpc^-1",
    source="https://doi.org/10.3847/1538-4357/ad630e",
    release="Breuval et al., ApJ 973 30 (2024)",
)


@dataclass(frozen=True)
class AcousticClosureAudit:
    target_id: str
    target_h0_km_s_mpc: float
    target_sigma_km_s_mpc: float
    target_units: str
    target_source: str
    target_release: str
    target_role: str
    gamma: float
    kappa: float
    background_steps: int
    distance_intervals: int
    bao_dataset: str
    bao_profiled_scale: float
    bao_profiled_chi2: float
    bao_dof: int
    required_rd_mpc: float
    required_rs_mpc: float
    external_reference_rd_mpc: float
    required_rd_over_external_reference: float
    external_reference_uniform_extra_density_over_baseline: float
    external_reference_uniform_extra_fraction_of_total: float
    standard_same_boundary_rd_mpc: float
    standard_same_boundary_rs_mpc: float
    required_rd_over_standard_same_boundary: float
    required_rs_over_standard_same_boundary: float
    common_rescaling_signed_gap: float
    common_rescaling_relative_mismatch: float
    declared_geometry_tolerance: float
    compressed_geometry_match_within_tolerance: bool
    rd_uniform_extra_density_over_same_boundary: float
    rd_uniform_extra_fraction_of_total_same_boundary: float
    rs_uniform_extra_density_over_same_boundary: float
    rs_uniform_extra_fraction_of_total_same_boundary: float
    omega_m_h2_same_boundary: float
    planck_compressed_omega_m_h2: float
    omega_m_h2_relative_offset: float
    planck_density_consistent_omega_m0_at_target_h0: float
    prediction: bool = False
    closure_status: str = "incomplete"
    role: str = "external_target_diagnostic"


@dataclass(frozen=True)
class CommonRescalingCalibration:
    lower_gamma: float
    upper_gamma: float
    iterations: int
    calibrated_gamma: float
    audit: AcousticClosureAudit
    prediction: bool = False
    role: str = "POSTHOC_COMPRESSED_GEOMETRY_CALIBRATION"


def acoustic_closure_audit(
    config: KineticClockConfig = KineticClockConfig(),
    *,
    target: ExternalHubbleTarget = SH0ES_2024_SMC_TARGET,
    distance_intervals: int = 4096,
    geometry_tolerance: float = 1.0e-3,
) -> AcousticClosureAudit:
    """Compute the BAO/CMB acoustic targets at one external H0 value."""

    if not math.isfinite(geometry_tolerance) or geometry_tolerance <= 0.0:
        raise ValueError("geometry_tolerance must be finite and positive")
    solution = solve_background(config)
    bao = profile_desi_bao(solution)
    cmb = compressed_cmb_acoustic_diagnostic(
        solution,
        h0_km_s_mpc=target.h0_km_s_mpc,
        distance_intervals=distance_intervals,
    )
    params = CEForwardParams(
        omega_b0=OMEGA_B0,
        omega_dm0=OMEGA_K0,
        omega_lambda0=OMEGA_V0,
        h0=target.h0_km_s_mpc,
    )
    standard_drag = early_universe_sound_horizon(params)

    required_rd = SPEED_OF_LIGHT_KM_S / (
        bao.scale * target.h0_km_s_mpc
    )
    required_rs = (
        PLANCK_100_THETA_STAR * cmb.transverse_distance_mpc / 100.0
    )
    rd_external_scale = required_rd / REFERENCE_RD_MPC
    rd_same_boundary_scale = required_rd / standard_drag.rd_mpc
    rs_same_boundary_scale = required_rs / cmb.sound_horizon_mpc
    signed_gap = rd_same_boundary_scale - rs_same_boundary_scale
    mean_scale = 0.5 * (rd_same_boundary_scale + rs_same_boundary_scale)
    relative_mismatch = abs(signed_gap) / mean_scale
    external_opening = uniform_opening_energy_requirement(rd_external_scale)
    rd_same_boundary_opening = uniform_opening_energy_requirement(
        rd_same_boundary_scale
    )
    rs_same_boundary_opening = uniform_opening_energy_requirement(
        rs_same_boundary_scale
    )
    h = target.h0_km_s_mpc / 100.0
    omega_m_h2_offset = (
        params.omega_m_h2 / PLANCK_COMPRESSED_OMEGA_M_H2 - 1.0
    )

    return AcousticClosureAudit(
        target_id=target.target_id,
        target_h0_km_s_mpc=target.h0_km_s_mpc,
        target_sigma_km_s_mpc=target.sigma_km_s_mpc,
        target_units=target.units,
        target_source=target.source,
        target_release=target.release,
        target_role=target.role,
        gamma=config.gamma,
        kappa=config.kappa,
        background_steps=config.steps,
        distance_intervals=distance_intervals,
        bao_dataset=bao.dataset,
        bao_profiled_scale=bao.scale,
        bao_profiled_chi2=bao.chi2,
        bao_dof=bao.dof,
        required_rd_mpc=required_rd,
        required_rs_mpc=required_rs,
        external_reference_rd_mpc=REFERENCE_RD_MPC,
        required_rd_over_external_reference=rd_external_scale,
        external_reference_uniform_extra_density_over_baseline=(
            external_opening.extra_density_over_baseline
        ),
        external_reference_uniform_extra_fraction_of_total=(
            external_opening.extra_fraction_of_total
        ),
        standard_same_boundary_rd_mpc=standard_drag.rd_mpc,
        standard_same_boundary_rs_mpc=cmb.sound_horizon_mpc,
        required_rd_over_standard_same_boundary=rd_same_boundary_scale,
        required_rs_over_standard_same_boundary=rs_same_boundary_scale,
        common_rescaling_signed_gap=signed_gap,
        common_rescaling_relative_mismatch=relative_mismatch,
        declared_geometry_tolerance=geometry_tolerance,
        compressed_geometry_match_within_tolerance=(
            relative_mismatch <= geometry_tolerance
        ),
        rd_uniform_extra_density_over_same_boundary=(
            rd_same_boundary_opening.extra_density_over_baseline
        ),
        rd_uniform_extra_fraction_of_total_same_boundary=(
            rd_same_boundary_opening.extra_fraction_of_total
        ),
        rs_uniform_extra_density_over_same_boundary=(
            rs_same_boundary_opening.extra_density_over_baseline
        ),
        rs_uniform_extra_fraction_of_total_same_boundary=(
            rs_same_boundary_opening.extra_fraction_of_total
        ),
        omega_m_h2_same_boundary=params.omega_m_h2,
        planck_compressed_omega_m_h2=PLANCK_COMPRESSED_OMEGA_M_H2,
        omega_m_h2_relative_offset=omega_m_h2_offset,
        planck_density_consistent_omega_m0_at_target_h0=(
            PLANCK_COMPRESSED_OMEGA_M_H2 / (h * h)
        ),
    )


def calibrate_common_rescaling_gamma(
    *,
    lower_gamma: float = 10.0,
    upper_gamma: float = 12.0,
    iterations: int = 10,
    steps: int = 2400,
    distance_intervals: int = 4096,
    target: ExternalHubbleTarget = SH0ES_2024_SMC_TARGET,
) -> CommonRescalingCalibration:
    """Post-hoc bisection where required r_d and r_s share one scale factor."""

    if lower_gamma <= 0.0 or upper_gamma <= lower_gamma:
        raise ValueError("gamma bounds must be positive and ordered")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    def evaluate(gamma: float) -> AcousticClosureAudit:
        return acoustic_closure_audit(
            KineticClockConfig(gamma=gamma, steps=steps),
            target=target,
            distance_intervals=distance_intervals,
        )

    low, high = lower_gamma, upper_gamma
    low_audit = evaluate(low)
    high_audit = evaluate(high)
    low_value = low_audit.common_rescaling_signed_gap
    high_value = high_audit.common_rescaling_signed_gap
    if low_value == 0.0:
        final_audit = low_audit
    elif high_value == 0.0:
        final_audit = high_audit
    else:
        if low_value * high_value > 0.0:
            raise ArithmeticError("common-rescaling gamma root is not bracketed")
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            middle_audit = evaluate(middle)
            middle_value = middle_audit.common_rescaling_signed_gap
            if middle_value * low_value > 0.0:
                low = middle
                low_value = middle_value
            else:
                high = middle
        final_audit = evaluate(0.5 * (low + high))

    return CommonRescalingCalibration(
        lower_gamma=lower_gamma,
        upper_gamma=upper_gamma,
        iterations=iterations,
        calibrated_gamma=final_audit.gamma,
        audit=final_audit,
    )


def main() -> int:
    audit = acoustic_closure_audit()
    calibration = calibrate_common_rescaling_gamma()
    print("# External-target dark-sector closure audit")
    print(json.dumps(asdict(audit), indent=2, sort_keys=True))
    print("# Post-hoc common-rescaling calibration")
    print(json.dumps(asdict(calibration), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
