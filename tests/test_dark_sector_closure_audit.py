from __future__ import annotations

import math

from examples.physics.dark_sector_closure_audit import (
    SH0ES_2022_COMPATIBILITY_TARGET,
    SH0ES_2024_SMC_TARGET,
    acoustic_closure_audit,
    calibrate_common_rescaling_gamma,
)
from examples.physics.kinetic_dark_sector_gate import KineticClockConfig


def test_hubble_target_versions_remain_separate() -> None:
    assert SH0ES_2022_COMPATIBILITY_TARGET.h0_km_s_mpc == 73.04
    assert SH0ES_2024_SMC_TARGET.h0_km_s_mpc == 73.17
    assert SH0ES_2024_SMC_TARGET.role == "external_target"
    assert SH0ES_2022_COMPATIBILITY_TARGET.target_id != SH0ES_2024_SMC_TARGET.target_id


def test_sh0es_target_exposes_acoustic_and_physical_density_conditions() -> None:
    audit = acoustic_closure_audit(
        KineticClockConfig(gamma=10.0, steps=1200),
        distance_intervals=2048,
    )

    assert 136.0 < audit.required_rd_mpc < 138.0
    assert 131.0 < audit.required_rs_mpc < 133.5
    assert math.isclose(
        audit.external_reference_uniform_extra_density_over_baseline,
        0.1535,
        rel_tol=2.0e-3,
    )
    assert math.isclose(
        audit.external_reference_uniform_extra_fraction_of_total,
        0.1331,
        rel_tol=2.0e-3,
    )
    assert (
        audit.external_reference_uniform_extra_density_over_baseline
        > audit.external_reference_uniform_extra_fraction_of_total
    )
    assert (
        audit.rd_uniform_extra_density_over_same_boundary
        > audit.rd_uniform_extra_fraction_of_total_same_boundary
    )
    assert (
        audit.rs_uniform_extra_density_over_same_boundary
        > audit.rs_uniform_extra_fraction_of_total_same_boundary
    )
    assert audit.common_rescaling_relative_mismatch < 2.0e-3
    assert audit.omega_m_h2_relative_offset > 0.17
    assert 0.26 < audit.planck_density_consistent_omega_m0_at_target_h0 < 0.27
    assert audit.prediction is False
    assert audit.closure_status == "incomplete"
    assert audit.role == "external_target_diagnostic"


def test_common_rescaling_gamma_is_explicitly_posthoc() -> None:
    result = calibrate_common_rescaling_gamma(
        lower_gamma=10.0,
        upper_gamma=12.0,
        iterations=7,
        steps=1200,
        distance_intervals=2048,
    )

    assert 10.0 < result.calibrated_gamma < 12.0
    assert result.audit.common_rescaling_relative_mismatch < 2.0e-4
    assert result.prediction is False
    assert result.role == "POSTHOC_COMPRESSED_GEOMETRY_CALIBRATION"
    assert math.isfinite(result.audit.bao_profiled_chi2)
