from __future__ import annotations

import math

from examples.physics.kinetic_dark_sector_gate import KineticClockConfig, solve_background
from examples.physics.kinetic_dark_sector_perturbation_gate import (
    evaluate_single_clock_gate,
    quasi_static_growth_diagnostic,
    scan_kappa_sensitivity,
)


def test_single_clock_gate_passes_without_claiming_matter_growth() -> None:
    solution = solve_background(KineticClockConfig(gamma=10.0, steps=600))
    gate = evaluate_single_clock_gate(solution)

    assert gate.status == "PASS_SINGLE_CLOCK_ONLY"
    assert gate.failed_gates == ()
    assert gate.matter_growth_likelihood.startswith("NOT_IMPLEMENTED")
    assert gate.min_friction > 0.0
    assert gate.max_tachyon_ratio < 1.0
    assert gate.max_log_growth_bound < 1.0
    assert gate.min_pump_slope > 0.0
    assert gate.min_zeta_decay_slope > 0.0
    assert gate.min_energy_cutoff_over_h > 1.0
    assert gate.min_wavenumber_cutoff_over_k_1mpc > 1.0
    assert math.isfinite(gate.fixed_coordinate_growth_minus_one)


def test_quasi_static_growth_is_explicitly_approximate_and_finite() -> None:
    solution = solve_background(KineticClockConfig(gamma=10.0, steps=600))
    diagnostic = quasi_static_growth_diagnostic(solution)

    assert 0.0 < diagnostic.predicted_fsigma8 < 1.0
    assert math.isfinite(diagnostic.pull)
    assert diagnostic.closure == "KINETIC_CLUSTERS_VACUUM_SMOOTH_GR_SUBHORIZON"
    assert diagnostic.role.startswith("APPROXIMATE_DIAGNOSTIC")


def test_kappa_scan_exposes_the_conditional_stability_threshold() -> None:
    rows = scan_kappa_sensitivity((3.0e11, 1.0e12), steps=600)

    assert rows[0].status == "FAIL_SINGLE_CLOCK_GATE"
    assert "positive_friction" in rows[0].failed_gates
    assert rows[1].status == "PASS_SINGLE_CLOCK_ONLY"
    assert rows[1].failed_gates == ()
