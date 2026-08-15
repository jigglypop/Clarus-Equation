from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from examples.physics.density_bridge_variational_audit import (
    CRITICAL_DENSITY_STATUS,
    EXTERNAL_INPUT_STATUS,
    POTENTIAL_CHOICE_STATUS,
    PREDICTION_STATUS,
    action_dimension_ledger,
    build_audit,
    conserved_dust_log_fraction_slope,
    matter_composition_audit,
    potential,
    required_tracker_transfer_ratio,
    small_fixed_point_root,
    static_scalar_stress,
    stationary_gradient,
    stationary_hessian,
    unit_branch_hessian,
    vacuum_offset_counterexample,
    weighted_event_audit,
)


D = 3.1777584234099736


def test_small_fixed_point_is_stationary_and_locally_stable() -> None:
    q = small_fixed_point_root(D)

    assert 0.0 < q < 1.0 / D
    assert q == pytest.approx(math.exp(-D * (1.0 - q)), abs=1.0e-15)
    assert stationary_gradient(q, D) == pytest.approx(0.0, abs=1.0e-14)
    assert stationary_hessian(q, D) > 0.0
    assert unit_branch_hessian(D) < 0.0


def test_declared_field_domain_includes_unit_endpoint() -> None:
    assert potential(1.0, D) == pytest.approx(-1.0 + 0.5 * D)
    assert stationary_gradient(1.0, D) == pytest.approx(0.0)
    assert stationary_hessian(1.0, D) == pytest.approx(1.0 - D)


def test_additive_offset_preserves_root_but_changes_density_fraction() -> None:
    example = vacuum_offset_counterexample(D)

    assert example.root_residual == pytest.approx(0.0, abs=1.0e-14)
    assert example.hessian == stationary_hessian(example.root, D)
    assert example.baseline_fraction == pytest.approx(0.0, abs=1.0e-14)
    assert example.shifted_fraction == pytest.approx(0.2, abs=1.0e-14)
    assert stationary_gradient(example.root, D) == pytest.approx(0.0, abs=1.0e-14)
    assert potential(example.root, D, example.shifted_offset) > potential(
        example.root, D, example.baseline_offset
    )


def test_weighted_event_identity_and_equal_energy_iff() -> None:
    equal = weighted_event_audit(0.2, 3.0, 3.0)
    unequal = weighted_event_audit(0.2, 2.0, 1.0)

    assert equal.equal_conditional_means
    assert equal.weighted_fraction == pytest.approx(equal.probability)
    assert equal.direct_difference == pytest.approx(equal.covariance_difference)

    assert not unequal.equal_conditional_means
    assert unequal.weighted_fraction != pytest.approx(unequal.probability)
    assert unequal.direct_difference == pytest.approx(unequal.covariance_difference)


def test_matter_composition_does_not_skip_total_matter_fraction() -> None:
    audit = matter_composition_audit(0.2, 0.4)

    assert audit.matter_composition_fraction == 0.2
    assert audit.critical_density_fraction == pytest.approx(0.08)
    assert not audit.equals_branching_probability
    assert audit.critical_density_bridge_status == CRITICAL_DENSITY_STATUS
    with pytest.raises(ValueError, match="must be positive"):
        matter_composition_audit(0.2, 0.0)


def test_candidate_action_dimension_ledger() -> None:
    ledger = action_dimension_ledger()

    assert ledger["x"] == 0
    assert ledger["d"] == 0
    assert ledger["log_argument"] == 0
    assert ledger["kinetic_action_density"] == 4
    assert ledger["potential_action_density"] == 4
    assert ledger["required_action_density"] == 4
    assert ledger["passes"] is True


@pytest.mark.parametrize(
    ("w_total", "slope", "tracker_ratio"),
    ((0.0, 0.0, 0.0), (1.0 / 3.0, 1.0, -1.0), (-1.0, -3.0, 3.0)),
)
def test_conserved_dust_fraction_no_go(
    w_total: float,
    slope: float,
    tracker_ratio: float,
) -> None:
    assert conserved_dust_log_fraction_slope(w_total) == pytest.approx(slope)
    assert required_tracker_transfer_ratio(w_total) == pytest.approx(tracker_ratio)


def test_static_canonical_scalar_has_vacuum_not_dust_stress() -> None:
    stress = static_scalar_stress(2.0)

    assert stress.pressure == -stress.energy_density
    assert stress.equation_of_state == -1.0
    assert stress.equation_of_state != 0.0


def test_structured_audit_passes_only_approved_mathematics() -> None:
    report = build_audit(D)

    assert report["approved_mathematical_checks_pass"] is True
    assert report["input"]["d_status"] == EXTERNAL_INPUT_STATUS
    assert report["claims"]["potential_choice"] == POTENTIAL_CHOICE_STATUS
    assert report["claims"]["critical_density_bridge"] == CRITICAL_DENSITY_STATUS
    assert report["claims"]["physical_prediction"] == PREDICTION_STATUS
    assert report["physical_bridge_complete"] is False
    assert report["is_physical_prediction"] is False


def test_cli_succeeds_for_math_and_fails_closed_for_physical_bridge() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "physics" / "density_bridge_variational_audit.py"

    mathematical = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    physical = subprocess.run(
        [sys.executable, "-B", str(script), "--require-physical-bridge"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    mathematical_report = json.loads(mathematical.stdout)
    physical_report = json.loads(physical.stdout)
    assert mathematical.returncode == 0
    assert mathematical_report["exit_policy"] == "APPROVED_MATH_ONLY"
    assert physical.returncode == 2
    assert physical_report["exit_policy"] == "REQUIRE_PHYSICAL_BRIDGE"
    assert physical_report["claims"]["critical_density_bridge"] == "INCOMPLETE"
    assert physical_report["physical_bridge_complete"] is False


def test_source_has_no_target_data_or_prediction_promotion() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "examples" / "physics" / "density_bridge_variational_audit.py"
    ).read_text(encoding="utf-8")

    for forbidden in ("Planck", "RECENT_BASELINES", "omega_b_h2", "H0", "0.0486"):
        assert forbidden not in source
    assert 'CRITICAL_DENSITY_STATUS = "INCOMPLETE"' in source
    assert 'PREDICTION_STATUS = "NONE"' in source
