from __future__ import annotations

import math

import pytest

from reality_stone.clarus.fusion_resonance_loop import (
    current_fusion_resonance_loop_report,
    scalar_line_audit,
    static_exchange_audit,
    wkb_counterfactual_audit,
)


def test_legacy_line_units_and_width_are_recomputed() -> None:
    audit = scalar_line_audit()

    assert audit.electron_width_mev == pytest.approx(9.58805e-15, rel=2.0e-6)
    assert audit.vacuum_quality_factor == pytest.approx(3.09218e15, rel=2.0e-6)
    assert audit.angular_frequency_rad_s == pytest.approx(4.50432e22, rel=2.0e-6)
    assert audit.cyclic_frequency_hz == pytest.approx(7.16885e21, rel=2.0e-6)
    assert audit.cyclic_linewidth_hz == pytest.approx(2.31838e6, rel=2.0e-6)
    assert audit.angular_frequency_rad_s / audit.cyclic_frequency_hz == pytest.approx(
        2.0 * math.pi,
        rel=2.0e-10,
    )


def test_collision_unit_correction_and_g_minus_two_control() -> None:
    audit = scalar_line_audit()

    assert audit.collision_cross_section_ansatz_m2 == pytest.approx(6.09996e-34, rel=3.0e-6)
    assert audit.collision_rate_ansatz_per_s == pytest.approx(3.62338e6, rel=3.0e-6)
    assert audit.collision_width_ansatz_mev == pytest.approx(2.38495e-15, rel=3.0e-6)
    assert audit.plasma_quality_factor_under_ansatz < audit.vacuum_quality_factor
    assert audit.electron_g_minus_two_one_loop == pytest.approx(2.13251e-19, rel=3.0e-6)
    assert not audit.collision_model_derived


def test_static_nuclear_exchange_cannot_hit_timelike_scalar_pole() -> None:
    audit = static_exchange_audit()

    assert audit.energy_transfer_mev == 0.0
    assert audit.spacelike_transfer
    assert audit.invariant_transfer_mev2 < 0.0
    assert audit.pole_invariant_mev2 > 0.0
    assert not audit.timelike_pole_reached
    assert not audit.static_propagator_has_quality_factor_enhancement
    assert not audit.driven_background_equals_pair_potential


def test_counterfactual_wkb_reproduces_q_1e9_without_claim_upgrade() -> None:
    audit = wkb_counterfactual_audit(supplied_quality_factor=1.0e9)

    assert audit.baseline_exponent == pytest.approx(2.81365, rel=2.0e-5)
    assert audit.modified_exponent == pytest.approx(0.958673, rel=3.0e-5)
    assert audit.counterfactual_tunnelling_enhancement == pytest.approx(40.8517, rel=4.0e-5)
    assert audit.inner_radius_cancellation_quality_factor == pytest.approx(
        6.04e7,
        rel=3.0e-3,
    )
    assert audit.whole_barrier_removal_quality_factor == pytest.approx(
        6.2974e10,
        rel=3.0e-3,
    )
    assert audit.whole_barrier_fractional_bandwidth == pytest.approx(1.58795e-11, rel=4.0e-3)
    assert not audit.whole_barrier_removed
    assert not audit.quality_factor_to_static_potential_bridge_derived
    assert not audit.thermal_reactivity_derived
    assert not audit.ignition_energy_derived


def test_whole_barrier_threshold_is_distinct_from_inner_radius_cancellation() -> None:
    seed = wkb_counterfactual_audit(supplied_quality_factor=0.0)
    threshold = wkb_counterfactual_audit(
        supplied_quality_factor=seed.whole_barrier_removal_quality_factor
    )

    assert threshold.whole_barrier_removed
    assert threshold.modified_exponent == pytest.approx(0.0, abs=2.0e-8)
    assert threshold.counterfactual_tunnelling_enhancement == pytest.approx(277.91, rel=2.0e-4)
    assert threshold.whole_barrier_removal_quality_factor > (
        1000.0 * threshold.inner_radius_cancellation_quality_factor
    )


def test_report_is_fail_closed_at_every_physical_upgrade() -> None:
    report = current_fusion_resonance_loop_report()
    statuses = {stage.name: stage.status for stage in report.stages}

    assert statuses["CANONICAL_Z2_LINEAR_NUCLEON_PORTAL"] == "CLOSED_OFF"
    assert statuses["STATIC_SPACELIKE_RESONANCE"] == "REJECT"
    assert statuses["COUNTERFACTUAL_WKB_POTENTIAL"] == "CONDITIONAL_PASS"
    assert statuses["MAXWELLIAN_D_T_REACTIVITY"] == "NOT_REACHED"
    assert report.legacy_counterfactual_wkb_reproduced
    assert not report.canonical_z2_linear_nucleon_coupling_present
    assert not report.physical_resonant_barrier_reduction_derived
    assert not report.thermal_reactivity_derived
    assert not report.nif_capsule_gain_derived
    assert not report.ignition_energy_derived
    assert report.maximum_supported_stage == "LEGACY_COUNTERFACTUAL_WKB_CONTROL_ONLY"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: scalar_line_audit(scalar_mass_mev=True), "real scalar"),
        (lambda: scalar_line_audit(mixing_angle_sine=1.1), "must not exceed"),
        (lambda: scalar_line_audit(integration_points=10), "at least 1001"),
        (lambda: static_exchange_audit(momentum_transfer_mev=-1.0), "nonnegative"),
        (lambda: wkb_counterfactual_audit(supplied_quality_factor=-1.0), "nonnegative"),
        (lambda: wkb_counterfactual_audit(supplied_quality_factor=1.0, grid_points=1000), "at least 1001"),
    ],
)
def test_invalid_inputs_fail_closed(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
