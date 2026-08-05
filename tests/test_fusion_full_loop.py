from __future__ import annotations

import math

import pytest

from reality_stone.clarus.fusion_full_loop import (
    audit_broken_z2_branch,
    audit_coherent_background,
    audit_icf_ignition,
    audit_thermal_reactivity,
    audit_z2_pair_branch,
    bosch_hale_dt_reactivity,
    current_full_fusion_loop_report,
)


def test_z2_pair_branch_stops_at_tree_vertex() -> None:
    audit = audit_z2_pair_branch()

    assert audit.tree_pair_vertex_present
    assert not audit.single_scalar_source_present
    assert audit.two_scalar_cut_threshold_mev == pytest.approx(59.29514)
    assert audit.two_scalar_asymptotic_range_fm == pytest.approx(3.32788, rel=2.0e-6)
    assert audit.zero_bare_mass_portal_pole_gev == pytest.approx(43.7677, rel=3.0e-6)
    assert not audit.registered_light_target_predicted
    assert not audit.light_target_portal_dominated
    assert audit.invisible_branching_fraction == pytest.approx(0.825312, rel=2.0e-6)
    assert audit.invisible_branching_fraction > audit.supplied_invisible_limit
    assert audit.maximum_lambda_under_supplied_limit == pytest.approx(0.00511074, rel=3.0e-6)
    assert not audit.supplied_portal_benchmark_allowed
    assert not audit.renormalized_two_scalar_exchange_amplitude_derived
    assert not audit.dt_scattering_residual_derived


def test_broken_z2_branch_fails_supplied_limit_and_has_only_tiny_static_force() -> None:
    audit = audit_broken_z2_branch()

    assert audit.mixing_ratio_to_limit == pytest.approx(11.4316, rel=3.0e-5)
    assert audit.branching_like_ratio_to_limit == pytest.approx(130.682, rel=4.0e-5)
    assert not audit.legacy_benchmark_allowed
    assert audit.legacy_static_force_ratio_at_nuclear_radius < 2.0e-8
    assert audit.maximum_static_force_ratio_under_supplied_limit < 2.0e-10
    assert not audit.timelike_quality_factor_enhances_static_force
    assert not audit.nonresonant_dt_amplitude_derived
    assert audit.status == "LEGACY_MIXING_REJECTED_BY_SUPPLIED_LIMIT"


def test_coherent_background_exposes_source_energy_scale_without_claim_upgrade() -> None:
    audit = audit_coherent_background()

    assert audit.fractional_nucleon_mass_modulation == 0.01
    assert audit.required_field_amplitude_mev > 1.0e5
    assert audit.energy_density_j_m3 > 1.0e38
    assert audit.quantum_number_density_m3 > 6.0e49
    assert audit.replenishment_power_density_w_m3 > 1.0e45
    assert audit.drive_to_transit_frequency_ratio == pytest.approx(12.9913, rel=2.0e-6)
    assert not audit.source_current_derived
    assert not audit.coherent_state_preparation_derived
    assert not audit.pump_work_accounted
    assert not audit.backreaction_solved
    assert not audit.floquet_dt_scattering_solved


def test_bosch_hale_dt_baseline_at_10_kev() -> None:
    theta, xi, reactivity = bosch_hale_dt_reactivity(10.0)

    assert theta == pytest.approx(11.9356225, rel=2.0e-8)
    assert xi == pytest.approx(2.9146850, rel=2.0e-8)
    assert reactivity == pytest.approx(1.1361655e-16, rel=2.0e-7)

    audit = audit_thermal_reactivity()
    assert audit.baseline_reactivity_cm3_s == pytest.approx(reactivity)
    assert audit.baseline_ignition_n_tau_cm3_s == pytest.approx(3.00052325e14, rel=2.0e-8)
    assert not audit.candidate_cross_section_supplied
    assert not audit.candidate_reactivity_derived
    assert not audit.counterfactual_wkb_factor_used_as_reactivity
    assert not audit.modified_lawson_value_derived


@pytest.mark.parametrize("temperature_kev", [0.199999, 100.000001, 1.0e-12, 1.0e6])
def test_bosch_hale_dt_reactivity_rejects_extrapolation(temperature_kev: float) -> None:
    with pytest.raises(ValueError, match="0.2--100 keV"):
        bosch_hale_dt_reactivity(temperature_kev)


def test_icf_loop_records_but_rejects_linear_energy_rescaling() -> None:
    audit = audit_icf_ignition(counterfactual_wkb_factor=40.85172379)

    assert audit.published_target_gain == pytest.approx(3.10 / 2.05)
    assert audit.rejected_linear_rescale_energy_kj == pytest.approx(50.1815, rel=2.0e-5)
    assert not audit.capsule_model_supplied
    assert not audit.hydrodynamic_gain_derived
    assert not audit.ignition_energy_derived
    assert audit.status == "LINEAR_LASER_ENERGY_RESCALING_REJECTED"


def test_full_report_exhausts_branches_without_illegal_promotion() -> None:
    report = current_full_fusion_loop_report()
    statuses = {stage.name: stage.status for stage in report.stages}

    assert report.all_candidate_branches_exhausted
    assert statuses["Z2_PAIR_PORTAL_VERTEX"] == "CONDITIONAL_PASS"
    assert statuses["Z2_PAIR_FUSION_AMPLITUDE"] == "NOT_REACHED"
    assert statuses["BROKEN_Z2_LEGACY_MIXING"] == "REJECT"
    assert statuses["COHERENT_BACKGROUND_ENERGY_SCALE"] == "NEGATIVE_CONTROL"
    assert statuses["BOSCH_HALE_DT_BASELINE"] == "PASS"
    assert statuses["MODIFIED_DT_REACTIVITY_AND_LAWSON"] == "NOT_REACHED"
    assert statuses["NIF_RADIATION_HYDRODYNAMIC_GAIN"] == "NOT_REACHED"
    assert not report.physical_dt_amplitude_modified
    assert not report.modified_thermal_reactivity_derived
    assert not report.modified_lawson_derived
    assert not report.nif_ignition_prediction_derived
    assert report.maximum_supported_stage == (
        "STANDARD_DT_BASELINE_PLUS_SOURCE_ENERGY_NEGATIVE_CONTROLS"
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: audit_broken_z2_branch(legacy_mixing_angle_sine=math.nan), "finite"),
        (lambda: audit_broken_z2_branch(supplied_mixing_limit=0.0), "must lie"),
        (lambda: audit_coherent_background(fractional_nucleon_mass_modulation=True), "real scalar"),
        (lambda: audit_coherent_background(mixing_angle_sine=0.0), "must lie"),
        (lambda: bosch_hale_dt_reactivity(-1.0), "positive"),
        (lambda: audit_icf_ignition(counterfactual_wkb_factor=0.0), "positive"),
    ],
)
def test_full_loop_invalid_inputs_fail_closed(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
