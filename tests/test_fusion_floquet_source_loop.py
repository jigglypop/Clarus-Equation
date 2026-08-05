from __future__ import annotations

import pytest

from reality_stone.clarus.fusion_floquet_source_loop import (
    audit_floquet_threshold,
    audit_floquet_volkov_reactivity,
    current_fusion_floquet_source_report,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_floquet_source_report()


def test_zero_field_recovers_the_bosch_hale_reactivity() -> None:
    audit = audit_floquet_volkov_reactivity(electric_field_v_m=0.0)

    assert audit.reactivity_ratio == pytest.approx(1.0, abs=2.0e-14)
    assert audit.ponderomotive_energy_kev == 0.0
    assert audit.maximum_sideband_probability_residual < 1.0e-14
    assert not audit.target_reached


def test_fv_formula_regression_point_reaches_four_percent_but_is_extrapolated(report) -> None:
    audit = report.regression_point

    assert audit.effective_charge_fraction == pytest.approx(0.1992318073, rel=2.0e-10)
    assert audit.reactivity_ratio == pytest.approx(1.042232376, rel=2.0e-8)
    assert audit.ponderomotive_energy_kev == pytest.approx(0.381743301, rel=2.0e-8)
    assert audit.maximum_sideband_probability_residual < 1.0e-12
    assert audit.reaction_weighted_out_of_fit_probability < 1.0e-9
    assert audit.target_reached
    assert audit.shifted_cross_section_domain_gate_pass
    assert not audit.temperature_matches_published_thermal_benchmark
    assert not audit.gamow_saddle_inside_published_cn_energy_window
    assert not audit.published_validation_support_pass


def test_one_percent_field_threshold_is_grid_converged(report) -> None:
    audit = report.qed_threshold

    assert audit.required_electric_field_v_m == pytest.approx(4.86159699e15, rel=3.0e-8)
    assert audit.ponderomotive_energy_ev == pytest.approx(90.22551, rel=3.0e-7)
    assert audit.gamow_saddle_energy_kev == pytest.approx(30.91766, rel=3.0e-6)
    assert audit.keldysh_gamow_parameter == pytest.approx(13.0895, rel=3.0e-5)
    assert audit.electric_energy_density_j_m3 == pytest.approx(1.04634919e20, rel=3.0e-7)
    assert audit.maximum_grid_fractional_gain_spread < 1.0e-8
    assert audit.multiphoton_regime
    assert audit.numerical_convergence_pass
    assert not audit.published_parameter_window_pass
    assert audit.formula_extrapolation_one_percent_pass
    assert not audit.prescribed_qed_reactivity_branch_pass


def test_published_one_kev_thermal_benchmark_closes_validation_gate() -> None:
    audit = audit_floquet_threshold(temperature_kev=1.0)

    assert audit.required_electric_field_v_m == pytest.approx(8.680352e14, rel=4.0e-7)
    assert audit.temperature_matches_published_thermal_benchmark
    assert audit.gamow_saddle_inside_published_cn_energy_window
    assert audit.published_parameter_window_pass
    assert audit.formula_extrapolation_one_percent_pass
    assert audit.prescribed_qed_reactivity_branch_pass


def test_pump_energy_is_explicit_but_not_a_reactor_gain(report) -> None:
    audit = report.pump_ledger

    assert audit.optical_cycles > 700.0
    assert audit.incident_pulse_energy_j == pytest.approx(0.09855, rel=2.0e-3)
    assert audit.incremental_fusion_energy_in_volume_j > 0.0
    assert audit.incremental_fusion_to_incident_pulse_energy_ratio < 1.0e-7
    assert audit.source_geometry_declared
    assert audit.incident_pump_energy_accounted
    assert not audit.absorption_and_propagation_solved
    assert not audit.net_energy_positive
    assert not audit.reactor_upgrade_derived


def test_exact_z2_beat_loophole_fails_scalar_energy_gate(report) -> None:
    audit = report.ce_scalar_beat

    assert audit.second_mode_momentum_mev == pytest.approx(0.133374, rel=4.0e-6)
    assert audit.beat_reduced_wavelength_fm == pytest.approx(1479.50, rel=4.0e-5)
    assert audit.beat_locally_uniform_over_barrier
    assert audit.quadratic_nucleon_coefficient_mev_inv == pytest.approx(
        9.1922e-11,
        rel=4.0e-5,
    )
    assert audit.required_fractional_mass_modulation == pytest.approx(
        0.3024396,
        rel=4.0e-6,
    )
    assert audit.required_equal_mode_amplitude_mev == pytest.approx(1.75701e6, rel=4.0e-5)
    assert audit.required_scalar_energy_density_j_m3 == pytest.approx(5.65818e40, rel=5.0e-5)
    assert audit.scalar_to_qed_field_energy_density_ratio > 1.0e20
    assert not audit.linearized_mass_modulation_valid
    assert not audit.scalar_source_preparation_derived
    assert not audit.scalar_specific_crank_nicolson_solved
    assert not audit.physical_ce_scalar_reactivity_branch_pass


def test_final_report_does_not_promote_qed_to_ce(report) -> None:
    assert report.qed_fv_formula_extrapolation_one_percent_derived
    assert not report.qed_prescribed_field_one_percent_reactivity_derived
    assert report.source_and_pump_numbers_explicit
    assert not report.qed_net_reactor_upgrade_derived
    assert not report.ce_scalar_one_percent_reactivity_derived
    assert not report.electromagnetic_result_promoted_to_scalar
    assert report.maximum_supported_stage == (
        "QED_FV_10KEV_FORMULA_EXTRAPOLATION_CE_SCALAR_SOURCE_NO_GO"
    )
