from __future__ import annotations

import pytest

from reality_stone.clarus.fusion_remaining_branches_loop import (
    audit_direct_operator_completion,
    audit_reactor_propagation,
    audit_time_dependent_drive,
    current_fusion_remaining_branches_report,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_remaining_branches_report()


def test_direct_operator_math_solution_fails_uv_and_experimental_gates(report) -> None:
    audit = report.direct_operator

    assert audit.massless_required_nucleon_coupling == pytest.approx(0.00569352, rel=3.0e-6)
    assert audit.registered_mass_required_nucleon_coupling == pytest.approx(
        0.0174265,
        rel=3.0e-6,
    )
    assert audit.registered_mass_equivalent_higgs_mixing_sine > 15.0
    assert audit.mass_proportional_completion_scale_registered_gev < 20.0
    assert audit.registered_completion_scale_to_higgs_vev < 0.07
    assert audit.registered_completion_scale_to_scalar_mass > 500.0
    assert audit.registered_nuclear_matter_mean_field_mev_per_nucleon > 0.2
    assert audit.perturbative_low_energy_coupling
    assert not audit.electroweak_symmetric_heavy_completion_separated
    assert not audit.selected_portal_action_contains_direct_operator
    assert not audit.experimental_constraint_gate_pass
    assert not audit.physical_operator_accepted


def test_time_dependent_em_control_is_not_a_ce_scalar_source(report) -> None:
    audit = report.time_dependent_drive

    assert audit.published_field_min_energy_density_j_m3 == pytest.approx(4.42709e18, rel=2e-6)
    assert audit.published_field_max_energy_density_j_m3 == pytest.approx(4.42709e20, rel=2e-6)
    assert audit.ce_energy_to_published_photon_ceiling_ratio == pytest.approx(29647.57)
    assert audit.quiver_amplitude_at_one_kev_and_max_field_fm > 50.0
    assert audit.quiver_amplitude_at_ce_frequency_and_max_field_fm < 1.0e-6
    assert audit.field_for_one_nuclear_radius_quiver_at_ce_frequency_v_m > 1.0e23
    assert audit.ce_scalar_to_published_max_em_energy_density_ratio > 1.0e17
    assert not audit.ce_frequency_inside_published_control_window
    assert not audit.electromagnetic_control_equals_scalar_drive
    assert not audit.source_geometry_supplied
    assert not audit.floquet_dt_scattering_solved
    assert not audit.physical_time_dependent_upgrade_derived


def test_reactor_and_icf_propagation_remains_a_bound_not_prediction(report) -> None:
    audit = report.reactor_propagation

    assert audit.allowed_static_reactivity_fractional_gain == pytest.approx(6.18129e-10, rel=8e-5)
    assert audit.higgs_model_class_reactivity_fractional_upper_bound == pytest.approx(
        4.01944e-4,
        rel=8e-5,
    )
    assert audit.rejected_nif_linear_energy_saving_upper_bound_j < 850.0
    assert audit.direct_operator_lawson_fractional_reduction == pytest.approx(0.01 / 1.01)
    assert not audit.direct_operator_physical_gate_pass
    assert not audit.radiation_hydrodynamic_capsule_model_supplied
    assert not audit.icf_prediction_derived


def test_final_remaining_report_is_exhaustive_but_fail_closed(report) -> None:
    assert report.all_declared_remaining_branches_audited
    assert report.static_equation_chain_closed
    assert not report.direct_operator_physical_gate_pass
    assert not report.time_dependent_physical_gate_pass
    assert not report.physical_one_percent_reactivity_gain_derived
    assert not report.physical_reactor_or_icf_upgrade_derived
    assert report.maximum_supported_stage == "MODEL_CLASS_NO_GO_PLUS_SOURCE_ENERGY_CONTROLS"


def test_public_audit_functions_are_deterministic(report) -> None:
    assert audit_direct_operator_completion() == report.direct_operator
    assert audit_time_dependent_drive() == report.time_dependent_drive
    assert audit_reactor_propagation() == report.reactor_propagation
