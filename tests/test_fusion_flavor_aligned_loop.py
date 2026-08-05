from __future__ import annotations

import pytest

from reality_stone.clarus.fusion_flavor_aligned_loop import (
    current_fusion_flavor_aligned_report,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_flavor_aligned_report()


def test_flavor_aligned_operator_reproduces_the_dt_target(report) -> None:
    audit = report.operator

    assert audit.universal_required_nucleon_coupling == pytest.approx(0.0174265, rel=4.0e-6)
    assert audit.aligned_scale_gev == pytest.approx(8.971, rel=5.0e-5)
    assert audit.proton_coupling == pytest.approx(0.017166, rel=5.0e-5)
    assert audit.neutron_coupling == pytest.approx(0.017612, rel=5.0e-5)
    assert audit.charge_product_relative_residual == pytest.approx(0.0, abs=2.0e-15)
    assert audit.static_one_percent_target_reproduced
    assert audit.gauge_invariant_operator_written
    assert audit.vlq_uv_example_supplied


def test_vlq_example_is_perturbative_but_not_naturalness_complete(report) -> None:
    audit = report.operator

    assert audit.required_vlq_kappa > 550.0
    assert not audit.effective_kappa_is_lagrangian_coupling
    assert audit.required_plot_coordinate_kappa_v_over_m == pytest.approx(27.42, rel=2.0e-4)
    assert audit.up_vlq_yukawa < 0.01
    assert audit.down_vlq_yukawa < 0.02
    assert audit.strange_vlq_yukawa == pytest.approx(0.299, rel=3.0e-3)
    assert audit.maximum_left_mixing_angle < 0.011
    assert audit.all_displayed_uv_couplings_perturbative
    assert not audit.full_smeft_wet_rg_matching_supplied
    assert not audit.scalar_mass_naturalness_protected
    assert not audit.radiative_mass_stability_gate_pass
    assert not audit.uv_action_gate_pass


def test_neutron_central_extrapolation_margin_is_not_a_pass(report) -> None:
    audit = report.neutron_constraint

    assert audit.extrapolated_equal_coupling_bound == pytest.approx(0.0175796, rel=2.0e-6)
    assert audit.flavor_matched_lead_effective_coupling == pytest.approx(0.0175242, rel=5.0e-5)
    assert 0.0 < audit.central_fractional_margin < 0.004
    assert audit.neutron_coupling_to_equal_bound_ratio > 1.0
    assert audit.candidate_outside_source_signal_mass_range
    assert audit.representative_q2_over_m2 > 0.19
    assert audit.contact_limit_correction_scale_exceeds_margin
    assert not audit.mass_specific_pb_differential_likelihood_supplied
    assert not audit.neutron_constraint_gate_pass


def test_central_kaon_curve_is_open_but_nlo_envelope_is_not(report) -> None:
    audit = report.rare_decay_constraint

    assert audit.central_bound_to_candidate_ratio > 6.0
    assert audit.central_curve_allows_candidate
    assert not audit.conservative_nlo_envelope_allows_candidate
    assert not audit.full_order_p4_weak_chpt_amplitude_supplied
    assert not audit.na62_e949_mass_bin_likelihood_supplied
    assert not audit.rare_decay_constraint_gate_pass


def test_invisible_decay_example_does_not_close_dark_constraints(report) -> None:
    audit = report.invisible_completion

    assert audit.partial_width_mev == pytest.approx(9.84e-9, rel=2.0e-3)
    assert audit.lifetime_s == pytest.approx(6.69e-14, rel=2.0e-3)
    assert audit.decay_length_m == pytest.approx(2.01e-5, rel=3.0e-3)
    assert audit.decay_kinematically_open
    assert audit.invisible_yukawa_perturbative
    assert audit.prompt_decay_to_invisible_states
    assert not audit.dark_sector_constraint_gate_pass


def test_candidate_remains_fail_closed(report) -> None:
    assert report.mathematical_one_percent_solution_reproduced
    assert report.gauge_invariant_perturbative_uv_candidate_supplied
    assert not report.all_existing_constraint_gates_pass
    assert not report.physical_ce_fusion_branch_accepted
    assert report.candidate_classification == (
        "CLOSEST_CONDITIONAL_CANDIDATE_NOT_CONSTRAINT_CLEARED"
    )
