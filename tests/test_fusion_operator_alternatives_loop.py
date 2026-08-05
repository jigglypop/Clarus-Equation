from __future__ import annotations

import pytest

from reality_stone.clarus.fusion_operator_alternatives_loop import (
    current_fusion_operator_alternatives_report,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_operator_alternatives_report()


def test_pure_trace_gluon_direction_fails_rare_decay(report) -> None:
    audit = report.trace_gluon

    assert audit.required_scale_over_trace_coefficient_gev == pytest.approx(44.92, rel=3.0e-4)
    assert audit.required_plot_coordinate_abs_k_theta_v_over_f == pytest.approx(5.48, rel=2.0e-3)
    assert audit.required_to_bound_ratio > 3.9e3
    assert audit.gauge_invariant_operator_available
    assert not audit.one_parameter_rare_decay_gate_pass
    assert not audit.physical_operator_gate_pass


def test_isospin_endpoints_do_not_open_an_attractive_blind_spot(report) -> None:
    audit = report.isospin

    assert audit.protophobic_required_neutron_coupling == pytest.approx(0.03018, rel=2.0e-4)
    assert audit.protophobic_to_neutron_bound_ratio > 1.3
    assert audit.neutron_phobic_required_proton_coupling == pytest.approx(0.042686, rel=2.0e-5)
    assert audit.neutron_phobic_kaon_combination_one_violation > 9.0e3
    assert audit.neutron_phobic_kaon_combination_two_violation > 2.7e4
    assert audit.lead_cancellation_gp_over_gn == pytest.approx(-126.0 / 82.0)
    assert audit.lead_cancellation_dt_product_coefficient < 0.0
    assert not audit.lead_cancellation_makes_dt_attraction
    assert audit.universal_minimizes_max_abs_nucleon_coupling
    assert audit.neutron_only_proxy_favors_neutron_phobic_limit
    assert not audit.protophobic_gate_pass
    assert not audit.neutron_phobic_gate_pass
    assert not audit.lead_blind_spot_gate_pass


def test_disformal_massless_upper_bound_is_grid_converged(report) -> None:
    audit = report.disformal

    assert audit.required_scale_for_one_percent_mev == pytest.approx(180.70494, rel=2.0e-6)
    assert audit.default_gain_at_required_scale == pytest.approx(0.01, rel=2.0e-8)
    assert audit.maximum_grid_gain_spread < 2.0e-6
    assert audit.massless_two_scalar_upper_bound


def test_supplied_disformal_bounds_all_miss_one_percent(report) -> None:
    audit = report.disformal

    assert audit.gain_at_hydrogen_bound == pytest.approx(0.00439587, rel=5.0e-5)
    assert audit.gain_at_stellar_bound == pytest.approx(6.02449e-8, rel=5.0e-5)
    assert audit.gain_at_atlas_bound == pytest.approx(2.59627e-33, rel=5.0e-5)
    assert audit.hydrogen_bound_derived_for_massless_scalar
    assert audit.stellar_bound_derived_for_massless_scalar
    assert not audit.mass_specific_atomic_or_stellar_bound_supplied
    assert audit.atlas_bound_applicable_in_light_mediator_limit
    assert audit.required_scale_below_applicable_atlas_bound
    assert audit.required_scale_below_all_supplied_bounds
    assert not audit.nonlinear_screening_completion_supplied
    assert not audit.experimental_constraint_gate_pass
    assert not audit.physical_operator_gate_pass


def test_no_alternative_is_promoted(report) -> None:
    assert report.all_declared_operator_alternatives_audited
    assert not report.any_alternative_constraint_cleared
    assert not report.physical_one_percent_ce_branch_derived
    assert report.maximum_supported_stage == "ALTERNATIVE_OPERATOR_MODEL_CLASS_NO_GO"
