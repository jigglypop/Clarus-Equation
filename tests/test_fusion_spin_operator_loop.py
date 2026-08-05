from __future__ import annotations

import pytest

from reality_stone.clarus.fusion_flavor_aligned_loop import (
    current_fusion_flavor_aligned_report,
)
from reality_stone.clarus.fusion_spin_operator_loop import (
    audit_axial_vector_operator,
    audit_pseudoscalar_operator,
    audit_spin_two_operator,
    audit_vector_operator,
    current_fusion_spin_operator_report,
)


@pytest.fixture(scope="module")
def report():
    return current_fusion_spin_operator_report()


def test_required_product_is_imported_from_existing_scalar_loop(report) -> None:
    scalar_product = current_fusion_flavor_aligned_report().operator.required_dt_charge_product

    assert report.required_dt_charge_product == scalar_product
    assert report.required_dt_charge_product == pytest.approx(0.00182209718)
    assert "fusion_flavor_aligned_loop" in report.scalar_required_product_source


def test_raw_unpolarized_trace_cancels_but_quartet_projector_survives(report) -> None:
    audit = report.spin_average

    assert audit.spin_space_dimension == 6
    assert audit.quartet_degeneracy == 4
    assert audit.doublet_degeneracy == 2
    assert audit.quartet_operator_eigenvalue == 1.0
    assert audit.doublet_operator_eigenvalue == -2.0
    assert audit.raw_unpolarized_operator_trace == 0.0
    assert audit.quartet_projector_formula == "P_3/2=(Sigma_D.Sigma_T+2)/3"
    assert audit.quartet_projected_unpolarized_trace == pytest.approx(2.0 / 3.0)
    assert audit.quartet_conditional_operator_expectation == 1.0
    assert audit.raw_unpolarized_first_order_spin_response_cancels
    assert audit.dt_s_wave_three_half_resonance_dominates
    assert not audit.exact_ncsmc_or_rmatrix_response_supplied


def test_pseudoscalar_match_is_strong_and_fail_closed(report) -> None:
    audit = report.pseudoscalar

    assert audit.required_abs_effective_nuclear_coupling_product == pytest.approx(
        131.05613,
        rel=1.0e-7,
    )
    assert audit.equal_abs_effective_nuclear_coupling == pytest.approx(
        11.447975,
        rel=1.0e-7,
    )
    assert audit.equal_coupling_fine_structure == pytest.approx(10.429116, rel=1.0e-7)
    assert audit.quartet_attractive_product_sign == -1
    assert audit.same_sign_quartet_force_is_repulsive
    assert audit.raw_unpolarized_trace_cancels
    assert audit.quartet_projected_first_order_term_survives
    assert not audit.perturbative_one_boson_exchange
    assert not audit.nuclear_pseudoscalar_form_factors_supplied
    assert not audit.exact_resonance_response_supplied
    assert not audit.physical_gate_pass


def test_axial_match_is_perturbative_math_but_not_constraint_closed(report) -> None:
    audit = report.axial_vector

    assert audit.required_effective_nuclear_coupling_product == pytest.approx(
        0.00273314576,
    )
    assert audit.equal_effective_nuclear_coupling == pytest.approx(0.052279497)
    assert audit.quartet_attractive_product_sign == 1
    assert audit.universal_quark_axial_bound_at_mass == pytest.approx(8.894271e-7)
    assert audit.naive_nuclear_coupling_to_quark_bound_ratio > 5.8e4
    assert "universal" in audit.universal_bound_scope
    assert not audit.nuclear_to_quark_matching_supplied
    assert not audit.nonuniversal_flavor_cancellation_supplied
    assert not audit.mass_specific_kaon_likelihood_supplied
    assert not audit.physical_gate_pass


def test_vector_has_attractive_minimax_and_zero_momentum_pb_blind_solutions(
    report,
) -> None:
    audit = report.vector

    assert audit.universal_same_sign_vector_is_repulsive
    assert audit.minimax_gp_over_gn == pytest.approx(-4.0 / 3.0)
    assert audit.minimax_proton_coupling == pytest.approx(0.120734326)
    assert audit.minimax_neutron_coupling == pytest.approx(-0.0905507443)
    assert audit.minimax_max_abs_nucleon_coupling == pytest.approx(
        (8.0 * report.required_dt_charge_product) ** 0.5
    )

    assert audit.lead_blind_gp_over_gn == pytest.approx(-126.0 / 82.0)
    assert audit.lead_blind_proton_coupling == pytest.approx(0.131534045)
    assert audit.lead_blind_neutron_coupling == pytest.approx(-0.0856015214)
    assert audit.lead_zero_momentum_charge == pytest.approx(0.0, abs=1.0e-14)
    assert audit.lead_blind_dt_charge_product == pytest.approx(-report.required_dt_charge_product)
    assert audit.lead_blind_isovector_quark_coupling == pytest.approx(0.217135566)
    assert audit.lead_blind_is_attractive_for_dt
    assert audit.lead_cancellation_is_zero_momentum_only
    assert audit.na48_mass_window_contains_candidate
    assert audit.required_to_prompt_visible_proxy_ratio > 540.0
    assert not audit.prompt_visible_proxy_is_mass_specific
    assert not audit.finite_momentum_pb_likelihood_supplied
    assert not audit.mass_specific_pion_kaon_likelihood_supplied
    assert not audit.anomaly_free_gauge_completion_supplied
    assert not audit.physical_gate_pass


def test_spin_two_match_exceeds_universal_babar_proxies(report) -> None:
    audit = report.spin_two

    assert audit.required_equal_c_over_lambda_per_gev == pytest.approx(0.022776659)
    assert audit.required_lambda_over_c_gev == pytest.approx(43.904596)
    assert audit.optimistic_dRGT_strong_coupling_scale_gev == pytest.approx(0.33793206)
    assert audit.required_to_visible_bound_ratio > 750.0
    assert audit.required_to_invisible_bound_ratio > 110.0
    assert audit.babar_bounds_require_universal_electron_stress_energy_coupling
    assert not audit.mass_specific_babar_likelihood_supplied
    assert not audit.nonuniversal_conserved_uv_completion_supplied
    assert not audit.physical_gate_pass


def test_derivative_on_shell_node_removes_yukawa_pole(report) -> None:
    audit = report.derivative_node

    assert audit.mediator_pole_invariant_mev2 == pytest.approx(878.9784069)
    assert audit.yukawa_range_fm == pytest.approx(6.65575561)
    assert audit.dt_reduced_mass_mev == pytest.approx(1124.647349)
    assert audit.incoming_gamow_momentum_mev == pytest.approx(8.33955587)
    assert audit.on_shell_node_cancels_yukawa_pole_residue
    assert audit.eom_operator_reduces_to_contact_interaction
    assert not audit.contact_interaction_lowers_long_range_barrier
    assert not audit.broad_spacelike_pb_node_demonstrated
    assert not audit.additional_light_pole_or_mediator_supplied
    assert not audit.mass_specific_differential_likelihood_supplied
    assert not audit.physical_gate_pass


def test_report_is_exhaustive_and_physical_gate_is_always_fail_closed(report) -> None:
    audits = (
        report.pseudoscalar,
        report.axial_vector,
        report.vector,
        report.spin_two,
        report.derivative_node,
    )

    assert report.all_declared_operator_math_audited
    assert all(not audit.physical_gate_pass for audit in audits)
    assert not report.exact_ncsmc_or_rmatrix_calculation_supplied
    assert not report.mass_specific_pion_kaon_babar_likelihoods_supplied
    assert not report.any_physical_operator_gate_pass
    assert not report.physical_one_percent_fusion_branch_accepted
    assert report.maximum_supported_stage == "OPERATOR_LEVEL_MATCHES_ONLY_FAIL_CLOSED"


@pytest.mark.parametrize(
    "function",
    [
        audit_pseudoscalar_operator,
        audit_axial_vector_operator,
        audit_vector_operator,
        audit_spin_two_operator,
    ],
)
@pytest.mark.parametrize("bad_product", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_required_products_fail_closed(function, bad_product) -> None:
    with pytest.raises(ValueError):
        function(bad_product)
