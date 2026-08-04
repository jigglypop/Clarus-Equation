from __future__ import annotations

import math

import pytest

from reality_stone.clarus.ce_two_point_vertex_certificate import (
    CONTROL_SCOPE,
    CONTROL_STATUS,
    DEFAULT_LAMBDA_HP,
    audit_hessian_vertex_nonidentifiability,
    audit_inverse_correlation_identifiability,
    audit_invisible_width_constraint,
    audit_light_pole_portal_compatibility,
    audit_portal_tree_vacuum,
    audit_tree_level_two_point,
    audit_z2_portal_vertices,
    ce_light_pole_q04_q05_certificate,
)
from reality_stone.clarus.q0_manifest_gate import (
    NOT_APPLIED_STATUS,
    q0_control_action_definition_payload,
    q0_control_action_definition_sha256,
)


def test_portal_certificate_is_bound_to_canonical_q0_action_definition() -> None:
    digest = q0_control_action_definition_sha256()
    payload = q0_control_action_definition_payload()
    report = ce_light_pole_q04_q05_certificate()

    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert payload["renormalization_status"] == NOT_APPLIED_STATUS
    assert payload["full_ce_sm_complete"] is False
    assert report.action_definition_sha256 == digest
    assert report.two_point.action_definition_sha256 == digest


def test_inverse_length_does_not_identify_pole_residue_or_lsz() -> None:
    audit = audit_inverse_correlation_identifiability(
        inverse_correlation_scale_gev=0.02964757,
    )

    assert audit.all_models_share_leading_exponential_scale
    assert audit.correlation_length_fm == pytest.approx(6.6557556, rel=1.0e-7)
    assert [model.isolated_delta_pole for model in audit.countermodels] == [
        True,
        True,
        True,
        False,
    ]
    assert [model.invariant_pole_residue for model in audit.countermodels] == [
        1.0,
        1.0e-12,
        -1.0,
        None,
    ]
    assert not audit.isolated_pole_identified_by_inverse_length
    assert not audit.residue_magnitude_identified_by_inverse_length
    assert not audit.residue_sign_identified_by_inverse_length
    assert not audit.reflection_positivity_identified_by_inverse_length
    assert not audit.lsz_particle_identified_by_inverse_length
    assert audit.maximum_supported_stage == "INVERSE_CORRELATION_SCALE_ANSATZ"


def test_same_gradient_and_hessian_allow_arbitrary_cubic_and_quartic_vertices() -> None:
    audit = audit_hessian_vertex_nonidentifiability(
        cubic_deformation=7.0,
        quartic_deformation=-11.0,
    )

    assert audit.deformation_gradient_at_background == 0.0
    assert audit.deformation_hessian_at_background == 0.0
    assert audit.deformation_cubic_derivative_at_background == 7.0
    assert audit.deformation_quartic_derivative_at_background == -11.0
    assert audit.background_gradient_unchanged
    assert audit.background_hessian_unchanged
    assert audit.cubic_vertex_changed
    assert audit.quartic_vertex_changed
    assert not audit.cubic_vertex_identified_by_hessian
    assert not audit.quartic_vertex_identified_by_hessian


def test_back_solved_tree_kernel_has_simple_positive_pole_and_dispersion() -> None:
    target_gev = 0.02964757
    portal_shift = DEFAULT_LAMBDA_HP * 246.22**2
    audit = audit_tree_level_two_point(
        bare_mass_squared_gev2=target_gev**2 - portal_shift,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        spatial_momentum_gev=3.0,
    )

    assert audit.pole_mass_gev == pytest.approx(target_gev, abs=2.0e-12)
    assert audit.positive_energy_gev == pytest.approx(math.sqrt(3.0**2 + target_gev**2))
    assert audit.on_shell_invariant_gev2 == pytest.approx(target_gev**2)
    assert audit.on_shell_kernel_residual_gev2 is not None
    assert audit.on_shell_kernel_residual_gev2 < 1.0e-12
    assert audit.invariant_pole_residue == 1.0
    assert audit.positive_energy_pole_residue_gev_inv == pytest.approx(
        1.0 / (2.0 * audit.positive_energy_gev)
    )
    assert audit.tachyon_free
    assert audit.isolated_massive_tree_pole
    assert audit.positive_tree_residue
    assert audit.relativistic_dispersion_identity_pass
    assert audit.tree_level_local_pole_candidate
    assert not audit.renormalized_two_point_derived
    assert not audit.full_ce_two_point_derived
    assert not audit.lsz_completed


def test_negative_effective_mass_squared_is_tachyon_not_physical_pole() -> None:
    audit = audit_tree_level_two_point(
        bare_mass_squared_gev2=-1.0,
        lambda_hp=0.0,
        higgs_vev_gev=246.22,
    )

    assert not audit.tachyon_free
    assert audit.pole_mass_gev is None
    assert audit.positive_energy_gev is None
    assert not audit.isolated_massive_tree_pole
    assert not audit.tree_level_local_pole_candidate


def test_z2_portal_has_pair_vertices_but_no_bilinear_or_single_phi_source() -> None:
    audit = audit_z2_portal_vertices(
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        singlet_self_coupling=0.1,
    )

    assert audit.h_phi_cross_hessian_gev == 0.0
    assert audit.h_phi_phi_derivative_gev == pytest.approx(-2.0 * DEFAULT_LAMBDA_HP * 246.22)
    assert audit.h_h_phi_phi_derivative == pytest.approx(-2.0 * DEFAULT_LAMBDA_HP)
    assert audit.chi_chi_phi_phi_derivative == pytest.approx(-2.0 * DEFAULT_LAMBDA_HP)
    assert audit.phi_four_derivative == pytest.approx(-0.6)
    assert audit.z2_odd_vacuum_derivatives_zero
    assert audit.bilinear_h_phi_mixing_zero
    assert audit.h_phi_phi_pair_vertex_present
    assert audit.h_h_phi_phi_pair_vertex_present
    assert audit.chi_chi_phi_phi_pair_vertex_present
    assert audit.local_derivative_identities_pass
    assert not audit.single_phi_source_derived
    assert not audit.direct_phi_squared_daughter_squared_vertex_derived


def test_zero_portal_coupling_removes_portal_vertices() -> None:
    audit = audit_z2_portal_vertices(
        lambda_hp=0.0,
        higgs_vev_gev=246.22,
        singlet_self_coupling=0.1,
    )

    assert not audit.h_phi_phi_pair_vertex_present
    assert not audit.h_h_phi_phi_pair_vertex_present
    assert not audit.chi_chi_phi_phi_pair_vertex_present
    assert audit.phi_four_derivative == pytest.approx(-0.6)
    assert audit.local_derivative_identities_pass


def test_nonnegative_bare_mass_theorem_excludes_light_same_field_pole() -> None:
    audit = audit_light_pole_portal_compatibility(
        target_mass_gev=0.02964757,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
    )

    assert audit.zero_bare_mass_portal_pole_gev == pytest.approx(
        43.7677,
        rel=2.0e-6,
    )
    assert audit.portal_to_target_mass_ratio > 1400.0
    assert audit.required_bare_mass_squared_gev2 == pytest.approx(
        -1915.6,
        rel=2.0e-4,
    )
    assert audit.required_bare_mass_sign == "negative"
    assert audit.required_bare_to_portal_shift_ratio > 0.999999
    assert audit.target_squared_to_portal_shift_ratio < 5.0e-7
    assert audit.cancellation_decimal_digits > 6.3
    assert audit.maximum_lambda_for_nonnegative_bare_mass < 1.5e-8
    assert audit.supplied_to_nonnegative_bare_lambda_ratio > 2.0e6
    assert not audit.target_reachable_with_nonnegative_bare_mass
    assert audit.target_reachable_with_back_solved_bare_mass
    assert not audit.portal_dominance_satisfied_by_required_bare_mass
    assert not audit.same_field_light_pole_and_portal_dominance_compatible
    assert audit.parameter_cancellation_required
    assert not audit.ce_matching_relation_derived


def test_target_above_portal_floor_can_use_nonnegative_bare_mass() -> None:
    audit = audit_light_pole_portal_compatibility(
        target_mass_gev=50.0,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        portal_dominance_max_bare_to_shift_ratio=1.0,
    )

    assert audit.required_bare_mass_squared_gev2 > 0.0
    assert audit.target_reachable_with_nonnegative_bare_mass
    assert audit.portal_dominance_satisfied_by_required_bare_mass
    assert audit.same_field_light_pole_and_portal_dominance_compatible
    assert not audit.parameter_cancellation_required


def test_small_negative_bare_mass_can_still_be_portal_dominated() -> None:
    portal_mass = 246.22 * math.sqrt(DEFAULT_LAMBDA_HP)
    audit = audit_light_pole_portal_compatibility(
        target_mass_gev=0.98 * portal_mass,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        portal_dominance_max_bare_to_shift_ratio=0.1,
    )

    assert audit.required_bare_mass_squared_gev2 < 0.0
    assert not audit.target_reachable_with_nonnegative_bare_mass
    assert audit.portal_dominance_satisfied_by_required_bare_mass
    assert audit.same_field_light_pole_and_portal_dominance_compatible


def test_back_solved_negative_bare_mass_can_still_have_z2_tree_vacuum() -> None:
    compatibility = audit_light_pole_portal_compatibility(
        target_mass_gev=0.02964757,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
    )
    higgs_quartic = 125.25**2 / (2.0 * 246.22**2)
    audit = audit_portal_tree_vacuum(
        bare_mass_squared_gev2=compatibility.required_bare_mass_squared_gev2,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        higgs_self_coupling=higgs_quartic,
        singlet_self_coupling=0.1,
    )

    assert audit.quartic_potential_bounded
    assert audit.selected_ew_vacuum_local_minimum
    assert audit.singlet_effective_mass_squared_gev2 == pytest.approx(
        0.02964757**2,
        abs=2.0e-12,
    )
    assert audit.singlet_only_stationary_exists
    assert not audit.mixed_stationary_exists
    assert audit.minimum_singlet_self_coupling_against_singlet_only_vacuum == (
        pytest.approx(0.00772, rel=2.0e-3)
    )
    assert audit.selected_ew_vacuum_global_among_tree_stationary_points
    assert audit.selected_vacuum_preserves_z2_despite_negative_bare_mass
    assert not audit.loop_and_thermal_vacuum_stability_derived


def test_too_small_singlet_quartic_loses_global_ew_vacuum() -> None:
    compatibility = audit_light_pole_portal_compatibility(
        target_mass_gev=0.02964757,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
    )
    audit = audit_portal_tree_vacuum(
        bare_mass_squared_gev2=compatibility.required_bare_mass_squared_gev2,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        higgs_self_coupling=125.25**2 / (2.0 * 246.22**2),
        singlet_self_coupling=0.001,
    )

    assert audit.selected_ew_vacuum_local_minimum
    assert not audit.selected_ew_vacuum_global_among_tree_stationary_points
    assert not audit.selected_vacuum_preserves_z2_despite_negative_bare_mass


def test_light_portal_benchmark_fails_supplied_invisible_width_limit() -> None:
    audit = audit_invisible_width_constraint(
        target_mass_gev=0.02964757,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        higgs_mass_gev=125.25,
        sm_higgs_width_gev=0.00407,
        branching_fraction_upper_limit=0.11,
    )

    assert audit.kinematically_open
    assert audit.phase_space_factor > 0.999999
    assert audit.partial_width_gev == pytest.approx(0.01925, rel=2.0e-3)
    assert audit.branching_fraction > 0.82
    assert audit.maximum_allowed_abs_lambda == pytest.approx(
        0.00511,
        rel=2.0e-3,
    )
    assert audit.supplied_to_maximum_coupling_ratio is not None
    assert audit.supplied_to_maximum_coupling_ratio > 6.0
    assert not audit.supplied_benchmark_allowed
    assert audit.limit_supplied_not_derived
    assert not audit.loop_and_global_fit_included


def test_closed_higgs_channel_does_not_bound_coupling_with_this_width_gate() -> None:
    audit = audit_invisible_width_constraint(
        target_mass_gev=70.0,
        lambda_hp=DEFAULT_LAMBDA_HP,
        higgs_vev_gev=246.22,
        higgs_mass_gev=125.25,
        sm_higgs_width_gev=0.00407,
        branching_fraction_upper_limit=0.11,
    )

    assert not audit.kinematically_open
    assert audit.partial_width_gev == 0.0
    assert audit.branching_fraction == 0.0
    assert audit.maximum_allowed_abs_lambda is None
    assert audit.supplied_to_maximum_coupling_ratio is None
    assert audit.supplied_benchmark_allowed


def test_aggregate_constructs_control_pole_without_promoting_ce_claims() -> None:
    report = ce_light_pole_q04_q05_certificate()
    payload = report.to_dict()

    assert report.scope == CONTROL_SCOPE
    assert report.status == CONTROL_STATUS
    assert report.singlet_block_q0_4_tree_control_pass
    assert report.singlet_block_q0_5_tree_control_pass
    assert report.conditional_portal_pair_vertex_derived
    assert report.registered_inverse_correlation_target_is_a_constructible_tree_pole
    assert not report.registered_target_is_predicted_by_portal_action
    assert not report.registered_target_equals_portal_dominated_pole
    assert not report.physical_clarus_pole_derived
    assert not report.renormalized_pole_and_residue_derived
    assert not report.full_lsz_passed
    assert not report.full_ce_production_vertex_derived
    assert not report.physical_sm_production_rate_derived
    assert not report.negative_stress_derived
    assert report.vacuum.selected_ew_vacuum_global_among_tree_stationary_points
    assert payload["physical_clarus_pole_derived"] is False
    assert not report.hessian_vertex_identifiability.cubic_vertex_identified_by_hessian
    assert "back-solving" in report.conclusion


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (
            audit_inverse_correlation_identifiability,
            {"inverse_correlation_scale_gev": True},
            "real scalar",
        ),
        (
            audit_hessian_vertex_nonidentifiability,
            {"cubic_deformation": 0.0, "quartic_deformation": 0.0},
            "at least one vertex deformation",
        ),
        (
            audit_tree_level_two_point,
            {
                "bare_mass_squared_gev2": math.inf,
                "lambda_hp": 0.1,
                "higgs_vev_gev": 246.22,
            },
            "finite",
        ),
        (
            audit_z2_portal_vertices,
            {
                "lambda_hp": -0.1,
                "higgs_vev_gev": 246.22,
                "singlet_self_coupling": 0.1,
            },
            "nonnegative",
        ),
        (
            audit_light_pole_portal_compatibility,
            {
                "target_mass_gev": 0.029,
                "lambda_hp": 0.0,
                "higgs_vev_gev": 246.22,
            },
            "positive",
        ),
        (
            audit_portal_tree_vacuum,
            {
                "bare_mass_squared_gev2": 1.0,
                "lambda_hp": 0.1,
                "higgs_vev_gev": 246.22,
                "higgs_self_coupling": 0.1,
                "singlet_self_coupling": 0.0,
            },
            "positive",
        ),
        (
            audit_invisible_width_constraint,
            {
                "target_mass_gev": 0.029,
                "lambda_hp": 0.01,
                "higgs_vev_gev": 246.22,
                "higgs_mass_gev": 125.25,
                "sm_higgs_width_gev": 0.00407,
                "branching_fraction_upper_limit": 1.0,
            },
            r"in \[0, 1\)",
        ),
    ],
)
def test_certificate_functions_reject_invalid_inputs(
    function: object,
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(**arguments)  # type: ignore[operator]
