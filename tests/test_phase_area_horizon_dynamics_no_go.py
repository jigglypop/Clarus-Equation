import math

import pytest

from examples.physics.cosmological_constant_holographic_gate import OMEGA_LAMBDA
from examples.physics.phase_area_horizon_dynamics_no_go import (
    PHASE_AREA_COEFFICIENT,
    apparent_horizon_log_entropy_relative,
    audit_boundary_label_phase_area,
    audit_phase_area_horizon_end_to_end,
    audit_phase_area_inputs,
    audit_physical_efold_phase_area,
    hubble_ratio_from_phase_label,
    phase_label_for_hubble_ratio,
)


def test_relative_horizon_entropy_and_phase_label_are_invertible() -> None:
    ratio = 2.75
    expected_log_entropy = -2.0 * math.log(ratio)

    assert math.isclose(
        apparent_horizon_log_entropy_relative(ratio),
        expected_log_entropy,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    label = phase_label_for_hubble_ratio(ratio)
    assert math.isclose(
        hubble_ratio_from_phase_label(label),
        ratio,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )


def test_physical_efold_interpretation_is_decelerating() -> None:
    audit = audit_physical_efold_phase_area()

    assert math.isclose(
        audit.epsilon_h,
        math.pi**2 / 4.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        audit.effective_w_flat_gr,
        -1.0 + math.pi**2 / 6.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        audit.deceleration_parameter,
        math.pi**2 / 4.0 - 1.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        audit.power_law_scale_factor_exponent,
        4.0 / math.pi**2,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert not audit.accelerates
    assert not audit.compatible_with_late_dark_energy_acceleration


def test_exact_de_sitter_is_a_counterexample_to_positive_universal_slope() -> None:
    audit = audit_physical_efold_phase_area()

    assert audit.entropy_slope_per_physical_efold == PHASE_AREA_COEFFICIENT
    assert audit.exact_de_sitter_entropy_slope == 0.0
    assert not audit.compatible_with_exact_de_sitter
    assert audit.entropy_growth_law_is_adopted_axiom
    assert not audit.unique_dark_energy_prediction


def test_boundary_label_reconstructs_distinct_histories_without_selecting_one() -> None:
    audit = audit_boundary_label_phase_area(z=1.0)

    assert audit.all_histories_reconstructed
    assert audit.histories_are_distinct
    assert len(audit.witnesses) == 2
    assert all(
        abs(item.reconstruction_residual) <= 1.0e-14
        for item in audit.witnesses
    )
    assert not audit.phase_relation_selects_one_history
    assert not audit.physical_efold_map_derived
    assert not audit.unique_hubble_history


def test_phase_area_scale_keeps_supplied_counts_epoch_and_omega() -> None:
    audit = audit_phase_area_inputs()
    xi = PHASE_AREA_COEFFICIENT

    assert math.isclose(audit.dln_h_d_d_eff, -9.0 * xi)
    assert math.isclose(audit.dln_density_d_d_eff, -18.0 * xi)
    assert math.isclose(audit.dln_quarter_scale_d_d_eff, -4.5 * xi)
    assert math.isclose(
        audit.density_ratio_for_delta_d_eff_0p01,
        math.exp(-0.18 * xi),
    )
    assert math.isclose(
        audit.mixed_over_true_quarter_scale,
        OMEGA_LAMBDA**0.25,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(audit.true_de_sitter_quarter_scale_mev, 2.4599, abs_tol=5.0e-4)
    assert math.isclose(audit.legacy_mixed_quarter_scale_mev, 2.2412, abs_tol=5.0e-4)
    assert audit.n_gauge_is_supplied
    assert audit.n_e_relation_is_supplied
    assert audit.omega_lambda_is_supplied
    assert audit.legacy_mixed_value_is_target_aware
    assert not audit.absolute_scale_unique


def test_planck_conventions_agree_when_used_consistently() -> None:
    audit = audit_phase_area_inputs()

    assert audit.consistent_planck_convention_coefficient_residual == 0.0
    assert math.isclose(
        audit.wrong_reduced_mass_in_unreduced_formula_factor,
        8.0 * math.pi,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def test_end_to_end_audit_retains_only_kinematic_subclaims() -> None:
    audit = audit_phase_area_horizon_end_to_end()

    assert audit.universal_entropy_growth_parent_refuted
    assert audit.physical_efold_dark_energy_parent_refuted
    assert audit.boundary_label_unique_hubble_parent_refuted
    assert not audit.unique_absolute_dark_energy_prediction
    assert audit.maximum_true_claims
    assert all(reason for _, reason in audit.dimensionless_arguments)
    assert audit.status == (
        'PHASE_AREA_DARK_ENERGY_ROUTE_REFUTED_KINEMATIC_SUBCLAIM_RETAINED'
    )


@pytest.mark.parametrize(
    ('call', 'message'),
    [
        (lambda: apparent_horizon_log_entropy_relative(0.0), 'hubble_ratio'),
        (lambda: phase_label_for_hubble_ratio(1.0, phase_area_coefficient=0.0), 'phase_area_coefficient'),
        (lambda: hubble_ratio_from_phase_label(math.inf), 'phase_label'),
        (lambda: audit_boundary_label_phase_area(z=-1.0), 'z'),
        (lambda: audit_boundary_label_phase_area(omega_m=0.4, omega_lambda=0.7), 'sum to one'),
    ],
)
def test_invalid_inputs_fail_closed(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
