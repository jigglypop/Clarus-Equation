from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.physics.zerod_selection_dark_energy_no_go import (
    audit_rendering_cosmology_no_go,
    audit_scale_supply_routes,
    audit_zerod_selection_dark_energy_end_to_end,
    certify_covariant_counterterm_no_go,
    eds_luminosity_distance_over_c_h0,
    eds_redshift_for_dimensionless_luminosity_distance,
    flat_e_of_z,
    flat_luminosity_distance_over_c_h0,
    quartic_hierarchy_coefficient,
)


ROOT = Path(__file__).resolve().parents[1]
CLOSURE_DIR = (
    ROOT / "paper" / "검증_원장" / "dark_energy_zerod_closure"
)


def test_covariant_counterterm_pair_refutes_absolute_source_uniqueness() -> None:
    witness = certify_covariant_counterterm_no_go()

    assert witness.q_update_model_a == witness.q_update_model_b
    assert witness.selection_recursion_identical
    assert witness.rendering_sequence_model_a == witness.rendering_sequence_model_b
    assert witness.rendering_readout_identical
    assert witness.vacuum_density_a == 0.0
    assert witness.vacuum_density_b == pytest.approx(0.2505916258330312)
    assert witness.hubble_squared_a == pytest.approx(1.0 / 3.0)
    assert witness.hubble_squared_b == pytest.approx(0.41686387527767704)
    assert witness.expansion_history_different
    assert witness.counterexample_complete
    assert witness.local_and_diffeomorphism_invariant
    assert witness.constant_vacuum_stress_bianchi_conserved
    assert not witness.absolute_source_unique_from_q


def test_counterterm_witness_has_dimension_four_source_and_dimension_two_h2() -> None:
    witness = certify_covariant_counterterm_no_go()

    assert witness.q_mass_dimension == 0
    assert witness.reference_mass_dimension == 1
    assert witness.vacuum_density_mass_dimension == 4
    assert witness.hubble_squared_mass_dimension == 2


def test_all_declared_scale_supplies_leave_an_external_input() -> None:
    audit = audit_scale_supply_routes()

    assert audit.required_planck_quartic_coefficient == pytest.approx(
        7.290112162620085e-121
    )
    assert quartic_hierarchy_coefficient(2.25e-3, 2.435e27) == pytest.approx(
        audit.required_planck_quartic_coefficient
    )
    assert audit.matter_tracking_effective_w == pytest.approx(0.0)
    assert audit.matter_tracking_deceleration_parameter == pytest.approx(0.5)
    assert audit.vacuum_pressure_matter_scaling_continuity_residual == pytest.approx(
        -3.0
    )
    assert audit.h_tracking_effective_w == pytest.approx(0.0)
    assert audit.h_tracking_deceleration_parameter == pytest.approx(0.5)
    assert audit.four_volume_mass_dimension == -4
    assert audit.inverse_sqrt_four_volume_mass_dimension == 2
    assert audit.missing_volume_lift_mass_dimension == 2
    assert audit.reference_scales_are_external_inputs
    assert audit.stationary_q_gradient_terms_zero
    assert not audit.planck_coefficient_derived_from_q
    assert not audit.matter_abundance_derived_from_q
    assert not audit.curvature_wilson_functions_derived_from_q
    assert not audit.global_boundary_datum_derived_from_q
    assert not audit.nonlocal_kernel_and_ir_scale_derived_from_q
    assert not audit.unique_dimensionful_scale_supply


def test_flat_distance_kernel_matches_exact_eds_and_declared_lcdm_values() -> None:
    redshifts = (0.1, 0.3, 0.5, 1.0, 1.5, 2.0)
    audit = audit_rendering_cosmology_no_go(redshifts=redshifts)

    assert [point.eds_luminosity_distance_mpc for point in audit.points] == pytest.approx(
        (438.477750, 1368.977205, 2357.697474, 5017.553029, 7870.504248, 10860.617286),
        rel=2.0e-9,
    )
    assert [
        point.target_luminosity_distance_mpc for point in audit.points
    ] == pytest.approx(
        (460.299936, 1552.715931, 2832.938094, 6607.657612, 10909.639416, 15539.586223),
        rel=2.0e-9,
    )
    assert flat_e_of_z(0.0, 1.0, 0.0) == pytest.approx(1.0)
    assert flat_luminosity_distance_over_c_h0(
        1.0, 1.0, 0.0
    ) == pytest.approx(eds_luminosity_distance_over_c_h0(1.0), rel=1.0e-12)


def test_static_rendering_is_calibration_not_dark_energy_distance_shape() -> None:
    audit = audit_rendering_cosmology_no_go()

    assert [point.eds_over_target_distance for point in audit.points] == pytest.approx(
        (0.95259138, 0.88166623, 0.83224462, 0.75935427, 0.72142662, 0.69890003),
        rel=2.0e-8,
    )
    assert audit.rendering_lambda_fixed == pytest.approx(0.9687005155694667)
    assert audit.constant_rendering_modulus_shift == pytest.approx(0.0690523456)
    assert audit.controlled_sequence_fractional_span == pytest.approx(
        0.00126015657
    )
    assert audit.controlled_sequence_modulus_span == pytest.approx(0.00273467254)
    assert audit.distance_ratio_span == pytest.approx(0.25369134499)
    assert audit.constant_factor_low_z_calibrated_shape_effect == 0.0
    assert audit.distance_ratio_is_redshift_dependent
    assert not audit.constant_rendering_matches_all_redshifts
    assert not audit.controlled_sequence_span_sufficient
    assert not audit.observation_only_dark_energy_explanation


def test_opacity_and_redshift_remap_require_target_inverse_functions() -> None:
    audit = audit_rendering_cosmology_no_go()

    assert [point.required_flux_survival for point in audit.points] == pytest.approx(
        (0.90743033, 0.77733535, 0.69263110, 0.57661891, 0.52045636, 0.48846125),
        rel=2.0e-8,
    )
    assert [
        point.required_distance_duality_eta for point in audit.points
    ] == pytest.approx(
        (1.04976806, 1.13421606, 1.20156981, 1.31690838, 1.38614237, 1.43081980),
        rel=2.0e-8,
    )
    assert [point.required_amplitude_steps for point in audit.points] == pytest.approx(
        (1.52734528, 3.96045868, 5.77453274, 8.65687900, 10.26814062, 11.26572452),
        rel=2.0e-8,
    )
    assert [
        point.target_inverse_eds_redshift for point in audit.points
    ] == pytest.approx(
        (0.10486390, 0.33799137, 0.59280009, 1.28208382, 2.00804366, 2.75094158),
        rel=2.0e-8,
    )
    for point in audit.points:
        target_distance = flat_luminosity_distance_over_c_h0(
            point.z, audit.target_omega_m, audit.target_omega_lambda
        )
        inverse_redshift = eds_redshift_for_dimensionless_luminosity_distance(
            target_distance
        )
        assert eds_luminosity_distance_over_c_h0(inverse_redshift) == pytest.approx(
            target_distance, rel=2.0e-13
        )
    assert audit.opacity_requires_distance_duality_violation
    assert audit.target_inverse_redshift_map_required
    assert not audit.benchmark_is_observational_fit
    assert audit.lambda_as_photon_amplitude_requires_new_axiom
    assert not audit.event_depth_to_redshift_derived
    assert not audit.photon_survival_law_derived
    assert not audit.growth_and_lensing_changed_by_readout


def test_end_to_end_claim_ceiling_and_dimensionless_contract() -> None:
    audit = audit_zerod_selection_dark_energy_end_to_end()

    assert audit.counterterm.counterexample_complete
    assert not audit.scale_supply.unique_dimensionful_scale_supply
    assert not audit.rendering.observation_only_dark_energy_explanation
    assert len(audit.pivots_tested) >= 3
    assert len(set(audit.pivots_tested)) == len(audit.pivots_tested)
    assert len(audit.escape_requirements) == 4
    assert audit.dimensionless_core_arguments == (
        ("D * (1 - q)", "D and q are dimensionless"),
        ("1 - delta/d - q**2", "delta, d, and q are dimensionless"),
        ("H0 * dL / c", "distance is normalized by c/H0"),
        ("low_scale / high_scale", "both scales use the same energy unit"),
    )
    assert audit.stationary_selection_absolute_source_parent_refuted
    assert audit.static_rendering_full_dark_energy_parent_refuted
    assert audit.direct_q_to_hubble_parent_refuted
    assert not audit.unique_dark_energy_prediction


def test_counterexample_and_pivot_portfolio_are_locked_to_live_witness() -> None:
    counterexample = json.loads(
        (CLOSURE_DIR / "counterexample.json").read_text(encoding="utf-8")
    )
    portfolio = json.loads(
        (CLOSURE_DIR / "portfolio.json").read_text(encoding="utf-8")
    )
    witness = certify_covariant_counterterm_no_go()

    assert counterexample["parent_claim_id"] == "DE-0D-ABSOLUTE-UNIQUE"
    assert counterexample["formal_status"] == "REFUTED_WITHIN_DECLARED_MODEL_CLASS"
    assert counterexample["common_inputs"]["q"] == pytest.approx(witness.q)
    assert counterexample["common_inputs"]["D"] == pytest.approx(witness.d_eff)
    assert counterexample["model_a"]["rho_vac"] == pytest.approx(
        witness.vacuum_density_a
    )
    assert counterexample["model_b"]["rho_vac"] == pytest.approx(
        witness.vacuum_density_b
    )
    assert counterexample["model_a"]["H_squared"] == pytest.approx(
        witness.hubble_squared_a
    )
    assert counterexample["model_b"]["H_squared"] == pytest.approx(
        witness.hubble_squared_b
    )
    assert counterexample["counterexample_complete"]

    routes = portfolio["routes"]
    fingerprints = {
        tuple(route["mechanism_fingerprint"]) for route in routes
    }
    assert len(routes) >= 3
    assert len(fingerprints) == len(routes)
    assert portfolio["declared_class_exhausted"]
    assert not portfolio["absolute_dark_energy_prediction"]
    assert len(portfolio["escape_requirements_all_required"]) == 4


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: certify_covariant_counterterm_no_go(q=-0.1), "q"),
        (
            lambda: certify_covariant_counterterm_no_go(reference_mass_scale=0.0),
            "reference_mass_scale",
        ),
        (
            lambda: audit_rendering_cosmology_no_go(redshifts=(0.5, 0.3)),
            "strictly increasing",
        ),
        (
            lambda: audit_rendering_cosmology_no_go(
                target_omega_m=0.3, target_omega_lambda=0.6
            ),
            "flat",
        ),
        (
            lambda: audit_rendering_cosmology_no_go(rendering_lambda=1.0),
            "below one",
        ),
        (
            lambda: audit_rendering_cosmology_no_go(integration_intervals=3),
            "positive even",
        ),
    ],
)
def test_end_to_end_inputs_fail_closed(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
