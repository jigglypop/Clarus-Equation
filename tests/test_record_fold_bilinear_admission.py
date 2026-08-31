from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.record_fold_bilinear_admission import (
    CANONICAL_BOUNDARY_PARAMETERS,
    CANONICAL_STABLE_PARAMETERS,
    CANONICAL_TACHYON_PARAMETERS,
    basis_obstruction_audit,
    bilinear_spectrum_audit,
    canonical_on_shell_ward_audit,
    certificate,
    dimension_audit,
    require_stable_spectrum,
    run,
    schur_complement_audit,
    source_accounting_audit,
    ward_exchange_audit,
)


def test_dimension_ledger_does_not_convert_probability_into_energy() -> None:
    receipt = dimension_audit()

    assert receipt.record_field_mass_dimension == 1
    assert receipt.fold_field_mass_dimension == 1
    assert receipt.mixing_kappa_mass_dimension == 2
    assert receipt.source_coefficient_mass_dimension == 3
    assert receipt.lagrangian_density_mass_dimension == 4
    assert receipt.stress_mass_dimension == 4
    assert receipt.ward_current_mass_dimension == 5
    assert receipt.action_mass_dimension == 0
    assert receipt.dimensions_pass
    assert not receipt.probability_used_as_source_coefficient


def test_stable_witness_has_the_analytic_eigenmass_squared_values() -> None:
    receipt = bilinear_spectrum_audit(*CANONICAL_STABLE_PARAMETERS)
    expected_high = 0.5 * (13.0 + math.sqrt(41.0))
    expected_low = 0.5 * (13.0 - math.sqrt(41.0))
    rotated = np.asarray(receipt.rotated_mass_squared_matrix)

    assert receipt.determinant_mass_four == pytest.approx(32.0)
    assert receipt.eigenmass_squared_high == pytest.approx(expected_high)
    assert receipt.eigenmass_squared_low == pytest.approx(expected_low)
    assert receipt.positive_by_principal_minors
    assert receipt.strictly_stable
    assert not receipt.tachyonic_mode_present
    assert not receipt.boundary_zero_mode_present
    assert receipt.canonical_kinetic_ghost_free
    assert receipt.rotated_off_diagonal_residual < 1.0e-12
    assert receipt.kinetic_rotation_residual < 1.0e-12
    assert sorted(np.diag(rotated)) == pytest.approx(
        sorted((expected_low, expected_high))
    )


def test_tachyon_and_zero_mode_counterexamples_fail_closed() -> None:
    tachyon = bilinear_spectrum_audit(*CANONICAL_TACHYON_PARAMETERS)
    boundary = bilinear_spectrum_audit(*CANONICAL_BOUNDARY_PARAMETERS)

    assert tachyon.determinant_mass_four == pytest.approx(-3.0)
    assert tachyon.eigenmass_squared_high == pytest.approx(3.0)
    assert tachyon.eigenmass_squared_low == pytest.approx(-1.0)
    assert tachyon.tachyonic_mode_present
    assert not tachyon.strictly_stable
    assert boundary.determinant_mass_four == pytest.approx(0.0)
    assert boundary.eigenmass_squared_low == pytest.approx(0.0)
    assert boundary.boundary_zero_mode_present
    assert not boundary.strictly_stable
    with pytest.raises(ValueError, match="not strictly stable"):
        require_stable_spectrum(*CANONICAL_TACHYON_PARAMETERS)
    with pytest.raises(ValueError, match="not strictly stable"):
        require_stable_spectrum(*CANONICAL_BOUNDARY_PARAMETERS)


def test_on_shell_ward_witness_has_equal_and_opposite_exchange() -> None:
    receipt = canonical_on_shell_ward_audit()
    free_fold = np.asarray(receipt.free_fold_stress_divergence)
    record_and_interaction = np.asarray(
        receipt.record_plus_interaction_divergence
    )

    assert receipt.source_coefficient == pytest.approx(1.0)
    assert receipt.record_eom_residual == pytest.approx(0.0)
    assert receipt.fold_eom_residual == pytest.approx(0.0)
    assert np.allclose(free_fold, -record_and_interaction)
    assert np.allclose(receipt.total_stress_divergence, np.zeros(4))
    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_exchange_balance_residual < 1.0e-12
    assert receipt.both_field_equations_on_shell
    assert receipt.total_stress_conserved_on_shell
    assert receipt.interaction_counted_once


def test_off_shell_ward_identity_is_not_misreported_as_conservation() -> None:
    receipt = ward_exchange_audit(
        record_value=0.5,
        fold_value=-0.25,
        box_record=0.0,
        box_fold=0.0,
        record_gradient_covector=(0.3, -0.2, 0.1, 0.0),
        fold_gradient_covector=(-0.4, 0.05, 0.0, 0.2),
    )

    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_total_divergence > 0.0
    assert not receipt.both_field_equations_on_shell
    assert not receipt.total_stress_conserved_on_shell


def test_action_sign_convention_puts_minus_j_on_the_fold_rhs() -> None:
    receipt = ward_exchange_audit(
        record_value=0.5,
        fold_value=0.0,
        box_record=4.5,
        box_fold=-1.0,
        record_gradient_covector=(1.0, 0.0, 0.0, 0.0),
        fold_gradient_covector=(0.0, 1.0, 0.0, 0.0),
    )

    assert receipt.source_coefficient == pytest.approx(1.0)
    assert receipt.fold_eom_residual == pytest.approx(0.0)
    assert receipt.free_fold_stress_divergence == pytest.approx(
        (0.0, -1.0, 0.0, 0.0)
    )


def test_static_schur_complement_is_positive_only_inside_the_stable_witness() -> None:
    stable = schur_complement_audit(*CANONICAL_STABLE_PARAMETERS)
    tachyon = schur_complement_audit(*CANONICAL_TACHYON_PARAMETERS)

    assert stable.static_effective_fold_mass_squared == pytest.approx(32.0 / 9.0)
    assert stable.determinant_over_record_mass_squared == pytest.approx(32.0 / 9.0)
    assert stable.positive_static_effective_mass
    assert stable.operator_kernel == "D_phi - kappa^2 D_R^{-1}"
    assert stable.zero_momentum_local_formula_only
    assert stable.inverse_boundary_or_state_prescription_required
    assert not stable.retarded_inverse_automatically_selected
    assert not stable.closed_time_path_noise_derived
    assert not stable.local_effective_stress_automatically_derived
    assert tachyon.static_effective_fold_mass_squared == pytest.approx(-3.0)
    assert not tachyon.positive_static_effective_mass


def test_retained_and_integrated_out_source_ledgers_are_exclusive() -> None:
    retained = source_accounting_audit("retained_fields")
    influence = source_accounting_audit("integrated_out_influence")

    assert retained.retained_record_and_fold_fields
    assert retained.original_bilinear_interaction_retained
    assert not retained.integrated_out_influence_kernel
    assert influence.integrated_out_influence_kernel
    assert not influence.retained_record_and_fold_fields
    assert not influence.original_bilinear_interaction_retained
    for receipt in (retained, influence):
        assert receipt.mutually_exclusive_representations
        assert not receipt.probability_rebooked_as_energy
        assert not receipt.source_stress_counted_twice
    with pytest.raises(ValueError, match="unknown source accounting mode"):
        source_accounting_audit("retained_plus_influence")


def test_basis_rotation_removes_mixing_but_does_not_select_a_record() -> None:
    receipt = basis_obstruction_audit()

    assert receipt.eigenmass_squared_set == pytest.approx((4.0, 6.0))
    assert receipt.absolute_rotation_angle_degrees == pytest.approx(45.0)
    assert receipt.rotated_off_diagonal_residual < 1.0e-12
    assert receipt.kinetic_rotation_residual < 1.0e-12
    assert receipt.hypothetical_pointer_vector_eigenbasis == pytest.approx(
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    )
    assert receipt.hypothetical_pointer_is_extra_input
    assert receipt.eigenmass_squared_set_basis_invariant
    assert not receipt.record_and_fold_labels_basis_invariant
    assert not receipt.bilinear_mixing_selects_pointer_basis
    assert not receipt.bilinear_mixing_derives_observed_outcome
    assert not receipt.bilinear_mixing_derives_dark_source


def test_zero_mixing_decouples_classical_equations_but_does_not_prove_gr_limit() -> None:
    spectrum = bilinear_spectrum_audit(9.0, 4.0, 0.0)
    top = certificate()

    assert spectrum.mass_squared_matrix == ((9.0, -0.0), (-0.0, 4.0))
    assert spectrum.strictly_stable
    assert not top.zero_stress_qm_gr_limit_derived
    assert not top.gravitational_solution_derived


def test_certificate_preserves_all_physical_claim_ceilings() -> None:
    receipt = certificate()

    assert receipt.status == "CONDITIONAL_CLASSICAL_TWO_FIELD_ADMISSION"
    assert receipt.one_total_action_accounting_admitted
    assert receipt.classical_principal_symbol_uses_metric_cone
    assert not receipt.nonselected_quantum_to_record_map_derived
    assert not receipt.pointer_selection_and_durable_record_derived
    assert not receipt.probability_deformation_defined
    assert not receipt.cptp_and_normalization_derived
    assert not receipt.qft_microcausality_derived
    assert not receipt.operational_no_signalling_derived
    assert not receipt.fixed_parameter_manifest_established
    assert not receipt.independent_holdout_prediction_derived
    assert not receipt.two_residual_classes_reduced
    assert not receipt.complexity_penalized_improvement_established
    assert "=-J_ns" in receipt.source_sign_convention


def test_invalid_numeric_inputs_fail_closed_and_run_is_serializable() -> None:
    with pytest.raises(ValueError, match="record_mass_squared"):
        bilinear_spectrum_audit(float("nan"), 4.0, 2.0)
    with pytest.raises(ValueError, match="record_mass_squared"):
        schur_complement_audit(0.0, 4.0, 2.0)
    with pytest.raises(ValueError, match="four finite"):
        ward_exchange_audit(
            record_value=0.0,
            fold_value=0.0,
            box_record=0.0,
            box_fold=0.0,
            record_gradient_covector=(1.0, 2.0),
            fold_gradient_covector=(0.0, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="reference_mass_scale"):
        ward_exchange_audit(
            record_value=0.0,
            fold_value=0.0,
            box_record=0.0,
            box_fold=0.0,
            record_gradient_covector=(0.0, 0.0, 0.0, 0.0),
            fold_gradient_covector=(0.0, 0.0, 0.0, 0.0),
            reference_mass_scale=0.0,
        )
    payload = run()
    assert payload["status"] == "CONDITIONAL_CLASSICAL_TWO_FIELD_ADMISSION"
    assert payload["stable_witness"]["strictly_stable"]
