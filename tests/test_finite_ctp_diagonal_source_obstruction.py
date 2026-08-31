from __future__ import annotations

import math

import pytest

from examples.physics.finite_ctp_diagonal_source_obstruction import (
    certify_finite_ctp_diagonal_source_obstruction,
    influence,
    influence_action,
)


def test_default_ctp_diagonal_and_difference_derivatives() -> None:
    certificate = certify_finite_ctp_diagonal_source_obstruction()
    assert certificate.influence == pytest.approx(0.99401997335 - 0.05960079924j, abs=2.0e-11)
    assert certificate.influence_diagonal_residual == 0.0
    assert certificate.action_diagonal_residual == 0.0
    assert certificate.h_c_derivative_at_diagonal == 0.0
    assert certificate.difference_source == pytest.approx(-0.6)
    assert certificate.central_difference_source == pytest.approx(-0.6, abs=2.0e-10)
    assert certificate.central_difference_residual < 2.0e-10


def test_local_ctp_expansion_has_unitary_linear_and_imaginary_noise_terms() -> None:
    certificate = certify_finite_ctp_diagonal_source_obstruction()
    assert certificate.linear_action_coefficient == pytest.approx(-0.6)
    assert certificate.quadratic_imaginary_action_coefficient == pytest.approx(0.42)
    assert certificate.symmetric_quadratic_coefficient.real == pytest.approx(0.0, abs=2.0e-9)
    # Symmetric finite differences retain an O(step**2) truncation term.
    assert certificate.symmetric_quadratic_coefficient.imag == pytest.approx(0.42, abs=1.0e-7)
    assert certificate.local_expansion_residual < 2.0e-4


def test_same_diagonal_readout_does_not_identify_difference_source() -> None:
    certificate = certify_finite_ctp_diagonal_source_obstruction()
    assert certificate.diagonal_readout_probabilities == (0.7, 0.3)
    assert certificate.diagonal_model_influence_residual == 0.0
    assert certificate.model_zero_difference_source == 0.0
    assert certificate.model_nonzero_difference_source == pytest.approx(-0.6)
    assert certificate.model_reference_frequency_residual == 0.0
    assert certificate.model_reference_hamiltonian_residual == 0.0
    assert certificate.limited_non_identifiability


def test_two_label_dilation_is_cptp_and_the_plus_witness_is_trace_preserving() -> None:
    certificate = certify_finite_ctp_diagonal_source_obstruction()
    assert certificate.environment_minimum_eigenvalue >= 0.0
    assert certificate.environment_trace_residual == 0.0
    assert certificate.controlled_unitary_residual < 2.0e-14
    assert certificate.gram_minimum_eigenvalue >= -2.0e-13
    assert certificate.gram_diagonal_residual < 2.0e-14
    assert certificate.schur_choi_minimum_eigenvalue >= -2.0e-13
    assert certificate.schur_trace_preservation_residual < 2.0e-14
    assert certificate.schur_output_trace_residual < 2.0e-14
    assert certificate.schur_completely_positive
    assert certificate.schur_trace_preserving
    assert abs(certificate.plus_state_coherence) < 0.5


def test_negative_controls_and_pure_environment_noise_limit() -> None:
    certificate = certify_finite_ctp_diagonal_source_obstruction()
    assert certificate.p_zero_source == certificate.tau_zero_source == certificate.slope_zero_source == 0.0
    assert certificate.p_zero_decoherence == certificate.tau_zero_decoherence == certificate.slope_zero_decoherence == 0.0
    assert certificate.p_one_quadratic_noise_coefficient == 0.0
    assert certificate.p_one_unitary_phase_present


def test_dimension_accounting_and_status_ceiling() -> None:
    certificate = certify_finite_ctp_diagonal_source_obstruction()
    assert certificate.dimensions_pass
    assert certificate.h_mass_dimension == 0
    assert certificate.omega_mass_dimension == certificate.slope_mass_dimension == 1
    assert certificate.tau_mass_dimension == -1
    assert certificate.tau_omega_mass_dimension == 0
    assert certificate.influence_mass_dimension == certificate.action_over_hbar_mass_dimension == 0
    assert certificate.difference_source_dimension == "action"
    assert certificate.accounting_mode == "integrated_out_influence_only"
    assert not certificate.retained_environment_stress_added
    assert not certificate.rn_reweighting_used
    assert not any((
        certificate.tensor_stress_derived, certificate.diffeo_Ward, certificate.retarded_causality,
        certificate.microcausality_or_c_front, certificate.attraction, certificate.mass_to_source,
        certificate.GR_lensing_spin2, certificate.energy_backreaction,
        certificate.physical_observation_or_selection, certificate.observational_holdout,
        certificate.gates_5_to_8, certificate.two_residuals, certificate.complexity_success,
    ))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"p": -0.1}, "p"), ({"p": 1.1}, "p"), ({"tau": -1.0}, "tau"),
        ({"hbar": 0.0}, "hbar"), ({"finite_difference_step": 0.0}, "finite_difference_step"),
        ({"slope": float("nan")}, "slope"),
    ],
)
def test_certificate_input_contract_fails_closed(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        certify_finite_ctp_diagonal_source_obstruction(**kwargs)


def test_zero_influence_and_unsafe_principal_branch_are_rejected_off_diagonal() -> None:
    with pytest.raises(ValueError, match="zero"):
        influence(0.5, -0.5, p=0.5, tau=1.0, omega_star=0.0, slope=math.pi, h_star=0.0)
    with pytest.raises(ValueError, match="principal"):
        influence_action(0.5, -0.5, p=1.0, tau=1.0, hbar=1.0, omega_star=0.0, slope=math.pi, h_star=0.0)
