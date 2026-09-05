from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.record.finite_ctp_diagonal_source_obstruction import (
    apply_influence_gram,
    audit_controlled_history_observable_expectation,
    audit_joint_environment_slice_memory,
    audit_observer_slice_memory,
    audit_quantum_kick_conservation,
    audit_thermal_forced_oscillator_ctp,
    certify_common_phase_vacuum_stress_no_go,
    certify_finite_ctp_diagonal_source_obstruction,
    certify_collective_diagonal_influence,
    influence,
    influence_action,
    joint_environment_influence_gram,
    product_environment_influence_gram,
    reduced_system_state,
    unitary_order_residual,
)


def test_exact_thermal_forced_oscillator_influence_witness() -> None:
    certificate = audit_thermal_forced_oscillator_ctp(
        system_mass=2.0,
        environment_mass=3.0,
        bilinear_coupling=2.0,
        vacuum_energy_density=5.0,
        volume=1.0,
        duration=math.pi / 3.0,
        field_left=1.0,
        field_right=0.0,
        mean_occupation=0.0,
        inverse_temperature=math.inf,
    )

    assert certificate.action_parameter_manifest == (2.0, 3.0, 2.0, 5.0)
    assert certificate.source_amplitudes == pytest.approx(
        (math.sqrt(2.0 / 3.0), 0.0)
    )
    assert certificate.noise_exponent == pytest.approx(4.0 / 27.0)
    assert certificate.influence_phase == pytest.approx(2.0 * math.pi / 27.0)
    assert abs(certificate.influence) == pytest.approx(math.exp(-4.0 / 27.0))
    assert certificate.influence == pytest.approx(
        np.exp(2j * math.pi / 27.0 - 4.0 / 27.0),
        abs=2.0e-12,
    )
    assert certificate.mass_matrix_determinant == pytest.approx(32.0)
    assert certificate.gram_minimum_eigenvalue > 0.13
    assert certificate.gram_diagonal_residual == 0.0
    assert certificate.closed_form_influence_residual < 2.0e-11
    assert certificate.diagonal_influence_residual == 0.0
    assert certificate.thermal_occupation_relative_residual == 0.0
    assert certificate.kms_detailed_balance_residual == 0.0
    assert certificate.retarded_future_support_residual == 0.0
    assert certificate.sampled_noise_kernel_minimum_eigenvalue > -2.0e-11
    assert certificate.mass_matrix_stable
    assert certificate.constant_history_noise_quadratic_form_nonnegative
    assert certificate.sampled_noise_kernel_positive_semidefinite
    assert certificate.analytic_retarded_support_contract
    assert certificate.kms_condition_verified
    assert certificate.gram_schur_channel_cptp
    assert certificate.dimensions_pass
    assert certificate.exact_gaussian_influence_computed
    assert certificate.representation.startswith("INTEGRATED_OUT")
    assert certificate.history_role.startswith("SUPPLIED_CTP")
    assert not certificate.canonical_hilbert_space_finite_dimensional
    assert not certificate.retained_environment_stress_added
    assert not certificate.local_stress_from_gram_derived
    assert not certificate.markovian_bath_derived


def test_gaussian_ctp_diagonal_and_vacuum_normalization_boundaries() -> None:
    common = dict(
        system_mass=2.0,
        environment_mass=3.0,
        bilinear_coupling=2.0,
        volume=1.0,
        duration=0.7,
        field_left=0.4,
        field_right=0.4,
        mean_occupation=1.5,
        inverse_temperature=math.log(5.0 / 3.0) / 3.0,
    )
    first = audit_thermal_forced_oscillator_ctp(
        **common,
        vacuum_energy_density=1.0,
    )
    second = audit_thermal_forced_oscillator_ctp(
        **common,
        vacuum_energy_density=9.0,
    )

    assert first.influence == pytest.approx(1.0 + 0.0j)
    assert second.influence == pytest.approx(first.influence)
    assert first.action_parameter_manifest[-1] == 1.0
    assert second.action_parameter_manifest[-1] == 9.0
    assert first.kms_boltzmann_factor == pytest.approx(0.6)
    assert first.thermal_occupation_relative_residual < 2.0e-11
    assert first.kms_detailed_balance_residual < 2.0e-11
    assert not first.local_stress_from_gram_derived


def test_gaussian_ctp_rejects_an_unstable_bilinear_action() -> None:
    with pytest.raises(ValueError, match="mass matrix"):
        audit_thermal_forced_oscillator_ctp(
            system_mass=1.0,
            environment_mass=1.0,
            bilinear_coupling=1.0,
            vacuum_energy_density=0.0,
            volume=1.0,
            duration=1.0,
            field_left=1.0,
            field_right=0.0,
            mean_occupation=0.0,
            inverse_temperature=math.inf,
        )

    with pytest.raises(ValueError, match="thermal occupation|KMS"):
        audit_thermal_forced_oscillator_ctp(
            system_mass=2.0,
            environment_mass=3.0,
            bilinear_coupling=2.0,
            vacuum_energy_density=0.0,
            volume=1.0,
            duration=1.0,
            field_left=1.0,
            field_right=0.0,
            mean_occupation=1.0,
            inverse_temperature=math.inf,
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


def test_unobserved_coherence_changes_exact_product_environment_influence() -> None:
    """Equal Z populations need not give equal influence under an X coupling."""

    theta = 0.07
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    conditional = np.stack(
        (identity, math.cos(theta) * identity - 1j * math.sin(theta) * sigma_x)
    )
    plus_x = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    coherent = np.outer(plus_x, plus_x.conj())
    z_dephased = 0.5 * identity

    coherent_gram = product_environment_influence_gram(
        (conditional,) * 100,
        (coherent,) * 100,
    )
    mixed_gram = product_environment_influence_gram(
        (conditional,) * 100,
        (z_dephased,) * 100,
    )

    assert coherent_gram[1, 0] == pytest.approx(np.exp(-1j * 100.0 * theta))
    assert abs(coherent_gram[1, 0]) == pytest.approx(1.0)
    assert mixed_gram[1, 0] == pytest.approx(math.cos(theta) ** 100)
    assert abs(mixed_gram[1, 0]) < 1.0


def test_entangled_environment_correlations_are_not_fixed_by_local_marginals() -> None:
    """Bell correlations and I/4 have equal one-qubit marginals but different G."""

    identity = np.eye(4, dtype=complex)
    sigma_z = np.diag((1.0, -1.0)).astype(complex)
    parity = np.kron(sigma_z, sigma_z)
    conditional = np.stack((identity, parity))
    bell_vector = np.array((1.0, 0.0, 0.0, 1.0), dtype=complex) / math.sqrt(2.0)
    entangled = np.outer(bell_vector, bell_vector.conj())
    product_of_local_marginals = 0.25 * identity

    entangled_gram = joint_environment_influence_gram(conditional, entangled)
    product_gram = joint_environment_influence_gram(
        conditional,
        product_of_local_marginals,
    )

    assert entangled_gram[1, 0] == pytest.approx(1.0)
    assert product_gram[1, 0] == pytest.approx(0.0)
    assert float(np.min(np.linalg.eigvalsh(entangled_gram))) >= -2.0e-11
    assert float(np.min(np.linalg.eigvalsh(product_gram))) >= -2.0e-11
    assert np.diag(entangled_gram) == pytest.approx(np.ones(2))
    assert np.diag(product_gram) == pytest.approx(np.ones(2))


def test_ten_observed_by_one_hundred_hidden_diagonal_influence_scales_by_correlations() -> None:
    observed_count = 10
    environment_count = 100
    probability = 0.3
    coupling = 0.002
    tau = 2.0
    histories = np.vstack(
        (np.zeros(observed_count), np.ones(observed_count))
    )
    couplings = np.full((observed_count, environment_count), coupling)
    probabilities = np.full(environment_count, probability)

    certificate = certify_collective_diagonal_influence(
        histories,
        couplings,
        probabilities,
        tau=tau,
    )
    gram = np.asarray(certificate.influence_gram)
    phase_per_environment = tau * observed_count * coupling
    expected_influence = (
        (1.0 - probability) + probability * np.exp(-1j * phase_per_environment)
    ) ** environment_count
    expected_mean = environment_count * coupling * probability
    expected_covariance = (
        environment_count * coupling**2 * probability * (1.0 - probability)
    )

    assert certificate.observed_count == observed_count
    assert certificate.environment_count == environment_count
    assert gram[1, 0] == pytest.approx(expected_influence)
    assert certificate.mean_angular_frequency_shift == pytest.approx(
        (expected_mean,) * observed_count
    )
    assert np.asarray(certificate.angular_frequency_covariance) == pytest.approx(
        np.full((observed_count, observed_count), expected_covariance)
    )
    assert certificate.history_mean_angular_frequency_shift == pytest.approx(
        (0.0, observed_count * expected_mean)
    )
    assert np.asarray(certificate.history_angular_frequency_covariance) == pytest.approx(
        np.array(
            [
                [0.0, 0.0],
                [0.0, observed_count**2 * expected_covariance],
            ]
        )
    )
    assert certificate.gram_minimum_eigenvalue >= -2.0e-11
    assert certificate.gram_hermiticity_residual < 2.0e-11
    assert certificate.gram_diagonal_residual < 2.0e-11
    assert certificate.schur_channel_completely_positive
    assert certificate.schur_channel_trace_preserving
    assert certificate.dimensionless_phase_input_contract_declared
    assert not certificate.environment_coherence_phases_resolved
    assert not certificate.physical_clarus_source_derived
    assert not certificate.retained_environment_stress_added

    plus_history_state = np.full((2, 2), 0.5, dtype=complex)
    output = apply_influence_gram(plus_history_state, gram)
    assert np.trace(output) == pytest.approx(1.0)
    assert float(np.min(np.linalg.eigvalsh(output))) >= -2.0e-11
    assert abs(output[1, 0]) < abs(plus_history_state[1, 0])


def test_zero_hidden_environment_is_the_identity_influence() -> None:
    histories = np.array([[0.0], [1.0]])
    certificate = certify_collective_diagonal_influence(
        histories,
        np.empty((1, 0)),
        np.empty(0),
        tau=1.0,
    )

    assert certificate.environment_count == 0
    assert certificate.mean_angular_frequency_shift == (0.0,)
    assert certificate.angular_frequency_covariance == ((0.0,),)
    assert certificate.history_mean_angular_frequency_shift == (0.0, 0.0)
    assert certificate.history_angular_frequency_covariance == (
        (0.0, 0.0),
        (0.0, 0.0),
    )
    assert np.asarray(certificate.influence_gram) == pytest.approx(
        np.ones((2, 2), dtype=complex)
    )


def test_same_environment_must_not_be_reset_at_each_observer_slice() -> None:
    phases = np.zeros((2, 2, 1))
    phases[:, 1, 0] = 0.4
    audit = audit_observer_slice_memory(phases, np.array([0.5]))
    same = np.asarray(audit.same_environment_gram)
    fresh = np.asarray(audit.fresh_environment_composed_gram)

    assert same[1, 0] == pytest.approx(0.5 + 0.5 * np.exp(-0.8j))
    assert fresh[1, 0] == pytest.approx((0.5 + 0.5 * np.exp(-0.4j)) ** 2)
    assert audit.naive_reduced_composition_residual > 1.0e-2
    assert audit.memory_aware_description_required
    assert audit.same_environment_retained
    assert audit.fresh_environment_reset_is_extra_assumption
    assert audit.commuting_diagonal_no_history_transition_assumption

    pure_audit = audit_observer_slice_memory(phases, np.array([1.0]))
    assert pure_audit.naive_reduced_composition_residual < 2.0e-11
    assert not pure_audit.memory_aware_description_required


def test_joint_environment_noncommuting_slices_are_ordered_before_trace() -> None:
    theta = math.pi / 3.0
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array(((0.0, 1.0), (1.0, 0.0)), dtype=complex)
    sigma_y = np.array(((0.0, -1j), (1j, 0.0)), dtype=complex)
    sigma_z = np.diag((1.0, -1.0)).astype(complex)
    rotation_x = math.cos(theta) * identity - 1j * math.sin(theta) * sigma_x
    rotation_z = math.cos(theta) * identity - 1j * math.sin(theta) * sigma_z
    plus_y = 0.5 * (identity + sigma_y)
    slices = np.stack(
        (
            np.stack((identity, rotation_x)),
            np.stack((identity, rotation_z)),
        )
    )

    audit = audit_joint_environment_slice_memory(slices, plus_y)
    same = np.asarray(audit.same_environment_gram)
    fresh = np.asarray(audit.fresh_environment_composed_gram)
    expected_same = math.cos(theta) ** 2 - 1j * math.sin(theta) ** 2

    assert same[1, 0] == pytest.approx(expected_same)
    assert fresh[1, 0] == pytest.approx(math.cos(theta) ** 2)
    assert audit.naive_reduced_composition_residual > 0.7
    assert audit.same_environment_gram_minimum_eigenvalue >= -2.0e-11
    assert audit.same_environment_gram_diagonal_residual < 2.0e-11
    assert audit.memory_aware_description_required
    assert audit.joint_environment_correlations_allowed
    assert audit.controlled_history_initial_product_assumption
    assert audit.conditional_slice_unitaries_may_be_noncommuting
    assert not audit.general_process_tensor_implemented


def test_initial_correlations_require_the_full_joint_preparation() -> None:
    """One reduced input state has two correlated preparations and two outputs."""

    identity = np.eye(4, dtype=complex)
    bell_vector = np.array((1.0, 0.0, 0.0, 1.0), dtype=complex) / math.sqrt(2.0)
    bell = np.outer(bell_vector, bell_vector.conj())
    classically_correlated = np.diag((0.5, 0.0, 0.0, 0.5)).astype(complex)
    controlled_not = np.array(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        dtype=complex,
    )

    bell_input = reduced_system_state(
        bell,
        identity,
        system_dimension=2,
        environment_dimension=2,
    )
    classical_input = reduced_system_state(
        classically_correlated,
        identity,
        system_dimension=2,
        environment_dimension=2,
    )
    bell_output = reduced_system_state(
        bell,
        controlled_not,
        system_dimension=2,
        environment_dimension=2,
    )
    classical_output = reduced_system_state(
        classically_correlated,
        controlled_not,
        system_dimension=2,
        environment_dimension=2,
    )

    assert bell_input == pytest.approx(0.5 * np.eye(2))
    assert classical_input == pytest.approx(bell_input)
    assert bell_output == pytest.approx(np.full((2, 2), 0.5))
    assert classical_output == pytest.approx(0.5 * np.eye(2))
    assert np.linalg.norm(bell_output - classical_output, ord=2) == pytest.approx(0.5)


def test_exact_system_kick_is_balanced_by_environment_recoil_operator() -> None:
    identity = np.eye(2, dtype=complex)
    occupation = np.diag((0.0, 1.0)).astype(complex)
    system_momentum = np.kron(occupation, identity)
    environment_momentum = np.kron(identity, occupation)
    sectors = np.stack((system_momentum, environment_momentum))[:, np.newaxis]
    swap = np.array(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=complex,
    )
    initial_vector = np.array((0.0, 0.0, 1.0, 0.0), dtype=complex)
    initial = np.outer(initial_vector, initial_vector.conj())

    audit = audit_quantum_kick_conservation(
        swap,
        initial,
        sectors,
        all_receivers_included=True,
    )

    assert np.asarray(audit.mean_kicks)[:, 0] == pytest.approx((-1.0, 1.0))
    assert audit.total_mean_kicks == pytest.approx((0.0,))
    assert max(audit.total_kick_operator_residuals) < 2.0e-11
    assert max(audit.total_momentum_commutator_residuals) < 2.0e-11
    assert audit.operator_conservation_certified
    assert audit.unitary_dimensionless
    assert audit.momentum_dimension_input_contract_declared
    assert not audit.force_time_window_derived
    assert not audit.four_vector_covariance_derived
    assert not audit.physical_clarus_source_derived
    assert not audit.stress_tensor_derived


def test_common_hamiltonian_phase_does_not_identify_absolute_vacuum_stress() -> None:
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array(((0.0, 1.0), (1.0, 0.0)), dtype=complex)
    unitaries = np.stack((identity, sigma_x))
    environment = np.array(((0.7, 0.1), (0.1, 0.3)), dtype=complex)
    minkowski = np.diag((-1.0, 1.0, 1.0, 1.0))

    certificate = certify_common_phase_vacuum_stress_no_go(
        unitaries,
        environment,
        hamiltonian_shift=2.5,
        duration=0.4,
        hbar=1.0,
        metric_covariant=minkowski,
        vacuum_energy_density_shift=0.7,
        reference_mass_scale=1.0,
    )

    assert certificate.common_phase_angle == pytest.approx(1.0)
    assert abs(certificate.common_phase_factor) == pytest.approx(1.0)
    assert certificate.maximum_gram_residual < 2.0e-11
    assert np.asarray(certificate.shifted_influence_gram) == pytest.approx(
        np.asarray(certificate.original_influence_gram)
    )
    assert np.asarray(certificate.vacuum_stress_shift_covariant) == pytest.approx(
        np.diag((0.7, -0.7, -0.7, -0.7))
    )
    assert certificate.dimensionless_vacuum_stress_difference == pytest.approx(0.7)
    assert certificate.hamiltonian_mass_dimension == 1
    assert certificate.duration_mass_dimension == -1
    assert certificate.phase_mass_dimension == 0
    assert certificate.vacuum_density_mass_dimension == 4
    assert certificate.stress_mass_dimension == 4
    assert certificate.dimensions_pass
    assert certificate.common_phase_has_unit_modulus
    assert certificate.influence_gram_invariant
    assert certificate.vacuum_stress_distinct
    assert certificate.absolute_vacuum_density_nonidentifiability_certified
    assert certificate.vacuum_action_supplied
    assert not certificate.quantum_identity_shift_to_vacuum_density_mapping_derived
    assert not certificate.absolute_vacuum_density_from_influence_gram_derived
    assert not certificate.physical_dark_energy_density_derived


def test_environment_only_observable_drops_history_interference_but_total_can_keep_it() -> None:
    identity = np.eye(2, dtype=complex)
    unitaries = np.stack((identity, identity))
    plus = 0.5 * np.ones((2, 2), dtype=complex)
    environment = np.diag((1.0, 0.0)).astype(complex)
    sigma_z = np.diag((1.0, -1.0)).astype(complex)

    environment_only_blocks = np.zeros((2, 2, 2, 2), dtype=complex)
    environment_only_blocks[0, 0] = sigma_z
    environment_only_blocks[1, 1] = sigma_z
    environment_only = audit_controlled_history_observable_expectation(
        plus,
        environment,
        unitaries,
        environment_only_blocks,
    )

    assert environment_only.full_expectation == pytest.approx(1.0)
    assert environment_only.diagonal_history_expectation == pytest.approx(1.0)
    assert environment_only.off_diagonal_history_expectation == pytest.approx(0.0)
    assert environment_only.environment_only_block_structure_detected
    assert environment_only.system_history_coherence_present
    assert not environment_only.off_diagonal_interference_present
    assert environment_only.environment_only_history_interference_absent

    total_blocks = np.zeros((2, 2, 2, 2), dtype=complex)
    total_blocks[0, 1] = identity
    total_blocks[1, 0] = identity
    total = audit_controlled_history_observable_expectation(
        plus,
        environment,
        unitaries,
        total_blocks,
    )

    assert total.full_expectation == pytest.approx(1.0)
    assert total.diagonal_history_expectation == pytest.approx(0.0)
    assert total.off_diagonal_history_expectation == pytest.approx(1.0)
    assert not total.environment_only_block_structure_detected
    assert total.system_history_coherence_present
    assert total.off_diagonal_interference_present
    assert total.exact_block_expectation_computed
    assert not total.observable_expectation_from_influence_gram_alone_derived
    assert not total.supplied_observable_is_physical_stress_derived
    assert not total.metric_variation_of_observable_derived
    assert not total.semiclassical_gravity_source_derived


def test_trivial_gram_does_not_identify_an_environment_observable_expectation() -> None:
    identity = np.eye(2, dtype=complex)
    unitaries = np.stack((identity, identity))
    system = np.diag((0.5, 0.5)).astype(complex)
    state_zero = np.diag((1.0, 0.0)).astype(complex)
    state_one = np.diag((0.0, 1.0)).astype(complex)
    sigma_z = np.diag((1.0, -1.0)).astype(complex)
    blocks = np.zeros((2, 2, 2, 2), dtype=complex)
    blocks[0, 0] = sigma_z
    blocks[1, 1] = sigma_z

    zero = audit_controlled_history_observable_expectation(
        system,
        state_zero,
        unitaries,
        blocks,
    )
    one = audit_controlled_history_observable_expectation(
        system,
        state_one,
        unitaries,
        blocks,
    )

    assert np.asarray(zero.influence_gram) == pytest.approx(np.ones((2, 2)))
    assert np.asarray(one.influence_gram) == pytest.approx(
        np.asarray(zero.influence_gram)
    )
    assert zero.full_expectation == pytest.approx(1.0)
    assert one.full_expectation == pytest.approx(-1.0)
    assert zero.full_expectation != one.full_expectation


def test_zero_mean_total_kick_in_one_state_does_not_prove_conservation() -> None:
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array(((0.0, 1.0), (1.0, 0.0)), dtype=complex)
    occupation = np.diag((0.0, 1.0)).astype(complex)
    system_momentum = np.kron(occupation, identity)
    environment_momentum = np.kron(identity, occupation)
    sectors = np.stack((system_momentum, environment_momentum))[:, np.newaxis]
    driven_unitary = np.kron(sigma_x, identity)

    audit = audit_quantum_kick_conservation(
        driven_unitary,
        0.25 * np.eye(4),
        sectors,
        all_receivers_included=True,
    )

    assert audit.total_mean_kicks == pytest.approx((0.0,))
    assert max(audit.total_kick_operator_residuals) > 0.9
    assert max(audit.total_momentum_commutator_residuals) > 0.9
    assert not audit.operator_conservation_certified


def test_disjoint_finite_gates_are_order_independent_but_overlap_need_not_be() -> None:
    theta = 0.37
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array(((0.0, 1.0), (1.0, 0.0)), dtype=complex)
    sigma_z = np.diag((1.0, -1.0)).astype(complex)
    rotation_x = math.cos(theta) * identity - 1j * math.sin(theta) * sigma_x
    rotation_z = math.cos(theta) * identity - 1j * math.sin(theta) * sigma_z

    assert unitary_order_residual(
        np.kron(rotation_x, identity),
        np.kron(identity, rotation_z),
    ) < 2.0e-11
    assert unitary_order_residual(
        np.kron(rotation_x, identity),
        np.kron(rotation_z, identity),
    ) > 0.2


def test_collective_influence_inputs_fail_closed() -> None:
    histories = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="environment_probabilities"):
        certify_collective_diagonal_influence(
            histories,
            np.ones((1, 2)),
            np.array([0.5]),
            tau=1.0,
        )
    with pytest.raises(ValueError, match="unitary"):
        product_environment_influence_gram(
            (np.ones((2, 2, 2), dtype=complex),),
            (0.5 * np.eye(2),),
        )
