import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_cubic_admission import (
    E68_BASE_WAVENUMBERS_BAR,
    audit_static_cubic_triad_precursor,
    flat_gauge_static_triad_lagrangian_bar_per_a3,
    flat_gauge_static_scalar_triad_lagrangian_bar_per_a3,
    normalized_cubic_profile_overlaps,
    rod_unitary_static_triad_lagrangian_bar_per_a3,
    rod_unitary_static_scalar_triad_lagrangian_bar_per_a3,
)
from examples.physics.qft_reference_flrw_scalar_stability import (
    exact_adm_scalar_mode_lagrangian_bar_per_a3,
    scalar_constraint_blocks,
)


def reference_state() -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=0.01,
    )
    u = 0.3
    b = 0.2
    return (
        ReferenceFlrwState(
            n=0.0,
            h=expanding_h_from_constraint(u=u, b=b, parameters=parameters),
            clock=0.0,
            u=u,
            b=b,
        ),
        parameters,
    )


def test_one_mode_is_a_false_negative_but_k_k_minus_2k_has_cubic_support() -> None:
    single, triad = normalized_cubic_profile_overlaps()

    assert abs(single) < 1.0e-14
    assert np.isclose(triad, 1.0 / np.sqrt(2.0))


def test_static_scalar_fields_use_the_e66_r_equals_k_s_coordinate() -> None:
    state, parameters = reference_state()
    base = 0.2
    mode = np.array([0.004, 0.01])
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=base,
    )
    constraint = -np.linalg.solve(
        blocks.constraint_matrix_bar,
        blocks.field_coupling_bar @ mode,
    )
    e66_action = exact_adm_scalar_mode_lagrangian_bar_per_a3(
        state,
        parameters,
        spatial_wavevector_bar=np.array([base, 0.0, 0.0]),
        amplitudes=np.array(
            [0.0, 0.0, constraint[0], constraint[1], mode[0], mode[1]]
        ),
        phase_points=512,
    )
    e68_action = flat_gauge_static_scalar_triad_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=base,
        mode_amplitudes=np.array([mode, [0.0, 0.0]]),
        phase_points=512,
    )

    assert np.isclose(e66_action, e68_action, rtol=1.0e-11, atol=1.0e-12)


def test_exact_static_spatial_diffeomorphism_matches_actions_before_expansion() -> None:
    state, parameters = reference_state()
    amplitudes = np.array([0.01, -0.006])
    flat = flat_gauge_static_triad_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        rod_amplitudes=amplitudes,
    )
    unitary = rod_unitary_static_triad_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        rod_amplitudes=amplitudes,
    )

    assert np.isclose(flat, unitary, rtol=1.0e-11, atol=1.0e-11)
    scalar_amplitudes = np.array([[0.004, 0.01], [-0.003, -0.006]])
    flat_scalar = flat_gauge_static_scalar_triad_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        mode_amplitudes=scalar_amplitudes,
    )
    unitary_scalar = rod_unitary_static_scalar_triad_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        mode_amplitudes=scalar_amplitudes,
    )
    assert np.isclose(flat_scalar, unitary_scalar, rtol=1.0e-11, atol=1.0e-11)


def test_static_cubic_triad_precursor_passes_but_keeps_strong_coupling_open() -> None:
    state, parameters = reference_state()
    audit = audit_static_cubic_triad_precursor(state, parameters)

    assert audit.single_mode_cubic_overlap < 1.0e-14
    assert np.isclose(audit.momentum_conserving_triad_overlap, 1.0 / np.sqrt(2.0))
    assert tuple(
        result.base_wavenumber_bar for result in audit.triad_results
    ) == E68_BASE_WAVENUMBERS_BAR
    assert audit.power_counting.covariant_power_counting_small
    assert not audit.power_counting.reduced_low_k_cutoff_derived
    assert all(result.flat_refinement_spread < 2.0e-4 for result in audit.triad_results)
    assert all(result.gauge_relative_residual < 1.0e-6 for result in audit.triad_results)
    assert all(
        result.coordinate_measure_negative_control_residual > 1.0e-6
        for result in audit.triad_results
    )
    assert all(result.static_two_gauge_gate_passed for result in audit.triad_results)
    assert all(
        result.tensor_refinement_spread < 2.0e-4
        for result in audit.scalar_tensor_results
    )
    assert all(
        result.tensor_gauge_relative_residual < 1.0e-6
        for result in audit.scalar_tensor_results
    )
    assert all(
        result.first_leg_permutation_residual < 1.0e-6
        for result in audit.scalar_tensor_results
    )
    assert all(
        result.static_scalar_tensor_gate_passed
        for result in audit.scalar_tensor_results
    )
    assert audit.static_off_shell_two_gauge_gate_passed
    assert audit.complete_static_qr_triad_tensor_computed
    assert not audit.on_shell_cubic_residue_computed
    assert not audit.all_scalar_vector_tensor_vertices_computed
    assert not audit.second_order_constraint_and_gauge_completion_computed
    assert not audit.physical_strong_coupling_scale_derived
    assert not audit.one_loop_st_identity_computed
    assert not audit.brst_physical_inner_product_constructed
    assert not audit.nonperturbative_m2_passed


def test_q_fixed_coordinate_proxies_are_finite_at_every_declared_nonzero_k() -> None:
    state, parameters = reference_state()
    audit = audit_static_cubic_triad_precursor(state, parameters)
    residues = np.array(
        [
            result.q_fixed_coordinate_direction_proxy
            for result in audit.triad_results
        ]
    )

    assert np.all(np.isfinite(residues))
