import numpy as np
import pytest

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_cubic_dynamics import (
    canonical_scalar_phase_space_map,
    direct_scalar_interaction_hamiltonian_tensor_pair,
    dynamic_reduced_scalar_cubic_tensor_pair,
    flat_gauge_dynamic_scalar_lagrangian_bar_per_a3,
    frozen_symplectic_scalar_modes,
    harmonic_scalar_phase_space_map,
    linear_dynamic_constraint_coefficients,
    project_frozen_scalar_hamiltonian_vertices,
    project_frozen_scalar_cubic_vertices,
    projected_dynamic_constraint_residual,
    quadratic_scalar_hamiltonian_cubic_negative_control,
    rod_unitary_dynamic_scalar_lagrangian_bar_per_a3,
    scalar_interaction_hamiltonian_cubic_tensor_pair,
    scalar_mode_symplectic_overlap,
    second_order_dynamic_constraint_tensor,
    solve_direct_scalar_legendre_point,
    solve_projected_dynamic_constraints,
)
from examples.physics.qft_reference_flrw_scalar_stability import (
    exact_adm_scalar_mode_lagrangian_bar_per_a3,
    reduced_scalar_matrices,
    scalar_constraint_blocks,
)


def reference_state() -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=0.01,
    )
    u = 0.3
    b = 0.2
    state = ReferenceFlrwState(
        n=0.0,
        h=expanding_h_from_constraint(u=u, b=b, parameters=parameters),
        clock=0.0,
        u=u,
        b=b,
    )
    return state, parameters


def test_dynamic_one_mode_is_exactly_the_e66_qdot_rdot_q_r_action() -> None:
    state, parameters = reference_state()
    base = 0.2
    first_mode = np.array([0.003, -0.005, 0.004, 0.01])
    physical = np.array([first_mode, np.zeros(4)])
    constraints = linear_dynamic_constraint_coefficients(
        state,
        parameters,
        base_wavenumber_bar=base,
        physical_modes=physical,
    )
    e66 = exact_adm_scalar_mode_lagrangian_bar_per_a3(
        state,
        parameters,
        spatial_wavevector_bar=np.array([base, 0.0, 0.0]),
        amplitudes=np.array(
            [
                first_mode[0],
                first_mode[1],
                constraints[1],
                constraints[5],
                first_mode[2],
                first_mode[3],
            ]
        ),
        phase_points=512,
    )
    dynamic = flat_gauge_dynamic_scalar_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=base,
        physical_modes=physical,
        constraint_coefficients=constraints,
        phase_points=512,
    )

    assert np.isclose(e66, dynamic, rtol=1.0e-11, atol=1.0e-12)


def test_time_dependent_rod_unitary_pullback_matches_and_needs_minus_ydot() -> None:
    state, parameters = reference_state()
    physical = np.array(
        [
            [0.003, -0.005, 0.004, 0.01],
            [-0.002, 0.004, -0.003, -0.006],
        ]
    )
    constraints = linear_dynamic_constraint_coefficients(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
    )
    flat = flat_gauge_dynamic_scalar_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
        constraint_coefficients=constraints,
        phase_points=512,
    )
    unitary = rod_unitary_dynamic_scalar_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
        constraint_coefficients=constraints,
        phase_points=512,
    )
    static_map_negative_control = rod_unitary_dynamic_scalar_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
        constraint_coefficients=constraints,
        phase_points=512,
        include_coordinate_time_shift=False,
    )

    assert np.isclose(flat, unitary, rtol=1.0e-11, atol=1.0e-11)
    assert abs(static_map_negative_control - flat) > 1.0e-6


def test_e66_linear_constraints_cancel_the_full_linear_projected_residual() -> None:
    state, parameters = reference_state()
    direction = np.array(
        [[0.3, -0.5, 0.4, 1.0], [-0.2, 0.4, -0.3, -0.6]]
    )
    epsilon = 1.0e-4
    physical = epsilon * direction
    linear = linear_dynamic_constraint_coefficients(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
    )
    residual = projected_dynamic_constraint_residual(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
        constraint_coefficients=linear,
        phase_points=512,
    )
    omitted = projected_dynamic_constraint_residual(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
        constraint_coefficients=np.zeros(9),
        phase_points=512,
    )

    assert np.max(np.abs(residual)) < 1.0e-7
    assert np.max(np.abs(omitted)) > 100.0 * np.max(np.abs(residual))


def test_exact_projected_constraint_solver_retains_the_quadratic_zero_mode() -> None:
    state, parameters = reference_state()
    physical = 0.01 * np.array(
        [[0.3, -0.5, 0.4, 1.0], [-0.2, 0.4, -0.3, -0.6]]
    )
    linear = linear_dynamic_constraint_coefficients(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
    )
    solved = solve_projected_dynamic_constraints(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        physical_modes=physical,
        phase_points=512,
    )

    assert solved.converged
    assert np.max(np.abs(solved.projected_residual)) < 1.0e-10
    assert abs(solved.coefficients[0]) > 1.0e-8
    assert np.linalg.norm(solved.coefficients - linear) > 1.0e-8


def test_second_order_constraint_tensor_refines_and_contains_zero_mode() -> None:
    state, parameters = reference_state()
    coarse = second_order_dynamic_constraint_tensor(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=2.0e-3,
        phase_points=256,
    )
    fine = second_order_dynamic_constraint_tensor(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=1.0e-3,
        phase_points=256,
    )
    fine_grid = second_order_dynamic_constraint_tensor(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=1.0e-3,
        phase_points=512,
    )
    spread = np.linalg.norm(coarse - fine) / max(1.0, np.linalg.norm(fine))
    grid_spread = np.linalg.norm(fine_grid - fine) / max(
        1.0, np.linalg.norm(fine)
    )

    assert spread < 2.0e-4
    assert grid_spread < 1.0e-8
    assert np.linalg.norm(fine[0]) > 1.0e-6
    assert np.allclose(fine, np.swapaxes(fine, 1, 2), atol=1.0e-10)


def test_full_dynamic_cubic_tensor_refines_and_matches_both_gauges() -> None:
    state, parameters = reference_state()
    coarse_flat, coarse_unitary = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=5.0e-3,
        phase_points=256,
    )
    fine_flat, fine_unitary = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=2.5e-3,
        phase_points=256,
    )
    grid_flat, grid_unitary = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=2.5e-3,
        phase_points=512,
    )
    linear_flat, _ = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=2.5e-3,
        phase_points=256,
        constraint_scheme='linear',
    )
    zero_flat, _ = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=2.5e-3,
        phase_points=256,
        constraint_scheme='zero',
    )
    scale = max(1.0, np.linalg.norm(fine_flat))

    assert np.linalg.norm(coarse_flat - fine_flat) / scale < 2.0e-4
    assert np.linalg.norm(grid_flat - fine_flat) / scale < 1.0e-8
    assert np.linalg.norm(grid_unitary - fine_unitary) / scale < 1.0e-8
    assert np.linalg.norm(fine_flat - fine_unitary) / scale < 1.0e-6
    assert np.linalg.norm(coarse_unitary - fine_unitary) / scale < 2.0e-4
    assert np.linalg.norm(fine_flat - linear_flat) / scale < 2.0e-4
    assert np.linalg.norm(fine_flat - zero_flat) / scale > 1.0e-6
    assert np.allclose(fine_flat, np.swapaxes(fine_flat, 0, 1), atol=1.0e-7)


def test_frozen_modes_have_unit_symplectic_norm_and_are_cross_orthogonal() -> None:
    state, parameters = reference_state()
    modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=0.2,
    )

    assert all(abs(mode.symplectic_norm - 1.0) < 1.0e-10 for mode in modes)
    assert all(mode.pencil_residual < 1.0e-8 for mode in modes)
    assert abs(scalar_mode_symplectic_overlap(modes[0], modes[1])) < 1.0e-8


def test_all_64_frozen_mode_assignments_are_finite_and_conjugate() -> None:
    state, parameters = reference_state()
    flat, unitary = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        epsilon=2.5e-3,
        phase_points=256,
    )
    first_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=0.2,
    )
    second_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=0.4,
    )
    vertices = project_frozen_scalar_cubic_vertices(
        flat,
        first_modes,
        second_modes,
        scale_factor=np.exp(state.n),
    )
    unitary_vertices = project_frozen_scalar_cubic_vertices(
        unitary,
        first_modes,
        second_modes,
        scale_factor=np.exp(state.n),
    )
    hamiltonian_flat, _ = scalar_interaction_hamiltonian_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        flat_lagrangian_tensor=flat,
        unitary_lagrangian_tensor=unitary,
    )
    hamiltonian_vertices = project_frozen_scalar_hamiltonian_vertices(
        hamiltonian_flat,
        first_modes,
        second_modes,
    )
    lookup = {
        (
            item.first_mode,
            item.second_mode,
            item.third_mode,
            item.first_frequency_sign,
            item.second_frequency_sign,
            item.third_frequency_sign,
        ): item.value
        for item in vertices
    }

    assert len(vertices) == 64
    assert all(np.isfinite(item.value) for item in vertices)
    assert max(
        abs(first.value - second.value)
        for first, second in zip(vertices, unitary_vertices, strict=True)
    ) < 1.0e-6
    assert max(
        abs(first.value + second.value)
        for first, second in zip(vertices, hamiltonian_vertices, strict=True)
    ) < 1.0e-8
    for key, value in lookup.items():
        conjugate_key = (*key[:3], -key[3], -key[4], -key[5])
        assert abs(lookup[conjugate_key] - value.conjugate()) < 1.0e-8


def test_quadratic_legendre_map_has_the_e66_sign_and_rejects_bad_kinetic() -> None:
    state, parameters = reference_state()
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=0.2,
    )
    kinetic, gyroscopic, _ = reduced_scalar_matrices(blocks)
    mapping = canonical_scalar_phase_space_map(
        kinetic,
        gyroscopic,
        scale_factor=np.exp(state.n),
    )
    phase_mode = np.array([0.003, -0.002, 0.004, 0.006])
    jet = mapping.matrix @ phase_mode
    recovered_momentum = np.exp(3.0 * state.n) * (
        kinetic @ jet[:2] - gyroscopic @ jet[2:]
    )

    assert np.all(mapping.kinetic_eigenvalues > 0.0)
    assert abs(mapping.determinant) > 1.0e-10
    assert np.allclose(jet[2:], phase_mode[2:], atol=1.0e-14)
    assert np.allclose(recovered_momentum, phase_mode[:2], atol=1.0e-13)
    with pytest.raises(ValueError, match='positive definite'):
        canonical_scalar_phase_space_map(
            np.diag([1.0, -1.0]),
            np.zeros((2, 2)),
            scale_factor=1.0,
        )


def test_direct_legendre_solver_recovers_momentum_without_using_r_map() -> None:
    state, parameters = reference_state()
    canonical = np.array(
        [
            [0.003, -0.002, 0.004, 0.006],
            [-0.001, 0.002, -0.003, 0.005],
        ]
    )
    solved = solve_direct_scalar_legendre_point(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        canonical_modes=canonical,
        phase_points=256,
    )

    assert solved.maximum_momentum_residual < 1.0e-11
    assert solved.maximum_constraint_residual < 1.0e-11
    assert solved.iterations > 0
    assert np.allclose(solved.recovered_momenta, canonical[:, :2], atol=1.0e-11)
    assert np.isfinite(solved.flat_interaction_hamiltonian_bar)
    assert abs(
        solved.flat_interaction_hamiltonian_bar
        - solved.unitary_interaction_hamiltonian_bar
    ) < 1.0e-10


def test_cubic_interaction_hamiltonian_matches_direct_legendre_tensor() -> None:
    state, parameters = reference_state()
    base = 0.2
    lagrangian_flat, lagrangian_unitary = (
        dynamic_reduced_scalar_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base,
            epsilon=2.5e-3,
            phase_points=256,
        )
    )
    analytic_flat, analytic_unitary = (
        scalar_interaction_hamiltonian_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base,
            flat_lagrangian_tensor=lagrangian_flat,
            unitary_lagrangian_tensor=lagrangian_unitary,
        )
    )
    direct = direct_scalar_interaction_hamiltonian_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base,
        epsilon=2.5e-3,
        phase_points=256,
    )
    scale = max(1.0, np.linalg.norm(analytic_flat))
    naive_coordinate_tensor = -np.exp(3.0 * state.n) * lagrangian_flat
    quadratic_control = quadratic_scalar_hamiltonian_cubic_negative_control(
        state,
        parameters,
        base_wavenumber_bar=base,
        epsilon=2.5e-3,
    )

    assert np.linalg.norm(direct.flat_tensor - analytic_flat) / scale < 2.0e-4
    assert np.max(np.abs(direct.flat_tensor - analytic_flat)) < 2.0e-4
    assert np.linalg.norm(direct.flat_tensor - direct.unitary_tensor) / scale < 1.0e-6
    assert np.linalg.norm(analytic_flat - analytic_unitary) / scale < 1.0e-6
    assert direct.maximum_momentum_residual < 1.0e-11
    assert direct.maximum_constraint_residual < 1.0e-11
    assert direct.maximum_iterations > 0
    assert np.linalg.norm(analytic_flat - naive_coordinate_tensor) > 1.0e-6
    assert np.linalg.norm(quadratic_control) < 1.0e-8


def test_phase_space_map_sends_each_kg_mode_to_its_frequency_jet() -> None:
    state, parameters = reference_state()
    for wavenumber in (0.2, 0.4):
        mapping = harmonic_scalar_phase_space_map(
            state,
            parameters,
            comoving_wavenumber_bar=wavenumber,
        )
        modes = frozen_symplectic_scalar_modes(
            state,
            parameters,
            comoving_wavenumber_bar=wavenumber,
        )
        for mode in modes:
            phase_mode = np.concatenate((mode.momentum, mode.configuration))
            jet = mapping.matrix @ phase_mode
            expected_velocity = -1j * mode.frequency_bar * mode.configuration

            assert np.linalg.norm(jet[:2] - expected_velocity) < 1.0e-10
            assert np.linalg.norm(jet[2:] - mode.configuration) < 1.0e-12
