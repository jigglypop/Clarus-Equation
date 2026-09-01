import numpy as np


def reduced_coefficients(rod_jacobian, clock_gradient, bare_momentum, mu_x, sqrt_q, c_zero):
    inverse_rod = np.linalg.inv(rod_jacobian)
    inverse_rod_metric = inverse_rod.T @ inverse_rod
    a_coef = 1.0 + clock_gradient @ inverse_rod_metric @ clock_gradient
    b_coef = bare_momentum @ inverse_rod_metric @ clock_gradient
    d_coef = (
        bare_momentum @ inverse_rod_metric @ bare_momentum
        + 2.0 * mu_x**2 * sqrt_q * c_zero
    )
    return inverse_rod, a_coef, b_coef, d_coef


def test_e55_reduced_roots_solve_all_four_constraints():
    rng = np.random.default_rng(5501)
    worst_residual = 0.0

    for _ in range(64):
        rod_jacobian = rng.normal(size=(3, 3)) + 2.0 * np.eye(3)
        clock_gradient = rng.normal(size=3)
        bare_momentum = rng.normal(size=3)
        mu_x = 1.3
        sqrt_q = 1.2
        inverse_preview = np.linalg.inv(rod_jacobian)
        inverse_metric_preview = inverse_preview.T @ inverse_preview
        bare_norm = bare_momentum @ inverse_metric_preview @ bare_momentum
        c_zero = -(bare_norm + 1.0) / (2.0 * mu_x**2 * sqrt_q)
        inverse_rod, a_coef, b_coef, d_coef = reduced_coefficients(
            rod_jacobian, clock_gradient, bare_momentum, mu_x, sqrt_q, c_zero
        )
        discriminant = b_coef**2 - a_coef * d_coef
        assert a_coef >= 1.0
        assert discriminant >= 0.0

        for sign in (-1.0, 1.0):
            p_time = (-b_coef + sign * np.sqrt(discriminant)) / a_coef
            p_rods = -inverse_rod @ (bare_momentum + p_time * clock_gradient)
            h_residual = c_zero + (p_time**2 + p_rods @ p_rods) / (
                2.0 * mu_x**2 * sqrt_q
            )
            d_residual = (
                bare_momentum + p_time * clock_gradient + rod_jacobian @ p_rods
            )
            worst_residual = max(
                worst_residual, abs(h_residual), float(np.max(np.abs(d_residual)))
            )

    assert worst_residual < 1.0e-10


def test_e55_clock_slicing_is_the_only_simple_square_root_limit():
    rod_jacobian = np.diag([1.2, 0.8, 1.5])
    bare_momentum = np.array([0.4, -0.7, 0.2])
    mu_x = 1.1
    sqrt_q = 0.9
    c_zero = -3.0
    _, a_zero, b_zero, _ = reduced_coefficients(
        rod_jacobian, np.zeros(3), bare_momentum, mu_x, sqrt_q, c_zero
    )
    _, a_tilted, b_tilted, _ = reduced_coefficients(
        rod_jacobian, np.array([0.3, -0.2, 0.4]), bare_momentum, mu_x, sqrt_q, c_zero
    )

    assert a_zero == 1.0
    assert b_zero == 0.0
    assert a_tilted > 1.0
    assert b_tilted != 0.0


def test_e56_commuting_spectral_sector_is_self_adjoint_and_unitary():
    a_coef = np.array([1.2, 1.7, 2.1])
    b_coef = np.array([2.0, 2.4, 3.1])
    d_coef = np.array([0.7, 1.1, 1.8])
    discriminant = b_coef**2 - a_coef * d_coef
    assert np.all(discriminant > 0.0)

    h_phys = (b_coef - np.sqrt(discriminant)) / a_coef
    quadratic_residual = a_coef * h_phys**2 - 2.0 * b_coef * h_phys + d_coef
    assert np.all(h_phys > 0.0)
    assert np.max(np.abs(quadratic_residual)) < 1.0e-14

    state = np.array([1.0 + 0.5j, -0.3j, 0.7 - 0.2j])
    evolved = np.exp(-1j * 0.37 * h_phys) * state
    assert abs(np.vdot(evolved, evolved) - np.vdot(state, state)) < 1.0e-14


def test_e56_naive_noncommuting_ordering_is_not_self_adjoint():
    a_coef = np.diag([1.0, 2.0])
    b_coef = np.array([[0.0, 1.0], [1.0, 0.0]])
    naive_ordering = np.linalg.inv(a_coef) @ b_coef

    assert np.allclose(a_coef, a_coef.T.conj())
    assert np.allclose(b_coef, b_coef.T.conj())
    assert not np.allclose(a_coef @ b_coef, b_coef @ a_coef)
    assert not np.allclose(naive_ordering, naive_ordering.T.conj())


def test_e57_scalar_cell_regulator_has_nonzero_b_d_commutator():
    levels = 12
    lowering = np.diag(np.sqrt(np.arange(1, levels)), 1)
    raising = lowering.T
    position = (lowering + raising) / np.sqrt(2.0)
    momentum = (lowering - raising) / (1j * np.sqrt(2.0))
    mass_squared = 1.7
    potential = 0.5 * mass_squared * (position @ position)
    commutator = momentum @ potential - potential @ momentum
    expected = -1j * mass_squared * position

    assert np.linalg.norm(commutator) > 1.0
    assert np.linalg.norm((commutator - expected)[:, :-2]) < 1.0e-12


def test_e58_scalar_cell_quadratic_form_is_positive_but_has_no_zero_mode():
    inverse_rod_metric = np.array(
        [[1.4, 0.2, -0.1], [0.2, 1.1, 0.15], [-0.1, 0.15, 0.9]]
    )
    clock_gradient = np.array([0.4, -0.3, 0.2])
    scalar_gradient = np.array([0.5, 0.1, -0.4])
    mu_x = 1.2
    a_coef = 1.0 + clock_gradient @ inverse_rod_metric @ clock_gradient
    beta = scalar_gradient @ inverse_rod_metric @ clock_gradient
    gamma = scalar_gradient @ inverse_rod_metric @ scalar_gradient + mu_x**2
    kinetic_matrix = np.array([[a_coef, beta], [beta, gamma]])
    assert np.min(np.linalg.eigvalsh(kinetic_matrix)) > 0.0
    assert a_coef * gamma - beta**2 >= mu_x**2

    time_momenta = np.diag(np.arange(-2.0, 3.0))
    levels = 12
    lowering = np.diag(np.sqrt(np.arange(1, levels)), 1)
    raising = lowering.T
    position = (lowering + raising) / np.sqrt(2.0)
    momentum = (lowering - raising) / (1j * np.sqrt(2.0))
    identity_t = np.eye(time_momenta.shape[0])
    identity_x = np.eye(levels)
    mass_squared = 1.7
    kappa = 2.0 * mu_x**2
    constraint = (
        a_coef * np.kron(time_momenta @ time_momenta, identity_x)
        + 2.0 * beta * np.kron(time_momenta, momentum)
        + gamma * np.kron(identity_t, momentum @ momentum)
        + 0.5 * kappa * mass_squared * np.kron(identity_t, position @ position)
    )
    eigenvalues = np.linalg.eigvalsh(constraint)
    assert np.allclose(constraint, constraint.T.conj())
    assert np.min(eigenvalues) > 0.0


def test_e59_finite_master_constraint_has_the_same_kernel():
    constraint = np.array(
        [[1.0, 1.0j, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=complex,
    )
    master = constraint.T.conj() @ constraint
    singular_values = np.linalg.svd(constraint, compute_uv=False)
    master_eigenvalues = np.linalg.eigvalsh(master)

    assert np.min(master_eigenvalues) >= -1.0e-14
    assert np.count_nonzero(singular_values < 1.0e-12) == 1
    assert np.count_nonzero(master_eigenvalues < 1.0e-12) == 1
    physical_state = np.array([0.0, 0.0, 1.0])
    assert np.linalg.norm(constraint @ physical_state) == 0.0
    assert np.vdot(physical_state, physical_state) == 1.0


def test_e59_closing_master_gap_does_not_create_a_finite_cutoff_kernel():
    gaps = []
    for cutoff in (4, 8, 16, 32, 64):
        constraint = np.array([[1.0 / cutoff]])
        master = constraint.T @ constraint
        gaps.append(master[0, 0])
        assert np.linalg.matrix_rank(constraint) == 1
        assert np.linalg.matrix_rank(master) == 1

    assert all(left > right for left, right in zip(gaps, gaps[1:]))
    assert gaps[-1] < gaps[0] / 100.0
