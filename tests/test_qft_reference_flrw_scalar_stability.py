import numpy as np
import pytest

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_scalar_stability import (
    analytic_scalar_mass_squared_bar,
    audit_flrw_scalar_sector,
    expected_unreduced_scalar_hessian,
    finite_difference_adm_scalar_hessian,
    reduced_scalar_matrices,
    scalar_constraint_blocks,
    wrong_sign_reference_principal_kinetic,
)


def reference_state(
    *,
    m: float = 10.0,
    lam: float = 0.01,
    u: float = 0.3,
    b: float = 0.2,
) -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=m,
        lambda_over_mu_x_squared=lam,
    )
    h = expanding_h_from_constraint(u=u, b=b, parameters=parameters)
    return ReferenceFlrwState(n=0.0, h=h, clock=0.0, u=u, b=b), parameters


def test_scalar_constraint_blocks_obey_both_background_identities() -> None:
    state, parameters = reference_state()
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=0.7,
    )

    assert blocks.friedmann_identity_residual < 1.0e-14
    assert np.isclose(
        blocks.positive_q_polynomial_bar,
        -state.b**2 * blocks.constraint_determinant_bar,
    )
    assert not blocks.lapse_has_time_derivative
    assert not blocks.longitudinal_shift_has_time_derivative


def test_exact_adm_scalar_hessian_reproduces_the_unreduced_six_by_six_block() -> None:
    state, parameters = reference_state()
    wavevector = np.array([0.4, -0.3, 0.8])
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=np.linalg.norm(wavevector),
    )
    expected = expected_unreduced_scalar_hessian(blocks)
    numerical = finite_difference_adm_scalar_hessian(
        state,
        parameters,
        spatial_wavevector_bar=wavevector,
    )

    assert np.allclose(numerical, expected, rtol=2.0e-6, atol=2.0e-6)


def test_reduced_matrices_are_the_exact_constraint_schur_complement() -> None:
    state, parameters = reference_state()
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=0.8,
    )
    kinetic, gyroscopic, potential = reduced_scalar_matrices(blocks)
    full = expected_unreduced_scalar_hessian(blocks)
    dynamical_indices = [0, 1, 4, 5]
    constraint_indices = [2, 3]
    dynamical = full[np.ix_(dynamical_indices, dynamical_indices)]
    coupling = full[np.ix_(constraint_indices, dynamical_indices)]
    schur = dynamical - coupling.T @ np.linalg.inv(
        full[np.ix_(constraint_indices, constraint_indices)]
    ) @ coupling

    assert np.allclose(schur[:2, :2], kinetic)
    assert np.allclose(schur[:2, 2:], -gyroscopic)
    assert np.allclose(schur[2:, :2], -gyroscopic.T)
    assert np.allclose(schur[2:, 2:], -potential)


@pytest.mark.parametrize('k_bar', [0.01, 0.1, 0.7, 3.0])
def test_positive_background_scalar_gate_has_exact_luminal_factorization(
    k_bar: float,
) -> None:
    state, parameters = reference_state()
    audit = audit_flrw_scalar_sector(
        state,
        parameters,
        comoving_wavenumber_bar=k_bar,
        chi_mass_over_mu_x=0.4,
    )

    roots = np.array(audit.coupled_frequency_squared_bar).real
    assert np.allclose(roots, audit.analytic_frequency_squared_bar, rtol=1.0e-8)
    assert np.isclose(roots[0], k_bar**2)
    assert np.isclose(roots[1], k_bar**2 + audit.scalar_mass_squared_bar)
    assert np.isclose(audit.spectator_chi_frequency_squared_bar, k_bar**2 + 0.4**2)
    assert audit.scalar_mass_squared_bar > 0.0
    assert audit.constraint_block_regular
    assert audit.reduced_kinetic_positive
    assert audit.frozen_roots_real
    assert audit.frozen_roots_nonnegative
    assert audit.adm_action_extraction_passed
    assert audit.finite_k_scalar_gate_passed
    assert audit.reduced_coupled_scalar_count == 2
    assert audit.spectator_scalar_count == 1
    assert audit.total_scalar_physical_count == 3
    assert not audit.homogeneous_k_zero_sector_resolved
    assert not audit.beta_zero_branch_resolved
    assert not audit.time_dependent_mode_equations_solved
    assert not audit.strong_coupling_scale_derived
    assert not audit.one_loop_st_identity_computed
    assert not audit.brst_physical_inner_product_constructed
    assert not audit.nonperturbative_m2_passed


def test_closed_scalar_mass_formula_matches_both_friedmann_forms() -> None:
    state, parameters = reference_state()
    direct = analytic_scalar_mass_squared_bar(state, parameters)
    m = parameters.m_planck_over_mu_x
    lam = parameters.lambda_over_mu_x_squared
    alternate = (
        3.0
        * (state.b**2 + m**2 * lam)
        * (state.b**2 + state.u**2)
        / (m**2 * (state.u**2 + 3.0 * state.b**2 + 2.0 * m**2 * lam))
    )

    assert np.isclose(direct, alternate)
    assert np.isclose(direct, 0.001835294117647059)


def test_high_k_kinetic_matrix_tends_to_the_canonical_identity() -> None:
    state, parameters = reference_state()
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=1.0e4,
    )
    kinetic, _, _ = reduced_scalar_matrices(blocks)

    assert np.allclose(kinetic, np.eye(2), rtol=1.0e-4, atol=1.0e-4)


def test_low_k_kinetic_determinant_vanishes_quadratically_but_stays_positive() -> None:
    state, parameters = reference_state()
    determinants = []
    for k_bar in (1.0e-4, 1.0e-3):
        blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=k_bar,
        )
        kinetic, _, _ = reduced_scalar_matrices(blocks)
        determinants.append(np.linalg.det(kinetic))

    assert determinants[0] > 0.0
    assert determinants[1] > determinants[0]
    assert np.isclose(determinants[1] / determinants[0], 100.0, rtol=2.0e-3)


def test_negative_lambda_can_make_a_constraint_pole() -> None:
    state, parameters = reference_state(m=1.0, lam=-1.7, u=2.0, b=1.0)
    kappa_pole = np.sqrt(
        -state.b**2 * (3.0 * state.b**2 + 2.0 * parameters.lambda_over_mu_x_squared)
        / (4.0 * state.h**2)
    )
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=kappa_pole,
    )

    assert abs(blocks.constraint_determinant_bar) < 1.0e-10
    with pytest.raises(ValueError, match='singular'):
        reduced_scalar_matrices(blocks)


def test_kinetic_positivity_alone_does_not_exclude_a_scalar_tachyon() -> None:
    state, parameters = reference_state(m=1.0, lam=-1.2, u=2.0, b=1.0)
    audit = audit_flrw_scalar_sector(
        state,
        parameters,
        comoving_wavenumber_bar=0.1,
    )

    assert audit.constraint_block_regular
    assert audit.reduced_kinetic_positive
    assert audit.scalar_mass_squared_bar < 0.0
    assert audit.frozen_roots_real
    assert not audit.frozen_roots_nonnegative
    assert not audit.finite_k_scalar_gate_passed


def test_wrong_reference_sign_is_an_immediate_scalar_principal_ghost() -> None:
    eigenvalues = np.linalg.eigvalsh(wrong_sign_reference_principal_kinetic())

    assert np.all(eigenvalues < 0.0)


def test_scalar_chart_rejects_k_zero_beta_zero_and_clock_degeneracy() -> None:
    state, parameters = reference_state()
    with pytest.raises(ValueError, match='positive momentum'):
        scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=0.0,
        )
    no_rods, no_rod_parameters = reference_state(b=0.0)
    with pytest.raises(ValueError, match='nonzero beta'):
        scalar_constraint_blocks(
            no_rods,
            no_rod_parameters,
            comoving_wavenumber_bar=1.0,
        )
    no_clock, no_clock_parameters = reference_state(u=0.0)
    with pytest.raises(ValueError, match='nondegenerate'):
        audit_flrw_scalar_sector(
            no_clock,
            no_clock_parameters,
            comoving_wavenumber_bar=1.0,
        )


def test_scalar_audit_rejects_negative_chi_mass_parameter() -> None:
    state, parameters = reference_state()
    with pytest.raises(ValueError, match='nonnegative'):
        audit_flrw_scalar_sector(
            state,
            parameters,
            comoving_wavenumber_bar=1.0,
            chi_mass_over_mu_x=-0.1,
        )
