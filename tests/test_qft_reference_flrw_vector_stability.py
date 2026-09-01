import numpy as np
import pytest

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
    state_from_conserved_charges,
)
from examples.physics.qft_reference_flrw_tensor_stability import (
    tensor_mass_squared_bar,
)
from examples.physics.qft_reference_flrw_vector_stability import (
    analytic_reduced_vector_kinetic_bar,
    audit_flrw_vector_sector,
    audit_homogeneous_vector_sector,
    exact_adm_vector_mode_lagrangian_bar,
    finite_difference_adm_vector_block,
    finite_difference_adm_vector_gradient_potential_bar,
    naive_shift_zero_omega_squared_bar,
    reduced_vector_kinetic_bar,
    uneliminated_vector_quadratic_block,
    transverse_vector_basis,
    vector_shift_denominator_bar,
)


def reference_state() -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=0.01,
    )
    u = 0.3
    b = 0.2
    h = expanding_h_from_constraint(u=u, b=b, parameters=parameters)
    return ReferenceFlrwState(n=0.0, h=h, clock=0.0, u=u, b=b), parameters


@pytest.mark.parametrize('k_bar', [0.01, 0.2, 1.0, 10.0])
def test_exact_shift_schur_complement_matches_closed_form(k_bar: float) -> None:
    state, parameters = reference_state()
    block = uneliminated_vector_quadratic_block(
        state,
        parameters,
        comoving_wavenumber_bar=k_bar,
    )
    schur = reduced_vector_kinetic_bar(
        state,
        parameters,
        comoving_wavenumber_bar=k_bar,
    )
    analytic = analytic_reduced_vector_kinetic_bar(
        state,
        parameters,
        comoving_wavenumber_bar=k_bar,
    )

    assert np.allclose(block, block.T)
    assert np.isclose(np.linalg.det(block), block[1, 1] * schur)
    assert np.isclose(schur, analytic, rtol=1.0e-13, atol=1.0e-13)


def test_admitted_vector_sector_has_two_degenerate_luminal_polarizations() -> None:
    state, parameters = reference_state()
    audit = audit_flrw_vector_sector(
        state,
        parameters,
        comoving_wavenumber_bar=1.0,
    )

    assert np.isclose(audit.shift_denominator_bar, 50.04)
    assert np.isclose(audit.schur_kinetic_bar, 50.0 / 50.04)
    assert np.isclose(audit.gradient_potential_coefficient_bar, 1.0)
    assert np.isclose(audit.vector_mass_squared_bar, 8.0e-4)
    assert np.isclose(audit.vector_mass_squared_bar, tensor_mass_squared_bar(state, parameters))
    assert np.isclose(audit.frozen_omega_squared_bar, 1.0008)
    assert np.isclose(audit.vector_speed_squared, 1.0)
    assert audit.adm_block_relative_residual < 1.0e-12
    assert audit.adm_gradient_relative_residual < 1.0e-12
    assert audit.adm_polarization_spread < 1.0e-12
    assert audit.adm_action_extraction_passed
    assert audit.transverse_polarization_count == 2
    assert audit.polarizations_degenerate
    assert not audit.shift_has_time_derivative
    assert audit.shift_constraint_regular
    assert audit.reduced_kinetic_positive
    assert audit.gradient_positive
    assert audit.frozen_pole_nonnegative
    assert audit.finite_k_vector_gate_passed
    assert not audit.homogeneous_k_zero_sector_resolved
    assert not audit.time_dependent_mode_equation_solved
    assert not audit.scalar_sector_computed
    assert not audit.strong_coupling_scale_derived
    assert not audit.one_loop_st_identity_computed
    assert not audit.brst_physical_inner_product_constructed
    assert not audit.nonperturbative_m2_passed


def test_full_adm_action_independently_reproduces_both_vector_blocks() -> None:
    state, parameters = reference_state()
    wavevector = np.array([1.0, 2.0, 3.0])
    polarizations = transverse_vector_basis(wavevector)
    expected = uneliminated_vector_quadratic_block(
        state,
        parameters,
        comoving_wavenumber_bar=np.linalg.norm(wavevector),
    )

    blocks = [
        finite_difference_adm_vector_block(
            state,
            parameters,
            spatial_wavevector_bar=wavevector,
            polarization=polarization,
        )
        for polarization in polarizations
    ]

    assert len(blocks) == 2
    assert np.allclose(blocks[0], expected, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(blocks[1], expected, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(blocks[0], blocks[1], rtol=1.0e-13, atol=1.0e-13)


def test_full_adm_action_independently_reproduces_vector_gradient() -> None:
    state, parameters = reference_state()
    wavevector = np.array([2.0, -1.0, 0.5])
    expected = np.linalg.norm(wavevector) ** 2
    gradients = [
        finite_difference_adm_vector_gradient_potential_bar(
            state,
            parameters,
            spatial_wavevector_bar=wavevector,
            polarization=polarization,
        )
        for polarization in transverse_vector_basis(wavevector)
    ]

    assert np.allclose(gradients, expected, rtol=1.0e-13, atol=1.0e-13)


def test_exact_adm_mode_rejects_nontransverse_or_unnormalized_polarization() -> None:
    state, parameters = reference_state()
    with pytest.raises(ValueError, match='transverse'):
        exact_adm_vector_mode_lagrangian_bar(
            state,
            parameters,
            spatial_wavevector_bar=np.array([0.0, 0.0, 1.0]),
            polarization=np.array([0.0, 0.0, 1.0]),
        )
    with pytest.raises(ValueError, match='unit norm'):
        exact_adm_vector_mode_lagrangian_bar(
            state,
            parameters,
            spatial_wavevector_bar=np.array([0.0, 0.0, 1.0]),
            polarization=np.array([2.0, 0.0, 0.0]),
        )


def test_high_and_low_wavenumber_kinetic_limits_are_explicit() -> None:
    state, parameters = reference_state()
    high = audit_flrw_vector_sector(
        state,
        parameters,
        comoving_wavenumber_bar=1.0e4,
    )
    k_small = 1.0e-4
    low = audit_flrw_vector_sector(
        state,
        parameters,
        comoving_wavenumber_bar=k_small,
    )
    beta_bar = state.b
    low_asymptotic = (
        parameters.m_planck_over_mu_x**2 * k_small**2 / (2.0 * beta_bar**2)
    )

    assert np.isclose(high.schur_kinetic_bar, high.high_k_kinetic_limit_bar, rtol=1.0e-10)
    assert np.isclose(low.schur_kinetic_bar, low_asymptotic, rtol=2.0e-5)
    assert low.schur_kinetic_bar > 0.0


def test_k_zero_is_not_misreported_as_a_ghost_or_propagating_pole() -> None:
    state, parameters = reference_state()
    homogeneous = audit_homogeneous_vector_sector()

    assert not homogeneous.transverse_decomposition_defined
    assert not homogeneous.reduced_formula_applied
    assert homogeneous.propagating_vector_count_claimed == 0
    assert 'SEPARATE_HOMOGENEOUS' in homogeneous.status
    with pytest.raises(ValueError, match='nonzero k'):
        reduced_vector_kinetic_bar(
            state,
            parameters,
            comoving_wavenumber_bar=0.0,
        )
    with pytest.raises(ValueError, match='positive k'):
        audit_flrw_vector_sector(
            state,
            parameters,
            comoving_wavenumber_bar=0.0,
        )


def test_setting_shift_to_zero_before_varying_erases_the_vector_mass() -> None:
    state, parameters = reference_state()
    k_bar = 0.7
    audit = audit_flrw_vector_sector(
        state,
        parameters,
        comoving_wavenumber_bar=k_bar,
    )
    naive = naive_shift_zero_omega_squared_bar(
        state,
        comoving_wavenumber_bar=k_bar,
    )

    assert np.isclose(audit.frozen_omega_squared_bar - naive, audit.vector_mass_squared_bar)
    assert audit.frozen_omega_squared_bar > naive


def test_wrong_rod_sign_exposes_pole_gradient_and_ghost_failures() -> None:
    state, parameters = reference_state()
    below = audit_flrw_vector_sector(
        state,
        parameters,
        comoving_wavenumber_bar=0.01,
        rod_kinetic_sign=-1.0,
    )
    above = audit_flrw_vector_sector(
        state,
        parameters,
        comoving_wavenumber_bar=0.1,
        rod_kinetic_sign=-1.0,
    )
    pole = np.sqrt(2.0) * state.b / parameters.m_planck_over_mu_x

    assert below.reduced_kinetic_positive
    assert not below.gradient_positive
    assert not below.frozen_pole_nonnegative
    assert not below.finite_k_vector_gate_passed
    assert not above.reduced_kinetic_positive
    assert not above.gradient_positive
    assert not above.finite_k_vector_gate_passed
    with pytest.raises(ValueError, match='singular'):
        audit_flrw_vector_sector(
            state,
            parameters,
            comoving_wavenumber_bar=pole,
            rod_kinetic_sign=-1.0,
        )


def test_vector_mass_redshifts_and_comoving_mass_is_conserved() -> None:
    state, parameters = reference_state()
    ns = (0.0, 0.3, 0.7)
    audits = [
        audit_flrw_vector_sector(
            state_from_conserved_charges(
                n=n,
                clock=0.0,
                clock_charge=state.u,
                rod_charge=state.b,
                parameters=parameters,
            ),
            parameters,
            comoving_wavenumber_bar=0.8,
        )
        for n in ns
    ]
    masses = [audit.vector_mass_squared_bar for audit in audits]
    invariants = [
        np.exp(2.0 * n) * audit.vector_mass_squared_bar
        for n, audit in zip(ns, audits, strict=True)
    ]

    assert masses[0] > masses[1] > masses[2]
    assert np.allclose(invariants, invariants[0])


def test_vector_api_rejects_degenerate_background_and_invalid_inputs() -> None:
    state, parameters = reference_state()
    no_rods = ReferenceFlrwState(
        n=state.n,
        h=expanding_h_from_constraint(u=state.u, b=0.0, parameters=parameters),
        clock=state.clock,
        u=state.u,
        b=0.0,
    )

    with pytest.raises(ValueError, match='nondegenerate'):
        audit_flrw_vector_sector(
            no_rods,
            parameters,
            comoving_wavenumber_bar=1.0,
        )
    with pytest.raises(ValueError, match='nonnegative'):
        vector_shift_denominator_bar(
            state,
            parameters,
            comoving_wavenumber_bar=-1.0,
        )
    with pytest.raises(ValueError, match='nonzero'):
        vector_shift_denominator_bar(
            state,
            parameters,
            comoving_wavenumber_bar=1.0,
            rod_kinetic_sign=0.0,
        )
