import numpy as np
import pytest

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
    state_from_conserved_charges,
)
from examples.physics.qft_reference_flrw_principal_stability import spatial_tt_basis
from examples.physics.qft_reference_flrw_tensor_stability import (
    audit_flrw_tensor_sector,
    audit_supplied_tensor_cutoff,
    finite_difference_tensor_mass_squared_bar,
    rod_lagrangian_density_bar,
    symmetric_matrix_exponential,
    symmetric_exponential_trace_increment,
    tensor_mass_squared_bar,
    validate_tt_generator,
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


def test_exponential_tt_parameterization_is_exactly_unimodular() -> None:
    generator = spatial_tt_basis(np.array([1.0, 2.0, 3.0]))[0]
    epsilon = 2.0e-3
    exponential = symmetric_matrix_exponential(epsilon * generator)

    assert abs(np.trace(generator)) < 1.0e-14
    assert np.isclose(np.linalg.det(exponential), 1.0, atol=1.0e-13)


def test_exact_rod_trace_has_the_half_trace_gamma_squared_coefficient() -> None:
    generator = spatial_tt_basis(np.array([0.0, 0.0, 1.0]))[1]
    epsilon = 1.0e-4
    norm_squared = np.einsum('ij,ij->', generator, generator)
    trace_increment = symmetric_exponential_trace_increment(-epsilon * generator)
    coefficient = trace_increment / (epsilon**2 * norm_squared)

    assert np.isclose(norm_squared, 1.0)
    assert np.isclose(coefficient, 0.5, rtol=2.0e-8, atol=2.0e-8)


def test_tensor_mass_matches_the_exact_rod_hessian() -> None:
    state, parameters = reference_state()
    generator = spatial_tt_basis(np.array([1.0, 0.0, 0.0]))[0]
    analytic = tensor_mass_squared_bar(state, parameters)
    numerical = finite_difference_tensor_mass_squared_bar(
        state,
        parameters,
        generator,
        epsilon=1.0e-4,
    )

    assert np.isclose(analytic, 2.0 * state.b**2 / parameters.m_planck_over_mu_x**2)
    assert np.isclose(numerical, analytic, atol=2.0e-8)


@pytest.mark.parametrize('polarization_index', [0, 1])
def test_both_tensor_polarizations_pass_the_finite_k_gate(
    polarization_index: int,
) -> None:
    state, parameters = reference_state()
    audit = audit_flrw_tensor_sector(
        state,
        parameters,
        spatial_wavevector_bar=np.array([1.0, 2.0, 3.0]),
        polarization_index=polarization_index,
    )

    assert np.isclose(audit.kinetic_coefficient_bar, 25.0)
    assert np.isclose(audit.gradient_coefficient_bar, 25.0)
    assert np.isclose(audit.expected_eh_coefficient_bar, 25.0)
    assert audit.kinetic_coefficient_residual < 1.0e-12
    assert audit.gradient_coefficient_residual < 1.0e-10
    assert audit.gradient_epsilon_grid_wavenumber_spread < 1.0e-10
    assert np.isclose(audit.tensor_speed_squared, 1.0)
    assert np.isclose(audit.tensor_mass_squared_bar, 8.0e-4)
    assert audit.mass_relative_residual < 2.0e-8
    assert audit.exponential_determinant_residual < 1.0e-12
    assert audit.kinetic_positive
    assert audit.gradient_positive
    assert audit.mass_nonnegative
    assert audit.finite_k_tensor_gate_passed
    assert not audit.vector_sector_computed
    assert not audit.scalar_sector_computed
    assert not audit.strong_coupling_scale_derived
    assert not audit.one_loop_st_identity_computed
    assert not audit.brst_physical_inner_product_constructed
    assert not audit.nonperturbative_m2_passed


def test_comoving_tensor_mass_is_constant_along_a_charge_trajectory() -> None:
    state, parameters = reference_state()
    clock_charge = state.u
    rod_charge = state.b
    audits = [
        audit_flrw_tensor_sector(
            state_from_conserved_charges(
                n=n,
                clock=0.0,
                clock_charge=clock_charge,
                rod_charge=rod_charge,
                parameters=parameters,
            ),
            parameters,
        )
        for n in (0.0, 0.3, 0.7)
    ]

    invariants = [audit.comoving_mass_invariant_bar for audit in audits]
    masses = [audit.tensor_mass_squared_bar for audit in audits]
    assert np.allclose(invariants, invariants[0])
    assert masses[0] > masses[1] > masses[2]


def test_beta_zero_recovers_the_massless_tensor_formula_but_not_a_reference_patch() -> None:
    state, parameters = reference_state()
    no_rods = ReferenceFlrwState(
        n=state.n,
        h=expanding_h_from_constraint(u=state.u, b=0.0, parameters=parameters),
        clock=state.clock,
        u=state.u,
        b=0.0,
    )

    assert tensor_mass_squared_bar(no_rods, parameters) == 0.0
    with pytest.raises(ValueError, match='nondegenerate'):
        audit_flrw_tensor_sector(no_rods, parameters)


def test_wrong_rod_sign_is_a_long_wavelength_tensor_tachyon() -> None:
    state, parameters = reference_state()
    audit = audit_flrw_tensor_sector(
        state,
        parameters,
        rod_kinetic_sign=-1.0,
    )

    assert audit.kinetic_positive
    assert audit.gradient_positive
    assert audit.tensor_mass_squared_bar < 0.0
    assert not audit.mass_nonnegative
    assert not audit.finite_k_tensor_gate_passed


def test_traceful_or_nontransverse_generators_fail_closed() -> None:
    with pytest.raises(ValueError, match='traceless'):
        validate_tt_generator(np.eye(3), np.array([0.0, 0.0, 1.0]))
    nontransverse = np.diag([1.0, -2.0, 1.0])
    with pytest.raises(ValueError, match='transverse'):
        validate_tt_generator(nontransverse, np.array([0.0, 0.0, 1.0]))


def test_rod_lagrangian_hessian_has_the_stable_negative_potential_sign() -> None:
    state, _ = reference_state()
    generator = spatial_tt_basis(np.array([0.0, 1.0, 0.0]))[0]
    epsilon = 1.0e-4
    l_plus = rod_lagrangian_density_bar(epsilon, generator, state)
    l_zero = rod_lagrangian_density_bar(0.0, generator, state)
    l_minus = rod_lagrangian_density_bar(-epsilon, generator, state)

    assert l_plus - 2.0 * l_zero + l_minus < 0.0


def test_supplied_cutoff_is_checked_but_never_claimed_as_derived() -> None:
    state, parameters = reference_state()
    high = audit_supplied_tensor_cutoff(
        state,
        parameters,
        cutoff_over_mu_x=10.0,
    )
    low = audit_supplied_tensor_cutoff(
        state,
        parameters,
        cutoff_over_mu_x=0.1,
    )

    assert high.below_supplied_cutoff
    assert not high.cutoff_derived_from_m1
    assert not low.below_supplied_cutoff


def test_tensor_api_rejects_invalid_polarization_and_epsilon() -> None:
    state, parameters = reference_state()
    with pytest.raises(ValueError, match='polarization'):
        audit_flrw_tensor_sector(state, parameters, polarization_index=2)
    generator = spatial_tt_basis(np.array([0.0, 0.0, 1.0]))[0]
    with pytest.raises(ValueError, match='epsilon'):
        finite_difference_tensor_mass_squared_bar(
            state,
            parameters,
            generator,
            epsilon=0.0,
        )
