import numpy as np
import pytest

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    audit_dimensionless_background_variables,
    audit_reference_flrw_background,
    bare_internal_kinetic_gram,
    clock_charge_bar,
    clock_dominated_past_time_bound,
    curvature_scalar_bar,
    energy_density_bar,
    expanding_h_from_constraint,
    flrw_rhs,
    friedmann_residual,
    integrate_reference_flrw_background,
    orthonormal_stress_bar,
    pressure_bar,
    reference_jacobian_bar,
    rod_charge_bar,
    state_from_conserved_charges,
)


def admitted_state(
    *,
    u: float = 0.3,
    b: float = 0.2,
    lam: float = 0.02,
) -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=lam,
    )
    h = expanding_h_from_constraint(u=u, b=b, parameters=parameters)
    return ReferenceFlrwState(n=0.0, h=h, clock=0.1, u=u, b=b), parameters


def test_all_flrw_core_variables_are_dimensionless() -> None:
    audit = audit_dimensionless_background_variables()

    assert audit.all_core_variables_dimensionless
    assert all(value == 0 for _, value in audit.combination_mass_dimensions)
    assert not audit.physical_correctness_implied


def test_clock_and_three_rods_give_exact_isotropic_stress() -> None:
    state, _ = admitted_state()
    stress = orthonormal_stress_bar(state)

    assert np.allclose(stress[0, 0], energy_density_bar(state))
    assert np.allclose(np.diag(stress)[1:], pressure_bar(state))
    assert np.linalg.norm(stress - np.diag(np.diag(stress))) < 1.0e-14
    assert np.isclose(energy_density_bar(state), 0.5 * state.u**2 + 1.5 * state.b**2)
    assert np.isclose(pressure_bar(state), 0.5 * state.u**2 - 0.5 * state.b**2)


def test_nondegenerate_constraint_state_passes_every_classical_residual() -> None:
    state, parameters = admitted_state()
    audit = audit_reference_flrw_background(state, parameters)

    assert audit.friedmann_relative_residual < 1.0e-14
    assert abs(audit.continuity_residual) < 1.0e-14
    assert abs(audit.constraint_propagation_residual) < 1.0e-14
    assert abs(audit.acceleration_residual) < 1.0e-14
    assert abs(audit.trace_residual) < 1.0e-14
    assert audit.isotropy_residual < 1.0e-14
    assert audit.reference_jacobian_bar != 0.0
    assert audit.clock_gradient_norm_bar < 0.0
    assert audit.local_reference_patch_admitted
    assert not audit.compact_global_real_rod_claim_admitted
    assert np.allclose(audit.bare_internal_kinetic_eigenvalues, np.ones(4))
    assert audit.bare_homogeneous_kinetic_positive
    assert not audit.perturbation_stability_computed
    assert not audit.one_loop_st_identity_computed
    assert not audit.physical_inner_product_constructed
    assert not audit.nonperturbative_m2_passed


@pytest.mark.parametrize(('u', 'b'), [(0.0, 0.2), (0.3, 0.0)])
def test_missing_clock_or_rod_charge_kills_the_relational_patch(u: float, b: float) -> None:
    state, parameters = admitted_state(u=u, b=b)
    audit = audit_reference_flrw_background(state, parameters)

    assert reference_jacobian_bar(state) == 0.0
    assert not audit.local_reference_patch_admitted
    assert audit.status == 'CLASSICAL_BACKGROUND_REJECTED'


def test_inconsistent_friedmann_initial_data_fail_closed() -> None:
    state, parameters = admitted_state()
    inconsistent = ReferenceFlrwState(
        n=state.n,
        h=state.h + 0.1,
        clock=state.clock,
        u=state.u,
        b=state.b,
    )

    assert abs(friedmann_residual(inconsistent, parameters)) > 1.0
    assert not audit_reference_flrw_background(inconsistent, parameters).local_reference_patch_admitted


def test_rhs_preserves_clock_rod_charges_and_friedmann_constraint() -> None:
    state, parameters = admitted_state()
    rhs = flrw_rhs(state, parameters)

    clock_charge_derivative = np.exp(3.0 * state.n) * (rhs.u + 3.0 * rhs.n * state.u)
    rod_charge_derivative = np.exp(state.n) * (rhs.b + rhs.n * state.b)
    assert abs(clock_charge_derivative) < 1.0e-14
    assert abs(rod_charge_derivative) < 1.0e-14

    epsilon = 1.0e-7
    moved = ReferenceFlrwState.from_array(state.as_array() + epsilon * rhs.as_array())
    assert abs(friedmann_residual(moved, parameters)) < 1.0e-10


def test_conserved_charge_constructor_reproduces_state_data() -> None:
    state, parameters = admitted_state()
    rebuilt = state_from_conserved_charges(
        n=state.n,
        clock=state.clock,
        clock_charge=clock_charge_bar(state),
        rod_charge=rod_charge_bar(state),
        parameters=parameters,
    )

    assert np.allclose(rebuilt.as_array(), state.as_array())


def test_short_rk4_trajectory_preserves_constraint_charges_and_reference_patch() -> None:
    state, parameters = admitted_state(lam=0.0)
    trajectory = integrate_reference_flrw_background(
        state,
        parameters,
        tau_final=0.25,
        steps=250,
        ricci_scalar_cutoff_bar=1.0,
    )

    assert len(trajectory.states) == 251
    assert trajectory.max_friedmann_relative_residual < 1.0e-12
    assert trajectory.max_clock_charge_drift < 1.0e-12
    assert trajectory.max_rod_charge_drift < 1.0e-12
    assert trajectory.minimum_abs_reference_jacobian_bar > 0.0
    assert trajectory.all_local_reference_patches_admitted
    assert not trajectory.perturbation_stability_computed
    assert not trajectory.one_loop_st_identity_computed
    assert not trajectory.nonperturbative_m2_passed


def test_curvature_cutoff_is_a_real_kill_gate() -> None:
    state, parameters = admitted_state(lam=0.0)
    curvature = abs(curvature_scalar_bar(state, parameters))

    assert audit_reference_flrw_background(
        state,
        parameters,
        ricci_scalar_cutoff_bar=2.0 * curvature,
    ).local_reference_patch_admitted
    assert not audit_reference_flrw_background(
        state,
        parameters,
        ricci_scalar_cutoff_bar=0.5 * curvature,
    ).local_reference_patch_admitted


def test_clock_charge_forces_a_finite_past_bound_and_curvature_growth() -> None:
    state, parameters = admitted_state(lam=0.0)
    bound = clock_dominated_past_time_bound(state, parameters)
    states = [
        state_from_conserved_charges(
            n=n,
            clock=0.0,
            clock_charge=clock_charge_bar(state),
            rod_charge=rod_charge_bar(state),
            parameters=parameters,
        )
        for n in (0.0, -1.0, -2.0)
    ]
    curvature = [abs(curvature_scalar_bar(item, parameters)) for item in states]

    assert np.isfinite(bound) and bound > 0.0
    assert state.h >= abs(state.u) / (
        np.sqrt(6.0) * parameters.m_planck_over_mu_x
    )
    assert curvature[0] < curvature[1] < curvature[2]


def test_bare_kinetic_gram_is_positive_but_not_a_stability_certificate() -> None:
    eigenvalues = np.linalg.eigvalsh(bare_internal_kinetic_gram())
    audit = audit_reference_flrw_background(*admitted_state())

    assert np.all(eigenvalues > 0.0)
    assert audit.bare_homogeneous_kinetic_positive
    assert not audit.perturbation_stability_computed


def test_rod_only_and_clock_only_limits_are_not_accelerating_reference_solutions() -> None:
    parameters = ReferenceFlrwParameters(m_planck_over_mu_x=3.0)
    rod = state_from_conserved_charges(
        n=0.0,
        clock=0.0,
        clock_charge=0.0,
        rod_charge=np.sqrt(2.0) * parameters.m_planck_over_mu_x,
        parameters=parameters,
    )
    clock = state_from_conserved_charges(
        n=0.0,
        clock=0.0,
        clock_charge=np.sqrt(2.0 / 3.0) * parameters.m_planck_over_mu_x,
        rod_charge=0.0,
        parameters=parameters,
    )

    rod_acceleration = flrw_rhs(rod, parameters).h + rod.h**2
    clock_acceleration = flrw_rhs(clock, parameters).h + clock.h**2
    assert abs(rod_acceleration) < 1.0e-14
    assert clock_acceleration < 0.0
    assert not audit_reference_flrw_background(rod, parameters).local_reference_patch_admitted
    assert not audit_reference_flrw_background(clock, parameters).local_reference_patch_admitted


def test_negative_friedmann_radicand_and_invalid_parameters_fail_closed() -> None:
    negative = ReferenceFlrwParameters(
        m_planck_over_mu_x=1.0,
        lambda_over_mu_x_squared=-1.0,
    )
    with pytest.raises(ValueError, match='negative radicand'):
        expanding_h_from_constraint(u=0.1, b=0.1, parameters=negative)
    with pytest.raises(ValueError, match='positive'):
        expanding_h_from_constraint(
            u=0.1,
            b=0.1,
            parameters=ReferenceFlrwParameters(m_planck_over_mu_x=0.0),
        )


def test_integrator_rejects_degenerate_initial_reference_data() -> None:
    state, parameters = admitted_state(u=0.0)
    with pytest.raises(ValueError, match='initial state'):
        integrate_reference_flrw_background(
            state,
            parameters,
            tau_final=0.1,
            steps=10,
        )
