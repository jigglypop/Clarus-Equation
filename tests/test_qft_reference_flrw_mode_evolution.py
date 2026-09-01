import numpy as np
import pytest

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_mode_evolution import (
    E67_CHI_MASS_OVER_MU_X,
    E67_COARSE_STEPS,
    E67_FINE_STEPS,
    E67_SECOND_ORDER_TOL,
    E67_TAU_FINAL,
    E67_WAVENUMBERS_BAR,
    audit_preregistered_e67_witness,
    canonical_mode_generators,
    fundamental_matrix_diagnostics,
    hamiltonian_generator_residual,
    integrate_finite_time_modes,
    scalar_second_order_equivalence_residual,
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


def test_all_instantaneous_mode_generators_are_canonical_hamiltonian() -> None:
    state, parameters = reference_state()
    generators = canonical_mode_generators(
        state,
        parameters,
        comoving_wavenumber_bar=0.2,
        chi_mass_over_mu_x=0.4,
    )

    for generator in (
        generators.tensor,
        generators.vector,
        generators.scalar,
        generators.chi,
    ):
        assert hamiltonian_generator_residual(generator) < 1.0e-12
        assert abs(np.trace(generator)) < 1.0e-12
    assert generators.vector_kinetic_ratio > 0.0
    assert generators.minimum_scalar_kinetic_eigenvalue > 0.0
    assert generators.maximum_hamiltonian_generator_residual < 1.0e-12
    residual = scalar_second_order_equivalence_residual(
        state,
        parameters,
        comoving_wavenumber_bar=0.2,
        chi_mass_over_mu_x=0.4,
        scalar_fundamental=np.eye(4),
        scalar_generator=generators.scalar,
    )
    assert residual < E67_SECOND_ORDER_TOL


def test_short_joint_run_preserves_every_canonical_symplectic_form() -> None:
    state, parameters = reference_state()
    run = integrate_finite_time_modes(
        state,
        parameters,
        tau_final=0.05,
        steps=200,
        wavenumbers_bar=(0.2, 1.0),
        chi_mass_over_mu_x=0.4,
    )

    assert run.maximum_friedmann_relative_residual < 1.0e-12
    assert run.minimum_abs_reference_jacobian_bar > 0.0
    for evolution in run.wavenumber_evolutions:
        assert evolution.maximum_symplectic_residual < 1.0e-10
        assert evolution.maximum_determinant_residual < 1.0e-10
        assert evolution.minimum_vector_kinetic_ratio > 0.0
        assert evolution.minimum_scalar_kinetic_eigenvalue > 0.0
        assert evolution.minimum_abs_scalar_constraint_determinant > 0.0
        assert evolution.maximum_scalar_second_order_residual < E67_SECOND_ORDER_TOL


def test_preregistered_e67_witness_passes_without_changing_its_contract() -> None:
    audit = audit_preregistered_e67_witness()

    assert audit.coarse_run.tau_final == E67_TAU_FINAL
    assert audit.coarse_run.steps == E67_COARSE_STEPS
    assert audit.fine_run.steps == E67_FINE_STEPS
    assert tuple(
        evolution.comoving_wavenumber_bar
        for evolution in audit.fine_run.wavenumber_evolutions
    ) == E67_WAVENUMBERS_BAR
    assert audit.maximum_refinement_residual < 1.0e-6
    assert audit.maximum_symplectic_residual < 1.0e-8
    assert audit.maximum_determinant_residual < 1.0e-8
    assert audit.maximum_largest_singular_value < 100.0
    assert audit.fine_run.maximum_friedmann_relative_residual < 1.0e-10
    assert audit.minimum_vector_kinetic_ratio > 0.0
    assert audit.minimum_scalar_kinetic_eigenvalue > 0.0
    assert audit.minimum_abs_scalar_constraint_determinant > 0.0
    assert audit.maximum_scalar_second_order_residual < E67_SECOND_ORDER_TOL
    assert audit.finite_time_symplectic_gate_passed
    assert audit.predeclared_finite_interval_only
    assert not audit.uniform_wavenumber_band_proved
    assert not audit.asymptotic_stability_proved
    assert not audit.nonlinear_stability_proved
    assert not audit.strong_coupling_scale_derived
    assert not audit.one_loop_st_identity_computed
    assert not audit.brst_physical_inner_product_constructed
    assert not audit.nonperturbative_m2_passed


def test_non_symplectic_scaling_is_a_real_negative_control() -> None:
    bad = 1.01 * np.eye(4)
    diagnostics = fundamental_matrix_diagnostics(bad)

    assert diagnostics.symplectic_residual > 1.0e-3
    assert diagnostics.determinant_residual > 1.0e-3


def test_mode_evolution_rejects_k_zero_duplicate_k_and_negative_chi_mass() -> None:
    state, parameters = reference_state()
    with pytest.raises(ValueError, match='positive k'):
        canonical_mode_generators(
            state,
            parameters,
            comoving_wavenumber_bar=0.0,
            chi_mass_over_mu_x=0.4,
        )
    with pytest.raises(ValueError, match='unique'):
        integrate_finite_time_modes(
            state,
            parameters,
            tau_final=0.1,
            steps=10,
            wavenumbers_bar=(0.2, 0.2),
            chi_mass_over_mu_x=0.4,
        )
    with pytest.raises(ValueError, match='nonnegative'):
        integrate_finite_time_modes(
            state,
            parameters,
            tau_final=0.1,
            steps=10,
            wavenumbers_bar=(0.2,),
            chi_mass_over_mu_x=-0.1,
        )


def test_preregistered_chi_mass_is_not_silently_changed() -> None:
    assert E67_CHI_MASS_OVER_MU_X == 0.4
