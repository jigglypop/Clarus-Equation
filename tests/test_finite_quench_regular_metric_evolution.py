"""Focused tests for finite-time evolution in pole-free metric variables."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(*, w_reservoir: float = 0.1) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=0.12,
            reservoir_present_density=0.21,
            w_reservoir=w_reservoir,
            w_open=2.1767e-4,
        )
    )


def _evolution(*, w: float = 0.1, kappa_initial: float = 0.05):
    bridge = _bridge(w_reservoir=w)
    return bridge, FiniteQuenchRegularMetricEvolution(
        bridge,
        n_initial=-5.0,
        kappa_initial=kappa_initial,
    )


def _construct(*, steps: int = 512, amplitude: float = 1.0e-5):
    bridge, evolution = _evolution()
    return bridge, evolution, evolution.construct(
        primordial_potential_amplitude=amplitude,
        coarse_step_count=steps,
        relative_tolerance=1.0e-8,
    )


def test_step_doubled_regular_evolution_crosses_source_and_reconstructs_today() -> None:
    _, _, receipt = _construct()
    assert receipt.initial_regular_mode_holds
    assert receipt.regular_metric_coefficients_continuous_on_domain
    assert receipt.source_support_was_traversed
    assert receipt.magnus_step_doubling_converged
    assert receipt.final_effective_full_reconstruction_holds
    assert receipt.final_regular_rhs_matches_full_system
    assert receipt.finite_time_source_on_evolution_numerically_verified
    assert receipt.kappa_final > receipt.kappa_initial
    assert not receipt.failure_reasons


def test_general_regular_metric_matrix_matches_full_system_across_source() -> None:
    bridge, evolution = _evolution()
    for n in (-5.0, -4.5, -4.0, -3.5, 0.0):
        clock = 0.01
        psi = 0.002
        background = evolution.reduced.construct(
            n=n,
            scalar_clock_shift=0.0,
            total_momentum_density=0.0,
        ).common_clock_second_tangent.common_clock_tangent.gr_linear_node.background
        kappa = evolution.reduced.k_over_a_h(n)
        coupling = background.gravity_constraint_coupling
        total_u = (
            kappa**2 * psi / (3.0 * coupling)
            - background.total_enthalpy * clock
        )
        full = evolution.reduced.construct(
            n=n,
            scalar_clock_shift=clock,
            total_momentum_density=total_u,
        )
        clock_rhs, psi_rhs = evolution.rhs(n, clock, psi)
        assert clock_rhs == pytest.approx(full.full_clock_log_derivative)
        assert psi_rhs == pytest.approx(
            full.algebraic_metric_tangent
            .direct_algebraic_curvature_potential_log_derivative
        )
        assert full.conditional_effective_full_reconstruction_holds
    assert bridge.config.n_minus == -4.5
    assert bridge.config.n_plus == -3.5


def test_trace_conditioned_generator_is_the_same_strict_trace_equation() -> None:
    _, evolution = _evolution()
    for n in (-5.0, -4.5, -4.0, -3.5, 0.0):
        clock = 0.01
        psi = 0.002
        _, psi_prime = evolution.rhs(n, clock, psi)
        reconstructed_clock = evolution._clock_from_trace_state(
            n,
            psi,
            psi_prime,
        )
        m11, m12, m21, m22 = evolution.trace_conditioned_matrix(n)
        psi_second = m21 * psi + m22 * psi_prime
        background = evolution.reduced.construct(
            n=n,
            scalar_clock_shift=0.0,
            total_momentum_density=0.0,
        ).common_clock_second_tangent.common_clock_tangent.gr_linear_node.background
        delta_pressure = (
            evolution.bridge.config.w_reservoir
            * background.reservoir_density_derivative
            * clock
        )
        trace_second = (
            background.gravity_constraint_coupling * delta_pressure
            - (4.0 + background.hubble_log_derivative) * psi_prime
            - (3.0 + 2.0 * background.hubble_log_derivative) * psi
        )
        assert m11 == 0.0
        assert m12 == 1.0
        assert reconstructed_clock == pytest.approx(clock)
        assert psi_second == pytest.approx(trace_second)


def test_magnus_step_doubling_displays_fourth_order_convergence() -> None:
    bridge = _bridge()
    evolution = FiniteQuenchRegularMetricEvolution(
        bridge,
        n_initial=-5.0,
        kappa_initial=0.05,
        n_final=bridge.config.n_minus,
    )
    coarse = evolution.construct(
        primordial_potential_amplitude=1.0e-2,
        coarse_step_count=16,
        relative_tolerance=1.0,
    )
    fine = evolution.construct(
        primordial_potential_amplitude=1.0e-2,
        coarse_step_count=32,
        relative_tolerance=1.0,
    )
    assert fine.curvature_richardson_error_estimate < (
        coarse.curvature_richardson_error_estimate / 12.0
    )
    assert fine.scalar_clock_richardson_error_estimate < (
        coarse.scalar_clock_richardson_error_estimate / 12.0
    )


def test_final_transfer_is_linear_in_free_initial_amplitude() -> None:
    _, evolution = _evolution()
    base = evolution.construct(
        primordial_potential_amplitude=2.0e-5,
        coarse_step_count=256,
        relative_tolerance=1.0e-7,
    )
    factor = -3.0
    scaled = evolution.construct(
        primordial_potential_amplitude=factor * 2.0e-5,
        coarse_step_count=256,
        relative_tolerance=1.0e-7,
    )
    assert scaled.refined_final_scalar_clock_shift == pytest.approx(
        factor * base.refined_final_scalar_clock_shift
    )
    assert scaled.refined_final_curvature_potential == pytest.approx(
        factor * base.refined_final_curvature_potential
    )
    assert scaled.curvature_transfer_per_unit_initial_amplitude == (
        pytest.approx(base.curvature_transfer_per_unit_initial_amplitude)
    )


def test_zero_free_amplitude_remains_zero_without_a_nan_transfer() -> None:
    _, evolution = _evolution()
    receipt = evolution.construct(
        primordial_potential_amplitude=0.0,
        coarse_step_count=128,
        relative_tolerance=1.0e-8,
    )
    assert receipt.finite_time_source_on_evolution_numerically_verified
    assert receipt.refined_final_scalar_clock_shift == 0.0
    assert receipt.refined_final_curvature_potential == 0.0
    assert receipt.curvature_transfer_per_unit_initial_amplitude is None


def test_final_regular_rhs_is_independently_reconstructed() -> None:
    _, _, receipt = _construct()
    assert receipt.final_clock_rhs_residual == pytest.approx(0.0, abs=1.0e-12)
    assert receipt.final_curvature_rhs_residual == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    final = receipt.final_reduced_ode
    assert final.scalar_clock_shift == pytest.approx(
        receipt.refined_final_scalar_clock_shift
    )
    assert final.total_momentum_density == pytest.approx(
        receipt.refined_final_total_momentum_density
    )


@pytest.mark.parametrize("w", [0.0, 1.0])
def test_causal_sound_speed_boundaries_can_cross_the_source(w: float) -> None:
    _, evolution = _evolution(w=w)
    receipt = evolution.construct(
        primordial_potential_amplitude=1.0e-6,
        coarse_step_count=512,
        relative_tolerance=2.0e-7,
    )
    assert receipt.finite_time_source_on_evolution_numerically_verified
    assert receipt.maximum_final_phase_step <= 1.0
    if w == 1.0:
        assert receipt.requested_coarse_step_count == 512
        assert receipt.coarse_step_count > receipt.requested_coarse_step_count


@pytest.mark.parametrize("kappa_initial", [0.02, 0.08])
def test_multiple_regular_initial_scales_cross_source(kappa_initial: float) -> None:
    _, evolution = _evolution(kappa_initial=kappa_initial)
    receipt = evolution.construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
        relative_tolerance=2.0e-7,
    )
    assert receipt.finite_time_source_on_evolution_numerically_verified


def test_stopping_inside_source_is_not_reported_as_source_traversal() -> None:
    bridge = _bridge()
    evolution = FiniteQuenchRegularMetricEvolution(
        bridge,
        n_initial=-5.0,
        kappa_initial=0.05,
        n_final=-4.0,
    )
    receipt = evolution.construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=128,
        relative_tolerance=1.0e-7,
    )
    assert not receipt.source_support_was_traversed
    assert not receipt.finite_time_source_on_evolution_numerically_verified
    assert "SOURCE_SUPPORT_NOT_TRAVERSED" in receipt.failure_reasons


def test_overly_strict_tolerance_fails_closed() -> None:
    _, evolution = _evolution()
    receipt = evolution.construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=32,
        relative_tolerance=1.0e-14,
    )
    assert not receipt.magnus_step_doubling_converged
    assert not receipt.finite_time_source_on_evolution_numerically_verified
    assert "MAGNUS_STEP_DOUBLING_NOT_CONVERGED" in receipt.failure_reasons


@pytest.mark.parametrize("bad_steps", [True, 1.5, 15])
def test_bad_step_counts_are_rejected(bad_steps) -> None:
    _, evolution = _evolution()
    with pytest.raises(ValueError, match="coarse_step_count"):
        evolution.construct(
            primordial_potential_amplitude=1.0e-5,
            coarse_step_count=bad_steps,
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True])
def test_bad_tolerances_are_rejected(bad) -> None:
    _, evolution = _evolution()
    with pytest.raises(ValueError):
        evolution.construct(
            primordial_potential_amplitude=1.0e-5,
            relative_tolerance=bad,
        )


def test_invalid_evolution_intervals_and_initial_kappa_are_rejected() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="pre-source"):
        FiniteQuenchRegularMetricEvolution(
            bridge,
            n_initial=-4.0,
            kappa_initial=0.05,
        )
    with pytest.raises(ValueError, match="kappa_initial"):
        FiniteQuenchRegularMetricEvolution(
            bridge,
            n_initial=-5.0,
            kappa_initial=0.2,
        )
    with pytest.raises(ValueError, match="n_final"):
        FiniteQuenchRegularMetricEvolution(
            bridge,
            n_initial=-5.0,
            kappa_initial=0.05,
            n_final=-5.1,
        )


def test_receipt_keeps_numerical_evolution_below_interval_and_observable_proof() -> None:
    _, _, receipt = _construct()
    assert not receipt.rigorous_interval_enclosure_proven
    assert not receipt.numerical_method_stability_theorem_proven
    assert not receipt.microphysical_covariant_transfer_law_proven
    assert not receipt.primordial_amplitude_predicted
    assert not receipt.observable_transfer_function_proven
    assert "STEP_DOUBLED_FINITE_TIME" in receipt.role
    assert "MAGNUS" in receipt.role
    assert "NOT_INTERVAL_MICROPHYSICAL" in receipt.role
