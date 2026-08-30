"""Focused tests for finite-time evolution in pole-free metric variables."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
)
from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(
    *,
    w_reservoir: float = 0.1,
    half_width: float = 0.5,
    omega_prod0: float = 0.12,
) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=half_width,
            omega_prod0=omega_prod0,
            reservoir_present_density=0.21,
            w_reservoir=w_reservoir,
            w_open=2.1767e-4,
        )
    )


def _evolution(
    *,
    w: float = 0.1,
    kappa_initial: float = 0.05,
    half_width: float = 0.5,
    omega_prod0: float = 0.12,
):
    bridge = _bridge(
        w_reservoir=w,
        half_width=half_width,
        omega_prod0=omega_prod0,
    )
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
    assert receipt.trace_generator_bound.trace_coefficients_bounded_on_interval
    assert (
        receipt.trace_flow_stability
        .finite_interval_continuous_dependence_bound_proven
    )
    assert receipt.source_edges_aligned_in_coarse_mesh
    assert receipt.analytic_resolution_bound_holds
    assert receipt.normalized_source_shape_resolution_holds
    assert receipt.refined_step_count == 2 * receipt.coarse_step_count
    assert receipt.maximum_characteristic_scale_step <= (
        1.0 + 256.0 * math.ulp(1.0)
    )
    assert receipt.kappa_final > receipt.kappa_initial
    assert not receipt.failure_reasons


@pytest.mark.parametrize(
    ("w", "half_width"),
    [
        (0.0, 0.5),
        (0.1, 0.5),
        (1.0, 0.5),
        (0.1, 1.0e-4),
    ],
)
def test_analytic_trace_bound_contains_sampled_coefficients(
    w: float,
    half_width: float,
) -> None:
    bridge, evolution = _evolution(w=w, half_width=half_width)
    bound = evolution.trace_generator_bound()
    expected_source_bound = (
        bridge.config.omega_prod0
        * math.exp(-3.0 * bridge.config.n_minus)
        * 15.0
        / (16.0 * bridge.config.half_width)
    )
    assert bound.source_upper_bound == pytest.approx(expected_source_bound)
    assert bound.source_right_endpoint == bridge.config.n_plus
    assert bound.source_enthalpy_lower_bound is not None
    assert bound.source_enthalpy_lower_bound > 0.0
    assert bound.component_density_nonnegativity_derived_from_bridge
    assert bound.source_bound_derived_analytically
    assert bound.enthalpy_monotonicity_proven
    assert bound.kappa_monotonicity_proven
    assert bound.trace_coefficients_bounded_on_interval
    assert not bound.mesh_rule_is_stability_or_error_theorem
    assert not bound.source_shape_step_floor_is_error_theorem

    nodes = {
        evolution.n_initial,
        evolution.n_final,
        bridge.config.n_minus,
        bridge.config.n_star,
        bridge.config.n_plus,
    }
    nodes.update(
        evolution.n_initial
        + (evolution.n_final - evolution.n_initial) * index / 256
        for index in range(257)
    )
    for n in sorted(nodes):
        _, _, negative_restoring, negative_damping = (
            evolution.trace_conditioned_matrix(n)
        )
        assert abs(negative_damping) <= (
            bound.damping_upper_bound
            + 1.0e-12 * max(1.0, bound.damping_upper_bound)
        )
        assert abs(negative_restoring) <= (
            bound.restoring_absolute_upper_bound
            + 1.0e-12 * max(
                1.0,
                bound.restoring_absolute_upper_bound,
            )
        )


@pytest.mark.parametrize(
    ("w", "half_width"),
    [
        (0.0, 0.5),
        (0.1, 0.5),
        (1.0, 0.5),
        (0.1, 1.0e-4),
    ],
)
def test_trace_coefficient_signs_follow_exact_positive_decompositions(
    w: float,
    half_width: float,
) -> None:
    bridge, evolution = _evolution(w=w, half_width=half_width)
    nodes = {
        evolution.n_initial,
        evolution.n_final,
        bridge.config.n_minus,
        bridge.config.n_star,
        bridge.config.n_plus,
    }
    nodes.update(
        evolution.n_initial
        + (evolution.n_final - evolution.n_initial) * index / 64
        for index in range(65)
    )
    for n in sorted(nodes):
        background = FiniteQuenchTwoFluidFlatGRBackground(
            bridge
        ).construct(n)
        rho_p = background.produced_density
        rho_r = background.reservoir_density
        total_density = background.total_density
        enthalpy = background.total_enthalpy
        source = background.produced_source
        pressure_ratio = (
            w * (3.0 * (1.0 + w) * rho_r + source) / enthalpy
        )
        kappa = evolution.reduced.k_over_a_h(n)
        damping_excess = (
            3.0
            * w
            * rho_r
            * (
                (1.0 + 2.0 * w) * rho_p
                + (1.0 + w) * rho_r
            )
            / (2.0 * total_density * enthalpy)
            + w * source / enthalpy
        )
        restoring_decomposition = (
            3.0
            * w
            * w
            * rho_p
            * rho_r
            / (total_density * enthalpy)
            + w * source / enthalpy
            + (kappa * kappa / 3.0) * pressure_ratio
        )
        _, _, negative_restoring, negative_damping = (
            evolution.trace_conditioned_matrix(n)
        )
        damping = -negative_damping
        restoring = -negative_restoring
        assert damping == pytest.approx(
            2.5 + damping_excess,
            rel=2.0e-12,
            abs=2.0e-12,
        )
        assert restoring == pytest.approx(
            restoring_decomposition,
            rel=2.0e-12,
            abs=2.0e-12,
        )
        assert damping >= 2.5 - 2.0e-12
        assert restoring >= -2.0e-12


@pytest.mark.parametrize(
    ("w", "half_width"),
    [
        (0.0, 0.5),
        (0.1, 0.5),
        (1.0, 0.5),
        (0.1, 1.0e-4),
    ],
)
def test_trace_flow_receipt_proves_only_the_safe_stability_statements(
    w: float,
    half_width: float,
) -> None:
    _, evolution = _evolution(w=w, half_width=half_width)
    flow = evolution.trace_flow_stability_bound()
    bound = flow.generator_bound
    assert flow.damping_lower_bound == 2.5
    assert flow.restoring_lower_bound == 0.0
    assert flow.wronskian_log_ratio_lower_bound <= (
        flow.wronskian_log_ratio_upper_bound
    )
    assert flow.wronskian_log_ratio_upper_bound < 0.0
    assert flow.wronskian_contraction_factor_representable
    assert flow.wronskian_contraction_factor_upper_bound is not None
    assert flow.wronskian_contraction_factor_upper_bound == pytest.approx(
        math.exp(flow.wronskian_log_ratio_upper_bound)
    )
    assert flow.direct_euclidean_weight == 1.0
    assert flow.balanced_weight == pytest.approx(
        0.5 * bound.restoring_absolute_upper_bound
    )
    assert flow.selected_log_amplification_upper_bound == min(
        flow.direct_euclidean_log_amplification_upper_bound,
        flow.balanced_log_amplification_upper_bound,
    )
    assert flow.selected_log_amplification_upper_bound == pytest.approx(
        flow.selected_euclidean_conversion_log_penalty
        + flow.selected_logarithmic_norm_rate_upper_bound
        * flow.interval_width
    )
    assert math.isfinite(flow.selected_log_amplification_upper_bound)
    assert flow.coefficient_signs_derived_analytically
    assert flow.frozen_generator_has_no_positive_real_eigenvalue
    assert flow.wronskian_identity_proven
    assert flow.fundamental_matrix_invertibility_proven
    assert flow.forward_phase_area_contraction_proven
    assert flow.finite_interval_continuous_dependence_bound_proven
    assert not flow.individual_solution_norm_monotone_decay_proven
    assert not flow.no_transient_growth_proven
    assert not flow.numerical_method_stability_theorem_proven
    assert not flow.rigorous_interval_enclosure_proven


def test_wronskian_underflow_keeps_the_log_bound_without_false_zero() -> None:
    representable = (
        FiniteQuenchRegularMetricEvolution
        ._representable_wronskian_contraction_factor(-1.0)
    )
    underflowed = (
        FiniteQuenchRegularMetricEvolution
        ._representable_wronskian_contraction_factor(-1_000.0)
    )
    assert representable == pytest.approx(math.exp(-1.0))
    assert underflowed is None


def test_conditional_duhamel_receipt_propagates_assumed_p_norm_bounds() -> None:
    _, evolution = _evolution()
    receipt = evolution.trace_residual_error_bound(
        initial_defect_p_upper_bound=2.0e-6,
        terminal_weighted_residual_p_upper_bound=3.0e-6,
    )
    expected_metric = (
        math.exp(receipt.metric_log_propagator_upper_bound) * 2.0e-6
        + 3.0e-6
    )
    expected_euclidean = expected_metric / math.sqrt(
        min(receipt.weight_p, 1.0)
    )
    assert receipt.metric_endpoint_error_upper_bound == pytest.approx(
        expected_metric
    )
    assert receipt.euclidean_endpoint_error_upper_bound == pytest.approx(
        expected_euclidean
    )
    assert receipt.endpoint_error_radius_representable
    assert not receipt.endpoint_error_exactly_zero_under_assumptions
    assert receipt.dimensionless_contract_assumed_by_normalized_system
    assert receipt.duhamel_identity_proven
    assert receipt.fixed_weight_metric_error_bound_proven
    assert receipt.conditional_a_posteriori_error_bound_proven
    assert not receipt.approximate_path_absolute_continuity_verified_by_module
    assert not receipt.dense_output_residual_certified_by_module
    assert not receipt.initial_defect_certified_by_module
    assert not receipt.piecewise_join_defects_included_by_module
    assert not receipt.coefficient_interval_enclosure_proven
    assert not receipt.outward_rounding_proven
    assert not receipt.floating_point_evaluation_is_rigorous
    assert not receipt.rigorous_interval_enclosure_proven
    assert receipt.weight_p > 1.0
    assert receipt.metric_to_euclidean_log_factor == 0.0
    assert (
        receipt.flow_stability.selected_euclidean_conversion_log_penalty
        > 0.0
    )


def test_conditional_duhamel_receipt_handles_zero_and_overflow_in_log_domain() -> None:
    _, evolution = _evolution()
    zero = evolution.trace_residual_error_bound(
        initial_defect_p_upper_bound=0.0,
        terminal_weighted_residual_p_upper_bound=0.0,
    )
    assert zero.metric_endpoint_error_log_upper_bound is None
    assert zero.euclidean_endpoint_error_log_upper_bound is None
    assert zero.metric_endpoint_error_upper_bound == 0.0
    assert zero.euclidean_endpoint_error_upper_bound == 0.0
    assert zero.endpoint_error_exactly_zero_under_assumptions
    assert zero.endpoint_error_radius_representable

    overflow = evolution.trace_residual_error_bound(
        initial_defect_p_upper_bound=1.0e308,
        terminal_weighted_residual_p_upper_bound=1.0e308,
    )
    assert overflow.metric_endpoint_error_log_upper_bound is not None
    assert overflow.euclidean_endpoint_error_log_upper_bound is not None
    assert overflow.euclidean_endpoint_error_upper_bound is None
    assert not overflow.endpoint_error_radius_representable


@pytest.mark.parametrize(
    ("initial_defect", "weighted_residual"),
    [(-1.0, 0.0), (0.0, -1.0), (math.inf, 0.0), (0.0, math.nan)],
)
def test_conditional_duhamel_receipt_rejects_invalid_assumed_bounds(
    initial_defect: float,
    weighted_residual: float,
) -> None:
    _, evolution = _evolution()
    with pytest.raises(ValueError):
        evolution.trace_residual_error_bound(
            initial_defect_p_upper_bound=initial_defect,
            terminal_weighted_residual_p_upper_bound=weighted_residual,
        )


def test_source_edges_are_exact_nodes_of_the_nested_mesh() -> None:
    bridge, evolution = _evolution(half_width=1.0e-4)
    bound = evolution.trace_generator_bound()
    target = max(
        512,
        math.ceil(
            (evolution.n_final - evolution.n_initial)
            * bound.characteristic_rate_upper_bound
        ),
    )
    coarse = evolution._piecewise_coarse_mesh(target)
    refined = evolution._refined_mesh(coarse)
    assert bridge.config.n_minus in coarse
    assert bridge.config.n_plus in coarse
    assert refined[::2] == coarse
    assert len(refined) - 1 == 2 * (len(coarse) - 1)
    assert evolution._active_source_step_count(coarse) >= (
        bound.minimum_active_source_coarse_steps
    )
    maximum_step = max(
        right - left
        for left, right in zip(coarse[:-1], coarse[1:], strict=True)
    )
    assert maximum_step * bound.characteristic_rate_upper_bound <= (
        1.0 + 256.0 * math.ulp(1.0)
    )


def test_narrow_source_is_resolved_and_step_doubling_converges() -> None:
    _, evolution = _evolution(half_width=1.0e-4)
    receipt = evolution.construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
        relative_tolerance=1.0e-8,
    )
    assert receipt.active_source_coarse_step_count >= (
        receipt.trace_generator_bound.minimum_active_source_coarse_steps
    )
    assert receipt.source_edges_aligned_in_coarse_mesh
    assert receipt.analytic_resolution_bound_holds
    assert receipt.normalized_source_shape_resolution_holds
    assert receipt.magnus_step_doubling_converged
    assert receipt.finite_time_source_on_evolution_numerically_verified
    assert not receipt.failure_reasons


def test_pathologically_narrow_source_fails_before_excessive_work() -> None:
    _, evolution = _evolution(half_width=1.0e-8)
    with pytest.raises(ValueError, match="source-aware resolution"):
        evolution.construct(
            primordial_potential_amplitude=1.0e-5,
            coarse_step_count=512,
            relative_tolerance=1.0e-8,
        )


@pytest.mark.parametrize("w", [-1.0, -0.5, 1.1, 1000.0])
def test_evolution_rejects_reservoir_barotropes_outside_strict_branch(
    w: float,
) -> None:
    bridge = _bridge(w_reservoir=w)
    with pytest.raises(ValueError, match="0 <= w_R <= 1"):
        FiniteQuenchRegularMetricEvolution(
            bridge,
            n_initial=-5.0,
            kappa_initial=0.05,
        )


def test_zero_source_needs_no_normalized_source_shape_floor() -> None:
    _, evolution = _evolution(omega_prod0=0.0)
    receipt = evolution.construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
        relative_tolerance=1.0e-8,
    )
    assert receipt.trace_generator_bound.source_upper_bound == 0.0
    assert receipt.active_source_coarse_step_count == 0
    assert receipt.normalized_source_shape_resolution_holds
    assert receipt.magnus_step_doubling_converged


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
    assert receipt.maximum_characteristic_scale_step <= (
        1.0 + 256.0 * math.ulp(1.0)
    )
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
    assert (
        not receipt.trace_flow_stability
        .numerical_method_stability_theorem_proven
    )
    assert not receipt.trace_flow_stability.no_transient_growth_proven
    assert not receipt.microphysical_covariant_transfer_law_proven
    assert not receipt.primordial_amplitude_predicted
    assert not receipt.observable_transfer_function_proven
    assert "STEP_DOUBLED_FINITE_TIME" in receipt.role
    assert "MAGNUS" in receipt.role
    assert "NOT_INTERVAL_MICROPHYSICAL" in receipt.role
