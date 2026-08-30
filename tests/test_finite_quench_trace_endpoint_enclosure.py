"""Focused tests for the exact-rational trace endpoint enclosure."""

from __future__ import annotations

from fractions import Fraction
import math

import pytest

import examples.physics.finite_quench_trace_endpoint_enclosure as endpoint_module
from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    FiniteQuenchTraceEndpointEnclosure,
    _rational_exp_bounds,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _evolution(
    *,
    w: float = 0.1,
    half_width: float = 0.5,
    reservoir_present_density: float = 0.21,
) -> FiniteQuenchRegularMetricEvolution:
    bridge = FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=half_width,
            omega_prod0=0.12,
            reservoir_present_density=reservoir_present_density,
            w_reservoir=w,
            w_open=2.1767e-4,
        )
    )
    return FiniteQuenchRegularMetricEvolution(
        bridge,
        n_initial=-5.0,
        kappa_initial=0.05,
    )


@pytest.mark.parametrize(
    "value",
    [Fraction(0), Fraction(1, 10), Fraction(27, 2), Fraction(21)],
)
def test_rational_taylor_exp_bounds_contain_reference_value(
    value: Fraction,
) -> None:
    lower, upper, term_count = _rational_exp_bounds(value)
    reference = math.exp(float(value))
    assert float(lower) <= reference <= float(upper)
    assert lower <= upper
    assert Fraction(term_count + 2) > value


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (Fraction(1), Fraction(1)),
        (Fraction(4), Fraction(9)),
        (Fraction(1, 3), Fraction(7, 5)),
    ],
)
def test_inverse_square_root_interval_uses_exact_square_inequalities(
    lower: Fraction,
    upper: Fraction,
) -> None:
    result = endpoint_module._inverse_sqrt_interval(
        endpoint_module._RationalInterval(lower, upper)
    )

    assert result.lower > 0
    assert result.lower * result.lower * upper <= 1
    assert result.upper * result.upper * lower >= 1


def test_inverse_square_root_interval_rejects_nonpositive_domain() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        endpoint_module._inverse_sqrt_interval(
            endpoint_module._RationalInterval(Fraction(0), Fraction(1))
        )


def test_outward_dyadic_rounding_and_negative_division_keep_inclusion() -> None:
    original = endpoint_module._RationalInterval(
        Fraction(-17, 13),
        Fraction(-11, 17),
    )
    rounded = endpoint_module._outward_dyadic(original)
    quantum = Fraction(1, 1 << endpoint_module._INTERVAL_DYADIC_BITS)

    assert rounded.lower <= original.lower <= original.upper <= rounded.upper
    assert original.lower - rounded.lower < quantum
    assert rounded.upper - original.upper < quantum

    numerator = endpoint_module._RationalInterval(
        Fraction(-7, 5),
        Fraction(11, 7),
    )
    denominator = endpoint_module._RationalInterval(
        Fraction(-13, 5),
        Fraction(-9, 8),
    )
    quotient = endpoint_module._interval_divide(numerator, denominator)
    exact_corners = tuple(
        value / divisor
        for value in (numerator.lower, numerator.upper)
        for divisor in (denominator.lower, denominator.upper)
    )
    assert quotient.lower <= min(exact_corners)
    assert quotient.upper >= max(exact_corners)


def _simpson_weighted_source_reference(
    n: Fraction,
    parameters: endpoint_module._FrozenTraceParameters,
) -> float:
    left = max(float(n), float(parameters.source_minus))
    right = min(0.0, float(parameters.source_plus))
    if left >= right:
        return 0.0
    subdivisions = 4096
    step = (right - left) / subdivisions
    center = float(parameters.center)
    width = float(parameters.width)
    omega = float(parameters.omega)
    w = float(parameters.w)

    def integrand(value: float) -> float:
        scaled = (value - center) / width
        bump = 15.0 * (1.0 - scaled * scaled) ** 2 / (16.0 * width)
        return omega * math.exp(3.0 * w * value) * bump

    total = integrand(left) + integrand(right)
    total += 4.0 * sum(
        integrand(left + index * step)
        for index in range(1, subdivisions, 2)
    )
    total += 2.0 * sum(
        integrand(left + index * step)
        for index in range(2, subdivisions, 2)
    )
    return total * step / 3.0


@pytest.mark.parametrize("w", [0.0, 1.0])
def test_weighted_source_interval_covers_support_and_causal_edges(
    w: float,
) -> None:
    enclosure = FiniteQuenchTraceEndpointEnclosure(_evolution(w=w))
    parameters = enclosure._frozen_parameters()
    points = (
        parameters.source_minus - parameters.width,
        parameters.source_minus,
        parameters.center,
        parameters.source_plus,
        parameters.source_plus + parameters.width,
    )

    for point in points:
        interval = endpoint_module._weighted_source_integral_interval(
            point,
            parameters,
        )
        if parameters.w == 0:
            active_left = max(point, parameters.source_minus)
            active_right = min(Fraction(0), parameters.source_plus)
            exact_reference = (
                Fraction(0)
                if active_left >= active_right
                else parameters.omega
                * (
                    endpoint_module._compact_cumulative_at(
                        active_right,
                        parameters,
                    )
                    - endpoint_module._compact_cumulative_at(
                        active_left,
                        parameters,
                    )
                )
            )
            assert interval.lower <= exact_reference <= interval.upper
        else:
            reference = _simpson_weighted_source_reference(point, parameters)
            assert float(interval.lower) <= reference <= float(interval.upper)
        assert interval.lower >= 0

    assert endpoint_module._weighted_source_integral_interval(
        parameters.source_plus,
        parameters,
    ) == endpoint_module._point_interval(0)
    assert endpoint_module._weighted_source_integral_interval(
        parameters.source_plus + parameters.width,
        parameters,
    ) == endpoint_module._point_interval(0)


def test_default_trace_endpoint_has_exact_rational_materialized_ball() -> None:
    evolution = _evolution()
    enclosure = FiniteQuenchTraceEndpointEnclosure(evolution).construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
    )
    numerical = evolution.construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
        relative_tolerance=1.0e-8,
    )

    assert (
        enclosure.coefficient_bounds
        .coefficient_bounds_proven_on_full_interval
    )
    assert (
        enclosure.coefficient_bounds.damping_lower_bound
        == Fraction(5, 2)
    )
    assert enclosure.coefficient_bounds.restoring_lower_bound == 0
    assert enclosure.coefficient_bounds.restoring_upper_bound > 0
    assert enclosure.residual_l1_integral_upper_bound > 0
    assert enclosure.propagation_exponent_upper_bound >= 0
    assert (
        enclosure.local_interval_residual_l1_integral_upper_bound
        < enclosure.global_triangle_residual_l1_integral_upper_bound
    )
    assert (
        enclosure
        .local_integrated_logarithmic_norm_exponent_upper_bound
        < enclosure.global_propagation_exponent_upper_bound
    )
    assert enclosure.materialized_rational_radius_upper_bound is not None
    assert enclosure.materialized_rational_radius_upper_bound > 0
    assert enclosure.materialized_float_radius_upper_bound is not None
    assert (
        enclosure.materialized_float_radius_upper_bound
        < abs(float(enclosure.endpoint_curvature_center))
    )
    assert enclosure.curvature_component_interval is not None
    assert enclosure.curvature_component_interval[0] > 0
    assert enclosure.curvature_component_certified_sign == 1
    assert enclosure.curvature_prime_component_certified_sign is None
    assert Fraction.from_float(
        enclosure.materialized_float_radius_upper_bound
    ) >= enclosure.materialized_rational_radius_upper_bound
    assert float(enclosure.endpoint_curvature_center) == (
        numerical.refined_final_curvature_potential
    )
    assert enclosure.refined_step_count == numerical.refined_step_count
    assert enclosure.refined_nodes_frozen_as_exact_binary_rationals
    assert enclosure.continuous_piecewise_linear_reconstruction_proven
    assert enclosure.piecewise_join_defect_zero_proven
    assert enclosure.local_coefficient_interval_enclosures_proven
    assert enclosure.local_residual_cancellation_retained
    assert enclosure.local_time_dependent_logarithmic_norm_integrated
    assert (
        enclosure
        .residual_integral_bound_proven_by_exact_rational_arithmetic
    )
    assert enclosure.logarithmic_norm_propagation_bound_proven
    assert enclosure.exact_symbolic_trace_endpoint_ball_proven
    assert enclosure.rigorous_materialized_trace_endpoint_enclosure_proven
    initial_bridge = enclosure.regular_initial_bridge
    assert initial_bridge.curvature_defect_abs_upper_bound > 0
    assert initial_bridge.curvature_prime_defect_abs_upper_bound > 0
    assert initial_bridge.initial_l1_defect_upper_bound == (
        initial_bridge.curvature_defect_abs_upper_bound
        + initial_bridge.curvature_prime_defect_abs_upper_bound
    )
    assert initial_bridge.exact_component_difference_enclosures_proven
    assert initial_bridge.euclidean_initial_defect_bounded_by_l1
    assert initial_bridge.floating_series_and_rhs_roundoff_absorbed
    assert initial_bridge.analytic_regular_trace_initial_state_enclosed
    assert not initial_bridge.physical_primordial_amplitude_supplied
    assert not initial_bridge.scalar_clock_initial_state_enclosed
    assert enclosure.analytic_regular_endpoint_radius.coefficient == (
        enclosure.residual_l1_integral_upper_bound
        + initial_bridge.initial_l1_defect_upper_bound
    )
    assert (
        enclosure.analytic_regular_materialized_rational_radius_upper_bound
        is not None
    )
    assert (
        enclosure.analytic_regular_materialized_rational_radius_upper_bound
        >= enclosure.materialized_rational_radius_upper_bound
    )
    assert (
        enclosure.analytic_regular_materialized_float_radius_upper_bound
        is not None
    )
    assert (
        enclosure.analytic_regular_materialized_float_radius_upper_bound
        < abs(float(enclosure.endpoint_curvature_center))
    )
    assert enclosure.analytic_regular_curvature_component_interval is not None
    assert enclosure.analytic_regular_curvature_component_interval[0] > 0
    assert enclosure.analytic_regular_curvature_component_certified_sign == 1
    assert (
        enclosure.analytic_regular_curvature_prime_component_certified_sign
        is None
    )
    assert enclosure.exact_symbolic_analytic_regular_endpoint_ball_proven
    assert (
        enclosure
        .rigorous_materialized_analytic_regular_endpoint_enclosure_proven
    )
    assert enclosure.analytic_source_off_regular_initial_condition_enclosed
    assert enclosure.numerical_node_roundoff_absorbed_into_frozen_path
    assert enclosure.normalized_dimensionless_model_assumed
    assert not enclosure.physical_primordial_initial_condition_enclosed
    clock = enclosure.scalar_clock_endpoint
    assert clock.trace_curvature_interval == (
        enclosure.analytic_regular_curvature_component_interval
    )
    assert clock.trace_curvature_prime_interval == (
        enclosure.analytic_regular_curvature_prime_component_interval
    )
    assert (
        clock.negative_hubble_log_derivative_interval[0]
        >= Fraction(3, 2)
    )
    assert clock.negative_hubble_log_derivative_interval[1] <= 3
    assert clock.kappa_squared_interval[0] > 0
    assert clock.clock_reconstruction_coefficient_interval[0] >= 1
    assert float(clock.frozen_numeric_scalar_clock_center) == (
        numerical.refined_final_scalar_clock_shift
    )
    assert clock.scalar_clock_interval is not None
    assert (
        clock.scalar_clock_interval[0]
        <= clock.frozen_numeric_scalar_clock_center
        <= clock.scalar_clock_interval[1]
    )
    assert clock.scalar_clock_interval[0] > 0
    assert clock.scalar_clock_certified_sign == 1
    assert clock.analytic_regular_trace_endpoint_used
    assert clock.trace_to_clock_algebraic_inversion_proven
    assert clock.negative_hubble_separated_from_zero
    assert clock.exact_rational_outward_interval_operations_proven
    assert clock.normalized_dimensionless_reconstruction_proven
    assert clock.scalar_clock_endpoint_enclosed
    assert not clock.independent_scalar_clock_dynamical_integration_proven
    assert not clock.physical_canonical_clock_identification_proven
    assert not clock.primordial_spectrum_supplied
    assert not clock.observable_transfer_function_enclosed
    assert enclosure.scalar_clock_endpoint_enclosed
    response = enclosure.amplitude_normalized_response
    assert response.supplied_amplitude == Fraction.from_float(1.0e-5)
    assert response.amplitude_sign == 1
    assert response.normalization_defined
    assert response.curvature_response_interval is not None
    assert response.curvature_prime_response_interval is not None
    assert response.common_ledger_clock_response_interval is not None
    assert response.curvature_response_interval[0] > 0
    assert response.common_ledger_clock_response_interval[0] > 0
    assert response.curvature_response_certified_sign == 1
    assert response.curvature_prime_response_certified_sign is None
    assert response.common_ledger_clock_response_certified_sign == 1
    normalized_curvature_center = (
        enclosure.endpoint_curvature_center
        / response.supplied_amplitude
    )
    assert (
        response.curvature_response_interval[0]
        <= normalized_curvature_center
        <= response.curvature_response_interval[1]
    )
    normalized_clock_center = (
        clock.frozen_numeric_scalar_clock_center
        / response.supplied_amplitude
    )
    assert (
        response.common_ledger_clock_response_interval[0]
        <= normalized_clock_center
        <= response.common_ledger_clock_response_interval[1]
    )
    assert response.exact_regular_series_linearity_proven
    assert response.analytic_trace_ode_homogeneity_proven
    assert response.common_ledger_clock_reconstruction_linearity_proven
    assert response.exact_rational_signed_point_division_proven
    assert response.fixed_amplitude_conditional_response_enclosed
    assert not response.frozen_recomputed_path_scale_invariance_proven
    assert (
        not response
        .residual_and_initial_bound_uniform_abs_amplitude_scaling_proven
    )
    assert not response.physical_primordial_normalization_supplied
    assert not response.physical_observable_transfer_function_enclosed
    assert enclosure.conditional_amplitude_normalized_response_enclosed
    weyl = enclosure.conditional_weyl_metric_endpoint
    assert weyl.curvature_potential_interval == (
        enclosure.analytic_regular_curvature_component_interval
    )
    assert weyl.lapse_potential_interval == weyl.curvature_potential_interval
    assert weyl.weyl_average_potential_interval == (
        weyl.curvature_potential_interval
    )
    assert weyl.weyl_sum_metric_source_interval is not None
    assert (
        weyl.weyl_sum_metric_source_interval[0]
        <= 2 * weyl.weyl_average_potential_interval[0]
    )
    assert (
        weyl.weyl_sum_metric_source_interval[1]
        >= 2 * weyl.weyl_average_potential_interval[1]
    )
    assert weyl.normalized_weyl_average_response_interval == (
        response.curvature_response_interval
    )
    assert (
        weyl.normalized_weyl_sum_metric_response_interval
        is not None
    )
    assert (
        weyl.normalized_weyl_sum_metric_response_interval[0]
        <= 2 * response.curvature_response_interval[0]
    )
    assert (
        weyl.normalized_weyl_sum_metric_response_interval[1]
        >= 2 * response.curvature_response_interval[1]
    )
    assert weyl.deterministic_weyl_average_squared_gain_interval == (
        response.curvature_response_interval[0] ** 2,
        response.curvature_response_interval[1] ** 2,
    )
    assert weyl.deterministic_weyl_sum_squared_gain_interval is not None
    assert weyl.weyl_average_response_certified_sign == 1
    assert weyl.weyl_sum_response_certified_sign == 1
    assert weyl.newtonian_gauge_metric_convention_fixed
    assert weyl.zero_total_anisotropic_stress_adopted_effective_closure
    assert weyl.lapse_equals_curvature_in_conditional_branch_proven
    assert weyl.conditional_metric_potential_endpoint_enclosed
    assert weyl.conditional_amplitude_normalized_metric_response_enclosed
    assert not weyl.line_of_sight_lensing_observable_enclosed
    assert not weyl.einstein_boltzmann_solution_enclosed
    assert not weyl.primordial_power_spectrum_supplied
    assert not weyl.physical_power_transfer_function_enclosed
    assert not weyl.cmb_lss_likelihood_enclosed
    assert enclosure.conditional_weyl_metric_endpoint_enclosed
    tube = enclosure.uniform_trace_path_tube_and_efold_integral
    assert tube.n_initial == enclosure.n_initial
    assert tube.n_final == enclosure.n_final
    assert tube.interval_width == enclosure.n_final - enclosure.n_initial
    assert tube.refined_step_count == enclosure.refined_step_count
    assert tube.frozen_pl_curvature_efold_integral > 0
    assert (
        tube.analytic_regular_materialized_uniform_radius_upper_bound
        == enclosure
        .analytic_regular_materialized_rational_radius_upper_bound
    )
    assert (
        tube.analytic_curvature_efold_integral_radius_upper_bound
        == tube.interval_width
        * tube.analytic_regular_materialized_uniform_radius_upper_bound
    )
    assert tube.analytic_curvature_efold_integral_interval is not None
    assert tube.analytic_curvature_efold_integral_interval[0] > 0
    assert tube.conditional_weyl_average_efold_integral_interval == (
        tube.analytic_curvature_efold_integral_interval
    )
    assert tube.conditional_weyl_sum_efold_integral_interval is not None
    assert (
        tube.conditional_weyl_sum_efold_integral_interval[0]
        <= 2 * tube.analytic_curvature_efold_integral_interval[0]
    )
    assert (
        tube.conditional_weyl_sum_efold_integral_interval[1]
        >= 2 * tube.analytic_curvature_efold_integral_interval[1]
    )
    assert tube.normalized_curvature_efold_response_interval is not None
    assert tube.normalized_curvature_efold_response_interval[0] > 0
    assert tube.normalized_weyl_sum_efold_response_interval is not None
    assert tube.normalized_curvature_efold_response_certified_sign == 1
    assert tube.normalized_weyl_sum_efold_response_certified_sign == 1
    assert tube.continuous_piecewise_linear_path_integrated_exactly
    assert tube.nonnegative_prefix_residual_budget_bounded_by_total
    assert tube.nonnegative_prefix_logarithmic_norm_bounded_by_total
    assert tube.uniform_trace_state_tube_covers_every_prefix
    assert tube.exact_symbolic_uniform_path_tube_proven
    assert tube.materialized_analytic_regular_uniform_path_tube_proven
    assert tube.unweighted_efold_metric_integral_enclosed
    assert not tube.prefix_sharp_radius_proven
    assert not tube.conformal_or_comoving_line_of_sight_integral_enclosed
    assert not tube.photon_geodesic_lensing_observable_enclosed
    assert not tube.integrated_sachs_wolfe_observable_enclosed
    assert not tube.primordial_power_spectrum_supplied
    assert enclosure.analytic_regular_uniform_trace_path_tube_proven
    conformal = enclosure.background_conformal_metric_time_integral
    assert conformal.n_initial == enclosure.n_initial
    assert conformal.n_final == enclosure.n_final
    assert conformal.refined_step_count == enclosure.refined_step_count
    assert conformal.conformal_weight_interval_hull[0] > 0
    assert (
        0
        < conformal.dimensionless_background_conformal_time_interval[0]
        <= conformal.dimensionless_background_conformal_time_interval[1]
    )
    frozen_conformal = (
        conformal
        .frozen_pl_weyl_average_conformal_time_integral_interval
    )
    assert frozen_conformal[0] > 0
    assert (
        conformal
        .analytic_regular_materialized_weyl_average_integral_radius_upper_bound
        == enclosure.analytic_regular_materialized_rational_radius_upper_bound
        * conformal.dimensionless_background_conformal_time_interval[1]
    )
    analytic_conformal = (
        conformal
        .analytic_regular_weyl_average_conformal_time_integral_interval
    )
    assert analytic_conformal is not None
    assert analytic_conformal[0] > 0
    normalized_conformal = (
        conformal
        .normalized_weyl_average_conformal_time_response_interval
    )
    assert normalized_conformal is not None
    assert normalized_conformal[0] > 0
    assert conformal.normalized_weyl_average_response_certified_sign == 1
    assert conformal.normalized_weyl_sum_response_certified_sign == 1
    assert conformal.flat_gr_radial_null_measure_identity_proven
    assert conformal.exact_rational_inverse_square_root_enclosures_proven
    assert conformal.positive_conformal_weight_on_every_mesh_cell
    assert conformal.cellwise_interval_weighted_pl_metric_integral_enclosed
    assert conformal.uniform_trace_tube_integrated_against_positive_measure
    assert conformal.materialized_analytic_regular_metric_time_integral_enclosed
    assert conformal.unperturbed_flat_background_radial_null_measure_used
    assert not conformal.physical_density_scale_calibration_supplied
    assert not conformal.spatial_mode_phase_on_null_path_supplied
    assert not conformal.lensing_source_distance_and_kernel_supplied
    assert not conformal.transverse_laplacian_or_angular_mode_supplied
    assert not conformal.photon_geodesic_lensing_observable_enclosed
    assert not conformal.integrated_sachs_wolfe_observable_enclosed
    assert not conformal.all_k_einstein_boltzmann_solution_enclosed
    assert not conformal.primordial_power_spectrum_supplied
    assert not conformal.cmb_lss_likelihood_enclosed
    assert enclosure.background_conformal_metric_time_integral_enclosed
    born = enclosure.fixed_mode_born_lensing_absolute_envelope
    assert born.n_source == enclosure.n_initial
    assert born.n_observer == enclosure.n_final
    assert born.refined_step_count == enclosure.refined_step_count
    assert len(born.dimensionless_conformal_cell_measure_intervals) == (
        enclosure.refined_step_count
    )
    assert len(
        born.dimensionless_source_side_distance_node_intervals
    ) == enclosure.refined_step_count + 1
    assert len(
        born.dimensionless_observer_side_distance_node_intervals
    ) == enclosure.refined_step_count + 1
    assert all(
        interval[0] > 0
        for interval in born.dimensionless_conformal_cell_measure_intervals
    )
    source_side_distances = (
        born.dimensionless_source_side_distance_node_intervals
    )
    observer_side_distances = (
        born.dimensionless_observer_side_distance_node_intervals
    )
    assert all(
        left[1] < right[1]
        for left, right in zip(
            source_side_distances,
            source_side_distances[1:],
        )
    )
    assert all(
        left[1] > right[1]
        for left, right in zip(
            observer_side_distances,
            observer_side_distances[1:],
        )
    )
    assert born.dimensionless_source_side_distance_node_intervals[0] == (
        Fraction(0),
        Fraction(0),
    )
    assert born.dimensionless_observer_side_distance_node_intervals[-1] == (
        Fraction(0),
        Fraction(0),
    )
    source_distance = born.dimensionless_source_distance_interval
    assert 0 < source_distance[0] <= source_distance[1]
    assert (
        conformal.dimensionless_background_conformal_time_interval[0]
        <= source_distance[0]
        <= source_distance[1]
        <= conformal.dimensionless_background_conformal_time_interval[1]
    )
    q_squared = born.dimensionless_fixed_wavenumber_squared_interval
    initial_density_reference = (
        evolution.bridge.production_density(evolution.n_initial)
        + evolution.bridge.reservoir_density(evolution.n_initial)
    )
    q_squared_reference = (
        evolution.kappa_initial**2
        * math.exp(2 * evolution.n_initial)
        * initial_density_reference
    )
    assert float(q_squared[0]) <= q_squared_reference <= float(q_squared[1])
    assert q_squared[0] > 0
    assert (
        0
        < born.dimensionless_geometric_kernel_upper_bound
        <= source_distance[1] / 4
    )
    for index, kernel_upper in enumerate(
        born.dimensionless_geometric_kernel_cell_upper_bounds
    ):
        independent_distance_upper = (
            source_side_distances[index + 1][1]
            * observer_side_distances[index][1]
            / source_distance[0]
        )
        assert kernel_upper == min(
            independent_distance_upper,
            source_distance[1] / 4,
        )
    assert born.frozen_pl_born_convergence_absolute_upper_bound > 0
    assert (
        born.analytic_regular_born_convergence_absolute_upper_bound
        is not None
    )
    assert (
        born.analytic_regular_born_convergence_absolute_upper_bound
        > born.frozen_pl_born_convergence_absolute_upper_bound
    )
    assert (
        born.normalized_analytic_regular_born_convergence_absolute_upper_bound
        == born.analytic_regular_born_convergence_absolute_upper_bound
        / abs(born.primordial_potential_amplitude)
    )
    assert born.single_mode_convergence_bound_strictly_below_unity
    assert born.source_and_observer_planes_fixed_at_interval_endpoints
    assert born.flat_background_born_weak_lensing_equation_adopted
    assert born.newtonian_gauge_zero_anisotropic_stress_adopted
    assert born.single_fixed_fourier_mode_adopted
    assert born.exact_rational_dimensionless_fixed_wavenumber_enclosed
    assert born.positive_conformal_cell_measure_enclosed
    assert born.prefix_and_suffix_distances_accumulated_independently
    assert born.source_distance_identity_enclosed_by_intersection
    assert born.nonnegative_flat_lensing_kernel_enclosed_cellwise
    assert born.transverse_wavenumber_bounded_by_total_wavenumber
    assert born.spatial_fourier_phase_modulus_bounded_by_one
    assert born.uniform_analytic_trace_tube_used
    assert born.conditional_single_mode_born_convergence_absolute_envelope_enclosed
    assert not born.signed_single_mode_convergence_enclosed
    assert not born.transverse_mode_orientation_supplied
    assert not born.spatial_mode_phase_on_null_path_supplied
    assert not born.source_redshift_calibration_supplied
    assert not born.source_population_distribution_supplied
    assert not born.born_weak_field_validity_independently_derived
    assert not born.perturbed_or_post_born_geodesic_enclosed
    assert not born.all_k_einstein_boltzmann_solution_enclosed
    assert not born.primordial_power_spectrum_supplied
    assert not born.shear_or_lensing_map_enclosed
    assert not born.angular_power_spectrum_enclosed
    assert not born.cmb_lss_likelihood_enclosed
    assert enclosure.conditional_fixed_mode_born_lensing_absolute_envelope_enclosed
    assert not enclosure.numerical_method_convergence_theorem_proven
    assert not enclosure.observable_transfer_function_enclosed


def test_uniform_path_tube_rejects_a_nonmonotone_mesh() -> None:
    zero_radius = endpoint_module.ExactExponentialRadius(
        coefficient=Fraction(0),
        exponent=Fraction(0),
        coefficient_nonnegative=True,
        exponent_nonnegative=True,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        FiniteQuenchTraceEndpointEnclosure._uniform_trace_path_tube_receipt(
            frozen_mesh=(
                Fraction(0),
                Fraction(2),
                Fraction(1),
                Fraction(3),
            ),
            frozen_nodes=((Fraction(0), Fraction(0)),) * 4,
            frozen_symbolic_radius=zero_radius,
            analytic_symbolic_radius=zero_radius,
            frozen_materialized_radius=Fraction(0),
            analytic_materialized_radius=Fraction(0),
            amplitude=Fraction(1),
        )
    owner = FiniteQuenchTraceEndpointEnclosure(_evolution())
    with pytest.raises(ValueError, match="strictly increasing"):
        owner._background_conformal_metric_time_integral_receipt(
            frozen_mesh=(
                Fraction(0),
                Fraction(2),
                Fraction(1),
                Fraction(3),
            ),
            frozen_nodes=((Fraction(0), Fraction(0)),) * 4,
            parameters=owner._frozen_parameters(),
            analytic_symbolic_radius=zero_radius,
            analytic_materialized_radius=Fraction(0),
            amplitude=Fraction(1),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        owner._fixed_mode_born_lensing_absolute_envelope_receipt(
            frozen_mesh=(
                Fraction(0),
                Fraction(2),
                Fraction(1),
                Fraction(3),
            ),
            frozen_nodes=((Fraction(0), Fraction(0)),) * 4,
            parameters=owner._frozen_parameters(),
            analytic_materialized_radius=Fraction(0),
            amplitude=Fraction(1),
        )


def test_zero_path_has_exact_zero_endpoint_radius() -> None:
    enclosure = FiniteQuenchTraceEndpointEnclosure(_evolution()).construct(
        primordial_potential_amplitude=0.0,
        coarse_step_count=64,
    )
    assert enclosure.residual_l1_integral_upper_bound == 0
    assert enclosure.materialized_rational_radius_upper_bound == 0
    assert enclosure.materialized_float_radius_upper_bound == 0.0
    assert enclosure.curvature_component_interval == (
        enclosure.endpoint_curvature_center,
        enclosure.endpoint_curvature_center,
    )
    assert enclosure.curvature_prime_component_interval == (
        enclosure.endpoint_curvature_prime_center,
        enclosure.endpoint_curvature_prime_center,
    )
    assert enclosure.curvature_component_certified_sign is None
    assert enclosure.curvature_prime_component_certified_sign is None
    assert (
        enclosure.regular_initial_bridge.initial_l1_defect_upper_bound
        == 0
    )
    assert (
        enclosure.analytic_regular_materialized_rational_radius_upper_bound
        == 0
    )
    assert enclosure.analytic_regular_curvature_component_interval == (
        enclosure.endpoint_curvature_center,
        enclosure.endpoint_curvature_center,
    )
    assert enclosure.analytic_regular_curvature_component_certified_sign is None
    assert enclosure.scalar_clock_endpoint.scalar_clock_interval == (
        Fraction(0),
        Fraction(0),
    )
    assert enclosure.scalar_clock_endpoint.scalar_clock_certified_sign is None
    assert enclosure.scalar_clock_endpoint_enclosed
    zero_tube = enclosure.uniform_trace_path_tube_and_efold_integral
    assert zero_tube.frozen_pl_curvature_efold_integral == 0
    assert zero_tube.analytic_curvature_efold_integral_interval == (
        Fraction(0),
        Fraction(0),
    )
    assert zero_tube.normalized_curvature_efold_response_interval is None
    assert zero_tube.uniform_trace_state_tube_covers_every_prefix
    assert zero_tube.unweighted_efold_metric_integral_enclosed
    zero_conformal = enclosure.background_conformal_metric_time_integral
    assert zero_conformal.dimensionless_background_conformal_time_interval[0] > 0
    assert (
        zero_conformal
        .frozen_pl_weyl_average_conformal_time_integral_interval
        == (Fraction(0), Fraction(0))
    )
    assert (
        zero_conformal
        .analytic_regular_materialized_weyl_average_integral_radius_upper_bound
        == 0
    )
    assert (
        zero_conformal
        .analytic_regular_weyl_average_conformal_time_integral_interval
        == (Fraction(0), Fraction(0))
    )
    assert (
        zero_conformal
        .normalized_weyl_average_conformal_time_response_interval
        is None
    )
    assert not zero_conformal.normalization_defined
    assert zero_conformal.materialized_analytic_regular_metric_time_integral_enclosed
    assert enclosure.background_conformal_metric_time_integral_enclosed
    zero_born = enclosure.fixed_mode_born_lensing_absolute_envelope
    assert zero_born.frozen_pl_born_convergence_absolute_upper_bound == 0
    assert (
        zero_born.analytic_regular_born_convergence_absolute_upper_bound
        == 0
    )
    assert (
        zero_born
        .normalized_analytic_regular_born_convergence_absolute_upper_bound
        is None
    )
    assert zero_born.single_mode_convergence_bound_strictly_below_unity
    assert zero_born.conditional_single_mode_born_convergence_absolute_envelope_enclosed
    assert enclosure.conditional_fixed_mode_born_lensing_absolute_envelope_enclosed
    zero_response = enclosure.amplitude_normalized_response
    assert zero_response.amplitude_sign == 0
    assert not zero_response.normalization_defined
    assert zero_response.curvature_response_interval is None
    assert zero_response.curvature_prime_response_interval is None
    assert zero_response.common_ledger_clock_response_interval is None
    assert not zero_response.exact_rational_signed_point_division_proven
    assert not zero_response.fixed_amplitude_conditional_response_enclosed
    assert not enclosure.conditional_amplitude_normalized_response_enclosed
    zero_weyl = enclosure.conditional_weyl_metric_endpoint
    assert zero_weyl.weyl_average_potential_interval == (
        Fraction(0),
        Fraction(0),
    )
    assert zero_weyl.weyl_sum_metric_source_interval == (
        Fraction(0),
        Fraction(0),
    )
    assert zero_weyl.normalized_weyl_average_response_interval is None
    assert zero_weyl.deterministic_weyl_average_squared_gain_interval is None
    assert zero_weyl.conditional_metric_potential_endpoint_enclosed
    assert (
        not zero_weyl
        .conditional_amplitude_normalized_metric_response_enclosed
    )
    assert enclosure.conditional_weyl_metric_endpoint_enclosed


@pytest.mark.parametrize(
    ("w", "half_width"),
    [(0.0, 0.5), (0.1, 1.0e-4)],
)
def test_edge_branches_keep_a_nonvacuous_positive_curvature_ball(
    w: float,
    half_width: float,
) -> None:
    enclosure = FiniteQuenchTraceEndpointEnclosure(
        _evolution(w=w, half_width=half_width)
    ).construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
    )
    assert enclosure.materialized_float_radius_upper_bound is not None
    assert (
        enclosure.materialized_float_radius_upper_bound
        < abs(float(enclosure.endpoint_curvature_center))
    )
    assert enclosure.curvature_component_certified_sign == 1
    assert (
        enclosure.analytic_regular_materialized_float_radius_upper_bound
        is not None
    )
    assert (
        enclosure.analytic_regular_materialized_float_radius_upper_bound
        < abs(float(enclosure.endpoint_curvature_center))
    )
    assert enclosure.analytic_regular_curvature_component_certified_sign == 1
    assert enclosure.scalar_clock_endpoint.scalar_clock_interval is not None
    assert (
        enclosure.scalar_clock_endpoint.scalar_clock_interval[0]
        <= enclosure.scalar_clock_endpoint.frozen_numeric_scalar_clock_center
        <= enclosure.scalar_clock_endpoint.scalar_clock_interval[1]
    )
    assert enclosure.scalar_clock_endpoint.scalar_clock_interval[0] > 0
    assert enclosure.scalar_clock_endpoint.scalar_clock_certified_sign == 1
    assert enclosure.scalar_clock_endpoint_enclosed
    edge_response = enclosure.amplitude_normalized_response
    assert edge_response.fixed_amplitude_conditional_response_enclosed
    assert edge_response.curvature_response_certified_sign == 1
    assert edge_response.common_ledger_clock_response_certified_sign == 1
    edge_weyl = enclosure.conditional_weyl_metric_endpoint
    assert edge_weyl.weyl_average_response_certified_sign == 1
    assert edge_weyl.weyl_sum_response_certified_sign == 1
    edge_tube = enclosure.uniform_trace_path_tube_and_efold_integral
    assert edge_tube.analytic_curvature_efold_integral_interval is not None
    assert edge_tube.analytic_curvature_efold_integral_interval[0] > 0
    assert edge_tube.normalized_curvature_efold_response_certified_sign == 1
    assert (
        enclosure.local_interval_residual_l1_integral_upper_bound
        < enclosure.global_triangle_residual_l1_integral_upper_bound
    )


def test_causal_w_one_requires_symbolic_radius_for_global_bound() -> None:
    bounds = FiniteQuenchTraceEndpointEnclosure(
        _evolution(w=1.0)
    ).coefficient_bounds()
    interval = Fraction(5)
    assert (
        bounds.euclidean_logarithmic_norm_rate_upper_bound * interval
        > 32
    )


def test_symbolic_radius_branch_never_emits_component_signs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        endpoint_module,
        "_MAX_MATERIALIZED_RADIUS_EXP_ARGUMENT",
        Fraction(0),
    )
    enclosure = FiniteQuenchTraceEndpointEnclosure(_evolution()).construct(
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=64,
    )

    assert enclosure.exact_symbolic_trace_endpoint_ball_proven
    assert enclosure.materialized_exponential_argument_upper_bound is None
    assert enclosure.materialized_rational_radius_upper_bound is None
    assert enclosure.materialized_float_radius_upper_bound is None
    assert enclosure.curvature_component_interval is None
    assert enclosure.curvature_prime_component_interval is None
    assert enclosure.curvature_component_certified_sign is None
    assert enclosure.curvature_prime_component_certified_sign is None
    assert not enclosure.rigorous_materialized_trace_endpoint_enclosure_proven
    assert (
        enclosure
        .analytic_regular_materialized_exponential_argument_upper_bound
        is None
    )
    assert (
        enclosure.analytic_regular_materialized_rational_radius_upper_bound
        is None
    )
    assert enclosure.analytic_regular_curvature_component_interval is None
    assert (
        enclosure.analytic_regular_curvature_prime_component_interval
        is None
    )
    assert (
        enclosure.analytic_regular_curvature_component_certified_sign
        is None
    )
    assert (
        enclosure
        .analytic_regular_curvature_prime_component_certified_sign
        is None
    )
    assert enclosure.exact_symbolic_analytic_regular_endpoint_ball_proven
    assert (
        not enclosure
        .rigorous_materialized_analytic_regular_endpoint_enclosure_proven
    )
    assert enclosure.scalar_clock_endpoint.scalar_clock_interval is None
    assert enclosure.scalar_clock_endpoint.scalar_clock_certified_sign is None
    assert not enclosure.scalar_clock_endpoint.scalar_clock_endpoint_enclosed
    assert not enclosure.scalar_clock_endpoint_enclosed
    symbolic_response = enclosure.amplitude_normalized_response
    assert symbolic_response.normalization_defined
    assert symbolic_response.curvature_response_interval is None
    assert symbolic_response.curvature_prime_response_interval is None
    assert symbolic_response.common_ledger_clock_response_interval is None
    assert not symbolic_response.fixed_amplitude_conditional_response_enclosed
    assert not enclosure.conditional_amplitude_normalized_response_enclosed
    symbolic_weyl = enclosure.conditional_weyl_metric_endpoint
    assert symbolic_weyl.weyl_average_potential_interval is None
    assert symbolic_weyl.weyl_sum_metric_source_interval is None
    assert symbolic_weyl.normalized_weyl_average_response_interval is None
    assert symbolic_weyl.deterministic_weyl_average_squared_gain_interval is None
    assert not symbolic_weyl.conditional_metric_potential_endpoint_enclosed
    assert not enclosure.conditional_weyl_metric_endpoint_enclosed
    symbolic_tube = enclosure.uniform_trace_path_tube_and_efold_integral
    assert symbolic_tube.exact_symbolic_uniform_path_tube_proven
    assert symbolic_tube.uniform_trace_state_tube_covers_every_prefix
    assert (
        symbolic_tube
        .analytic_regular_materialized_uniform_radius_upper_bound
        is None
    )
    assert symbolic_tube.analytic_curvature_efold_integral_interval is None
    assert symbolic_tube.normalized_curvature_efold_response_interval is None
    assert (
        not symbolic_tube
        .materialized_analytic_regular_uniform_path_tube_proven
    )
    assert not symbolic_tube.unweighted_efold_metric_integral_enclosed
    assert enclosure.analytic_regular_uniform_trace_path_tube_proven
    symbolic_conformal = enclosure.background_conformal_metric_time_integral
    assert symbolic_conformal.dimensionless_background_conformal_time_interval[0] > 0
    assert (
        symbolic_conformal
        .analytic_regular_symbolic_weyl_average_integral_radius
        .coefficient
        > 0
    )
    assert (
        symbolic_conformal
        .analytic_regular_materialized_weyl_average_integral_radius_upper_bound
        is None
    )
    assert (
        symbolic_conformal
        .analytic_regular_weyl_average_conformal_time_integral_interval
        is None
    )
    assert (
        symbolic_conformal
        .normalized_weyl_average_conformal_time_response_interval
        is None
    )
    assert not symbolic_conformal.materialized_analytic_regular_metric_time_integral_enclosed
    assert not enclosure.background_conformal_metric_time_integral_enclosed
    symbolic_born = enclosure.fixed_mode_born_lensing_absolute_envelope
    assert symbolic_born.frozen_pl_born_convergence_absolute_upper_bound > 0
    assert (
        symbolic_born.analytic_regular_born_convergence_absolute_upper_bound
        is None
    )
    assert (
        symbolic_born
        .normalized_analytic_regular_born_convergence_absolute_upper_bound
        is None
    )
    assert (
        symbolic_born.single_mode_convergence_bound_strictly_below_unity
        is None
    )
    assert (
        not symbolic_born
        .conditional_single_mode_born_convergence_absolute_envelope_enclosed
    )
    assert (
        not enclosure
        .conditional_fixed_mode_born_lensing_absolute_envelope_enclosed
    )


def test_negative_curvature_path_has_certified_negative_sign() -> None:
    enclosure = FiniteQuenchTraceEndpointEnclosure(_evolution()).construct(
        primordial_potential_amplitude=-1.0e-5,
        coarse_step_count=512,
    )

    assert enclosure.curvature_component_interval is not None
    assert enclosure.curvature_component_interval[1] < 0
    assert enclosure.curvature_component_certified_sign == -1
    assert enclosure.analytic_regular_curvature_component_interval is not None
    assert enclosure.analytic_regular_curvature_component_interval[1] < 0
    assert enclosure.analytic_regular_curvature_component_certified_sign == -1
    assert enclosure.scalar_clock_endpoint.scalar_clock_interval is not None
    assert (
        enclosure.scalar_clock_endpoint.scalar_clock_interval[0]
        <= enclosure.scalar_clock_endpoint.frozen_numeric_scalar_clock_center
        <= enclosure.scalar_clock_endpoint.scalar_clock_interval[1]
    )
    assert enclosure.scalar_clock_endpoint.scalar_clock_interval[1] < 0
    assert enclosure.scalar_clock_endpoint.scalar_clock_certified_sign == -1
    negative_response = enclosure.amplitude_normalized_response
    assert negative_response.amplitude_sign == -1
    assert negative_response.normalization_defined
    assert negative_response.curvature_response_interval is not None
    assert negative_response.common_ledger_clock_response_interval is not None
    assert negative_response.curvature_response_interval[0] > 0
    assert negative_response.common_ledger_clock_response_interval[0] > 0
    assert negative_response.curvature_response_certified_sign == 1
    assert negative_response.common_ledger_clock_response_certified_sign == 1
    assert negative_response.fixed_amplitude_conditional_response_enclosed
    negative_weyl = enclosure.conditional_weyl_metric_endpoint
    assert negative_weyl.normalized_weyl_average_response_interval is not None
    assert (
        negative_weyl.normalized_weyl_sum_metric_response_interval
        is not None
    )
    assert negative_weyl.weyl_average_response_certified_sign == 1
    assert negative_weyl.weyl_sum_response_certified_sign == 1
    negative_tube = enclosure.uniform_trace_path_tube_and_efold_integral
    assert negative_tube.normalized_curvature_efold_response_interval is not None
    assert negative_tube.normalized_curvature_efold_response_interval[0] > 0
    assert negative_tube.normalized_curvature_efold_response_certified_sign == 1
    negative_conformal = enclosure.background_conformal_metric_time_integral
    assert (
        negative_conformal
        .analytic_regular_weyl_average_conformal_time_integral_interval
        is not None
    )
    assert (
        negative_conformal
        .analytic_regular_weyl_average_conformal_time_integral_interval[1]
        < 0
    )
    assert (
        negative_conformal
        .normalized_weyl_average_conformal_time_response_interval
        is not None
    )
    assert (
        negative_conformal
        .normalized_weyl_average_conformal_time_response_interval[0]
        > 0
    )
    assert negative_conformal.normalized_weyl_average_response_certified_sign == 1
    assert negative_conformal.normalized_weyl_sum_response_certified_sign == 1
    negative_born = enclosure.fixed_mode_born_lensing_absolute_envelope
    assert (
        negative_born.analytic_regular_born_convergence_absolute_upper_bound
        is not None
    )
    assert (
        negative_born.analytic_regular_born_convergence_absolute_upper_bound
        > 0
    )
    assert (
        negative_born
        .normalized_analytic_regular_born_convergence_absolute_upper_bound
        is not None
    )
    assert (
        negative_born
        .normalized_analytic_regular_born_convergence_absolute_upper_bound
        > 0
    )
    assert not negative_born.signed_single_mode_convergence_enclosed


def test_rational_enclosure_requires_positive_present_reservoir() -> None:
    enclosure = FiniteQuenchTraceEndpointEnclosure(
        _evolution(reservoir_present_density=0.0)
    )
    with pytest.raises(ValueError, match="positive present reservoir"):
        enclosure.coefficient_bounds()
