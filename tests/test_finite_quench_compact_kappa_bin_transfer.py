"""Focused tests for the compact initial-kappa Duhamel transfer tube."""

from __future__ import annotations

from fractions import Fraction

import pytest

from examples.physics.finite_quench_compact_kappa_bin_transfer import (
    _initial_normalized_state_and_sensitivity_bounds,
    construct_compact_kappa_bin_weyl_transfer_enclosure,
)
from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.finite_quench_trace_endpoint_enclosure import (
    FiniteQuenchTraceEndpointEnclosure,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge() -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=0.12,
            reservoir_present_density=0.21,
            w_reservoir=0.1,
            w_open=2.1767e-4,
        )
    )


@pytest.fixture(scope="module")
def compact_receipt():
    evolution = FiniteQuenchRegularMetricEvolution(
        _bridge(),
        n_initial=-5.0,
        kappa_initial=0.05,
    )
    return construct_compact_kappa_bin_weyl_transfer_enclosure(
        FiniteQuenchTraceEndpointEnclosure(evolution),
        initial_kappa_lower=Fraction(49, 1000),
        initial_kappa_upper=Fraction(51, 1000),
        primordial_potential_amplitude=1.0e-5,
        coarse_step_count=512,
    )


def test_compact_bin_cellwise_sensitivity_receipt_is_aligned(
    compact_receipt,
) -> None:
    receipt = compact_receipt
    cell_count = len(receipt.refined_mesh) - 1

    assert receipt.initial_kappa_interval == (
        Fraction(49, 1000),
        Fraction(51, 1000),
    )
    assert receipt.initial_kappa_squared_interval == (
        Fraction(2401, 1_000_000),
        Fraction(2601, 1_000_000),
    )
    assert all(ratio < 1 for ratio in receipt.initial_series_ratio_upper_bounds)
    assert len(receipt.kappa_growth_factor_cell_intervals) == cell_count
    assert len(receipt.parameter_generator_lipschitz_cell_upper_bounds) == cell_count
    assert len(receipt.cell_exponential_propagator_upper_bounds) == cell_count
    assert len(receipt.normalized_state_node_upper_bounds) == cell_count + 1
    assert len(receipt.normalized_parameter_derivative_node_upper_bounds) == (
        cell_count + 1
    )
    assert all(
        left <= right
        for left, right in zip(
            receipt.normalized_parameter_derivative_node_upper_bounds,
            receipt.normalized_parameter_derivative_node_upper_bounds[1:],
        )
    )
    assert all(value >= 1 for value in receipt.cell_exponential_propagator_upper_bounds)


def test_compact_bin_adds_finite_parameter_radius_to_central_tube(
    compact_receipt,
) -> None:
    receipt = compact_receipt
    central = receipt.central_analytic_regular_uniform_tube_radius_upper_bound
    totals = receipt.compact_bin_total_curvature_node_radius_upper_bounds

    assert central is not None
    assert totals is not None
    assert receipt.parameter_variation_curvature_node_radius_upper_bounds[-1] > 0
    assert receipt.parameter_variation_curvature_node_radius_upper_bounds[-1] < Fraction(1, 10_000_000)
    assert totals[-1] > central
    assert receipt.compact_bin_weyl_average_cell_intervals is not None
    assert receipt.normalized_compact_bin_weyl_average_cell_intervals is not None
    assert receipt.compact_kappa_bin_uniform_weyl_path_tube_enclosed


def test_low_center_high_numeric_paths_lie_inside_final_bin_cell(
    compact_receipt,
) -> None:
    receipt = compact_receipt
    mesh = tuple(float(value) for value in receipt.refined_mesh)
    amplitude = float(receipt.primordial_potential_amplitude)
    cell_intervals = (
        receipt.normalized_compact_bin_weyl_average_cell_intervals
    )
    endpoint_values = {}

    for kappa in (0.049, 0.05, 0.051):
        evolution = FiniteQuenchRegularMetricEvolution(
            _bridge(),
            n_initial=-5.0,
            kappa_initial=kappa,
        )
        path = FiniteQuenchTraceEndpointEnclosure(
            evolution
        )._trace_nodes(
            primordial_potential_amplitude=amplitude,
            mesh=mesh,
        )
        normalized_path = tuple(
            Fraction.from_float(curvature)
            / receipt.primordial_potential_amplitude
            for curvature, _ in path
        )
        for index, interval in enumerate(cell_intervals):
            assert interval[0] <= normalized_path[index] <= interval[1]
            assert interval[0] <= normalized_path[index + 1] <= interval[1]
        endpoint_values[kappa] = normalized_path[-1]

    u_low = Fraction.from_float(0.049) ** 2
    u_high = Fraction.from_float(0.051) ** 2
    endpoint_secant = abs(
        endpoint_values[0.051] - endpoint_values[0.049]
    ) / (u_high - u_low)
    assert endpoint_secant <= (
        receipt.normalized_parameter_derivative_node_upper_bounds[-1]
    )


def test_compact_bin_locks_proof_status_and_nonclaims(compact_receipt) -> None:
    receipt = compact_receipt

    assert receipt.initial_bin_is_superhorizon
    assert receipt.trace_generator_affine_in_initial_kappa_squared_proven
    assert (
        receipt
        .exact_regular_series_parameter_derivative_conservatively_enclosed
    )
    assert receipt.cellwise_parameter_generator_lipschitz_enclosed
    assert receipt.cellwise_duhamel_sensitivity_recurrence_proven
    assert receipt.central_trace_tube_reused
    assert receipt.central_trace_radius_is_global_uniform_not_prefix_sharp
    assert receipt.zero_anisotropic_stress_weyl_average_equals_curvature_adopted
    assert not receipt.physical_wavenumber_bin_calibrated
    assert not receipt.primordial_curvature_normalization_supplied
    assert not receipt.spherical_bessel_harmonic_projection_enclosed
    assert not receipt.all_k_einstein_boltzmann_transfer_enclosed
    assert not receipt.angular_power_spectrum_enclosed
    assert not receipt.covariance_or_likelihood_enclosed


def test_dust_limit_has_zero_initial_kappa_parameter_sensitivity() -> None:
    state, sensitivity, ratios = (
        _initial_normalized_state_and_sensitivity_bounds(
            reservoir_equation_of_state=Fraction(0),
            initial_kappa_squared_upper_bound=Fraction(1, 100),
        )
    )

    assert state == 1
    assert sensitivity == 0
    assert ratios == (0, 0, 0)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (0, Fraction(51, 1000)),
        (Fraction(51, 1000), Fraction(49, 1000)),
        (Fraction(49, 1000), Fraction(11, 100)),
        (Fraction(3, 50), Fraction(7, 100)),
    ],
)
def test_invalid_or_non_superhorizon_bins_fail_before_construction(
    lower,
    upper,
) -> None:
    evolution = FiniteQuenchRegularMetricEvolution(
        _bridge(),
        n_initial=-5.0,
        kappa_initial=0.05,
    )
    with pytest.raises(ValueError, match="compact bin requires"):
        construct_compact_kappa_bin_weyl_transfer_enclosure(
            FiniteQuenchTraceEndpointEnclosure(evolution),
            initial_kappa_lower=lower,
            initial_kappa_upper=upper,
            primordial_potential_amplitude=1.0e-5,
            coarse_step_count=64,
        )
