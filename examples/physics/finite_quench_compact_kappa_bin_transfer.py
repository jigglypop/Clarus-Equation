"""Parameter-uniform CE trace tube on one compact initial-kappa bin.

The trace generator depends on ``u = kappa_i**2`` only through

    B(n; u) = 3 + 2 h(n) + p(n) * (1 + u E(n) / 3),

so ``||partial_u M||_2 <= p_+ E_+ / 3``.  This module propagates an
exact-rational bound for both the state norm and its u-derivative on every
refined cell.  Adding ``delta_u * ||partial_u z||`` to the existing central
trace tube encloses every mode in the supplied compact bin.

This is a conditional deterministic Weyl-response tube per free potential
amplitude.  It is not yet a primordial-curvature transfer, spherical-Bessel
projection, angular power spectrum, or likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral

from examples.physics.finite_quench_trace_endpoint_enclosure import (
    FiniteQuenchTraceEndpointEnclosure,
    TraceEndpointEnclosureReceipt,
    _FrozenTraceParameters,
    _RationalInterval,
    _binary_fraction,
    _dyadic_upper_fraction,
    _interval_add,
    _interval_divide,
    _interval_multiply,
    _interval_scale,
    _monotone_exp_range,
    _outward_dyadic,
    _point_interval,
    _produced_density_interval,
    _rational_exp_interval,
    _reservoir_point_interval,
    _source_density_interval,
    _total_density_interval,
    _total_density_point_interval,
)


_SENSITIVITY_DYADIC_BITS = 160


def _finite_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar, not bool")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(value)
    if isinstance(value, (Fraction, Integral)):
        return Fraction(value)
    raise ValueError(f"{name} must be an int, Fraction, or finite float")


@dataclass(frozen=True)
class _KappaFamilyCellBounds:
    kappa_growth_factor_interval: tuple[Fraction, Fraction]
    pressure_ratio_interval: tuple[Fraction, Fraction]
    damping_interval: tuple[Fraction, Fraction]
    restoring_interval: tuple[Fraction, Fraction]
    parameter_generator_lipschitz_upper_bound: Fraction
    euclidean_logarithmic_norm_rate_upper_bound: Fraction


def _local_kappa_family_cell_bounds(
    *,
    left: Fraction,
    right: Fraction,
    parameters: _FrozenTraceParameters,
    initial_kappa_squared_interval: _RationalInterval,
) -> _KappaFamilyCellBounds:
    """Enclose M(n; u), E(n), and ||partial_u M|| on one cell."""

    produced = _produced_density_interval(left, right, parameters)
    reservoir_left = _reservoir_point_interval(left, parameters)
    reservoir_right = _reservoir_point_interval(right, parameters)
    reservoir = _RationalInterval(
        reservoir_right.lower,
        reservoir_left.upper,
    )
    source = _source_density_interval(left, right, parameters)
    total_density = _total_density_interval(left, right, parameters)
    enthalpy = _interval_add(
        produced,
        _interval_scale(reservoir, 1 + parameters.w),
    )
    if enthalpy.lower <= 0:
        raise ValueError("kappa-family enthalpy interval lost positivity")

    hubble_log_derivative = _interval_scale(
        _interval_divide(enthalpy, total_density),
        Fraction(-3, 2),
    )
    pressure_numerator = _interval_add(
        _interval_scale(reservoir, 3 * (1 + parameters.w)),
        source,
    )
    pressure_ratio = _interval_scale(
        _interval_divide(pressure_numerator, enthalpy),
        parameters.w,
    )
    if pressure_ratio.upper < 0:
        raise ValueError("kappa-family pressure ratio lost nonnegativity")

    initial_density = _total_density_point_interval(
        parameters.n_initial,
        parameters,
    )
    scale_factor = _monotone_exp_range(
        -2 * (right - parameters.n_initial),
        -2 * (left - parameters.n_initial),
    )
    growth_factor = _interval_multiply(
        scale_factor,
        _interval_divide(initial_density, total_density),
    )
    if growth_factor.lower <= 0:
        raise ValueError("kappa-family growth factor lost positivity")
    kappa_squared = _interval_multiply(
        growth_factor,
        initial_kappa_squared_interval,
    )

    damping_natural = _interval_add(
        _interval_add(_point_interval(4), hubble_log_derivative),
        pressure_ratio,
    )
    restoring_natural = _interval_add(
        _interval_add(
            _point_interval(3),
            _interval_scale(hubble_log_derivative, 2),
        ),
        _interval_multiply(
            pressure_ratio,
            _interval_add(
                _point_interval(1),
                _interval_scale(kappa_squared, Fraction(1, 3)),
            ),
        ),
    )
    damping = _RationalInterval(
        max(Fraction(5, 2), damping_natural.lower),
        damping_natural.upper,
    )
    restoring = _RationalInterval(
        max(Fraction(0), restoring_natural.lower),
        restoring_natural.upper,
    )
    if damping.lower <= 0 or damping.lower > damping.upper:
        raise ValueError("kappa-family damping enclosure is invalid")
    if restoring.lower > restoring.upper:
        raise ValueError("kappa-family restoring enclosure is invalid")

    off_diagonal = max(
        abs(1 - restoring.lower),
        abs(1 - restoring.upper),
    )
    logarithmic_rate = min(
        off_diagonal / 2,
        off_diagonal * off_diagonal / (4 * damping.lower),
    )
    parameter_lipschitz = (
        max(Fraction(0), pressure_ratio.upper)
        * growth_factor.upper
        / 3
    )
    return _KappaFamilyCellBounds(
        kappa_growth_factor_interval=(
            growth_factor.lower,
            growth_factor.upper,
        ),
        pressure_ratio_interval=(
            pressure_ratio.lower,
            pressure_ratio.upper,
        ),
        damping_interval=(damping.lower, damping.upper),
        restoring_interval=(restoring.lower, restoring.upper),
        parameter_generator_lipschitz_upper_bound=(
            parameter_lipschitz
        ),
        euclidean_logarithmic_norm_rate_upper_bound=(
            logarithmic_rate
        ),
    )


def _initial_normalized_state_and_sensitivity_bounds(
    *,
    reservoir_equation_of_state: Fraction,
    initial_kappa_squared_upper_bound: Fraction,
) -> tuple[Fraction, Fraction, tuple[Fraction, Fraction, Fraction]]:
    """Bound ||z_i||/|A| and ||partial_u z_i||/|A| by series ratios."""

    w = reservoir_equation_of_state
    u_upper = initial_kappa_squared_upper_bound
    rate = 1 + 3 * w
    friction = (5 + 3 * w) / 2
    q0 = w * u_upper / (rate * (rate + friction))
    q1 = w * u_upper / (rate * (2 * rate + friction))
    q2 = 2 * w * u_upper / (rate * (2 * rate + friction))
    if not (
        Fraction(0) <= q0 < 1
        and Fraction(0) <= q1 < 1
        and Fraction(0) <= q2 < 1
    ):
        raise ValueError("initial kappa-family derivative series is not contractive")

    state_bound = (
        1 / (1 - q0)
        + (w * u_upper / (rate + friction)) / (1 - q1)
    )
    sensitivity_bound = (
        (w / (rate * (rate + friction))) / (1 - q1)
        + (w / (rate + friction)) / (1 - q2)
    )
    return state_bound, sensitivity_bound, (q0, q1, q2)


@dataclass(frozen=True)
class CompactKappaBinWeylTransferEnclosureReceipt:
    """Cellwise uniform Weyl trace tube for one compact kappa_i bin."""

    central_trace_receipt: TraceEndpointEnclosureReceipt
    initial_kappa_interval: tuple[Fraction, Fraction]
    central_initial_kappa: Fraction
    initial_kappa_squared_interval: tuple[Fraction, Fraction]
    central_initial_kappa_squared: Fraction
    maximum_initial_kappa_squared_distance_from_center: Fraction
    primordial_potential_amplitude: Fraction
    refined_mesh: tuple[Fraction, ...]
    central_frozen_trace_nodes: tuple[tuple[Fraction, Fraction], ...]
    initial_series_ratio_upper_bounds: tuple[Fraction, Fraction, Fraction]
    normalized_initial_state_l2_upper_bound_via_l1: Fraction
    normalized_initial_parameter_derivative_l2_upper_bound_via_l1: Fraction
    kappa_growth_factor_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    pressure_ratio_cell_intervals: tuple[tuple[Fraction, Fraction], ...]
    damping_cell_intervals: tuple[tuple[Fraction, Fraction], ...]
    restoring_cell_intervals: tuple[tuple[Fraction, Fraction], ...]
    parameter_generator_lipschitz_cell_upper_bounds: tuple[Fraction, ...]
    euclidean_logarithmic_norm_cell_rate_upper_bounds: tuple[Fraction, ...]
    cell_exponential_propagator_upper_bounds: tuple[Fraction, ...]
    normalized_state_node_upper_bounds: tuple[Fraction, ...]
    normalized_parameter_derivative_node_upper_bounds: tuple[Fraction, ...]
    parameter_variation_curvature_node_radius_upper_bounds: (
        tuple[Fraction, ...]
    )
    central_analytic_regular_uniform_tube_radius_upper_bound: Fraction | None
    compact_bin_total_curvature_node_radius_upper_bounds: (
        tuple[Fraction, ...] | None
    )
    compact_bin_weyl_average_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...] | None
    )
    normalized_compact_bin_weyl_average_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...] | None
    )
    initial_bin_is_superhorizon: bool
    trace_generator_affine_in_initial_kappa_squared_proven: bool
    exact_regular_series_parameter_derivative_conservatively_enclosed: bool
    cellwise_parameter_generator_lipschitz_enclosed: bool
    cellwise_duhamel_sensitivity_recurrence_proven: bool
    central_trace_tube_reused: bool
    central_trace_radius_is_global_uniform_not_prefix_sharp: bool
    compact_kappa_bin_uniform_weyl_path_tube_enclosed: bool
    zero_anisotropic_stress_weyl_average_equals_curvature_adopted: bool
    physical_wavenumber_bin_calibrated: bool = False
    primordial_curvature_normalization_supplied: bool = False
    spherical_bessel_harmonic_projection_enclosed: bool = False
    all_k_einstein_boltzmann_transfer_enclosed: bool = False
    angular_power_spectrum_enclosed: bool = False
    covariance_or_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_COMPACT_INITIAL_KAPPA_BIN_UNIFORM_WEYL_TRACE_TRANSFER_"
        "TUBE_PER_FREE_POTENTIAL_AMPLITUDE_NOT_PRIMORDIAL_CURVATURE_ALL_K_"
        "HARMONIC_POWER_COVARIANCE_OR_LIKELIHOOD"
    )


def construct_compact_kappa_bin_weyl_transfer_enclosure(
    endpoint_enclosure: FiniteQuenchTraceEndpointEnclosure,
    *,
    initial_kappa_lower: object,
    initial_kappa_upper: object,
    primordial_potential_amplitude: object,
    coarse_step_count: object = 512,
) -> CompactKappaBinWeylTransferEnclosureReceipt:
    """Promote one central trace receipt to its whole compact kappa_i bin."""

    if not isinstance(endpoint_enclosure, FiniteQuenchTraceEndpointEnclosure):
        raise ValueError("endpoint_enclosure has the wrong type")
    lower = _finite_fraction(initial_kappa_lower, "initial kappa lower")
    upper = _finite_fraction(initial_kappa_upper, "initial kappa upper")
    center = _binary_fraction(endpoint_enclosure.evolution.kappa_initial)
    if not Fraction(0) < lower <= center <= upper <= Fraction(1, 10):
        raise ValueError(
            "compact bin requires 0 < kappa_lower <= center <= "
            "kappa_upper <= 0.1"
        )

    reference = endpoint_enclosure.construct(
        primordial_potential_amplitude=primordial_potential_amplitude,
        coarse_step_count=coarse_step_count,
    )
    _, refined = endpoint_enclosure._refined_mesh(coarse_step_count)
    nodes = endpoint_enclosure._trace_nodes(
        primordial_potential_amplitude=primordial_potential_amplitude,
        mesh=refined,
    )
    frozen_mesh = tuple(_binary_fraction(value) for value in refined)
    frozen_nodes = tuple(
        (_binary_fraction(curvature), _binary_fraction(curvature_prime))
        for curvature, curvature_prime in nodes
    )
    if (
        len(frozen_mesh) != reference.refined_step_count + 1
        or len(frozen_nodes) != len(frozen_mesh)
        or frozen_nodes[-1][0] != reference.endpoint_curvature_center
        or frozen_nodes[-1][1] != reference.endpoint_curvature_prime_center
    ):
        raise ValueError("central trace replay did not match its receipt")

    u_interval = _RationalInterval(lower * lower, upper * upper)
    u_center = center * center
    delta_u = max(u_center - u_interval.lower, u_interval.upper - u_center)
    amplitude = (
        reference.regular_initial_bridge.regular_mode_enclosure
        .primordial_potential_amplitude
    )
    parameters = endpoint_enclosure._frozen_parameters()
    state_bound, derivative_bound, ratios = (
        _initial_normalized_state_and_sensitivity_bounds(
            reservoir_equation_of_state=parameters.w,
            initial_kappa_squared_upper_bound=u_interval.upper,
        )
    )

    family_cells: list[_KappaFamilyCellBounds] = []
    propagators: list[Fraction] = []
    state_nodes = [state_bound]
    derivative_nodes = [derivative_bound]
    current_state = state_bound
    current_derivative = derivative_bound
    for left, right in zip(frozen_mesh[:-1], frozen_mesh[1:], strict=True):
        step = right - left
        if step <= 0:
            raise ValueError("compact kappa-bin mesh is not increasing")
        cell = _local_kappa_family_cell_bounds(
            left=left,
            right=right,
            parameters=parameters,
            initial_kappa_squared_interval=u_interval,
        )
        exponent = (
            cell.euclidean_logarithmic_norm_rate_upper_bound * step
        )
        propagator = _dyadic_upper_fraction(
            _rational_exp_interval(exponent).upper,
            bits=_SENSITIVITY_DYADIC_BITS,
        )
        next_derivative = _dyadic_upper_fraction(
            propagator
            * (
                current_derivative
                + step
                * cell.parameter_generator_lipschitz_upper_bound
                * current_state
            ),
            bits=_SENSITIVITY_DYADIC_BITS,
        )
        next_state = _dyadic_upper_fraction(
            propagator * current_state,
            bits=_SENSITIVITY_DYADIC_BITS,
        )
        family_cells.append(cell)
        propagators.append(propagator)
        state_nodes.append(next_state)
        derivative_nodes.append(next_derivative)
        current_state = next_state
        current_derivative = next_derivative

    parameter_radii = tuple(
        abs(amplitude) * delta_u * bound for bound in derivative_nodes
    )
    central_radius = (
        reference.uniform_trace_path_tube_and_efold_integral
        .analytic_regular_materialized_uniform_radius_upper_bound
    )
    if central_radius is None:
        total_radii = None
        weyl_cells = None
        normalized_weyl_cells = None
    else:
        total_radii = tuple(
            central_radius + radius for radius in parameter_radii
        )
        analytic_cells: list[tuple[Fraction, Fraction]] = []
        normalized_cells: list[tuple[Fraction, Fraction]] = []
        for index, (left_node, right_node) in enumerate(
            zip(frozen_nodes[:-1], frozen_nodes[1:], strict=True)
        ):
            radius = total_radii[index + 1]
            central_curvature = _RationalInterval(
                min(left_node[0], right_node[0]),
                max(left_node[0], right_node[0]),
            )
            analytic = _interval_add(
                central_curvature,
                _RationalInterval(-radius, radius),
            )
            analytic_cells.append((analytic.lower, analytic.upper))
            if amplitude != 0:
                normalized = _interval_divide(
                    analytic,
                    _point_interval(amplitude),
                )
                normalized_cells.append(
                    (normalized.lower, normalized.upper)
                )
        weyl_cells = tuple(analytic_cells)
        normalized_weyl_cells = (
            None if amplitude == 0 else tuple(normalized_cells)
        )

    return CompactKappaBinWeylTransferEnclosureReceipt(
        central_trace_receipt=reference,
        initial_kappa_interval=(lower, upper),
        central_initial_kappa=center,
        initial_kappa_squared_interval=(
            u_interval.lower,
            u_interval.upper,
        ),
        central_initial_kappa_squared=u_center,
        maximum_initial_kappa_squared_distance_from_center=delta_u,
        primordial_potential_amplitude=amplitude,
        refined_mesh=frozen_mesh,
        central_frozen_trace_nodes=frozen_nodes,
        initial_series_ratio_upper_bounds=ratios,
        normalized_initial_state_l2_upper_bound_via_l1=state_bound,
        normalized_initial_parameter_derivative_l2_upper_bound_via_l1=(
            derivative_bound
        ),
        kappa_growth_factor_cell_intervals=tuple(
            cell.kappa_growth_factor_interval for cell in family_cells
        ),
        pressure_ratio_cell_intervals=tuple(
            cell.pressure_ratio_interval for cell in family_cells
        ),
        damping_cell_intervals=tuple(
            cell.damping_interval for cell in family_cells
        ),
        restoring_cell_intervals=tuple(
            cell.restoring_interval for cell in family_cells
        ),
        parameter_generator_lipschitz_cell_upper_bounds=tuple(
            cell.parameter_generator_lipschitz_upper_bound
            for cell in family_cells
        ),
        euclidean_logarithmic_norm_cell_rate_upper_bounds=tuple(
            cell.euclidean_logarithmic_norm_rate_upper_bound
            for cell in family_cells
        ),
        cell_exponential_propagator_upper_bounds=tuple(propagators),
        normalized_state_node_upper_bounds=tuple(state_nodes),
        normalized_parameter_derivative_node_upper_bounds=tuple(
            derivative_nodes
        ),
        parameter_variation_curvature_node_radius_upper_bounds=(
            parameter_radii
        ),
        central_analytic_regular_uniform_tube_radius_upper_bound=(
            central_radius
        ),
        compact_bin_total_curvature_node_radius_upper_bounds=total_radii,
        compact_bin_weyl_average_cell_intervals=weyl_cells,
        normalized_compact_bin_weyl_average_cell_intervals=(
            normalized_weyl_cells
        ),
        initial_bin_is_superhorizon=upper <= Fraction(1, 10),
        trace_generator_affine_in_initial_kappa_squared_proven=True,
        exact_regular_series_parameter_derivative_conservatively_enclosed=(
            True
        ),
        cellwise_parameter_generator_lipschitz_enclosed=True,
        cellwise_duhamel_sensitivity_recurrence_proven=True,
        central_trace_tube_reused=True,
        central_trace_radius_is_global_uniform_not_prefix_sharp=True,
        compact_kappa_bin_uniform_weyl_path_tube_enclosed=(
            weyl_cells is not None
        ),
        zero_anisotropic_stress_weyl_average_equals_curvature_adopted=True,
    )
