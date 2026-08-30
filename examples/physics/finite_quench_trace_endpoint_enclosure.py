"""Rigorous trace-endpoint enclosures for frozen and analytic-regular CE IVPs.

The numerical Magnus nodes are not assumed exact. They are frozen as exact
binary rationals and joined by a continuous piecewise-linear path. Exact
rational coefficient bounds and a Duhamel residual estimate then enclose the
endpoint of the analytic trace ODE that starts at the first frozen node.

An independent exact-rational regular-series enclosure also bounds the
difference between that frozen node and the analytic source-off bounded-past
trace mode. Duhamel propagation then supplies a second endpoint ball for that
analytic regular IVP. The potential amplitude remains free: this does not
supply a primordial spectrum, inflationary state, scalar-clock endpoint, or
observable transfer function.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from numbers import Integral

from examples.physics.finite_quench_regular_metric_evolution import (
    FiniteQuenchRegularMetricEvolution,
)
from examples.physics.finite_quench_superhorizon_regularity import (
    ExactRegularModeInitialEnclosureReceipt,
    FiniteQuenchSuperhorizonRegularity,
)


_MAX_AUTOMATIC_COARSE_STEPS = 200_000
_MAX_COEFFICIENT_EXP_ARGUMENT = Fraction(128)
_MAX_MATERIALIZED_RADIUS_EXP_ARGUMENT = Fraction(32)
_WEIGHTED_SOURCE_SUBINTERVALS = 8
_LOCAL_COEFFICIENT_BLOCK_STEPS = 1
_INTERVAL_DYADIC_BITS = 160


def _binary_fraction(value: float) -> Fraction:
    """Freeze one finite Python float as its exact binary rational."""

    if isinstance(value, bool) or not isinstance(value, float):
        raise ValueError("binary-fraction input must be a float")
    if not math.isfinite(value):
        raise ValueError("binary-fraction input must be finite")
    return Fraction.from_float(value)


def _unit_direction_cosine_fraction(value: object) -> Fraction:
    """Freeze an exact supplied sightline direction cosine in [-1, 1]."""

    if isinstance(value, bool):
        raise ValueError("direction cosine must be a real scalar")
    if isinstance(value, float):
        result = _binary_fraction(value)
    elif isinstance(value, (Fraction, Integral)):
        result = Fraction(value)
    else:
        raise ValueError(
            "direction cosine must be an int, Fraction, or finite float"
        )
    if not -1 <= result <= 1:
        raise ValueError("direction cosine must lie in [-1, 1]")
    return result


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _dyadic_upper_fraction(
    value: Fraction,
    *,
    bits: int = 48,
) -> Fraction:
    if value < 0 or bits < 1:
        raise ValueError("dyadic upper conversion requires value >= 0")
    scale = 1 << bits
    scaled = value * scale
    upper_integer = -(-scaled.numerator // scaled.denominator)
    result = Fraction(upper_integer, scale)
    if result < value:
        raise ValueError("dyadic upper conversion rounded downward")
    return result


@lru_cache(maxsize=None)
def _rational_exp_bounds(
    value: Fraction,
) -> tuple[Fraction, Fraction, int]:
    """Enclose exp(value) by a Taylor lower sum and geometric tail.

    Only nonnegative arguments are needed here. With the partial sum through
    N and N + 2 > value, the omitted terms are positive and their successive
    ratios are at most value / (N + 2).
    """

    if value < 0:
        raise ValueError("rational exponential enclosure requires value >= 0")
    if value > _MAX_COEFFICIENT_EXP_ARGUMENT:
        raise ValueError(
            "rational exponential coefficient argument exceeds the "
            "certified work limit"
        )
    term_count = max(16, _ceil_fraction(value) + 16)
    if not Fraction(term_count + 2) > value:
        raise ValueError("Taylor tail ratio is not strictly below one")

    term = Fraction(1)
    lower = term
    for index in range(1, term_count + 1):
        term *= value / index
        lower += term
    first_omitted = term * value / (term_count + 1)
    tail_ratio = value / (term_count + 2)
    upper = lower + first_omitted / (1 - tail_ratio)
    if not Fraction(0) < lower <= upper:
        raise ValueError("rational exponential enclosure is invalid")
    return lower, upper, term_count


@dataclass(frozen=True)
class _RationalInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("rational interval endpoints are reversed")


def _outward_dyadic(
    interval: _RationalInterval,
) -> _RationalInterval:
    scale = 1 << _INTERVAL_DYADIC_BITS
    lower_scaled = interval.lower * scale
    upper_scaled = interval.upper * scale
    lower_integer = (
        lower_scaled.numerator // lower_scaled.denominator
    )
    upper_integer = -(
        -upper_scaled.numerator // upper_scaled.denominator
    )
    return _RationalInterval(
        Fraction(lower_integer, scale),
        Fraction(upper_integer, scale),
    )


def _point_interval(value: Fraction | int) -> _RationalInterval:
    point = Fraction(value)
    return _RationalInterval(point, point)


def _interval_add(
    left: _RationalInterval,
    right: _RationalInterval,
) -> _RationalInterval:
    return _outward_dyadic(
        _RationalInterval(
            left.lower + right.lower,
            left.upper + right.upper,
        )
    )


def _interval_subtract(
    left: _RationalInterval,
    right: _RationalInterval,
) -> _RationalInterval:
    return _outward_dyadic(
        _RationalInterval(
            left.lower - right.upper,
            left.upper - right.lower,
        )
    )


def _interval_multiply(
    left: _RationalInterval,
    right: _RationalInterval,
) -> _RationalInterval:
    products = (
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    )
    return _outward_dyadic(
        _RationalInterval(min(products), max(products))
    )


def _interval_divide(
    numerator: _RationalInterval,
    denominator: _RationalInterval,
) -> _RationalInterval:
    if denominator.lower <= 0 <= denominator.upper:
        raise ValueError("rational interval division crossed zero")
    reciprocal = _outward_dyadic(
        _RationalInterval(
            1 / denominator.upper,
            1 / denominator.lower,
        )
    )
    return _interval_multiply(numerator, reciprocal)


def _interval_scale(
    interval: _RationalInterval,
    scalar: Fraction | int,
) -> _RationalInterval:
    return _interval_multiply(interval, _point_interval(Fraction(scalar)))


def _interval_abs_upper(interval: _RationalInterval) -> Fraction:
    return max(abs(interval.lower), abs(interval.upper))


@lru_cache(maxsize=None)
def _rational_exp_interval(value: Fraction) -> _RationalInterval:
    if value >= 0:
        lower, upper, _ = _rational_exp_bounds(value)
        return _outward_dyadic(_RationalInterval(lower, upper))
    lower, upper, _ = _rational_exp_bounds(-value)
    return _outward_dyadic(
        _RationalInterval(1 / upper, 1 / lower)
    )


def _monotone_exp_range(
    left_exponent: Fraction,
    right_exponent: Fraction,
) -> _RationalInterval:
    low_exponent = min(left_exponent, right_exponent)
    high_exponent = max(left_exponent, right_exponent)
    low = _rational_exp_interval(low_exponent)
    high = _rational_exp_interval(high_exponent)
    return _outward_dyadic(
        _RationalInterval(low.lower, high.upper)
    )


def _inverse_sqrt_interval(
    interval: _RationalInterval,
) -> _RationalInterval:
    """Return an exact dyadic enclosure of ``1 / sqrt(interval)``."""

    if interval.lower <= 0:
        raise ValueError("inverse-square-root interval must be positive")
    scale = 1 << _INTERVAL_DYADIC_BITS
    scale_squared = scale * scale

    def point_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
        target_floor = (
            scale_squared * value.denominator // value.numerator
        )
        lower_integer = math.isqrt(target_floor)
        lower = Fraction(lower_integer, scale)
        if (
            lower_integer * lower_integer * value.numerator
            == scale_squared * value.denominator
        ):
            upper = lower
        else:
            upper = Fraction(lower_integer + 1, scale)
        return lower, upper

    lower, _ = point_bounds(interval.upper)
    _, upper = point_bounds(interval.lower)
    result = _RationalInterval(lower, upper)
    if result.lower * result.lower * interval.upper > 1:
        raise ValueError("inverse-square-root lower enclosure failed")
    if result.upper * result.upper * interval.lower < 1:
        raise ValueError("inverse-square-root upper enclosure failed")
    return result


@dataclass(frozen=True)
class _FrozenTraceParameters:
    w: Fraction
    omega: Fraction
    reservoir_today: Fraction
    width: Fraction
    center: Fraction
    n_initial: Fraction
    n_final: Fraction
    kappa_initial: Fraction

    @property
    def source_minus(self) -> Fraction:
        return self.center - self.width

    @property
    def source_plus(self) -> Fraction:
        return self.center + self.width


def _compact_bump_at(
    n: Fraction,
    parameters: _FrozenTraceParameters,
) -> Fraction:
    if n <= parameters.source_minus or n >= parameters.source_plus:
        return Fraction(0)
    x = (n - parameters.center) / parameters.width
    return Fraction(15, 16) * (1 - x * x) ** 2 / parameters.width


def _compact_cumulative_at(
    n: Fraction,
    parameters: _FrozenTraceParameters,
) -> Fraction:
    if n <= parameters.source_minus:
        return Fraction(0)
    if n >= parameters.source_plus:
        return Fraction(1)
    x = (n - parameters.center) / parameters.width
    primitive = (
        x
        - Fraction(2, 3) * x**3
        + Fraction(1, 5) * x**5
        + Fraction(8, 15)
    )
    return Fraction(15, 16) * primitive


def _compact_bump_interval(
    left: Fraction,
    right: Fraction,
    parameters: _FrozenTraceParameters,
) -> _RationalInterval:
    active_left = max(left, parameters.source_minus)
    active_right = min(right, parameters.source_plus)
    if active_left >= active_right:
        return _point_interval(0)
    candidates = [
        _compact_bump_at(active_left, parameters),
        _compact_bump_at(active_right, parameters),
    ]
    if active_left <= parameters.center <= active_right:
        candidates.append(Fraction(15, 16) / parameters.width)
    lower = min(candidates)
    if left < parameters.source_minus or right > parameters.source_plus:
        lower = Fraction(0)
    return _RationalInterval(lower, max(candidates))


def _produced_density_interval(
    left: Fraction,
    right: Fraction,
    parameters: _FrozenTraceParameters,
) -> _RationalInterval:
    exponential = _monotone_exp_range(-3 * right, -3 * left)
    cumulative = _RationalInterval(
        _compact_cumulative_at(left, parameters),
        _compact_cumulative_at(right, parameters),
    )
    return _interval_scale(
        _interval_multiply(exponential, cumulative),
        parameters.omega,
    )


def _source_density_interval(
    left: Fraction,
    right: Fraction,
    parameters: _FrozenTraceParameters,
) -> _RationalInterval:
    exponential = _monotone_exp_range(-3 * right, -3 * left)
    bump = _compact_bump_interval(left, right, parameters)
    return _interval_scale(
        _interval_multiply(exponential, bump),
        parameters.omega,
    )


@lru_cache(maxsize=None)
def _weighted_source_integral_interval(
    n: Fraction,
    parameters: _FrozenTraceParameters,
) -> _RationalInterval:
    active_left = max(n, parameters.source_minus)
    active_right = min(Fraction(0), parameters.source_plus)
    if (
        parameters.omega == 0
        or active_left >= active_right
    ):
        return _point_interval(0)

    step = (
        active_right - active_left
    ) / _WEIGHTED_SOURCE_SUBINTERVALS
    result = _point_interval(0)
    for index in range(_WEIGHTED_SOURCE_SUBINTERVALS):
        left = active_left + index * step
        right = left + step
        mass = (
            _compact_cumulative_at(right, parameters)
            - _compact_cumulative_at(left, parameters)
        )
        if mass < 0:
            raise ValueError("compact cumulative mass became negative")
        left_exp = _rational_exp_interval(
            3 * parameters.w * left
        )
        right_exp = _rational_exp_interval(
            3 * parameters.w * right
        )
        contribution = _interval_scale(
            _RationalInterval(left_exp.lower, right_exp.upper),
            parameters.omega * mass,
        )
        result = _interval_add(result, contribution)
    return result


@lru_cache(maxsize=None)
def _reservoir_point_interval(
    n: Fraction,
    parameters: _FrozenTraceParameters,
) -> _RationalInterval:
    lam = 3 * (1 + parameters.w)
    prefactor = _rational_exp_interval(-lam * n)
    paid = _weighted_source_integral_interval(n, parameters)
    bracket = _interval_add(
        _point_interval(parameters.reservoir_today),
        paid,
    )
    return _interval_multiply(prefactor, bracket)


@lru_cache(maxsize=None)
def _total_density_point_interval(
    n: Fraction,
    parameters: _FrozenTraceParameters,
) -> _RationalInterval:
    produced = _produced_density_interval(n, n, parameters)
    reservoir = _reservoir_point_interval(n, parameters)
    return _interval_add(produced, reservoir)


def _total_density_interval(
    left: Fraction,
    right: Fraction,
    parameters: _FrozenTraceParameters,
) -> _RationalInterval:
    left_value = _total_density_point_interval(left, parameters)
    right_value = _total_density_point_interval(right, parameters)
    interval = _RationalInterval(
        right_value.lower,
        left_value.upper,
    )
    if interval.lower <= 0:
        raise ValueError("total-density interval lost positivity")
    return interval


def _fraction_to_float_upper(value: Fraction) -> float | None:
    """Return a finite float no smaller than a nonnegative exact fraction."""

    if value < 0:
        raise ValueError("float upper conversion requires value >= 0")
    try:
        result = float(value)
    except OverflowError:
        return None
    if not math.isfinite(result):
        return None
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result) or Fraction.from_float(result) < value:
        return None
    return result


def _materialize_exponential_radius(
    coefficient: Fraction,
    exponent: Fraction,
) -> tuple[Fraction | None, Fraction | None]:
    if coefficient < 0 or exponent < 0:
        raise ValueError("endpoint-radius factors must be nonnegative")
    if coefficient == 0:
        return Fraction(0), Fraction(0)
    argument = _dyadic_upper_fraction(exponent)
    if argument > _MAX_MATERIALIZED_RADIUS_EXP_ARGUMENT:
        return None, None
    _, exponential_upper, _ = _rational_exp_bounds(argument)
    return argument, coefficient * exponential_upper


def _component_interval(
    center: Fraction,
    radius: Fraction | None,
) -> tuple[Fraction, Fraction] | None:
    if radius is None:
        return None
    return center - radius, center + radius


def _certified_component_sign(
    interval: tuple[Fraction, Fraction] | None,
) -> int | None:
    if interval is None:
        return None
    if interval[0] > 0:
        return 1
    if interval[1] < 0:
        return -1
    return None


@dataclass(frozen=True)
class ExactExponentialRadius:
    """Exact symbolic radius coefficient times exp(exponent)."""

    coefficient: Fraction
    exponent: Fraction
    coefficient_nonnegative: bool
    exponent_nonnegative: bool
    role: str = "EXACT_REAL_RADIUS_COEFFICIENT_TIMES_EXPONENTIALED_RATIONAL"


@dataclass(frozen=True)
class RationalTraceCoefficientBoundReceipt:
    """Exact-rational global bounds for the analytic trace generator."""

    source_intersects_interval: bool
    source_left_endpoint: Fraction | None
    source_right_endpoint: Fraction | None
    source_upper_bound: Fraction
    enthalpy_lower_bound: Fraction
    pressure_ratio_upper_bound: Fraction
    kappa_upper_bound: Fraction
    damping_lower_bound: Fraction
    damping_upper_bound: Fraction
    restoring_lower_bound: Fraction
    restoring_upper_bound: Fraction
    generator_entrywise_norm_upper_bound: Fraction
    euclidean_logarithmic_norm_rate_upper_bound: Fraction
    exact_binary_float_parameters_frozen: bool
    rational_taylor_exponential_enclosures_proven: bool
    weighted_source_integral_partition_count: int
    component_and_source_nonnegativity_used: bool
    kappa_squared_monotonicity_proven: bool
    coefficient_bounds_proven_on_full_interval: bool
    role: str = (
        "EXACT_RATIONAL_GLOBAL_TRACE_GENERATOR_BOUNDS_"
        "FOR_FROZEN_BINARY_FLOAT_PARAMETERS"
    )


@dataclass(frozen=True)
class RegularInitialTraceBridgeReceipt:
    """Exact component defect from the analytic regular mode to the PL path."""

    regular_mode_enclosure: ExactRegularModeInitialEnclosureReceipt
    frozen_initial_curvature: Fraction
    frozen_initial_curvature_prime: Fraction
    curvature_defect_abs_upper_bound: Fraction
    curvature_prime_defect_abs_upper_bound: Fraction
    initial_l1_defect_upper_bound: Fraction
    exact_component_difference_enclosures_proven: bool
    euclidean_initial_defect_bounded_by_l1: bool
    floating_series_and_rhs_roundoff_absorbed: bool
    analytic_regular_trace_initial_state_enclosed: bool
    physical_primordial_amplitude_supplied: bool = False
    scalar_clock_initial_state_enclosed: bool = False
    role: str = (
        "EXACT_RATIONAL_ANALYTIC_REGULAR_TO_FROZEN_TRACE_INITIAL_DEFECT_"
        "NOT_PRIMORDIAL_AMPLITUDE_OR_SCALAR_CLOCK_CERTIFICATE"
    )


@dataclass(frozen=True)
class TraceScalarClockEndpointReceipt:
    """Algebraic common-clock endpoint enclosure from the trace-state ball."""

    n: Fraction
    total_density_interval: tuple[Fraction, Fraction]
    total_enthalpy_interval: tuple[Fraction, Fraction]
    negative_hubble_log_derivative_interval: tuple[Fraction, Fraction]
    kappa_squared_interval: tuple[Fraction, Fraction]
    clock_reconstruction_coefficient_interval: tuple[Fraction, Fraction]
    frozen_numeric_scalar_clock_center: Fraction
    trace_curvature_interval: tuple[Fraction, Fraction] | None
    trace_curvature_prime_interval: tuple[Fraction, Fraction] | None
    reconstruction_numerator_interval: tuple[Fraction, Fraction] | None
    scalar_clock_interval: tuple[Fraction, Fraction] | None
    scalar_clock_certified_sign: int | None
    analytic_regular_trace_endpoint_used: bool
    trace_to_clock_algebraic_inversion_proven: bool
    negative_hubble_separated_from_zero: bool
    exact_rational_outward_interval_operations_proven: bool
    normalized_dimensionless_reconstruction_proven: bool
    scalar_clock_endpoint_enclosed: bool
    independent_scalar_clock_dynamical_integration_proven: bool = False
    physical_canonical_clock_identification_proven: bool = False
    primordial_spectrum_supplied: bool = False
    observable_transfer_function_enclosed: bool = False
    role: str = (
        "CONDITIONAL_EXACT_RATIONAL_COMMON_LEDGER_CLOCK_ENDPOINT_FROM_"
        "ANALYTIC_REGULAR_TRACE_BALL_NOT_CANONICAL_CLOCK_OR_OBSERVABLE"
    )


@dataclass(frozen=True)
class AmplitudeNormalizedEndpointResponseReceipt:
    """Per-input analytic response coefficients, not physical observables.

    The true linearity flags concern the exact recurrence and analytic ODE.
    Separately recomputed floating paths are deliberately not scale-certified.
    """

    supplied_amplitude: Fraction
    amplitude_sign: int
    normalization_defined: bool
    curvature_response_interval: tuple[Fraction, Fraction] | None
    curvature_prime_response_interval: tuple[Fraction, Fraction] | None
    common_ledger_clock_response_interval: (
        tuple[Fraction, Fraction] | None
    )
    curvature_response_certified_sign: int | None
    curvature_prime_response_certified_sign: int | None
    common_ledger_clock_response_certified_sign: int | None
    exact_regular_series_linearity_proven: bool
    analytic_trace_ode_homogeneity_proven: bool
    common_ledger_clock_reconstruction_linearity_proven: bool
    exact_rational_signed_point_division_proven: bool
    fixed_amplitude_conditional_response_enclosed: bool
    frozen_recomputed_path_scale_invariance_proven: bool = False
    residual_and_initial_bound_uniform_abs_amplitude_scaling_proven: (
        bool
    ) = False
    physical_primordial_normalization_supplied: bool = False
    physical_observable_transfer_function_enclosed: bool = False
    role: str = (
        "PER_SUPPLIED_AMPLITUDE_NORMALIZED_ANALYTIC_TRACE_AND_COMMON_LEDGER_"
        "CLOCK_RESPONSE_NOT_AMPLITUDE_UNIFORM_ERROR_OR_PHYSICAL_OBSERVABLE"
    )


@dataclass(frozen=True)
class ConditionalWeylMetricEndpointReceipt:
    """Zero-stress Newtonian-gauge metric readout, not a lensing observable."""

    curvature_potential_interval: tuple[Fraction, Fraction] | None
    lapse_potential_interval: tuple[Fraction, Fraction] | None
    weyl_average_potential_interval: tuple[Fraction, Fraction] | None
    weyl_sum_metric_source_interval: tuple[Fraction, Fraction] | None
    normalized_weyl_average_response_interval: (
        tuple[Fraction, Fraction] | None
    )
    normalized_weyl_sum_metric_response_interval: (
        tuple[Fraction, Fraction] | None
    )
    deterministic_weyl_average_squared_gain_interval: (
        tuple[Fraction, Fraction] | None
    )
    deterministic_weyl_sum_squared_gain_interval: (
        tuple[Fraction, Fraction] | None
    )
    weyl_average_response_certified_sign: int | None
    weyl_sum_response_certified_sign: int | None
    newtonian_gauge_metric_convention_fixed: bool
    zero_total_anisotropic_stress_adopted_effective_closure: bool
    lapse_equals_curvature_in_conditional_branch_proven: bool
    conditional_metric_potential_endpoint_enclosed: bool
    conditional_amplitude_normalized_metric_response_enclosed: bool
    line_of_sight_lensing_observable_enclosed: bool = False
    einstein_boltzmann_solution_enclosed: bool = False
    primordial_power_spectrum_supplied: bool = False
    physical_power_transfer_function_enclosed: bool = False
    cmb_lss_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_ZERO_STRESS_WEYL_METRIC_ENDPOINT_AND_DETERMINISTIC_GAIN_"
        "NOT_LINE_OF_SIGHT_LENSING_POWER_SPECTRUM_OR_LIKELIHOOD"
    )


@dataclass(frozen=True)
class UniformTracePathTubeAndEfoldIntegralReceipt:
    """Uniform analytic trace tube and unweighted e-fold metric integral."""

    n_initial: Fraction
    n_final: Fraction
    interval_width: Fraction
    refined_step_count: int
    frozen_pl_curvature_efold_integral: Fraction
    frozen_ivp_symbolic_uniform_radius: ExactExponentialRadius
    analytic_regular_symbolic_uniform_radius: ExactExponentialRadius
    frozen_ivp_materialized_uniform_radius_upper_bound: Fraction | None
    analytic_regular_materialized_uniform_radius_upper_bound: Fraction | None
    analytic_curvature_efold_integral_radius_upper_bound: Fraction | None
    analytic_curvature_efold_integral_interval: (
        tuple[Fraction, Fraction] | None
    )
    conditional_weyl_average_efold_integral_interval: (
        tuple[Fraction, Fraction] | None
    )
    conditional_weyl_sum_efold_integral_interval: (
        tuple[Fraction, Fraction] | None
    )
    normalized_curvature_efold_response_interval: (
        tuple[Fraction, Fraction] | None
    )
    normalized_weyl_sum_efold_response_interval: (
        tuple[Fraction, Fraction] | None
    )
    normalized_curvature_efold_response_certified_sign: int | None
    normalized_weyl_sum_efold_response_certified_sign: int | None
    continuous_piecewise_linear_path_integrated_exactly: bool
    nonnegative_prefix_residual_budget_bounded_by_total: bool
    nonnegative_prefix_logarithmic_norm_bounded_by_total: bool
    uniform_trace_state_tube_covers_every_prefix: bool
    exact_symbolic_uniform_path_tube_proven: bool
    materialized_analytic_regular_uniform_path_tube_proven: bool
    unweighted_efold_metric_integral_enclosed: bool
    prefix_sharp_radius_proven: bool = False
    conformal_or_comoving_line_of_sight_integral_enclosed: bool = False
    photon_geodesic_lensing_observable_enclosed: bool = False
    integrated_sachs_wolfe_observable_enclosed: bool = False
    primordial_power_spectrum_supplied: bool = False
    role: str = (
        "UNIFORM_ANALYTIC_REGULAR_TRACE_PATH_TUBE_AND_UNWEIGHTED_EFOLD_"
        "WEYL_METRIC_INTEGRAL_NOT_LINE_OF_SIGHT_OR_PHOTON_OBSERVABLE"
    )


@dataclass(frozen=True)
class BackgroundConformalMetricTimeIntegralReceipt:
    """Flat-background conformal-time metric integral, not an observable."""

    n_initial: Fraction
    n_final: Fraction
    refined_step_count: int
    primordial_potential_amplitude: Fraction
    conformal_weight_interval_hull: tuple[Fraction, Fraction]
    dimensionless_background_conformal_time_interval: (
        tuple[Fraction, Fraction]
    )
    frozen_pl_weyl_average_conformal_time_integral_interval: (
        tuple[Fraction, Fraction]
    )
    analytic_regular_symbolic_weyl_average_integral_radius: (
        ExactExponentialRadius
    )
    analytic_regular_materialized_weyl_average_integral_radius_upper_bound: (
        Fraction | None
    )
    analytic_regular_weyl_average_conformal_time_integral_interval: (
        tuple[Fraction, Fraction] | None
    )
    analytic_regular_weyl_sum_conformal_time_integral_interval: (
        tuple[Fraction, Fraction] | None
    )
    normalized_weyl_average_conformal_time_response_interval: (
        tuple[Fraction, Fraction] | None
    )
    normalized_weyl_sum_conformal_time_response_interval: (
        tuple[Fraction, Fraction] | None
    )
    normalized_weyl_average_response_certified_sign: int | None
    normalized_weyl_sum_response_certified_sign: int | None
    normalization_defined: bool
    flat_gr_radial_null_measure_identity_proven: bool
    exact_rational_inverse_square_root_enclosures_proven: bool
    positive_conformal_weight_on_every_mesh_cell: bool
    cellwise_interval_weighted_pl_metric_integral_enclosed: bool
    uniform_trace_tube_integrated_against_positive_measure: bool
    materialized_analytic_regular_metric_time_integral_enclosed: bool
    unperturbed_flat_background_radial_null_measure_used: bool
    physical_density_scale_calibration_supplied: bool = False
    spatial_mode_phase_on_null_path_supplied: bool = False
    lensing_source_distance_and_kernel_supplied: bool = False
    transverse_laplacian_or_angular_mode_supplied: bool = False
    photon_geodesic_lensing_observable_enclosed: bool = False
    integrated_sachs_wolfe_observable_enclosed: bool = False
    all_k_einstein_boltzmann_solution_enclosed: bool = False
    primordial_power_spectrum_supplied: bool = False
    cmb_lss_likelihood_enclosed: bool = False
    role: str = (
        "DIMENSIONLESS_FLAT_BACKGROUND_CONFORMAL_TIME_WEIGHTED_WEYL_METRIC_"
        "INTEGRAL_NOT_SPATIALLY_EVALUATED_LINE_OF_SIGHT_LENSING_ISW_POWER_"
        "SPECTRUM_OR_LIKELIHOOD"
    )


@dataclass(frozen=True)
class FixedModeBornLensingOrientationEnvelopeReceipt:
    """Orientation-resolved modulus law for one fixed Born-lensing mode."""

    supplied_direction_cosine: Fraction
    direction_cosine_squared: Fraction
    transverse_wavenumber_squared_fraction: Fraction
    unoriented_frozen_pl_absolute_upper_bound: Fraction
    oriented_frozen_pl_absolute_upper_bound: Fraction
    unoriented_analytic_regular_absolute_upper_bound: Fraction | None
    oriented_analytic_regular_absolute_upper_bound: Fraction | None
    unoriented_normalized_absolute_upper_bound: Fraction | None
    oriented_normalized_absolute_upper_bound: Fraction | None
    uniform_direction_cosine_mean_absolute_upper_bound: Fraction | None
    uniform_direction_cosine_mean_normalized_upper_bound: Fraction | None
    oriented_bound_strictly_below_unity: bool | None
    exact_supplied_direction_cosine_frozen: bool
    transverse_wavenumber_identity_used: bool
    orientation_resolved_absolute_envelope_enclosed: bool
    uniform_direction_cosine_measure_adopted: bool
    uniform_direction_cosine_mean_absolute_envelope_enclosed: bool
    signed_convergence_enclosed: bool = False
    spatial_mode_phase_supplied: bool = False
    physical_orientation_distribution_supplied: bool = False
    isotropic_cosmological_ensemble_claimed: bool = False
    shear_or_lensing_map_enclosed: bool = False
    angular_power_spectrum_enclosed: bool = False
    primordial_power_spectrum_supplied: bool = False
    cmb_lss_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_SUPPLIED_DIRECTION_FIXED_FOURIER_MODE_BORN_LENSING_"
        "ABSOLUTE_ENVELOPE_AND_FORMAL_UNIFORM_MU_MEAN_NOT_SIGNED_MAP_"
        "COSMOLOGICAL_ENSEMBLE_POWER_SPECTRUM_OR_LIKELIHOOD"
    )


@dataclass(frozen=True)
class FixedModeBornLensingAbsoluteEnvelopeReceipt:
    """Conditional absolute envelope for one Born-lensing Fourier mode."""

    n_source: Fraction
    n_observer: Fraction
    refined_step_count: int
    primordial_potential_amplitude: Fraction
    initial_k_over_a_h: Fraction
    dimensionless_fixed_wavenumber_squared_interval: (
        tuple[Fraction, Fraction]
    )
    dimensionless_conformal_cell_measure_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    dimensionless_source_side_distance_node_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    dimensionless_observer_side_distance_node_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    dimensionless_source_distance_interval: tuple[Fraction, Fraction]
    dimensionless_geometric_kernel_cell_upper_bounds: tuple[Fraction, ...]
    dimensionless_geometric_kernel_upper_bound: Fraction
    frozen_pl_weyl_average_absolute_geometry_integral_upper_bound: Fraction
    analytic_uniform_tube_geometry_measure_upper_bound: Fraction
    frozen_pl_born_convergence_absolute_upper_bound: Fraction
    analytic_regular_born_convergence_absolute_upper_bound: Fraction | None
    normalized_analytic_regular_born_convergence_absolute_upper_bound: (
        Fraction | None
    )
    single_mode_convergence_bound_strictly_below_unity: bool | None
    source_and_observer_planes_fixed_at_interval_endpoints: bool
    flat_background_born_weak_lensing_equation_adopted: bool
    newtonian_gauge_zero_anisotropic_stress_adopted: bool
    single_fixed_fourier_mode_adopted: bool
    exact_rational_dimensionless_fixed_wavenumber_enclosed: bool
    positive_conformal_cell_measure_enclosed: bool
    prefix_and_suffix_distances_accumulated_independently: bool
    source_distance_identity_enclosed_by_intersection: bool
    nonnegative_flat_lensing_kernel_enclosed_cellwise: bool
    transverse_wavenumber_bounded_by_total_wavenumber: bool
    spatial_fourier_phase_modulus_bounded_by_one: bool
    uniform_analytic_trace_tube_used: bool
    conditional_single_mode_born_convergence_absolute_envelope_enclosed: bool
    signed_single_mode_convergence_enclosed: bool = False
    transverse_mode_orientation_supplied: bool = False
    spatial_mode_phase_on_null_path_supplied: bool = False
    source_redshift_calibration_supplied: bool = False
    source_population_distribution_supplied: bool = False
    born_weak_field_validity_independently_derived: bool = False
    perturbed_or_post_born_geodesic_enclosed: bool = False
    all_k_einstein_boltzmann_solution_enclosed: bool = False
    primordial_power_spectrum_supplied: bool = False
    shear_or_lensing_map_enclosed: bool = False
    angular_power_spectrum_enclosed: bool = False
    cmb_lss_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_FLAT_BACKGROUND_BORN_SINGLE_FIXED_FOURIER_MODE_"
        "LENSING_CONVERGENCE_ABSOLUTE_ENVELOPE_NOT_SIGNED_MAP_SHEAR_"
        "POWER_SPECTRUM_OR_LIKELIHOOD"
    )

    def at_direction_cosine(
        self,
        direction_cosine: object,
    ) -> FixedModeBornLensingOrientationEnvelopeReceipt:
        """Resolve the modulus envelope at an exactly supplied direction."""

        mu = _unit_direction_cosine_fraction(direction_cosine)
        mu_squared = mu * mu
        transverse_fraction = 1 - mu_squared

        def scale(value: Fraction | None) -> Fraction | None:
            if value is None:
                return None
            return transverse_fraction * value

        oriented_analytic = scale(
            self.analytic_regular_born_convergence_absolute_upper_bound
        )
        oriented_normalized = scale(
            self
            .normalized_analytic_regular_born_convergence_absolute_upper_bound
        )
        uniform_mean = (
            None
            if self.analytic_regular_born_convergence_absolute_upper_bound
            is None
            else Fraction(2, 3)
            * self.analytic_regular_born_convergence_absolute_upper_bound
        )
        uniform_normalized_mean = (
            None
            if self
            .normalized_analytic_regular_born_convergence_absolute_upper_bound
            is None
            else Fraction(2, 3)
            * self
            .normalized_analytic_regular_born_convergence_absolute_upper_bound
        )
        return FixedModeBornLensingOrientationEnvelopeReceipt(
            supplied_direction_cosine=mu,
            direction_cosine_squared=mu_squared,
            transverse_wavenumber_squared_fraction=transverse_fraction,
            unoriented_frozen_pl_absolute_upper_bound=(
                self.frozen_pl_born_convergence_absolute_upper_bound
            ),
            oriented_frozen_pl_absolute_upper_bound=(
                transverse_fraction
                * self.frozen_pl_born_convergence_absolute_upper_bound
            ),
            unoriented_analytic_regular_absolute_upper_bound=(
                self.analytic_regular_born_convergence_absolute_upper_bound
            ),
            oriented_analytic_regular_absolute_upper_bound=(
                oriented_analytic
            ),
            unoriented_normalized_absolute_upper_bound=(
                self
                .normalized_analytic_regular_born_convergence_absolute_upper_bound
            ),
            oriented_normalized_absolute_upper_bound=oriented_normalized,
            uniform_direction_cosine_mean_absolute_upper_bound=uniform_mean,
            uniform_direction_cosine_mean_normalized_upper_bound=(
                uniform_normalized_mean
            ),
            oriented_bound_strictly_below_unity=(
                None
                if oriented_analytic is None
                else oriented_analytic < 1
            ),
            exact_supplied_direction_cosine_frozen=True,
            transverse_wavenumber_identity_used=True,
            orientation_resolved_absolute_envelope_enclosed=(
                oriented_analytic is not None
            ),
            uniform_direction_cosine_measure_adopted=True,
            uniform_direction_cosine_mean_absolute_envelope_enclosed=(
                uniform_mean is not None
            ),
        )


@dataclass(frozen=True)
class TraceEndpointEnclosureReceipt:
    """Endpoint balls for frozen-node and analytic-regular trace IVPs."""

    coefficient_bounds: RationalTraceCoefficientBoundReceipt
    n_initial: Fraction
    n_final: Fraction
    requested_coarse_step_count: int
    coarse_step_count: int
    refined_step_count: int
    local_coefficient_block_step_count: int
    endpoint_curvature_center: Fraction
    endpoint_curvature_prime_center: Fraction
    global_triangle_residual_l1_integral_upper_bound: Fraction
    local_interval_residual_l1_integral_upper_bound: Fraction
    residual_l1_integral_upper_bound: Fraction
    global_propagation_exponent_upper_bound: Fraction
    local_integrated_logarithmic_norm_exponent_upper_bound: Fraction
    propagation_exponent_upper_bound: Fraction
    endpoint_radius: ExactExponentialRadius
    materialized_exponential_argument_upper_bound: Fraction | None
    materialized_rational_radius_upper_bound: Fraction | None
    materialized_float_radius_upper_bound: float | None
    curvature_component_interval: tuple[Fraction, Fraction] | None
    curvature_prime_component_interval: tuple[Fraction, Fraction] | None
    curvature_component_certified_sign: int | None
    curvature_prime_component_certified_sign: int | None
    regular_initial_bridge: RegularInitialTraceBridgeReceipt
    analytic_regular_endpoint_radius: ExactExponentialRadius
    analytic_regular_materialized_exponential_argument_upper_bound: (
        Fraction | None
    )
    analytic_regular_materialized_rational_radius_upper_bound: Fraction | None
    analytic_regular_materialized_float_radius_upper_bound: float | None
    analytic_regular_curvature_component_interval: (
        tuple[Fraction, Fraction] | None
    )
    analytic_regular_curvature_prime_component_interval: (
        tuple[Fraction, Fraction] | None
    )
    analytic_regular_curvature_component_certified_sign: int | None
    analytic_regular_curvature_prime_component_certified_sign: int | None
    scalar_clock_endpoint: TraceScalarClockEndpointReceipt
    amplitude_normalized_response: (
        AmplitudeNormalizedEndpointResponseReceipt
    )
    conditional_weyl_metric_endpoint: (
        ConditionalWeylMetricEndpointReceipt
    )
    uniform_trace_path_tube_and_efold_integral: (
        UniformTracePathTubeAndEfoldIntegralReceipt
    )
    background_conformal_metric_time_integral: (
        BackgroundConformalMetricTimeIntegralReceipt
    )
    fixed_mode_born_lensing_absolute_envelope: (
        FixedModeBornLensingAbsoluteEnvelopeReceipt
    )
    refined_nodes_frozen_as_exact_binary_rationals: bool
    continuous_piecewise_linear_reconstruction_proven: bool
    piecewise_join_defect_zero_proven: bool
    local_coefficient_interval_enclosures_proven: bool
    local_residual_cancellation_retained: bool
    local_time_dependent_logarithmic_norm_integrated: bool
    residual_integral_bound_proven_by_exact_rational_arithmetic: bool
    logarithmic_norm_propagation_bound_proven: bool
    exact_symbolic_trace_endpoint_ball_proven: bool
    rigorous_materialized_trace_endpoint_enclosure_proven: bool
    exact_symbolic_analytic_regular_endpoint_ball_proven: bool
    rigorous_materialized_analytic_regular_endpoint_enclosure_proven: bool
    analytic_source_off_regular_initial_condition_enclosed: bool
    conditional_amplitude_normalized_response_enclosed: bool
    conditional_weyl_metric_endpoint_enclosed: bool
    analytic_regular_uniform_trace_path_tube_proven: bool
    background_conformal_metric_time_integral_enclosed: bool
    conditional_fixed_mode_born_lensing_absolute_envelope_enclosed: bool
    numerical_node_roundoff_absorbed_into_frozen_path: bool
    normalized_dimensionless_model_assumed: bool
    physical_primordial_initial_condition_enclosed: bool = False
    scalar_clock_endpoint_enclosed: bool = False
    numerical_method_convergence_theorem_proven: bool = False
    observable_transfer_function_enclosed: bool = False
    role: str = (
        "RIGOROUS_PIECEWISE_LINEAR_RESIDUAL_ENDPOINT_BALLS_FOR_"
        "FROZEN_AND_ANALYTIC_SOURCE_OFF_REGULAR_TRACE_INITIAL_VALUE_PROBLEMS_"
        "WITH_CONDITIONAL_COMMON_LEDGER_CLOCK_AND_PER_AMPLITUDE_RESPONSE_"
        "NOT_PRIMORDIAL_SPECTRUM_CANONICAL_CLOCK_OBSERVABLE_OR_NUMERICAL_"
        "CONVERGENCE_PROOF"
    )


class FiniteQuenchTraceEndpointEnclosure:
    """Build a rigorous endpoint ball around a frozen piecewise-linear path."""

    def __init__(self, evolution: FiniteQuenchRegularMetricEvolution) -> None:
        if not isinstance(evolution, FiniteQuenchRegularMetricEvolution):
            raise ValueError(
                "evolution must be a FiniteQuenchRegularMetricEvolution"
            )
        self.evolution = evolution

    def _frozen_parameters(self) -> _FrozenTraceParameters:
        config = self.evolution.bridge.config
        return _FrozenTraceParameters(
            w=_binary_fraction(config.w_reservoir),
            omega=_binary_fraction(config.omega_prod0),
            reservoir_today=_binary_fraction(
                config.reservoir_present_density
            ),
            width=_binary_fraction(config.half_width),
            center=_binary_fraction(config.n_star),
            n_initial=_binary_fraction(self.evolution.n_initial),
            n_final=_binary_fraction(self.evolution.n_final),
            kappa_initial=_binary_fraction(self.evolution.kappa_initial),
        )

    def _local_trace_coefficient_intervals(
        self,
        left: Fraction,
        right: Fraction,
        parameters: _FrozenTraceParameters,
        global_bounds: RationalTraceCoefficientBoundReceipt,
    ) -> tuple[_RationalInterval, _RationalInterval]:
        produced = _produced_density_interval(left, right, parameters)
        reservoir_left = _reservoir_point_interval(left, parameters)
        reservoir_right = _reservoir_point_interval(right, parameters)
        reservoir = _RationalInterval(
            reservoir_right.lower,
            reservoir_left.upper,
        )
        source = _source_density_interval(left, right, parameters)
        total_density = _total_density_interval(
            left,
            right,
            parameters,
        )
        enthalpy = _interval_add(
            produced,
            _interval_scale(reservoir, 1 + parameters.w),
        )
        if enthalpy.lower <= 0:
            raise ValueError("local enthalpy interval lost positivity")

        hubble_log_derivative = _interval_scale(
            _interval_divide(enthalpy, total_density),
            Fraction(-3, 2),
        )
        pressure_numerator = _interval_add(
            _interval_scale(
                reservoir,
                3 * (1 + parameters.w),
            ),
            source,
        )
        pressure_ratio = _interval_scale(
            _interval_divide(pressure_numerator, enthalpy),
            parameters.w,
        )

        initial_density = _total_density_point_interval(
            parameters.n_initial,
            parameters,
        )
        scale_factor = _monotone_exp_range(
            -2 * (right - parameters.n_initial),
            -2 * (left - parameters.n_initial),
        )
        kappa_squared = _interval_scale(
            _interval_multiply(
                scale_factor,
                _interval_divide(initial_density, total_density),
            ),
            parameters.kappa_initial * parameters.kappa_initial,
        )

        damping_natural = _interval_add(
            _interval_add(
                _point_interval(4),
                hubble_log_derivative,
            ),
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
            max(
                Fraction(5, 2),
                damping_natural.lower,
            ),
            min(
                global_bounds.damping_upper_bound,
                damping_natural.upper,
            ),
        )
        restoring = _RationalInterval(
            max(Fraction(0), restoring_natural.lower),
            min(
                global_bounds.restoring_upper_bound,
                restoring_natural.upper,
            ),
        )
        return damping, restoring

    def _scalar_clock_endpoint_receipt(
        self,
        *,
        parameters: _FrozenTraceParameters,
        endpoint_curvature: Fraction,
        endpoint_curvature_prime: Fraction,
        analytic_curvature_interval: tuple[Fraction, Fraction] | None,
        analytic_curvature_prime_interval: (
            tuple[Fraction, Fraction] | None
        ),
    ) -> TraceScalarClockEndpointReceipt:
        """Apply T = [psi' + (1 + K/3) psi] / (-h) at n_final."""

        n = parameters.n_final
        produced = _produced_density_interval(n, n, parameters)
        reservoir = _reservoir_point_interval(n, parameters)
        total_density = _interval_add(produced, reservoir)
        total_enthalpy = _interval_add(
            produced,
            _interval_scale(reservoir, 1 + parameters.w),
        )
        negative_hubble_natural = _interval_scale(
            _interval_divide(total_enthalpy, total_density),
            Fraction(3, 2),
        )
        negative_hubble = _RationalInterval(
            max(Fraction(3, 2), negative_hubble_natural.lower),
            min(Fraction(3), negative_hubble_natural.upper),
        )
        if negative_hubble.lower <= 0:
            raise ValueError(
                "scalar-clock reconstruction denominator crossed zero"
            )

        initial_density = _total_density_point_interval(
            parameters.n_initial,
            parameters,
        )
        scale_factor = _rational_exp_interval(
            -2 * (n - parameters.n_initial)
        )
        kappa_squared = _interval_scale(
            _interval_multiply(
                scale_factor,
                _interval_divide(initial_density, total_density),
            ),
            parameters.kappa_initial * parameters.kappa_initial,
        )
        clock_coefficient = _interval_add(
            _point_interval(1),
            _interval_scale(kappa_squared, Fraction(1, 3)),
        )
        if clock_coefficient.lower < 1:
            raise ValueError(
                "scalar-clock reconstruction coefficient lost positivity"
            )

        frozen_clock = _binary_fraction(
            self.evolution._clock_from_trace_state(
                float(n),
                float(endpoint_curvature),
                float(endpoint_curvature_prime),
            )
        )
        if (
            analytic_curvature_interval is None
            or analytic_curvature_prime_interval is None
        ):
            if (
                analytic_curvature_interval is not None
                or analytic_curvature_prime_interval is not None
            ):
                raise ValueError(
                    "analytic trace component intervals are inconsistent"
                )
            reconstruction_numerator = None
            scalar_clock_interval = None
        else:
            curvature = _RationalInterval(*analytic_curvature_interval)
            curvature_prime = _RationalInterval(
                *analytic_curvature_prime_interval
            )
            numerator = _interval_add(
                curvature_prime,
                _interval_multiply(clock_coefficient, curvature),
            )
            scalar_clock = _interval_divide(
                numerator,
                negative_hubble,
            )
            reconstruction_numerator = (
                numerator.lower,
                numerator.upper,
            )
            scalar_clock_interval = (
                scalar_clock.lower,
                scalar_clock.upper,
            )

        return TraceScalarClockEndpointReceipt(
            n=n,
            total_density_interval=(
                total_density.lower,
                total_density.upper,
            ),
            total_enthalpy_interval=(
                total_enthalpy.lower,
                total_enthalpy.upper,
            ),
            negative_hubble_log_derivative_interval=(
                negative_hubble.lower,
                negative_hubble.upper,
            ),
            kappa_squared_interval=(
                kappa_squared.lower,
                kappa_squared.upper,
            ),
            clock_reconstruction_coefficient_interval=(
                clock_coefficient.lower,
                clock_coefficient.upper,
            ),
            frozen_numeric_scalar_clock_center=frozen_clock,
            trace_curvature_interval=analytic_curvature_interval,
            trace_curvature_prime_interval=(
                analytic_curvature_prime_interval
            ),
            reconstruction_numerator_interval=reconstruction_numerator,
            scalar_clock_interval=scalar_clock_interval,
            scalar_clock_certified_sign=_certified_component_sign(
                scalar_clock_interval
            ),
            analytic_regular_trace_endpoint_used=True,
            trace_to_clock_algebraic_inversion_proven=True,
            negative_hubble_separated_from_zero=True,
            exact_rational_outward_interval_operations_proven=True,
            normalized_dimensionless_reconstruction_proven=True,
            scalar_clock_endpoint_enclosed=(
                scalar_clock_interval is not None
            ),
        )

    @staticmethod
    def _amplitude_normalized_response_receipt(
        *,
        amplitude: Fraction,
        analytic_curvature_interval: tuple[Fraction, Fraction] | None,
        analytic_curvature_prime_interval: (
            tuple[Fraction, Fraction] | None
        ),
        scalar_clock_interval: tuple[Fraction, Fraction] | None,
    ) -> AmplitudeNormalizedEndpointResponseReceipt:
        """Divide one fixed-amplitude analytic certificate by its exact A."""

        normalization_defined = amplitude != 0

        def normalize(
            interval: tuple[Fraction, Fraction] | None,
        ) -> tuple[Fraction, Fraction] | None:
            if not normalization_defined or interval is None:
                return None
            normalized = _interval_divide(
                _RationalInterval(*interval),
                _point_interval(amplitude),
            )
            return normalized.lower, normalized.upper

        curvature_response = normalize(analytic_curvature_interval)
        curvature_prime_response = normalize(
            analytic_curvature_prime_interval
        )
        clock_response = normalize(scalar_clock_interval)
        fixed_amplitude_enclosed = (
            normalization_defined
            and curvature_response is not None
            and curvature_prime_response is not None
            and clock_response is not None
        )
        return AmplitudeNormalizedEndpointResponseReceipt(
            supplied_amplitude=amplitude,
            amplitude_sign=(
                1 if amplitude > 0 else (-1 if amplitude < 0 else 0)
            ),
            normalization_defined=normalization_defined,
            curvature_response_interval=curvature_response,
            curvature_prime_response_interval=curvature_prime_response,
            common_ledger_clock_response_interval=clock_response,
            curvature_response_certified_sign=(
                _certified_component_sign(curvature_response)
            ),
            curvature_prime_response_certified_sign=(
                _certified_component_sign(curvature_prime_response)
            ),
            common_ledger_clock_response_certified_sign=(
                _certified_component_sign(clock_response)
            ),
            exact_regular_series_linearity_proven=True,
            analytic_trace_ode_homogeneity_proven=True,
            common_ledger_clock_reconstruction_linearity_proven=True,
            exact_rational_signed_point_division_proven=(
                normalization_defined
            ),
            fixed_amplitude_conditional_response_enclosed=(
                fixed_amplitude_enclosed
            ),
        )

    @staticmethod
    def _conditional_weyl_metric_endpoint_receipt(
        *,
        analytic_curvature_interval: tuple[Fraction, Fraction] | None,
        amplitude_response: AmplitudeNormalizedEndpointResponseReceipt,
    ) -> ConditionalWeylMetricEndpointReceipt:
        """Use phi = psi to read out Weyl-average and Weyl-sum intervals."""

        def scale(
            interval: tuple[Fraction, Fraction] | None,
            factor: Fraction,
        ) -> tuple[Fraction, Fraction] | None:
            if interval is None:
                return None
            scaled = _interval_scale(
                _RationalInterval(*interval),
                factor,
            )
            return scaled.lower, scaled.upper

        def square(
            interval: tuple[Fraction, Fraction] | None,
        ) -> tuple[Fraction, Fraction] | None:
            if interval is None:
                return None
            lower, upper = interval
            squared_upper = max(lower * lower, upper * upper)
            squared_lower = (
                Fraction(0)
                if lower <= 0 <= upper
                else min(lower * lower, upper * upper)
            )
            return squared_lower, squared_upper

        weyl_average = analytic_curvature_interval
        weyl_sum = scale(analytic_curvature_interval, Fraction(2))
        normalized_weyl = amplitude_response.curvature_response_interval
        normalized_weyl_sum = scale(normalized_weyl, Fraction(2))
        return ConditionalWeylMetricEndpointReceipt(
            curvature_potential_interval=analytic_curvature_interval,
            lapse_potential_interval=analytic_curvature_interval,
            weyl_average_potential_interval=weyl_average,
            weyl_sum_metric_source_interval=weyl_sum,
            normalized_weyl_average_response_interval=normalized_weyl,
            normalized_weyl_sum_metric_response_interval=(
                normalized_weyl_sum
            ),
            deterministic_weyl_average_squared_gain_interval=(
                square(normalized_weyl)
            ),
            deterministic_weyl_sum_squared_gain_interval=(
                square(normalized_weyl_sum)
            ),
            weyl_average_response_certified_sign=(
                _certified_component_sign(normalized_weyl)
            ),
            weyl_sum_response_certified_sign=(
                _certified_component_sign(normalized_weyl_sum)
            ),
            newtonian_gauge_metric_convention_fixed=True,
            zero_total_anisotropic_stress_adopted_effective_closure=True,
            lapse_equals_curvature_in_conditional_branch_proven=True,
            conditional_metric_potential_endpoint_enclosed=(
                analytic_curvature_interval is not None
            ),
            conditional_amplitude_normalized_metric_response_enclosed=(
                normalized_weyl is not None
            ),
        )

    @staticmethod
    def _uniform_trace_path_tube_receipt(
        *,
        frozen_mesh: tuple[Fraction, ...],
        frozen_nodes: tuple[tuple[Fraction, Fraction], ...],
        frozen_symbolic_radius: ExactExponentialRadius,
        analytic_symbolic_radius: ExactExponentialRadius,
        frozen_materialized_radius: Fraction | None,
        analytic_materialized_radius: Fraction | None,
        amplitude: Fraction,
    ) -> UniformTracePathTubeAndEfoldIntegralReceipt:
        """Promote the total Duhamel budget to every path prefix."""

        if (
            len(frozen_mesh) < 2
            or len(frozen_nodes) != len(frozen_mesh)
        ):
            raise ValueError("uniform path tube requires aligned PL nodes")
        if any(
            left_n >= right_n
            for left_n, right_n in zip(frozen_mesh, frozen_mesh[1:])
        ):
            raise ValueError(
                "uniform path tube mesh must be strictly increasing"
            )
        interval_width = frozen_mesh[-1] - frozen_mesh[0]
        if interval_width <= 0:
            raise ValueError("uniform path tube interval must be positive")

        pl_integral = sum(
            (right_n - left_n)
            * (left_y[0] + right_y[0])
            / 2
            for left_n, right_n, left_y, right_y in zip(
                frozen_mesh[:-1],
                frozen_mesh[1:],
                frozen_nodes[:-1],
                frozen_nodes[1:],
                strict=True,
            )
        )
        if analytic_materialized_radius is None:
            integral_radius = None
            curvature_integral = None
        else:
            integral_radius = (
                interval_width * analytic_materialized_radius
            )
            curvature_integral = (
                pl_integral - integral_radius,
                pl_integral + integral_radius,
            )

        def scale(
            interval: tuple[Fraction, Fraction] | None,
            factor: Fraction,
        ) -> tuple[Fraction, Fraction] | None:
            if interval is None:
                return None
            scaled = _interval_scale(
                _RationalInterval(*interval),
                factor,
            )
            return scaled.lower, scaled.upper

        def normalize(
            interval: tuple[Fraction, Fraction] | None,
        ) -> tuple[Fraction, Fraction] | None:
            if interval is None or amplitude == 0:
                return None
            normalized = _interval_divide(
                _RationalInterval(*interval),
                _point_interval(amplitude),
            )
            return normalized.lower, normalized.upper

        weyl_sum_integral = scale(curvature_integral, Fraction(2))
        normalized_curvature_integral = normalize(curvature_integral)
        normalized_weyl_sum_integral = normalize(weyl_sum_integral)
        return UniformTracePathTubeAndEfoldIntegralReceipt(
            n_initial=frozen_mesh[0],
            n_final=frozen_mesh[-1],
            interval_width=interval_width,
            refined_step_count=len(frozen_mesh) - 1,
            frozen_pl_curvature_efold_integral=pl_integral,
            frozen_ivp_symbolic_uniform_radius=frozen_symbolic_radius,
            analytic_regular_symbolic_uniform_radius=(
                analytic_symbolic_radius
            ),
            frozen_ivp_materialized_uniform_radius_upper_bound=(
                frozen_materialized_radius
            ),
            analytic_regular_materialized_uniform_radius_upper_bound=(
                analytic_materialized_radius
            ),
            analytic_curvature_efold_integral_radius_upper_bound=(
                integral_radius
            ),
            analytic_curvature_efold_integral_interval=curvature_integral,
            conditional_weyl_average_efold_integral_interval=(
                curvature_integral
            ),
            conditional_weyl_sum_efold_integral_interval=(
                weyl_sum_integral
            ),
            normalized_curvature_efold_response_interval=(
                normalized_curvature_integral
            ),
            normalized_weyl_sum_efold_response_interval=(
                normalized_weyl_sum_integral
            ),
            normalized_curvature_efold_response_certified_sign=(
                _certified_component_sign(normalized_curvature_integral)
            ),
            normalized_weyl_sum_efold_response_certified_sign=(
                _certified_component_sign(normalized_weyl_sum_integral)
            ),
            continuous_piecewise_linear_path_integrated_exactly=True,
            nonnegative_prefix_residual_budget_bounded_by_total=True,
            nonnegative_prefix_logarithmic_norm_bounded_by_total=True,
            uniform_trace_state_tube_covers_every_prefix=True,
            exact_symbolic_uniform_path_tube_proven=True,
            materialized_analytic_regular_uniform_path_tube_proven=(
                analytic_materialized_radius is not None
            ),
            unweighted_efold_metric_integral_enclosed=(
                curvature_integral is not None
            ),
        )

    @staticmethod
    def _background_conformal_metric_time_integral_receipt(
        *,
        frozen_mesh: tuple[Fraction, ...],
        frozen_nodes: tuple[tuple[Fraction, Fraction], ...],
        parameters: _FrozenTraceParameters,
        analytic_symbolic_radius: ExactExponentialRadius,
        analytic_materialized_radius: Fraction | None,
        amplitude: Fraction,
    ) -> BackgroundConformalMetricTimeIntegralReceipt:
        """Enclose ``H_rho integral Phi_W d eta`` on the background path."""

        if (
            len(frozen_mesh) < 2
            or len(frozen_nodes) != len(frozen_mesh)
        ):
            raise ValueError(
                "background conformal integral requires aligned PL nodes"
            )
        if any(
            left_n >= right_n
            for left_n, right_n in zip(frozen_mesh, frozen_mesh[1:])
        ):
            raise ValueError(
                "background conformal integral mesh must be strictly increasing"
            )

        conformal_time = _point_interval(0)
        weighted_pl_integral = _point_interval(0)
        weight_intervals: list[_RationalInterval] = []
        for left_n, right_n, left_y, right_y in zip(
            frozen_mesh[:-1],
            frozen_mesh[1:],
            frozen_nodes[:-1],
            frozen_nodes[1:],
            strict=True,
        ):
            step = right_n - left_n
            density = _total_density_interval(
                left_n,
                right_n,
                parameters,
            )
            inverse_hubble_ratio = _inverse_sqrt_interval(density)
            inverse_scale_factor = _monotone_exp_range(
                -right_n,
                -left_n,
            )
            weight = _interval_multiply(
                inverse_scale_factor,
                inverse_hubble_ratio,
            )
            if weight.lower <= 0:
                raise ValueError("background conformal weight lost positivity")
            weight_intervals.append(weight)
            conformal_time = _interval_add(
                conformal_time,
                _interval_scale(weight, step),
            )
            curvature = _RationalInterval(
                min(left_y[0], right_y[0]),
                max(left_y[0], right_y[0]),
            )
            weighted_pl_integral = _interval_add(
                weighted_pl_integral,
                _interval_scale(
                    _interval_multiply(weight, curvature),
                    step,
                ),
            )

        weight_hull = _RationalInterval(
            min(interval.lower for interval in weight_intervals),
            max(interval.upper for interval in weight_intervals),
        )
        symbolic_integral_radius = ExactExponentialRadius(
            coefficient=(
                analytic_symbolic_radius.coefficient
                * conformal_time.upper
            ),
            exponent=analytic_symbolic_radius.exponent,
            coefficient_nonnegative=(
                analytic_symbolic_radius.coefficient_nonnegative
                and conformal_time.upper >= 0
            ),
            exponent_nonnegative=(
                analytic_symbolic_radius.exponent_nonnegative
            ),
        )
        if analytic_materialized_radius is None:
            materialized_integral_radius = None
            analytic_weyl_average_integral = None
        else:
            materialized_integral_radius = (
                analytic_materialized_radius * conformal_time.upper
            )
            analytic_interval = _interval_add(
                weighted_pl_integral,
                _RationalInterval(
                    -materialized_integral_radius,
                    materialized_integral_radius,
                ),
            )
            analytic_weyl_average_integral = (
                analytic_interval.lower,
                analytic_interval.upper,
            )

        def scale(
            interval: tuple[Fraction, Fraction] | None,
            factor: Fraction,
        ) -> tuple[Fraction, Fraction] | None:
            if interval is None:
                return None
            scaled = _interval_scale(
                _RationalInterval(*interval),
                factor,
            )
            return scaled.lower, scaled.upper

        def normalize(
            interval: tuple[Fraction, Fraction] | None,
        ) -> tuple[Fraction, Fraction] | None:
            if interval is None or amplitude == 0:
                return None
            normalized = _interval_divide(
                _RationalInterval(*interval),
                _point_interval(amplitude),
            )
            return normalized.lower, normalized.upper

        analytic_weyl_sum_integral = scale(
            analytic_weyl_average_integral,
            Fraction(2),
        )
        normalized_weyl_average = normalize(
            analytic_weyl_average_integral
        )
        normalized_weyl_sum = normalize(analytic_weyl_sum_integral)
        return BackgroundConformalMetricTimeIntegralReceipt(
            n_initial=frozen_mesh[0],
            n_final=frozen_mesh[-1],
            refined_step_count=len(frozen_mesh) - 1,
            primordial_potential_amplitude=amplitude,
            conformal_weight_interval_hull=(
                weight_hull.lower,
                weight_hull.upper,
            ),
            dimensionless_background_conformal_time_interval=(
                conformal_time.lower,
                conformal_time.upper,
            ),
            frozen_pl_weyl_average_conformal_time_integral_interval=(
                weighted_pl_integral.lower,
                weighted_pl_integral.upper,
            ),
            analytic_regular_symbolic_weyl_average_integral_radius=(
                symbolic_integral_radius
            ),
            analytic_regular_materialized_weyl_average_integral_radius_upper_bound=(
                materialized_integral_radius
            ),
            analytic_regular_weyl_average_conformal_time_integral_interval=(
                analytic_weyl_average_integral
            ),
            analytic_regular_weyl_sum_conformal_time_integral_interval=(
                analytic_weyl_sum_integral
            ),
            normalized_weyl_average_conformal_time_response_interval=(
                normalized_weyl_average
            ),
            normalized_weyl_sum_conformal_time_response_interval=(
                normalized_weyl_sum
            ),
            normalized_weyl_average_response_certified_sign=(
                _certified_component_sign(normalized_weyl_average)
            ),
            normalized_weyl_sum_response_certified_sign=(
                _certified_component_sign(normalized_weyl_sum)
            ),
            normalization_defined=amplitude != 0,
            flat_gr_radial_null_measure_identity_proven=True,
            exact_rational_inverse_square_root_enclosures_proven=True,
            positive_conformal_weight_on_every_mesh_cell=True,
            cellwise_interval_weighted_pl_metric_integral_enclosed=True,
            uniform_trace_tube_integrated_against_positive_measure=True,
            materialized_analytic_regular_metric_time_integral_enclosed=(
                analytic_weyl_average_integral is not None
            ),
            unperturbed_flat_background_radial_null_measure_used=True,
        )

    @staticmethod
    def _fixed_mode_born_lensing_absolute_envelope_receipt(
        *,
        frozen_mesh: tuple[Fraction, ...],
        frozen_nodes: tuple[tuple[Fraction, Fraction], ...],
        parameters: _FrozenTraceParameters,
        analytic_materialized_radius: Fraction | None,
        amplitude: Fraction,
    ) -> FixedModeBornLensingAbsoluteEnvelopeReceipt:
        """Enclose one fixed-mode Born convergence contribution in modulus.

        Adopt the flat-background Born equation

        kappa = integral dchi K(chi) nabla_perp^2 Phi_W

        with a source at the initial slice, an observer at the final slice,
        K = chi_O chi_S / chi_s, and Phi_W = psi on the conditional
        zero-anisotropic-stress branch.  For one Fourier mode,
        k_perp^2 <= k^2 and the phase has modulus one.  The result is an
        absolute upper bound only; it does not determine a sign or a map.
        """

        if (
            len(frozen_mesh) < 2
            or len(frozen_nodes) != len(frozen_mesh)
        ):
            raise ValueError(
                "fixed-mode Born envelope requires aligned PL nodes"
            )
        if any(
            left_n >= right_n
            for left_n, right_n in zip(frozen_mesh, frozen_mesh[1:])
        ):
            raise ValueError(
                "fixed-mode Born envelope mesh must be strictly increasing"
            )
        if (
            parameters.n_initial != frozen_mesh[0]
            or parameters.n_final != frozen_mesh[-1]
        ):
            raise ValueError(
                "fixed-mode Born envelope parameters and mesh disagree"
            )

        cell_measures: list[_RationalInterval] = []
        for left_n, right_n in zip(
            frozen_mesh[:-1],
            frozen_mesh[1:],
            strict=True,
        ):
            density = _total_density_interval(
                left_n,
                right_n,
                parameters,
            )
            weight = _interval_multiply(
                _monotone_exp_range(-right_n, -left_n),
                _inverse_sqrt_interval(density),
            )
            measure = _interval_scale(weight, right_n - left_n)
            if measure.lower <= 0:
                raise ValueError(
                    "fixed-mode Born conformal cell measure lost positivity"
                )
            cell_measures.append(measure)

        source_side = [_point_interval(0)]
        for measure in cell_measures:
            source_side.append(_interval_add(source_side[-1], measure))

        observer_side = [_point_interval(0) for _ in frozen_mesh]
        for index in range(len(cell_measures) - 1, -1, -1):
            observer_side[index] = _interval_add(
                cell_measures[index],
                observer_side[index + 1],
            )

        source_distance_lower = max(
            source_side[-1].lower,
            observer_side[0].lower,
        )
        source_distance_upper = min(
            source_side[-1].upper,
            observer_side[0].upper,
        )
        if not 0 < source_distance_lower <= source_distance_upper:
            raise ValueError(
                "fixed-mode Born source-distance enclosures do not overlap"
            )
        source_distance = _RationalInterval(
            source_distance_lower,
            source_distance_upper,
        )

        initial_density = _total_density_point_interval(
            frozen_mesh[0],
            parameters,
        )
        fixed_wavenumber_squared = _interval_multiply(
            _point_interval(
                parameters.kappa_initial * parameters.kappa_initial
            ),
            _interval_multiply(
                _rational_exp_interval(2 * frozen_mesh[0]),
                initial_density,
            ),
        )
        if fixed_wavenumber_squared.lower < 0:
            raise ValueError(
                "dimensionless fixed wavenumber squared lost nonnegativity"
            )

        kernel_upper_bounds: list[Fraction] = []
        frozen_geometry_integral = Fraction(0)
        tube_geometry_measure = Fraction(0)
        global_kernel_upper = source_distance.upper / 4
        for index, (left_y, right_y, measure) in enumerate(
            zip(
                frozen_nodes[:-1],
                frozen_nodes[1:],
                cell_measures,
                strict=True,
            )
        ):
            independent_distance_upper = (
                source_side[index + 1].upper
                * observer_side[index].upper
                / source_distance.lower
            )
            kernel_upper = min(
                independent_distance_upper,
                global_kernel_upper,
            )
            if kernel_upper < 0:
                raise ValueError(
                    "fixed-mode Born geometric kernel lost nonnegativity"
                )
            kernel_upper_bounds.append(kernel_upper)
            weighted_geometry_upper = measure.upper * kernel_upper
            pl_absolute_upper = max(abs(left_y[0]), abs(right_y[0]))
            frozen_geometry_integral += (
                weighted_geometry_upper * pl_absolute_upper
            )
            tube_geometry_measure += weighted_geometry_upper

        q_squared_upper = fixed_wavenumber_squared.upper
        frozen_born_upper = q_squared_upper * frozen_geometry_integral
        if analytic_materialized_radius is None:
            analytic_born_upper = None
            normalized_analytic_born_upper = None
            below_unity = None
        else:
            analytic_born_upper = q_squared_upper * (
                frozen_geometry_integral
                + analytic_materialized_radius * tube_geometry_measure
            )
            normalized_analytic_born_upper = (
                None
                if amplitude == 0
                else analytic_born_upper / abs(amplitude)
            )
            below_unity = analytic_born_upper < 1

        def pairs(
            intervals: list[_RationalInterval],
        ) -> tuple[tuple[Fraction, Fraction], ...]:
            return tuple(
                (interval.lower, interval.upper)
                for interval in intervals
            )

        return FixedModeBornLensingAbsoluteEnvelopeReceipt(
            n_source=frozen_mesh[0],
            n_observer=frozen_mesh[-1],
            refined_step_count=len(frozen_mesh) - 1,
            primordial_potential_amplitude=amplitude,
            initial_k_over_a_h=parameters.kappa_initial,
            dimensionless_fixed_wavenumber_squared_interval=(
                fixed_wavenumber_squared.lower,
                fixed_wavenumber_squared.upper,
            ),
            dimensionless_conformal_cell_measure_intervals=pairs(
                cell_measures
            ),
            dimensionless_source_side_distance_node_intervals=pairs(
                source_side
            ),
            dimensionless_observer_side_distance_node_intervals=pairs(
                observer_side
            ),
            dimensionless_source_distance_interval=(
                source_distance.lower,
                source_distance.upper,
            ),
            dimensionless_geometric_kernel_cell_upper_bounds=tuple(
                kernel_upper_bounds
            ),
            dimensionless_geometric_kernel_upper_bound=max(
                kernel_upper_bounds
            ),
            frozen_pl_weyl_average_absolute_geometry_integral_upper_bound=(
                frozen_geometry_integral
            ),
            analytic_uniform_tube_geometry_measure_upper_bound=(
                tube_geometry_measure
            ),
            frozen_pl_born_convergence_absolute_upper_bound=(
                frozen_born_upper
            ),
            analytic_regular_born_convergence_absolute_upper_bound=(
                analytic_born_upper
            ),
            normalized_analytic_regular_born_convergence_absolute_upper_bound=(
                normalized_analytic_born_upper
            ),
            single_mode_convergence_bound_strictly_below_unity=below_unity,
            source_and_observer_planes_fixed_at_interval_endpoints=True,
            flat_background_born_weak_lensing_equation_adopted=True,
            newtonian_gauge_zero_anisotropic_stress_adopted=True,
            single_fixed_fourier_mode_adopted=True,
            exact_rational_dimensionless_fixed_wavenumber_enclosed=True,
            positive_conformal_cell_measure_enclosed=True,
            prefix_and_suffix_distances_accumulated_independently=True,
            source_distance_identity_enclosed_by_intersection=True,
            nonnegative_flat_lensing_kernel_enclosed_cellwise=True,
            transverse_wavenumber_bounded_by_total_wavenumber=True,
            spatial_fourier_phase_modulus_bounded_by_one=True,
            uniform_analytic_trace_tube_used=True,
            conditional_single_mode_born_convergence_absolute_envelope_enclosed=(
                analytic_born_upper is not None
            ),
        )

    def coefficient_bounds(self) -> RationalTraceCoefficientBoundReceipt:
        """Derive exact-rational A, B, kappa, and logarithmic-norm bounds."""

        evolution = self.evolution
        config = evolution.bridge.config
        w = _binary_fraction(config.w_reservoir)
        omega = _binary_fraction(config.omega_prod0)
        reservoir_today = _binary_fraction(
            config.reservoir_present_density
        )
        width = _binary_fraction(config.half_width)
        center = _binary_fraction(config.n_star)
        n_initial = _binary_fraction(evolution.n_initial)
        n_final = _binary_fraction(evolution.n_final)
        kappa_initial = _binary_fraction(evolution.kappa_initial)
        if not Fraction(0) <= w <= Fraction(1):
            raise ValueError("rational trace enclosure requires 0 <= w <= 1")
        if reservoir_today <= 0:
            raise ValueError(
                "rational trace enclosure requires positive present reservoir"
            )
        if width <= 0 or not n_initial < n_final <= 0:
            raise ValueError("rational trace enclosure interval is invalid")

        source_minus = center - width
        source_plus = center + width
        source_intersects = (
            n_final > source_minus and n_initial < source_plus and omega > 0
        )
        source_left: Fraction | None = None
        source_right: Fraction | None = None
        source_upper = Fraction(0)
        if source_intersects:
            source_left = max(n_initial, source_minus)
            source_right = min(n_final, source_plus)
            if not source_left < source_right <= 0:
                raise ValueError("source intersection is invalid")
            _, source_exp_upper, _ = _rational_exp_bounds(-3 * source_left)
            source_upper = (
                omega
                * source_exp_upper
                * Fraction(15, 16)
                / width
            )
            enthalpy_exp_argument = -3 * (1 + w) * source_right
        else:
            enthalpy_exp_argument = -3 * (1 + w) * n_final

        enthalpy_exp_lower, _, _ = _rational_exp_bounds(
            enthalpy_exp_argument
        )
        enthalpy_lower = reservoir_today * enthalpy_exp_lower
        if enthalpy_lower <= 0:
            raise ValueError("rational enthalpy lower bound must be positive")

        pressure_ratio_upper = 3 * w
        if source_upper > 0:
            pressure_ratio_upper += w * source_upper / enthalpy_lower
        damping_lower = Fraction(5, 2)
        damping_upper = damping_lower + pressure_ratio_upper

        interval_width = n_final - n_initial
        kappa_rate_upper = Fraction(1, 2) + Fraction(3, 2) * w
        _, kappa_exp_upper, _ = _rational_exp_bounds(
            kappa_rate_upper * interval_width
        )
        kappa_upper = kappa_initial * kappa_exp_upper
        restoring_lower = Fraction(0)
        restoring_upper = 3 + pressure_ratio_upper * (
            1 + kappa_upper * kappa_upper / 3
        )
        generator_norm_upper = 1 + damping_upper + restoring_upper

        off_diagonal = max(Fraction(1), restoring_upper - 1)
        rate_upper = min(
            off_diagonal / 2,
            off_diagonal * off_diagonal / 10,
        )
        if not (
            source_upper >= 0
            and enthalpy_lower > 0
            and pressure_ratio_upper >= 0
            and kappa_upper > 0
            and damping_upper >= damping_lower
            and restoring_upper >= 0
            and generator_norm_upper > 0
            and rate_upper >= 0
        ):
            raise ValueError("rational trace coefficient bound is invalid")

        return RationalTraceCoefficientBoundReceipt(
            source_intersects_interval=source_intersects,
            source_left_endpoint=source_left,
            source_right_endpoint=source_right,
            source_upper_bound=source_upper,
            enthalpy_lower_bound=enthalpy_lower,
            pressure_ratio_upper_bound=pressure_ratio_upper,
            kappa_upper_bound=kappa_upper,
            damping_lower_bound=damping_lower,
            damping_upper_bound=damping_upper,
            restoring_lower_bound=restoring_lower,
            restoring_upper_bound=restoring_upper,
            generator_entrywise_norm_upper_bound=generator_norm_upper,
            euclidean_logarithmic_norm_rate_upper_bound=rate_upper,
            exact_binary_float_parameters_frozen=True,
            rational_taylor_exponential_enclosures_proven=True,
            weighted_source_integral_partition_count=(
                _WEIGHTED_SOURCE_SUBINTERVALS
            ),
            component_and_source_nonnegativity_used=True,
            kappa_squared_monotonicity_proven=True,
            coefficient_bounds_proven_on_full_interval=True,
        )

    def _refined_mesh(
        self,
        coarse_step_count: object,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if (
            isinstance(coarse_step_count, bool)
            or not isinstance(coarse_step_count, Integral)
        ):
            raise ValueError("coarse_step_count must be an integer")
        steps = int(coarse_step_count)
        if steps < 16:
            raise ValueError("coarse_step_count must be >= 16")
        flow = self.evolution.trace_flow_stability_bound()
        resolution_floor = math.ceil(
            flow.interval_width
            * flow.generator_bound.characteristic_rate_upper_bound
        )
        target = max(steps, resolution_floor)
        if target > _MAX_AUTOMATIC_COARSE_STEPS:
            raise ValueError(
                "trace endpoint enclosure exceeds the automatic mesh limit"
            )
        coarse = self.evolution._piecewise_coarse_mesh(target)
        refined = self.evolution._refined_mesh(coarse)
        return coarse, refined

    def _trace_nodes(
        self,
        *,
        primordial_potential_amplitude: object,
        mesh: tuple[float, ...],
    ) -> tuple[tuple[float, float], ...]:
        regular = FiniteQuenchSuperhorizonRegularity(
            self.evolution.bridge
        ).construct_regular_mode(
            n=self.evolution.n_initial,
            k_over_a_h=self.evolution.kappa_initial,
            primordial_potential_amplitude=(
                primordial_potential_amplitude
            ),
        )
        curvature = regular.series_curvature_potential
        curvature_prime = self.evolution.rhs(
            self.evolution.n_initial,
            regular.required_scalar_clock_shift,
            curvature,
        )[1]
        nodes = [(curvature, curvature_prime)]
        for n_start, n_end in zip(
            mesh[:-1],
            mesh[1:],
            strict=True,
        ):
            curvature, curvature_prime = self.evolution._magnus_step(
                n_start,
                n_end,
                curvature,
                curvature_prime,
            )
            nodes.append((curvature, curvature_prime))
        return tuple(nodes)

    def construct(
        self,
        *,
        primordial_potential_amplitude: object,
        coarse_step_count: object = 512,
    ) -> TraceEndpointEnclosureReceipt:
        """Enclose the trace endpoint around the refined linear path."""

        coefficient_bounds = self.coefficient_bounds()
        coarse, refined = self._refined_mesh(coarse_step_count)
        nodes = self._trace_nodes(
            primordial_potential_amplitude=primordial_potential_amplitude,
            mesh=refined,
        )
        if len(nodes) != len(refined):
            raise ValueError("trace node and mesh lengths do not match")

        generator_bound = (
            coefficient_bounds.generator_entrywise_norm_upper_bound
        )
        parameters = self._frozen_parameters()
        global_residual_integral = Fraction(0)
        local_residual_integral = Fraction(0)
        local_logarithmic_norm_exponent = Fraction(0)
        frozen_nodes = tuple(
            (
                _binary_fraction(curvature),
                _binary_fraction(curvature_prime),
            )
            for curvature, curvature_prime in nodes
        )
        frozen_mesh = tuple(_binary_fraction(node) for node in refined)
        exact_regular = FiniteQuenchSuperhorizonRegularity(
            self.evolution.bridge
        ).construct_exact_regular_initial_enclosure(
            n=self.evolution.n_initial,
            k_over_a_h=self.evolution.kappa_initial,
            primordial_potential_amplitude=primordial_potential_amplitude,
        )
        frozen_initial_curvature = frozen_nodes[0][0]
        frozen_initial_curvature_prime = frozen_nodes[0][1]
        curvature_initial_defect = max(
            abs(
                frozen_initial_curvature
                - exact_regular.curvature_interval[0]
            ),
            abs(
                frozen_initial_curvature
                - exact_regular.curvature_interval[1]
            ),
        )
        curvature_prime_initial_defect = max(
            abs(
                frozen_initial_curvature_prime
                - exact_regular.curvature_prime_interval[0]
            ),
            abs(
                frozen_initial_curvature_prime
                - exact_regular.curvature_prime_interval[1]
            ),
        )
        initial_l1_defect = (
            curvature_initial_defect
            + curvature_prime_initial_defect
        )
        regular_initial_bridge = RegularInitialTraceBridgeReceipt(
            regular_mode_enclosure=exact_regular,
            frozen_initial_curvature=frozen_initial_curvature,
            frozen_initial_curvature_prime=(
                frozen_initial_curvature_prime
            ),
            curvature_defect_abs_upper_bound=curvature_initial_defect,
            curvature_prime_defect_abs_upper_bound=(
                curvature_prime_initial_defect
            ),
            initial_l1_defect_upper_bound=initial_l1_defect,
            exact_component_difference_enclosures_proven=True,
            euclidean_initial_defect_bounded_by_l1=True,
            floating_series_and_rhs_roundoff_absorbed=True,
            analytic_regular_trace_initial_state_enclosed=True,
        )
        local_coefficients: list[
            tuple[_RationalInterval, _RationalInterval]
        ] = []
        interval_count = len(frozen_mesh) - 1
        for block_start in range(
            0,
            interval_count,
            _LOCAL_COEFFICIENT_BLOCK_STEPS,
        ):
            block_stop = min(
                interval_count,
                block_start + _LOCAL_COEFFICIENT_BLOCK_STEPS,
            )
            block_coefficients = (
                self._local_trace_coefficient_intervals(
                    frozen_mesh[block_start],
                    frozen_mesh[block_stop],
                    parameters,
                    coefficient_bounds,
                )
            )
            local_coefficients.extend(
                block_coefficients
                for _ in range(block_start, block_stop)
            )
        if len(local_coefficients) != interval_count:
            raise ValueError("local coefficient block count is inconsistent")

        for left_n, right_n, left_y, right_y, coefficients in zip(
            frozen_mesh[:-1],
            frozen_mesh[1:],
            frozen_nodes[:-1],
            frozen_nodes[1:],
            local_coefficients,
            strict=True,
        ):
            step = right_n - left_n
            if step <= 0:
                raise ValueError("frozen refined mesh is not strictly increasing")
            delta_l1 = (
                abs(right_y[0] - left_y[0])
                + abs(right_y[1] - left_y[1])
            )
            state_l1 = max(
                abs(left_y[0]) + abs(left_y[1]),
                abs(right_y[0]) + abs(right_y[1]),
            )
            global_residual_integral += (
                delta_l1 + step * generator_bound * state_l1
            )

            damping, restoring = coefficients
            off_diagonal = max(
                abs(1 - restoring.lower),
                abs(1 - restoring.upper),
            )
            local_rate_upper = min(
                off_diagonal / 2,
                off_diagonal
                * off_diagonal
                / (4 * damping.lower),
            )
            local_logarithmic_norm_exponent += (
                step * local_rate_upper
            )
            curvature = _RationalInterval(
                min(left_y[0], right_y[0]),
                max(left_y[0], right_y[0]),
            )
            curvature_prime = _RationalInterval(
                min(left_y[1], right_y[1]),
                max(left_y[1], right_y[1]),
            )
            slope_curvature = (
                right_y[0] - left_y[0]
            ) / step
            slope_curvature_prime = (
                right_y[1] - left_y[1]
            ) / step
            first_residual = _interval_subtract(
                _point_interval(slope_curvature),
                curvature_prime,
            )
            second_residual = _interval_add(
                _point_interval(slope_curvature_prime),
                _interval_add(
                    _interval_multiply(restoring, curvature),
                    _interval_multiply(damping, curvature_prime),
                ),
            )
            local_residual_integral += step * (
                _interval_abs_upper(first_residual)
                + _interval_abs_upper(second_residual)
            )

        residual_integral = min(
            global_residual_integral,
            local_residual_integral,
        )

        interval_width = frozen_mesh[-1] - frozen_mesh[0]
        global_propagation_exponent = (
            coefficient_bounds.euclidean_logarithmic_norm_rate_upper_bound
            * interval_width
        )
        propagation_exponent = min(
            global_propagation_exponent,
            local_logarithmic_norm_exponent,
        )
        endpoint_radius = ExactExponentialRadius(
            coefficient=residual_integral,
            exponent=propagation_exponent,
            coefficient_nonnegative=residual_integral >= 0,
            exponent_nonnegative=propagation_exponent >= 0,
        )
        # The local logarithmic norm above is the unweighted Euclidean
        # (P = I) norm, and ||delta||_2 <= ||delta||_1 componentwise.
        analytic_radius_coefficient = (
            residual_integral + initial_l1_defect
        )
        analytic_endpoint_radius = ExactExponentialRadius(
            coefficient=analytic_radius_coefficient,
            exponent=propagation_exponent,
            coefficient_nonnegative=analytic_radius_coefficient >= 0,
            exponent_nonnegative=propagation_exponent >= 0,
        )
        (
            materialized_exp_argument,
            materialized_radius,
        ) = _materialize_exponential_radius(
            residual_integral,
            propagation_exponent,
        )
        (
            analytic_materialized_exp_argument,
            analytic_materialized_radius,
        ) = _materialize_exponential_radius(
            analytic_radius_coefficient,
            propagation_exponent,
        )
        float_radius = (
            None
            if materialized_radius is None
            else _fraction_to_float_upper(materialized_radius)
        )
        analytic_float_radius = (
            None
            if analytic_materialized_radius is None
            else _fraction_to_float_upper(analytic_materialized_radius)
        )

        endpoint_curvature = frozen_nodes[-1][0]
        endpoint_curvature_prime = frozen_nodes[-1][1]
        curvature_interval = _component_interval(
            endpoint_curvature,
            materialized_radius,
        )
        curvature_prime_interval = _component_interval(
            endpoint_curvature_prime,
            materialized_radius,
        )
        analytic_curvature_interval = _component_interval(
            endpoint_curvature,
            analytic_materialized_radius,
        )
        analytic_curvature_prime_interval = _component_interval(
            endpoint_curvature_prime,
            analytic_materialized_radius,
        )
        curvature_sign = _certified_component_sign(curvature_interval)
        curvature_prime_sign = _certified_component_sign(
            curvature_prime_interval
        )
        analytic_curvature_sign = _certified_component_sign(
            analytic_curvature_interval
        )
        analytic_curvature_prime_sign = _certified_component_sign(
            analytic_curvature_prime_interval
        )
        scalar_clock_endpoint = self._scalar_clock_endpoint_receipt(
            parameters=parameters,
            endpoint_curvature=endpoint_curvature,
            endpoint_curvature_prime=endpoint_curvature_prime,
            analytic_curvature_interval=analytic_curvature_interval,
            analytic_curvature_prime_interval=(
                analytic_curvature_prime_interval
            ),
        )
        amplitude_normalized_response = (
            self._amplitude_normalized_response_receipt(
                amplitude=exact_regular.primordial_potential_amplitude,
                analytic_curvature_interval=analytic_curvature_interval,
                analytic_curvature_prime_interval=(
                    analytic_curvature_prime_interval
                ),
                scalar_clock_interval=(
                    scalar_clock_endpoint.scalar_clock_interval
                ),
            )
        )
        conditional_weyl_metric_endpoint = (
            self._conditional_weyl_metric_endpoint_receipt(
                analytic_curvature_interval=analytic_curvature_interval,
                amplitude_response=amplitude_normalized_response,
            )
        )
        uniform_trace_path_tube = self._uniform_trace_path_tube_receipt(
            frozen_mesh=frozen_mesh,
            frozen_nodes=frozen_nodes,
            frozen_symbolic_radius=endpoint_radius,
            analytic_symbolic_radius=analytic_endpoint_radius,
            frozen_materialized_radius=materialized_radius,
            analytic_materialized_radius=analytic_materialized_radius,
            amplitude=exact_regular.primordial_potential_amplitude,
        )
        background_conformal_metric_time_integral = (
            self._background_conformal_metric_time_integral_receipt(
                frozen_mesh=frozen_mesh,
                frozen_nodes=frozen_nodes,
                parameters=parameters,
                analytic_symbolic_radius=analytic_endpoint_radius,
                analytic_materialized_radius=analytic_materialized_radius,
                amplitude=exact_regular.primordial_potential_amplitude,
            )
        )
        fixed_mode_born_lensing_absolute_envelope = (
            self._fixed_mode_born_lensing_absolute_envelope_receipt(
                frozen_mesh=frozen_mesh,
                frozen_nodes=frozen_nodes,
                parameters=parameters,
                analytic_materialized_radius=analytic_materialized_radius,
                amplitude=exact_regular.primordial_potential_amplitude,
            )
        )

        return TraceEndpointEnclosureReceipt(
            coefficient_bounds=coefficient_bounds,
            n_initial=frozen_mesh[0],
            n_final=frozen_mesh[-1],
            requested_coarse_step_count=int(coarse_step_count),
            coarse_step_count=len(coarse) - 1,
            refined_step_count=len(refined) - 1,
            local_coefficient_block_step_count=(
                _LOCAL_COEFFICIENT_BLOCK_STEPS
            ),
            endpoint_curvature_center=endpoint_curvature,
            endpoint_curvature_prime_center=endpoint_curvature_prime,
            global_triangle_residual_l1_integral_upper_bound=(
                global_residual_integral
            ),
            local_interval_residual_l1_integral_upper_bound=(
                local_residual_integral
            ),
            residual_l1_integral_upper_bound=residual_integral,
            global_propagation_exponent_upper_bound=(
                global_propagation_exponent
            ),
            local_integrated_logarithmic_norm_exponent_upper_bound=(
                local_logarithmic_norm_exponent
            ),
            propagation_exponent_upper_bound=propagation_exponent,
            endpoint_radius=endpoint_radius,
            materialized_exponential_argument_upper_bound=(
                materialized_exp_argument
            ),
            materialized_rational_radius_upper_bound=materialized_radius,
            materialized_float_radius_upper_bound=float_radius,
            curvature_component_interval=curvature_interval,
            curvature_prime_component_interval=curvature_prime_interval,
            curvature_component_certified_sign=curvature_sign,
            curvature_prime_component_certified_sign=(
                curvature_prime_sign
            ),
            regular_initial_bridge=regular_initial_bridge,
            analytic_regular_endpoint_radius=analytic_endpoint_radius,
            analytic_regular_materialized_exponential_argument_upper_bound=(
                analytic_materialized_exp_argument
            ),
            analytic_regular_materialized_rational_radius_upper_bound=(
                analytic_materialized_radius
            ),
            analytic_regular_materialized_float_radius_upper_bound=(
                analytic_float_radius
            ),
            analytic_regular_curvature_component_interval=(
                analytic_curvature_interval
            ),
            analytic_regular_curvature_prime_component_interval=(
                analytic_curvature_prime_interval
            ),
            analytic_regular_curvature_component_certified_sign=(
                analytic_curvature_sign
            ),
            analytic_regular_curvature_prime_component_certified_sign=(
                analytic_curvature_prime_sign
            ),
            scalar_clock_endpoint=scalar_clock_endpoint,
            amplitude_normalized_response=amplitude_normalized_response,
            conditional_weyl_metric_endpoint=(
                conditional_weyl_metric_endpoint
            ),
            uniform_trace_path_tube_and_efold_integral=(
                uniform_trace_path_tube
            ),
            background_conformal_metric_time_integral=(
                background_conformal_metric_time_integral
            ),
            fixed_mode_born_lensing_absolute_envelope=(
                fixed_mode_born_lensing_absolute_envelope
            ),
            refined_nodes_frozen_as_exact_binary_rationals=True,
            continuous_piecewise_linear_reconstruction_proven=True,
            piecewise_join_defect_zero_proven=True,
            local_coefficient_interval_enclosures_proven=True,
            local_residual_cancellation_retained=True,
            local_time_dependent_logarithmic_norm_integrated=True,
            residual_integral_bound_proven_by_exact_rational_arithmetic=True,
            logarithmic_norm_propagation_bound_proven=True,
            exact_symbolic_trace_endpoint_ball_proven=True,
            rigorous_materialized_trace_endpoint_enclosure_proven=(
                materialized_radius is not None
            ),
            exact_symbolic_analytic_regular_endpoint_ball_proven=True,
            rigorous_materialized_analytic_regular_endpoint_enclosure_proven=(
                analytic_materialized_radius is not None
            ),
            analytic_source_off_regular_initial_condition_enclosed=True,
            conditional_amplitude_normalized_response_enclosed=(
                amplitude_normalized_response
                .fixed_amplitude_conditional_response_enclosed
            ),
            conditional_weyl_metric_endpoint_enclosed=(
                conditional_weyl_metric_endpoint
                .conditional_metric_potential_endpoint_enclosed
            ),
            analytic_regular_uniform_trace_path_tube_proven=(
                uniform_trace_path_tube
                .uniform_trace_state_tube_covers_every_prefix
            ),
            background_conformal_metric_time_integral_enclosed=(
                background_conformal_metric_time_integral
                .materialized_analytic_regular_metric_time_integral_enclosed
            ),
            conditional_fixed_mode_born_lensing_absolute_envelope_enclosed=(
                fixed_mode_born_lensing_absolute_envelope
                .conditional_single_mode_born_convergence_absolute_envelope_enclosed
            ),
            numerical_node_roundoff_absorbed_into_frozen_path=True,
            normalized_dimensionless_model_assumed=True,
            scalar_clock_endpoint_enclosed=(
                scalar_clock_endpoint.scalar_clock_endpoint_enclosed
            ),
        )
