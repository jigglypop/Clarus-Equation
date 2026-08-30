"""Finite-interval evolution in the pole-free ``(T,psi)`` variables.

For the complete strict common-clock branch, not only in the early pure-fluid
era, the Einstein constraints imply

    U = K psi/(3C) - W T,

and the exact reduced system becomes

    T'   = (h+K/3) T + (1+K/3+K^2/(9h)) psi,
    psi' = -h T - (1+K/3) psi.

Here ``K=kappa^2`` and ``h<0`` on the causal two-fluid branch.  This basis has
no inverse power of K and is continuous across the compact source edges.

The module evolves the unique past-bounded regular mode with a fourth-order
Magnus step for the equivalent trace-conditioned ``(psi, psi')`` system and
checks step doubling.  The mesh contains both source edges, resolves the
normalized compact-source shape, and uses analytic coefficient bounds for a
conservative characteristic-rate floor.  Exponentiating the frozen linear
generator removes the explicit-RK stability failure when a causal ``w=1``
reservoir drives ``kappa`` deep inside the horizon.  This supplies a
reproducible finite-time numerical trajectory through the source, but not a
rigorous interval enclosure or a general numerical-stability theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
)
from examples.physics.finite_quench_reduced_ode_closure import (
    FiniteQuenchReducedODEClosure,
    ReducedODEClosureReceipt,
)
from examples.physics.finite_quench_superhorizon_regularity import (
    FiniteQuenchSuperhorizonRegularity,
    SuperhorizonRegularModeReceipt,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
)

_MAX_AUTOMATIC_COARSE_STEPS = 200_000
_MINIMUM_ACTIVE_SOURCE_COARSE_STEPS = 16
_TRACE_DAMPING_LOWER_BOUND = 2.5


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _finite_sum(name: str, *values: float) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} left the finite domain") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} left the finite domain")
    return result


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 256.0 * math.ulp(scale)


@dataclass(frozen=True)
class TraceGeneratorBoundReceipt:
    """Analytic coefficient bounds used by the mesh-resolution heuristic."""

    n_initial: float
    n_final: float
    source_intersects_interval: bool
    source_right_endpoint: float | None
    source_enthalpy_lower_bound: float | None
    source_upper_bound: float
    pressure_ratio_upper_bound: float
    kappa_upper_bound: float
    damping_upper_bound: float
    restoring_absolute_upper_bound: float
    characteristic_rate_upper_bound: float
    minimum_active_source_coarse_steps: int
    component_density_nonnegativity_derived_from_bridge: bool
    source_bound_derived_analytically: bool
    enthalpy_monotonicity_proven: bool
    kappa_monotonicity_proven: bool
    trace_coefficients_bounded_on_interval: bool
    mesh_rule_is_stability_or_error_theorem: bool = False
    source_shape_step_floor_is_error_theorem: bool = False
    role: str = (
        "ANALYTIC_TRACE_COEFFICIENT_BOUND_WITH_CHARACTERISTIC_RESOLUTION_"
        "HEURISTIC_NOT_STABILITY_OR_ERROR_THEOREM"
    )


@dataclass(frozen=True)
class TraceFlowStabilityReceipt:
    """Exact-flow bounds under the validated flat-GR bridge assumptions.

    The coefficient signs use the background equations together with
    nonnegative component densities and source, and 0 <= w <= 1.
    """

    generator_bound: TraceGeneratorBoundReceipt
    n_initial: float
    n_final: float
    interval_width: float
    damping_lower_bound: float
    restoring_lower_bound: float
    wronskian_log_ratio_lower_bound: float
    wronskian_log_ratio_upper_bound: float
    wronskian_contraction_factor_upper_bound: float | None
    wronskian_contraction_factor_representable: bool
    direct_euclidean_weight: float
    direct_euclidean_logarithmic_norm_rate_upper_bound: float
    direct_euclidean_log_amplification_upper_bound: float
    balanced_weight: float
    balanced_logarithmic_norm_rate_upper_bound: float
    balanced_log_amplification_upper_bound: float
    selected_conservative_weight: float
    selected_euclidean_conversion_log_penalty: float
    selected_logarithmic_norm_rate_upper_bound: float
    selected_log_amplification_upper_bound: float
    coefficient_signs_derived_analytically: bool
    frozen_generator_has_no_positive_real_eigenvalue: bool
    wronskian_identity_proven: bool
    fundamental_matrix_invertibility_proven: bool
    forward_phase_area_contraction_proven: bool
    finite_interval_continuous_dependence_bound_proven: bool
    individual_solution_norm_monotone_decay_proven: bool = False
    no_transient_growth_proven: bool = False
    numerical_method_stability_theorem_proven: bool = False
    rigorous_interval_enclosure_proven: bool = False
    role: str = (
        "CONDITIONAL_TRACE_SIGN_WRONSKIAN_AND_WEIGHTED_PROPAGATOR_BOUND_"
        "NOT_INDIVIDUAL_DECAY_TRANSIENT_GROWTH_OR_NUMERICAL_STABILITY_PROOF"
    )


@dataclass(frozen=True)
class TraceResidualErrorBoundReceipt:
    """Conditional Duhamel endpoint-error theorem in a fixed P norm."""

    flow_stability: TraceFlowStabilityReceipt
    weight_p: float
    metric_logarithmic_norm_rate_upper_bound: float
    metric_log_propagator_upper_bound: float
    metric_to_euclidean_log_factor: float
    assumed_initial_defect_p_upper_bound: float
    assumed_terminal_weighted_residual_p_upper_bound: float
    metric_endpoint_error_log_upper_bound: float | None
    euclidean_endpoint_error_log_upper_bound: float | None
    metric_endpoint_error_upper_bound: float | None
    euclidean_endpoint_error_upper_bound: float | None
    endpoint_error_exactly_zero_under_assumptions: bool
    endpoint_error_radius_representable: bool
    dimensionless_contract_assumed_by_normalized_system: bool
    duhamel_identity_proven: bool
    fixed_weight_metric_error_bound_proven: bool
    conditional_a_posteriori_error_bound_proven: bool
    approximate_path_absolute_continuity_verified_by_module: bool = False
    dense_output_residual_certified_by_module: bool = False
    initial_defect_certified_by_module: bool = False
    piecewise_join_defects_included_by_module: bool = False
    coefficient_interval_enclosure_proven: bool = False
    outward_rounding_proven: bool = False
    floating_point_evaluation_is_rigorous: bool = False
    rigorous_interval_enclosure_proven: bool = False
    role: str = (
        "CONDITIONAL_DUHAMEL_A_POSTERIORI_ERROR_THEOREM_"
        "EXTERNAL_P_NORM_DEFECT_AND_RESIDUAL_BOUNDS_REQUIRED_"
        "NOT_A_RIGOROUS_INTERVAL_ENCLOSURE"
    )


@dataclass(frozen=True)
class RegularMetricEvolutionReceipt:
    """Step-doubled finite-interval evolution and final reconstruction."""

    regular_initial_mode: SuperhorizonRegularModeReceipt
    final_reduced_ode: ReducedODEClosureReceipt
    trace_generator_bound: TraceGeneratorBoundReceipt
    trace_flow_stability: TraceFlowStabilityReceipt
    n_initial: float
    n_final: float
    kappa_initial: float
    kappa_final: float
    primordial_potential_amplitude: float
    requested_coarse_step_count: int
    coarse_step_count: int
    refined_step_count: int
    maximum_coarse_step: float
    maximum_final_phase_step: float
    maximum_characteristic_scale_step: float
    active_source_coarse_step_count: int
    relative_tolerance: float
    coarse_final_scalar_clock_shift: float
    refined_final_scalar_clock_shift: float
    coarse_final_curvature_potential: float
    refined_final_curvature_potential: float
    scalar_clock_richardson_error_estimate: float
    curvature_richardson_error_estimate: float
    scalar_clock_relative_error_estimate: float
    curvature_relative_error_estimate: float
    refined_final_total_momentum_density: float
    final_regular_clock_rhs: float
    final_regular_curvature_rhs: float
    final_full_clock_rhs: float
    final_full_curvature_rhs: float
    final_clock_rhs_residual: float
    final_curvature_rhs_residual: float
    max_abs_refined_scalar_clock_shift: float
    max_abs_refined_curvature_potential: float
    curvature_transfer_per_unit_initial_amplitude: float | None
    initial_regular_mode_holds: bool
    regular_metric_coefficients_continuous_on_domain: bool
    source_support_was_traversed: bool
    source_edges_aligned_in_coarse_mesh: bool
    analytic_resolution_bound_holds: bool
    normalized_source_shape_resolution_holds: bool
    magnus_step_doubling_converged: bool
    final_effective_full_reconstruction_holds: bool
    final_regular_rhs_matches_full_system: bool
    finite_time_source_on_evolution_numerically_verified: bool
    failure_reasons: tuple[str, ...]
    rigorous_interval_enclosure_proven: bool = False
    numerical_method_stability_theorem_proven: bool = False
    microphysical_covariant_transfer_law_proven: bool = False
    primordial_amplitude_predicted: bool = False
    observable_transfer_function_proven: bool = False
    role: str = (
        "CONDITIONAL_SOURCE_ALIGNED_MAGNUS_STEP_DOUBLED_FINITE_TIME_"
        "REGULAR_METRIC_EVOLUTION_NOT_INTERVAL_MICROPHYSICAL_PRIMORDIAL_"
        "OR_OBSERVABLE_PROOF"
    )


class FiniteQuenchRegularMetricEvolution:
    """Evolve the past-bounded mode across the compact source in (T,psi)."""

    def __init__(
        self,
        bridge: FiniteQuenchBridge,
        *,
        n_initial: object,
        kappa_initial: object,
        n_final: object = 0.0,
    ) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge
        self.n_initial = _finite_real(n_initial, "n_initial")
        self.n_final = _finite_real(n_final, "n_final")
        self.kappa_initial = _finite_real(kappa_initial, "kappa_initial")
        if not bridge.config.n_initial <= self.n_initial <= bridge.config.n_minus:
            raise ValueError("evolution must start in the pre-source era")
        if not self.n_initial < self.n_final <= 0.0:
            raise ValueError("n_final must be after n_initial and no later than today")
        if not 0.0 < self.kappa_initial <= 0.1:
            raise ValueError("regular initial mode requires 0 < kappa_initial <= 0.1")
        self.reduced = FiniteQuenchReducedODEClosure(
            bridge,
            n_reference=self.n_initial,
            kappa_reference=self.kappa_initial,
        )

    def regular_matrix(self, n: object) -> tuple[float, float, float, float]:
        """Return the exact pole-free matrix for ``(T,psi)``."""

        n_value = _finite_real(n, "n")
        if not self.n_initial <= n_value <= self.n_final:
            raise ValueError("n is outside the evolution interval")
        background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).construct(n_value)
        h = background.hubble_log_derivative
        if h >= 0.0:
            raise ValueError("regular metric basis requires h < 0")
        kappa = self.reduced.k_over_a_h(n_value)
        kappa_squared = kappa * kappa
        b11 = h + kappa_squared / 3.0
        b12 = (
            1.0
            + kappa_squared / 3.0
            + kappa_squared * kappa_squared / (9.0 * h)
        )
        b21 = -h
        b22 = -(1.0 + kappa_squared / 3.0)
        if any(not math.isfinite(value) for value in (b11, b12, b21, b22)):
            raise ValueError("regular metric matrix left the finite domain")
        return b11, b12, b21, b22

    def rhs(self, n: object, clock: object, curvature: object) -> tuple[float, float]:
        """Evaluate the exact regular vector field."""

        n_value = _finite_real(n, "n")
        t_value = _finite_real(clock, "clock")
        psi_value = _finite_real(curvature, "curvature")
        b11, b12, b21, b22 = self.regular_matrix(n_value)
        return (
            _finite_sum(
                "regular metric clock RHS",
                b11 * t_value,
                b12 * psi_value,
            ),
            _finite_sum(
                "regular metric curvature RHS",
                b21 * t_value,
                b22 * psi_value,
            ),
        )

    def trace_conditioned_matrix(
        self,
        n: object,
    ) -> tuple[float, float, float, float]:
        """Return the equivalent ``(psi, psi')`` generator.

        The strict trace equation and common clock give

            psi'' + A psi' + B psi = 0,

        with ``A=4+h+C P'/h`` and
        ``B=3+2h+(C P'/h)(1+K/3)``.  Unlike the direct ``(T,psi)``
        representation, this generator contains no ``K^2`` coefficient.
        """

        n_value = _finite_real(n, "n")
        if not self.n_initial <= n_value <= self.n_final:
            raise ValueError("n is outside the evolution interval")
        background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).construct(n_value)
        h = background.hubble_log_derivative
        if h >= 0.0:
            raise ValueError("trace-conditioned basis requires h < 0")
        coupling = background.gravity_constraint_coupling
        pressure_prime = (
            self.bridge.config.w_reservoir
            * background.reservoir_density_derivative
        )
        pressure_ratio = coupling * pressure_prime / h
        kappa = self.reduced.k_over_a_h(n_value)
        kappa_squared = kappa * kappa
        damping = _finite_sum("trace damping", 4.0, h, pressure_ratio)
        restoring = _finite_sum(
            "trace restoring coefficient",
            3.0,
            2.0 * h,
            pressure_ratio * (1.0 + kappa_squared / 3.0),
        )
        matrix = (0.0, 1.0, -restoring, -damping)
        if any(not math.isfinite(value) for value in matrix):
            raise ValueError("trace-conditioned matrix left the finite domain")
        return matrix

    def trace_generator_bound(self) -> TraceGeneratorBoundReceipt:
        """Return analytic bounds for the trace-conditioned coefficients.

        The proof uses C/h = -1/W, monotone total enthalpy W, the exact
        compact-source maximum, and monotone kappa on 0 <= w_R <= 1.  The
        resulting characteristic rate is only a conservative mesh-resolution
        scale; it is not an operator-norm or global-error theorem.
        """

        config = self.bridge.config
        w = config.w_reservoir
        if not 0.0 <= w <= 1.0:
            raise ValueError(
                "analytic trace bound requires the strict 0 <= w_R <= 1 branch"
            )
        component_nonnegativity = (
            config.omega_prod0 >= 0.0
            and config.reservoir_present_density >= 0.0
        )
        if not component_nonnegativity:
            raise ValueError(
                "analytic trace bound requires non-negative component inputs"
            )
        source_intersects = (
            self.n_final > config.n_minus
            and self.n_initial < config.n_plus
        )
        source_right_endpoint = (
            min(self.n_final, config.n_plus)
            if source_intersects
            else None
        )
        source_enthalpy_lower_bound: float | None = None
        if source_right_endpoint is not None:
            source_background = FiniteQuenchTwoFluidFlatGRBackground(
                self.bridge
            ).construct(source_right_endpoint)
            source_enthalpy_lower_bound = source_background.total_enthalpy
            if source_enthalpy_lower_bound <= 0.0:
                raise ValueError(
                    "source interval requires a positive enthalpy lower bound"
                )

        if config.omega_prod0 == 0.0 or not source_intersects:
            source_upper_bound = 0.0
        else:
            try:
                source_upper_bound = (
                    config.omega_prod0
                    * math.exp(-3.0 * config.n_minus)
                    * 15.0
                    / (16.0 * config.half_width)
                )
            except OverflowError as error:
                raise ValueError(
                    "analytic source upper bound left the finite domain"
                ) from error
            if not math.isfinite(source_upper_bound):
                raise ValueError(
                    "analytic source upper bound left the finite domain"
                )

        source_pressure_term = (
            0.0
            if source_enthalpy_lower_bound is None
            else w * source_upper_bound / source_enthalpy_lower_bound
        )
        pressure_ratio_upper_bound = _finite_sum(
            "pressure ratio upper bound",
            3.0 * w,
            source_pressure_term,
        )
        kappa_upper_bound = self.reduced.k_over_a_h(self.n_final)
        damping_upper_bound = _finite_sum(
            "trace damping upper bound",
            2.5,
            pressure_ratio_upper_bound,
        )
        kappa_squared = kappa_upper_bound * kappa_upper_bound
        restoring_absolute_upper_bound = _finite_sum(
            "trace restoring absolute upper bound",
            3.0,
            pressure_ratio_upper_bound * (1.0 + kappa_squared / 3.0),
        )
        spectral_discriminant_bound = math.hypot(
            damping_upper_bound,
            2.0 * math.sqrt(restoring_absolute_upper_bound),
        )
        characteristic_rate_upper_bound = 0.5 * _finite_sum(
            "trace characteristic rate upper bound",
            damping_upper_bound,
            spectral_discriminant_bound,
        )
        if characteristic_rate_upper_bound <= 0.0:
            raise ValueError("trace characteristic rate bound must be positive")

        return TraceGeneratorBoundReceipt(
            n_initial=self.n_initial,
            n_final=self.n_final,
            source_intersects_interval=source_intersects,
            source_right_endpoint=source_right_endpoint,
            source_enthalpy_lower_bound=source_enthalpy_lower_bound,
            source_upper_bound=source_upper_bound,
            pressure_ratio_upper_bound=pressure_ratio_upper_bound,
            kappa_upper_bound=kappa_upper_bound,
            damping_upper_bound=damping_upper_bound,
            restoring_absolute_upper_bound=restoring_absolute_upper_bound,
            characteristic_rate_upper_bound=characteristic_rate_upper_bound,
            minimum_active_source_coarse_steps=(
                _MINIMUM_ACTIVE_SOURCE_COARSE_STEPS
            ),
            component_density_nonnegativity_derived_from_bridge=(
                component_nonnegativity
            ),
            source_bound_derived_analytically=True,
            enthalpy_monotonicity_proven=True,
            kappa_monotonicity_proven=True,
            trace_coefficients_bounded_on_interval=True,
        )

    @staticmethod
    def _representable_wronskian_contraction_factor(
        log_upper_bound: float,
    ) -> float | None:
        """Exponentiate a nonpositive log bound without returning false zero."""

        value = _finite_real(log_upper_bound, "Wronskian log upper bound")
        if value > 0.0:
            raise ValueError("Wronskian contraction log bound must be <= 0")
        return FiniteQuenchRegularMetricEvolution._positive_exp_or_none(value)

    @staticmethod
    def _positive_exp_or_none(log_value: float) -> float | None:
        """Return a positive finite exponential, or None outside float range."""

        value = _finite_real(log_value, "positive exponential log value")
        try:
            result = math.exp(value)
        except OverflowError:
            return None
        if result <= 0.0 or not math.isfinite(result):
            return None
        return result

    def trace_flow_stability_bound(self) -> TraceFlowStabilityReceipt:
        """Return proof-safe exact-flow bounds, not a numerical error bound.

        With total density E, enthalpy W, reservoir density rho_R, produced
        density rho_p, source q, and 0 <= w <= 1, the background equations give

          A - 5/2 =
              3 w rho_R ((1+2w) rho_p + (1+w) rho_R) / (2 E W)
              + w q / W,
          B = 3 w^2 rho_p rho_R / (E W)
              + w q / W + (kappa^2 / 3) R.

        Every term is nonnegative on the validated branch.
        """

        generator_bound = self.trace_generator_bound()
        interval_width = self.n_final - self.n_initial
        damping_lower_bound = _TRACE_DAMPING_LOWER_BOUND
        restoring_lower_bound = 0.0
        restoring_upper_bound = (
            generator_bound.restoring_absolute_upper_bound
        )
        if restoring_upper_bound <= 0.0:
            raise ValueError(
                "trace flow bound requires a positive restoring upper bound"
            )

        wronskian_log_ratio_lower_bound = (
            -generator_bound.damping_upper_bound * interval_width
        )
        wronskian_log_ratio_upper_bound = (
            -damping_lower_bound * interval_width
        )
        wronskian_contraction_factor_upper_bound = (
            self._representable_wronskian_contraction_factor(
                wronskian_log_ratio_upper_bound
            )
        )

        direct_weight = 1.0
        direct_off_diagonal_bound = max(
            direct_weight,
            restoring_upper_bound - direct_weight,
        )
        direct_rate = 0.5 * _finite_sum(
            "direct logarithmic norm rate upper bound",
            -damping_lower_bound,
            math.hypot(
                damping_lower_bound,
                direct_off_diagonal_bound,
            ),
        )
        direct_log_amplification = direct_rate * interval_width

        balanced_weight = 0.5 * restoring_upper_bound
        balanced_off_diagonal_bound = math.sqrt(balanced_weight)
        balanced_rate = 0.5 * _finite_sum(
            "balanced logarithmic norm rate upper bound",
            -damping_lower_bound,
            math.hypot(
                damping_lower_bound,
                balanced_off_diagonal_bound,
            ),
        )
        direct_conversion_log_penalty = 0.0
        balanced_conversion_log_penalty = 0.5 * abs(
            math.log(balanced_weight)
        )
        balanced_log_amplification = _finite_sum(
            "balanced log amplification upper bound",
            balanced_conversion_log_penalty,
            balanced_rate * interval_width,
        )

        if balanced_log_amplification < direct_log_amplification:
            selected_weight = balanced_weight
            selected_conversion_log_penalty = (
                balanced_conversion_log_penalty
            )
            selected_rate = balanced_rate
            selected_log_amplification = balanced_log_amplification
        else:
            selected_weight = direct_weight
            selected_conversion_log_penalty = direct_conversion_log_penalty
            selected_rate = direct_rate
            selected_log_amplification = direct_log_amplification
        finite_values = (
            wronskian_log_ratio_lower_bound,
            wronskian_log_ratio_upper_bound,
            direct_rate,
            direct_log_amplification,
            balanced_weight,
            balanced_rate,
            balanced_log_amplification,
            selected_weight,
            selected_conversion_log_penalty,
            selected_rate,
            selected_log_amplification,
        )
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError("trace flow stability bound left the finite domain")
        if (
            wronskian_contraction_factor_upper_bound is not None
            and (
                not math.isfinite(wronskian_contraction_factor_upper_bound)
                or wronskian_contraction_factor_upper_bound <= 0.0
            )
        ):
            raise ValueError(
                "representable Wronskian contraction factor must be positive"
            )

        return TraceFlowStabilityReceipt(
            generator_bound=generator_bound,
            n_initial=self.n_initial,
            n_final=self.n_final,
            interval_width=interval_width,
            damping_lower_bound=damping_lower_bound,
            restoring_lower_bound=restoring_lower_bound,
            wronskian_log_ratio_lower_bound=(
                wronskian_log_ratio_lower_bound
            ),
            wronskian_log_ratio_upper_bound=(
                wronskian_log_ratio_upper_bound
            ),
            wronskian_contraction_factor_upper_bound=(
                wronskian_contraction_factor_upper_bound
            ),
            wronskian_contraction_factor_representable=(
                wronskian_contraction_factor_upper_bound is not None
            ),
            direct_euclidean_weight=direct_weight,
            direct_euclidean_logarithmic_norm_rate_upper_bound=direct_rate,
            direct_euclidean_log_amplification_upper_bound=(
                direct_log_amplification
            ),
            balanced_weight=balanced_weight,
            balanced_logarithmic_norm_rate_upper_bound=balanced_rate,
            balanced_log_amplification_upper_bound=(
                balanced_log_amplification
            ),
            selected_conservative_weight=selected_weight,
            selected_euclidean_conversion_log_penalty=(
                selected_conversion_log_penalty
            ),
            selected_logarithmic_norm_rate_upper_bound=selected_rate,
            selected_log_amplification_upper_bound=(
                selected_log_amplification
            ),
            coefficient_signs_derived_analytically=True,
            frozen_generator_has_no_positive_real_eigenvalue=True,
            wronskian_identity_proven=True,
            fundamental_matrix_invertibility_proven=True,
            forward_phase_area_contraction_proven=True,
            finite_interval_continuous_dependence_bound_proven=True,
        )

    def trace_residual_error_bound(
        self,
        *,
        initial_defect_p_upper_bound: object,
        terminal_weighted_residual_p_upper_bound: object,
    ) -> TraceResidualErrorBoundReceipt:
        """Apply Duhamel to assumed fixed-P defect and residual bounds.

        For r = y_h' - M y_h and e = y - y_h,

          e(T) = Phi(T,n0)e(n0) - integral Phi(T,s)r(s) ds.

        The residual input is assumed to bound

          integral exp(mu_p (T-s)) ||r(s)||_P ds.

        This method proves the implication but does not certify either input
        or the floating-point evaluation of the reported radius.
        """

        initial_defect = _finite_real(
            initial_defect_p_upper_bound,
            "initial P-norm defect upper bound",
        )
        weighted_residual = _finite_real(
            terminal_weighted_residual_p_upper_bound,
            "terminal-weighted P-norm residual upper bound",
        )
        if initial_defect < 0.0:
            raise ValueError("initial P-norm defect upper bound must be >= 0")
        if weighted_residual < 0.0:
            raise ValueError(
                "terminal-weighted P-norm residual upper bound must be >= 0"
            )

        flow = self.trace_flow_stability_bound()
        weight = flow.selected_conservative_weight
        rate = flow.selected_logarithmic_norm_rate_upper_bound
        metric_log_propagator = rate * flow.interval_width
        metric_to_euclidean_log_factor = -0.5 * math.log(
            min(weight, 1.0)
        )
        finite_values = (
            weight,
            rate,
            metric_log_propagator,
            metric_to_euclidean_log_factor,
        )
        if (
            weight <= 0.0
            or rate < 0.0
            or any(not math.isfinite(value) for value in finite_values)
        ):
            raise ValueError("Duhamel metric bound left the finite domain")

        log_terms: list[float] = []
        if initial_defect > 0.0:
            log_terms.append(
                _finite_sum(
                    "propagated initial P-norm defect log bound",
                    metric_log_propagator,
                    math.log(initial_defect),
                )
            )
        if weighted_residual > 0.0:
            log_terms.append(math.log(weighted_residual))

        if not log_terms:
            metric_log_error: float | None = None
            euclidean_log_error: float | None = None
            metric_error: float | None = 0.0
            euclidean_error: float | None = 0.0
            exactly_zero = True
            radius_representable = True
        else:
            high = max(log_terms)
            scaled_sum = sum(math.exp(term - high) for term in log_terms)
            metric_log_error = _finite_sum(
                "metric endpoint error log upper bound",
                high,
                math.log(scaled_sum),
            )
            euclidean_log_error = _finite_sum(
                "Euclidean endpoint error log upper bound",
                metric_log_error,
                metric_to_euclidean_log_factor,
            )
            metric_error = self._positive_exp_or_none(metric_log_error)
            euclidean_error = self._positive_exp_or_none(
                euclidean_log_error
            )
            exactly_zero = False
            radius_representable = euclidean_error is not None

        return TraceResidualErrorBoundReceipt(
            flow_stability=flow,
            weight_p=weight,
            metric_logarithmic_norm_rate_upper_bound=rate,
            metric_log_propagator_upper_bound=metric_log_propagator,
            metric_to_euclidean_log_factor=(
                metric_to_euclidean_log_factor
            ),
            assumed_initial_defect_p_upper_bound=initial_defect,
            assumed_terminal_weighted_residual_p_upper_bound=(
                weighted_residual
            ),
            metric_endpoint_error_log_upper_bound=metric_log_error,
            euclidean_endpoint_error_log_upper_bound=euclidean_log_error,
            metric_endpoint_error_upper_bound=metric_error,
            euclidean_endpoint_error_upper_bound=euclidean_error,
            endpoint_error_exactly_zero_under_assumptions=exactly_zero,
            endpoint_error_radius_representable=radius_representable,
            dimensionless_contract_assumed_by_normalized_system=True,
            duhamel_identity_proven=True,
            fixed_weight_metric_error_bound_proven=True,
            conditional_a_posteriori_error_bound_proven=True,
        )

    def _piecewise_coarse_mesh(
        self,
        target_step_count: int,
    ) -> tuple[float, ...]:
        """Build a mesh that contains every source edge in the interval."""

        if (
            isinstance(target_step_count, bool)
            or not isinstance(target_step_count, Integral)
            or target_step_count < 1
        ):
            raise ValueError("target_step_count must be a positive integer")
        interval_width = self.n_final - self.n_initial
        interior_edges = sorted(
            edge
            for edge in (
                self.bridge.config.n_minus,
                self.bridge.config.n_plus,
            )
            if self.n_initial < edge < self.n_final
        )
        boundaries = (
            self.n_initial,
            *interior_edges,
            self.n_final,
        )
        mesh = [self.n_initial]
        for left, right in zip(
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            segment_width = right - left
            segment_step_count = max(
                1,
                math.ceil(
                    target_step_count * segment_width / interval_width
                ),
            )
            source_segment_is_active = (
                self.bridge.config.omega_prod0 > 0.0
                and right > self.bridge.config.n_minus
                and left < self.bridge.config.n_plus
            )
            if source_segment_is_active:
                segment_step_count = max(
                    segment_step_count,
                    _MINIMUM_ACTIVE_SOURCE_COARSE_STEPS,
                )
            for index in range(1, segment_step_count + 1):
                node = (
                    right
                    if index == segment_step_count
                    else left
                    + segment_width * index / segment_step_count
                )
                mesh.append(node)
        if any(
            not left < right
            for left, right in zip(mesh[:-1], mesh[1:], strict=True)
        ):
            raise ValueError("source-aligned coarse mesh is not increasing")
        return tuple(mesh)

    @staticmethod
    def _refined_mesh(mesh: tuple[float, ...]) -> tuple[float, ...]:
        """Bisect each coarse interval to obtain an exactly nested mesh."""

        if len(mesh) < 2:
            raise ValueError("coarse mesh must contain at least two nodes")
        refined = [mesh[0]]
        for left, right in zip(mesh[:-1], mesh[1:], strict=True):
            midpoint = 0.5 * (left + right)
            if not left < midpoint < right:
                raise ValueError("coarse interval is too narrow to bisect")
            refined.extend((midpoint, right))
        return tuple(refined)

    def _source_edges_aligned(self, mesh: tuple[float, ...]) -> bool:
        relevant_edges = (
            edge
            for edge in (
                self.bridge.config.n_minus,
                self.bridge.config.n_plus,
            )
            if self.n_initial <= edge <= self.n_final
        )
        return all(edge in mesh for edge in relevant_edges)

    def _active_source_step_count(self, mesh: tuple[float, ...]) -> int:
        return sum(
            1
            for left, right in zip(mesh[:-1], mesh[1:], strict=True)
            if (
                self.bridge.config.omega_prod0 > 0.0
                and right > self.bridge.config.n_minus
                and left < self.bridge.config.n_plus
            )
        )

    @staticmethod
    def _matrix_product(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        a, b, c, d = left
        e, f, g, h = right
        return (
            _finite_sum("matrix product 00", a * e, b * g),
            _finite_sum("matrix product 01", a * f, b * h),
            _finite_sum("matrix product 10", c * e, d * g),
            _finite_sum("matrix product 11", c * f, d * h),
        )

    @staticmethod
    def _matrix_exponential_apply(
        matrix: tuple[float, float, float, float],
        vector: tuple[float, float],
    ) -> tuple[float, float]:
        """Apply the exact exponential of a finite real 2 by 2 matrix."""

        a, b, c, d = matrix
        y0, y1 = vector
        half_trace = 0.5 * (a + d)
        diagonal = 0.5 * (a - d)
        discriminant = diagonal * diagonal + b * c
        if abs(discriminant) <= 1.0e-12:
            scalar = 1.0 + 0.5 * discriminant
            factor = 1.0 + discriminant / 6.0
        elif discriminant > 0.0:
            root = math.sqrt(discriminant)
            scalar = math.cosh(root)
            factor = math.sinh(root) / root
        else:
            root = math.sqrt(-discriminant)
            scalar = math.cos(root)
            factor = math.sin(root) / root
        prefactor = math.exp(half_trace)
        result0 = prefactor * _finite_sum(
            "matrix exponential row 0",
            (scalar + factor * diagonal) * y0,
            factor * b * y1,
        )
        result1 = prefactor * _finite_sum(
            "matrix exponential row 1",
            factor * c * y0,
            (scalar - factor * diagonal) * y1,
        )
        if not math.isfinite(result0) or not math.isfinite(result1):
            raise ValueError("matrix exponential left the finite domain")
        return result0, result1

    def _magnus_step(
        self,
        n_start: float,
        n_end: float,
        curvature: float,
        curvature_prime: float,
    ) -> tuple[float, float]:
        """Advance one fourth-order two-node Gauss--Magnus step."""

        step = n_end - n_start
        midpoint = 0.5 * (n_start + n_end)
        node_offset = math.sqrt(3.0) * step / 6.0
        first = self.trace_conditioned_matrix(midpoint - node_offset)
        second = self.trace_conditioned_matrix(midpoint + node_offset)
        first_second = self._matrix_product(first, second)
        second_first = self._matrix_product(second, first)
        commutator = tuple(
            left - right
            for left, right in zip(first_second, second_first, strict=True)
        )
        commutator_weight = -math.sqrt(3.0) * step * step / 12.0
        omega = tuple(
            0.5 * step * (left + right) + commutator_weight * bracket
            for left, right, bracket in zip(
                first,
                second,
                commutator,
                strict=True,
            )
        )
        if any(not math.isfinite(value) for value in omega):
            raise ValueError("Magnus generator left the finite domain")
        return self._matrix_exponential_apply(
            omega,
            (curvature, curvature_prime),
        )

    def _clock_from_trace_state(
        self,
        n: float,
        curvature: float,
        curvature_prime: float,
    ) -> float:
        background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).construct(n)
        h = background.hubble_log_derivative
        kappa = self.reduced.k_over_a_h(n)
        numerator = _finite_sum(
            "clock reconstruction numerator",
            curvature_prime,
            (1.0 + kappa * kappa / 3.0) * curvature,
        )
        clock = -numerator / h
        if not math.isfinite(clock):
            raise ValueError("clock reconstruction left the finite domain")
        return clock

    def _magnus4(
        self,
        *,
        clock_initial: float,
        curvature_initial: float,
        mesh: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        if (
            len(mesh) < 2
            or mesh[0] != self.n_initial
            or mesh[-1] != self.n_final
        ):
            raise ValueError("Magnus mesh must span the evolution interval")
        clock = clock_initial
        curvature = curvature_initial
        curvature_prime = self.rhs(
            self.n_initial,
            clock_initial,
            curvature_initial,
        )[1]
        max_clock = abs(clock)
        max_curvature = abs(curvature)
        for n_start, n_end in zip(
            mesh[:-1],
            mesh[1:],
            strict=True,
        ):
            curvature, curvature_prime = self._magnus_step(
                n_start,
                n_end,
                curvature,
                curvature_prime,
            )
            clock = self._clock_from_trace_state(
                n_end,
                curvature,
                curvature_prime,
            )
            max_clock = max(max_clock, abs(clock))
            max_curvature = max(max_curvature, abs(curvature))
        return clock, curvature, max_clock, max_curvature

    def construct(
        self,
        *,
        primordial_potential_amplitude: object,
        coarse_step_count: object = 512,
        relative_tolerance: object = 1.0e-8,
    ) -> RegularMetricEvolutionReceipt:
        """Run independent N and 2N Magnus paths and reconstruct the final node."""

        amplitude = _finite_real(
            primordial_potential_amplitude,
            "primordial_potential_amplitude",
        )
        if isinstance(coarse_step_count, bool) or not isinstance(
            coarse_step_count,
            Integral,
        ):
            raise ValueError("coarse_step_count must be an integer")
        steps = int(coarse_step_count)
        if steps < 16:
            raise ValueError("coarse_step_count must be >= 16")
        tolerance = _finite_real(relative_tolerance, "relative_tolerance")
        if tolerance <= 0.0:
            raise ValueError("relative_tolerance must be > 0")
        final_kappa = self.reduced.k_over_a_h(self.n_final)
        interval_width = self.n_final - self.n_initial
        trace_flow_stability = self.trace_flow_stability_bound()
        trace_bound = trace_flow_stability.generator_bound
        resolution_floor = math.ceil(
            interval_width * trace_bound.characteristic_rate_upper_bound
        )
        target_steps = max(steps, resolution_floor)
        if target_steps > _MAX_AUTOMATIC_COARSE_STEPS:
            raise ValueError(
                "source-aware resolution requires more than "
                f"{_MAX_AUTOMATIC_COARSE_STEPS} automatic coarse steps"
            )
        coarse_mesh = self._piecewise_coarse_mesh(target_steps)
        refined_mesh = self._refined_mesh(coarse_mesh)
        coarse_steps = len(coarse_mesh) - 1
        refined_steps = len(refined_mesh) - 1
        if coarse_steps > _MAX_AUTOMATIC_COARSE_STEPS:
            raise ValueError(
                "source-aware mesh exceeds the automatic coarse-step limit "
                f"of {_MAX_AUTOMATIC_COARSE_STEPS}"
            )
        maximum_coarse_step = max(
            right - left
            for left, right in zip(
                coarse_mesh[:-1],
                coarse_mesh[1:],
                strict=True,
            )
        )
        maximum_final_phase_step = maximum_coarse_step * final_kappa
        maximum_characteristic_scale_step = (
            maximum_coarse_step
            * trace_bound.characteristic_rate_upper_bound
        )
        source_edges_aligned = self._source_edges_aligned(coarse_mesh)
        active_source_steps = self._active_source_step_count(coarse_mesh)
        analytic_resolution_holds = (
            maximum_characteristic_scale_step
            <= 1.0 + 256.0 * math.ulp(1.0)
        )
        source_shape_resolution_holds = (
            not trace_bound.source_intersects_interval
            or self.bridge.config.omega_prod0 == 0.0
            or active_source_steps
            >= trace_bound.minimum_active_source_coarse_steps
        )
        regular = FiniteQuenchSuperhorizonRegularity(
            self.bridge
        ).construct_regular_mode(
            n=self.n_initial,
            k_over_a_h=self.kappa_initial,
            primordial_potential_amplitude=amplitude,
        )
        clock_initial = regular.required_scalar_clock_shift
        curvature_initial = regular.series_curvature_potential
        coarse_t, coarse_p, _, _ = self._magnus4(
            clock_initial=clock_initial,
            curvature_initial=curvature_initial,
            mesh=coarse_mesh,
        )
        refined_t, refined_p, max_t, max_p = self._magnus4(
            clock_initial=clock_initial,
            curvature_initial=curvature_initial,
            mesh=refined_mesh,
        )
        error_t = abs(refined_t - coarse_t) / 15.0
        error_p = abs(refined_p - coarse_p) / 15.0
        scale_t = max(abs(amplitude), abs(refined_t), 1.0e-300)
        scale_p = max(abs(amplitude), abs(refined_p), 1.0e-300)
        relative_error_t = error_t / scale_t
        relative_error_p = error_p / scale_p
        converged = (
            relative_error_t <= tolerance and relative_error_p <= tolerance
        )

        final_background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).construct(self.n_final)
        final_kappa_squared = final_kappa * final_kappa
        final_coupling = final_background.gravity_constraint_coupling
        final_enthalpy = final_background.total_enthalpy
        final_u = _finite_sum(
            "final reconstructed total momentum",
            final_kappa_squared * refined_p / (3.0 * final_coupling),
            -final_enthalpy * refined_t,
        )
        final_reduced = self.reduced.construct(
            n=self.n_final,
            scalar_clock_shift=refined_t,
            total_momentum_density=final_u,
        )
        regular_t_rhs, regular_p_rhs = self.rhs(
            self.n_final,
            refined_t,
            refined_p,
        )
        full_t_rhs = final_reduced.full_clock_log_derivative
        full_p_rhs = (
            final_reduced.algebraic_metric_tangent
            .direct_algebraic_curvature_potential_log_derivative
        )
        final_t_residual = _finite_sum(
            "final clock RHS residual",
            regular_t_rhs,
            -full_t_rhs,
        )
        final_p_residual = _finite_sum(
            "final curvature RHS residual",
            regular_p_rhs,
            -full_p_rhs,
        )
        final_t_holds = _within_roundoff(
            final_t_residual,
            regular_t_rhs,
            full_t_rhs,
        )
        final_p_holds = _within_roundoff(
            final_p_residual,
            regular_p_rhs,
            full_p_rhs,
        )
        initial_holds = regular.full_regular_mode_holds
        domain = self.reduced.domain_receipt()
        continuous = (
            domain.reduced_matrix_is_continuous
            and domain.source_regularity_derived_from_piecewise_analytic_matching
        )
        traversed = (
            self.n_initial <= self.bridge.config.n_minus
            and self.n_final >= self.bridge.config.n_plus
        )
        final_holds = final_reduced.conditional_effective_full_reconstruction_holds
        all_holds = (
            initial_holds
            and continuous
            and traversed
            and trace_bound.trace_coefficients_bounded_on_interval
            and trace_flow_stability.coefficient_signs_derived_analytically
            and (
                trace_flow_stability
                .finite_interval_continuous_dependence_bound_proven
            )
            and source_edges_aligned
            and analytic_resolution_holds
            and source_shape_resolution_holds
            and converged
            and final_holds
            and final_t_holds
            and final_p_holds
        )
        failures: list[str] = []
        if not initial_holds:
            failures.append("REGULAR_INITIAL_MODE_FAILED")
        if not continuous:
            failures.append("REGULAR_METRIC_COEFFICIENT_CONTINUITY_FAILED")
        if not traversed:
            failures.append("SOURCE_SUPPORT_NOT_TRAVERSED")
        if not trace_bound.trace_coefficients_bounded_on_interval:
            failures.append("ANALYTIC_TRACE_COEFFICIENT_BOUND_FAILED")
        if (
            not trace_flow_stability.coefficient_signs_derived_analytically
            or not (
                trace_flow_stability
                .finite_interval_continuous_dependence_bound_proven
            )
        ):
            failures.append("ANALYTIC_TRACE_FLOW_BOUND_FAILED")
        if not source_edges_aligned:
            failures.append("SOURCE_EDGES_NOT_ALIGNED")
        if not analytic_resolution_holds:
            failures.append("ANALYTIC_RESOLUTION_BOUND_FAILED")
        if not source_shape_resolution_holds:
            failures.append("NORMALIZED_SOURCE_SHAPE_UNDERRESOLVED")
        if not converged:
            failures.append("MAGNUS_STEP_DOUBLING_NOT_CONVERGED")
        if not final_holds:
            failures.append("FINAL_EFFECTIVE_RECONSTRUCTION_FAILED")
        if not final_t_holds or not final_p_holds:
            failures.append("FINAL_REGULAR_FULL_RHS_MISMATCH")

        transfer = None if amplitude == 0.0 else refined_p / amplitude
        return RegularMetricEvolutionReceipt(
            regular_initial_mode=regular,
            final_reduced_ode=final_reduced,
            trace_generator_bound=trace_bound,
            trace_flow_stability=trace_flow_stability,
            n_initial=self.n_initial,
            n_final=self.n_final,
            kappa_initial=self.kappa_initial,
            kappa_final=final_kappa,
            primordial_potential_amplitude=amplitude,
            requested_coarse_step_count=steps,
            coarse_step_count=coarse_steps,
            refined_step_count=refined_steps,
            maximum_coarse_step=maximum_coarse_step,
            maximum_final_phase_step=maximum_final_phase_step,
            maximum_characteristic_scale_step=(
                maximum_characteristic_scale_step
            ),
            active_source_coarse_step_count=active_source_steps,
            relative_tolerance=tolerance,
            coarse_final_scalar_clock_shift=coarse_t,
            refined_final_scalar_clock_shift=refined_t,
            coarse_final_curvature_potential=coarse_p,
            refined_final_curvature_potential=refined_p,
            scalar_clock_richardson_error_estimate=error_t,
            curvature_richardson_error_estimate=error_p,
            scalar_clock_relative_error_estimate=relative_error_t,
            curvature_relative_error_estimate=relative_error_p,
            refined_final_total_momentum_density=final_u,
            final_regular_clock_rhs=regular_t_rhs,
            final_regular_curvature_rhs=regular_p_rhs,
            final_full_clock_rhs=full_t_rhs,
            final_full_curvature_rhs=full_p_rhs,
            final_clock_rhs_residual=final_t_residual,
            final_curvature_rhs_residual=final_p_residual,
            max_abs_refined_scalar_clock_shift=max_t,
            max_abs_refined_curvature_potential=max_p,
            curvature_transfer_per_unit_initial_amplitude=transfer,
            initial_regular_mode_holds=initial_holds,
            regular_metric_coefficients_continuous_on_domain=continuous,
            source_support_was_traversed=traversed,
            source_edges_aligned_in_coarse_mesh=source_edges_aligned,
            analytic_resolution_bound_holds=analytic_resolution_holds,
            normalized_source_shape_resolution_holds=(
                source_shape_resolution_holds
            ),
            magnus_step_doubling_converged=converged,
            final_effective_full_reconstruction_holds=final_holds,
            final_regular_rhs_matches_full_system=(
                final_t_holds and final_p_holds
            ),
            finite_time_source_on_evolution_numerically_verified=all_holds,
            failure_reasons=tuple(failures),
        )
