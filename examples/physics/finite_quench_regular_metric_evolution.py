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
checks step doubling.  Exponentiating the frozen linear generator removes the
explicit-RK stability failure when a causal ``w=1`` reservoir drives
``kappa`` deep inside the horizon.  This supplies a reproducible finite-time
numerical trajectory through the source, but not a rigorous interval
enclosure or a general numerical-stability theorem.
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
class RegularMetricEvolutionReceipt:
    """Step-doubled finite-interval evolution and final reconstruction."""

    regular_initial_mode: SuperhorizonRegularModeReceipt
    final_reduced_ode: ReducedODEClosureReceipt
    n_initial: float
    n_final: float
    kappa_initial: float
    kappa_final: float
    primordial_potential_amplitude: float
    requested_coarse_step_count: int
    coarse_step_count: int
    refined_step_count: int
    maximum_final_phase_step: float
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
        "CONDITIONAL_MAGNUS_STEP_DOUBLED_FINITE_TIME_REGULAR_METRIC_EVOLUTION_"
        "NOT_INTERVAL_MICROPHYSICAL_PRIMORDIAL_OR_OBSERVABLE_PROOF"
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
        step_count: int,
    ) -> tuple[float, float, float, float]:
        step = (self.n_final - self.n_initial) / step_count
        clock = clock_initial
        curvature = curvature_initial
        curvature_prime = self.rhs(
            self.n_initial,
            clock_initial,
            curvature_initial,
        )[1]
        max_clock = abs(clock)
        max_curvature = abs(curvature)
        for index in range(step_count):
            n_start = self.n_initial + index * step
            n_end = (
                self.n_final
                if index == step_count - 1
                else self.n_initial + (index + 1) * step
            )
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
        phase_resolution_floor = math.ceil(
            interval_width * max(1.0, final_kappa)
        )
        effective_steps = max(steps, phase_resolution_floor)
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
            step_count=effective_steps,
        )
        refined_t, refined_p, max_t, max_p = self._magnus4(
            clock_initial=clock_initial,
            curvature_initial=curvature_initial,
            step_count=2 * effective_steps,
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
            n_initial=self.n_initial,
            n_final=self.n_final,
            kappa_initial=self.kappa_initial,
            kappa_final=final_kappa,
            primordial_potential_amplitude=amplitude,
            requested_coarse_step_count=steps,
            coarse_step_count=effective_steps,
            refined_step_count=2 * effective_steps,
            maximum_final_phase_step=(
                interval_width * final_kappa / effective_steps
            ),
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
            magnus_step_doubling_converged=converged,
            final_effective_full_reconstruction_holds=final_holds,
            final_regular_rhs_matches_full_system=(
                final_t_holds and final_p_holds
            ),
            finite_time_source_on_evolution_numerically_verified=all_holds,
            failure_reasons=tuple(failures),
        )
