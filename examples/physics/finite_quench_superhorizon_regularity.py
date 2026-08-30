"""Regular super-horizon variables and modes before the finite source opens.

In the source-off pure-reservoir era define ``W=rho+P``, ``K=kappa^2`` and
``g=CW=3(1+w)/2``.  The raw reduced variables contain

    psi = 3 C (U+WT) / K.

Using ``(T,psi)`` instead of ``(T,U)`` removes every apparent ``1/K`` pole:

    T'   = (-g+K/3) T + (1+K/3-K^2/(9g)) psi,
    psi' = g T - (1+K/3) psi.

At K=0 the eigenvalues are 0 and ``-(1+g)=-(5+3w)/2``.  Thus there is no
forward-growing super-horizon mode for ``0<=w<=1``.  Requiring boundedness as
``a->0`` removes the second (past-divergent, forward-decaying) mode and leaves
the one-dimensional constant-potential adiabatic branch.

This statement is restricted to the source-off pure perfect-fluid era.  It
does not prove source-on matching, sub-horizon stability, a primordial power
spectrum, or a microphysical transfer law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral, Real

from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
    TwoFluidFlatGRBackgroundReceipt,
)
from examples.physics.finite_quench_reduced_ode_closure import (
    FiniteQuenchReducedODEClosure,
    ReducedODEClosureReceipt,
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


def _binary_fraction(value: float) -> Fraction:
    """Freeze one finite Python float as its exact binary rational."""

    if not math.isfinite(value):
        raise ValueError("binary-fraction input must be finite")
    return Fraction.from_float(value)


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
    return abs(residual) <= 192.0 * math.ulp(scale)


@dataclass(frozen=True)
class SuperhorizonRegularSystemReceipt:
    """Exact transformed vector field and K->0 mode audit at one early node."""

    reduced_ode: ReducedODEClosureReceipt
    background: TwoFluidFlatGRBackgroundReceipt
    n: float
    reservoir_equation_of_state: float
    k_over_a_h: float
    k_over_a_h_squared: float
    gravity_enthalpy_coupling: float
    scalar_clock_shift: float
    total_momentum_density: float
    cancellation_numerator_j: float
    curvature_potential: float
    transformed_matrix_b11: float
    transformed_matrix_b12: float
    transformed_matrix_b21: float
    transformed_matrix_b22: float
    transformed_clock_log_derivative: float
    transformed_curvature_log_derivative: float
    full_clock_log_derivative: float
    full_curvature_log_derivative: float
    clock_transformation_residual: float
    curvature_transformation_residual: float
    perfect_fluid_potential_equation_residual: float
    adiabatic_limit_eigenvalue: float
    decaying_limit_eigenvalue: float
    decaying_kappa_power: float
    adiabatic_limit_eigenvector_residuals: tuple[float, float]
    decaying_limit_eigenvector_residuals: tuple[float, float]
    source_off_pure_reservoir_holds: bool
    raw_numerator_to_curvature_identity_holds: bool
    transformed_clock_equation_holds: bool
    transformed_curvature_equation_holds: bool
    perfect_fluid_potential_equation_holds: bool
    k_zero_eigenpairs_hold: bool
    no_forward_growing_superhorizon_mode: bool
    bounded_past_mode_dimension: int
    produced_intrinsic_force_vanishes: bool
    full_superhorizon_regular_system_holds: bool
    failure_reasons: tuple[str, ...]
    source_on_matching_proven: bool = False
    subhorizon_stability_proven: bool = False
    primordial_spectrum_supplied: bool = False
    microphysical_covariant_transfer_law_proven: bool = False
    source: str = (
        "Ma_Bertschinger_1995_Eqs_99_to_101_and_"
        "VMM_constraint_convention_reduction"
    )
    role: str = (
        "CONDITIONAL_SOURCE_OFF_PURE_FLUID_SUPERHORIZON_REGULAR_SYSTEM_"
        "NOT_SOURCE_ON_SUBHORIZON_PRIMORDIAL_OR_MICROPHYSICAL_PROOF"
    )


@dataclass(frozen=True)
class SuperhorizonRegularModeReceipt:
    """Past-bounded analytic power-series mode with zero decaying amplitude."""

    regular_system: SuperhorizonRegularSystemReceipt
    primordial_potential_amplitude: float
    series_terms_used: int
    first_kappa_squared_series_coefficient: float
    next_series_term_bound: float
    series_curvature_potential: float
    series_curvature_log_derivative: float
    series_curvature_second_log_derivative: float
    provided_scalar_clock_shift: float
    required_scalar_clock_shift: float
    provided_total_momentum_density: float
    required_total_momentum_density: float
    provided_curvature_potential: float
    provided_curvature_log_derivative: float
    provided_curvature_second_log_derivative: float
    clock_mode_residual: float
    momentum_mode_residual: float
    curvature_mode_residual: float
    curvature_first_mode_residual: float
    curvature_second_mode_residual: float
    regular_numerator_residual: float
    series_potential_equation_residual: float
    past_bounded_regular_series_holds: bool
    zero_decaying_mode_selected: bool
    constraint_compatible_mode_holds: bool
    full_regular_mode_holds: bool
    failure_reasons: tuple[str, ...]
    bounded_past_initial_subspace_dimension: int = 1
    potential_amplitude_is_free_initial_data: bool = True
    primordial_amplitude_predicted: bool = False
    finite_time_source_on_evolution_certified: bool = False
    role: str = (
        "CONDITIONAL_PAST_BOUNDED_SUPERHORIZON_ADIABATIC_MODE_"
        "WITH_FREE_AMPLITUDE_NOT_PRIMORDIAL_SPECTRUM_OR_SOURCE_ON_SOLUTION"
    )


@dataclass(frozen=True)
class ExactRegularModeInitialEnclosureReceipt:
    """Exact-rational enclosure of the analytic regular trace initial state."""

    n: Fraction
    source_minus: Fraction
    reservoir_equation_of_state: Fraction
    kappa_initial: Fraction
    kappa_initial_squared: Fraction
    primordial_potential_amplitude: Fraction
    exponential_rate: Fraction
    potential_friction: Fraction
    highest_partial_sum_order: int
    terms_in_partial_sum: int
    curvature_partial_sum: Fraction
    first_omitted_curvature_term_abs: Fraction
    curvature_tail_ratio_upper_bound: Fraction
    curvature_tail_abs_upper_bound: Fraction
    curvature_interval: tuple[Fraction, Fraction]
    curvature_prime_partial_sum: Fraction
    first_omitted_curvature_prime_term_abs: Fraction
    curvature_prime_tail_ratio_upper_bound: Fraction
    curvature_prime_tail_abs_upper_bound: Fraction
    curvature_prime_interval: tuple[Fraction, Fraction]
    exact_binary_float_inputs_frozen: bool
    source_off_pure_reservoir_series_equation_proven: bool
    exact_series_recurrence_proven: bool
    tail_ratios_monotone_and_strictly_below_one: bool
    exact_rational_tail_enclosures_proven: bool
    unique_past_bounded_regular_mode_enclosed: bool
    normalized_dimensionless_series_proven: bool
    potential_amplitude_is_free_initial_data: bool = True
    physical_primordial_amplitude_supplied: bool = False
    scalar_clock_initial_interval_enclosed: bool = False
    role: str = (
        "EXACT_RATIONAL_SOURCE_OFF_REGULAR_TRACE_INITIAL_STATE_"
        "WITH_FREE_AMPLITUDE_NOT_PHYSICAL_PRIMORDIAL_SPECTRUM"
    )


class FiniteQuenchSuperhorizonRegularity:
    """Audit the regularized source-off system and its bounded-past mode."""

    _SERIES_TERMS = 64

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        if not 0.0 <= bridge.config.w_reservoir <= 1.0:
            raise ValueError("regular branch requires 0 <= w_R <= 1")
        self.bridge = bridge

    def _early_background(self, n: object) -> TwoFluidFlatGRBackgroundReceipt:
        n_value = _finite_real(n, "n")
        if not self.bridge.config.n_initial <= n_value <= self.bridge.config.n_minus:
            raise ValueError("regular mode requires n in the pre-source era")
        background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).construct(n_value)
        source_off = (
            self.bridge.source(n_value) == 0.0
            and self.bridge.source_derivative(n_value) == 0.0
            and background.produced_density == 0.0
            and background.produced_density_derivative == 0.0
            and background.reservoir_density > 0.0
        )
        if not source_off:
            raise ValueError("regular mode requires a source-off pure reservoir")
        return background

    @staticmethod
    def _validated_kappa(k_over_a_h: object) -> float:
        kappa = _finite_real(k_over_a_h, "k_over_a_h")
        if not 0.0 < kappa <= 0.1:
            raise ValueError("super-horizon audit requires 0 < kappa <= 0.1")
        if not math.isfinite(kappa * kappa) or kappa * kappa <= 0.0:
            raise ValueError("kappa squared must remain positive and finite")
        return kappa

    def construct_exact_regular_initial_enclosure(
        self,
        *,
        n: object,
        k_over_a_h: object,
        primordial_potential_amplitude: object,
        highest_partial_sum_order: object = 16,
    ) -> ExactRegularModeInitialEnclosureReceipt:
        """Enclose the source-off analytic regular mode by exact rationals.

        With x = n - n_i, r = 1 + 3w, f = (5 + 3w)/2, and
        K_i = kappa_i^2, the bounded-past solution of

            psi'' + f psi' + w K_i exp(r x) psi = 0

        has psi = sum_m t_m exp(m r x), where

            t_m / t_(m-1) = -w K_i / [m r (m r + f)].

        At x = 0 the decreasing absolute term ratios give geometric
        enclosures for both psi and psi'. All inputs are the exact binary
        rationals represented by the validated Python floats.
        """

        background = self._early_background(n)
        kappa = self._validated_kappa(k_over_a_h)
        amplitude = _finite_real(
            primordial_potential_amplitude,
            "primordial_potential_amplitude",
        )
        if isinstance(highest_partial_sum_order, bool) or not isinstance(
            highest_partial_sum_order,
            Integral,
        ):
            raise ValueError("highest_partial_sum_order must be an integer")
        order = int(highest_partial_sum_order)
        if not 0 <= order <= 256:
            raise ValueError(
                "highest_partial_sum_order must lie between 0 and 256"
            )

        config = self.bridge.config
        n_exact = _binary_fraction(background.n)
        source_minus = (
            _binary_fraction(config.n_star)
            - _binary_fraction(config.half_width)
        )
        if n_exact > source_minus:
            raise ValueError(
                "exact regular series requires a source-off initial node"
            )

        w = _binary_fraction(config.w_reservoir)
        kappa_exact = _binary_fraction(kappa)
        kappa_squared = kappa_exact * kappa_exact
        amplitude_exact = _binary_fraction(amplitude)
        rate = 1 + 3 * w
        friction = (5 + 3 * w) / 2
        coupling = w * kappa_squared
        if (
            not Fraction(0) <= w <= Fraction(1)
            or rate <= 0
            or friction <= 0
            or kappa_squared <= 0
        ):
            raise ValueError("exact regular-series assumptions failed")

        term = amplitude_exact
        curvature_partial = term
        curvature_prime_partial = Fraction(0)
        for m in range(1, order + 1):
            m_exact = Fraction(m)
            term *= (
                -coupling
                / (
                    m_exact
                    * rate
                    * (m_exact * rate + friction)
                )
            )
            curvature_partial += term
            curvature_prime_partial += m_exact * rate * term

        first_omitted_index = order + 1
        first_omitted_index_exact = Fraction(first_omitted_index)
        first_omitted = term * (
            -coupling
            / (
                first_omitted_index_exact
                * rate
                * (
                    first_omitted_index_exact * rate
                    + friction
                )
            )
        )
        first_omitted_abs = abs(first_omitted)
        next_index = Fraction(order + 2)
        curvature_tail_ratio = coupling / (
            next_index * rate * (next_index * rate + friction)
        )
        curvature_prime_first_abs = (
            first_omitted_index_exact * rate * first_omitted_abs
        )
        curvature_prime_tail_ratio = coupling / (
            first_omitted_index_exact
            * rate
            * (next_index * rate + friction)
        )
        if (
            not Fraction(0) <= curvature_tail_ratio < Fraction(1)
            or not Fraction(0)
            <= curvature_prime_tail_ratio
            < Fraction(1)
        ):
            raise ValueError("regular-series tail ratio is not contractive")

        curvature_tail = first_omitted_abs / (
            1 - curvature_tail_ratio
        )
        curvature_prime_tail = curvature_prime_first_abs / (
            1 - curvature_prime_tail_ratio
        )
        curvature_interval = (
            curvature_partial - curvature_tail,
            curvature_partial + curvature_tail,
        )
        curvature_prime_interval = (
            curvature_prime_partial - curvature_prime_tail,
            curvature_prime_partial + curvature_prime_tail,
        )

        return ExactRegularModeInitialEnclosureReceipt(
            n=n_exact,
            source_minus=source_minus,
            reservoir_equation_of_state=w,
            kappa_initial=kappa_exact,
            kappa_initial_squared=kappa_squared,
            primordial_potential_amplitude=amplitude_exact,
            exponential_rate=rate,
            potential_friction=friction,
            highest_partial_sum_order=order,
            terms_in_partial_sum=order + 1,
            curvature_partial_sum=curvature_partial,
            first_omitted_curvature_term_abs=first_omitted_abs,
            curvature_tail_ratio_upper_bound=curvature_tail_ratio,
            curvature_tail_abs_upper_bound=curvature_tail,
            curvature_interval=curvature_interval,
            curvature_prime_partial_sum=curvature_prime_partial,
            first_omitted_curvature_prime_term_abs=(
                curvature_prime_first_abs
            ),
            curvature_prime_tail_ratio_upper_bound=(
                curvature_prime_tail_ratio
            ),
            curvature_prime_tail_abs_upper_bound=curvature_prime_tail,
            curvature_prime_interval=curvature_prime_interval,
            exact_binary_float_inputs_frozen=True,
            source_off_pure_reservoir_series_equation_proven=True,
            exact_series_recurrence_proven=True,
            tail_ratios_monotone_and_strictly_below_one=True,
            exact_rational_tail_enclosures_proven=True,
            unique_past_bounded_regular_mode_enclosed=True,
            normalized_dimensionless_series_proven=True,
        )

    def construct_system(
        self,
        *,
        n: object,
        k_over_a_h: object,
        scalar_clock_shift: object,
        total_momentum_density: object,
    ) -> SuperhorizonRegularSystemReceipt:
        """Transform a raw reduced state and audit the pole-free equations."""

        background = self._early_background(n)
        kappa = self._validated_kappa(k_over_a_h)
        clock = _finite_real(scalar_clock_shift, "scalar_clock_shift")
        total_u = _finite_real(
            total_momentum_density,
            "total_momentum_density",
        )
        reduced = FiniteQuenchReducedODEClosure(
            self.bridge,
            n_reference=background.n,
            kappa_reference=kappa,
        ).construct(
            n=background.n,
            scalar_clock_shift=clock,
            total_momentum_density=total_u,
        )
        node = reduced.common_clock_second_tangent.common_clock_tangent.gr_linear_node
        einstein = node.einstein_constraint
        trace = reduced.common_clock_second_tangent.einstein_trace_evolution
        coupling = background.gravity_constraint_coupling
        enthalpy = background.total_enthalpy
        g = coupling * enthalpy
        kappa_squared = kappa * kappa
        j = _finite_sum(
            "regular cancellation numerator",
            total_u,
            enthalpy * clock,
        )
        psi = einstein.curvature_potential
        identity_residual = _finite_sum(
            "raw numerator to curvature residual",
            psi,
            -3.0 * coupling * j / kappa_squared,
        )
        b11 = -g + kappa_squared / 3.0
        b12 = (
            1.0
            + kappa_squared / 3.0
            - kappa_squared * kappa_squared / (9.0 * g)
        )
        b21 = g
        b22 = -(1.0 + kappa_squared / 3.0)
        transformed_t_prime = _finite_sum(
            "transformed clock derivative",
            b11 * clock,
            b12 * psi,
        )
        transformed_psi_prime = _finite_sum(
            "transformed curvature derivative",
            b21 * clock,
            b22 * psi,
        )
        full_t_prime = reduced.full_clock_log_derivative
        full_psi_prime = (
            reduced.algebraic_metric_tangent
            .direct_algebraic_curvature_potential_log_derivative
        )
        clock_residual = _finite_sum(
            "transformed clock residual",
            full_t_prime,
            -transformed_t_prime,
        )
        curvature_residual = _finite_sum(
            "transformed curvature residual",
            full_psi_prime,
            -transformed_psi_prime,
        )
        w = self.bridge.config.w_reservoir
        potential_friction = (5.0 + 3.0 * w) / 2.0
        potential_equation_residual = _finite_sum(
            "perfect-fluid potential equation residual",
            trace.provided_curvature_potential_second_log_derivative,
            potential_friction * full_psi_prime,
            w * kappa_squared * psi,
        )
        lambda_ad = 0.0
        lambda_dec = -(1.0 + g)
        kappa_power = (1.0 + g) / ((1.0 + 3.0 * w) / 2.0)
        ad_t = 1.0 / g
        ad_psi = 1.0
        ad_residuals = (
            _finite_sum("adiabatic eigenvector row 1", -g * ad_t, ad_psi),
            _finite_sum("adiabatic eigenvector row 2", g * ad_t, -ad_psi),
        )
        dec_t = 1.0
        dec_psi = -1.0
        dec_residuals = (
            _finite_sum(
                "decaying eigenvector row 1",
                -g * dec_t,
                dec_psi,
                -lambda_dec * dec_t,
            ),
            _finite_sum(
                "decaying eigenvector row 2",
                g * dec_t,
                -dec_psi,
                -lambda_dec * dec_psi,
            ),
        )
        identity_holds = _within_roundoff(
            identity_residual,
            psi,
            3.0 * coupling * j / kappa_squared,
        )
        clock_holds = _within_roundoff(
            clock_residual,
            full_t_prime,
            transformed_t_prime,
        )
        curvature_holds = _within_roundoff(
            curvature_residual,
            full_psi_prime,
            transformed_psi_prime,
        )
        potential_holds = _within_roundoff(
            potential_equation_residual,
            trace.provided_curvature_potential_second_log_derivative,
            potential_friction * full_psi_prime,
            w * kappa_squared * psi,
        )
        eigenpairs_hold = all(
            _within_roundoff(value, value, 0.0)
            for value in (*ad_residuals, *dec_residuals)
        )
        source_off = (
            background.produced_density == 0.0
            and self.bridge.source(background.n) == 0.0
        )
        produced_force = (
            reduced.required_effective_produced_intrinsic_force
        )
        force_vanishes = _within_roundoff(produced_force, produced_force, 0.0)
        no_growth = lambda_ad == 0.0 and lambda_dec < 0.0
        all_holds = (
            reduced.conditional_effective_full_reconstruction_holds
            and source_off
            and identity_holds
            and clock_holds
            and curvature_holds
            and potential_holds
            and eigenpairs_hold
            and no_growth
            and force_vanishes
        )
        failures: list[str] = []
        if not source_off:
            failures.append("SOURCE_OFF_PURE_RESERVOIR_FAILED")
        if not identity_holds:
            failures.append("RAW_NUMERATOR_CURVATURE_IDENTITY_FAILED")
        if not clock_holds:
            failures.append("TRANSFORMED_CLOCK_EQUATION_FAILED")
        if not curvature_holds:
            failures.append("TRANSFORMED_CURVATURE_EQUATION_FAILED")
        if not potential_holds:
            failures.append("PERFECT_FLUID_POTENTIAL_EQUATION_FAILED")
        if not eigenpairs_hold:
            failures.append("K_ZERO_EIGENPAIR_FAILED")
        if not no_growth:
            failures.append("FORWARD_GROWING_SUPERHORIZON_MODE_FOUND")
        if not force_vanishes:
            failures.append("EMPTY_PRODUCED_FORCE_NONZERO")
        if not reduced.conditional_effective_full_reconstruction_holds:
            failures.append("PARENT_REDUCED_ODE_FAILED")

        return SuperhorizonRegularSystemReceipt(
            reduced_ode=reduced,
            background=background,
            n=background.n,
            reservoir_equation_of_state=w,
            k_over_a_h=kappa,
            k_over_a_h_squared=kappa_squared,
            gravity_enthalpy_coupling=g,
            scalar_clock_shift=clock,
            total_momentum_density=total_u,
            cancellation_numerator_j=j,
            curvature_potential=psi,
            transformed_matrix_b11=b11,
            transformed_matrix_b12=b12,
            transformed_matrix_b21=b21,
            transformed_matrix_b22=b22,
            transformed_clock_log_derivative=transformed_t_prime,
            transformed_curvature_log_derivative=transformed_psi_prime,
            full_clock_log_derivative=full_t_prime,
            full_curvature_log_derivative=full_psi_prime,
            clock_transformation_residual=clock_residual,
            curvature_transformation_residual=curvature_residual,
            perfect_fluid_potential_equation_residual=(
                potential_equation_residual
            ),
            adiabatic_limit_eigenvalue=lambda_ad,
            decaying_limit_eigenvalue=lambda_dec,
            decaying_kappa_power=kappa_power,
            adiabatic_limit_eigenvector_residuals=ad_residuals,
            decaying_limit_eigenvector_residuals=dec_residuals,
            source_off_pure_reservoir_holds=source_off,
            raw_numerator_to_curvature_identity_holds=identity_holds,
            transformed_clock_equation_holds=clock_holds,
            transformed_curvature_equation_holds=curvature_holds,
            perfect_fluid_potential_equation_holds=potential_holds,
            k_zero_eigenpairs_hold=eigenpairs_hold,
            no_forward_growing_superhorizon_mode=no_growth,
            bounded_past_mode_dimension=1,
            produced_intrinsic_force_vanishes=force_vanishes,
            full_superhorizon_regular_system_holds=all_holds,
            failure_reasons=tuple(failures),
        )

    def _regular_series(
        self,
        *,
        w: float,
        kappa_squared: float,
        amplitude: float,
    ) -> tuple[float, float, float, int, float, float]:
        rate = 1.0 + 3.0 * w
        friction = (5.0 + 3.0 * w) / 2.0
        terms = [amplitude]
        first_coefficient = (
            0.0
            if w == 0.0
            else -w / (rate * (rate + friction))
        )
        term = amplitude
        for m in range(1, self._SERIES_TERMS):
            term *= (
                -w
                * kappa_squared
                / (m * rate * (m * rate + friction))
            )
            terms.append(term)
        psi = math.fsum(terms)
        psi_prime = math.fsum(
            m * rate * value for m, value in enumerate(terms)
        )
        psi_second = math.fsum(
            (m * rate) ** 2 * value for m, value in enumerate(terms)
        )
        next_m = self._SERIES_TERMS
        next_term = term * (
            -w
            * kappa_squared
            / (next_m * rate * (next_m * rate + friction))
        )
        if any(not math.isfinite(value) for value in (psi, psi_prime, psi_second)):
            raise ValueError("regular mode series left the finite domain")
        return (
            psi,
            psi_prime,
            psi_second,
            self._SERIES_TERMS,
            first_coefficient,
            abs(next_term),
        )

    def construct_regular_mode(
        self,
        *,
        n: object,
        k_over_a_h: object,
        primordial_potential_amplitude: object,
    ) -> SuperhorizonRegularModeReceipt:
        """Construct the unique past-bounded power-series mode at one node."""

        background = self._early_background(n)
        kappa = self._validated_kappa(k_over_a_h)
        amplitude = _finite_real(
            primordial_potential_amplitude,
            "primordial_potential_amplitude",
        )
        kappa_squared = kappa * kappa
        w = self.bridge.config.w_reservoir
        psi, psi_prime, psi_second, _, _, _ = self._regular_series(
            w=w,
            kappa_squared=kappa_squared,
            amplitude=amplitude,
        )
        coupling = background.gravity_constraint_coupling
        h = background.hubble_log_derivative
        clock = -(
            kappa_squared * psi + 3.0 * (psi_prime + psi)
        ) / (3.0 * h)
        total_u = -(psi_prime + psi) / coupling
        return self.audit_regular_mode(
            n=background.n,
            k_over_a_h=kappa,
            primordial_potential_amplitude=amplitude,
            scalar_clock_shift=clock,
            total_momentum_density=total_u,
            curvature_potential=psi,
            curvature_potential_log_derivative=psi_prime,
            curvature_potential_second_log_derivative=psi_second,
        )

    def audit_regular_mode(
        self,
        *,
        n: object,
        k_over_a_h: object,
        primordial_potential_amplitude: object,
        scalar_clock_shift: object,
        total_momentum_density: object,
        curvature_potential: object,
        curvature_potential_log_derivative: object,
        curvature_potential_second_log_derivative: object,
    ) -> SuperhorizonRegularModeReceipt:
        """Audit independent state and metric candidates against the series."""

        background = self._early_background(n)
        kappa = self._validated_kappa(k_over_a_h)
        amplitude = _finite_real(
            primordial_potential_amplitude,
            "primordial_potential_amplitude",
        )
        provided_clock = _finite_real(scalar_clock_shift, "scalar_clock_shift")
        provided_u = _finite_real(
            total_momentum_density,
            "total_momentum_density",
        )
        provided_psi = _finite_real(curvature_potential, "curvature_potential")
        provided_psi_prime = _finite_real(
            curvature_potential_log_derivative,
            "curvature_potential_log_derivative",
        )
        provided_psi_second = _finite_real(
            curvature_potential_second_log_derivative,
            "curvature_potential_second_log_derivative",
        )
        kappa_squared = kappa * kappa
        w = self.bridge.config.w_reservoir
        (
            series_psi,
            series_psi_prime,
            series_psi_second,
            terms_used,
            first_coefficient,
            next_bound,
        ) = self._regular_series(
            w=w,
            kappa_squared=kappa_squared,
            amplitude=amplitude,
        )
        coupling = background.gravity_constraint_coupling
        h = background.hubble_log_derivative
        enthalpy = background.total_enthalpy
        required_clock = -(
            kappa_squared * series_psi
            + 3.0 * (series_psi_prime + series_psi)
        ) / (3.0 * h)
        required_u = -(series_psi_prime + series_psi) / coupling
        system = self.construct_system(
            n=background.n,
            k_over_a_h=kappa,
            scalar_clock_shift=provided_clock,
            total_momentum_density=provided_u,
        )
        clock_residual = _finite_sum(
            "regular clock mode residual",
            provided_clock,
            -required_clock,
        )
        momentum_residual = _finite_sum(
            "regular momentum mode residual",
            provided_u,
            -required_u,
        )
        psi_residual = _finite_sum(
            "regular curvature mode residual",
            provided_psi,
            -series_psi,
        )
        psi_prime_residual = _finite_sum(
            "regular curvature first mode residual",
            provided_psi_prime,
            -series_psi_prime,
        )
        psi_second_residual = _finite_sum(
            "regular curvature second mode residual",
            provided_psi_second,
            -series_psi_second,
        )
        j = _finite_sum(
            "regular mode numerator",
            provided_u,
            enthalpy * provided_clock,
        )
        numerator_residual = _finite_sum(
            "regular numerator residual",
            j,
            -kappa_squared * provided_psi / (3.0 * coupling),
        )
        friction = (5.0 + 3.0 * w) / 2.0
        series_equation_residual = _finite_sum(
            "regular series potential residual",
            series_psi_second,
            friction * series_psi_prime,
            w * kappa_squared * series_psi,
        )
        clock_holds = _within_roundoff(
            clock_residual,
            provided_clock,
            required_clock,
        )
        momentum_holds = _within_roundoff(
            momentum_residual,
            provided_u,
            required_u,
        )
        psi_holds = _within_roundoff(psi_residual, provided_psi, series_psi)
        psi_prime_holds = _within_roundoff(
            psi_prime_residual,
            provided_psi_prime,
            series_psi_prime,
        )
        psi_second_holds = _within_roundoff(
            psi_second_residual,
            provided_psi_second,
            series_psi_second,
        )
        numerator_holds = _within_roundoff(
            numerator_residual,
            j,
            kappa_squared * provided_psi / (3.0 * coupling),
            provided_u,
            enthalpy * provided_clock,
        )
        series_holds = _within_roundoff(
            series_equation_residual,
            series_psi_second,
            friction * series_psi_prime,
            w * kappa_squared * series_psi,
            next_bound,
        )
        constraint_mode_holds = (
            clock_holds
            and momentum_holds
            and psi_holds
            and psi_prime_holds
            and psi_second_holds
            and numerator_holds
            and system.full_superhorizon_regular_system_holds
        )
        all_holds = series_holds and constraint_mode_holds
        failures: list[str] = []
        if not series_holds:
            failures.append("PAST_BOUNDED_SERIES_EQUATION_FAILED")
        if not clock_holds:
            failures.append("REGULAR_CLOCK_MODE_FAILED")
        if not momentum_holds:
            failures.append("REGULAR_MOMENTUM_MODE_FAILED")
        if not psi_holds or not psi_prime_holds or not psi_second_holds:
            failures.append("REGULAR_METRIC_SERIES_FAILED")
        if not numerator_holds:
            failures.append("REGULAR_NUMERATOR_CANCELLATION_FAILED")
        if not system.full_superhorizon_regular_system_holds:
            failures.append("PARENT_REGULAR_SYSTEM_FAILED")

        return SuperhorizonRegularModeReceipt(
            regular_system=system,
            primordial_potential_amplitude=amplitude,
            series_terms_used=terms_used,
            first_kappa_squared_series_coefficient=first_coefficient,
            next_series_term_bound=next_bound,
            series_curvature_potential=series_psi,
            series_curvature_log_derivative=series_psi_prime,
            series_curvature_second_log_derivative=series_psi_second,
            provided_scalar_clock_shift=provided_clock,
            required_scalar_clock_shift=required_clock,
            provided_total_momentum_density=provided_u,
            required_total_momentum_density=required_u,
            provided_curvature_potential=provided_psi,
            provided_curvature_log_derivative=provided_psi_prime,
            provided_curvature_second_log_derivative=provided_psi_second,
            clock_mode_residual=clock_residual,
            momentum_mode_residual=momentum_residual,
            curvature_mode_residual=psi_residual,
            curvature_first_mode_residual=psi_prime_residual,
            curvature_second_mode_residual=psi_second_residual,
            regular_numerator_residual=numerator_residual,
            series_potential_equation_residual=series_equation_residual,
            past_bounded_regular_series_holds=series_holds,
            zero_decaying_mode_selected=True,
            constraint_compatible_mode_holds=constraint_mode_holds,
            full_regular_mode_holds=all_holds,
            failure_reasons=tuple(failures),
        )
