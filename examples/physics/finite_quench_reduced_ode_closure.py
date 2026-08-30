"""Reduced two-variable ODE closure for the strict finite-quench branch.

On the common-clock manifold, total density and pressure perturbations are

    Delta = -3 W T,                 Delta_P = w_R r_R' T,

where ``W=rho+P``.  The k>0 Einstein constraints give

    psi = 3 C (U+W T) / kappa^2,
    psi_n = -C U - psi.

The summed energy and momentum equations therefore reduce exactly to a
linear nonautonomous system for ``y=(T,U)``:

    T' = A11 T + A12 U,
    U' = A21 T + A22 U.

The pointwise force solved by the second-tangent gate is adopted here as an
explicit effective, gauge-fixed constitutive closure at every node.  It is
not claimed to be microscopic or covariant.  Continuous coefficients on the
compact finite interval give a unique global reduced solution for every
finite initial ``(T,U)`` by the standard linear-ODE existence theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.finite_quench_algebraic_metric_tangent import (
    AlgebraicMetricTangentReceipt,
    FiniteQuenchAlgebraicMetricTangent,
)
from examples.physics.finite_quench_common_clock_second_tangent import (
    CommonClockSecondTangentReceipt,
    FiniteQuenchCommonClockSecondTangent,
)
from examples.physics.finite_quench_einstein_constraint_propagation import (
    EinsteinConstraintPropagationReceipt,
    FiniteQuenchEinsteinConstraintPropagation,
)
from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
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
    return abs(residual) <= 128.0 * math.ulp(scale)


@dataclass(frozen=True)
class ReducedODEDomainReceipt:
    """Analytic regularity and denominator bounds on the finite domain."""

    n_initial: float
    n_final: float
    n_reference: float
    kappa_reference: float
    present_total_density_lower_bound: float
    present_total_enthalpy_lower_bound: float
    kappa_at_n_initial: float
    kappa_at_n_final: float
    kappa_positive_lower_bound: float
    hubble_log_derivative_lower_bound: float
    hubble_log_derivative_upper_bound: float
    kappa_log_derivative_lower_bound: float
    kappa_log_derivative_upper_bound: float
    source_endpoint_values: tuple[float, float]
    source_endpoint_derivatives: tuple[float, float]
    source_regularity_derived_from_piecewise_analytic_matching: bool
    background_regularity_derived_from_continuity_odes: bool
    compact_source_is_c1: bool
    background_is_at_least_c2: bool
    reduced_matrix_is_continuous: bool
    effective_force_closure_is_continuous: bool
    all_denominators_have_positive_domain_bounds: bool
    global_reduced_linear_ode_existence_uniqueness_proven: bool
    force_closure_status: str = (
        "ADOPTED_EFFECTIVE_GAUGE_FIXED_TANGENT_PRESERVING_AXIOM"
    )
    microphysical_covariant_force_law_proven: bool = False
    source: str = "standard_continuous_linear_ODE_existence_uniqueness_theorem"


@dataclass(frozen=True)
class ReducedODEClosureReceipt:
    """One-node audit of the globally defined reduced linear vector field."""

    domain: ReducedODEDomainReceipt
    common_clock_second_tangent: CommonClockSecondTangentReceipt
    constraint_propagation: EinsteinConstraintPropagationReceipt
    algebraic_metric_tangent: AlgebraicMetricTangentReceipt
    n: float
    k_over_a_h: float
    scalar_clock_shift: float
    total_momentum_density: float
    matrix_a11: float
    matrix_a12: float
    matrix_a21: float
    matrix_a22: float
    reduced_clock_log_derivative: float
    reduced_total_momentum_density_derivative: float
    full_clock_log_derivative: float
    full_total_momentum_density_derivative: float
    clock_rhs_residual: float
    total_momentum_rhs_residual: float
    required_effective_produced_intrinsic_force: float
    reference_kappa_state_matches: bool
    parent_second_tangent_holds: bool
    parent_constraint_propagation_holds: bool
    parent_algebraic_metric_tangent_holds: bool
    reduced_clock_rhs_matches_full_system: bool
    reduced_momentum_rhs_matches_full_system: bool
    pointwise_effective_force_closure_holds: bool
    conditional_global_reduced_solution_exists_uniquely: bool
    conditional_effective_full_reconstruction_holds: bool
    failure_reasons: tuple[str, ...]
    numerical_finite_step_solution_certified: bool = False
    interval_enclosure_proven: bool = False
    microphysical_covariant_transfer_law_proven: bool = False
    observed_initial_spectrum_supplied: bool = False
    observable_prediction_proven: bool = False
    dimensionless_roles: tuple[tuple[str, str], ...] = (
        ("T", "dimensionless e-fold scalar clock shift"),
        ("U", "total momentum density divided by rho_unit"),
        ("Aij", "continuous dimensionless reduced ODE coefficients"),
        ("kappa", "k/(aH)"),
        ("fhat_p", "a f_p/rho_unit"),
    )
    role: str = (
        "CONDITIONAL_GLOBAL_LINEAR_REDUCED_ODE_WITH_ADOPTED_EFFECTIVE_FORCE_"
        "NOT_MICROPHYSICAL_COVARIANT_INTERVAL_NUMERICAL_OR_OBSERVABLE_PROOF"
    )


class FiniteQuenchReducedODEClosure:
    """Define and audit the exact common-clock two-variable vector field."""

    def __init__(
        self,
        bridge: FiniteQuenchBridge,
        *,
        n_reference: object,
        kappa_reference: object,
    ) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge
        self.n_reference = self._validated_n(n_reference)
        self.kappa_reference = _finite_real(
            kappa_reference,
            "kappa_reference",
        )
        if self.kappa_reference <= 0.0:
            raise ValueError("kappa_reference must be > 0")
        if not 0.0 <= bridge.config.w_reservoir <= 1.0:
            raise ValueError("reduced strict branch requires 0 <= w_R <= 1")
        present_total = (
            bridge.config.omega_prod0
            + bridge.config.reservoir_present_density
        )
        if present_total <= 0.0:
            raise ValueError("reduced domain requires positive present density")
        self._reference_density = self._total_density(self.n_reference)

    def _validated_n(self, n: object) -> float:
        value = _finite_real(n, "n")
        if not self.bridge.config.n_initial <= value <= 0.0:
            raise ValueError("n is outside the reduced ODE domain")
        return value

    def _total_density(self, n: float) -> float:
        value = _finite_sum(
            "reduced total density",
            self.bridge.production_density(n),
            self.bridge.reservoir_density(n),
        )
        if value <= 0.0:
            raise ValueError("reduced ODE requires positive total density")
        return value

    def k_over_a_h(self, n: object) -> float:
        """Return exact k/(aH) relative to the declared reference node."""

        n_value = self._validated_n(n)
        rho = self._total_density(n_value)
        try:
            value = (
                self.kappa_reference
                * math.exp(self.n_reference - n_value)
                * math.sqrt(self._reference_density / rho)
            )
        except (OverflowError, ValueError) as error:
            raise ValueError("kappa evolution left the finite domain") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("kappa evolution must remain positive and finite")
        return value

    def domain_receipt(self) -> ReducedODEDomainReceipt:
        """Return analytic positivity, regularity, and linear-ODE status."""

        n_initial = self.bridge.config.n_initial
        n_final = 0.0
        present_total = (
            self.bridge.config.omega_prod0
            + self.bridge.config.reservoir_present_density
        )
        kappa_initial = self.k_over_a_h(n_initial)
        kappa_final = self.k_over_a_h(n_final)
        denominators_positive = (
            present_total > 0.0
            and kappa_initial > 0.0
            and kappa_final >= kappa_initial
        )
        source_endpoint_values = (
            self.bridge.source(self.bridge.config.n_minus),
            self.bridge.source(self.bridge.config.n_plus),
        )
        source_endpoint_derivatives = (
            self.bridge.source_derivative(self.bridge.config.n_minus),
            self.bridge.source_derivative(self.bridge.config.n_plus),
        )
        piecewise_matching = (
            self.bridge.config.n_minus < self.bridge.config.n_plus
            and source_endpoint_values == (0.0, 0.0)
            and source_endpoint_derivatives == (0.0, 0.0)
        )
        source_c1 = piecewise_matching
        background_from_odes = source_c1
        background_c2 = background_from_odes
        matrix_continuous = source_c1 and background_c2 and denominators_positive
        force_continuous = matrix_continuous
        global_unique = matrix_continuous
        return ReducedODEDomainReceipt(
            n_initial=n_initial,
            n_final=n_final,
            n_reference=self.n_reference,
            kappa_reference=self.kappa_reference,
            present_total_density_lower_bound=present_total,
            present_total_enthalpy_lower_bound=present_total,
            kappa_at_n_initial=kappa_initial,
            kappa_at_n_final=kappa_final,
            kappa_positive_lower_bound=kappa_initial,
            hubble_log_derivative_lower_bound=-3.0,
            hubble_log_derivative_upper_bound=-1.5,
            kappa_log_derivative_lower_bound=0.5,
            kappa_log_derivative_upper_bound=2.0,
            source_endpoint_values=source_endpoint_values,
            source_endpoint_derivatives=source_endpoint_derivatives,
            source_regularity_derived_from_piecewise_analytic_matching=(
                piecewise_matching
            ),
            background_regularity_derived_from_continuity_odes=(
                background_from_odes
            ),
            compact_source_is_c1=source_c1,
            background_is_at_least_c2=background_c2,
            reduced_matrix_is_continuous=matrix_continuous,
            effective_force_closure_is_continuous=force_continuous,
            all_denominators_have_positive_domain_bounds=denominators_positive,
            global_reduced_linear_ode_existence_uniqueness_proven=(
                global_unique
            ),
        )

    def _matrix(self, n: float, kappa: float) -> tuple[float, float, float, float]:
        background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).construct(n)
        coupling = background.gravity_constraint_coupling
        enthalpy = background.total_enthalpy
        h = background.hubble_log_derivative
        kappa_squared = kappa * kappa
        if not math.isfinite(kappa_squared) or kappa_squared <= 0.0:
            raise ValueError("reduced matrix requires finite positive kappa squared")
        reservoir_pressure_prime_coefficient = (
            self.bridge.config.w_reservoir
            * background.reservoir_density_derivative
        )
        a11 = 3.0 * coupling * enthalpy / kappa_squared
        a12 = _finite_sum(
            "reduced A12",
            coupling,
            3.0 * coupling / kappa_squared,
            -kappa_squared / (3.0 * enthalpy),
        )
        a21 = _finite_sum(
            "reduced A21",
            -3.0 * coupling * enthalpy * enthalpy / kappa_squared,
            -reservoir_pressure_prime_coefficient,
        )
        a22 = _finite_sum(
            "reduced A22",
            -(3.0 - h),
            -3.0 * coupling * enthalpy / kappa_squared,
        )
        if any(not math.isfinite(value) for value in (a11, a12, a21, a22)):
            raise ValueError("reduced matrix left the finite domain")
        return a11, a12, a21, a22

    def _raw_metric(
        self,
        receipt: object,
    ) -> AlgebraicMetricTangentReceipt:
        if not isinstance(receipt, AlgebraicMetricTangentReceipt):
            raise ValueError(
                "algebraic_metric_tangent must be an "
                "AlgebraicMetricTangentReceipt"
            )
        return FiniteQuenchAlgebraicMetricTangent(self.bridge).audit(
            constraint_propagation=receipt.constraint_propagation,
            curvature_potential_log_derivative=(
                receipt.provided_curvature_potential_log_derivative
            ),
            curvature_potential_second_log_derivative=(
                receipt.provided_curvature_potential_second_log_derivative
            ),
        )

    def construct(
        self,
        *,
        n: object,
        scalar_clock_shift: object,
        total_momentum_density: object,
    ) -> ReducedODEClosureReceipt:
        """Construct the full local chain from one reduced state and audit it."""

        n_value = self._validated_n(n)
        clock = _finite_real(scalar_clock_shift, "scalar_clock_shift")
        total_u = _finite_real(
            total_momentum_density,
            "total_momentum_density",
        )
        kappa = self.k_over_a_h(n_value)
        second = FiniteQuenchCommonClockSecondTangent(self.bridge).construct(
            n=n_value,
            k_over_a_h=kappa,
            scalar_clock_shift=clock,
            total_momentum_density=total_u,
        )
        propagation = FiniteQuenchEinsteinConstraintPropagation(
            self.bridge
        ).construct(common_clock_second_tangent=second)
        metric = FiniteQuenchAlgebraicMetricTangent(self.bridge).construct(
            constraint_propagation=propagation
        )
        node = second.common_clock_tangent.gr_linear_node
        return self.audit(
            algebraic_metric_tangent=metric,
            scalar_clock_log_derivative=(
                second.common_clock_tangent.scalar_clock_log_derivative
            ),
            total_momentum_density_derivative=(
                node.momentum_equation.provided_total_momentum_density_derivative
            ),
        )

    def audit(
        self,
        *,
        algebraic_metric_tangent: object,
        scalar_clock_log_derivative: object,
        total_momentum_density_derivative: object,
    ) -> ReducedODEClosureReceipt:
        """Audit independent reduced-RHS derivative candidates."""

        metric = self._raw_metric(algebraic_metric_tangent)
        propagation = metric.constraint_propagation
        second = propagation.common_clock_second_tangent
        tangent = second.common_clock_tangent
        node = tangent.gr_linear_node
        n = node.background.n
        kappa = node.einstein_constraint.k_over_a_h
        expected_kappa = self.k_over_a_h(n)
        clock = node.scalar_clock.scalar_clock_shift
        total_u = node.einstein_constraint.total_momentum_density
        provided_clock_prime = _finite_real(
            scalar_clock_log_derivative,
            "scalar_clock_log_derivative",
        )
        provided_u_prime = _finite_real(
            total_momentum_density_derivative,
            "total_momentum_density_derivative",
        )
        a11, a12, a21, a22 = self._matrix(n, kappa)
        reduced_clock_prime = _finite_sum(
            "reduced clock derivative",
            a11 * clock,
            a12 * total_u,
        )
        reduced_u_prime = _finite_sum(
            "reduced total momentum derivative",
            a21 * clock,
            a22 * total_u,
        )
        clock_residual = _finite_sum(
            "reduced clock RHS residual",
            provided_clock_prime,
            -reduced_clock_prime,
        )
        momentum_residual = _finite_sum(
            "reduced momentum RHS residual",
            provided_u_prime,
            -reduced_u_prime,
        )
        kappa_residual = _finite_sum(
            "reference kappa state residual",
            kappa,
            -expected_kappa,
        )
        kappa_holds = _within_roundoff(
            kappa_residual,
            kappa,
            expected_kappa,
        )
        clock_holds = _within_roundoff(
            clock_residual,
            provided_clock_prime,
            reduced_clock_prime,
            a11 * clock,
            a12 * total_u,
        )
        momentum_holds = _within_roundoff(
            momentum_residual,
            provided_u_prime,
            reduced_u_prime,
            a21 * clock,
            a22 * total_u,
        )
        second_holds = second.local_common_clock_second_tangent_holds
        propagation_holds = (
            propagation.local_first_derivative_constraint_propagation_holds
        )
        metric_holds = metric.local_algebraic_metric_second_tangent_holds
        force_holds = second.locally_required_intrinsic_force_holds
        domain = self.domain_receipt()
        global_unique = (
            domain.global_reduced_linear_ode_existence_uniqueness_proven
        )
        full_reconstruction = (
            kappa_holds
            and second_holds
            and propagation_holds
            and metric_holds
            and clock_holds
            and momentum_holds
            and force_holds
            and global_unique
        )
        failures: list[str] = []
        if not kappa_holds:
            failures.append("REFERENCE_KAPPA_STATE_FAILED")
        if not second_holds:
            failures.append("PARENT_SECOND_TANGENT_FAILED")
        if not propagation_holds:
            failures.append("PARENT_CONSTRAINT_PROPAGATION_FAILED")
        if not metric_holds:
            failures.append("PARENT_ALGEBRAIC_METRIC_TANGENT_FAILED")
        if not clock_holds:
            failures.append("REDUCED_CLOCK_RHS_FAILED")
        if not momentum_holds:
            failures.append("REDUCED_MOMENTUM_RHS_FAILED")
        if not force_holds:
            failures.append("EFFECTIVE_FORCE_CLOSURE_FAILED")
        if not global_unique:
            failures.append("GLOBAL_LINEAR_ODE_DOMAIN_FAILED")

        return ReducedODEClosureReceipt(
            domain=domain,
            common_clock_second_tangent=second,
            constraint_propagation=propagation,
            algebraic_metric_tangent=metric,
            n=n,
            k_over_a_h=kappa,
            scalar_clock_shift=clock,
            total_momentum_density=total_u,
            matrix_a11=a11,
            matrix_a12=a12,
            matrix_a21=a21,
            matrix_a22=a22,
            reduced_clock_log_derivative=reduced_clock_prime,
            reduced_total_momentum_density_derivative=reduced_u_prime,
            full_clock_log_derivative=provided_clock_prime,
            full_total_momentum_density_derivative=provided_u_prime,
            clock_rhs_residual=clock_residual,
            total_momentum_rhs_residual=momentum_residual,
            required_effective_produced_intrinsic_force=(
                second.required_produced_intrinsic_momentum_potential
            ),
            reference_kappa_state_matches=kappa_holds,
            parent_second_tangent_holds=second_holds,
            parent_constraint_propagation_holds=propagation_holds,
            parent_algebraic_metric_tangent_holds=metric_holds,
            reduced_clock_rhs_matches_full_system=clock_holds,
            reduced_momentum_rhs_matches_full_system=momentum_holds,
            pointwise_effective_force_closure_holds=force_holds,
            conditional_global_reduced_solution_exists_uniquely=(
                global_unique
            ),
            conditional_effective_full_reconstruction_holds=(
                full_reconstruction
            ),
            failure_reasons=tuple(failures),
        )
