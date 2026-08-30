"""Local second-tangent closure with a solved intrinsic transfer force.

The first common-clock tangent satisfies, for the produced component,

    S_p = r_p' T' - 3 H_p psi_n - kappa^2 U_p
          - q (phi+hT) = 0.

Differentiate this identity once.  The spatial-trace Einstein equation fixes
``psi_nn`` and the summed momentum equation fixes ``U_total'``.  Consequently
``D_n S_p=0`` is linear in the intrinsic momentum-transfer potential and
solves one required value ``fhat_p`` at the node.  The reservoir value is its
negative, and its second tangent follows from the summed identity.

This is a local fitted-force theorem.  It is not a microphysical or covariant
law for ``fhat_p(n)``, and it does not prove finite-time propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.finite_quench_common_clock_tangent import (
    CommonClockTangentReceipt,
    FiniteQuenchCommonClockTangent,
)
from examples.physics.finite_quench_einstein_trace_evolution import (
    EinsteinTraceEvolutionReceipt,
    FiniteQuenchEinsteinTraceEvolution,
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


def _finite_product(name: str, *values: float) -> float:
    result = 1.0
    for value in values:
        result *= value
        if not math.isfinite(result):
            raise ValueError(f"{name} left the finite domain")
    return result


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 96.0 * math.ulp(scale)


@dataclass(frozen=True)
class _SecondTangentState:
    produced_enthalpy_derivative: float
    reservoir_enthalpy_derivative: float
    total_enthalpy_derivative: float
    produced_density_second_derivative: float
    reservoir_density_second_derivative: float
    total_density_second_derivative: float
    hubble_second_log_derivative: float
    kappa_squared_derivative: float
    required_total_momentum_density_derivative: float
    required_scalar_clock_second_log_derivative: float
    required_produced_momentum_density_derivative: float
    required_reservoir_momentum_density_derivative: float
    required_produced_intrinsic_momentum_potential: float
    clock_metric_bracket: float
    clock_metric_bracket_derivative: float


@dataclass(frozen=True)
class CommonClockSecondTangentReceipt:
    """One-node second-tangent and locally fitted-force audit receipt."""

    common_clock_tangent: CommonClockTangentReceipt
    einstein_trace_evolution: EinsteinTraceEvolutionReceipt
    produced_enthalpy_derivative: float
    reservoir_enthalpy_derivative: float
    total_enthalpy_derivative: float
    produced_density_second_derivative: float
    reservoir_density_second_derivative: float
    total_density_second_derivative: float
    hubble_second_log_derivative: float
    kappa_squared_derivative: float
    required_total_momentum_density_derivative: float
    provided_total_momentum_density_derivative: float
    provided_scalar_clock_second_log_derivative: float
    required_scalar_clock_second_log_derivative: float
    required_produced_momentum_density_derivative: float
    required_reservoir_momentum_density_derivative: float
    provided_produced_momentum_density_derivative: float
    provided_reservoir_momentum_density_derivative: float
    provided_produced_intrinsic_momentum_potential: float
    required_produced_intrinsic_momentum_potential: float
    scalar_clock_second_derivative_residual: float
    produced_intrinsic_force_residual: float
    produced_momentum_derivative_target_residual: float
    reservoir_momentum_derivative_target_residual: float
    total_momentum_derivative_residual: float
    produced_second_tangent_residual: float
    reservoir_second_tangent_residual: float
    total_second_tangent_residual: float
    cross_residuals: tuple[tuple[str, float], ...]
    parent_first_tangent_holds: bool
    parent_metric_trace_holds: bool
    trace_and_tangent_same_node: bool
    total_momentum_derivative_holds: bool
    scalar_clock_second_derivative_holds: bool
    locally_required_intrinsic_force_holds: bool
    component_momentum_derivatives_hit_second_tangent_targets: bool
    produced_second_tangent_holds: bool
    reservoir_second_tangent_holds: bool
    total_second_tangent_holds: bool
    local_common_clock_second_tangent_holds: bool
    failure_reasons: tuple[str, ...]
    free_declared_inputs: tuple[str, ...] = (
        "scalar_clock_shift",
        "total_momentum_density",
    )
    fitted_node_force_is_not_a_force_law: bool = True
    unreduced_qmu_second_derivative_cancellation_proven: bool = False
    microphysical_covariant_transfer_law_proven: bool = False
    finite_step_constraint_propagation_proven: bool = False
    dimensionless_roles: tuple[tuple[str, str], ...] = (
        ("T_nn", "d^2 T/d(ln a)^2"),
        ("h_n", "d^2 ln H/d(ln a)^2"),
        ("kappa2_n", "d[k/(aH)]^2/d ln a"),
        ("U_A_n", "d U_A/d ln a"),
        ("fhat_A", "a f_A/rho_unit"),
    )
    source: str = (
        "Derivative_of_VMM_2008_Eq_20_common_clock_identity_"
        "closed_locally_with_Eq_21_and_MB_1995_Eq_23c"
    )
    role: str = (
        "CONDITIONAL_LOCAL_COMMON_CLOCK_SECOND_TANGENT_WITH_FITTED_FORCE_"
        "NOT_MICROPHYSICAL_FORCE_LAW_TIME_INTEGRATION_OR_PROPAGATION_PROOF"
    )


class FiniteQuenchCommonClockSecondTangent:
    """Solve and audit the force required by one local second tangent."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _raw_tangent(self, receipt: object) -> CommonClockTangentReceipt:
        if not isinstance(receipt, CommonClockTangentReceipt):
            raise ValueError(
                "common_clock_tangent must be a CommonClockTangentReceipt"
            )
        return FiniteQuenchCommonClockTangent(self.bridge).audit(
            gr_linear_node=receipt.gr_linear_node,
            scalar_clock_log_derivative=receipt.scalar_clock_log_derivative,
        )

    def _raw_trace(
        self,
        receipt: object,
    ) -> EinsteinTraceEvolutionReceipt:
        if not isinstance(receipt, EinsteinTraceEvolutionReceipt):
            raise ValueError(
                "einstein_trace_evolution must be an "
                "EinsteinTraceEvolutionReceipt"
            )
        return FiniteQuenchEinsteinTraceEvolution(self.bridge).audit(
            gr_linear_node=receipt.gr_linear_node,
            lapse_potential_log_derivative=(
                receipt.lapse_potential_log_derivative
            ),
            curvature_potential_second_log_derivative=(
                receipt.provided_curvature_potential_second_log_derivative
            ),
        )

    def _derive(
        self,
        tangent: CommonClockTangentReceipt,
        trace: EinsteinTraceEvolutionReceipt,
    ) -> _SecondTangentState:
        node = tangent.gr_linear_node
        background = node.background
        closure = node.closure
        einstein = node.einstein_constraint
        clock = node.scalar_clock
        rho = background.total_density
        total_enthalpy = background.total_enthalpy
        if rho <= 0.0 or total_enthalpy <= 0.0:
            raise ValueError("second tangent requires positive density and enthalpy")

        q = self.bridge.source(background.n)
        q_prime = self.bridge.source_derivative(background.n)
        w_r = self.bridge.config.w_reservoir
        h_p = background.produced_density + background.produced_pressure
        h_r = background.reservoir_density + background.reservoir_pressure
        h_p_prime = _finite_sum(
            "produced enthalpy derivative",
            background.produced_density_derivative,
            closure.produced_background_pressure_derivative,
        )
        h_r_prime = _finite_sum(
            "reservoir enthalpy derivative",
            background.reservoir_density_derivative,
            closure.reservoir_background_pressure_derivative,
        )
        h_total_prime = _finite_sum(
            "total enthalpy derivative",
            h_p_prime,
            h_r_prime,
        )
        rho_p_second = _finite_sum(
            "produced density second derivative",
            q_prime,
            -3.0 * h_p_prime,
        )
        rho_r_second = _finite_sum(
            "reservoir density second derivative",
            -q_prime,
            -3.0 * (1.0 + w_r) * background.reservoir_density_derivative,
        )
        rho_total_second = _finite_sum(
            "total density second derivative",
            rho_p_second,
            rho_r_second,
        )
        enthalpy_fraction = total_enthalpy / rho
        density_log_prime = background.total_density_derivative / rho
        h_prime = -1.5 * (
            h_total_prime / rho - enthalpy_fraction * density_log_prime
        )
        if not math.isfinite(h_prime):
            raise ValueError("hubble second log derivative left the finite domain")

        kappa_squared = einstein.k_over_a_h_squared
        kappa_squared_prime = _finite_product(
            "kappa squared derivative",
            -2.0,
            1.0 + background.hubble_log_derivative,
            kappa_squared,
        )
        total_u = einstein.total_momentum_density
        delta_pressure = _finite_sum(
            "total pressure perturbation",
            closure.produced_pressure_perturbation,
            closure.reservoir_pressure_perturbation,
        )
        total_u_prime = _finite_sum(
            "required total momentum derivative",
            -(3.0 - background.hubble_log_derivative) * total_u,
            -total_enthalpy * einstein.lapse_potential,
            -delta_pressure,
        )
        t_prime = tangent.scalar_clock_log_derivative
        psi_second = (
            trace.provided_curvature_potential_second_log_derivative
        )
        momentum_numerator_prime = _finite_sum(
            "clock second derivative momentum numerator",
            kappa_squared_prime * total_u,
            kappa_squared * total_u_prime,
        )
        clock_second = _finite_sum(
            "required scalar clock second derivative",
            -psi_second,
            -momentum_numerator_prime / (3.0 * total_enthalpy),
            kappa_squared
            * total_u
            * h_total_prime
            / (3.0 * total_enthalpy * total_enthalpy),
        )
        bracket = _finite_sum(
            "clock-metric bracket",
            einstein.lapse_potential,
            background.hubble_log_derivative * clock.scalar_clock_shift,
        )
        bracket_prime = _finite_sum(
            "clock-metric bracket derivative",
            einstein.curvature_potential_log_derivative,
            h_prime * clock.scalar_clock_shift,
            background.hubble_log_derivative * t_prime,
        )
        numerator_prime = _finite_sum(
            "produced split numerator derivative",
            rho_p_second * t_prime,
            background.produced_density_derivative * clock_second,
            -3.0 * h_p_prime * einstein.curvature_potential_log_derivative,
            -3.0 * h_p * psi_second,
            -q_prime * bracket,
            -q * bracket_prime,
        )
        target_u_p_prime = (
            numerator_prime
            - kappa_squared_prime * einstein.produced_momentum_density
        ) / kappa_squared
        if not math.isfinite(target_u_p_prime):
            raise ValueError("produced momentum target left the finite domain")
        target_u_r_prime = _finite_sum(
            "reservoir momentum target",
            total_u_prime,
            -target_u_p_prime,
        )
        total_velocity = total_u / total_enthalpy
        required_force = _finite_sum(
            "required produced intrinsic force",
            target_u_p_prime,
            (3.0 - background.hubble_log_derivative)
            * einstein.produced_momentum_density,
            h_p * einstein.lapse_potential,
            closure.produced_pressure_perturbation,
            -q * total_velocity,
        )
        return _SecondTangentState(
            produced_enthalpy_derivative=h_p_prime,
            reservoir_enthalpy_derivative=h_r_prime,
            total_enthalpy_derivative=h_total_prime,
            produced_density_second_derivative=rho_p_second,
            reservoir_density_second_derivative=rho_r_second,
            total_density_second_derivative=rho_total_second,
            hubble_second_log_derivative=h_prime,
            kappa_squared_derivative=kappa_squared_prime,
            required_total_momentum_density_derivative=total_u_prime,
            required_scalar_clock_second_log_derivative=clock_second,
            required_produced_momentum_density_derivative=target_u_p_prime,
            required_reservoir_momentum_density_derivative=target_u_r_prime,
            required_produced_intrinsic_momentum_potential=required_force,
            clock_metric_bracket=bracket,
            clock_metric_bracket_derivative=bracket_prime,
        )

    def construct(
        self,
        *,
        n: object,
        k_over_a_h: object,
        scalar_clock_shift: object,
        total_momentum_density: object,
    ) -> CommonClockSecondTangentReceipt:
        """Solve the one-node intrinsic force, then rebuild and audit Eq. 21."""

        tangent_builder = FiniteQuenchCommonClockTangent(self.bridge)
        preliminary = tangent_builder.construct(
            n=n,
            k_over_a_h=k_over_a_h,
            scalar_clock_shift=scalar_clock_shift,
            total_momentum_density=total_momentum_density,
            produced_intrinsic_momentum_potential=0.0,
        )
        preliminary_trace = FiniteQuenchEinsteinTraceEvolution(
            self.bridge
        ).construct(gr_linear_node=preliminary.gr_linear_node)
        preliminary_state = self._derive(preliminary, preliminary_trace)
        tangent = tangent_builder.construct(
            n=n,
            k_over_a_h=k_over_a_h,
            scalar_clock_shift=scalar_clock_shift,
            total_momentum_density=total_momentum_density,
            produced_intrinsic_momentum_potential=(
                preliminary_state.required_produced_intrinsic_momentum_potential
            ),
        )
        trace = FiniteQuenchEinsteinTraceEvolution(self.bridge).construct(
            gr_linear_node=tangent.gr_linear_node
        )
        state = self._derive(tangent, trace)
        return self.audit(
            common_clock_tangent=tangent,
            einstein_trace_evolution=trace,
            scalar_clock_second_log_derivative=(
                state.required_scalar_clock_second_log_derivative
            ),
        )

    def audit(
        self,
        *,
        common_clock_tangent: object,
        einstein_trace_evolution: object,
        scalar_clock_second_log_derivative: object,
    ) -> CommonClockSecondTangentReceipt:
        """Audit the trace, fitted force, Eq. 21 targets, and all S2 residuals."""

        tangent = self._raw_tangent(common_clock_tangent)
        trace = self._raw_trace(einstein_trace_evolution)
        clock_second = _finite_real(
            scalar_clock_second_log_derivative,
            "scalar_clock_second_log_derivative",
        )
        state = self._derive(tangent, trace)
        node = tangent.gr_linear_node
        trace_node = trace.gr_linear_node
        background = node.background
        closure = node.closure
        einstein = node.einstein_constraint
        clock = node.scalar_clock
        momentum = node.momentum_equation
        q = self.bridge.source(background.n)
        q_prime = self.bridge.source_derivative(background.n)
        h_p = background.produced_density + background.produced_pressure
        h_r = background.reservoir_density + background.reservoir_pressure
        kappa_squared = einstein.k_over_a_h_squared
        psi_n = einstein.curvature_potential_log_derivative
        psi_second = trace.provided_curvature_potential_second_log_derivative
        u_p_prime = momentum.provided_produced_momentum_density_derivative
        u_r_prime = momentum.provided_reservoir_momentum_density_derivative
        u_total_prime = momentum.provided_total_momentum_density_derivative
        provided_force = (
            node.transfer_projection.produced_intrinsic_momentum_potential
        )

        cross_pairs = (
            ("n", background.n, trace_node.background.n),
            (
                "kappa",
                einstein.k_over_a_h,
                trace_node.einstein_constraint.k_over_a_h,
            ),
            (
                "clock",
                clock.scalar_clock_shift,
                trace_node.scalar_clock.scalar_clock_shift,
            ),
            (
                "U_p",
                einstein.produced_momentum_density,
                trace_node.einstein_constraint.produced_momentum_density,
            ),
            (
                "U_R",
                einstein.reservoir_momentum_density,
                trace_node.einstein_constraint.reservoir_momentum_density,
            ),
            (
                "fhat_p",
                provided_force,
                trace_node.transfer_projection.produced_intrinsic_momentum_potential,
            ),
        )
        cross_residuals: list[tuple[str, float]] = []
        same_node = True
        for name, left, right in cross_pairs:
            residual = _finite_sum(f"{name} cross residual", left, -right)
            cross_residuals.append((name, residual))
            same_node = same_node and _within_roundoff(
                residual,
                left,
                right,
            )

        clock_second_residual = _finite_sum(
            "scalar clock second derivative residual",
            clock_second,
            -state.required_scalar_clock_second_log_derivative,
        )
        force_residual = _finite_sum(
            "produced intrinsic force residual",
            provided_force,
            -state.required_produced_intrinsic_momentum_potential,
        )
        u_p_target_residual = _finite_sum(
            "produced momentum derivative target residual",
            u_p_prime,
            -state.required_produced_momentum_density_derivative,
        )
        u_r_target_residual = _finite_sum(
            "reservoir momentum derivative target residual",
            u_r_prime,
            -state.required_reservoir_momentum_density_derivative,
        )
        u_total_residual = _finite_sum(
            "total momentum derivative residual",
            u_total_prime,
            -state.required_total_momentum_density_derivative,
        )
        produced_s2 = _finite_sum(
            "produced second tangent residual",
            state.produced_density_second_derivative
            * tangent.scalar_clock_log_derivative,
            background.produced_density_derivative * clock_second,
            -3.0 * state.produced_enthalpy_derivative * psi_n,
            -3.0 * h_p * psi_second,
            -state.kappa_squared_derivative
            * einstein.produced_momentum_density,
            -kappa_squared * u_p_prime,
            -q_prime * state.clock_metric_bracket,
            -q * state.clock_metric_bracket_derivative,
        )
        reservoir_s2 = _finite_sum(
            "reservoir second tangent residual",
            state.reservoir_density_second_derivative
            * tangent.scalar_clock_log_derivative,
            background.reservoir_density_derivative * clock_second,
            -3.0 * state.reservoir_enthalpy_derivative * psi_n,
            -3.0 * h_r * psi_second,
            -state.kappa_squared_derivative
            * einstein.reservoir_momentum_density,
            -kappa_squared * u_r_prime,
            q_prime * state.clock_metric_bracket,
            q * state.clock_metric_bracket_derivative,
        )
        total_s2 = _finite_sum(
            "total second tangent residual",
            state.total_density_second_derivative
            * tangent.scalar_clock_log_derivative,
            background.total_density_derivative * clock_second,
            -3.0 * state.total_enthalpy_derivative * psi_n,
            -3.0 * background.total_enthalpy * psi_second,
            -state.kappa_squared_derivative
            * einstein.total_momentum_density,
            -kappa_squared * u_total_prime,
        )

        clock_second_holds = _within_roundoff(
            clock_second_residual,
            clock_second,
            state.required_scalar_clock_second_log_derivative,
        )
        force_holds = _within_roundoff(
            force_residual,
            provided_force,
            state.required_produced_intrinsic_momentum_potential,
        )
        momentum_targets_hold = (
            _within_roundoff(
                u_p_target_residual,
                u_p_prime,
                state.required_produced_momentum_density_derivative,
            )
            and _within_roundoff(
                u_r_target_residual,
                u_r_prime,
                state.required_reservoir_momentum_density_derivative,
            )
        )
        total_momentum_holds = _within_roundoff(
            u_total_residual,
            u_total_prime,
            state.required_total_momentum_density_derivative,
        )
        p_holds = _within_roundoff(
            produced_s2,
            state.produced_density_second_derivative
            * tangent.scalar_clock_log_derivative,
            background.produced_density_derivative * clock_second,
            3.0 * state.produced_enthalpy_derivative * psi_n,
            3.0 * h_p * psi_second,
            state.kappa_squared_derivative
            * einstein.produced_momentum_density,
            kappa_squared * u_p_prime,
            q_prime * state.clock_metric_bracket,
            q * state.clock_metric_bracket_derivative,
        )
        r_holds = _within_roundoff(
            reservoir_s2,
            state.reservoir_density_second_derivative
            * tangent.scalar_clock_log_derivative,
            background.reservoir_density_derivative * clock_second,
            3.0 * state.reservoir_enthalpy_derivative * psi_n,
            3.0 * h_r * psi_second,
            state.kappa_squared_derivative
            * einstein.reservoir_momentum_density,
            kappa_squared * u_r_prime,
            q_prime * state.clock_metric_bracket,
            q * state.clock_metric_bracket_derivative,
        )
        total_holds = _within_roundoff(
            total_s2,
            state.total_density_second_derivative
            * tangent.scalar_clock_log_derivative,
            background.total_density_derivative * clock_second,
            3.0 * state.total_enthalpy_derivative * psi_n,
            3.0 * background.total_enthalpy * psi_second,
            state.kappa_squared_derivative
            * einstein.total_momentum_density,
            kappa_squared * u_total_prime,
        )
        parent_tangent_holds = tangent.local_common_clock_first_tangent_holds
        parent_trace_holds = trace.one_node_metric_second_derivative_holds
        all_holds = (
            parent_tangent_holds
            and parent_trace_holds
            and same_node
            and total_momentum_holds
            and clock_second_holds
            and force_holds
            and momentum_targets_hold
            and p_holds
            and r_holds
            and total_holds
        )
        failures: list[str] = []
        if not parent_tangent_holds:
            failures.append("PARENT_FIRST_TANGENT_FAILED")
        if not parent_trace_holds:
            failures.append("PARENT_METRIC_TRACE_FAILED")
        if not same_node:
            failures.append("TRACE_TANGENT_NODE_MISMATCH")
        if not total_momentum_holds:
            failures.append("TOTAL_MOMENTUM_DERIVATIVE_FAILED")
        if not clock_second_holds:
            failures.append("SCALAR_CLOCK_SECOND_DERIVATIVE_FAILED")
        if not force_holds:
            failures.append("LOCALLY_REQUIRED_INTRINSIC_FORCE_FAILED")
        if not momentum_targets_hold:
            failures.append("COMPONENT_MOMENTUM_DERIVATIVE_TARGET_FAILED")
        if not p_holds:
            failures.append("PRODUCED_SECOND_TANGENT_FAILED")
        if not r_holds:
            failures.append("RESERVOIR_SECOND_TANGENT_FAILED")
        if not total_holds:
            failures.append("TOTAL_SECOND_TANGENT_FAILED")

        return CommonClockSecondTangentReceipt(
            common_clock_tangent=tangent,
            einstein_trace_evolution=trace,
            produced_enthalpy_derivative=state.produced_enthalpy_derivative,
            reservoir_enthalpy_derivative=state.reservoir_enthalpy_derivative,
            total_enthalpy_derivative=state.total_enthalpy_derivative,
            produced_density_second_derivative=(
                state.produced_density_second_derivative
            ),
            reservoir_density_second_derivative=(
                state.reservoir_density_second_derivative
            ),
            total_density_second_derivative=state.total_density_second_derivative,
            hubble_second_log_derivative=state.hubble_second_log_derivative,
            kappa_squared_derivative=state.kappa_squared_derivative,
            required_total_momentum_density_derivative=(
                state.required_total_momentum_density_derivative
            ),
            provided_total_momentum_density_derivative=u_total_prime,
            provided_scalar_clock_second_log_derivative=clock_second,
            required_scalar_clock_second_log_derivative=(
                state.required_scalar_clock_second_log_derivative
            ),
            required_produced_momentum_density_derivative=(
                state.required_produced_momentum_density_derivative
            ),
            required_reservoir_momentum_density_derivative=(
                state.required_reservoir_momentum_density_derivative
            ),
            provided_produced_momentum_density_derivative=u_p_prime,
            provided_reservoir_momentum_density_derivative=u_r_prime,
            provided_produced_intrinsic_momentum_potential=provided_force,
            required_produced_intrinsic_momentum_potential=(
                state.required_produced_intrinsic_momentum_potential
            ),
            scalar_clock_second_derivative_residual=clock_second_residual,
            produced_intrinsic_force_residual=force_residual,
            produced_momentum_derivative_target_residual=u_p_target_residual,
            reservoir_momentum_derivative_target_residual=u_r_target_residual,
            total_momentum_derivative_residual=u_total_residual,
            produced_second_tangent_residual=produced_s2,
            reservoir_second_tangent_residual=reservoir_s2,
            total_second_tangent_residual=total_s2,
            cross_residuals=tuple(cross_residuals),
            parent_first_tangent_holds=parent_tangent_holds,
            parent_metric_trace_holds=parent_trace_holds,
            trace_and_tangent_same_node=same_node,
            total_momentum_derivative_holds=total_momentum_holds,
            scalar_clock_second_derivative_holds=clock_second_holds,
            locally_required_intrinsic_force_holds=force_holds,
            component_momentum_derivatives_hit_second_tangent_targets=(
                momentum_targets_hold
            ),
            produced_second_tangent_holds=p_holds,
            reservoir_second_tangent_holds=r_holds,
            total_second_tangent_holds=total_holds,
            local_common_clock_second_tangent_holds=all_holds,
            failure_reasons=tuple(failures),
        )
