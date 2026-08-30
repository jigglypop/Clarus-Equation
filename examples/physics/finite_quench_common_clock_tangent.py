"""Local tangent closure for the finite-quench common-clock GR node.

For the strict constant-barotrope branch, impose at one node

    Delta_A = r_A' T,
    delta Qhat_A = (q_A' + h q_A) T,
    Theta_A = -kappa^2 U_A / (r_A+p_A).

Substitution into the VMM energy equation gives the division-free tangent
identity

    r_A' T' = 3(r_A+p_A) psi_n + kappa^2 U_A
               + q_A(phi+hT).

Summing the paired species fixes T', and the produced equation fixes the
momentum split.  This is only a local first-tangent theorem.  It does not show
that the split itself is preserved by the momentum equation at the next node,
nor does it integrate or prove finite-step constraint propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.finite_quench_gr_linear_node import (
    FiniteQuenchGRLinearNode,
    GRLinearNodeReceipt,
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


def _finite_product(name: str, left: float, right: float) -> float:
    result = left * right
    if not math.isfinite(result):
        raise ValueError(f"{name} left the finite domain")
    return result


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 64.0 * math.ulp(scale)


@dataclass(frozen=True)
class CommonClockTangentReceipt:
    """One-node first-tangent and momentum-split audit receipt."""

    gr_linear_node: GRLinearNodeReceipt
    scalar_clock_log_derivative: float
    required_scalar_clock_log_derivative: float
    required_produced_momentum_density: float
    required_reservoir_momentum_density: float
    produced_tangent_equation_residual: float
    reservoir_tangent_equation_residual: float
    total_tangent_equation_residual: float
    scalar_clock_derivative_residual: float
    produced_momentum_split_residual: float
    reservoir_momentum_split_residual: float
    expected_produced_density_perturbation_derivative: float
    expected_reservoir_density_perturbation_derivative: float
    produced_density_tangent_derivative_residual: float
    reservoir_density_tangent_derivative_residual: float
    parent_gr_linear_node_holds: bool
    total_clock_derivative_holds: bool
    produced_tangent_equation_holds: bool
    reservoir_tangent_equation_holds: bool
    momentum_split_holds: bool
    density_tangent_derivatives_match_energy_equations: bool
    local_common_clock_first_tangent_holds: bool
    failure_reasons: tuple[str, ...]
    free_declared_inputs: tuple[str, ...] = (
        "scalar_clock_shift",
        "total_momentum_density",
        "produced_intrinsic_momentum_potential",
    )
    momentum_split_tangent_preservation_proven: bool = False
    finite_step_constraint_propagation_proven: bool = False
    role: str = (
        "CONDITIONAL_LOCAL_COMMON_CLOCK_FIRST_TANGENT_CLOSURE_"
        "NOT_SECOND_TANGENT_TIME_INTEGRATION_OR_PROPAGATION_PROOF"
    )


class FiniteQuenchCommonClockTangent:
    """Construct or audit the local common-clock first tangent."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _raw_node(self, node: object) -> GRLinearNodeReceipt:
        if not isinstance(node, GRLinearNodeReceipt):
            raise ValueError("gr_linear_node must be a GRLinearNodeReceipt")
        return FiniteQuenchGRLinearNode(self.bridge).audit(
            background=node.background,
            scalar_clock=node.scalar_clock,
            closure=node.closure,
            einstein_constraint=node.einstein_constraint,
            transfer_projection=node.transfer_projection,
            energy_equation=node.energy_equation,
            momentum_equation=node.momentum_equation,
        )

    @staticmethod
    def _required_clock_derivative_raw(node: GRLinearNodeReceipt) -> float:
        total_enthalpy = node.background.total_enthalpy
        if total_enthalpy <= 0.0:
            raise ValueError("clock tangent requires positive total enthalpy")
        einstein = node.einstein_constraint
        momentum_term = (
            einstein.k_over_a_h_squared
            * einstein.total_momentum_density
            / (3.0 * total_enthalpy)
        )
        result = -einstein.curvature_potential_log_derivative - momentum_term
        if not math.isfinite(result):
            raise ValueError("required clock derivative left the finite domain")
        return result

    def required_clock_log_derivative(self, node: object) -> float:
        """Return T' required by the summed two-fluid energy equation."""

        return self._required_clock_derivative_raw(self._raw_node(node))

    def _required_split(
        self,
        node: GRLinearNodeReceipt,
        clock_prime: float,
    ) -> tuple[float, float]:
        background = node.background
        clock = node.scalar_clock
        einstein = node.einstein_constraint
        q = self.bridge.source(background.n)
        phi_plus_h_t = _finite_sum(
            "clock-metric source bracket",
            einstein.lapse_potential,
            background.hubble_log_derivative * clock.scalar_clock_shift,
        )
        numerator = _finite_sum(
            "produced tangent momentum numerator",
            background.produced_density_derivative * clock_prime,
            -3.0
            * (background.produced_density + background.produced_pressure)
            * einstein.curvature_potential_log_derivative,
            -q * phi_plus_h_t,
        )
        produced = numerator / einstein.k_over_a_h_squared
        if not math.isfinite(produced):
            raise ValueError("required produced momentum left the finite domain")
        reservoir = _finite_sum(
            "required reservoir momentum",
            einstein.total_momentum_density,
            -produced,
        )
        return produced, reservoir

    def construct(
        self,
        *,
        n: object,
        k_over_a_h: object,
        scalar_clock_shift: object,
        total_momentum_density: object,
        produced_intrinsic_momentum_potential: object,
    ) -> CommonClockTangentReceipt:
        """Solve T' and the component momentum split, then audit both."""

        total_momentum = _finite_real(
            total_momentum_density,
            "total_momentum_density",
        )
        builder = FiniteQuenchGRLinearNode(self.bridge)
        preliminary = builder.construct(
            n=n,
            k_over_a_h=k_over_a_h,
            scalar_clock_shift=scalar_clock_shift,
            produced_momentum_density=0.0,
            reservoir_momentum_density=total_momentum,
            produced_intrinsic_momentum_potential=(
                produced_intrinsic_momentum_potential
            ),
        )
        clock_prime = self._required_clock_derivative_raw(preliminary)
        momentum_p, momentum_r = self._required_split(
            preliminary,
            clock_prime,
        )
        node = builder.construct(
            n=n,
            k_over_a_h=k_over_a_h,
            scalar_clock_shift=scalar_clock_shift,
            produced_momentum_density=momentum_p,
            reservoir_momentum_density=momentum_r,
            produced_intrinsic_momentum_potential=(
                produced_intrinsic_momentum_potential
            ),
        )
        return self.audit(
            gr_linear_node=node,
            scalar_clock_log_derivative=clock_prime,
        )

    def audit(
        self,
        *,
        gr_linear_node: object,
        scalar_clock_log_derivative: object,
    ) -> CommonClockTangentReceipt:
        """Audit T', both component tangents, and the solved momentum split."""

        node = self._raw_node(gr_linear_node)
        clock_prime = _finite_real(
            scalar_clock_log_derivative,
            "scalar_clock_log_derivative",
        )
        required_clock_prime = self._required_clock_derivative_raw(node)
        required_u_p, required_u_r = self._required_split(
            node,
            required_clock_prime,
        )
        background = node.background
        clock = node.scalar_clock
        einstein = node.einstein_constraint
        energy = node.energy_equation
        q = self.bridge.source(background.n)
        q_prime = self.bridge.source_derivative(background.n)
        kappa_squared = einstein.k_over_a_h_squared
        h_p = background.produced_density + background.produced_pressure
        h_r = background.reservoir_density + background.reservoir_pressure
        phi_plus_h_t = _finite_sum(
            "clock-metric source bracket",
            einstein.lapse_potential,
            background.hubble_log_derivative * clock.scalar_clock_shift,
        )
        p_tangent = _finite_sum(
            "produced tangent residual",
            background.produced_density_derivative * clock_prime,
            -3.0 * h_p * einstein.curvature_potential_log_derivative,
            -kappa_squared * einstein.produced_momentum_density,
            -q * phi_plus_h_t,
        )
        r_tangent = _finite_sum(
            "reservoir tangent residual",
            background.reservoir_density_derivative * clock_prime,
            -3.0 * h_r * einstein.curvature_potential_log_derivative,
            -kappa_squared * einstein.reservoir_momentum_density,
            q * phi_plus_h_t,
        )
        total_tangent = _finite_sum(
            "total tangent residual",
            background.total_density_derivative * clock_prime,
            -3.0
            * background.total_enthalpy
            * einstein.curvature_potential_log_derivative,
            -kappa_squared * einstein.total_momentum_density,
        )
        clock_prime_residual = _finite_sum(
            "clock derivative residual",
            clock_prime,
            -required_clock_prime,
        )
        split_p_residual = _finite_sum(
            "produced momentum split residual",
            einstein.produced_momentum_density,
            -required_u_p,
        )
        split_r_residual = _finite_sum(
            "reservoir momentum split residual",
            einstein.reservoir_momentum_density,
            -required_u_r,
        )

        rho_p_second = _finite_sum(
            "produced density second derivative",
            q_prime,
            -3.0 * background.produced_density_derivative,
        )
        rho_r_second = _finite_sum(
            "reservoir density second derivative",
            -q_prime,
            -3.0
            * (1.0 + self.bridge.config.w_reservoir)
            * background.reservoir_density_derivative,
        )
        expected_delta_prime_p = _finite_sum(
            "produced clock density derivative",
            rho_p_second * clock.scalar_clock_shift,
            background.produced_density_derivative * clock_prime,
        )
        expected_delta_prime_r = _finite_sum(
            "reservoir clock density derivative",
            rho_r_second * clock.scalar_clock_shift,
            background.reservoir_density_derivative * clock_prime,
        )
        delta_prime_residual_p = _finite_sum(
            "produced density tangent derivative residual",
            energy.provided_produced_density_perturbation_derivative,
            -expected_delta_prime_p,
        )
        delta_prime_residual_r = _finite_sum(
            "reservoir density tangent derivative residual",
            energy.provided_reservoir_density_perturbation_derivative,
            -expected_delta_prime_r,
        )
        clock_holds = _within_roundoff(
            clock_prime_residual,
            clock_prime,
            required_clock_prime,
        )
        p_holds = _within_roundoff(
            p_tangent,
            background.produced_density_derivative * clock_prime,
            3.0 * h_p * einstein.curvature_potential_log_derivative,
            kappa_squared * einstein.produced_momentum_density,
            q * phi_plus_h_t,
        )
        r_holds = _within_roundoff(
            r_tangent,
            background.reservoir_density_derivative * clock_prime,
            3.0 * h_r * einstein.curvature_potential_log_derivative,
            kappa_squared * einstein.reservoir_momentum_density,
            q * phi_plus_h_t,
        )
        total_holds = _within_roundoff(
            total_tangent,
            background.total_density_derivative * clock_prime,
            3.0
            * background.total_enthalpy
            * einstein.curvature_potential_log_derivative,
            kappa_squared * einstein.total_momentum_density,
        )
        split_holds = (
            _within_roundoff(
                split_p_residual,
                einstein.produced_momentum_density,
                required_u_p,
            )
            and _within_roundoff(
                split_r_residual,
                einstein.reservoir_momentum_density,
                required_u_r,
            )
        )
        derivative_holds = (
            _within_roundoff(
                delta_prime_residual_p,
                energy.provided_produced_density_perturbation_derivative,
                expected_delta_prime_p,
            )
            and _within_roundoff(
                delta_prime_residual_r,
                energy.provided_reservoir_density_perturbation_derivative,
                expected_delta_prime_r,
            )
        )
        parent_holds = node.full_declared_gr_linear_node_holds
        all_holds = (
            parent_holds
            and clock_holds
            and p_holds
            and r_holds
            and total_holds
            and split_holds
            and derivative_holds
        )
        failures: list[str] = []
        if not parent_holds:
            failures.append("PARENT_GR_LINEAR_NODE_FAILED")
        if not clock_holds or not total_holds:
            failures.append("TOTAL_CLOCK_DERIVATIVE_FAILED")
        if not p_holds:
            failures.append("PRODUCED_CLOCK_TANGENT_FAILED")
        if not r_holds:
            failures.append("RESERVOIR_CLOCK_TANGENT_FAILED")
        if not split_holds:
            failures.append("COMPONENT_MOMENTUM_SPLIT_FAILED")
        if not derivative_holds:
            failures.append("ENERGY_DERIVATIVE_NOT_CLOCK_TANGENT")

        return CommonClockTangentReceipt(
            gr_linear_node=node,
            scalar_clock_log_derivative=clock_prime,
            required_scalar_clock_log_derivative=required_clock_prime,
            required_produced_momentum_density=required_u_p,
            required_reservoir_momentum_density=required_u_r,
            produced_tangent_equation_residual=p_tangent,
            reservoir_tangent_equation_residual=r_tangent,
            total_tangent_equation_residual=total_tangent,
            scalar_clock_derivative_residual=clock_prime_residual,
            produced_momentum_split_residual=split_p_residual,
            reservoir_momentum_split_residual=split_r_residual,
            expected_produced_density_perturbation_derivative=(
                expected_delta_prime_p
            ),
            expected_reservoir_density_perturbation_derivative=(
                expected_delta_prime_r
            ),
            produced_density_tangent_derivative_residual=(
                delta_prime_residual_p
            ),
            reservoir_density_tangent_derivative_residual=(
                delta_prime_residual_r
            ),
            parent_gr_linear_node_holds=parent_holds,
            total_clock_derivative_holds=(clock_holds and total_holds),
            produced_tangent_equation_holds=p_holds,
            reservoir_tangent_equation_holds=r_holds,
            momentum_split_holds=split_holds,
            density_tangent_derivatives_match_energy_equations=(
                derivative_holds
            ),
            local_common_clock_first_tangent_holds=all_holds,
            failure_reasons=tuple(failures),
        )
