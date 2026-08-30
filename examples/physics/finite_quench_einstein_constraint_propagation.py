"""Local first-derivative propagation of the scalar Einstein constraints.

For the strict zero-anisotropic-stress branch define

    E = kappa^2 psi + 3(psi_n+phi) + C Delta,
    M = psi_n + phi + C U,
    T = psi_nn + (3+h)psi_n + phi_n + (3+2h)phi - C Delta_P.

Using the flat-GR background identities, the summed VMM energy and momentum
equations, and ``phi=psi``, direct differentiation gives

    M_n = T - (3+h) M,
    E_n = 3 T - (3+2h) E + kappa^2 M.

Thus E=M=T=0 implies E_n=M_n=0 at the same node.  This is a local
first-derivative propagation theorem.  It is not a finite-step integration,
an interval enclosure, or a substitute for a covariant transfer law.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.finite_quench_common_clock_second_tangent import (
    CommonClockSecondTangentReceipt,
    FiniteQuenchCommonClockSecondTangent,
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
class _ConstraintDerivativeState:
    gravity_coupling_derivative: float
    kappa_squared_derivative: float
    direct_energy_constraint_derivative: float
    direct_momentum_constraint_derivative: float
    identity_energy_constraint_derivative: float
    identity_momentum_constraint_derivative: float


@dataclass(frozen=True)
class EinsteinConstraintPropagationReceipt:
    """Audit receipt for the first derivative of the 00 and 0i constraints."""

    common_clock_second_tangent: CommonClockSecondTangentReceipt
    gravity_coupling_derivative: float
    kappa_squared_derivative: float
    provided_energy_constraint_log_derivative: float
    provided_momentum_constraint_log_derivative: float
    direct_energy_constraint_log_derivative: float
    direct_momentum_constraint_log_derivative: float
    identity_energy_constraint_log_derivative: float
    identity_momentum_constraint_log_derivative: float
    energy_derivative_candidate_residual: float
    momentum_derivative_candidate_residual: float
    energy_propagation_identity_residual: float
    momentum_propagation_identity_residual: float
    parent_energy_constraint_residual: float
    parent_momentum_constraint_residual: float
    parent_trace_evolution_residual: float
    parent_second_tangent_holds: bool
    zero_stress_functional_branch_holds: bool
    energy_derivative_candidate_matches_direct: bool
    momentum_derivative_candidate_matches_direct: bool
    energy_propagation_identity_holds: bool
    momentum_propagation_identity_holds: bool
    energy_constraint_derivative_vanishes: bool
    momentum_constraint_derivative_vanishes: bool
    local_first_derivative_constraint_propagation_holds: bool
    failure_reasons: tuple[str, ...]
    finite_step_constraint_propagation_proven: bool = False
    interval_certified: bool = False
    microphysical_covariant_transfer_law_proven: bool = False
    dimensionless_roles: tuple[tuple[str, str], ...] = (
        ("E", "dimensionless 00 Einstein constraint residual"),
        ("M", "dimensionless 0i Einstein constraint residual"),
        ("T", "dimensionless trace-ij Einstein residual"),
        ("D_n", "derivative with respect to ln(a)"),
    )
    source: str = (
        "Contracted_Bianchi_local_identity_realized_with_"
        "VMM_2008_Eqs_20_21_52_54_and_MB_1995_Eq_23c"
    )
    role: str = (
        "CONDITIONAL_LOCAL_FIRST_DERIVATIVE_EINSTEIN_CONSTRAINT_PROPAGATION_"
        "NOT_FINITE_STEP_INTERVAL_OR_COVARIANT_MICROPHYSICAL_PROOF"
    )


class FiniteQuenchEinsteinConstraintPropagation:
    """Construct or audit the local differentiated constraint identities."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _raw_second_tangent(
        self,
        receipt: object,
    ) -> CommonClockSecondTangentReceipt:
        if not isinstance(receipt, CommonClockSecondTangentReceipt):
            raise ValueError(
                "common_clock_second_tangent must be a "
                "CommonClockSecondTangentReceipt"
            )
        return FiniteQuenchCommonClockSecondTangent(self.bridge).audit(
            common_clock_tangent=receipt.common_clock_tangent,
            einstein_trace_evolution=receipt.einstein_trace_evolution,
            scalar_clock_second_log_derivative=(
                receipt.provided_scalar_clock_second_log_derivative
            ),
        )

    @staticmethod
    def _derive(
        second: CommonClockSecondTangentReceipt,
    ) -> _ConstraintDerivativeState:
        tangent = second.common_clock_tangent
        node = tangent.gr_linear_node
        trace = second.einstein_trace_evolution
        background = node.background
        einstein = node.einstein_constraint
        energy = node.energy_equation
        momentum = node.momentum_equation
        h = background.hubble_log_derivative
        coupling = background.gravity_constraint_coupling
        kappa_squared = einstein.k_over_a_h_squared
        coupling_prime = -2.0 * h * coupling
        kappa_squared_prime = -2.0 * (1.0 + h) * kappa_squared
        if not math.isfinite(coupling_prime) or not math.isfinite(
            kappa_squared_prime
        ):
            raise ValueError("constraint coefficients left the finite domain")

        phi_n = trace.lapse_potential_log_derivative
        psi_nn = trace.provided_curvature_potential_second_log_derivative
        direct_energy = _finite_sum(
            "direct energy constraint derivative",
            kappa_squared_prime * einstein.curvature_potential,
            kappa_squared * einstein.curvature_potential_log_derivative,
            3.0 * psi_nn,
            3.0 * phi_n,
            coupling_prime * einstein.total_density_perturbation,
            coupling * energy.provided_total_density_perturbation_derivative,
        )
        direct_momentum = _finite_sum(
            "direct momentum constraint derivative",
            psi_nn,
            phi_n,
            coupling_prime * einstein.total_momentum_density,
            coupling * momentum.provided_total_momentum_density_derivative,
        )
        energy_residual = einstein.energy_constraint_residual
        momentum_residual = einstein.momentum_constraint_residual
        trace_residual = trace.general_spatial_trace_residual
        identity_energy = _finite_sum(
            "energy propagation identity",
            3.0 * trace_residual,
            -(3.0 + 2.0 * h) * energy_residual,
            kappa_squared * momentum_residual,
        )
        identity_momentum = _finite_sum(
            "momentum propagation identity",
            trace_residual,
            -(3.0 + h) * momentum_residual,
        )
        return _ConstraintDerivativeState(
            gravity_coupling_derivative=coupling_prime,
            kappa_squared_derivative=kappa_squared_prime,
            direct_energy_constraint_derivative=direct_energy,
            direct_momentum_constraint_derivative=direct_momentum,
            identity_energy_constraint_derivative=identity_energy,
            identity_momentum_constraint_derivative=identity_momentum,
        )

    def construct(
        self,
        *,
        common_clock_second_tangent: object,
    ) -> EinsteinConstraintPropagationReceipt:
        """Construct the two direct derivatives, then audit the identities."""

        second = self._raw_second_tangent(common_clock_second_tangent)
        state = self._derive(second)
        return self.audit(
            common_clock_second_tangent=second,
            energy_constraint_log_derivative=(
                state.direct_energy_constraint_derivative
            ),
            momentum_constraint_log_derivative=(
                state.direct_momentum_constraint_derivative
            ),
        )

    def audit(
        self,
        *,
        common_clock_second_tangent: object,
        energy_constraint_log_derivative: object,
        momentum_constraint_log_derivative: object,
    ) -> EinsteinConstraintPropagationReceipt:
        """Audit direct differentiation and both propagation identities."""

        second = self._raw_second_tangent(common_clock_second_tangent)
        provided_energy = _finite_real(
            energy_constraint_log_derivative,
            "energy_constraint_log_derivative",
        )
        provided_momentum = _finite_real(
            momentum_constraint_log_derivative,
            "momentum_constraint_log_derivative",
        )
        state = self._derive(second)
        node = second.common_clock_tangent.gr_linear_node
        einstein = node.einstein_constraint
        trace = second.einstein_trace_evolution

        energy_candidate_residual = _finite_sum(
            "energy derivative candidate residual",
            provided_energy,
            -state.direct_energy_constraint_derivative,
        )
        momentum_candidate_residual = _finite_sum(
            "momentum derivative candidate residual",
            provided_momentum,
            -state.direct_momentum_constraint_derivative,
        )
        energy_identity_residual = _finite_sum(
            "energy propagation identity residual",
            state.direct_energy_constraint_derivative,
            -state.identity_energy_constraint_derivative,
        )
        momentum_identity_residual = _finite_sum(
            "momentum propagation identity residual",
            state.direct_momentum_constraint_derivative,
            -state.identity_momentum_constraint_derivative,
        )
        energy_candidate_holds = _within_roundoff(
            energy_candidate_residual,
            provided_energy,
            state.direct_energy_constraint_derivative,
        )
        momentum_candidate_holds = _within_roundoff(
            momentum_candidate_residual,
            provided_momentum,
            state.direct_momentum_constraint_derivative,
        )
        energy_identity_holds = _within_roundoff(
            energy_identity_residual,
            state.direct_energy_constraint_derivative,
            state.identity_energy_constraint_derivative,
        )
        momentum_identity_holds = _within_roundoff(
            momentum_identity_residual,
            state.direct_momentum_constraint_derivative,
            state.identity_momentum_constraint_derivative,
        )
        energy_vanishes = _within_roundoff(
            provided_energy,
            state.direct_energy_constraint_derivative,
            state.identity_energy_constraint_derivative,
        )
        momentum_vanishes = _within_roundoff(
            provided_momentum,
            state.direct_momentum_constraint_derivative,
            state.identity_momentum_constraint_derivative,
        )
        parent_holds = second.local_common_clock_second_tangent_holds
        zero_stress_holds = (
            trace.functional_zero_anisotropic_stress_declared
            and trace.lapse_curvature_derivative_identity_holds
        )
        all_holds = (
            parent_holds
            and zero_stress_holds
            and energy_candidate_holds
            and momentum_candidate_holds
            and energy_identity_holds
            and momentum_identity_holds
            and energy_vanishes
            and momentum_vanishes
        )
        failures: list[str] = []
        if not parent_holds:
            failures.append("PARENT_SECOND_TANGENT_FAILED")
        if not zero_stress_holds:
            failures.append("ZERO_STRESS_FUNCTIONAL_BRANCH_FAILED")
        if not energy_candidate_holds:
            failures.append("ENERGY_DERIVATIVE_CANDIDATE_FAILED")
        if not momentum_candidate_holds:
            failures.append("MOMENTUM_DERIVATIVE_CANDIDATE_FAILED")
        if not energy_identity_holds:
            failures.append("ENERGY_PROPAGATION_IDENTITY_FAILED")
        if not momentum_identity_holds:
            failures.append("MOMENTUM_PROPAGATION_IDENTITY_FAILED")
        if not energy_vanishes:
            failures.append("ENERGY_CONSTRAINT_DERIVATIVE_NONZERO")
        if not momentum_vanishes:
            failures.append("MOMENTUM_CONSTRAINT_DERIVATIVE_NONZERO")

        return EinsteinConstraintPropagationReceipt(
            common_clock_second_tangent=second,
            gravity_coupling_derivative=state.gravity_coupling_derivative,
            kappa_squared_derivative=state.kappa_squared_derivative,
            provided_energy_constraint_log_derivative=provided_energy,
            provided_momentum_constraint_log_derivative=provided_momentum,
            direct_energy_constraint_log_derivative=(
                state.direct_energy_constraint_derivative
            ),
            direct_momentum_constraint_log_derivative=(
                state.direct_momentum_constraint_derivative
            ),
            identity_energy_constraint_log_derivative=(
                state.identity_energy_constraint_derivative
            ),
            identity_momentum_constraint_log_derivative=(
                state.identity_momentum_constraint_derivative
            ),
            energy_derivative_candidate_residual=energy_candidate_residual,
            momentum_derivative_candidate_residual=momentum_candidate_residual,
            energy_propagation_identity_residual=energy_identity_residual,
            momentum_propagation_identity_residual=momentum_identity_residual,
            parent_energy_constraint_residual=(
                einstein.energy_constraint_residual
            ),
            parent_momentum_constraint_residual=(
                einstein.momentum_constraint_residual
            ),
            parent_trace_evolution_residual=(
                trace.general_spatial_trace_residual
            ),
            parent_second_tangent_holds=parent_holds,
            zero_stress_functional_branch_holds=zero_stress_holds,
            energy_derivative_candidate_matches_direct=energy_candidate_holds,
            momentum_derivative_candidate_matches_direct=(
                momentum_candidate_holds
            ),
            energy_propagation_identity_holds=energy_identity_holds,
            momentum_propagation_identity_holds=momentum_identity_holds,
            energy_constraint_derivative_vanishes=energy_vanishes,
            momentum_constraint_derivative_vanishes=momentum_vanishes,
            local_first_derivative_constraint_propagation_holds=all_holds,
            failure_reasons=tuple(failures),
        )
