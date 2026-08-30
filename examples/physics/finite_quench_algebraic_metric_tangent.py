"""Tangent consistency of the algebraically reconstructed metric potentials.

The k>0 Einstein constraints reconstruct ``psi`` and ``psi_n`` algebraically
at every node.  A time-dependent solution also needs those algebraic values to
be actual derivatives along the fluid flow.  This module checks both links:

    D_n [ C(3U-Delta)/kappa^2 ] = psi_n,
    D_n [ -phi-CU ] = psi_nn.

The first derivative uses the summed energy and momentum equations.  The
second uses the differentiated 0i constraint and the zero-stress identity
``phi_n=psi_n``.  This remains a local tangent audit, not a numerical
integration or a microphysical transfer derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.finite_quench_einstein_constraint_propagation import (
    EinsteinConstraintPropagationReceipt,
    FiniteQuenchEinsteinConstraintPropagation,
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
class AlgebraicMetricTangentReceipt:
    """Audit the first two derivatives of algebraic metric reconstruction."""

    constraint_propagation: EinsteinConstraintPropagationReceipt
    density_momentum_numerator: float
    density_momentum_numerator_derivative: float
    provided_curvature_potential_log_derivative: float
    direct_algebraic_curvature_potential_log_derivative: float
    provided_curvature_potential_second_log_derivative: float
    direct_algebraic_curvature_potential_second_log_derivative: float
    curvature_first_tangent_residual: float
    curvature_second_tangent_residual: float
    parent_constraint_propagation_holds: bool
    functional_zero_stress_branch_holds: bool
    algebraic_curvature_first_tangent_holds: bool
    algebraic_curvature_second_tangent_holds: bool
    local_algebraic_metric_second_tangent_holds: bool
    failure_reasons: tuple[str, ...]
    finite_step_metric_evolution_proven: bool = False
    interval_certified: bool = False
    dimensionless_roles: tuple[tuple[str, str], ...] = (
        ("psi", "dimensionless curvature potential"),
        ("psi_n", "d psi/d ln(a)"),
        ("psi_nn", "d^2 psi/d[ln(a)]^2"),
        ("C", "4 pi G rho_unit/H^2"),
        ("kappa", "k/(aH)"),
    )
    source: str = (
        "Derivative_of_VMM_2008_Eqs_52_53_under_total_Eqs_20_21"
    )
    role: str = (
        "CONDITIONAL_LOCAL_ALGEBRAIC_METRIC_SECOND_TANGENT_"
        "NOT_FINITE_STEP_INTERVAL_OR_MICROPHYSICAL_PROOF"
    )


class FiniteQuenchAlgebraicMetricTangent:
    """Construct or audit metric tangency along the declared fluid flow."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _raw_propagation(
        self,
        receipt: object,
    ) -> EinsteinConstraintPropagationReceipt:
        if not isinstance(receipt, EinsteinConstraintPropagationReceipt):
            raise ValueError(
                "constraint_propagation must be an "
                "EinsteinConstraintPropagationReceipt"
            )
        return FiniteQuenchEinsteinConstraintPropagation(self.bridge).audit(
            common_clock_second_tangent=(
                receipt.common_clock_second_tangent
            ),
            energy_constraint_log_derivative=(
                receipt.provided_energy_constraint_log_derivative
            ),
            momentum_constraint_log_derivative=(
                receipt.provided_momentum_constraint_log_derivative
            ),
        )

    @staticmethod
    def _direct_derivatives(
        propagation: EinsteinConstraintPropagationReceipt,
    ) -> tuple[float, float, float, float]:
        second = propagation.common_clock_second_tangent
        node = second.common_clock_tangent.gr_linear_node
        background = node.background
        einstein = node.einstein_constraint
        energy = node.energy_equation
        momentum = node.momentum_equation
        coupling = background.gravity_constraint_coupling
        coupling_prime = propagation.gravity_coupling_derivative
        kappa_squared = einstein.k_over_a_h_squared
        kappa_squared_prime = propagation.kappa_squared_derivative
        numerator = _finite_sum(
            "algebraic curvature numerator",
            3.0 * einstein.total_momentum_density,
            -einstein.total_density_perturbation,
        )
        numerator_prime = _finite_sum(
            "algebraic curvature numerator derivative",
            3.0 * momentum.provided_total_momentum_density_derivative,
            -energy.provided_total_density_perturbation_derivative,
        )
        first = _finite_sum(
            "direct algebraic curvature derivative",
            coupling_prime * numerator / kappa_squared,
            coupling * numerator_prime / kappa_squared,
            -coupling
            * numerator
            * kappa_squared_prime
            / (kappa_squared * kappa_squared),
        )
        second_derivative = _finite_sum(
            "direct algebraic curvature second derivative",
            -first,
            -coupling_prime * einstein.total_momentum_density,
            -coupling * momentum.provided_total_momentum_density_derivative,
        )
        return numerator, numerator_prime, first, second_derivative

    def construct(
        self,
        *,
        constraint_propagation: object,
    ) -> AlgebraicMetricTangentReceipt:
        """Use declared metric derivatives as candidates and audit them."""

        propagation = self._raw_propagation(constraint_propagation)
        second = propagation.common_clock_second_tangent
        node = second.common_clock_tangent.gr_linear_node
        trace = second.einstein_trace_evolution
        return self.audit(
            constraint_propagation=propagation,
            curvature_potential_log_derivative=(
                node.einstein_constraint.curvature_potential_log_derivative
            ),
            curvature_potential_second_log_derivative=(
                trace.provided_curvature_potential_second_log_derivative
            ),
        )

    def audit(
        self,
        *,
        constraint_propagation: object,
        curvature_potential_log_derivative: object,
        curvature_potential_second_log_derivative: object,
    ) -> AlgebraicMetricTangentReceipt:
        """Audit independent ``psi_n`` and ``psi_nn`` candidates."""

        propagation = self._raw_propagation(constraint_propagation)
        provided_first = _finite_real(
            curvature_potential_log_derivative,
            "curvature_potential_log_derivative",
        )
        provided_second = _finite_real(
            curvature_potential_second_log_derivative,
            "curvature_potential_second_log_derivative",
        )
        numerator, numerator_prime, direct_first, direct_second = (
            self._direct_derivatives(propagation)
        )
        first_residual = _finite_sum(
            "curvature first tangent residual",
            provided_first,
            -direct_first,
        )
        second_residual = _finite_sum(
            "curvature second tangent residual",
            provided_second,
            -direct_second,
        )
        first_holds = _within_roundoff(
            first_residual,
            provided_first,
            direct_first,
        )
        second_holds = _within_roundoff(
            second_residual,
            provided_second,
            direct_second,
        )
        parent_holds = (
            propagation.local_first_derivative_constraint_propagation_holds
        )
        zero_stress_holds = (
            propagation.zero_stress_functional_branch_holds
        )
        all_holds = (
            parent_holds
            and zero_stress_holds
            and first_holds
            and second_holds
        )
        failures: list[str] = []
        if not parent_holds:
            failures.append("PARENT_CONSTRAINT_PROPAGATION_FAILED")
        if not zero_stress_holds:
            failures.append("FUNCTIONAL_ZERO_STRESS_BRANCH_FAILED")
        if not first_holds:
            failures.append("ALGEBRAIC_CURVATURE_FIRST_TANGENT_FAILED")
        if not second_holds:
            failures.append("ALGEBRAIC_CURVATURE_SECOND_TANGENT_FAILED")

        return AlgebraicMetricTangentReceipt(
            constraint_propagation=propagation,
            density_momentum_numerator=numerator,
            density_momentum_numerator_derivative=numerator_prime,
            provided_curvature_potential_log_derivative=provided_first,
            direct_algebraic_curvature_potential_log_derivative=direct_first,
            provided_curvature_potential_second_log_derivative=provided_second,
            direct_algebraic_curvature_potential_second_log_derivative=(
                direct_second
            ),
            curvature_first_tangent_residual=first_residual,
            curvature_second_tangent_residual=second_residual,
            parent_constraint_propagation_holds=parent_holds,
            functional_zero_stress_branch_holds=zero_stress_holds,
            algebraic_curvature_first_tangent_holds=first_holds,
            algebraic_curvature_second_tangent_holds=second_holds,
            local_algebraic_metric_second_tangent_holds=all_holds,
            failure_reasons=tuple(failures),
        )
