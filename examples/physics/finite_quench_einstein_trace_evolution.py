"""One-node spatial-trace Einstein evolution for the strict finite-quench branch.

In VMM notation and Newtonian gauge, the general flat-FLRW scalar trace is

    psi_nn + (3+h) psi_n + phi_n + (3+2h) phi
        + kappa^2 (psi-phi)/3 = C Delta_P.

The strict barotropic branch declares zero anisotropic stress as a
constitutive closure at every node, not merely as an accidental zero at one
node.  The traceless Einstein equation then gives ``phi=psi`` as a functional
identity and therefore ``phi_n=psi_n``.  On that narrow branch,

    psi_nn + (4+h) psi_n + (3+2h) psi = C Delta_P.

This module solves only the second metric derivative at one node.  It does not
integrate the perturbations or prove finite-step constraint propagation.
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


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 64.0 * math.ulp(scale)


@dataclass(frozen=True)
class EinsteinTraceEvolutionReceipt:
    """Raw one-node audit of the zero-stress spatial Einstein trace."""

    gr_linear_node: GRLinearNodeReceipt
    total_pressure_perturbation: float
    lapse_potential_log_derivative: float
    provided_curvature_potential_second_log_derivative: float
    required_curvature_potential_second_log_derivative: float
    lapse_curvature_derivative_residual: float
    general_spatial_trace_residual: float
    reduced_zero_stress_trace_residual: float
    parent_gr_linear_node_holds: bool
    functional_zero_anisotropic_stress_declared: bool
    lapse_curvature_derivative_identity_holds: bool
    general_spatial_trace_holds: bool
    reduced_zero_stress_trace_holds: bool
    one_node_metric_second_derivative_holds: bool
    failure_reasons: tuple[str, ...]
    dimensionless_roles: tuple[tuple[str, str], ...] = (
        ("psi_nn", "d^2 psi/d(ln a)^2"),
        ("h", "d ln H/d ln a"),
        ("kappa", "k/(aH)"),
        ("C", "4 pi G rho_unit/H^2"),
        ("Delta_P", "delta P/rho_unit"),
    )
    zero_stress_is_constitutive_not_pointwise: bool = True
    finite_step_constraint_propagation_proven: bool = False
    source: str = (
        "Ma_Bertschinger_1995_Eqs_23c_23d_and_"
        "Valiviita_Majerotto_Maartens_2008_Eqs_54_55"
    )
    role: str = (
        "CONDITIONAL_ZERO_STRESS_ONE_NODE_SPATIAL_TRACE_EVOLUTION_"
        "NOT_TIME_INTEGRATION_OR_CONSTRAINT_PROPAGATION_PROOF"
    )


class FiniteQuenchEinsteinTraceEvolution:
    """Construct or audit the second metric derivative on the strict branch."""

    _STRICT_MODEL = "strict_dust_plus_causal_constant_barotrope_zero_pi"

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
    def _required_second_derivative(node: GRLinearNodeReceipt) -> float:
        background = node.background
        einstein = node.einstein_constraint
        closure = node.closure
        delta_pressure = _finite_sum(
            "total pressure perturbation",
            closure.produced_pressure_perturbation,
            closure.reservoir_pressure_perturbation,
        )
        result = _finite_sum(
            "required curvature second derivative",
            background.gravity_constraint_coupling * delta_pressure,
            -(4.0 + background.hubble_log_derivative)
            * einstein.curvature_potential_log_derivative,
            -(3.0 + 2.0 * background.hubble_log_derivative)
            * einstein.curvature_potential,
        )
        return result

    def construct(
        self,
        *,
        gr_linear_node: object,
    ) -> EinsteinTraceEvolutionReceipt:
        """Solve ``psi_nn`` from the reduced zero-stress trace equation."""

        node = self._raw_node(gr_linear_node)
        required = self._required_second_derivative(node)
        return self.audit(
            gr_linear_node=node,
            lapse_potential_log_derivative=(
                node.einstein_constraint.curvature_potential_log_derivative
            ),
            curvature_potential_second_log_derivative=required,
        )

    def audit(
        self,
        *,
        gr_linear_node: object,
        lapse_potential_log_derivative: object,
        curvature_potential_second_log_derivative: object,
    ) -> EinsteinTraceEvolutionReceipt:
        """Audit independently supplied ``phi_n`` and ``psi_nn`` candidates."""

        node = self._raw_node(gr_linear_node)
        phi_n = _finite_real(
            lapse_potential_log_derivative,
            "lapse_potential_log_derivative",
        )
        psi_nn = _finite_real(
            curvature_potential_second_log_derivative,
            "curvature_potential_second_log_derivative",
        )
        background = node.background
        closure = node.closure
        einstein = node.einstein_constraint
        h = background.hubble_log_derivative
        coupling = background.gravity_constraint_coupling
        kappa_squared = einstein.k_over_a_h_squared
        psi = einstein.curvature_potential
        phi = einstein.lapse_potential
        psi_n = einstein.curvature_potential_log_derivative
        delta_pressure = _finite_sum(
            "total pressure perturbation",
            closure.produced_pressure_perturbation,
            closure.reservoir_pressure_perturbation,
        )
        required = self._required_second_derivative(node)
        derivative_identity_residual = _finite_sum(
            "lapse-curvature derivative residual",
            phi_n,
            -psi_n,
        )
        general_residual = _finite_sum(
            "general spatial trace residual",
            psi_nn,
            (3.0 + h) * psi_n,
            phi_n,
            (3.0 + 2.0 * h) * phi,
            (kappa_squared / 3.0) * (psi - phi),
            -coupling * delta_pressure,
        )
        reduced_residual = _finite_sum(
            "reduced zero-stress trace residual",
            psi_nn,
            (4.0 + h) * psi_n,
            (3.0 + 2.0 * h) * psi,
            -coupling * delta_pressure,
        )
        functional_zero_stress = (
            closure.model == self._STRICT_MODEL
            and closure.zero_anisotropic_stress_holds
            and einstein.zero_stress_traceless_spatial_constraint_holds
        )
        derivative_identity_holds = _within_roundoff(
            derivative_identity_residual,
            phi_n,
            psi_n,
        )
        general_holds = _within_roundoff(
            general_residual,
            psi_nn,
            (3.0 + h) * psi_n,
            phi_n,
            (3.0 + 2.0 * h) * phi,
            (kappa_squared / 3.0) * (psi - phi),
            coupling * delta_pressure,
        )
        reduced_holds = _within_roundoff(
            reduced_residual,
            psi_nn,
            (4.0 + h) * psi_n,
            (3.0 + 2.0 * h) * psi,
            coupling * delta_pressure,
        )
        parent_holds = node.full_declared_gr_linear_node_holds
        all_holds = (
            parent_holds
            and functional_zero_stress
            and derivative_identity_holds
            and general_holds
            and reduced_holds
        )
        failures: list[str] = []
        if not parent_holds:
            failures.append("PARENT_GR_LINEAR_NODE_FAILED")
        if not functional_zero_stress:
            failures.append("FUNCTIONAL_ZERO_STRESS_CLOSURE_FAILED")
        if not derivative_identity_holds:
            failures.append("LAPSE_CURVATURE_DERIVATIVE_IDENTITY_FAILED")
        if not general_holds:
            failures.append("GENERAL_SPATIAL_TRACE_FAILED")
        if not reduced_holds:
            failures.append("REDUCED_ZERO_STRESS_TRACE_FAILED")

        return EinsteinTraceEvolutionReceipt(
            gr_linear_node=node,
            total_pressure_perturbation=delta_pressure,
            lapse_potential_log_derivative=phi_n,
            provided_curvature_potential_second_log_derivative=psi_nn,
            required_curvature_potential_second_log_derivative=required,
            lapse_curvature_derivative_residual=derivative_identity_residual,
            general_spatial_trace_residual=general_residual,
            reduced_zero_stress_trace_residual=reduced_residual,
            parent_gr_linear_node_holds=parent_holds,
            functional_zero_anisotropic_stress_declared=functional_zero_stress,
            lapse_curvature_derivative_identity_holds=derivative_identity_holds,
            general_spatial_trace_holds=general_holds,
            reduced_zero_stress_trace_holds=reduced_holds,
            one_node_metric_second_derivative_holds=all_holds,
            failure_reasons=tuple(failures),
        )
