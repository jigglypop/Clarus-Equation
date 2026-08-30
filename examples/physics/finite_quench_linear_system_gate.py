"""Cross-receipt gate for one finite-quench linear perturbation node.

The lower-Qmu projection, normalized energy equation, and normalized momentum
equation test different obligations.  None may stand in for the others.  This
module checks that their receipts describe the same bridge/node and combines
their independent statuses without promoting the result to an integrated
Einstein--Boltzmann solution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from examples.physics.finite_quench_linear_energy_equation import (
    LinearEnergyEquationReceipt,
)
from examples.physics.finite_quench_linear_momentum_equation import (
    LinearMomentumEquationReceipt,
)
from examples.physics.finite_quench_qmu_projection_ledger import (
    LowerQmuProjectionReceipt,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
)


def _consistent(left: float, right: float) -> tuple[float, bool]:
    residual = left - right
    if not math.isfinite(residual):
        return residual, False
    scale = max(1.0, abs(left), abs(right))
    return residual, abs(residual) <= 64.0 * math.ulp(scale)


def _zero_with_terms(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 64.0 * math.ulp(scale)


@dataclass(frozen=True)
class LinearSystemNodeGateReceipt:
    """Stable-snapshot cross-gate for one declared scalar Fourier node."""

    n: float
    cross_receipt_residuals: tuple[tuple[str, float], ...]
    cross_receipt_consistency_holds: bool
    projection_background_pair_holds: bool
    projection_physical_energy_pair_holds: bool
    projection_intrinsic_momentum_pair_holds: bool
    projection_lower_time_pair_holds: bool
    projection_lower_spatial_pair_holds: bool
    projection_all_component_pairs_hold: bool
    projection_common_physical_clock_holds: bool
    energy_equations_and_exchange_hold: bool
    momentum_equations_and_exchange_hold: bool
    nondegenerate_total_energy_frame_holds: bool
    algebraic_energy_momentum_node_holds: bool
    common_clock_energy_momentum_node_holds: bool
    full_declared_nondegenerate_node_holds: bool
    failure_reasons: tuple[str, ...]
    role: str = (
        "CONDITIONAL_CROSS_RECEIPT_LINEAR_NODE_GATE_"
        "NOT_INTEGRATED_EINSTEIN_BOLTZMANN_OR_OBSERVABLE_SOLUTION"
    )


class FiniteQuenchLinearSystemNodeGate:
    """Require Qmu, energy, momentum, clock, and frame receipts together."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def audit(
        self,
        *,
        transfer_projection: object,
        energy_equation: object,
        momentum_equation: object,
    ) -> LinearSystemNodeGateReceipt:
        if not isinstance(transfer_projection, LowerQmuProjectionReceipt):
            raise ValueError("transfer_projection must be a LowerQmuProjectionReceipt")
        if not isinstance(energy_equation, LinearEnergyEquationReceipt):
            raise ValueError("energy_equation must be a LinearEnergyEquationReceipt")
        if not isinstance(momentum_equation, LinearMomentumEquationReceipt):
            raise ValueError("momentum_equation must be a LinearMomentumEquationReceipt")

        projection = transfer_projection
        energy = energy_equation
        momentum = momentum_equation
        rho_p = self.bridge.production_density(projection.n)
        rho_r = self.bridge.reservoir_density(projection.n)
        pressure_p = 0.0
        pressure_r = self.bridge.config.w_reservoir * rho_r
        enthalpy_p = rho_p + pressure_p
        enthalpy_r = rho_r + pressure_r
        expected_q = self.bridge.source(projection.n)
        expected_q_prime = self.bridge.source_derivative(projection.n)
        try:
            kappa_squared = projection.k_over_a_h**2
        except OverflowError as error:
            raise ValueError("cross-gate kappa squared left the finite domain") from error
        if not math.isfinite(kappa_squared):
            raise ValueError("cross-gate kappa squared left the finite domain")
        expected_energy_source_p = (
            projection.produced_background_q * projection.lapse_potential
            + projection.produced_physical_energy_perturbation
        )
        expected_energy_source_r = (
            projection.reservoir_background_q * projection.lapse_potential
            + projection.reservoir_physical_energy_perturbation
        )
        expected_momentum_source_p = (
            projection.produced_background_q
            * projection.normalized_total_velocity_potential
            + projection.produced_intrinsic_momentum_potential
        )
        expected_momentum_source_r = (
            projection.reservoir_background_q
            * projection.normalized_total_velocity_potential
            + projection.reservoir_intrinsic_momentum_potential
        )

        comparisons = (
            ("energy_n", energy.n, projection.n),
            ("momentum_n", momentum.n, projection.n),
            ("projection_q_p", projection.produced_background_q, expected_q),
            ("projection_q_r", projection.reservoir_background_q, -expected_q),
            (
                "projection_q_prime",
                projection.produced_background_q_derivative,
                expected_q_prime,
            ),
            ("energy_phi", energy.lapse_potential, projection.lapse_potential),
            ("momentum_phi", momentum.lapse_potential, projection.lapse_potential),
            ("momentum_kappa", momentum.k_over_a_h, projection.k_over_a_h),
            (
                "momentum_hubble_log_derivative",
                momentum.hubble_log_derivative,
                projection.hubble_log_derivative,
            ),
            (
                "momentum_total_velocity",
                momentum.normalized_total_velocity_potential,
                projection.normalized_total_velocity_potential,
            ),
            (
                "energy_source_p",
                energy.produced_energy_transfer_source,
                expected_energy_source_p,
            ),
            (
                "energy_source_r",
                energy.reservoir_energy_transfer_source,
                expected_energy_source_r,
            ),
            (
                "momentum_source_p",
                momentum.produced_momentum_transfer_source,
                expected_momentum_source_p,
            ),
            (
                "momentum_source_r",
                momentum.reservoir_momentum_transfer_source,
                expected_momentum_source_r,
            ),
            ("energy_rho_p", energy.produced_background_density, rho_p),
            ("energy_rho_r", energy.reservoir_background_density, rho_r),
            ("energy_pressure_p", energy.produced_background_pressure, pressure_p),
            ("energy_pressure_r", energy.reservoir_background_pressure, pressure_r),
            (
                "energy_enthalpy_p",
                energy.produced_background_enthalpy,
                enthalpy_p,
            ),
            (
                "energy_enthalpy_r",
                energy.reservoir_background_enthalpy,
                enthalpy_r,
            ),
            ("momentum_rho_p", momentum.produced_background_density, rho_p),
            ("momentum_rho_r", momentum.reservoir_background_density, rho_r),
            (
                "momentum_pressure_p",
                momentum.produced_background_pressure,
                pressure_p,
            ),
            (
                "momentum_pressure_r",
                momentum.reservoir_background_pressure,
                pressure_r,
            ),
            (
                "momentum_enthalpy_p",
                momentum.produced_background_enthalpy,
                enthalpy_p,
            ),
            (
                "momentum_enthalpy_r",
                momentum.reservoir_background_enthalpy,
                enthalpy_r,
            ),
            (
                "produced_pressure_perturbation",
                energy.produced_pressure_perturbation,
                momentum.produced_pressure_perturbation,
            ),
            (
                "reservoir_pressure_perturbation",
                energy.reservoir_pressure_perturbation,
                momentum.reservoir_pressure_perturbation,
            ),
            (
                "produced_velocity_state",
                enthalpy_p * energy.produced_theta_over_a_h,
                -kappa_squared * momentum.produced_momentum_density,
            ),
            (
                "reservoir_velocity_state",
                enthalpy_r * energy.reservoir_theta_over_a_h,
                -kappa_squared * momentum.reservoir_momentum_density,
            ),
        )
        residuals: list[tuple[str, float]] = []
        consistent = True
        for name, left, right in comparisons:
            residual, holds = _consistent(left, right)
            residuals.append((name, residual))
            consistent = consistent and holds

        try:
            projection_background_pair_residual = math.fsum(
                (
                    projection.produced_background_q,
                    projection.reservoir_background_q,
                )
            )
            projection_energy_pair_residual = math.fsum(
                (
                    projection.produced_physical_energy_perturbation,
                    projection.reservoir_physical_energy_perturbation,
                )
            )
            projection_momentum_pair_residual = math.fsum(
                (
                    projection.produced_intrinsic_momentum_potential,
                    projection.reservoir_intrinsic_momentum_potential,
                )
            )
            expected_clock_delta_q = (
                expected_q_prime
                + projection.hubble_log_derivative * expected_q
            ) * projection.scalar_clock_shift
            clock_residual_p = (
                projection.produced_physical_energy_perturbation
                - expected_clock_delta_q
            )
            clock_residual_r = (
                projection.reservoir_physical_energy_perturbation
                + expected_clock_delta_q
            )
            lower_time_p = -(
                projection.produced_background_q
                * (1.0 + projection.lapse_potential)
                + projection.produced_physical_energy_perturbation
            )
            lower_time_r = -(
                projection.reservoir_background_q
                * (1.0 + projection.lapse_potential)
                + projection.reservoir_physical_energy_perturbation
            )
            lower_time_pair_residual = math.fsum(
                (lower_time_p, lower_time_r)
            )
            lower_spatial_p = projection.k_over_a_h * (
                projection.produced_intrinsic_momentum_potential
                + projection.produced_background_q
                * projection.normalized_total_velocity_potential
            )
            lower_spatial_r = projection.k_over_a_h * (
                projection.reservoir_intrinsic_momentum_potential
                + projection.reservoir_background_q
                * projection.normalized_total_velocity_potential
            )
            lower_spatial_pair_residual = math.fsum(
                (lower_spatial_p, lower_spatial_r)
            )

            required_energy_p = (
                expected_energy_source_p
                - 3.0
                * (
                    energy.produced_density_perturbation
                    + energy.produced_pressure_perturbation
                )
                + 3.0
                * enthalpy_p
                * energy.metric_curvature_log_derivative
                - enthalpy_p * energy.produced_theta_over_a_h
            )
            required_energy_r = (
                expected_energy_source_r
                - 3.0
                * (
                    energy.reservoir_density_perturbation
                    + energy.reservoir_pressure_perturbation
                )
                + 3.0
                * enthalpy_r
                * energy.metric_curvature_log_derivative
                - enthalpy_r * energy.reservoir_theta_over_a_h
            )
            energy_residual_p = (
                energy.provided_produced_density_perturbation_derivative
                - required_energy_p
            )
            energy_residual_r = (
                energy.provided_reservoir_density_perturbation_derivative
                - required_energy_r
            )
            energy_source_pair_residual = math.fsum(
                (expected_energy_source_p, expected_energy_source_r)
            )

            required_momentum_p = (
                expected_momentum_source_p
                - (3.0 - projection.hubble_log_derivative)
                * momentum.produced_momentum_density
                - enthalpy_p * projection.lapse_potential
                - momentum.produced_pressure_perturbation
                + (2.0 / 3.0)
                * kappa_squared
                * momentum.produced_normalized_anisotropic_stress
            )
            required_momentum_r = (
                expected_momentum_source_r
                - (3.0 - projection.hubble_log_derivative)
                * momentum.reservoir_momentum_density
                - enthalpy_r * projection.lapse_potential
                - momentum.reservoir_pressure_perturbation
                + (2.0 / 3.0)
                * kappa_squared
                * momentum.reservoir_normalized_anisotropic_stress
            )
            momentum_residual_p = (
                momentum.provided_produced_momentum_density_derivative
                - required_momentum_p
            )
            momentum_residual_r = (
                momentum.provided_reservoir_momentum_density_derivative
                - required_momentum_r
            )
            momentum_source_pair_residual = math.fsum(
                (expected_momentum_source_p, expected_momentum_source_r)
            )
            total_enthalpy = enthalpy_p + enthalpy_r
            total_frame_residual = math.fsum(
                (
                    momentum.produced_momentum_density,
                    momentum.reservoir_momentum_density,
                )
            ) - (
                total_enthalpy
                * projection.normalized_total_velocity_potential
            )
        except OverflowError as error:
            raise ValueError("cross-gate recomputation left the finite domain") from error
        recomputed_values = (
            projection_background_pair_residual,
            projection_energy_pair_residual,
            projection_momentum_pair_residual,
            expected_clock_delta_q,
            clock_residual_p,
            clock_residual_r,
            lower_time_pair_residual,
            lower_spatial_p,
            lower_spatial_r,
            lower_spatial_pair_residual,
            required_energy_p,
            required_energy_r,
            energy_residual_p,
            energy_residual_r,
            energy_source_pair_residual,
            required_momentum_p,
            required_momentum_r,
            momentum_residual_p,
            momentum_residual_r,
            momentum_source_pair_residual,
            total_enthalpy,
            total_frame_residual,
        )
        if any(not math.isfinite(value) for value in recomputed_values):
            raise ValueError("cross-gate recomputation left the finite domain")

        projection_background_pair_holds = _zero_with_terms(
            projection_background_pair_residual,
            projection.produced_background_q,
            projection.reservoir_background_q,
        )
        projection_energy_pair_holds = _zero_with_terms(
            projection_energy_pair_residual,
            projection.produced_physical_energy_perturbation,
            projection.reservoir_physical_energy_perturbation,
        )
        projection_momentum_pair_holds = _zero_with_terms(
            projection_momentum_pair_residual,
            projection.produced_intrinsic_momentum_potential,
            projection.reservoir_intrinsic_momentum_potential,
        )
        projection_time_pair_holds = _zero_with_terms(
            lower_time_pair_residual,
            lower_time_p,
            lower_time_r,
        )
        projection_spatial_pair_holds = _zero_with_terms(
            lower_spatial_pair_residual,
            lower_spatial_p,
            lower_spatial_r,
        )
        projection_pairs = (
            projection_background_pair_holds
            and projection_energy_pair_holds
            and projection_momentum_pair_holds
            and projection_time_pair_holds
            and projection_spatial_pair_holds
        )
        common_clock_holds = (
            _zero_with_terms(
                clock_residual_p,
                projection.produced_physical_energy_perturbation,
                expected_clock_delta_q,
            )
            and _zero_with_terms(
                clock_residual_r,
                projection.reservoir_physical_energy_perturbation,
                -expected_clock_delta_q,
            )
        )
        energy_holds = (
            _zero_with_terms(
                energy_residual_p,
                energy.provided_produced_density_perturbation_derivative,
                required_energy_p,
            )
            and _zero_with_terms(
                energy_residual_r,
                energy.provided_reservoir_density_perturbation_derivative,
                required_energy_r,
            )
            and _zero_with_terms(
                energy_source_pair_residual,
                expected_energy_source_p,
                expected_energy_source_r,
            )
        )
        momentum_holds = (
            _zero_with_terms(
                momentum_residual_p,
                momentum.provided_produced_momentum_density_derivative,
                required_momentum_p,
            )
            and _zero_with_terms(
                momentum_residual_r,
                momentum.provided_reservoir_momentum_density_derivative,
                required_momentum_r,
            )
            and _zero_with_terms(
                momentum_source_pair_residual,
                expected_momentum_source_p,
                expected_momentum_source_r,
            )
        )
        total_frame_holds = _zero_with_terms(
            total_frame_residual,
            momentum.produced_momentum_density,
            momentum.reservoir_momentum_density,
            total_enthalpy * projection.normalized_total_velocity_potential,
        )
        nondegenerate_frame_holds = (
            total_frame_holds
            and total_enthalpy != 0.0
            and projection.k_over_a_h != 0.0
        )
        algebraic_holds = (
            consistent and projection_pairs and energy_holds and momentum_holds
        )
        clock_holds = (
            algebraic_holds and common_clock_holds
        )
        full_holds = (
            clock_holds and nondegenerate_frame_holds
        )
        failures: list[str] = []
        if not consistent:
            failures.append("CROSS_RECEIPT_MISMATCH")
        if not projection_pairs:
            failures.append("PROJECTION_COMPONENT_PAIR_CLOSURE_FAILED")
        if not energy_holds:
            failures.append("ENERGY_EQUATION_OR_EXCHANGE_FAILED")
        if not momentum_holds:
            failures.append("MOMENTUM_EQUATION_OR_EXCHANGE_FAILED")
        if not common_clock_holds:
            failures.append("COMMON_PHYSICAL_CLOCK_FAILED")
        if not nondegenerate_frame_holds:
            failures.append("NONDEGENERATE_TOTAL_ENERGY_FRAME_FAILED")

        return LinearSystemNodeGateReceipt(
            n=projection.n,
            cross_receipt_residuals=tuple(residuals),
            cross_receipt_consistency_holds=consistent,
            projection_background_pair_holds=(
                projection_background_pair_holds
            ),
            projection_physical_energy_pair_holds=(
                projection_energy_pair_holds
            ),
            projection_intrinsic_momentum_pair_holds=(
                projection_momentum_pair_holds
            ),
            projection_lower_time_pair_holds=(
                projection_time_pair_holds
            ),
            projection_lower_spatial_pair_holds=(
                projection_spatial_pair_holds
            ),
            projection_all_component_pairs_hold=projection_pairs,
            projection_common_physical_clock_holds=(
                common_clock_holds
            ),
            energy_equations_and_exchange_hold=energy_holds,
            momentum_equations_and_exchange_hold=momentum_holds,
            nondegenerate_total_energy_frame_holds=(
                nondegenerate_frame_holds
            ),
            algebraic_energy_momentum_node_holds=algebraic_holds,
            common_clock_energy_momentum_node_holds=clock_holds,
            full_declared_nondegenerate_node_holds=full_holds,
            failure_reasons=tuple(failures),
        )
