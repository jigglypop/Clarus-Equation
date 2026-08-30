"""Construct and audit one fully cross-linked scalar GR fluid node.

This narrow common-clock branch links the flat-GR background, a declared
e-fold clock shift, strict barotropic pressure, kappa>0 Einstein constraints,
the lower Q_mu projection, and the VMM energy and momentum equations.

The result is an existence and consistency receipt for one Fourier node.  It
does not integrate time, prove constraint propagation, derive Q_mu from
quench microphysics, or predict observables.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from examples.physics.finite_quench_barotropic_closure import (
    FiniteQuenchStrictBarotropicClosure,
    StrictBarotropicClosureReceipt,
)
from examples.physics.finite_quench_einstein_constraint import (
    FiniteQuenchScalarEinsteinConstraint,
    ScalarEinsteinConstraintReceipt,
)
from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
    TwoFluidFlatGRBackgroundReceipt,
)
from examples.physics.finite_quench_linear_energy_equation import (
    FiniteQuenchLinearEnergyEquation,
    LinearEnergyEquationReceipt,
)
from examples.physics.finite_quench_linear_momentum_equation import (
    FiniteQuenchLinearMomentumEquation,
    LinearMomentumEquationReceipt,
)
from examples.physics.finite_quench_linear_system_gate import (
    FiniteQuenchLinearSystemNodeGate,
    LinearSystemNodeGateReceipt,
)
from examples.physics.finite_quench_qmu_projection_ledger import (
    FiniteQuenchLowerQmuProjectionLedger,
    LowerQmuProjectionReceipt,
)
from examples.physics.finite_quench_scalar_clock_ledger import (
    FiniteQuenchScalarClockLedger,
    ScalarClockLedgerReceipt,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
)


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 64.0 * math.ulp(scale)


def _finite_sum(name: str, *values: float) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} left the finite domain") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} left the finite domain")
    return result


def _compare(left: float, right: float) -> tuple[float, bool]:
    residual = _finite_sum("GR-node cross comparison", left, -right)
    return residual, _within_roundoff(residual, left, right)


@dataclass(frozen=True)
class GRLinearNodeReceipt:
    """Cross-linked raw receipts for one declared scalar Fourier node."""

    background: TwoFluidFlatGRBackgroundReceipt
    scalar_clock: ScalarClockLedgerReceipt
    closure: StrictBarotropicClosureReceipt
    einstein_constraint: ScalarEinsteinConstraintReceipt
    transfer_projection: LowerQmuProjectionReceipt
    energy_equation: LinearEnergyEquationReceipt
    momentum_equation: LinearMomentumEquationReceipt
    linear_system_gate: LinearSystemNodeGateReceipt
    cross_residuals: tuple[tuple[str, float], ...]
    background_holds: bool
    common_scalar_clock_holds: bool
    strict_barotropic_closure_holds: bool
    scalar_einstein_constraints_hold: bool
    lower_qmu_projection_holds: bool
    energy_equations_hold: bool
    momentum_equations_hold: bool
    prior_linear_system_gate_holds: bool
    all_cross_receipt_state_identifications_hold: bool
    full_declared_gr_linear_node_holds: bool
    failure_reasons: tuple[str, ...]
    free_declared_inputs: tuple[str, ...] = (
        "scalar_clock_shift",
        "produced_momentum_density",
        "reservoir_momentum_density",
        "produced_intrinsic_momentum_potential",
    )
    common_clock_tangent_preservation_proven: bool = False
    finite_step_constraint_propagation_proven: bool = False
    role: str = (
        "CONDITIONAL_COMMON_CLOCK_K_POSITIVE_ONE_NODE_GR_FLUID_CLOSURE_"
        "NOT_TIME_INTEGRATED_MICROPHYSICAL_OR_OBSERVABLE_SOLUTION"
    )


class FiniteQuenchGRLinearNode:
    """Build or audit the common-clock one-node GR-fluid branch."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    @staticmethod
    def _canonical_theta(
        *,
        enthalpy: float,
        momentum: float,
        kappa_squared: float,
        species: str,
    ) -> float:
        if enthalpy == 0.0:
            if momentum != 0.0:
                raise ValueError(
                    f"{species} momentum must vanish at zero enthalpy"
                )
            return 0.0
        theta = -kappa_squared * momentum / enthalpy
        if not math.isfinite(theta):
            raise ValueError(f"{species} velocity divergence left the finite domain")
        return theta

    def construct(
        self,
        *,
        n: object,
        k_over_a_h: object,
        scalar_clock_shift: object,
        produced_momentum_density: object,
        reservoir_momentum_density: object,
        produced_intrinsic_momentum_potential: object,
    ) -> GRLinearNodeReceipt:
        """Construct every one-node receipt from four declared inputs."""

        scalar_clock = FiniteQuenchScalarClockLedger(self.bridge).construct(
            n=n,
            scalar_clock_shift=scalar_clock_shift,
        )
        background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).construct(scalar_clock.n)
        closure = FiniteQuenchStrictBarotropicClosure(self.bridge).construct(
            n=scalar_clock.n,
            produced_density_perturbation=(
                scalar_clock.produced_density_perturbation
            ),
            reservoir_density_perturbation=(
                scalar_clock.reservoir_density_perturbation
            ),
        )
        einstein = FiniteQuenchScalarEinsteinConstraint(self.bridge).construct(
            background=background,
            closure=closure,
            k_over_a_h=k_over_a_h,
            produced_momentum_density=produced_momentum_density,
            reservoir_momentum_density=reservoir_momentum_density,
        )
        if background.total_enthalpy <= 0.0:
            raise ValueError(
                "common total velocity requires positive total enthalpy"
            )
        total_velocity = (
            einstein.total_momentum_density / background.total_enthalpy
        )
        if not math.isfinite(total_velocity):
            raise ValueError("total velocity potential left the finite domain")
        theta_p = self._canonical_theta(
            enthalpy=(
                background.produced_density + background.produced_pressure
            ),
            momentum=einstein.produced_momentum_density,
            kappa_squared=einstein.k_over_a_h_squared,
            species="produced",
        )
        theta_r = self._canonical_theta(
            enthalpy=(
                background.reservoir_density + background.reservoir_pressure
            ),
            momentum=einstein.reservoir_momentum_density,
            kappa_squared=einstein.k_over_a_h_squared,
            species="reservoir",
        )
        projection = FiniteQuenchLowerQmuProjectionLedger(
            self.bridge
        ).construct_common_clock(
            n=scalar_clock.n,
            k_over_a_h=einstein.k_over_a_h,
            scalar_clock_shift=scalar_clock.scalar_clock_shift,
            hubble_log_derivative=background.hubble_log_derivative,
            lapse_potential=einstein.lapse_potential,
            normalized_total_velocity_potential=total_velocity,
            produced_intrinsic_momentum_potential=(
                produced_intrinsic_momentum_potential
            ),
        )
        energy = FiniteQuenchLinearEnergyEquation(
            self.bridge
        ).construct_required_derivatives(
            transfer_projection=projection,
            produced_density_perturbation=(
                closure.produced_density_perturbation
            ),
            reservoir_density_perturbation=(
                closure.reservoir_density_perturbation
            ),
            produced_pressure_perturbation=(
                closure.produced_pressure_perturbation
            ),
            reservoir_pressure_perturbation=(
                closure.reservoir_pressure_perturbation
            ),
            metric_curvature_log_derivative=(
                einstein.curvature_potential_log_derivative
            ),
            produced_theta_over_a_h=theta_p,
            reservoir_theta_over_a_h=theta_r,
        )
        momentum = FiniteQuenchLinearMomentumEquation(
            self.bridge
        ).construct_required_derivatives(
            transfer_projection=projection,
            produced_momentum_density=einstein.produced_momentum_density,
            reservoir_momentum_density=einstein.reservoir_momentum_density,
            produced_pressure_perturbation=(
                closure.produced_pressure_perturbation
            ),
            reservoir_pressure_perturbation=(
                closure.reservoir_pressure_perturbation
            ),
            produced_normalized_anisotropic_stress=(
                closure.produced_normalized_anisotropic_stress
            ),
            reservoir_normalized_anisotropic_stress=(
                closure.reservoir_normalized_anisotropic_stress
            ),
        )
        return self.audit(
            background=background,
            scalar_clock=scalar_clock,
            closure=closure,
            einstein_constraint=einstein,
            transfer_projection=projection,
            energy_equation=energy,
            momentum_equation=momentum,
        )

    def audit(
        self,
        *,
        background: object,
        scalar_clock: object,
        closure: object,
        einstein_constraint: object,
        transfer_projection: object,
        energy_equation: object,
        momentum_equation: object,
    ) -> GRLinearNodeReceipt:
        """Recompute every receipt from raw fields and cross-link the states."""

        if not isinstance(background, TwoFluidFlatGRBackgroundReceipt):
            raise ValueError("background has the wrong receipt type")
        if not isinstance(scalar_clock, ScalarClockLedgerReceipt):
            raise ValueError("scalar_clock has the wrong receipt type")
        if not isinstance(closure, StrictBarotropicClosureReceipt):
            raise ValueError("closure has the wrong receipt type")
        if not isinstance(einstein_constraint, ScalarEinsteinConstraintReceipt):
            raise ValueError("einstein_constraint has the wrong receipt type")
        if not isinstance(transfer_projection, LowerQmuProjectionReceipt):
            raise ValueError("transfer_projection has the wrong receipt type")
        if not isinstance(energy_equation, LinearEnergyEquationReceipt):
            raise ValueError("energy_equation has the wrong receipt type")
        if not isinstance(momentum_equation, LinearMomentumEquationReceipt):
            raise ValueError("momentum_equation has the wrong receipt type")

        raw_background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).audit(
            background.n,
            normalized_hubble_squared=(
                background.hubble_squared_over_eight_pi_g_rho_unit_over_three
            ),
            hubble_log_derivative=background.hubble_log_derivative,
        )
        raw_clock = FiniteQuenchScalarClockLedger(self.bridge).audit(
            n=scalar_clock.n,
            scalar_clock_shift=scalar_clock.scalar_clock_shift,
            produced_density_perturbation=(
                scalar_clock.produced_density_perturbation
            ),
            reservoir_density_perturbation=(
                scalar_clock.reservoir_density_perturbation
            ),
            produced_energy_transfer_perturbation=(
                scalar_clock.produced_energy_transfer_perturbation
            ),
            reservoir_energy_transfer_perturbation=(
                scalar_clock.reservoir_energy_transfer_perturbation
            ),
        )
        raw_closure = FiniteQuenchStrictBarotropicClosure(self.bridge).audit(
            n=closure.n,
            produced_density_perturbation=(
                closure.produced_density_perturbation
            ),
            reservoir_density_perturbation=(
                closure.reservoir_density_perturbation
            ),
            produced_pressure_perturbation=(
                closure.produced_pressure_perturbation
            ),
            reservoir_pressure_perturbation=(
                closure.reservoir_pressure_perturbation
            ),
            produced_normalized_anisotropic_stress=(
                closure.produced_normalized_anisotropic_stress
            ),
            reservoir_normalized_anisotropic_stress=(
                closure.reservoir_normalized_anisotropic_stress
            ),
            produced_background_pressure_derivative=(
                closure.produced_background_pressure_derivative
            ),
            reservoir_background_pressure_derivative=(
                closure.reservoir_background_pressure_derivative
            ),
        )
        raw_einstein = FiniteQuenchScalarEinsteinConstraint(
            self.bridge
        ).audit(
            background=raw_background,
            closure=raw_closure,
            k_over_a_h=einstein_constraint.k_over_a_h,
            produced_momentum_density=(
                einstein_constraint.produced_momentum_density
            ),
            reservoir_momentum_density=(
                einstein_constraint.reservoir_momentum_density
            ),
            lapse_potential=einstein_constraint.lapse_potential,
            curvature_potential=einstein_constraint.curvature_potential,
            curvature_potential_log_derivative=(
                einstein_constraint.curvature_potential_log_derivative
            ),
        )
        raw_projection = FiniteQuenchLowerQmuProjectionLedger(
            self.bridge
        ).audit(
            n=transfer_projection.n,
            k_over_a_h=transfer_projection.k_over_a_h,
            scalar_clock_shift=transfer_projection.scalar_clock_shift,
            hubble_log_derivative=(
                transfer_projection.hubble_log_derivative
            ),
            lapse_potential=transfer_projection.lapse_potential,
            normalized_total_velocity_potential=(
                transfer_projection.normalized_total_velocity_potential
            ),
            produced_physical_energy_perturbation=(
                transfer_projection.produced_physical_energy_perturbation
            ),
            reservoir_physical_energy_perturbation=(
                transfer_projection.reservoir_physical_energy_perturbation
            ),
            produced_intrinsic_momentum_potential=(
                transfer_projection.produced_intrinsic_momentum_potential
            ),
            reservoir_intrinsic_momentum_potential=(
                transfer_projection.reservoir_intrinsic_momentum_potential
            ),
        )
        raw_energy = FiniteQuenchLinearEnergyEquation(self.bridge).audit(
            transfer_projection=raw_projection,
            produced_density_perturbation=(
                energy_equation.produced_density_perturbation
            ),
            reservoir_density_perturbation=(
                energy_equation.reservoir_density_perturbation
            ),
            produced_pressure_perturbation=(
                energy_equation.produced_pressure_perturbation
            ),
            reservoir_pressure_perturbation=(
                energy_equation.reservoir_pressure_perturbation
            ),
            metric_curvature_log_derivative=(
                energy_equation.metric_curvature_log_derivative
            ),
            produced_theta_over_a_h=energy_equation.produced_theta_over_a_h,
            reservoir_theta_over_a_h=(
                energy_equation.reservoir_theta_over_a_h
            ),
            produced_density_perturbation_derivative=(
                energy_equation.provided_produced_density_perturbation_derivative
            ),
            reservoir_density_perturbation_derivative=(
                energy_equation.provided_reservoir_density_perturbation_derivative
            ),
        )
        raw_momentum = FiniteQuenchLinearMomentumEquation(self.bridge).audit(
            transfer_projection=raw_projection,
            produced_momentum_density=(
                momentum_equation.produced_momentum_density
            ),
            reservoir_momentum_density=(
                momentum_equation.reservoir_momentum_density
            ),
            produced_pressure_perturbation=(
                momentum_equation.produced_pressure_perturbation
            ),
            reservoir_pressure_perturbation=(
                momentum_equation.reservoir_pressure_perturbation
            ),
            produced_normalized_anisotropic_stress=(
                momentum_equation.produced_normalized_anisotropic_stress
            ),
            reservoir_normalized_anisotropic_stress=(
                momentum_equation.reservoir_normalized_anisotropic_stress
            ),
            produced_momentum_density_derivative=(
                momentum_equation.provided_produced_momentum_density_derivative
            ),
            reservoir_momentum_density_derivative=(
                momentum_equation.provided_reservoir_momentum_density_derivative
            ),
        )
        raw_linear_gate = FiniteQuenchLinearSystemNodeGate(self.bridge).audit(
            transfer_projection=raw_projection,
            energy_equation=raw_energy,
            momentum_equation=raw_momentum,
        )

        enthalpy_p = _finite_sum(
            "produced enthalpy",
            raw_background.produced_density,
            raw_background.produced_pressure,
        )
        enthalpy_r = _finite_sum(
            "reservoir enthalpy",
            raw_background.reservoir_density,
            raw_background.reservoir_pressure,
        )
        if raw_background.total_enthalpy <= 0.0:
            raise ValueError("GR node requires positive total enthalpy")
        expected_velocity = (
            raw_einstein.total_momentum_density
            / raw_background.total_enthalpy
        )
        if not math.isfinite(expected_velocity):
            raise ValueError("expected total velocity left the finite domain")
        expected_theta_p = self._canonical_theta(
            enthalpy=enthalpy_p,
            momentum=raw_einstein.produced_momentum_density,
            kappa_squared=raw_einstein.k_over_a_h_squared,
            species="produced",
        )
        expected_theta_r = self._canonical_theta(
            enthalpy=enthalpy_r,
            momentum=raw_einstein.reservoir_momentum_density,
            kappa_squared=raw_einstein.k_over_a_h_squared,
            species="reservoir",
        )
        hubble_clock_term = (
            raw_background.hubble_log_derivative
            * self.bridge.source(raw_clock.n)
            * raw_clock.scalar_clock_shift
        )
        if not math.isfinite(hubble_clock_term):
            raise ValueError(
                "physical-source clock correction left the finite domain"
            )
        expected_physical_source_p = _finite_sum(
            "physical source clock conversion",
            raw_clock.produced_energy_transfer_perturbation,
            hubble_clock_term,
        )
        expected_physical_source_r = -expected_physical_source_p

        comparisons = (
            ("clock_n", raw_clock.n, raw_background.n),
            ("closure_n", raw_closure.n, raw_background.n),
            ("einstein_n", raw_einstein.n, raw_background.n),
            ("projection_n", raw_projection.n, raw_background.n),
            ("energy_n", raw_energy.n, raw_background.n),
            ("momentum_n", raw_momentum.n, raw_background.n),
            (
                "projection_kappa",
                raw_projection.k_over_a_h,
                raw_einstein.k_over_a_h,
            ),
            (
                "momentum_kappa",
                raw_momentum.k_over_a_h,
                raw_einstein.k_over_a_h,
            ),
            (
                "projection_hubble_log_derivative",
                raw_projection.hubble_log_derivative,
                raw_background.hubble_log_derivative,
            ),
            (
                "momentum_hubble_log_derivative",
                raw_momentum.hubble_log_derivative,
                raw_background.hubble_log_derivative,
            ),
            (
                "projection_phi",
                raw_projection.lapse_potential,
                raw_einstein.lapse_potential,
            ),
            (
                "energy_phi",
                raw_energy.lapse_potential,
                raw_einstein.lapse_potential,
            ),
            (
                "momentum_phi",
                raw_momentum.lapse_potential,
                raw_einstein.lapse_potential,
            ),
            (
                "energy_psi_prime",
                raw_energy.metric_curvature_log_derivative,
                raw_einstein.curvature_potential_log_derivative,
            ),
            (
                "clock_delta_rho_p_closure",
                raw_clock.produced_density_perturbation,
                raw_closure.produced_density_perturbation,
            ),
            (
                "clock_delta_rho_r_closure",
                raw_clock.reservoir_density_perturbation,
                raw_closure.reservoir_density_perturbation,
            ),
            (
                "clock_delta_rho_p_einstein",
                raw_clock.produced_density_perturbation,
                raw_einstein.produced_density_perturbation,
            ),
            (
                "clock_delta_rho_r_einstein",
                raw_clock.reservoir_density_perturbation,
                raw_einstein.reservoir_density_perturbation,
            ),
            (
                "clock_delta_rho_p_energy",
                raw_clock.produced_density_perturbation,
                raw_energy.produced_density_perturbation,
            ),
            (
                "clock_delta_rho_r_energy",
                raw_clock.reservoir_density_perturbation,
                raw_energy.reservoir_density_perturbation,
            ),
            (
                "closure_delta_pressure_p_energy",
                raw_closure.produced_pressure_perturbation,
                raw_energy.produced_pressure_perturbation,
            ),
            (
                "closure_delta_pressure_r_energy",
                raw_closure.reservoir_pressure_perturbation,
                raw_energy.reservoir_pressure_perturbation,
            ),
            (
                "closure_delta_pressure_p_momentum",
                raw_closure.produced_pressure_perturbation,
                raw_momentum.produced_pressure_perturbation,
            ),
            (
                "closure_delta_pressure_r_momentum",
                raw_closure.reservoir_pressure_perturbation,
                raw_momentum.reservoir_pressure_perturbation,
            ),
            (
                "closure_pi_p_momentum",
                raw_closure.produced_normalized_anisotropic_stress,
                raw_momentum.produced_normalized_anisotropic_stress,
            ),
            (
                "closure_pi_r_momentum",
                raw_closure.reservoir_normalized_anisotropic_stress,
                raw_momentum.reservoir_normalized_anisotropic_stress,
            ),
            (
                "einstein_u_p_momentum",
                raw_einstein.produced_momentum_density,
                raw_momentum.produced_momentum_density,
            ),
            (
                "einstein_u_r_momentum",
                raw_einstein.reservoir_momentum_density,
                raw_momentum.reservoir_momentum_density,
            ),
            (
                "total_velocity",
                raw_projection.normalized_total_velocity_potential,
                expected_velocity,
            ),
            (
                "produced_theta",
                raw_energy.produced_theta_over_a_h,
                expected_theta_p,
            ),
            (
                "reservoir_theta",
                raw_energy.reservoir_theta_over_a_h,
                expected_theta_r,
            ),
            (
                "projection_clock",
                raw_projection.scalar_clock_shift,
                raw_clock.scalar_clock_shift,
            ),
            (
                "physical_source_p",
                raw_projection.produced_physical_energy_perturbation,
                expected_physical_source_p,
            ),
            (
                "physical_source_r",
                raw_projection.reservoir_physical_energy_perturbation,
                expected_physical_source_r,
            ),
        )
        residuals: list[tuple[str, float]] = []
        cross_holds = True
        for name, left, right in comparisons:
            residual, holds = _compare(left, right)
            residuals.append((name, residual))
            cross_holds = cross_holds and holds

        background_holds = raw_background.all_background_constraints_hold
        clock_holds = raw_clock.all_declared_clock_constraints_hold
        closure_holds = raw_closure.all_strict_barotropic_constraints_hold
        einstein_holds = raw_einstein.all_declared_scalar_constraints_hold
        projection_holds = (
            raw_projection.all_declared_lower_component_constraints_hold
        )
        energy_holds = raw_energy.common_clock_energy_branch_holds
        momentum_holds = (
            raw_momentum.total_energy_frame_momentum_branch_holds
        )
        linear_gate_holds = (
            raw_linear_gate.full_declared_nondegenerate_node_holds
        )
        all_holds = (
            background_holds
            and clock_holds
            and closure_holds
            and einstein_holds
            and projection_holds
            and energy_holds
            and momentum_holds
            and linear_gate_holds
            and cross_holds
        )
        failures: list[str] = []
        if not background_holds:
            failures.append("BACKGROUND_CONTRACT_FAILED")
        if not clock_holds:
            failures.append("COMMON_SCALAR_CLOCK_FAILED")
        if not closure_holds:
            failures.append("BAROTROPIC_CLOSURE_FAILED")
        if not einstein_holds:
            failures.append("SCALAR_EINSTEIN_CONSTRAINT_FAILED")
        if not projection_holds:
            failures.append("LOWER_QMU_PROJECTION_FAILED")
        if not energy_holds:
            failures.append("ENERGY_EQUATION_FAILED")
        if not momentum_holds:
            failures.append("MOMENTUM_EQUATION_FAILED")
        if not linear_gate_holds:
            failures.append("PRIOR_LINEAR_SYSTEM_GATE_FAILED")
        if not cross_holds:
            failures.append("GR_NODE_CROSS_RECEIPT_MISMATCH")

        return GRLinearNodeReceipt(
            background=raw_background,
            scalar_clock=raw_clock,
            closure=raw_closure,
            einstein_constraint=raw_einstein,
            transfer_projection=raw_projection,
            energy_equation=raw_energy,
            momentum_equation=raw_momentum,
            linear_system_gate=raw_linear_gate,
            cross_residuals=tuple(residuals),
            background_holds=background_holds,
            common_scalar_clock_holds=clock_holds,
            strict_barotropic_closure_holds=closure_holds,
            scalar_einstein_constraints_hold=einstein_holds,
            lower_qmu_projection_holds=projection_holds,
            energy_equations_hold=energy_holds,
            momentum_equations_hold=momentum_holds,
            prior_linear_system_gate_holds=linear_gate_holds,
            all_cross_receipt_state_identifications_hold=cross_holds,
            full_declared_gr_linear_node_holds=all_holds,
            failure_reasons=tuple(failures),
        )
