"""One-node Newtonian-gauge linear energy equation for the finite quench.

Starting from Eq. (20) of Valiviita, Majerotto and Maartens,
arXiv:0804.0232, divide by the conformal Hubble rate and by a constant density
unit.  In Newtonian gauge (B=E=0), with n=ln(a) and
Theta_A=theta_A/(aH), the normalized equation is

    D_n Delta rho_A + 3(Delta rho_A + Delta P_A)
      - 3(rho_A + P_A) D_n psi + (rho_A + P_A) Theta_A
      = q_A phi + deltaQhat_A.

This module computes or audits the density-perturbation derivative required by
that equation at one node.  It does not integrate the equation and does not
provide the pressure, velocity, metric, or microphysical transfer closures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.finite_quench_qmu_projection_ledger import (
    LowerQmuProjectionReceipt,
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


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 64.0 * math.ulp(scale)


@dataclass(frozen=True)
class LinearEnergyEquationReceipt:
    """One-node residual audit for the two normalized energy equations."""

    n: float
    lapse_potential: float
    metric_curvature_log_derivative: float
    produced_background_density: float
    reservoir_background_density: float
    produced_background_pressure: float
    reservoir_background_pressure: float
    produced_background_enthalpy: float
    reservoir_background_enthalpy: float
    produced_density_perturbation: float
    reservoir_density_perturbation: float
    produced_pressure_perturbation: float
    reservoir_pressure_perturbation: float
    produced_theta_over_a_h: float
    reservoir_theta_over_a_h: float
    produced_energy_transfer_source: float
    reservoir_energy_transfer_source: float
    total_energy_transfer_source_residual: float
    required_produced_density_perturbation_derivative: float
    required_reservoir_density_perturbation_derivative: float
    provided_produced_density_perturbation_derivative: float
    provided_reservoir_density_perturbation_derivative: float
    produced_energy_equation_residual: float
    reservoir_energy_equation_residual: float
    summed_energy_equation_residual: float
    required_total_density_perturbation_derivative: float
    provided_total_density_perturbation_derivative: float
    transfer_projection_common_clock_holds: bool
    transfer_source_pair_cancels: bool
    produced_energy_equation_holds: bool
    reservoir_energy_equation_holds: bool
    both_energy_equations_hold: bool
    energy_equations_and_exchange_hold: bool
    common_clock_energy_branch_holds: bool
    dimensionless_roles: tuple[tuple[str, str], ...]
    source: str = "Valiviita_Majerotto_Maartens_2008_Eq_20"
    role: str = (
        "CONDITIONAL_ONE_NODE_LINEAR_ENERGY_BALANCE_"
        "NOT_INTEGRATED_EINSTEIN_BOLTZMANN_SOLUTION"
    )


class FiniteQuenchLinearEnergyEquation:
    """Compute and audit Eq. (20) after the declared e-fold normalization."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _validate_projection(
        self,
        projection: object,
    ) -> LowerQmuProjectionReceipt:
        if not isinstance(projection, LowerQmuProjectionReceipt):
            raise ValueError("transfer_projection must be a LowerQmuProjectionReceipt")
        expected_q = self.bridge.source(projection.n)
        q_scale = max(1.0, abs(expected_q), abs(projection.produced_background_q))
        if abs(projection.produced_background_q - expected_q) > (
            64.0 * math.ulp(q_scale)
        ):
            raise ValueError("transfer projection does not match this bridge")
        if abs(projection.reservoir_background_q + expected_q) > (
            64.0 * math.ulp(q_scale)
        ):
            raise ValueError("reservoir transfer projection does not match this bridge")
        return projection

    def construct_required_derivatives(
        self,
        *,
        transfer_projection: object,
        produced_density_perturbation: object,
        reservoir_density_perturbation: object,
        produced_pressure_perturbation: object,
        reservoir_pressure_perturbation: object,
        metric_curvature_log_derivative: object,
        produced_theta_over_a_h: object,
        reservoir_theta_over_a_h: object,
    ) -> LinearEnergyEquationReceipt:
        """Construct the two derivatives required at the supplied state node."""

        projection = self._validate_projection(transfer_projection)
        state = self._state_terms(
            projection=projection,
            produced_density_perturbation=produced_density_perturbation,
            reservoir_density_perturbation=reservoir_density_perturbation,
            produced_pressure_perturbation=produced_pressure_perturbation,
            reservoir_pressure_perturbation=reservoir_pressure_perturbation,
            metric_curvature_log_derivative=metric_curvature_log_derivative,
            produced_theta_over_a_h=produced_theta_over_a_h,
            reservoir_theta_over_a_h=reservoir_theta_over_a_h,
        )
        required_p, required_r = self._required_derivatives(*state)
        return self.audit(
            transfer_projection=projection,
            produced_density_perturbation=state[6],
            reservoir_density_perturbation=state[7],
            produced_pressure_perturbation=state[8],
            reservoir_pressure_perturbation=state[9],
            metric_curvature_log_derivative=state[10],
            produced_theta_over_a_h=state[11],
            reservoir_theta_over_a_h=state[12],
            produced_density_perturbation_derivative=required_p,
            reservoir_density_perturbation_derivative=required_r,
        )

    def _state_terms(
        self,
        *,
        projection: LowerQmuProjectionReceipt,
        produced_density_perturbation: object,
        reservoir_density_perturbation: object,
        produced_pressure_perturbation: object,
        reservoir_pressure_perturbation: object,
        metric_curvature_log_derivative: object,
        produced_theta_over_a_h: object,
        reservoir_theta_over_a_h: object,
    ) -> tuple[float, ...]:
        rho_p = self.bridge.production_density(projection.n)
        rho_r = self.bridge.reservoir_density(projection.n)
        pressure_p = 0.0
        pressure_r = self.bridge.config.w_reservoir * rho_r
        enthalpy_p = rho_p + pressure_p
        enthalpy_r = rho_r + pressure_r
        values = (
            rho_p,
            rho_r,
            pressure_p,
            pressure_r,
            enthalpy_p,
            enthalpy_r,
            _finite_real(
                produced_density_perturbation,
                "produced_density_perturbation",
            ),
            _finite_real(
                reservoir_density_perturbation,
                "reservoir_density_perturbation",
            ),
            _finite_real(
                produced_pressure_perturbation,
                "produced_pressure_perturbation",
            ),
            _finite_real(
                reservoir_pressure_perturbation,
                "reservoir_pressure_perturbation",
            ),
            _finite_real(
                metric_curvature_log_derivative,
                "metric_curvature_log_derivative",
            ),
            _finite_real(produced_theta_over_a_h, "produced_theta_over_a_h"),
            _finite_real(
                reservoir_theta_over_a_h,
                "reservoir_theta_over_a_h",
            ),
            projection.produced_background_q * projection.lapse_potential
            + projection.produced_physical_energy_perturbation,
            projection.reservoir_background_q * projection.lapse_potential
            + projection.reservoir_physical_energy_perturbation,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("linear energy state left the finite domain")
        return values

    @staticmethod
    def _required_derivatives(*state: float) -> tuple[float, float]:
        (
            _rho_p,
            _rho_r,
            _pressure_p,
            _pressure_r,
            enthalpy_p,
            enthalpy_r,
            delta_rho_p,
            delta_rho_r,
            delta_pressure_p,
            delta_pressure_r,
            psi_prime,
            theta_p,
            theta_r,
            source_p,
            source_r,
        ) = state
        required_p = (
            source_p
            - 3.0 * (delta_rho_p + delta_pressure_p)
            + 3.0 * enthalpy_p * psi_prime
            - enthalpy_p * theta_p
        )
        required_r = (
            source_r
            - 3.0 * (delta_rho_r + delta_pressure_r)
            + 3.0 * enthalpy_r * psi_prime
            - enthalpy_r * theta_r
        )
        if not math.isfinite(required_p) or not math.isfinite(required_r):
            raise ValueError("required energy derivative left the finite domain")
        return required_p, required_r

    def audit(
        self,
        *,
        transfer_projection: object,
        produced_density_perturbation: object,
        reservoir_density_perturbation: object,
        produced_pressure_perturbation: object,
        reservoir_pressure_perturbation: object,
        metric_curvature_log_derivative: object,
        produced_theta_over_a_h: object,
        reservoir_theta_over_a_h: object,
        produced_density_perturbation_derivative: object,
        reservoir_density_perturbation_derivative: object,
    ) -> LinearEnergyEquationReceipt:
        """Audit supplied derivatives against the normalized energy equation."""

        projection = self._validate_projection(transfer_projection)
        state = self._state_terms(
            projection=projection,
            produced_density_perturbation=produced_density_perturbation,
            reservoir_density_perturbation=reservoir_density_perturbation,
            produced_pressure_perturbation=produced_pressure_perturbation,
            reservoir_pressure_perturbation=reservoir_pressure_perturbation,
            metric_curvature_log_derivative=metric_curvature_log_derivative,
            produced_theta_over_a_h=produced_theta_over_a_h,
            reservoir_theta_over_a_h=reservoir_theta_over_a_h,
        )
        required_p, required_r = self._required_derivatives(*state)
        derivative_p = _finite_real(
            produced_density_perturbation_derivative,
            "produced_density_perturbation_derivative",
        )
        derivative_r = _finite_real(
            reservoir_density_perturbation_derivative,
            "reservoir_density_perturbation_derivative",
        )
        residual_p = derivative_p - required_p
        residual_r = derivative_r - required_r
        summed_residual = math.fsum((residual_p, residual_r))
        source_pair = math.fsum((state[13], state[14]))
        required_total = math.fsum((required_p, required_r))
        provided_total = math.fsum((derivative_p, derivative_r))
        finite_outputs = (
            residual_p,
            residual_r,
            summed_residual,
            source_pair,
            required_total,
            provided_total,
        )
        if any(not math.isfinite(value) for value in finite_outputs):
            raise ValueError("linear energy audit left the finite domain")
        produced_holds = _within_roundoff(residual_p, derivative_p, required_p)
        reservoir_holds = _within_roundoff(residual_r, derivative_r, required_r)
        pair_holds = _within_roundoff(source_pair, state[13], state[14])
        both_hold = produced_holds and reservoir_holds

        return LinearEnergyEquationReceipt(
            n=projection.n,
            lapse_potential=projection.lapse_potential,
            metric_curvature_log_derivative=state[10],
            produced_background_density=state[0],
            reservoir_background_density=state[1],
            produced_background_pressure=state[2],
            reservoir_background_pressure=state[3],
            produced_background_enthalpy=state[4],
            reservoir_background_enthalpy=state[5],
            produced_density_perturbation=state[6],
            reservoir_density_perturbation=state[7],
            produced_pressure_perturbation=state[8],
            reservoir_pressure_perturbation=state[9],
            produced_theta_over_a_h=state[11],
            reservoir_theta_over_a_h=state[12],
            produced_energy_transfer_source=state[13],
            reservoir_energy_transfer_source=state[14],
            total_energy_transfer_source_residual=source_pair,
            required_produced_density_perturbation_derivative=required_p,
            required_reservoir_density_perturbation_derivative=required_r,
            provided_produced_density_perturbation_derivative=derivative_p,
            provided_reservoir_density_perturbation_derivative=derivative_r,
            produced_energy_equation_residual=residual_p,
            reservoir_energy_equation_residual=residual_r,
            summed_energy_equation_residual=summed_residual,
            required_total_density_perturbation_derivative=required_total,
            provided_total_density_perturbation_derivative=provided_total,
            transfer_projection_common_clock_holds=(
                projection.common_clock_physical_source_holds
            ),
            transfer_source_pair_cancels=pair_holds,
            produced_energy_equation_holds=produced_holds,
            reservoir_energy_equation_holds=reservoir_holds,
            both_energy_equations_hold=both_hold,
            energy_equations_and_exchange_hold=(both_hold and pair_holds),
            common_clock_energy_branch_holds=(
                both_hold
                and pair_holds
                and projection.common_clock_physical_source_holds
            ),
            dimensionless_roles=(
                ("Delta_rho_A", "delta rho_A divided by rho_unit"),
                ("Delta_P_A", "delta P_A divided by rho_unit"),
                ("D_n", "derivative with respect to dimensionless ln(a)"),
                ("phi,psi", "dimensionless Newtonian metric potentials"),
                ("Theta_A", "dimensionless theta_A/(aH)"),
                ("q_A", "Q_A divided by H rho_unit"),
                ("deltaQhat_A", "delta Q_A divided by H rho_unit"),
            ),
        )
