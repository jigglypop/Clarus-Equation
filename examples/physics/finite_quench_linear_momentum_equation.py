"""One-node Newtonian-gauge linear momentum equation for the finite quench.

Equation (21) of Valiviita, Majerotto and Maartens, arXiv:0804.0232,
is kept regular at k=0 and at zero enthalpy by evolving the normalized
enthalpy-weighted momentum

    U_A = (rho_A + P_A) V_A,     V_A = aH v_A.

For n=ln(a), h=D_n ln(H), kappa=k/(aH), constant density unit, and
pihat_A=H^2 pi_A/rho_unit, the normalized one-node equation is

    D_n U_A + (3-h)U_A + (rho_A+P_A)phi + Delta P_A
      - (2/3) kappa^2 pihat_A = q_A V + fhat_A.

Here V=aH(v+B) is the common total velocity potential and
fhat_A=a f_A/rho_unit.  The module computes or audits the derivative required
by this equation.  It neither integrates the system nor derives pressure,
anisotropic stress, metric, velocity, or transfer closures.
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
class _MomentumState:
    rho_p: float
    rho_r: float
    pressure_p: float
    pressure_r: float
    enthalpy_p: float
    enthalpy_r: float
    momentum_p: float
    momentum_r: float
    delta_pressure_p: float
    delta_pressure_r: float
    anisotropic_stress_p: float
    anisotropic_stress_r: float
    source_p: float
    source_r: float


@dataclass(frozen=True)
class LinearMomentumEquationReceipt:
    """One-node residual audit for two normalized momentum equations."""

    n: float
    k_over_a_h: float
    hubble_log_derivative: float
    lapse_potential: float
    normalized_total_velocity_potential: float
    produced_background_density: float
    reservoir_background_density: float
    produced_background_pressure: float
    reservoir_background_pressure: float
    produced_background_enthalpy: float
    reservoir_background_enthalpy: float
    total_background_enthalpy: float
    produced_momentum_density: float
    reservoir_momentum_density: float
    total_energy_frame_momentum_residual: float
    produced_pressure_perturbation: float
    reservoir_pressure_perturbation: float
    produced_normalized_anisotropic_stress: float
    reservoir_normalized_anisotropic_stress: float
    produced_momentum_transfer_source: float
    reservoir_momentum_transfer_source: float
    total_momentum_transfer_source_residual: float
    required_produced_momentum_density_derivative: float
    required_reservoir_momentum_density_derivative: float
    provided_produced_momentum_density_derivative: float
    provided_reservoir_momentum_density_derivative: float
    produced_momentum_equation_residual: float
    reservoir_momentum_equation_residual: float
    summed_momentum_equation_residual: float
    required_total_momentum_density_derivative: float
    provided_total_momentum_density_derivative: float
    total_velocity_identifiable_from_total_enthalpy: bool
    homogeneous_fourier_mode_degenerate: bool
    total_energy_frame_relation_holds: bool
    transfer_source_pair_cancels: bool
    produced_momentum_equation_holds: bool
    reservoir_momentum_equation_holds: bool
    both_momentum_equations_hold: bool
    momentum_equations_and_exchange_hold: bool
    total_energy_frame_momentum_branch_holds: bool
    dimensionless_roles: tuple[tuple[str, str], ...]
    source: str = "Valiviita_Majerotto_Maartens_2008_Eq_21"
    role: str = (
        "CONDITIONAL_ONE_NODE_LINEAR_MOMENTUM_BALANCE_"
        "NOT_INTEGRATED_EINSTEIN_BOLTZMANN_SOLUTION"
    )


class FiniteQuenchLinearMomentumEquation:
    """Compute and audit Eq. (21) using the nonsingular momentum variable U."""

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

    def _background_enthalpies(
        self,
        projection: LowerQmuProjectionReceipt,
    ) -> tuple[float, float]:
        rho_p = self.bridge.production_density(projection.n)
        rho_r = self.bridge.reservoir_density(projection.n)
        return rho_p, (1.0 + self.bridge.config.w_reservoir) * rho_r

    def reservoir_momentum_for_total_energy_frame(
        self,
        *,
        transfer_projection: object,
        produced_momentum_density: object,
    ) -> float:
        """Return U_R required by U_p+U_R=(enthalpy total)V without division."""

        projection = self._validate_projection(transfer_projection)
        momentum_p = _finite_real(
            produced_momentum_density,
            "produced_momentum_density",
        )
        enthalpy_p, enthalpy_r = self._background_enthalpies(projection)
        result = (
            (enthalpy_p + enthalpy_r)
            * projection.normalized_total_velocity_potential
            - momentum_p
        )
        if not math.isfinite(result):
            raise ValueError("total-energy-frame momentum left the finite domain")
        return result

    def _state(
        self,
        *,
        projection: LowerQmuProjectionReceipt,
        produced_momentum_density: object,
        reservoir_momentum_density: object,
        produced_pressure_perturbation: object,
        reservoir_pressure_perturbation: object,
        produced_normalized_anisotropic_stress: object,
        reservoir_normalized_anisotropic_stress: object,
    ) -> _MomentumState:
        rho_p = self.bridge.production_density(projection.n)
        rho_r = self.bridge.reservoir_density(projection.n)
        pressure_p = 0.0
        pressure_r = self.bridge.config.w_reservoir * rho_r
        enthalpy_p = rho_p + pressure_p
        enthalpy_r = rho_r + pressure_r
        state = _MomentumState(
            rho_p=rho_p,
            rho_r=rho_r,
            pressure_p=pressure_p,
            pressure_r=pressure_r,
            enthalpy_p=enthalpy_p,
            enthalpy_r=enthalpy_r,
            momentum_p=_finite_real(
                produced_momentum_density,
                "produced_momentum_density",
            ),
            momentum_r=_finite_real(
                reservoir_momentum_density,
                "reservoir_momentum_density",
            ),
            delta_pressure_p=_finite_real(
                produced_pressure_perturbation,
                "produced_pressure_perturbation",
            ),
            delta_pressure_r=_finite_real(
                reservoir_pressure_perturbation,
                "reservoir_pressure_perturbation",
            ),
            anisotropic_stress_p=_finite_real(
                produced_normalized_anisotropic_stress,
                "produced_normalized_anisotropic_stress",
            ),
            anisotropic_stress_r=_finite_real(
                reservoir_normalized_anisotropic_stress,
                "reservoir_normalized_anisotropic_stress",
            ),
            source_p=(
                projection.produced_background_q
                * projection.normalized_total_velocity_potential
                + projection.produced_intrinsic_momentum_potential
            ),
            source_r=(
                projection.reservoir_background_q
                * projection.normalized_total_velocity_potential
                + projection.reservoir_intrinsic_momentum_potential
            ),
        )
        if any(not math.isfinite(value) for value in state.__dict__.values()):
            raise ValueError("linear momentum state left the finite domain")
        return state

    @staticmethod
    def _required_derivatives(
        state: _MomentumState,
        *,
        kappa: float,
        hubble_log_prime: float,
        phi: float,
    ) -> tuple[float, float]:
        try:
            kappa_squared = kappa**2
        except OverflowError as error:
            raise ValueError("kappa squared left the finite domain") from error
        if not math.isfinite(kappa_squared):
            raise ValueError("kappa squared left the finite domain")
        required_p = (
            state.source_p
            - (3.0 - hubble_log_prime) * state.momentum_p
            - state.enthalpy_p * phi
            - state.delta_pressure_p
            + (2.0 / 3.0) * kappa_squared * state.anisotropic_stress_p
        )
        required_r = (
            state.source_r
            - (3.0 - hubble_log_prime) * state.momentum_r
            - state.enthalpy_r * phi
            - state.delta_pressure_r
            + (2.0 / 3.0) * kappa_squared * state.anisotropic_stress_r
        )
        if not math.isfinite(required_p) or not math.isfinite(required_r):
            raise ValueError("required momentum derivative left the finite domain")
        return required_p, required_r

    def construct_required_derivatives(
        self,
        *,
        transfer_projection: object,
        produced_momentum_density: object,
        reservoir_momentum_density: object,
        produced_pressure_perturbation: object,
        reservoir_pressure_perturbation: object,
        produced_normalized_anisotropic_stress: object,
        reservoir_normalized_anisotropic_stress: object,
    ) -> LinearMomentumEquationReceipt:
        """Construct the two momentum derivatives required at one node."""

        projection = self._validate_projection(transfer_projection)
        state = self._state(
            projection=projection,
            produced_momentum_density=produced_momentum_density,
            reservoir_momentum_density=reservoir_momentum_density,
            produced_pressure_perturbation=produced_pressure_perturbation,
            reservoir_pressure_perturbation=reservoir_pressure_perturbation,
            produced_normalized_anisotropic_stress=(
                produced_normalized_anisotropic_stress
            ),
            reservoir_normalized_anisotropic_stress=(
                reservoir_normalized_anisotropic_stress
            ),
        )
        required_p, required_r = self._required_derivatives(
            state,
            kappa=projection.k_over_a_h,
            hubble_log_prime=projection.hubble_log_derivative,
            phi=projection.lapse_potential,
        )
        return self.audit(
            transfer_projection=projection,
            produced_momentum_density=state.momentum_p,
            reservoir_momentum_density=state.momentum_r,
            produced_pressure_perturbation=state.delta_pressure_p,
            reservoir_pressure_perturbation=state.delta_pressure_r,
            produced_normalized_anisotropic_stress=(
                state.anisotropic_stress_p
            ),
            reservoir_normalized_anisotropic_stress=(
                state.anisotropic_stress_r
            ),
            produced_momentum_density_derivative=required_p,
            reservoir_momentum_density_derivative=required_r,
        )

    def audit(
        self,
        *,
        transfer_projection: object,
        produced_momentum_density: object,
        reservoir_momentum_density: object,
        produced_pressure_perturbation: object,
        reservoir_pressure_perturbation: object,
        produced_normalized_anisotropic_stress: object,
        reservoir_normalized_anisotropic_stress: object,
        produced_momentum_density_derivative: object,
        reservoir_momentum_density_derivative: object,
    ) -> LinearMomentumEquationReceipt:
        """Audit supplied U derivatives against the normalized Eq. (21)."""

        projection = self._validate_projection(transfer_projection)
        state = self._state(
            projection=projection,
            produced_momentum_density=produced_momentum_density,
            reservoir_momentum_density=reservoir_momentum_density,
            produced_pressure_perturbation=produced_pressure_perturbation,
            reservoir_pressure_perturbation=reservoir_pressure_perturbation,
            produced_normalized_anisotropic_stress=(
                produced_normalized_anisotropic_stress
            ),
            reservoir_normalized_anisotropic_stress=(
                reservoir_normalized_anisotropic_stress
            ),
        )
        required_p, required_r = self._required_derivatives(
            state,
            kappa=projection.k_over_a_h,
            hubble_log_prime=projection.hubble_log_derivative,
            phi=projection.lapse_potential,
        )
        derivative_p = _finite_real(
            produced_momentum_density_derivative,
            "produced_momentum_density_derivative",
        )
        derivative_r = _finite_real(
            reservoir_momentum_density_derivative,
            "reservoir_momentum_density_derivative",
        )
        residual_p = derivative_p - required_p
        residual_r = derivative_r - required_r
        residual_sum = math.fsum((residual_p, residual_r))
        source_pair = math.fsum((state.source_p, state.source_r))
        total_enthalpy = state.enthalpy_p + state.enthalpy_r
        frame_residual = math.fsum((state.momentum_p, state.momentum_r)) - (
            total_enthalpy * projection.normalized_total_velocity_potential
        )
        required_total = math.fsum((required_p, required_r))
        provided_total = math.fsum((derivative_p, derivative_r))
        finite_outputs = (
            residual_p,
            residual_r,
            residual_sum,
            source_pair,
            total_enthalpy,
            frame_residual,
            required_total,
            provided_total,
        )
        if any(not math.isfinite(value) for value in finite_outputs):
            raise ValueError("linear momentum audit left the finite domain")

        produced_holds = _within_roundoff(residual_p, derivative_p, required_p)
        reservoir_holds = _within_roundoff(residual_r, derivative_r, required_r)
        source_pair_holds = _within_roundoff(
            source_pair,
            state.source_p,
            state.source_r,
        )
        frame_holds = _within_roundoff(
            frame_residual,
            state.momentum_p,
            state.momentum_r,
            total_enthalpy * projection.normalized_total_velocity_potential,
        )
        both_hold = produced_holds and reservoir_holds

        return LinearMomentumEquationReceipt(
            n=projection.n,
            k_over_a_h=projection.k_over_a_h,
            hubble_log_derivative=projection.hubble_log_derivative,
            lapse_potential=projection.lapse_potential,
            normalized_total_velocity_potential=(
                projection.normalized_total_velocity_potential
            ),
            produced_background_density=state.rho_p,
            reservoir_background_density=state.rho_r,
            produced_background_pressure=state.pressure_p,
            reservoir_background_pressure=state.pressure_r,
            produced_background_enthalpy=state.enthalpy_p,
            reservoir_background_enthalpy=state.enthalpy_r,
            total_background_enthalpy=total_enthalpy,
            produced_momentum_density=state.momentum_p,
            reservoir_momentum_density=state.momentum_r,
            total_energy_frame_momentum_residual=frame_residual,
            produced_pressure_perturbation=state.delta_pressure_p,
            reservoir_pressure_perturbation=state.delta_pressure_r,
            produced_normalized_anisotropic_stress=(
                state.anisotropic_stress_p
            ),
            reservoir_normalized_anisotropic_stress=(
                state.anisotropic_stress_r
            ),
            produced_momentum_transfer_source=state.source_p,
            reservoir_momentum_transfer_source=state.source_r,
            total_momentum_transfer_source_residual=source_pair,
            required_produced_momentum_density_derivative=required_p,
            required_reservoir_momentum_density_derivative=required_r,
            provided_produced_momentum_density_derivative=derivative_p,
            provided_reservoir_momentum_density_derivative=derivative_r,
            produced_momentum_equation_residual=residual_p,
            reservoir_momentum_equation_residual=residual_r,
            summed_momentum_equation_residual=residual_sum,
            required_total_momentum_density_derivative=required_total,
            provided_total_momentum_density_derivative=provided_total,
            total_velocity_identifiable_from_total_enthalpy=(
                total_enthalpy != 0.0
            ),
            homogeneous_fourier_mode_degenerate=(
                projection.k_over_a_h == 0.0
            ),
            total_energy_frame_relation_holds=frame_holds,
            transfer_source_pair_cancels=source_pair_holds,
            produced_momentum_equation_holds=produced_holds,
            reservoir_momentum_equation_holds=reservoir_holds,
            both_momentum_equations_hold=both_hold,
            momentum_equations_and_exchange_hold=(
                both_hold and source_pair_holds
            ),
            total_energy_frame_momentum_branch_holds=(
                both_hold
                and source_pair_holds
                and frame_holds
                and total_enthalpy != 0.0
                and projection.k_over_a_h != 0.0
            ),
            dimensionless_roles=(
                ("U_A", "(rho_A+P_A) aH v_A divided by rho_unit"),
                ("V", "dimensionless common aH(v+B), with B=0 here"),
                ("h", "dimensionless d ln H / d ln a"),
                ("kappa", "dimensionless k/(aH)"),
                ("Delta_P_A", "delta P_A divided by rho_unit"),
                ("pihat_A", "H^2 pi_A divided by rho_unit"),
                ("q_A", "Q_A divided by H rho_unit"),
                ("fhat_A", "a f_A divided by rho_unit"),
            ),
        )
