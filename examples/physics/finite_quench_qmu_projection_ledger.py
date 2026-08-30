"""Lower-component transfer projection ledger for the finite quench.

The conventions follow Eqs. (15)--(19) of Valiviita, Majerotto and
Maartens, arXiv:0804.0232.  For the lower spacetime components,

    Q^A_0 = -a [Q_A (1 + phi) + delta Q_A],
    Q^A_i =  a partial_i [f_A + Q_A (v + B)].

With a constant density unit, q_A = Q_A/(H rho_unit),
kappa = k/(aH), V = aH(v+B), and fhat_A = a f_A/rho_unit, this module audits
the dimensionless coefficients

    Q0hat_A = -[q_A(1+phi) + deltaQhat_A],
    Qihat_A = kappa [fhat_A + q_A V],

where the omitted Fourier factor is ``i k_hat_i``.  A common e-fold clock T
acts on the physical rate Q_A = H rho_unit q_A as

    deltaQhat_A = [q_A' + (ln H)' q_A] T.

This is a conditional lower-component pair-closure ledger.  It does not
derive a microphysical four-vector, verify u_mu F^mu=0 beyond the declared
linear scalar projection, or solve Einstein--Boltzmann dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

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
class LowerQmuProjectionAxioms:
    """Conventions that must be fixed before using the projection formulas."""

    gauge: str = "newtonian_B_equals_zero"
    metric_signature: str = "minus_plus_plus_plus"
    density_normalization: str = "constant_rho_unit"
    source_identification: str = "q_equals_Q_over_H_rho_unit"
    time_shift_convention: str = "delta_X_equals_dX_dn_times_T"
    fourier_convention: str = "exp_plus_i_k_dot_x"

    def __post_init__(self) -> None:
        required = {
            "gauge": "newtonian_B_equals_zero",
            "metric_signature": "minus_plus_plus_plus",
            "density_normalization": "constant_rho_unit",
            "source_identification": "q_equals_Q_over_H_rho_unit",
            "time_shift_convention": "delta_X_equals_dX_dn_times_T",
            "fourier_convention": "exp_plus_i_k_dot_x",
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r}")


@dataclass(frozen=True)
class LowerQmuProjectionReceipt:
    """One-node audit of the declared lower-component transfer projection."""

    n: float
    k_over_a_h: float
    scalar_clock_shift: float
    hubble_log_derivative: float
    lapse_potential: float
    normalized_total_velocity_potential: float
    produced_background_q: float
    reservoir_background_q: float
    background_q_pair_residual: float
    produced_background_q_derivative: float
    expected_clock_physical_energy_perturbation: float
    produced_physical_energy_perturbation: float
    reservoir_physical_energy_perturbation: float
    produced_physical_energy_clock_residual: float
    reservoir_physical_energy_clock_residual: float
    physical_energy_perturbation_pair_residual: float
    q_prime_only_physical_clock_residual: float
    missing_hubble_normalization_term: float
    produced_intrinsic_momentum_potential: float
    reservoir_intrinsic_momentum_potential: float
    intrinsic_momentum_pair_residual: float
    produced_normalized_lower_time_component: float
    reservoir_normalized_lower_time_component: float
    normalized_lower_time_component_sum_residual: float
    produced_normalized_spatial_bracket: float
    reservoir_normalized_spatial_bracket: float
    produced_normalized_lower_spatial_fourier_scalar: float
    reservoir_normalized_lower_spatial_fourier_scalar: float
    normalized_lower_spatial_component_sum_residual: float
    common_clock_physical_source_holds: bool
    background_source_pair_cancels: bool
    physical_energy_perturbation_pair_cancels: bool
    intrinsic_momentum_pair_cancels: bool
    lower_time_component_pair_cancels: bool
    lower_spatial_component_pair_cancels: bool
    all_declared_lower_component_constraints_hold: bool
    dimensionless_roles: tuple[tuple[str, str], ...]
    source: str = "Valiviita_Majerotto_Maartens_2008_Eqs_15_to_19"
    role: str = (
        "CONDITIONAL_LOWER_COMPONENT_PAIR_CLOSURE_"
        "NOT_MICROPHYSICAL_QMU_OR_DYNAMICAL_SOLUTION"
    )


class FiniteQuenchLowerQmuProjectionLedger:
    """Construct and audit a declared Newtonian-gauge scalar projection."""

    def __init__(
        self,
        bridge: FiniteQuenchBridge,
        axioms: LowerQmuProjectionAxioms | None = None,
    ) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        if axioms is None:
            axioms = LowerQmuProjectionAxioms()
        if not isinstance(axioms, LowerQmuProjectionAxioms):
            raise ValueError("axioms must be LowerQmuProjectionAxioms")
        self.bridge = bridge
        self.axioms = axioms

    def _background(self, n: object) -> tuple[float, float, float]:
        n_value = _finite_real(n, "n")
        if not self.bridge.config.n_initial <= n_value <= 0.0:
            raise ValueError("n is outside the Qmu projection ledger domain")
        return (
            n_value,
            self.bridge.source(n_value),
            self.bridge.source_derivative(n_value),
        )

    def construct_common_clock(
        self,
        *,
        n: object,
        k_over_a_h: object,
        scalar_clock_shift: object,
        hubble_log_derivative: object,
        lapse_potential: object,
        normalized_total_velocity_potential: object,
        produced_intrinsic_momentum_potential: object,
    ) -> LowerQmuProjectionReceipt:
        """Construct paired lower components from the declared common clock."""

        n_value, q_p, q_p_prime = self._background(n)
        clock = _finite_real(scalar_clock_shift, "scalar_clock_shift")
        hubble_log_prime = _finite_real(
            hubble_log_derivative,
            "hubble_log_derivative",
        )
        f_p = _finite_real(
            produced_intrinsic_momentum_potential,
            "produced_intrinsic_momentum_potential",
        )
        delta_q_physical = (q_p_prime + hubble_log_prime * q_p) * clock
        if not math.isfinite(delta_q_physical):
            raise ValueError("common-clock physical source left the finite domain")
        return self.audit(
            n=n_value,
            k_over_a_h=k_over_a_h,
            scalar_clock_shift=clock,
            hubble_log_derivative=hubble_log_prime,
            lapse_potential=lapse_potential,
            normalized_total_velocity_potential=(
                normalized_total_velocity_potential
            ),
            produced_physical_energy_perturbation=delta_q_physical,
            reservoir_physical_energy_perturbation=-delta_q_physical,
            produced_intrinsic_momentum_potential=f_p,
            reservoir_intrinsic_momentum_potential=-f_p,
        )

    def audit(
        self,
        *,
        n: object,
        k_over_a_h: object,
        scalar_clock_shift: object,
        hubble_log_derivative: object,
        lapse_potential: object,
        normalized_total_velocity_potential: object,
        produced_physical_energy_perturbation: object,
        reservoir_physical_energy_perturbation: object,
        produced_intrinsic_momentum_potential: object,
        reservoir_intrinsic_momentum_potential: object,
    ) -> LowerQmuProjectionReceipt:
        """Audit arbitrary paired scalar data against the projection ledger."""

        n_value, q_p, q_p_prime = self._background(n)
        kappa = _finite_real(k_over_a_h, "k_over_a_h")
        if kappa < 0.0:
            raise ValueError("k_over_a_h must be >= 0")
        clock = _finite_real(scalar_clock_shift, "scalar_clock_shift")
        hubble_log_prime = _finite_real(
            hubble_log_derivative,
            "hubble_log_derivative",
        )
        phi = _finite_real(lapse_potential, "lapse_potential")
        velocity = _finite_real(
            normalized_total_velocity_potential,
            "normalized_total_velocity_potential",
        )
        delta_q_p = _finite_real(
            produced_physical_energy_perturbation,
            "produced_physical_energy_perturbation",
        )
        delta_q_r = _finite_real(
            reservoir_physical_energy_perturbation,
            "reservoir_physical_energy_perturbation",
        )
        f_p = _finite_real(
            produced_intrinsic_momentum_potential,
            "produced_intrinsic_momentum_potential",
        )
        f_r = _finite_real(
            reservoir_intrinsic_momentum_potential,
            "reservoir_intrinsic_momentum_potential",
        )
        q_r = -q_p
        expected_delta_q_p = (q_p_prime + hubble_log_prime * q_p) * clock
        expected_delta_q_r = -expected_delta_q_p
        q_prime_only = q_p_prime * clock
        missing_hubble_term = hubble_log_prime * q_p * clock
        try:
            background_pair = math.fsum((q_p, q_r))
            clock_residual_p = delta_q_p - expected_delta_q_p
            clock_residual_r = delta_q_r - expected_delta_q_r
            source_pair = math.fsum((delta_q_p, delta_q_r))
            q_prime_only_residual = q_prime_only - expected_delta_q_p
            momentum_pair = math.fsum((f_p, f_r))
            time_p = -(q_p * (1.0 + phi) + delta_q_p)
            time_r = -(q_r * (1.0 + phi) + delta_q_r)
            time_pair = math.fsum((time_p, time_r))
            spatial_bracket_p = f_p + q_p * velocity
            spatial_bracket_r = f_r + q_r * velocity
            spatial_p = kappa * spatial_bracket_p
            spatial_r = kappa * spatial_bracket_r
            spatial_pair = math.fsum((spatial_p, spatial_r))
        except OverflowError as error:
            raise ValueError("Qmu projection audit left the finite domain") from error
        finite_outputs = (
            expected_delta_q_p,
            q_prime_only,
            missing_hubble_term,
            background_pair,
            clock_residual_p,
            clock_residual_r,
            source_pair,
            q_prime_only_residual,
            momentum_pair,
            time_p,
            time_r,
            time_pair,
            spatial_bracket_p,
            spatial_bracket_r,
            spatial_p,
            spatial_r,
            spatial_pair,
        )
        if any(not math.isfinite(value) for value in finite_outputs):
            raise ValueError("Qmu projection audit left the finite domain")

        clock_p_holds = _within_roundoff(
            clock_residual_p,
            delta_q_p,
            expected_delta_q_p,
        )
        clock_r_holds = _within_roundoff(
            clock_residual_r,
            delta_q_r,
            expected_delta_q_r,
        )
        background_pair_holds = _within_roundoff(background_pair, q_p, q_r)
        source_pair_holds = _within_roundoff(
            source_pair,
            delta_q_p,
            delta_q_r,
        )
        momentum_pair_holds = _within_roundoff(momentum_pair, f_p, f_r)
        time_pair_holds = _within_roundoff(time_pair, time_p, time_r)
        spatial_pair_holds = _within_roundoff(
            spatial_pair,
            spatial_p,
            spatial_r,
        )

        return LowerQmuProjectionReceipt(
            n=n_value,
            k_over_a_h=kappa,
            scalar_clock_shift=clock,
            hubble_log_derivative=hubble_log_prime,
            lapse_potential=phi,
            normalized_total_velocity_potential=velocity,
            produced_background_q=q_p,
            reservoir_background_q=q_r,
            background_q_pair_residual=background_pair,
            produced_background_q_derivative=q_p_prime,
            expected_clock_physical_energy_perturbation=expected_delta_q_p,
            produced_physical_energy_perturbation=delta_q_p,
            reservoir_physical_energy_perturbation=delta_q_r,
            produced_physical_energy_clock_residual=clock_residual_p,
            reservoir_physical_energy_clock_residual=clock_residual_r,
            physical_energy_perturbation_pair_residual=source_pair,
            q_prime_only_physical_clock_residual=q_prime_only_residual,
            missing_hubble_normalization_term=missing_hubble_term,
            produced_intrinsic_momentum_potential=f_p,
            reservoir_intrinsic_momentum_potential=f_r,
            intrinsic_momentum_pair_residual=momentum_pair,
            produced_normalized_lower_time_component=time_p,
            reservoir_normalized_lower_time_component=time_r,
            normalized_lower_time_component_sum_residual=time_pair,
            produced_normalized_spatial_bracket=spatial_bracket_p,
            reservoir_normalized_spatial_bracket=spatial_bracket_r,
            produced_normalized_lower_spatial_fourier_scalar=spatial_p,
            reservoir_normalized_lower_spatial_fourier_scalar=spatial_r,
            normalized_lower_spatial_component_sum_residual=spatial_pair,
            common_clock_physical_source_holds=(
                clock_p_holds and clock_r_holds
            ),
            background_source_pair_cancels=background_pair_holds,
            physical_energy_perturbation_pair_cancels=source_pair_holds,
            intrinsic_momentum_pair_cancels=momentum_pair_holds,
            lower_time_component_pair_cancels=time_pair_holds,
            lower_spatial_component_pair_cancels=spatial_pair_holds,
            all_declared_lower_component_constraints_hold=(
                clock_p_holds
                and clock_r_holds
                and background_pair_holds
                and source_pair_holds
                and momentum_pair_holds
                and time_pair_holds
                and spatial_pair_holds
            ),
            dimensionless_roles=(
                ("q_A", "Q_A divided by H rho_unit"),
                ("deltaQhat_A", "delta Q_A divided by H rho_unit"),
                ("hubble_log_derivative", "d ln H / d ln a"),
                ("scalar_clock_shift", "dimensionless e-fold shift T"),
                ("lapse_potential", "dimensionless Newtonian phi"),
                ("V", "dimensionless aH(v+B), with B=0 here"),
                ("fhat_A", "dimensionless a f_A / rho_unit"),
                ("k_over_a_h", "dimensionless k/(aH)"),
            ),
        )
