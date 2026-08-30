"""Strict causal barotropic constitutive branch for finite-quench fluids.

This is an explicit effective-fluid axiom, not a consequence of the
microscopic quench.  The produced component is strict dust.  The reservoir is
a constant local barotrope with 0 <= w_R <= 1.  Thus

    Delta P_p = 0,              pihat_p = 0,
    Delta P_R = w_R Delta rho_R, pihat_R = 0,

and c_s^2=c_a^2=w for both components.  The nonadiabatic interaction term in
Eq. (29) of Valiviita et al., arXiv:0804.0232, vanishes because its coefficient
c_s^2-c_a^2 is zero.  General interacting fluids do not inherit this closure.
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
class StrictBarotropicClosureReceipt:
    """One-node constitutive residual audit for the two-fluid branch."""

    n: float
    reservoir_equation_of_state: float
    produced_density_perturbation: float
    reservoir_density_perturbation: float
    produced_pressure_perturbation: float
    reservoir_pressure_perturbation: float
    produced_normalized_anisotropic_stress: float
    reservoir_normalized_anisotropic_stress: float
    produced_pressure_closure_residual: float
    reservoir_pressure_closure_residual: float
    produced_anisotropic_stress_residual: float
    reservoir_anisotropic_stress_residual: float
    produced_background_pressure_derivative: float
    reservoir_background_pressure_derivative: float
    produced_background_barotrope_derivative_residual: float
    reservoir_background_barotrope_derivative_residual: float
    produced_rest_frame_sound_speed_squared: float
    reservoir_rest_frame_sound_speed_squared: float
    produced_adiabatic_sound_speed_squared: float
    reservoir_adiabatic_sound_speed_squared: float
    nonadiabatic_interaction_coefficient_produced: float
    nonadiabatic_interaction_coefficient_reservoir: float
    pressure_closure_holds: bool
    zero_anisotropic_stress_holds: bool
    background_barotrope_derivative_holds: bool
    causal_nonnegative_sound_speed_branch: bool
    all_strict_barotropic_constraints_hold: bool
    model: str = "strict_dust_plus_causal_constant_barotrope_zero_pi"
    role: str = (
        "CONDITIONAL_CONSTITUTIVE_FLUID_CLOSURE_"
        "NOT_MICROPHYSICAL_QUENCH_DERIVATION"
    )


class FiniteQuenchStrictBarotropicClosure:
    """Construct or audit the strict dust-plus-barotrope closure."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        w_r = bridge.config.w_reservoir
        if not 0.0 <= w_r <= 1.0:
            raise ValueError(
                "strict causal barotropic reservoir requires 0 <= w_reservoir <= 1"
            )
        self.bridge = bridge

    def construct(
        self,
        *,
        n: object,
        produced_density_perturbation: object,
        reservoir_density_perturbation: object,
    ) -> StrictBarotropicClosureReceipt:
        delta_rho_p = _finite_real(
            produced_density_perturbation,
            "produced_density_perturbation",
        )
        delta_rho_r = _finite_real(
            reservoir_density_perturbation,
            "reservoir_density_perturbation",
        )
        n_value = _finite_real(n, "n")
        if not self.bridge.config.n_initial <= n_value <= 0.0:
            raise ValueError("n is outside the barotropic closure domain")
        rho_r_prime = self.bridge.reservoir_derivative(n_value)
        return self.audit(
            n=n_value,
            produced_density_perturbation=delta_rho_p,
            reservoir_density_perturbation=delta_rho_r,
            produced_pressure_perturbation=0.0,
            reservoir_pressure_perturbation=(
                self.bridge.config.w_reservoir * delta_rho_r
            ),
            produced_normalized_anisotropic_stress=0.0,
            reservoir_normalized_anisotropic_stress=0.0,
            produced_background_pressure_derivative=0.0,
            reservoir_background_pressure_derivative=(
                self.bridge.config.w_reservoir * rho_r_prime
            ),
        )

    def audit(
        self,
        *,
        n: object,
        produced_density_perturbation: object,
        reservoir_density_perturbation: object,
        produced_pressure_perturbation: object,
        reservoir_pressure_perturbation: object,
        produced_normalized_anisotropic_stress: object,
        reservoir_normalized_anisotropic_stress: object,
        produced_background_pressure_derivative: object,
        reservoir_background_pressure_derivative: object,
    ) -> StrictBarotropicClosureReceipt:
        n_value = _finite_real(n, "n")
        if not self.bridge.config.n_initial <= n_value <= 0.0:
            raise ValueError("n is outside the barotropic closure domain")
        delta_rho_p = _finite_real(
            produced_density_perturbation,
            "produced_density_perturbation",
        )
        delta_rho_r = _finite_real(
            reservoir_density_perturbation,
            "reservoir_density_perturbation",
        )
        delta_pressure_p = _finite_real(
            produced_pressure_perturbation,
            "produced_pressure_perturbation",
        )
        delta_pressure_r = _finite_real(
            reservoir_pressure_perturbation,
            "reservoir_pressure_perturbation",
        )
        pi_p = _finite_real(
            produced_normalized_anisotropic_stress,
            "produced_normalized_anisotropic_stress",
        )
        pi_r = _finite_real(
            reservoir_normalized_anisotropic_stress,
            "reservoir_normalized_anisotropic_stress",
        )
        pressure_prime_p = _finite_real(
            produced_background_pressure_derivative,
            "produced_background_pressure_derivative",
        )
        pressure_prime_r = _finite_real(
            reservoir_background_pressure_derivative,
            "reservoir_background_pressure_derivative",
        )
        w_r = self.bridge.config.w_reservoir
        expected_pressure_p = 0.0
        expected_pressure_r = w_r * delta_rho_r
        pressure_residual_p = delta_pressure_p - expected_pressure_p
        pressure_residual_r = delta_pressure_r - expected_pressure_r
        rho_p_prime = self.bridge.production_derivative(n_value)
        rho_r_prime = self.bridge.reservoir_derivative(n_value)
        expected_pressure_prime_p = 0.0
        expected_pressure_prime_r = w_r * rho_r_prime
        background_derivative_residual_p = (
            pressure_prime_p - expected_pressure_prime_p
        )
        background_derivative_residual_r = (
            pressure_prime_r - expected_pressure_prime_r
        )
        pressure_holds = (
            _within_roundoff(
                pressure_residual_p,
                delta_pressure_p,
                expected_pressure_p,
            )
            and _within_roundoff(
                pressure_residual_r,
                delta_pressure_r,
                expected_pressure_r,
            )
        )
        anisotropic_holds = (
            _within_roundoff(pi_p, pi_p, 0.0)
            and _within_roundoff(pi_r, pi_r, 0.0)
        )
        background_holds = (
            _within_roundoff(
                background_derivative_residual_p,
                pressure_prime_p,
                expected_pressure_prime_p,
            )
            and _within_roundoff(
                background_derivative_residual_r,
                pressure_prime_r,
                expected_pressure_prime_r,
            )
        )
        causal = 0.0 <= w_r <= 1.0

        return StrictBarotropicClosureReceipt(
            n=n_value,
            reservoir_equation_of_state=w_r,
            produced_density_perturbation=delta_rho_p,
            reservoir_density_perturbation=delta_rho_r,
            produced_pressure_perturbation=delta_pressure_p,
            reservoir_pressure_perturbation=delta_pressure_r,
            produced_normalized_anisotropic_stress=pi_p,
            reservoir_normalized_anisotropic_stress=pi_r,
            produced_pressure_closure_residual=pressure_residual_p,
            reservoir_pressure_closure_residual=pressure_residual_r,
            produced_anisotropic_stress_residual=pi_p,
            reservoir_anisotropic_stress_residual=pi_r,
            produced_background_pressure_derivative=pressure_prime_p,
            reservoir_background_pressure_derivative=pressure_prime_r,
            produced_background_barotrope_derivative_residual=(
                background_derivative_residual_p
            ),
            reservoir_background_barotrope_derivative_residual=(
                background_derivative_residual_r
            ),
            produced_rest_frame_sound_speed_squared=0.0,
            reservoir_rest_frame_sound_speed_squared=w_r,
            produced_adiabatic_sound_speed_squared=0.0,
            reservoir_adiabatic_sound_speed_squared=w_r,
            nonadiabatic_interaction_coefficient_produced=0.0,
            nonadiabatic_interaction_coefficient_reservoir=0.0,
            pressure_closure_holds=pressure_holds,
            zero_anisotropic_stress_holds=anisotropic_holds,
            background_barotrope_derivative_holds=background_holds,
            causal_nonnegative_sound_speed_branch=causal,
            all_strict_barotropic_constraints_hold=(
                pressure_holds and anisotropic_holds and background_holds and causal
            ),
        )
