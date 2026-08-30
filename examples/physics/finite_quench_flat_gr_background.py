"""Two-component-only flat-GR background normalization for the finite quench.

This is a deliberately narrow cosmological branch.  The produced and
reservoir fluids are declared to be the complete background species manifest,
the density unit is constant, spatial curvature is zero, and the expanding GR
Friedmann branch is selected.  In density-unit variables,

    H^2 / [(8 pi G/3) rho_unit] = rho_total,
    Omega_unit = 1/rho_total,
    C = 4 pi G rho_unit/H^2 = 3/(2 rho_total),
    h = D_n ln H = -(3/2) (rho_total+P_total)/rho_total.

This does not assert that the real universe contains only these two fluids.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
)


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 64.0 * math.ulp(scale)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


@dataclass(frozen=True)
class TwoFluidFlatGRBackgroundReceipt:
    """One-node Friedmann/Raychaudhuri/continuity normalization receipt."""

    n: float
    produced_density: float
    reservoir_density: float
    total_density: float
    produced_pressure: float
    reservoir_pressure: float
    total_pressure: float
    total_enthalpy: float
    produced_density_derivative: float
    reservoir_density_derivative: float
    total_density_derivative: float
    produced_source: float
    reservoir_source: float
    source_pair_residual: float
    total_continuity_residual: float
    hubble_squared_over_eight_pi_g_rho_unit_over_three: float
    omega_density_unit: float
    gravity_constraint_coupling: float
    hubble_log_derivative: float
    kappa_log_derivative_at_fixed_comoving_k: float
    friedmann_normalization_residual: float
    raychaudhuri_normalization_residual: float
    source_pair_cancels: bool
    total_continuity_holds: bool
    friedmann_normalization_holds: bool
    raychaudhuri_normalization_holds: bool
    all_background_constraints_hold: bool
    species_manifest: tuple[str, ...] = ("produced", "reservoir")
    external_background_species_assumed_absent: bool = True
    role: str = (
        "CONDITIONAL_TWO_COMPONENT_ONLY_FLAT_GR_BACKGROUND_"
        "NOT_OBSERVED_COSMOLOGY"
    )


class FiniteQuenchTwoFluidFlatGRBackground:
    """Audit the two-fluid finite bridge as a complete flat-GR background."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _validated_n(self, n: object) -> float:
        n_value = _finite_real(n, "n")
        if not self.bridge.config.n_initial <= n_value <= 0.0:
            raise ValueError("n is outside the flat-GR background domain")
        return n_value

    def construct(self, n: object) -> TwoFluidFlatGRBackgroundReceipt:
        """Construct H and D_n ln H from the declared flat-GR branch."""

        n_value = self._validated_n(n)
        rho_p = self.bridge.production_density(n_value)
        rho_r = self.bridge.reservoir_density(n_value)
        rho_total = math.fsum((rho_p, rho_r))
        pressure_total = self.bridge.config.w_reservoir * rho_r
        if not math.isfinite(rho_total) or rho_total <= 0.0:
            raise ValueError("two-fluid flat-GR branch requires positive total density")
        expected_hubble_log_prime = (
            -1.5 * (rho_total + pressure_total) / rho_total
        )
        return self.audit(
            n=n_value,
            normalized_hubble_squared=rho_total,
            hubble_log_derivative=expected_hubble_log_prime,
        )

    def audit(
        self,
        n: object,
        *,
        normalized_hubble_squared: object,
        hubble_log_derivative: object,
    ) -> TwoFluidFlatGRBackgroundReceipt:
        """Audit independently supplied H^2 and D_n ln H candidates."""

        n_value = self._validated_n(n)
        hubble_squared_normalized = _finite_real(
            normalized_hubble_squared,
            "normalized_hubble_squared",
        )
        if hubble_squared_normalized <= 0.0:
            raise ValueError("expanding flat-GR branch requires positive H squared")
        hubble_log_prime = _finite_real(
            hubble_log_derivative,
            "hubble_log_derivative",
        )

        rho_p = self.bridge.production_density(n_value)
        rho_r = self.bridge.reservoir_density(n_value)
        pressure_p = 0.0
        pressure_r = self.bridge.config.w_reservoir * rho_r
        rho_total = math.fsum((rho_p, rho_r))
        pressure_total = math.fsum((pressure_p, pressure_r))
        enthalpy_total = rho_total + pressure_total
        if not math.isfinite(rho_total) or rho_total <= 0.0:
            raise ValueError("two-fluid flat-GR branch requires positive total density")
        rho_p_prime = self.bridge.production_derivative(n_value)
        rho_r_prime = self.bridge.reservoir_derivative(n_value)
        rho_total_prime = math.fsum((rho_p_prime, rho_r_prime))
        source_p = self.bridge.source(n_value)
        source_r = -source_p
        source_pair = math.fsum((source_p, source_r))
        continuity_residual = rho_total_prime + 3.0 * enthalpy_total
        omega_unit = 1.0 / hubble_squared_normalized
        gravity_coupling = 1.5 * omega_unit
        kappa_log_prime = -1.0 - hubble_log_prime
        friedmann_residual = hubble_squared_normalized - rho_total
        raychaudhuri_residual = (
            hubble_log_prime + 1.5 * enthalpy_total / rho_total
        )
        finite_outputs = (
            pressure_total,
            enthalpy_total,
            rho_total_prime,
            source_pair,
            continuity_residual,
            hubble_squared_normalized,
            omega_unit,
            gravity_coupling,
            hubble_log_prime,
            kappa_log_prime,
            friedmann_residual,
            raychaudhuri_residual,
        )
        if any(not math.isfinite(value) for value in finite_outputs):
            raise ValueError("flat-GR background receipt left the finite domain")
        source_holds = _within_roundoff(source_pair, source_p, source_r)
        continuity_holds = _within_roundoff(
            continuity_residual,
            rho_total_prime,
            3.0 * enthalpy_total,
        )
        friedmann_holds = _within_roundoff(
            friedmann_residual,
            hubble_squared_normalized,
            rho_total,
        )
        raychaudhuri_holds = _within_roundoff(
            raychaudhuri_residual,
            hubble_log_prime,
            1.5 * enthalpy_total / rho_total,
        )

        return TwoFluidFlatGRBackgroundReceipt(
            n=n_value,
            produced_density=rho_p,
            reservoir_density=rho_r,
            total_density=rho_total,
            produced_pressure=pressure_p,
            reservoir_pressure=pressure_r,
            total_pressure=pressure_total,
            total_enthalpy=enthalpy_total,
            produced_density_derivative=rho_p_prime,
            reservoir_density_derivative=rho_r_prime,
            total_density_derivative=rho_total_prime,
            produced_source=source_p,
            reservoir_source=source_r,
            source_pair_residual=source_pair,
            total_continuity_residual=continuity_residual,
            hubble_squared_over_eight_pi_g_rho_unit_over_three=(
                hubble_squared_normalized
            ),
            omega_density_unit=omega_unit,
            gravity_constraint_coupling=gravity_coupling,
            hubble_log_derivative=hubble_log_prime,
            kappa_log_derivative_at_fixed_comoving_k=kappa_log_prime,
            friedmann_normalization_residual=friedmann_residual,
            raychaudhuri_normalization_residual=raychaudhuri_residual,
            source_pair_cancels=source_holds,
            total_continuity_holds=continuity_holds,
            friedmann_normalization_holds=friedmann_holds,
            raychaudhuri_normalization_holds=raychaudhuri_holds,
            all_background_constraints_hold=(
                source_holds
                and continuity_holds
                and friedmann_holds
                and raychaudhuri_holds
            ),
        )
