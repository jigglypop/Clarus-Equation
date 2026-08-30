"""Singularity-free scalar-clock ledger for finite-quench perturbations.

This module records one small conditional algebraic theorem. Given a declared
dimensionless e-fold clock shift ``T`` and the finite-quench backgrounds,
define absolute (not fractional) density perturbations and source
perturbations by

    Delta rho_p = rho_p' T,       delta q_p = q' T,
    Delta rho_R = rho_R' T,       delta q_R = -delta q_p.

The clock residuals and paired-source residual then vanish. The formulation
never divides by ``rho_A``, ``rho_A'``, ``q``, or ``q'`` and therefore remains
defined at source endpoints and stationary background nodes. It is a declared
scalar ledger, not a covariant four-vector ``Q^mu``, not an Einstein--Boltzmann
system, and not a perturbation evolution solution.
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
class ScalarClockLedgerReceipt:
    """Algebraic audit receipt for one declared clock and Fourier node."""

    n: float
    scalar_clock_shift: float
    produced_density: float
    reservoir_density: float
    background_source: float
    produced_density_derivative: float
    reservoir_density_derivative: float
    background_source_derivative: float
    produced_density_perturbation: float
    reservoir_density_perturbation: float
    produced_energy_transfer_perturbation: float
    reservoir_energy_transfer_perturbation: float
    produced_density_clock_residual: float
    reservoir_density_clock_residual: float
    source_clock_residual: float
    paired_source_residual: float
    produced_density_clock_holds: bool
    reservoir_density_clock_holds: bool
    source_clock_holds: bool
    paired_source_cancels: bool
    all_declared_clock_constraints_hold: bool
    source_derivative_exact_float_zero: bool
    produced_clock_identifiable_from_density_derivative: bool
    reservoir_clock_identifiable_from_density_derivative: bool
    produced_density_contrast_or_none: float | None
    reservoir_density_contrast_or_none: float | None
    produced_density_contrast_status: str
    reservoir_density_contrast_status: str
    dimensionless_roles: tuple[tuple[str, str], ...]
    model: str = "declared_common_scalar_clock_ledger"
    role: str = (
        "CONDITIONAL_ALGEBRAIC_CLOCK_LEDGER_"
        "NOT_COVARIANT_QMU_OR_DYNAMICAL_SOLUTION"
    )


class FiniteQuenchScalarClockLedger:
    """Construct or audit the declared common-scalar-clock identities."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _background(
        self,
        n: object,
    ) -> tuple[float, float, float, float, float, float, float]:
        n_value = _finite_real(n, "n")
        if not self.bridge.config.n_initial <= n_value <= 0.0:
            raise ValueError("n is outside the scalar-clock ledger domain")
        return (
            n_value,
            self.bridge.production_density(n_value),
            self.bridge.reservoir_density(n_value),
            self.bridge.source(n_value),
            self.bridge.production_derivative(n_value),
            self.bridge.reservoir_derivative(n_value),
            self.bridge.source_derivative(n_value),
        )

    def construct(
        self,
        *,
        n: object,
        scalar_clock_shift: object,
    ) -> ScalarClockLedgerReceipt:
        """Construct the conditional identity from a declared clock shift."""

        (
            n_value,
            _rho_p,
            _rho_r,
            _source,
            rho_p_prime,
            rho_r_prime,
            source_prime,
        ) = self._background(n)
        clock = _finite_real(scalar_clock_shift, "scalar_clock_shift")
        try:
            delta_rho_p = rho_p_prime * clock
            delta_rho_r = rho_r_prime * clock
            delta_q_p = source_prime * clock
        except OverflowError as error:
            raise ValueError(
                "scalar-clock construction left the finite domain"
            ) from error
        return self.audit(
            n=n_value,
            scalar_clock_shift=clock,
            produced_density_perturbation=delta_rho_p,
            reservoir_density_perturbation=delta_rho_r,
            produced_energy_transfer_perturbation=delta_q_p,
            reservoir_energy_transfer_perturbation=-delta_q_p,
        )

    def audit(
        self,
        *,
        n: object,
        scalar_clock_shift: object,
        produced_density_perturbation: object,
        reservoir_density_perturbation: object,
        produced_energy_transfer_perturbation: object,
        reservoir_energy_transfer_perturbation: object,
    ) -> ScalarClockLedgerReceipt:
        """Check arbitrary absolute perturbations against the clock ledger."""

        (
            n_value,
            rho_p,
            rho_r,
            source,
            rho_p_prime,
            rho_r_prime,
            source_prime,
        ) = self._background(n)
        clock = _finite_real(scalar_clock_shift, "scalar_clock_shift")
        delta_rho_p = _finite_real(
            produced_density_perturbation,
            "produced_density_perturbation",
        )
        delta_rho_r = _finite_real(
            reservoir_density_perturbation,
            "reservoir_density_perturbation",
        )
        delta_q_p = _finite_real(
            produced_energy_transfer_perturbation,
            "produced_energy_transfer_perturbation",
        )
        delta_q_r = _finite_real(
            reservoir_energy_transfer_perturbation,
            "reservoir_energy_transfer_perturbation",
        )
        try:
            expected_delta_rho_p = rho_p_prime * clock
            expected_delta_rho_r = rho_r_prime * clock
            expected_delta_q_p = source_prime * clock
            residual_rho_p = delta_rho_p - expected_delta_rho_p
            residual_rho_r = delta_rho_r - expected_delta_rho_r
            residual_q = delta_q_p - expected_delta_q_p
            paired_q = math.fsum((delta_q_p, delta_q_r))
        except OverflowError as error:
            raise ValueError("scalar-clock audit left the finite domain") from error
        finite_outputs = (
            expected_delta_rho_p,
            expected_delta_rho_r,
            expected_delta_q_p,
            residual_rho_p,
            residual_rho_r,
            residual_q,
            paired_q,
        )
        if any(not math.isfinite(value) for value in finite_outputs):
            raise ValueError("scalar-clock audit left the finite domain")

        density_p_holds = _within_roundoff(
            residual_rho_p,
            delta_rho_p,
            expected_delta_rho_p,
        )
        density_r_holds = _within_roundoff(
            residual_rho_r,
            delta_rho_r,
            expected_delta_rho_r,
        )
        source_holds = _within_roundoff(
            residual_q,
            delta_q_p,
            expected_delta_q_p,
        )
        pair_holds = _within_roundoff(paired_q, delta_q_p, delta_q_r)

        if rho_p == 0.0:
            contrast_p = None
            contrast_p_status = "UNDEFINED_ZERO_BACKGROUND_DENSITY"
        else:
            contrast_p = delta_rho_p / rho_p
            if math.isfinite(contrast_p):
                contrast_p_status = "FINITE_READOUT"
            else:
                contrast_p = None
                contrast_p_status = "UNAVAILABLE_FLOAT_REPRESENTATION"
        if rho_r == 0.0:
            contrast_r = None
            contrast_r_status = "UNDEFINED_ZERO_BACKGROUND_DENSITY"
        else:
            contrast_r = delta_rho_r / rho_r
            if math.isfinite(contrast_r):
                contrast_r_status = "FINITE_READOUT"
            else:
                contrast_r = None
                contrast_r_status = "UNAVAILABLE_FLOAT_REPRESENTATION"

        return ScalarClockLedgerReceipt(
            n=n_value,
            scalar_clock_shift=clock,
            produced_density=rho_p,
            reservoir_density=rho_r,
            background_source=source,
            produced_density_derivative=rho_p_prime,
            reservoir_density_derivative=rho_r_prime,
            background_source_derivative=source_prime,
            produced_density_perturbation=delta_rho_p,
            reservoir_density_perturbation=delta_rho_r,
            produced_energy_transfer_perturbation=delta_q_p,
            reservoir_energy_transfer_perturbation=delta_q_r,
            produced_density_clock_residual=residual_rho_p,
            reservoir_density_clock_residual=residual_rho_r,
            source_clock_residual=residual_q,
            paired_source_residual=paired_q,
            produced_density_clock_holds=density_p_holds,
            reservoir_density_clock_holds=density_r_holds,
            source_clock_holds=source_holds,
            paired_source_cancels=pair_holds,
            all_declared_clock_constraints_hold=(
                density_p_holds
                and density_r_holds
                and source_holds
                and pair_holds
            ),
            source_derivative_exact_float_zero=(source_prime == 0.0),
            produced_clock_identifiable_from_density_derivative=(
                rho_p_prime != 0.0
            ),
            reservoir_clock_identifiable_from_density_derivative=(
                rho_r_prime != 0.0
            ),
            produced_density_contrast_or_none=contrast_p,
            reservoir_density_contrast_or_none=contrast_r,
            produced_density_contrast_status=contrast_p_status,
            reservoir_density_contrast_status=contrast_r_status,
            dimensionless_roles=(
                ("n", "dimensionless ln(a)"),
                ("scalar_clock_shift", "dimensionless e-fold shift T"),
                ("rho_A", "normalized background density"),
                ("Delta_rho_A", "absolute normalized density perturbation"),
                ("q", "normalized density transfer per e-fold"),
                ("delta_q_A", "perturbation of normalized transfer q"),
            ),
        )
