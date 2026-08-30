"""Minimal linear exchange contract for the finite-quench two-fluid branch.

Background conservation does not determine perturbations.  This module makes a
new, deliberately small effective-model choice in one declared scalar ledger
frame:

    delta_q_p = b_q q delta_p,       delta_q_R = -delta_q_p,

    f_p = -b_f q (theta_p-theta_R),  f_R = -f_p.

Here ``theta_A`` and ``k/(aH)`` are dimensionless, so ``f_A`` has the
same normalized-density-per-e-fold role as ``q``.  The opposite pairs prove
algebraic exchange cancellation, and ``b_f >= 0`` makes the quadratic proxy
``b_f q (theta_p-theta_R)^2`` non-negative.  For a regular scalar Fourier
potential the spatial vector is proportional to ``i k_i f_A``; the
normalized gradient proxy therefore vanishes at ``k=0``.

This is an algebraic source ledger, not the Einstein-Boltzmann equations.  It
does not derive ``delta q``, momentum transfer, sound speeds, metric
constraints, or initial conditions from the microscopic quench.
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


@dataclass(frozen=True)
class FiniteQuenchPerturbationAxioms:
    """Explicit choices needed before perturbation evolution can be written."""

    gauge: str
    transfer_frame: str
    initial_mode: str
    anisotropic_stress_model: str
    energy_transfer_bias: float
    momentum_drag_bias: float
    produced_equation_of_state: float
    produced_sound_speed_squared: float
    reservoir_sound_speed_squared: float

    def __post_init__(self) -> None:
        if self.gauge != "newtonian":
            raise ValueError("this branch requires gauge='newtonian'")
        if self.transfer_frame != "declared_common_scalar_frame":
            raise ValueError(
                "this branch requires "
                "transfer_frame='declared_common_scalar_frame'"
            )
        if self.initial_mode != "density_equal_time_shift_seed":
            raise ValueError(
                "this branch requires "
                "initial_mode='density_equal_time_shift_seed'"
            )
        if self.anisotropic_stress_model != "zero":
            raise ValueError(
                "this branch requires anisotropic_stress_model='zero'"
            )
        for name in (
            "energy_transfer_bias",
            "momentum_drag_bias",
            "produced_equation_of_state",
            "produced_sound_speed_squared",
            "reservoir_sound_speed_squared",
        ):
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name),
            )
        if self.momentum_drag_bias < 0.0:
            raise ValueError("momentum_drag_bias must be >= 0")
        if self.produced_equation_of_state != 0.0:
            raise ValueError(
                "the finite-quench background requires "
                "produced_equation_of_state == 0"
            )
        for name in (
            "produced_sound_speed_squared",
            "reservoir_sound_speed_squared",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class PerturbationExchangeReceipt:
    """One-mode algebraic receipt for the declared perturbation source split."""

    n: float
    k_over_a_h: float
    background_source: float
    background_source_derivative: float
    produced_energy_transfer: float
    reservoir_energy_transfer: float
    energy_transfer_sum_residual: float
    produced_momentum_potential: float
    reservoir_momentum_potential: float
    momentum_potential_sum_residual: float
    produced_normalized_gradient_proxy: float
    reservoir_normalized_gradient_proxy: float
    normalized_gradient_sum_residual: float
    nonnegative_drag_quadratic_proxy: float
    noninteracting_barotropic_entropy_proxy: float
    density_equal_time_shift_residual: float | None
    density_equal_time_shift_status: str
    density_equal_time_shift_at_this_node: bool | None
    declared_source_time_shift_residual: float | None
    density_and_source_equal_time_shift_at_this_node: bool | None
    source_time_shift_diagnostic_status: str
    declared_sound_speeds_in_unit_interval: bool
    dimensionless_roles: tuple[tuple[str, str], ...]
    role: str = (
        "CONDITIONAL_LINEAR_EXCHANGE_LEDGER_NOT_EINSTEIN_BOLTZMANN_SOLUTION"
    )


class FiniteQuenchPerturbationContract:
    """Pair a finite background source with explicit linear exchange axioms."""

    def __init__(
        self,
        bridge: FiniteQuenchBridge,
        axioms: FiniteQuenchPerturbationAxioms,
    ) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        if not isinstance(axioms, FiniteQuenchPerturbationAxioms):
            raise ValueError("axioms must be FiniteQuenchPerturbationAxioms")
        if bridge.config.w_reservoir <= -1.0:
            raise ValueError(
                "this fluid perturbation branch requires w_reservoir > -1"
            )
        self.bridge = bridge
        self.axioms = axioms

    def reservoir_delta_for_density_equal_time_shift(
        self,
        n: object,
        produced_delta: object,
    ) -> float:
        n_value = _finite_real(n, "n")
        delta_p = _finite_real(produced_delta, "produced_delta")
        if not self.bridge.config.n_initial <= n_value <= 0.0:
            raise ValueError("n is outside the perturbation contract domain")
        rho_p = self.bridge.production_density(n_value)
        rho_r = self.bridge.reservoir_density(n_value)
        rho_p_prime = self.bridge.production_derivative(n_value)
        rho_r_prime = self.bridge.reservoir_derivative(n_value)
        if (
            rho_p == 0.0
            or rho_r == 0.0
            or rho_p_prime == 0.0
            or rho_r_prime == 0.0
        ):
            raise ValueError(
                "equal-time-shift density pair is undefined at this node"
            )
        produced_time_shift = rho_p * delta_p / rho_p_prime
        result = produced_time_shift * rho_r_prime / rho_r
        if not math.isfinite(result):
            raise ValueError(
                "density equal-time-shift delta left the finite domain"
            )
        return result

    def receipt(
        self,
        *,
        n: object,
        k_over_a_h: object,
        produced_delta: object,
        reservoir_delta: object,
        produced_velocity_divergence: object,
        reservoir_velocity_divergence: object,
    ) -> PerturbationExchangeReceipt:
        n_value = _finite_real(n, "n")
        k_value = _finite_real(k_over_a_h, "k_over_a_h")
        delta_p = _finite_real(produced_delta, "produced_delta")
        delta_r = _finite_real(reservoir_delta, "reservoir_delta")
        theta_p = _finite_real(
            produced_velocity_divergence,
            "produced_velocity_divergence",
        )
        theta_r = _finite_real(
            reservoir_velocity_divergence,
            "reservoir_velocity_divergence",
        )
        if not self.bridge.config.n_initial <= n_value <= 0.0:
            raise ValueError("n is outside the perturbation contract domain")
        if k_value < 0.0:
            raise ValueError("k_over_a_h must be >= 0")

        source = self.bridge.source(n_value)
        source_prime = self.bridge.source_derivative(n_value)
        relative_velocity = theta_p - theta_r
        if not math.isfinite(relative_velocity):
            raise ValueError("relative velocity left the finite domain")
        try:
            delta_q_p = (
                self.axioms.energy_transfer_bias * source * delta_p
            )
            delta_q_r = -delta_q_p
            f_p = (
                -self.axioms.momentum_drag_bias
                * source
                * relative_velocity
            )
            f_r = -f_p
            gradient_p = k_value * f_p
            gradient_r = k_value * f_r
            entropy_proxy = (
                delta_p
                - delta_r / (1.0 + self.bridge.config.w_reservoir)
            )
            drag_quadratic = (
                self.axioms.momentum_drag_bias
                * source
                * relative_velocity**2
            )
        except OverflowError as error:
            raise ValueError(
                "perturbation receipt left the finite domain"
            ) from error

        rho_p = self.bridge.production_density(n_value)
        rho_r = self.bridge.reservoir_density(n_value)
        rho_p_prime = self.bridge.production_derivative(n_value)
        rho_r_prime = self.bridge.reservoir_derivative(n_value)
        if (
            rho_p == 0.0
            or rho_r == 0.0
            or rho_p_prime == 0.0
            or rho_r_prime == 0.0
        ):
            produced_time_shift = None
            time_shift_residual = None
            time_shift_status = (
                "UNDEFINED_AT_ZERO_DENSITY_OR_STATIONARY_BACKGROUND_NODE"
            )
            density_time_shift_equal = None
        else:
            produced_time_shift = rho_p * delta_p / rho_p_prime
            reservoir_time_shift = rho_r * delta_r / rho_r_prime
            time_shift_residual = produced_time_shift - reservoir_time_shift
            time_shift_scale = max(
                1.0,
                abs(produced_time_shift),
                abs(reservoir_time_shift),
            )
            density_time_shift_equal = (
                abs(time_shift_residual)
                <= 64.0 * math.ulp(time_shift_scale)
            )
            time_shift_status = (
                "DEFINED_FROM_INTERACTING_BACKGROUND_DERIVATIVES"
            )
        if produced_time_shift is None:
            source_time_shift_residual = None
            density_and_source_equal = None
            source_time_shift_status = "DENSITY_CLOCK_UNAVAILABLE"
        else:
            expected_delta_q = source_prime * produced_time_shift
            source_time_shift_residual = delta_q_p - expected_delta_q
            source_shift_scale = max(
                1.0,
                abs(delta_q_p),
                abs(expected_delta_q),
            )
            source_time_shift_equal = (
                abs(source_time_shift_residual)
                <= 64.0 * math.ulp(source_shift_scale)
            )
            density_and_source_equal = bool(
                density_time_shift_equal and source_time_shift_equal
            )
            if source == 0.0 and source_prime == 0.0:
                source_time_shift_status = "VACUOUS_NO_SOURCE"
            elif produced_time_shift == 0.0:
                source_time_shift_status = "TRIVIAL_ZERO_CLOCK_SHIFT"
            else:
                source_time_shift_status = "NONDEGENERATE_NODE_DIAGNOSTIC"
        finite_outputs = (
            source,
            source_prime,
            delta_q_p,
            f_p,
            gradient_p,
            entropy_proxy,
            drag_quadratic,
        )
        if time_shift_residual is not None:
            finite_outputs = (*finite_outputs, time_shift_residual)
        if source_time_shift_residual is not None:
            finite_outputs = (*finite_outputs, source_time_shift_residual)
        if any(not math.isfinite(value) for value in finite_outputs):
            raise ValueError("perturbation receipt left the finite domain")
        if drag_quadratic < 0.0:
            raise ValueError("relative drag quadratic must be non-negative")

        return PerturbationExchangeReceipt(
            n=n_value,
            k_over_a_h=k_value,
            background_source=source,
            background_source_derivative=source_prime,
            produced_energy_transfer=delta_q_p,
            reservoir_energy_transfer=delta_q_r,
            energy_transfer_sum_residual=math.fsum((delta_q_p, delta_q_r)),
            produced_momentum_potential=f_p,
            reservoir_momentum_potential=f_r,
            momentum_potential_sum_residual=math.fsum((f_p, f_r)),
            produced_normalized_gradient_proxy=gradient_p,
            reservoir_normalized_gradient_proxy=gradient_r,
            normalized_gradient_sum_residual=math.fsum(
                (gradient_p, gradient_r)
            ),
            nonnegative_drag_quadratic_proxy=drag_quadratic,
            noninteracting_barotropic_entropy_proxy=entropy_proxy,
            density_equal_time_shift_residual=time_shift_residual,
            density_equal_time_shift_status=time_shift_status,
            density_equal_time_shift_at_this_node=density_time_shift_equal,
            declared_source_time_shift_residual=source_time_shift_residual,
            density_and_source_equal_time_shift_at_this_node=(
                density_and_source_equal
            ),
            source_time_shift_diagnostic_status=source_time_shift_status,
            declared_sound_speeds_in_unit_interval=(
                0.0 <= self.axioms.produced_sound_speed_squared <= 1.0
                and 0.0 <= self.axioms.reservoir_sound_speed_squared <= 1.0
            ),
            dimensionless_roles=(
                ("n", "dimensionless ln(a)"),
                ("k_over_a_h", "dimensionless Fourier scale k/(aH)"),
                ("delta_A", "dimensionless density contrast"),
                ("theta_A", "dimensionless velocity divergence theta/(aH)"),
                ("delta_q_A", "normalized density transfer per e-fold"),
                (
                    "f_A",
                    "declared normalized scalar momentum-transfer potential",
                ),
                (
                    "k_over_a_h*f_A",
                    "normalized Fourier-gradient proxy, not physical Q_A^i",
                ),
            ),
        )
