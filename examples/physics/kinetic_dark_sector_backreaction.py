"""Exact dimensionless FLRW Ward/backreaction closure for a dynamic clock.

The declared action is

    S = integral sqrt(-g) [
        Mpl^2 R/2 - (partial T)^2/2 - V(T)
        - sum_s sum_A ((partial phi_sA)^2
                       + (m_s(T)^2 + xi_s R) phi_sA^2)/2
    ].

Use theta=T/H0, N=log(a), E=H/H0, M=Mpl/H0, r=rho/H0^4,
pi=p/H0^4, v=V/H0^4, and Phi_s^2=<phi_sA^2>/H0^2 for one real
component.  With g_s identical components,

    j_s = g_s (d mu_s^2/d theta) Phi_s^2 / 2,
    r_s,N + 3(r_s+pi_s) = theta_N j_s,

while the canonical clock obeys

    E^2[theta_NN + (3+E_N/E)theta_N] + v_theta + sum_s j_s = 0.

The clock Ward residual is theta_N times this equation.  Consequently the
total Ward identity is algebraic, not a fitted numerical cancellation.  If
the Raychaudhuri equation also holds, the derivative of the Friedmann
constraint is exactly zero:

    C_N = 6 M^2 R_Raychaudhuri - W_total.

This module proves that conditional closure.  The scalar inputs must already
come from a common regulator/counterterm triplet; the module does not create
renormalized stress by ledger arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ScalarBackreactionChannel:
    """One species with explicit single-field versus multiplet conventions.

    ``field_squared`` is for one of ``degeneracy`` identical fields, whereas
    ``energy_density``, ``pressure``, and ``energy_density_d_n`` are already
    degeneracy-summed multiplet totals.
    """

    degeneracy: int
    energy_density: float
    pressure: float
    energy_density_d_n: float
    field_squared: float
    mass_squared_d_theta: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.degeneracy, bool)
            or not isinstance(self.degeneracy, int)
            or self.degeneracy < 1
        ):
            raise ValueError("degeneracy must be a positive integer")
        for name, value in (
            ("energy_density", self.energy_density),
            ("pressure", self.pressure),
            ("energy_density_d_n", self.energy_density_d_n),
            ("field_squared", self.field_squared),
            ("mass_squared_d_theta", self.mass_squared_d_theta),
        ):
            _finite(name, value)

    @property
    def clock_force(self) -> float:
        return (
            0.5
            * self.degeneracy
            * self.mass_squared_d_theta
            * self.field_squared
        )

    def ward_residual(self, theta_d_n: float) -> float:
        theta_d_n = _finite("theta_d_n", theta_d_n)
        return (
            self.energy_density_d_n
            + 3.0 * (self.energy_density + self.pressure)
            - theta_d_n * self.clock_force
        )


@dataclass(frozen=True)
class ConservedFluid:
    energy_density: float
    pressure: float
    energy_density_d_n: float

    def __post_init__(self) -> None:
        for name, value in (
            ("energy_density", self.energy_density),
            ("pressure", self.pressure),
            ("energy_density_d_n", self.energy_density_d_n),
        ):
            _finite(name, value)

    @property
    def ward_residual(self) -> float:
        return self.energy_density_d_n + 3.0 * (
            self.energy_density + self.pressure
        )


@dataclass(frozen=True)
class BackreactionClosureReceipt:
    scalar_ward_residuals: tuple[float, ...]
    clock_equation_residual: float
    clock_ward_residual: float
    clock_ward_factorization_residual: float
    total_ward_residual: float
    friedmann_constraint_residual: float
    raychaudhuri_residual: float
    friedmann_constraint_derivative: float
    constraint_propagation_identity_residual: float
    clock_energy_density: float
    clock_pressure: float
    total_energy_density: float
    total_pressure: float
    total_clock_force: float
    status: str = "CONDITIONAL_DYNAMIC_CLOCK_FLRW_CLOSURE"


def backreaction_closure_receipt(
    *,
    e: float,
    d_log_e_d_n: float,
    reduced_planck_over_h0: float,
    theta_d_n: float,
    theta_d2_n: float,
    potential: float,
    potential_d_theta: float,
    scalar_channels: tuple[ScalarBackreactionChannel, ...],
    conserved_fluids: tuple[ConservedFluid, ...] = (),
) -> BackreactionClosureReceipt:
    """Audit scalar transfer, clock response, and constraint propagation."""

    e = _finite("e", e)
    d_log_e_d_n = _finite("d_log_e_d_n", d_log_e_d_n)
    reduced_planck_over_h0 = _finite(
        "reduced_planck_over_h0", reduced_planck_over_h0
    )
    theta_d_n = _finite("theta_d_n", theta_d_n)
    theta_d2_n = _finite("theta_d2_n", theta_d2_n)
    potential = _finite("potential", potential)
    potential_d_theta = _finite("potential_d_theta", potential_d_theta)
    if e <= 0.0:
        raise ValueError("e=H/H0 must be positive")
    if reduced_planck_over_h0 <= 0.0:
        raise ValueError("reduced_planck_over_h0 must be positive")
    if not isinstance(scalar_channels, tuple) or not isinstance(
        conserved_fluids, tuple
    ):
        raise ValueError("channels and fluids must be tuples")

    e_squared = e * e
    clock_kinetic = 0.5 * e_squared * theta_d_n * theta_d_n
    clock_energy_density = clock_kinetic + potential
    clock_pressure = clock_kinetic - potential
    total_clock_force = math.fsum(
        channel.clock_force for channel in scalar_channels
    )
    clock_equation_residual = (
        e_squared
        * (theta_d2_n + (3.0 + d_log_e_d_n) * theta_d_n)
        + potential_d_theta
        + total_clock_force
    )

    clock_energy_density_d_n = (
        e_squared * theta_d_n * theta_d2_n
        + e_squared * d_log_e_d_n * theta_d_n * theta_d_n
        + potential_d_theta * theta_d_n
    )
    clock_ward_residual = (
        clock_energy_density_d_n
        + 3.0 * (clock_energy_density + clock_pressure)
        + theta_d_n * total_clock_force
    )
    factored_clock_ward = theta_d_n * clock_equation_residual

    scalar_ward_residuals = tuple(
        channel.ward_residual(theta_d_n) for channel in scalar_channels
    )
    fluid_ward_residuals = tuple(
        fluid.ward_residual for fluid in conserved_fluids
    )
    total_ward_residual = math.fsum(
        (
            *scalar_ward_residuals,
            *fluid_ward_residuals,
            clock_ward_residual,
        )
    )

    total_energy_density = math.fsum(
        (
            clock_energy_density,
            *(channel.energy_density for channel in scalar_channels),
            *(fluid.energy_density for fluid in conserved_fluids),
        )
    )
    total_pressure = math.fsum(
        (
            clock_pressure,
            *(channel.pressure for channel in scalar_channels),
            *(fluid.pressure for fluid in conserved_fluids),
        )
    )
    planck_squared = reduced_planck_over_h0**2
    friedmann_constraint_residual = (
        3.0 * planck_squared * e_squared - total_energy_density
    )
    raychaudhuri_residual = (
        e_squared * d_log_e_d_n
        + (total_energy_density + total_pressure) / (2.0 * planck_squared)
    )
    friedmann_constraint_derivative = (
        6.0 * planck_squared * e_squared * d_log_e_d_n
        - (
            clock_energy_density_d_n
            + math.fsum(
                channel.energy_density_d_n for channel in scalar_channels
            )
            + math.fsum(
                fluid.energy_density_d_n for fluid in conserved_fluids
            )
        )
    )
    propagated_derivative = (
        6.0 * planck_squared * raychaudhuri_residual - total_ward_residual
    )

    return BackreactionClosureReceipt(
        scalar_ward_residuals=scalar_ward_residuals,
        clock_equation_residual=clock_equation_residual,
        clock_ward_residual=clock_ward_residual,
        clock_ward_factorization_residual=(
            clock_ward_residual - factored_clock_ward
        ),
        total_ward_residual=total_ward_residual,
        friedmann_constraint_residual=friedmann_constraint_residual,
        raychaudhuri_residual=raychaudhuri_residual,
        friedmann_constraint_derivative=friedmann_constraint_derivative,
        constraint_propagation_identity_residual=(
            friedmann_constraint_derivative - propagated_derivative
        ),
        clock_energy_density=clock_energy_density,
        clock_pressure=clock_pressure,
        total_energy_density=total_energy_density,
        total_pressure=total_pressure,
        total_clock_force=total_clock_force,
    )
