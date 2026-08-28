"""Dimensionless FLRW scalar-mode evolution on a declared CE background.

With c=1 and a0=1, the independent variables and fields are

    N = log(a),  x = H0 * eta,  q = k / H0,  mu = m / H0,
    U = sqrt(H0) * u_phys,  V = dU / dx.

They obey

    dU/dN = V / (a E),
    dV/dN = -Omega^2 U / (a E),

where E=H/H0 and

    Omega^2 = q^2 + a^2 mu^2 + (xi - 1/6) a^2 R/H0^2.

This module supplies a mode-function and canonical-Wronskian audit only.  It
does not implement fourth-order adiabatic subtraction, a renormalized stress
tensor, a Ward identity, backreaction, or a cosmological likelihood.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Protocol


class BackgroundNodeLike(Protocol):
    n: float
    e2: float


class FLRWBackgroundLike(Protocol):
    nodes: tuple[BackgroundNodeLike, ...]

    def at_n(self, n: float) -> BackgroundNodeLike: ...


@dataclass(frozen=True)
class FLRWModeSpec:
    """All frequency inputs expressed as dimensionless H0 ratios."""

    comoving_wavenumber_over_h0: float
    mass_over_h0: Callable[[float], float]
    curvature_coupling: float = 1.0 / 6.0
    initial_n: float | None = None
    final_n: float | None = None
    steps: int = 1200
    curvature_derivative_step_n: float = 1.0e-4
    adiabatic_derivative_step_n: float = 1.0e-4
    max_initial_adiabaticity: float | None = None

    def __post_init__(self) -> None:
        q = self.comoving_wavenumber_over_h0
        if not math.isfinite(q) or q <= 0.0:
            raise ValueError("comoving_wavenumber_over_h0 must be finite and positive")
        if not callable(self.mass_over_h0):
            raise ValueError("mass_over_h0 must be callable")
        if not math.isfinite(self.curvature_coupling):
            raise ValueError("curvature_coupling must be finite")
        for name, value in (("initial_n", self.initial_n), ("final_n", self.final_n)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 20:
            raise ValueError("steps must be an integer of at least 20")
        for name, derivative_step in (
            ("curvature_derivative_step_n", self.curvature_derivative_step_n),
            ("adiabatic_derivative_step_n", self.adiabatic_derivative_step_n),
        ):
            if not math.isfinite(derivative_step) or derivative_step <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        maximum = self.max_initial_adiabaticity
        if maximum is not None and (not math.isfinite(maximum) or maximum <= 0.0):
            raise ValueError(
                "max_initial_adiabaticity must be finite and positive when provided"
            )


@dataclass(frozen=True)
class FLRWAdiabaticInitialState:
    n: float
    omega: float
    u: complex
    du_dx: complex
    adiabaticity: float
    wronskian_residual: float
    amplitude_residual: float


@dataclass(frozen=True)
class FLRWModeNode:
    n: float
    x: float
    omega_squared: float
    u: complex
    du_dx: complex
    wronskian_residual: float


@dataclass(frozen=True)
class FLRWModeSolution:
    spec: FLRWModeSpec
    nodes: tuple[FLRWModeNode, ...]
    background_window: tuple[float, float]
    initial_adiabaticity: float
    initial_amplitude_residual: float
    max_wronskian_residual: float
    status: str = "MODE_ONLY_NO_RENORMALIZED_STRESS_OR_BACKREACTION"
    dimensionless_contract: str = (
        "N=log(a); x=H0*eta; q=k/H0; mu=m/H0; U=sqrt(H0)*u_phys"
    )


def _background_bounds(background: FLRWBackgroundLike) -> tuple[float, float]:
    nodes = tuple(background.nodes)
    if len(nodes) < 2:
        raise ValueError("background must contain at least two ordered nodes")
    n_values = tuple(float(node.n) for node in nodes)
    if not all(math.isfinite(value) for value in n_values):
        raise ValueError("background e-fold nodes must be finite")
    if any(right <= left for left, right in zip(n_values, n_values[1:])):
        raise ValueError("background e-fold nodes must be strictly increasing")
    return n_values[0], n_values[-1]


def _resolved_interval(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
) -> tuple[float, float, tuple[float, float]]:
    bounds = _background_bounds(background)
    initial_n = bounds[0] if spec.initial_n is None else spec.initial_n
    final_n = bounds[1] if spec.final_n is None else spec.final_n
    assert initial_n is not None and final_n is not None
    if initial_n < bounds[0] or final_n > bounds[1]:
        raise ValueError("mode interval lies outside the solved background window")
    if final_n <= initial_n:
        raise ValueError("final_n must be greater than initial_n")
    return initial_n, final_n, bounds


def _e2_at_n(background: FLRWBackgroundLike, n: float) -> float:
    e2 = float(background.at_n(n).e2)
    if not math.isfinite(e2) or e2 <= 0.0:
        raise ValueError("background e2 must be finite and positive")
    return e2


def _mass_ratio_at_n(spec: FLRWModeSpec, n: float) -> float:
    mass_ratio = float(spec.mass_over_h0(n))
    if not math.isfinite(mass_ratio) or mass_ratio < 0.0:
        raise ValueError("mass_over_h0(n) must be finite and non-negative")
    return mass_ratio


def _dimensionless_ricci_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    bounds: tuple[float, float],
) -> float:
    step = spec.curvature_derivative_step_n
    if n - step < bounds[0] or n + step > bounds[1]:
        raise ValueError("curvature derivative stencil leaves the background window")
    e2 = _e2_at_n(background, n)
    log_e2_left = math.log(_e2_at_n(background, n - step))
    log_e2_right = math.log(_e2_at_n(background, n + step))
    d_log_h_d_n = (log_e2_right - log_e2_left) / (4.0 * step)
    return 6.0 * e2 * (2.0 + d_log_h_d_n)


def _omega_squared_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    bounds: tuple[float, float],
) -> float:
    if n < bounds[0] or n > bounds[1]:
        raise ValueError("requested e-fold is outside the solved background window")
    scale_factor = math.exp(n)
    mass_ratio = _mass_ratio_at_n(spec, n)
    curvature_coefficient = spec.curvature_coupling - 1.0 / 6.0
    curvature_term = 0.0
    if curvature_coefficient != 0.0:
        curvature_term = (
            curvature_coefficient
            * scale_factor**2
            * _dimensionless_ricci_at_n(background, spec, n, bounds)
        )
    omega_squared = (
        spec.comoving_wavenumber_over_h0**2
        + scale_factor**2 * mass_ratio**2
        + curvature_term
    )
    if not math.isfinite(omega_squared) or omega_squared <= 0.0:
        raise ValueError("dimensionless omega_squared must be finite and positive")
    return omega_squared


def omega_squared_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
) -> float:
    """Return dimensionless Omega^2, failing closed outside its domain."""

    return _omega_squared_at_n(background, spec, n, _background_bounds(background))


def _wronskian_residual(u: complex, du_dx: complex) -> float:
    wronskian = u * du_dx.conjugate() - u.conjugate() * du_dx
    return abs(wronskian - 1.0j)


def _omega_derivative_at_n(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    difference_step: float,
    bounds: tuple[float, float],
) -> float:
    def omega(at_n: float) -> float:
        return math.sqrt(_omega_squared_at_n(background, spec, at_n, bounds))

    if n - difference_step >= bounds[0] and n + difference_step <= bounds[1]:
        return (omega(n + difference_step) - omega(n - difference_step)) / (
            2.0 * difference_step
        )
    if n + 2.0 * difference_step <= bounds[1]:
        return (
            -3.0 * omega(n)
            + 4.0 * omega(n + difference_step)
            - omega(n + 2.0 * difference_step)
        ) / (2.0 * difference_step)
    if n - 2.0 * difference_step >= bounds[0]:
        return (
            3.0 * omega(n)
            - 4.0 * omega(n - difference_step)
            + omega(n - 2.0 * difference_step)
        ) / (2.0 * difference_step)
    raise ValueError("omega derivative stencil leaves the background window")


def _adiabatic_initial_mode(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float,
    bounds: tuple[float, float],
) -> FLRWAdiabaticInitialState:
    omega = math.sqrt(_omega_squared_at_n(background, spec, n, bounds))
    d_omega_d_n = _omega_derivative_at_n(
        background,
        spec,
        n,
        spec.adiabatic_derivative_step_n,
        bounds,
    )
    scale_factor_times_e = math.exp(n) * math.sqrt(_e2_at_n(background, n))
    d_omega_dx = scale_factor_times_e * d_omega_d_n
    u = complex(1.0 / math.sqrt(2.0 * omega))
    logarithmic_amplitude_rate = -d_omega_dx / (2.0 * omega)
    du_dx = complex(logarithmic_amplitude_rate, -omega) * u
    adiabaticity = abs(d_omega_dx) / omega**2
    amplitude_residual = abs(2.0 * omega * abs(u) ** 2 - 1.0)
    wronskian_residual = _wronskian_residual(u, du_dx)
    maximum = spec.max_initial_adiabaticity
    if maximum is not None and adiabaticity > maximum:
        raise ValueError(
            "initial adiabaticity exceeds max_initial_adiabaticity"
        )
    return FLRWAdiabaticInitialState(
        n=n,
        omega=omega,
        u=u,
        du_dx=du_dx,
        adiabaticity=adiabaticity,
        wronskian_residual=wronskian_residual,
        amplitude_residual=amplitude_residual,
    )


def adiabatic_initial_mode(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    n: float | None = None,
) -> FLRWAdiabaticInitialState:
    """Construct the zeroth/first-derivative adiabatic canonical state."""

    initial_n, final_n, bounds = _resolved_interval(background, spec)
    target_n = initial_n if n is None else n
    if target_n < initial_n or target_n > final_n:
        raise ValueError("initial-state e-fold lies outside the mode interval")
    return _adiabatic_initial_mode(
        background,
        spec,
        target_n,
        bounds,
    )


def _mode_rhs(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    bounds: tuple[float, float],
    n: float,
    state: tuple[float, complex, complex],
) -> tuple[float, complex, complex]:
    edge_tolerance = (
        16.0
        * math.ulp(1.0)
        * max(1.0, abs(bounds[0]), abs(bounds[1]))
    )
    if bounds[0] - edge_tolerance <= n < bounds[0]:
        n = bounds[0]
    elif bounds[1] < n <= bounds[1] + edge_tolerance:
        n = bounds[1]
    _, u, du_dx = state
    inverse_a_e = 1.0 / (math.exp(n) * math.sqrt(_e2_at_n(background, n)))
    omega_squared = _omega_squared_at_n(background, spec, n, bounds)
    return (
        inverse_a_e,
        du_dx * inverse_a_e,
        -omega_squared * u * inverse_a_e,
    )


def _rk4_mode_step(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
    bounds: tuple[float, float],
    n: float,
    state: tuple[float, complex, complex],
    step: float,
) -> tuple[float, complex, complex]:
    k1 = _mode_rhs(background, spec, bounds, n, state)
    second = (
        state[0] + 0.5 * step * k1[0],
        state[1] + 0.5 * step * k1[1],
        state[2] + 0.5 * step * k1[2],
    )
    k2 = _mode_rhs(background, spec, bounds, n + 0.5 * step, second)
    third = (
        state[0] + 0.5 * step * k2[0],
        state[1] + 0.5 * step * k2[1],
        state[2] + 0.5 * step * k2[2],
    )
    k3 = _mode_rhs(background, spec, bounds, n + 0.5 * step, third)
    fourth = (
        state[0] + step * k3[0],
        state[1] + step * k3[1],
        state[2] + step * k3[2],
    )
    k4 = _mode_rhs(background, spec, bounds, n + step, fourth)
    return (
        state[0] + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + step * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )


def solve_flrw_mode(
    background: FLRWBackgroundLike,
    spec: FLRWModeSpec,
) -> FLRWModeSolution:
    """Solve one canonical scalar mode and preserve its Wronskian receipt."""

    initial_n, final_n, bounds = _resolved_interval(background, spec)
    step = (final_n - initial_n) / spec.steps
    initial = _adiabatic_initial_mode(
        background,
        spec,
        initial_n,
        bounds,
    )
    state = (0.0, initial.u, initial.du_dx)
    nodes: list[FLRWModeNode] = [
        FLRWModeNode(
            n=initial_n,
            x=state[0],
            omega_squared=initial.omega**2,
            u=state[1],
            du_dx=state[2],
            wronskian_residual=initial.wronskian_residual,
        )
    ]
    for index in range(spec.steps):
        n = initial_n + index * step
        state = _rk4_mode_step(background, spec, bounds, n, state, step)
        next_n = initial_n + (index + 1) * step
        nodes.append(
            FLRWModeNode(
                n=next_n,
                x=state[0],
                omega_squared=_omega_squared_at_n(
                    background,
                    spec,
                    next_n,
                    bounds,
                ),
                u=state[1],
                du_dx=state[2],
                wronskian_residual=_wronskian_residual(state[1], state[2]),
            )
        )
    max_wronskian_residual = max(node.wronskian_residual for node in nodes)
    return FLRWModeSolution(
        spec=spec,
        nodes=tuple(nodes),
        background_window=bounds,
        initial_adiabaticity=initial.adiabaticity,
        initial_amplitude_residual=initial.amplitude_residual,
        max_wronskian_residual=max_wronskian_residual,
    )
