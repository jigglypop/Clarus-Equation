"""Single-clock perturbation and EFT-cutoff gate for the kinetic dark sector.

This promotes the reproducible, low-data part of the former R2 diagnostic into
the main examples tree.  It tests the clock+Einstein subsystem only.  The
fixed-background ``pi`` growth below is a diagnostic of the clock coordinate,
not the observable matter-growth function or an ``f sigma_8`` likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from examples.physics.kinetic_dark_sector_gate import (
    BackgroundSolution,
    KineticClockConfig,
    _densities,
    _rhs,
    solve_background,
)


H0_KM_S_MPC = 67.4
MPC_IN_M = 3.0856775814913673e22
HBAR_EV_S = 6.582119569e-16
HBAR_C_EV_M = 1.973269804e-7
REDUCED_MPL_EV = 2.435e27
H0_EV = H0_KM_S_MPC * 1000.0 / MPC_IN_M * HBAR_EV_S
RHO_CRIT0_EV4 = 3.0 * REDUCED_MPL_EV**2 * H0_EV**2
MPC_INV_EV = HBAR_C_EV_M / MPC_IN_M


@dataclass(frozen=True)
class PerturbationNode:
    n: float
    e2: float
    friction: float
    tachyon_ratio: float
    cs2: float
    q_s_over_mpl2: float
    pump_slope: float
    zeta_decay_slope: float
    energy_cutoff_ev: float
    wavenumber_cutoff_ev: float


@dataclass(frozen=True)
class SingleClockGate:
    gamma: float
    min_friction: float
    max_tachyon_ratio: float
    fixed_coordinate_growth_minus_one: float
    max_log_growth_bound: float
    min_pump_slope: float
    min_zeta_decay_slope: float
    min_energy_cutoff_over_h: float
    min_wavenumber_cutoff_over_k_1mpc: float
    status: str = "PASS_SINGLE_CLOCK_ONLY"
    failed_gates: tuple[str, ...] = ()
    matter_growth_likelihood: str = "NOT_IMPLEMENTED_COUPLED_EQUATIONS_REQUIRED"


@dataclass(frozen=True)
class QuasiStaticGrowthDiagnostic:
    redshift: float
    predicted_fsigma8: float
    observed_fsigma8: float
    observed_sigma: float
    pull: float
    sigma8_0: float
    closure: str = "KINETIC_CLUSTERS_VACUUM_SMOOTH_GR_SUBHORIZON"
    role: str = "APPROXIMATE_DIAGNOSTIC_NOT_FULL_COUPLED_LIKELIHOOD"


@dataclass(frozen=True)
class KappaSensitivityRow:
    kappa: float
    min_cs2: float
    min_friction: float
    max_log_growth_bound: float
    min_energy_cutoff_over_h: float
    status: str
    failed_gates: tuple[str, ...]


def _nodes(solution: BackgroundSolution) -> tuple[PerturbationNode, ...]:
    config = solution.config
    rho_inf_ev4 = solution.amplitude * RHO_CRIT0_EV4
    lambda3_ev = 2.0 * (config.kappa * rho_inf_ev4) ** 0.25
    result: list[PerturbationNode] = []
    for node in solution.nodes:
        data = _densities(node.n, node.tau, node.u, config, solution.amplitude)
        rho_b, rho_r, _, rho_k, p_k, _ = data
        _, u_prime = _rhs(
            node.n, (node.tau, node.u), config, solution.amplitude
        )
        delta = node.u / config.kappa
        delta_prime = u_prime / config.kappa
        h_prime_over_h = (
            -3.0 * rho_b - 4.0 * rho_r - 3.0 * (rho_k + p_k)
        ) / (2.0 * node.e2)
        kinetic_prime = 3.0 * delta_prime / (2.0 + 3.0 * delta)
        friction = 3.0 + h_prime_over_h + kinetic_prime
        tachyon_ratio = (
            config.gamma**2
            * config.x_star
            * math.exp(-config.gamma * node.tau)
            / (config.kappa * (2.0 + 3.0 * delta) * node.e2)
        )
        pump = (
            3.0
            + delta_prime / (1.0 + delta)
            + 3.0 * delta_prime / (2.0 + 3.0 * delta)
            - 2.0 * h_prime_over_h
        )
        energy_cutoff = lambda3_ev * node.cs2 ** (7.0 / 8.0)
        wavenumber_cutoff = lambda3_ev * node.cs2 ** (3.0 / 8.0)
        result.append(
            PerturbationNode(
                n=node.n,
                e2=node.e2,
                friction=friction,
                tachyon_ratio=tachyon_ratio,
                cs2=node.cs2,
                q_s_over_mpl2=node.q_s_over_mpl2,
                pump_slope=pump,
                zeta_decay_slope=pump + h_prime_over_h,
                energy_cutoff_ev=energy_cutoff,
                wavenumber_cutoff_ev=wavenumber_cutoff,
            )
        )
    return tuple(result)


def _fixed_coordinate_growth(nodes: tuple[PerturbationNode, ...]) -> float:
    y = 0.0
    velocity = 0.0
    for left, right in zip(nodes, nodes[1:]):
        step = right.n - left.n
        friction = 0.5 * (left.friction + right.friction)
        ratio = 0.5 * (left.tachyon_ratio + right.tachyon_ratio)

        def derivative(yy: float, vv: float) -> tuple[float, float]:
            return vv, -friction * vv + ratio * (1.0 + yy)

        k1 = derivative(y, velocity)
        k2 = derivative(y + step * k1[0] / 2.0, velocity + step * k1[1] / 2.0)
        k3 = derivative(y + step * k2[0] / 2.0, velocity + step * k2[1] / 2.0)
        k4 = derivative(y + step * k3[0], velocity + step * k3[1])
        y += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        velocity += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    return y


def evaluate_single_clock_gate(
    solution: BackgroundSolution | None = None,
) -> SingleClockGate:
    selected = solution or solve_background(KineticClockConfig())
    nodes = _nodes(selected)
    min_friction = min(node.friction for node in nodes)
    max_ratio = max(node.tachyon_ratio for node in nodes)
    interval = nodes[-1].n - nodes[0].n
    positive_root = 2.0 * max_ratio / (
        math.sqrt(min_friction**2 + 4.0 * max_ratio) + min_friction
    )
    bound = positive_root * interval
    min_energy_over_h = min(
        node.energy_cutoff_ev / (H0_EV * math.sqrt(node.e2)) for node in nodes
    )
    min_wavenumber_over_k = min(
        node.wavenumber_cutoff_ev / (MPC_INV_EV * math.exp(-node.n))
        for node in nodes
    )
    gate_values = {
        "positive_friction": min_friction > 0.0,
        "sub_hubble_tachyon": max_ratio < 1.0,
        "sub_order_one_growth_bound": bound < 1.0,
        "positive_pump": min(node.pump_slope for node in nodes) > 0.0,
        "decaying_zeta_integrand": min(node.zeta_decay_slope for node in nodes) > 0.0,
        "energy_cutoff_above_h": min_energy_over_h > 1.0,
        "momentum_cutoff_above_1mpc": min_wavenumber_over_k > 1.0,
    }
    failed = tuple(name for name, passed in gate_values.items() if not passed)
    return SingleClockGate(
        gamma=selected.config.gamma,
        min_friction=min_friction,
        max_tachyon_ratio=max_ratio,
        fixed_coordinate_growth_minus_one=_fixed_coordinate_growth(nodes),
        max_log_growth_bound=bound,
        min_pump_slope=min(node.pump_slope for node in nodes),
        min_zeta_decay_slope=min(node.zeta_decay_slope for node in nodes),
        min_energy_cutoff_over_h=min_energy_over_h,
        min_wavenumber_cutoff_over_k_1mpc=min_wavenumber_over_k,
        status=("PASS_SINGLE_CLOCK_ONLY" if not failed else "FAIL_SINGLE_CLOCK_GATE"),
        failed_gates=failed,
    )


def quasi_static_growth_diagnostic(
    solution: BackgroundSolution | None = None,
    *,
    redshift: float = 0.07,
    observed_fsigma8: float = 0.4497,
    observed_sigma: float = 0.0548,
    sigma8_0: float = 0.811,
) -> QuasiStaticGrowthDiagnostic:
    """Solve a declared subhorizon closure and compare one compact datum.

    The kinetic inventory and baryons source the Poisson term, the saturated
    readout is smooth, GR is retained, and radiation affects only H(a).  This
    is the strongest low-data diagnostic available before deriving the full
    multi-component perturbation system.  ``sigma8_0`` remains external.
    """

    if redshift < 0.0 or observed_sigma <= 0.0 or sigma8_0 <= 0.0:
        raise ValueError("growth diagnostic inputs are outside their domain")
    selected = solution or solve_background(KineticClockConfig())
    target_n = -math.log1p(redshift)
    if target_n < selected.nodes[0].n:
        raise ValueError("growth redshift is outside the solved window")

    # Growing-mode matter-era seed at a>=0.01.  This deliberately avoids
    # pretending that the closure supplies an adiabatic radiation-era transfer.
    growth_nodes = tuple(node for node in selected.nodes if node.n >= math.log(0.01))
    if len(growth_nodes) < 10:
        raise ValueError("background grid is too sparse for the growth closure")
    d_value = math.exp(growth_nodes[0].n)
    velocity = d_value
    history: list[tuple[float, float, float]] = [
        (growth_nodes[0].n, d_value, velocity)
    ]

    def coefficients(n: float) -> tuple[float, float]:
        node = selected.at_n(n)
        rho_b, rho_r, _, rho_k, p_k, _ = _densities(
            n, node.tau, node.u, selected.config, selected.amplitude
        )
        h_prime_over_h = (
            -3.0 * rho_b - 4.0 * rho_r - 3.0 * (rho_k + p_k)
        ) / (2.0 * node.e2)
        omega_cluster = (rho_b + rho_k) / node.e2
        return 2.0 + h_prime_over_h, 1.5 * omega_cluster

    def derivative(n: float, d: float, v: float) -> tuple[float, float]:
        drag, source = coefficients(n)
        return v, -drag * v + source * d

    for left, right in zip(growth_nodes, growth_nodes[1:]):
        step = right.n - left.n
        n = left.n
        k1 = derivative(n, d_value, velocity)
        k2 = derivative(
            n + step / 2.0,
            d_value + step * k1[0] / 2.0,
            velocity + step * k1[1] / 2.0,
        )
        k3 = derivative(
            n + step / 2.0,
            d_value + step * k2[0] / 2.0,
            velocity + step * k2[1] / 2.0,
        )
        k4 = derivative(
            n + step,
            d_value + step * k3[0],
            velocity + step * k3[1],
        )
        d_value += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        velocity += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        history.append((right.n, d_value, velocity))

    normalization = history[-1][1]
    for left, right in zip(history, history[1:]):
        if left[0] <= target_n <= right[0]:
            weight = (target_n - left[0]) / (right[0] - left[0])
            d_target = left[1] + weight * (right[1] - left[1])
            v_target = left[2] + weight * (right[2] - left[2])
            break
    else:
        d_target, v_target = history[-1][1], history[-1][2]
    # f*sigma8 = (D'/D) * sigma8_0*(D/D0) = sigma8_0*D'/D0.
    prediction = sigma8_0 * v_target / normalization
    return QuasiStaticGrowthDiagnostic(
        redshift=redshift,
        predicted_fsigma8=prediction,
        observed_fsigma8=observed_fsigma8,
        observed_sigma=observed_sigma,
        pull=(prediction - observed_fsigma8) / observed_sigma,
        sigma8_0=sigma8_0,
    )


def scan_kappa_sensitivity(
    kappa_values: tuple[float, ...] = (1.0e10, 3.0e11, 1.0e12, 1.0e14, 1.0e17, 1.0e20),
    *,
    gamma: float = 10.0,
    steps: int = 1200,
) -> tuple[KappaSensitivityRow, ...]:
    """Expose which role the otherwise external stiffness scale plays."""

    if not kappa_values or any(value <= 0.0 for value in kappa_values):
        raise ValueError("kappa scan values must be non-empty and positive")
    rows: list[KappaSensitivityRow] = []
    for kappa in kappa_values:
        solution = solve_background(
            KineticClockConfig(gamma=gamma, kappa=kappa, steps=steps)
        )
        gate = evaluate_single_clock_gate(solution)
        rows.append(
            KappaSensitivityRow(
                kappa=kappa,
                min_cs2=solution.min_cs2,
                min_friction=gate.min_friction,
                max_log_growth_bound=gate.max_log_growth_bound,
                min_energy_cutoff_over_h=gate.min_energy_cutoff_over_h,
                status=gate.status,
                failed_gates=gate.failed_gates,
            )
        )
    return tuple(rows)


def main() -> int:
    gate = evaluate_single_clock_gate()
    for name, value in gate.__dict__.items():
        print(name, value)
    growth = quasi_static_growth_diagnostic()
    for name, value in growth.__dict__.items():
        print(f"growth_{name}", value)
    for row in scan_kappa_sensitivity():
        print(
            "kappa_scan",
            row.kappa,
            row.min_cs2,
            row.min_friction,
            row.max_log_growth_bound,
            row.min_energy_cutoff_over_h,
            row.status,
            ",".join(row.failed_gates) or "none",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
