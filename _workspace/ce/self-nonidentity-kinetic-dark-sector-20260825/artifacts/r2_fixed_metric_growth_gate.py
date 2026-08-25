"""R2: 포화 readout의 관측구간 섭동 및 EFT cutoff 진단.

현재 clock EFT의 포화 potential은 V''<0이므로 fixed-metric scalar mode에
음의 유효 질량제곱을 준다. 이 스크립트는 그 부호를 감추지 않고 기존
homogeneous background를 독립적으로 다시 풀어 다음 양을 계산한다.

    r(N) = |m_eff^2| / H^2
    S_tach = integral r(N) dN / 3

또한 정확한 fixed-background 마찰

    f_N = 3 + H'/H + A'/A

를 사용한 k=0 선형식

    pi'' + f_N pi' - r pi = 0

의 작은 성장분을 직접 적분한다. 이어서 Einstein 중력과 clock만 남긴
single-clock ADM 부분계의 Q_s, c_s^2, super-Hubble pump를 평가하고, 작은
sound speed에서의 보수적 strong-coupling power counting을 계산한다.

고정계량 pi는 gauge-dependent 진단이다. single-clock 결과도 baryon,
radiation, Gaussian reservoir의 perturbation을 함께 풀지 않으므로 full
cosmological stability 판정이 아니다. cutoff 계산은 delta -> 0 decoupling
연산자의 tree-level power counting이며 UV completion의 존재 증명이 아니다.
에너지 cutoff Lambda_E와 물리 파수 cutoff q_sc를 구분한다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


OMEGA_B0 = 0.049
OMEGA_R0 = 9.0e-5
OMEGA_V0 = 0.687
OMEGA_K0 = 1.0 - OMEGA_B0 - OMEGA_R0 - OMEGA_V0
KAPPA = 1.0e17
X_STAR = 0.5
N_INITIAL = math.log(1.0e-4)
GAMMAS = (3.5, 5.0, 10.0, 20.0, 30.0)
H0_KM_S_MPC = 67.4
MPC_IN_M = 3.0856775814913673e22
HBAR_EV_S = 6.582119569e-16
HBAR_C_EV_M = 1.973269804e-7
REDUCED_MPL_EV = 2.435e27
H0_EV = H0_KM_S_MPC * 1000.0 / MPC_IN_M * HBAR_EV_S
RHO_CRIT0_EV4 = 3.0 * REDUCED_MPL_EV**2 * H0_EV**2
MPC_INV_EV = HBAR_C_EV_M / MPC_IN_M


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def positive_u_from_density(target_over_amplitude: float) -> float:
    """2u + 3u^2/(2 kappa) = y의 cancellation-safe 양의 근."""

    y = target_over_amplitude
    return y / (math.sqrt(1.0 + 1.5 * y / KAPPA) + 1.0)


def amplitude_from_b(b: float) -> float:
    denominator = -math.expm1(-b)
    require(denominator > 0.0, "1-exp(-b)가 양수가 아니다")
    return OMEGA_V0 / denominator


def densities(n: float, tau: float, u: float, gamma: float, amplitude: float) -> dict[str, float]:
    rho_b = OMEGA_B0 * math.exp(-3.0 * n)
    rho_r = OMEGA_R0 * math.exp(-4.0 * n)
    rho_v = amplitude * (1.0 - math.exp(-gamma * tau))
    rho_k = amplitude * (2.0 * u + 1.5 * u * u / KAPPA)
    p_k = amplitude * u * u / (2.0 * KAPPA)
    e2 = rho_b + rho_r + rho_v + rho_k
    if not math.isfinite(e2) or e2 <= 0.0:
        raise ArithmeticError("shooting path에서 E^2<=0")
    return {
        "rho_b": rho_b,
        "rho_r": rho_r,
        "rho_v": rho_v,
        "rho_k": rho_k,
        "p_k": p_k,
        "e2": e2,
    }


def background_rhs(
    n: float,
    state: Sequence[float],
    gamma: float,
    amplitude: float,
) -> tuple[float, float]:
    tau, u = state
    require(u > -0.99 * KAPPA, "sqrt 정의역 밖의 u")
    data = densities(n, tau, u, gamma, amplitude)
    e = math.sqrt(data["e2"])
    root = math.sqrt(1.0 + u / KAPPA)
    current_shape = u * root
    current_derivative = (1.0 + 1.5 * u / KAPPA) / root
    tau_prime = root / e
    u_prime = (
        -3.0 * current_shape
        - gamma * math.exp(-gamma * tau) / (2.0 * e)
    ) / current_derivative
    return tau_prime, u_prime


def rk4_step(
    rhs: Callable[[float, Sequence[float]], Sequence[float]],
    n: float,
    state: Sequence[float],
    step: float,
) -> list[float]:
    k1 = list(rhs(n, state))
    y2 = [y + 0.5 * step * k for y, k in zip(state, k1)]
    k2 = list(rhs(n + 0.5 * step, y2))
    y3 = [y + 0.5 * step * k for y, k in zip(state, k2)]
    k3 = list(rhs(n + 0.5 * step, y3))
    y4 = [y + step * k for y, k in zip(state, k3)]
    k4 = list(rhs(n + step, y4))
    return [
        y + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for y, a, b, c, d in zip(state, k1, k2, k3, k4)
    ]


def integrate_backward(
    gamma: float,
    b: float,
    steps: int,
    keep_grid: bool,
) -> tuple[float, float, list[tuple[float, float, float]]]:
    amplitude = amplitude_from_b(b)
    tau0 = b / gamma
    u0 = positive_u_from_density(OMEGA_K0 / amplitude)
    state = [tau0, u0]
    n = 0.0
    step = N_INITIAL / steps
    grid: list[tuple[float, float, float]] = [(n, state[0], state[1])] if keep_grid else []

    def rhs(x: float, y: Sequence[float]) -> Sequence[float]:
        return background_rhs(x, y, gamma, amplitude)

    for _ in range(steps):
        state = rk4_step(rhs, n, state, step)
        n += step
        if keep_grid:
            grid.append((n, state[0], state[1]))
    return state[0], amplitude, grid


def shoot_b(gamma: float, steps: int = 1800) -> float:
    """tau(N_initial)=0을 만족하는 b=gamma*tau_0를 이분법으로 찾는다."""

    low = 0.5 * gamma
    high = 1.2 * gamma

    def endpoint(b: float) -> float:
        try:
            value, _, _ = integrate_backward(gamma, b, steps, keep_grid=False)
            return value
        except (ArithmeticError, AssertionError, OverflowError):
            return -math.inf

    low_value = endpoint(low)
    high_value = endpoint(high)
    while high_value <= 0.0:
        high *= 1.25
        high_value = endpoint(high)
        require(high < 10.0 * gamma, "shooting 상단 bracket을 찾지 못했다")
    require(low_value < 0.0, "shooting 하단이 tau_i<0을 만들지 않는다")

    for _ in range(52):
        middle = 0.5 * (low + high)
        value = endpoint(middle)
        if value > 0.0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class Node:
    n: float
    tau: float
    u: float
    e2: float
    friction: float
    tachyon_ratio: float
    cs2: float
    q_s_over_mpl2: float
    pump_slope: float
    zeta_integrand_decay_slope: float


def build_nodes(gamma: float, steps: int = 6000) -> tuple[float, float, list[Node]]:
    b = shoot_b(gamma)
    _, amplitude, descending = integrate_backward(gamma, b, steps, keep_grid=True)
    ascending = list(reversed(descending))
    nodes: list[Node] = []

    for n, tau, u in ascending:
        data = densities(n, tau, u, gamma, amplitude)
        _, u_prime = background_rhs(n, (tau, u), gamma, amplitude)
        h_prime_over_h = (
            -3.0 * data["rho_b"]
            -4.0 * data["rho_r"]
            -3.0 * (data["rho_k"] + data["p_k"])
        ) / (2.0 * data["e2"])
        kinetic_prime_over_kinetic = (
            3.0 * u_prime / KAPPA
        ) / (2.0 + 3.0 * u / KAPPA)
        friction = 3.0 + h_prime_over_h + kinetic_prime_over_kinetic
        tachyon_ratio = (
            gamma
            * gamma
            * X_STAR
            * math.exp(-gamma * tau)
            / (KAPPA * (2.0 + 3.0 * u / KAPPA) * data["e2"])
        )
        delta = u / KAPPA
        cs2 = delta / (2.0 + 3.0 * delta)
        delta_prime = u_prime / KAPPA
        q_s_over_mpl2 = (
            3.0
            * KAPPA
            * amplitude
            * (1.0 + delta)
            * (2.0 + 3.0 * delta)
            / data["e2"]
        )
        pump_slope = (
            3.0
            + delta_prime / (1.0 + delta)
            + 3.0 * delta_prime / (2.0 + 3.0 * delta)
            - 2.0 * h_prime_over_h
        )
        # d ln[H a^3 Q_s]/dN. 양의 하한은 zeta_2 적분함수의 감쇠를 뜻한다.
        zeta_integrand_decay_slope = pump_slope + h_prime_over_h
        nodes.append(
            Node(
                n=n,
                tau=tau,
                u=u,
                e2=data["e2"],
                friction=friction,
                tachyon_ratio=tachyon_ratio,
                cs2=cs2,
                q_s_over_mpl2=q_s_over_mpl2,
                pump_slope=pump_slope,
                zeta_integrand_decay_slope=zeta_integrand_decay_slope,
            )
        )
    return b, amplitude, nodes


def trapezoid(nodes: Sequence[Node], value: Callable[[Node], float]) -> float:
    return sum(
        0.5
        * (value(left) + value(right))
        * (right.n - left.n)
        for left, right in zip(nodes, nodes[1:])
    )


def integrate_small_growth(nodes: Sequence[Node]) -> tuple[float, float]:
    """pi=1+y로 두어 1e-20 수준 성장도 cancellation 없이 적분한다."""

    y = 0.0
    velocity = 0.0
    maximum = 0.0
    for left, right in zip(nodes, nodes[1:]):
        step = right.n - left.n
        f0, f1 = left.friction, right.friction
        r0, r1 = left.tachyon_ratio, right.tachyon_ratio
        fm, rm = 0.5 * (f0 + f1), 0.5 * (r0 + r1)

        def derivative(state: tuple[float, float], friction: float, ratio: float) -> tuple[float, float]:
            yy, vv = state
            return vv, -friction * vv + ratio * (1.0 + yy)

        k1 = derivative((y, velocity), f0, r0)
        k2_state = (y + 0.5 * step * k1[0], velocity + 0.5 * step * k1[1])
        k2 = derivative(k2_state, fm, rm)
        k3_state = (y + 0.5 * step * k2[0], velocity + 0.5 * step * k2[1])
        k3 = derivative(k3_state, fm, rm)
        k4_state = (y + step * k3[0], velocity + step * k3[1])
        k4 = derivative(k4_state, f1, r1)
        y += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        velocity += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        maximum = max(maximum, y)
    return y, maximum


def frozen_rows() -> dict[float, dict[str, float]]:
    path = Path(__file__).with_name("numerical-results.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        float(row["gamma"]): row
        for row in payload["rows"]
        if row.get("stable") and float(row["gamma"]) in GAMMAS
    }


def diagnose(
    gamma: float,
    frozen: dict[float, dict[str, float]],
    steps: int = 6000,
) -> dict[str, float]:
    b, amplitude, nodes = build_nodes(gamma, steps=steps)
    require(all(node.friction > 0.0 for node in nodes), "fixed-background e-fold 마찰이 음수가 됐다")
    require(all(node.tachyon_ratio >= 0.0 for node in nodes), "tachyon ratio 부호 오류")
    require(all(node.cs2 >= 0.0 for node in nodes), "delta>=0 가지에서 cs2<0")
    require(all(node.q_s_over_mpl2 > 0.0 for node in nodes), "single-clock Q_s가 양수가 아니다")
    require(all(node.pump_slope > 0.0 for node in nodes), "a^3 Q_s pump가 감소한다")
    require(
        all(node.zeta_integrand_decay_slope > 0.0 for node in nodes),
        "관측 배경 구간에서 zeta_2 적분함수가 증가한다",
    )

    r_max = max(node.tachyon_ratio for node in nodes)
    f_min = min(node.friction for node in nodes)
    slow_roll_exponent = trapezoid(nodes, lambda node: node.tachyon_ratio / 3.0)
    def positive_root(node: Node) -> float:
        # (-f+sqrt(f^2+4r))/2를 그대로 계산하면 r<<f^2에서 상쇄된다.
        return 2.0 * node.tachyon_ratio / (
            math.sqrt(node.friction * node.friction + 4.0 * node.tachyon_ratio)
            + node.friction
        )

    exact_friction_exponent = trapezoid(nodes, positive_root)
    growth_end, growth_max = integrate_small_growth(nodes)
    constant_root = 2.0 * r_max / (
        math.sqrt(f_min * f_min + 4.0 * r_max) + f_min
    )
    constant_coefficient_bound = constant_root * (nodes[-1].n - nodes[0].n)

    # X_*=1/2, delta->0에서 canonically normalized phi=sqrt(A) pi에 대해
    # L3=[dot(phi)^3-dot(phi)(grad phi)^2]/[4 sqrt(kappa rho_inf)]이다.
    # 따라서 bare derivative scale Lambda_3=2(kappa rho_inf)^(1/4).
    # 작은 c_s에서 공간 cubic의 energy cutoff는
    # Lambda_E ~= Lambda_3 c_s^(7/4), physical-wavenumber cutoff는
    # q_sc=Lambda_E/c_s ~= Lambda_3 c_s^(3/4)이다. 수치 계수는 EFT convention에
    # 의존하므로 여기서는 관측 energy/momentum과의 hierarchy만 gate한다.
    rho_inf_ev4 = amplitude * RHO_CRIT0_EV4
    lambda3_ev = 2.0 * (KAPPA * rho_inf_ev4) ** 0.25
    lambda4_ev = (32.0 * KAPPA * rho_inf_ev4) ** 0.25

    def energy_cutoff(node: Node) -> float:
        return lambda3_ev * node.cs2 ** (7.0 / 8.0)

    def physical_wavenumber_cutoff(node: Node) -> float:
        return lambda3_ev * node.cs2 ** (3.0 / 8.0)

    def kinetic_coefficient_ev4(node: Node) -> float:
        delta = node.u / KAPPA
        return KAPPA * rho_inf_ev4 / X_STAR * (2.0 + 3.0 * delta)

    energy_cutoff_values = [energy_cutoff(node) for node in nodes]
    wavenumber_cutoff_values = [physical_wavenumber_cutoff(node) for node in nodes]
    require(
        all(
            math.isclose(energy, math.sqrt(node.cs2) * momentum, rel_tol=2.0e-15)
            for energy, momentum, node in zip(
                energy_cutoff_values, wavenumber_cutoff_values, nodes
            )
        ),
        "energy cutoff와 physical-wavenumber cutoff의 c_s 변환이 깨졌다",
    )
    energy_cutoff_over_h = [
        cutoff / (H0_EV * math.sqrt(node.e2))
        for cutoff, node in zip(energy_cutoff_values, nodes)
    ]
    wavenumber_cutoff_over_k_1mpc = [
        cutoff / (MPC_INV_EV * math.exp(-node.n))
        for cutoff, node in zip(wavenumber_cutoff_values, nodes)
    ]
    # q_x=c_s sqrt(A)/barM가 q_sc보다 먼저 나타나는 최소 barM.
    k4_bar_m_required = [
        math.sqrt(node.cs2)
        * math.sqrt(kinetic_coefficient_ev4(node))
        / cutoff
        for cutoff, node in zip(wavenumber_cutoff_values, nodes)
    ]

    reference = frozen[gamma]
    amplitude_relative_error = abs(amplitude - float(reference["A"])) / float(reference["A"])
    max_delta = max(node.u / KAPPA for node in nodes)
    max_delta_relative_error = abs(max_delta - float(reference["max_delta_a_1e-4_to_1"])) / float(
        reference["max_delta_a_1e-4_to_1"]
    )
    require(amplitude_relative_error < 2.0e-7, "독립 shooting amplitude가 동결 결과와 맞지 않는다")
    require(max_delta_relative_error < 2.0e-5, "독립 background delta가 동결 결과와 맞지 않는다")
    require(r_max < 1.0, "tachyon timescale가 Hubble보다 빠르다")
    require(constant_coefficient_bound < 1.0, "보수적 fixed-background log 증폭상계가 order one 이상이다")
    require(min(energy_cutoff_over_h) > 1.0, "관측 배경 구간에서 energy cutoff가 H 아래로 내려갔다")
    require(
        min(wavenumber_cutoff_over_k_1mpc) > 1.0,
        "관측 배경 구간에서 wavenumber cutoff가 physical k=1 Mpc^-1 아래로 내려갔다",
    )

    return {
        "gamma": gamma,
        "b": b,
        "amplitude": amplitude,
        "amplitude_relative_error_vs_frozen": amplitude_relative_error,
        "max_delta_relative_error_vs_frozen": max_delta_relative_error,
        "min_friction_per_efold": f_min,
        "max_abs_m_eff2_over_H2": r_max,
        "S_tach_integral_mu2_over_3H_dt": slow_roll_exponent,
        "integrated_local_growth_exponent": exact_friction_exponent,
        "direct_pi_growth_minus_one_at_present": growth_end,
        "direct_pi_growth_minus_one_max": growth_max,
        "constant_coefficient_log_growth_bound": constant_coefficient_bound,
        "max_cs2": max(node.cs2 for node in nodes),
        "min_cs2": min(node.cs2 for node in nodes),
        "min_Qs_over_Mpl2": min(node.q_s_over_mpl2 for node in nodes),
        "min_dln_a3Qs_dN": min(node.pump_slope for node in nodes),
        "min_dln_Ha3Qs_dN": min(node.zeta_integrand_decay_slope for node in nodes),
        "rho_inf_eV4": rho_inf_ev4,
        "bare_Lambda3_eV": lambda3_ev,
        "bare_Lambda4_eV": lambda4_ev,
        "min_strong_coupling_energy_cutoff_eV": min(energy_cutoff_values),
        "max_strong_coupling_energy_cutoff_eV": max(energy_cutoff_values),
        "min_physical_wavenumber_cutoff_eV": min(wavenumber_cutoff_values),
        "max_physical_wavenumber_cutoff_eV": max(wavenumber_cutoff_values),
        "min_energy_cutoff_over_H": min(energy_cutoff_over_h),
        "min_wavenumber_cutoff_over_physical_k_1_Mpc_inv": min(
            wavenumber_cutoff_over_k_1mpc
        ),
        "k4_barM_required_at_present_eV": k4_bar_m_required[-1],
        "max_k4_barM_required_over_observation_window_eV": max(k4_bar_m_required),
    }


def structural_no_go_gate() -> dict[str, str]:
    """문서화된 미적분 정리와 k=0에서 k^4의 한계를 명시한다."""

    return {
        "saturating_convex_no_go": (
            "비상수·C2·단조 증가·상계 함수는 [0,∞) 전체에서 V''>=0일 수 없다"
        ),
        "proof": (
            "V''>=0이면 V'는 비감소이고, 어느 점의 V'>0는 이후 선형 이상의 "
            "무한 증가를 강제한다; 따라서 상계이면 V'≡0뿐이다"
        ),
        "k4_limit": "양의 k^4 항은 UV gradient를 안정화하지만 k=0 음의 질량항을 제거하지 않는다",
    }


def refinement_gate(
    fine_rows: Sequence[dict[str, float]],
    frozen: dict[float, dict[str, float]],
) -> dict[str, float]:
    """gamma=10에서 배경/섭동 격자를 절반으로 줄여 수렴을 독립 확인한다."""

    fine = next(row for row in fine_rows if row["gamma"] == 10.0)
    coarse = diagnose(10.0, frozen, steps=3000)
    keys = (
        "max_abs_m_eff2_over_H2",
        "S_tach_integral_mu2_over_3H_dt",
        "integrated_local_growth_exponent",
        "direct_pi_growth_minus_one_at_present",
    )
    relative = {
        key: abs(fine[key] - coarse[key]) / max(abs(fine[key]), 1.0e-40)
        for key in keys
    }
    maximum = max(relative.values())
    require(maximum < 2.0e-5, "R2 성장량의 2:1 격자 수렴이 부족하다")
    return {
        "gamma": 10.0,
        "coarse_steps": 3000.0,
        "fine_steps": 6000.0,
        "max_relative_change": maximum,
        **{f"relative_change__{key}": value for key, value in relative.items()},
    }


def main() -> None:
    frozen = frozen_rows()
    require(set(frozen) == set(GAMMAS), "동결 background 비교 행이 누락됐다")
    rows = [diagnose(gamma, frozen) for gamma in GAMMAS]
    output = {
        "schema": "ce-r2-observation-window-perturbation-cutoff-gate-v2",
        "status": "OBSERVATION_WINDOW_DIAGNOSTICS_PASS__MULTICOMPONENT_AND_FUTURE_COMPLETION_OPEN",
        "structural_result": structural_no_go_gate(),
        "rows": rows,
        "grid_refinement": refinement_gate(rows, frozen),
        "global_max_abs_m_eff2_over_H2": max(row["max_abs_m_eff2_over_H2"] for row in rows),
        "global_max_log_growth_bound": max(
            row["constant_coefficient_log_growth_bound"] for row in rows
        ),
        "single_clock_adm_observation_window": {
            "global_min_cs2": min(row["min_cs2"] for row in rows),
            "global_min_Qs_over_Mpl2": min(row["min_Qs_over_Mpl2"] for row in rows),
            "global_min_dln_a3Qs_dN": min(row["min_dln_a3Qs_dN"] for row in rows),
            "global_min_dln_Ha3Qs_dN": min(row["min_dln_Ha3Qs_dN"] for row in rows),
        },
        "strong_coupling_power_counting": {
            "H0_eV": H0_EV,
            "Mpc_inverse_eV": MPC_INV_EV,
            "global_min_energy_cutoff_eV": min(
                row["min_strong_coupling_energy_cutoff_eV"] for row in rows
            ),
            "global_min_physical_wavenumber_cutoff_eV": min(
                row["min_physical_wavenumber_cutoff_eV"] for row in rows
            ),
            "global_min_energy_cutoff_over_H": min(
                row["min_energy_cutoff_over_H"] for row in rows
            ),
            "global_min_wavenumber_cutoff_over_physical_k_1_Mpc_inv": min(
                row["min_wavenumber_cutoff_over_physical_k_1_Mpc_inv"] for row in rows
            ),
            "global_max_k4_barM_required_over_observation_window_eV": max(
                row["max_k4_barM_required_over_observation_window_eV"] for row in rows
            ),
            "global_max_k4_barM_required_at_present_eV": max(
                row["k4_barM_required_at_present_eV"] for row in rows
            ),
            "scope": (
                "Lambda_E=Lambda_3 c_s^(7/4)와 q_sc=Lambda_E/c_s는 "
                "delta->0 decoupling 연산자의 tree-level power-counting estimate다. "
                "유한 관측창의 scale separation만 검사하며 UV completion을 "
                "만들거나 그 계수를 유도하지 않는다."
            ),
        },
        "not_tested": [
            "baryon/radiation/reservoir perturbation을 포함한 coupled ADM kinetic/gradient matrix",
            "bath perturbation의 retarded self-energy와 pole",
            "유한 계수를 가진 ghost-condensate k^4 또는 다른 UV completion",
            "multi-component gauge-invariant adiabatic/isocurvature mode",
            "Einstein-Boltzmann CMB/LSS/lensing likelihood",
            "delta->0 무한 미래까지의 EFT 유효성",
        ],
        "single_clock_adm_scope": (
            "Q_s>0, c_s^2>0 및 a^3Q_s pump는 clock+GR 단일-clock 부분계의 "
            "필요조건이다. baryon/radiation/reservoir entropy perturbation을 "
            "끈 계산은 full gauge-invariant cosmology가 아니다."
        ),
        "interpretation": (
            "현재 동결 background와 kappa=1e17에서는 gauge-dependent fixed-metric "
            "tachyon 진단의 누적 성장이 미미하고, clock+GR 단일-clock 부분계의 "
            "Q_s와 c_s^2가 양수이며, 관측창의 cosmological scale은 energy와 "
            "wavenumber power-counting cutoff보다 충분히 낮다. 이는 multi-component full "
            "stability, 미래 UV completion, dark-sector origin의 증명이 아니다."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
