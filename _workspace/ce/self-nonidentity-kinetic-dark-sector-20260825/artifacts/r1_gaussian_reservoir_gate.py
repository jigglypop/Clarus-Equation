"""R1 유계 Gaussian 저장소의 최소 일관성 게이트.

이 스크립트는 CE의 0차원 기원이나 암흑부문 존재량을 증명하지 않는다.
다음의 더 좁고 재현 가능한 명제를 검사한다.

1. 유계 결합을 가진 명시적 clock+bath 작용의 질량차원이 맞는다.
2. 변위된 Gaussian 초기상태가 <T>=0, <J>>0와 양자 불확정성을 함께
   만족할 수 있다.
3. 유한 Gaussian bath의 retarded kernel은 인과적이고 noise kernel은
   양의 준정부호다.
4. 축약 clock 에너지의 손실과 bath+interaction 에너지의 이득이 정확히
   상쇄된다.
5. 비유계 선형 T-phi 결합은 하한을 잃지만 유계 source는 하한을 가진다.
6. 현재 두-미분 bulk에는 장파장 음의 유효 질량제곱이라는 열린 반례가
   남아 있음을 숨기지 않는다.

외부 패키지를 사용하지 않아 저장소 Python hook만으로 실행할 수 있다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


TOL = 1.0e-11


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _jacobi_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    """작은 실대칭 행렬의 고윳값을 외부 선형대수 패키지 없이 계산한다."""

    a = [list(row) for row in matrix]
    n = len(a)
    _require(n > 0 and all(len(row) == n for row in a), "정방행렬이 필요하다")
    for i in range(n):
        for j in range(n):
            _require(abs(a[i][j] - a[j][i]) < 1.0e-12, "행렬이 대칭이 아니다")

    for _ in range(100 * n * n):
        p, q = 0, 1 if n > 1 else 0
        largest = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    largest = abs(a[i][j])
                    p, q = i, j
        if largest < 1.0e-14 or n == 1:
            break

        angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k in (p, q):
                continue
            akp, akq = a[k][p], a[k][q]
            a[k][p] = a[p][k] = c * akp - s * akq
            a[k][q] = a[q][k] = s * akp + c * akq

    return sorted(a[i][i] for i in range(n))


def dimension_gate() -> dict[str, int]:
    """자연단위에서 각 작용항의 총 질량차원을 검사한다."""

    dim = {
        "T": -1,
        "X": 0,
        "P": 4,
        "J": 4,
        "Pi_F": 4,
        "phi": 1,
        "m": 1,
        "Gamma": 1,
        "mu": 1,
        "s": 3,
        "s_prime": 4,
        "D_R": 2,
        "N": 2,
        "K_R": 10,
        "K_Sigma": 5,
        "N_Sigma_bilocal": 8,
        "n_Sigma_local": 5,
    }
    checks = {
        "Gamma_T": dim["Gamma"] + dim["T"],
        "bulk_source_density": dim["s"] + dim["phi"],
        "ctp_response_action": -8 + dim["s"] + dim["D_R"] + dim["s"],
        "ctp_noise_action": -8 + dim["s"] + dim["N"] + dim["s"],
        "boundary_momentum_action": -3 + dim["Pi_F"] + dim["T"],
        "boundary_K_action": -3 + dim["K_Sigma"] + 2 * dim["T"],
        "boundary_N_action": -6 + dim["N_Sigma_bilocal"] + 2 * dim["T"],
        "linearized_memory_equation": -4 + dim["K_R"] + dim["T"],
    }
    for key in (
        "Gamma_T",
        "ctp_response_action",
        "ctp_noise_action",
        "boundary_momentum_action",
        "boundary_K_action",
        "boundary_N_action",
    ):
        _require(checks[key] == 0, f"{key}가 무차원이 아니다")
    _require(checks["bulk_source_density"] == 4, "s(T) phi가 벌크 밀도 차원 4가 아니다")
    _require(checks["linearized_memory_equation"] == 5, "기억항이 T 방정식 차원 5가 아니다")
    return checks


def gaussian_state_gate() -> dict[str, float]:
    """한 coarse-grained mode의 Robertson 양성 및 평균 변위를 검사한다."""

    hbar = 1.0
    var_t = 0.8
    var_p = 0.6
    cov_tp = 0.1
    determinant = var_t * var_p - cov_tp * cov_tp
    lower_bound = (hbar / 2.0) ** 2
    mean_t = 0.0
    mean_p = 2.75

    _require(var_t > 0.0 and var_p > 0.0, "Gaussian 분산이 양수가 아니다")
    _require(determinant >= lower_bound, "Robertson 불확정성 조건을 위반했다")
    _require(mean_t == 0.0 and mean_p > 0.0, "요구한 평균 앵커/운동량 변위가 아니다")
    return {
        "mean_T": mean_t,
        "mean_cell_momentum": mean_p,
        "covariance_determinant": determinant,
        "robertson_lower_bound": lower_bound,
        "margin": determinant - lower_bound,
    }


def kernel_gate() -> dict[str, float | int]:
    """규제된 bath에서 retarded support와 Gaussian noise PSD를 검사한다."""

    times = [0.13 * i for i in range(9)]
    frequencies = (0.7, 1.3, 2.1, 3.0)
    beta = 1.7

    def d_ret(t: float, tp: float) -> float:
        if t < tp:
            return 0.0
        return sum(math.sin(w * (t - tp)) / w for w in frequencies)

    def thermal_weight(w: float) -> float:
        return 1.0 / (2.0 * w * math.tanh(beta * w / 2.0))

    noise = [
        [
            sum(thermal_weight(w) * math.cos(w * (t - tp)) for w in frequencies)
            for tp in times
        ]
        for t in times
    ]
    eigenvalues = _jacobi_eigenvalues(noise)
    acausal_max = max(
        (abs(d_ret(t, tp)) for t in times for tp in times if t < tp),
        default=0.0,
    )
    _require(acausal_max <= TOL, "retarded kernel에 미래 지지가 있다")
    _require(eigenvalues[0] >= -1.0e-10, "noise kernel이 양의 준정부호가 아니다")
    return {
        "time_nodes": len(times),
        "bath_modes": len(frequencies),
        "max_acausal_entry": acausal_max,
        "min_noise_eigenvalue": eigenvalues[0],
        "max_noise_eigenvalue": eigenvalues[-1],
    }


@dataclass(frozen=True)
class Model:
    rho_inf: float = 1.0
    kappa: float = 12.0
    x_star: float = 0.5
    gamma: float = 0.04
    masses: tuple[float, ...] = (1.1, 1.7, 2.3)
    mus: tuple[float, ...] = (0.16, 0.12, 0.10)

    def source(self, index: int, t_field: float) -> float:
        return self.mus[index] ** 3 * math.tanh(self.gamma * t_field)

    def source_prime(self, index: int, t_field: float) -> float:
        z = self.gamma * t_field
        return self.mus[index] ** 3 * self.gamma / math.cosh(z) ** 2


def _delta(model: Model, velocity: float) -> float:
    return velocity * velocity / (2.0 * model.x_star) - 1.0


def _rho_x(model: Model, velocity: float) -> float:
    delta = _delta(model, velocity)
    return model.rho_inf * model.kappa * (2.0 + 3.0 * delta) / model.x_star


def _rho_clock(model: Model, t_field: float, velocity: float) -> float:
    delta = _delta(model, velocity)
    return model.rho_inf * (
        2.0 * model.kappa * delta
        + 1.5 * model.kappa * delta * delta
        + 1.0
        - math.exp(-model.gamma * t_field)
    )


def _total_energy(model: Model, state: Sequence[float]) -> float:
    t_field, velocity = state[0], state[1]
    total = _rho_clock(model, t_field, velocity)
    for i, (mass, _) in enumerate(zip(model.masses, model.mus)):
        phi = state[2 + 2 * i]
        phi_dot = state[3 + 2 * i]
        total += 0.5 * phi_dot * phi_dot + 0.5 * mass * mass * phi * phi
        total += model.source(i, t_field) * phi
    return total


def _rhs(model: Model, state: Sequence[float]) -> list[float]:
    t_field, velocity = state[0], state[1]
    p_t = -model.rho_inf * model.gamma * math.exp(-model.gamma * t_field)
    force_from_bath = sum(
        model.source_prime(i, t_field) * state[2 + 2 * i]
        for i in range(len(model.masses))
    )
    acceleration = (p_t - force_from_bath) / _rho_x(model, velocity)
    result = [velocity, acceleration]
    for i, mass in enumerate(model.masses):
        phi = state[2 + 2 * i]
        phi_dot = state[3 + 2 * i]
        result.extend((phi_dot, -mass * mass * phi - model.source(i, t_field)))
    return result


def _rk4_step(
    rhs: Callable[[Sequence[float]], Sequence[float]],
    state: Sequence[float],
    dt: float,
) -> list[float]:
    k1 = list(rhs(state))
    k2_state = [y + 0.5 * dt * k for y, k in zip(state, k1)]
    k2 = list(rhs(k2_state))
    k3_state = [y + 0.5 * dt * k for y, k in zip(state, k2)]
    k3 = list(rhs(k3_state))
    k4_state = [y + dt * k for y, k in zip(state, k3)]
    k4 = list(rhs(k4_state))
    return [
        y + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for y, a, b, c, d in zip(state, k1, k2, k3, k4)
    ]


def conservation_gate() -> dict[str, float]:
    """명시적 clock+bath 계의 에너지 보존과 안정 branch를 수치 검산한다."""

    model = Model()
    delta_initial = 0.22
    velocity_initial = math.sqrt(2.0 * model.x_star * (1.0 + delta_initial))
    state = [0.0, velocity_initial]
    for i in range(len(model.masses)):
        state.extend((0.025 * (-1.0) ** i, 0.01 / (i + 1)))

    initial_energy = _total_energy(model, state)
    max_energy_error = 0.0
    min_delta = _delta(model, state[1])
    max_exchange_residual = 0.0
    dt = 0.002
    steps = 5000

    for step in range(steps):
        if step % 25 == 0:
            t_field, velocity = state[0], state[1]
            derivative = _rhs(model, state)
            bath_force = sum(
                model.source_prime(i, t_field) * state[2 + 2 * i]
                for i in range(len(model.masses))
            )
            clock_rate = (
                _rho_x(model, velocity) * velocity * derivative[1]
                + model.rho_inf
                * model.gamma
                * math.exp(-model.gamma * t_field)
                * velocity
            )
            bath_rate = 0.0
            for i, mass in enumerate(model.masses):
                phi = state[2 + 2 * i]
                phi_dot = state[3 + 2 * i]
                phi_ddot = derivative[3 + 2 * i]
                source = model.source(i, t_field)
                source_prime = model.source_prime(i, t_field)
                bath_rate += (
                    phi_dot * phi_ddot
                    + mass * mass * phi * phi_dot
                    + source_prime * velocity * phi
                    + source * phi_dot
                )
            expected_clock_rate = -bath_force * velocity
            max_exchange_residual = max(
                max_exchange_residual,
                abs(clock_rate - expected_clock_rate),
                abs(clock_rate + bath_rate),
            )

        state = _rk4_step(lambda y: _rhs(model, y), state, dt)
        energy = _total_energy(model, state)
        max_energy_error = max(max_energy_error, abs(energy - initial_energy))
        min_delta = min(min_delta, _delta(model, state[1]))

    relative_energy_drift = max_energy_error / max(1.0, abs(initial_energy))
    _require(min_delta > 0.0, "적분 경로가 delta>0 안정 branch를 벗어났다")
    _require(relative_energy_drift < 1.0e-10, "전체 에너지 수치 보존이 허용오차를 넘었다")
    _require(max_exchange_residual < 1.0e-11, "clock-bath 교환식이 닫히지 않는다")
    return {
        "integration_time": dt * steps,
        "min_delta": min_delta,
        "relative_total_energy_drift": relative_energy_drift,
        "max_exchange_identity_residual": max_exchange_residual,
    }


def boundedness_gate() -> dict[str, float]:
    """선형 결합 음성대조군과 유계 source의 하한을 비교한다."""

    mass = 1.4
    coupling = 0.8
    linear_minima = [-(coupling * t) ** 2 / (2.0 * mass * mass) for t in (1.0, 10.0, 100.0)]
    _require(
        linear_minima[2] < linear_minima[1] < linear_minima[0],
        "선형 결합의 무하한 음성대조군이 재현되지 않았다",
    )

    mu = 0.7
    gamma = 0.3
    bound = -(mu**6) / (2.0 * mass * mass)
    bounded_minima = [
        -(mu**3 * math.tanh(gamma * t)) ** 2 / (2.0 * mass * mass)
        for t in (0.0, 1.0, 10.0, 100.0)
    ]
    _require(min(bounded_minima) >= bound - TOL, "유계 source의 완성제곱 하한이 깨졌다")
    return {
        "linear_minimum_at_T_100": linear_minima[-1],
        "bounded_sample_minimum": min(bounded_minima),
        "analytic_bounded_lower_bound": bound,
    }


def open_bulk_falsifier() -> dict[str, float | str]:
    """두-미분 clock bulk의 미해결 장파장 곡률 부호를 기록한다."""

    model = Model()
    t_field = 0.5
    delta = 0.1
    velocity = math.sqrt(2.0 * model.x_star * (1.0 + delta))
    p_tt = model.rho_inf * model.gamma**2 * math.exp(-model.gamma * t_field)
    effective_mass_squared = -p_tt / _rho_x(model, velocity)
    sound_speed_squared = delta / (2.0 + 3.0 * delta)
    near_condensate_sound_speed_squared = 1.0e-12 / (2.0 + 3.0e-12)

    _require(effective_mass_squared < 0.0, "예상한 장파장 음의 곡률이 포착되지 않았다")
    _require(sound_speed_squared > 0.0, "선택한 delta>0 가지의 기울기 조건이 깨졌다")
    return {
        "status": "OPEN: full metric-mixed perturbation and k^4 completion required",
        "fixed_background_m_eff_squared": effective_mass_squared,
        "sound_speed_squared_at_delta_0_1": sound_speed_squared,
        "sound_speed_squared_near_delta_zero": near_condensate_sound_speed_squared,
    }


def main() -> None:
    result = {
        "schema": "ce-r1-gaussian-reservoir-gate-v1",
        "claim_ceiling": (
            "유계 source와 규제된 Gaussian reservoir 및 양의 초기 Gaussian 상태를 "
            "채택하면 조건부 인과성·잡음 양성·총 에너지 보존을 함께 구현할 수 있다"
        ),
        "not_derived": [
            "Pi_F의 수치",
            "0차원 접힘과 bath의 동일성",
            "점형 기록에서 균일 FLRW 초기면으로 가는 사상",
            "내재적 시간 화살",
            "암흑물질·암흑에너지 존재량",
            "full CMB/LSS/metric-mixed perturbation 안정성",
        ],
        "dimensions": dimension_gate(),
        "gaussian_state": gaussian_state_gate(),
        "kernels": kernel_gate(),
        "conservation": conservation_gate(),
        "boundedness": boundedness_gate(),
        "bulk_long_wavelength_gate": open_bulk_falsifier(),
        "overall_status": "CONDITIONAL_PASS_WITH_OPEN_COSMOLOGY_GATES",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
