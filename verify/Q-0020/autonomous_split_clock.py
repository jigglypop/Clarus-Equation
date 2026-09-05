"""분할 생성자를 양의 위치시계와 합성하고 유한 준비 오차를 검산한다.

무차원 x=X/L_*, P=L_* p/hbar, mu=M E_* L_*^2/hbar^2,
tau=E_* t/hbar를 사용한다. G=R^T K R/2와 f, 초기 시계는 공급값이다.
H/E_*=U_f(P^2/(2mu)+H0/E_*)U_f^dagger로 정의하며 사차항 G^2를 유지한다.
이 모형은 기하, 초기 준비 장치, 재충전이나 CE 작용을 유도하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag, expm
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import split_quadratic_generator as generator


def profile(x, length=1.0):
    """구간 밖에서 정확히 일정하고 두 번 연속 미분 가능한 회전 비율."""
    if not math.isfinite(length) or length <= 0:
        raise ValueError("결합 구간 길이는 양의 유한수여야 한다")
    z = np.clip(np.asarray(x, dtype=float)/length, 0, 1)
    return z**3*(10-15*z+6*z*z), 30*z*z*(1-z)**2/length


def _clock(sigma, distance, momentum, mass, length):
    values = np.array([sigma, distance, momentum, mass, length], dtype=float)
    if not np.isfinite(values).all() or np.min(values) <= 0:
        raise ValueError("시계 폭·거리·운동량·질량·구간 길이는 양의 유한수여야 한다")


def clock_bound(time, sigma=1.0, distance=8.0, momentum=4.0, mass=1.0, length=1.0):
    """참조계를 포함한 상태 구별도 D=||rho-sigma||_1/2의 상계."""
    _clock(sigma, distance, momentum, mass, length)
    if not math.isfinite(time) or time < 0:
        raise ValueError("시간은 음이 아닌 유한수여야 한다")
    initial_tail = float(ndtr(-distance/sigma))
    negative_momentum = float(ndtr(-2*sigma*momentum))
    center = -distance+momentum*time/mass
    width = math.hypot(sigma, time/(2*mass*sigma))
    unfinished = float(ndtr((length-center)/width))
    return {
        "time": time, "initial_position_tail": initial_tail,
        "negative_momentum_probability": negative_momentum,
        "unfinished_probability": unfinished,
        "trace_distance_bound": min(1.0, 2*(math.sqrt(initial_tail)+math.sqrt(unfinished))),
        "limiting_bound": min(1.0, 2*(math.sqrt(initial_tail)+math.sqrt(negative_momentum))),
        "clock_kinetic_energy": (momentum**2+1/(4*sigma**2))/(2*mass),
    }


def completion_time(tolerance, sigma=1.0, distance=8.0, momentum=4.0, mass=1.0, length=1.0):
    """이후 모든 시간에 구별도 상계를 만족시키는 시각의 수치 평가."""
    data = clock_bound(0, sigma, distance, momentum, mass, length)
    if not math.isfinite(tolerance) or not data["limiting_bound"] < tolerance < 1:
        raise ValueError("허용오차는 장시간 상계보다 크고 1보다 작아야 한다")
    allowed = (tolerance/2-math.sqrt(data["initial_position_tail"]))**2
    q = -float(ndtri(allowed))
    a, b, c = length+distance, momentum/mass, 1/(2*mass*sigma)
    denominator = b*b-q*q*c*c
    if denominator <= 0:
        raise ArithmeticError("유한 완료 시각의 분모가 양수가 아니다")
    return (a*b+q*math.sqrt(b*b*sigma*sigma+c*c*a*a-q*q*c*c*sigma*sigma))/denominator


def transported_form(k, fraction=1.0, winding=0):
    """정준행렬과 양자 생성자를 구분해 출력 에너지 형식을 전달한다."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("경로 비율은 0과 1 사이여야 한다")
    data = generator.witness(k, winding)
    omega = np.kron(np.eye(k), generator.source.J)
    symplectic = expm(fraction*omega @ data["generator"])
    inverse = np.linalg.solve(symplectic, np.eye(2*k))
    return symplectic, inverse.T @ inverse


def initial_vacuum_energy(k, sigma=1.0, distance=8.0, momentum=4.0,
                          mass=1.0, length=1.0, winding=0):
    """독립 곱 상태의 전체 초기 에너지. 시계 꼬리의 결합 에너지도 포함."""
    _clock(sigma, distance, momentum, mass, length)
    data = generator.witness(k, winding)
    metric = data["generator"]
    omega = np.kron(np.eye(k), generator.source.J)
    mean_g = float(np.trace(metric)/4)
    variance_g = float((np.trace(metric @ metric)+np.trace(metric @ omega @ metric @ omega))/8)
    second_g = mean_g**2+variance_g
    density = lambda x: math.exp(-(x+distance)**2/(2*sigma*sigma))/(math.sqrt(2*math.pi)*sigma)
    integrate = lambda fun: quad(fun, 0, length, epsabs=1e-27, epsrel=1e-10)[0]
    mean_slope = integrate(lambda x: density(x)*float(profile(x, length)[1]))
    slope_square = integrate(lambda x: density(x)*float(profile(x, length)[1])**2)

    def potential_excess(x):
        fraction = float(profile(x, length)[0])
        inverse = expm(-fraction*omega @ metric)
        return float(np.trace(inverse.T @ inverse)/4-k/2)

    potential = integrate(lambda x: density(x)*potential_excess(x))
    potential += float(ndtr(-(distance+length)/sigma))*potential_excess(length)
    kinetic = (momentum**2+1/(4*sigma*sigma))/(2*mass)
    cross = momentum*mean_slope*mean_g/mass
    square = slope_square*second_g/(2*mass)
    return {"branching": k, "winding": winding, "clock_kinetic_energy": kinetic,
            "bare_internal_energy": k/2, "gauge_cross_energy": cross,
            "gauge_square_energy": square, "transported_potential_excess": potential,
            "total_energy": kinetic+k/2+cross+square+potential,
            "generator_vacuum_mean": mean_g, "generator_vacuum_second_moment": second_g}


def missing_square_witness(k, occupation, mass=1.0):
    """완전제곱에서 G^2를 빼면 이차형식이 아래로 무한함을 보이는 상태족."""
    if isinstance(occupation, bool) or not isinstance(occupation, int) or occupation < 0:
        raise ValueError("점유수는 음이 아닌 정수여야 한다")
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("시계 질량은 양의 유한수여야 한다")
    data = generator.witness(k)
    levels = np.full(k, .5)
    levels[0] += occupation
    covariance = data["basis"] @ np.diag(np.repeat(levels, 2)) @ data["basis"].T
    internal = float(np.trace(covariance)/2)
    eigen_g = float(data["frequencies"] @ levels)
    # chi(x)=4/sqrt(5)*sin(pi*x)^3, 0<x<1. 확장 함수는 H^2에 속한다.
    slope_square = quad(lambda x: 16/5*math.sin(math.pi*x)**6*
                        float(profile(x)[1])**2, 0, 1, epsabs=1e-12)[0]
    clock_energy = 9*math.pi**2/(10*mass)
    good = clock_energy+internal
    omitted = slope_square*eigen_g**2/(2*mass)
    return {"branching": k, "occupation": occupation, "full_square_energy": good,
            "omitted_square_energy": omitted, "truncated_form_energy": good-omitted}


def finite_grid_witness():
    """유한 격자의 직접 행렬 진화 대조. 무한 직선의 산란 증명으로 쓰지 않는다."""
    x = np.linspace(-12, 12, 65)
    dx = x[1]-x[0]
    free_clock = (2*np.eye(len(x))-np.eye(len(x), k=1)-np.eye(len(x), k=-1))/(2*dx*dx)
    internal = np.diag([.5, 1.5])
    gate_generator = np.array([[1.4, .6], [.6, .8]])
    rotations = [expm(-1j*float(profile(value)[0])*gate_generator) for value in x]
    dressing = block_diag(*rotations)
    free = np.kron(free_clock, np.eye(2))+np.kron(np.eye(len(x)), internal)
    total = dressing @ free @ dressing.conj().T
    clock = np.exp(-(x+5)**2/(4*.6**2)+2j*x)
    clock /= np.linalg.norm(clock)
    bell = np.eye(2)/math.sqrt(2)
    initial = np.einsum("x,ij->xij", clock, bell).reshape(2*len(x), 2)
    time = 7.0
    clock_evolution = expm(-1j*time*free_clock)
    internal_evolution = expm(-1j*time*internal)
    free_evolution = np.kron(clock_evolution, internal_evolution)
    direct = expm(-1j*time*total) @ initial
    factored = dressing @ free_evolution @ dressing.conj().T @ initial
    evolved_clock = clock_evolution @ clock
    target = np.einsum("x,ij->xij", evolved_clock,
                      expm(-1j*gate_generator) @ internal_evolution @ bell).reshape(2*len(x), 2)
    overlap = np.vdot(target, direct)
    distance = math.sqrt(max(0.0, 1-abs(overlap)**2))
    initial_tail = float(sum(abs(clock[x > 0])**2))
    unfinished = float(sum(abs(evolved_clock[x < 1])**2))
    bound = min(1.0, 2*(math.sqrt(initial_tail)+math.sqrt(unfinished)))
    energy = lambda state: float(np.trace(state.conj().T @ total @ state).real)
    return {
        "internal_dimension": 2, "reference_dimension": 2, "clock_sites": len(x),
        "minimum_total_eigenvalue": float(np.linalg.eigvalsh(total)[0]),
        "noncommutation_norm": float(np.linalg.norm(gate_generator @ internal-internal @ gate_generator)),
        "state_factorization_residual": float(np.linalg.norm(direct-factored)),
        "energy_conservation_residual": abs(energy(direct)-energy(initial)),
        "norm_residual": abs(float(np.linalg.norm(direct))-1),
        "product_input_trace_distance": distance, "discrete_trace_distance_bound": bound,
        "continuum_scattering_or_infinite_time_proved_by_grid": False,
    }


def run():
    target_time = completion_time(1e-3)
    transport = []
    for k in (2, 3):
        symplectic, form = transported_form(k)
        omega = np.kron(np.eye(k), generator.source.J)
        residual = max(float(np.linalg.norm(expm(time*omega @ form) @ symplectic
                                             -symplectic @ expm(time*omega)))
                       for time in (.3, 1.0, 2.7))
        _, wound = transported_form(k, winding=1)
        transport.append({"branching": k, "minimum_output_form_eigenvalue": float(np.linalg.eigvalsh(form)[0]),
                          "free_evolution_intertwining_residual": residual,
                          "winding_output_form_residual": float(np.linalg.norm(form-wound))})
    paths = (Path(__file__), HERE/"split_quadratic_generator.py", HERE/"split_quantum_source.py")
    return {
        "status": "conditional",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "clock_parameters": {"sigma": 1.0, "distance": 8.0, "momentum": 4.0, "mass": 1.0, "length": 1.0},
        "tolerance": 1e-3, "completion_time": target_time,
        "clock_bounds": [clock_bound(time) for time in (target_time, 8, 16, 32)],
        "initial_product_energies": [initial_vacuum_energy(k, winding=n) for k in (2, 3) for n in (0, 1)],
        "overlap_preparation_control": [initial_vacuum_energy(2, distance=1, momentum=1, winding=n) for n in (0, 1)],
        "transport": transport, "finite_grid": finite_grid_witness(),
        "missing_square": [missing_square_witness(k, n) for k in (2, 3) for n in (0, 4, 16, 64)],
        "scope": {
            "positive_time_independent_clock_action_constructed": True,
            "finite_product_preparation_error_bounded_with_reference": True,
            "output_subspace_preserved_under_transported_hamiltonian": True,
            "same_bare_input_output_hamiltonian": False,
            "clock_and_ancilla_preparation_derived": False,
            "full_action_is_quadratic": False,
            "kinetic_square_retained": True,
            "finite_clock_exact_zero_error_proved": False,
            "spatial_locality_or_CE_action_derived": False,
            "common_metric_selected": False,
            "charging_merge_recycling_cycle_closed": False,
        },
    }


if __name__ == "__main__":
    result = run()
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "completion_time": result["completion_time"],
                      "grid_residual": result["finite_grid"]["state_factorization_residual"]}))

