"""실제 Regge 이차 작용과 길이 차이에 결합한 조화환경을 검산한다.

Q-0020의 같은 작용에서 감쇠·잡음·측도를 얻는 경로에 대한 조건부 검사다.
q, X는 기준 길이로 나눈 섭동, tau=t E_*/hbar, T=k_B T_physical/E_*.
N, W, omega, g는 무차원이며 Regge 계수 length_*^2/(8pi length_P^2)=1을
공급한다. W>0 및 환경 주파수·결합·자유 열적 초기상태도 외부 입력이다.
선형 원천항은 생략했다. 이는 Hessian이나 Gaussian 수렴 여부를 바꾸지 않는다.

공급된 Euclidean 이차 작용은
 S_E = integral dtau [qdot^T W qdot/2 + q^T N q/2
       + sum_j (Xdot_j^2+omega_j^2(X_j+g_j C_j q/omega_j^2)^2)/2].
환경만 적분하는 것은 N의 부호와 무관하게 잘 정의된다. 전체 q 적분의
수렴은 별도 문제다. 완전제곱의 counterterm은 원래 정적 N을 보존하는 입력이다.
K_eff(s)=N+s^2 W+C^T diag[g^2 s^2/(omega^2(s^2+omega^2))] C.
저주파 운동항 보정은 delta_W=C^T diag(g^2/omega^4) C이다.
자유 열적 환경 초기조건에서
 W qddot+N q+integral_0^t Gamma(t-u) qdot(u)du+Gamma(t)q(0)=xi(t),
 Gamma(t)=C^T diag[g^2 cos(omega t)/omega^2]C,
 <{xi(t),xi(0)}>/2=C^T diag[g^2 coth(omega/(2T)) cos(omega t)/(2omega)]C.
T=0이면 coth 인자는 1이다. 조건부 이동 환경은 다른 초기조건이다.
표준 출처: https://doi.org/10.1016/0378-4371(83)90013-4.

Cv=0이면 모든 주파수의 환경 보정, 직접 잡음, delta_W가 v에서 사라진다.
N이나 W가 이 부분공간을 보존하지 않으면 실제 흐름에서 다른 방향과 섞일 수
있으므로 영구 무감쇠 모드라고 부르지 않는다.

regular boundary length sqrt(2), theta=pi-acos(1/4)에서 Schlaefli에 의해
 H=theta Hessian(sum areas)-sum grad(area) grad(dihedral)^T.
삼각형 미분과 역 Gram 미분은 다음 세 entry를 준다:
 diag=-5sqrt(3)theta/6-sqrt(5)/10,
 adjacent=2sqrt(3)theta/9+sqrt(5)/15, disjoint=-sqrt(5)/10.
Johnson J(5,2)의 adjacent 고윳값 6,1,-2, disjoint 고윳값 3,-2,1로
 lambda_1=sqrt(3)theta/2,
 lambda_4=(3sqrt(5)-11sqrt(3)theta)/18,
 lambda_5=-(6sqrt(5)+23sqrt(3)theta)/18.
각 중복도는 1,4,5다. 단위 cycle=(e01-e02-e13+e23)/2는 lambda_5<0.
그 barycentric 내부 길이 미분은 0이며, 길이 사본으로 올린 v는 Cv=0.
따라서 N+C^T D C에 임의의 양의 D를 더해도 v^T N v는 바뀌지 않는다.
상수 시간모드와 경계 섭동을 적분하는 전체 실수 Gaussian은 수렴하지 않는다.
이는 선형 섭동 R^n 전체에서의 이차 근사에 관한 명제다.

경계 고정 후 접착과 내부 정점 게이지를 제거한 내부 곡률은 40sqrt(5)>0.
그 적분에는 위 경계 음의 방향이 포함되지 않는다. 대안으로 공급된 경계
정밀도 hI는 h>-lambda_5에서 정칙 실수 Gaussian을 주지만 h를 선택하지 않는다.
음의 고유방향의 복소 contour rotation도 Gaussian을 수렴시킬 수 있으나
복소 길이와 Jacobian 위상은 추가 처방이며 실수 길이 확률로 승격하지 않는다.
Euclidean 부호에서 Lorentzian 불안정성을 추론하지 않는다:
https://arxiv.org/abs/2004.06635.
전체 비선형 Regge 적분, 물리적 시간·W·환경의 유도, 공통 계량 선택과
Einstein 극한은 이 검산 범위 밖이다.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

import length_gluing_bath as gluing


HERE = Path(__file__).resolve().parent


def exact_coarse_hessian():
    theta = math.pi-math.acos(.25)
    diagonal = -5*math.sqrt(3)*theta/6-math.sqrt(5)/10
    adjacent = 2*math.sqrt(3)*theta/9+math.sqrt(5)/15
    disjoint = -math.sqrt(5)/10
    edges = list(combinations(range(5), 2))
    return np.array([
        [diagonal if e == f else adjacent if set(e) & set(f) else disjoint for f in edges]
        for e in edges
    ])


def exact_coarse_spectrum():
    theta = math.pi-math.acos(.25)
    return np.r_[
        np.full(5, -(6*math.sqrt(5)+23*math.sqrt(3)*theta)/18),
        np.full(4, (3*math.sqrt(5)-11*math.sqrt(3)*theta)/18),
        math.sqrt(3)*theta/2,
    ]


@lru_cache(maxsize=2)
def regge_geometry(step=2e-3):
    step = float(step)
    if not math.isfinite(step) or step <= 0 or step > .005:
        raise ValueError("Hessian 차분 간격은 0과 0.005 사이여야 합니다")
    data, r = gluing.length_gluing(1), gluing.reference()
    a, c = data["assembly"], data["constraint"]
    kappas = r.equal_split_kappas(data["cells"], tuple(range(5)), np.full(10, math.pi))
    n = np.zeros((len(a), len(a)))
    for i, (cell, kappa) in enumerate(zip(data["cells"], kappas)):
        lengths = r.cell_lengths(cell, data["points"])
        n[10*i:10*i+10, 10*i:10*i+10] = r.richardson_hessian(
            lambda q: r.simplex_action(q, kappa), lengths, step
        )
    boundary = [i for i, e in enumerate(data["global_edges"]) if max(e) < 5]
    internal = [i for i, e in enumerate(data["global_edges"]) if max(e) >= 5]
    section = np.zeros((a.shape[1], 10))
    section[boundary] = np.eye(10)
    section[internal] = r.RG.barycentric_section_jacobian(np.full(10, math.sqrt(2)))
    cycle = np.array([1., -1, 0, 0, 0, -1, 0, 1, 0, 0])/2
    coarse_numeric = r.richardson_hessian(
        lambda q: r.simplex_action(q, np.full(10, math.pi)), np.full(10, math.sqrt(2)), step
    )
    return {
        "n": n, "a": a, "c": c, "section": section, "cycle": cycle,
        "lifted_cycle": a @ section @ cycle, "internal": internal,
        "coarse_numeric": coarse_numeric,
    }


def _bath(c, omega, coupling):
    c = np.asarray(c, dtype=float)
    omega, coupling = np.asarray(omega, dtype=float), np.asarray(coupling, dtype=float)
    if c.ndim != 2 or min(c.shape) == 0 or not np.isfinite(c).all():
        raise ValueError("결합 지도는 유한한 비어 있지 않은 행렬이어야 합니다")
    if omega.shape != (len(c),) or coupling.shape != omega.shape:
        raise ValueError("환경 주파수와 결합은 제약 행 수와 같아야 합니다")
    if not np.isfinite(omega).all() or np.any(omega <= 0) or not np.isfinite(coupling).all():
        raise ValueError("주파수는 유한한 양수이고 결합은 유한해야 합니다")
    return c, omega, coupling


def euclidean_kernels(n, masses, c, omega, coupling, frequency):
    c, omega, coupling = _bath(c, omega, coupling)
    n, masses = np.asarray(n, dtype=float), np.asarray(masses, dtype=float)
    frequency = float(frequency)
    size = c.shape[1]
    if n.shape != (size, size) or not np.isfinite(n).all() or not np.allclose(n, n.T, atol=1e-12, rtol=0):
        raise ValueError("시스템 Hessian은 유한한 대칭 행렬이어야 합니다")
    if masses.shape != (size,) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("시스템 운동항은 유한한 양수 벡터여야 합니다")
    if not math.isfinite(frequency):
        raise ValueError("유클리드 주파수는 유한해야 합니다")
    s2, coupling2, omega2 = frequency**2, coupling**2, omega**2
    bare = n+s2*np.diag(masses)
    cross = c.T*coupling
    full = np.block([
        [bare+c.T @ ((coupling2/omega2)[:, None]*c), cross],
        [cross.T, np.diag(omega2+s2)],
    ])
    correction = c.T @ ((coupling2*s2/(omega2*(omega2+s2)))[:, None]*c)
    return full, bare+correction, correction


def environment_kernels(c, omega, coupling, time=0., temperature=0.):
    c, omega, coupling = _bath(c, omega, coupling)
    time, temperature = float(time), float(temperature)
    if not math.isfinite(time) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("시간은 유한하고 온도는 유한한 음이 아닌 값이어야 합니다")
    thermal = np.ones(len(omega)) if temperature == 0 else 1/np.tanh(omega/(2*temperature))
    cosine = np.cos(omega*time)
    memory = c.T @ ((coupling**2*cosine/omega**2)[:, None]*c)
    noise = c.T @ ((coupling**2*thermal*cosine/(2*omega))[:, None]*c)
    mass_shift = c.T @ ((coupling**2/omega**4)[:, None]*c)
    return memory, noise, mass_shift


def langevin_slip_case(samples=241):
    """전체 조화계의 정확한 평균 궤적과 환경 소거식을 비교한다."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 5 or samples % 2 == 0:
        raise ValueError("적분 표본 수는 5 이상의 홀수여야 합니다")
    data = regge_geometry()
    n, c = data["n"], data["c"]
    omega, coupling = np.linspace(.7, 1.9, len(c)), np.linspace(.2, .6, len(c))
    masses = np.linspace(.8, 1.2, len(n))
    potential, _, _ = euclidean_kernels(n, masses, c, omega, coupling, 0.)
    inverse_root = 1/np.sqrt(np.r_[masses, np.ones(len(c))])
    values, vectors = np.linalg.eigh(inverse_root[:, None]*potential*inverse_root[None, :])
    initial = np.r_[np.linspace(-.03, .04, len(n)), np.zeros(len(c))]
    amplitudes = vectors.T @ (initial/inverse_root)
    grid = np.linspace(0., .2, samples)
    roots = np.sqrt(np.abs(values))
    phase = grid[:, None]*roots
    cosine = np.where(values[None, :] >= 0, np.cos(phase), np.cosh(phase))
    derivative = np.where(values[None, :] >= 0, -roots*np.sin(phase), roots*np.sinh(phase))
    positions = (cosine*amplitudes) @ vectors.T * inverse_root
    velocities = (derivative*amplitudes) @ vectors.T * inverse_root
    acceleration = -(potential @ positions[-1])*inverse_root**2
    # 초기 환경 평균은 0이며 자유 열적 환경의 평균 방정식에는 slip이 남는다.
    forcing_history = (velocities[:, :len(n)] @ c.T)*(coupling**2/omega**2)
    integrand = forcing_history*np.cos((grid[-1]-grid)[:, None]*omega)
    weights = np.ones(samples)
    weights[1:-1:2], weights[2:-1:2] = 4, 2
    convolution = c.T @ (weights @ integrand)*(grid[1]-grid[0])/3
    memory, _, _ = environment_kernels(c, omega, coupling, time=grid[-1])
    slip = memory @ initial[:len(n)]
    without_slip = masses*acceleration[:len(n)]+n @ positions[-1, :len(n)]+convolution
    return {
        "samples": samples, "time": float(grid[-1]),
        "with_initial_slip_residual": float(np.linalg.norm(without_slip+slip)),
        "omitting_initial_slip_residual": float(np.linalg.norm(without_slip)),
        "slip_norm": float(np.linalg.norm(slip)),
        "bath_mean_initially_zero": True, "conditional_displaced_initial_state": False,
    }


def certificate(step=2e-3):
    data = regge_geometry(step)
    n, a, c, v = data["n"], data["a"], data["c"], data["lifted_cycle"]
    coarse = exact_coarse_hessian()
    expected = exact_coarse_spectrum()
    glued = a.T @ n @ a
    radial = np.ones(5)/math.sqrt(5)
    internal = glued[np.ix_(data["internal"], data["internal"])]
    omega, coupling = np.linspace(.7, 1.9, len(c)), np.linspace(.2, .6, len(c))
    masses = np.linspace(.8, 1.2, len(n))
    frequencies = []
    for frequency in (0., .1, 1., 10.):
        full, effective, correction = euclidean_kernels(n, masses, c, omega, coupling, frequency)
        direct = full[:len(n), :len(n)]-full[:len(n), len(n):] @ np.linalg.solve(
            full[len(n):, len(n):], full[len(n):, :len(n)]
        )
        frequencies.append({
            "frequency": frequency,
            "schur_residual": float(np.linalg.norm(effective-direct)),
            "correction_on_glued_residual": float(np.linalg.norm(correction @ a)),
        })
    memory, noise, mass_shift = environment_kernels(c, omega, coupling, time=.3, temperature=.4)
    threshold = -float(expected[0])
    alternatives = []
    for margin in (.5, 1.):
        precision = coarse+(threshold+margin)*np.eye(10)
        covariance = np.linalg.solve(precision, np.eye(10))
        alternatives.append({
            "supplied_boundary_precision": threshold+margin,
            "smallest_curvature": float(np.linalg.eigvalsh(precision)[0]),
            "cycle_variance": float(data["cycle"] @ covariance @ data["cycle"]),
        })
    eigenvalues, basis = np.linalg.eigh(coarse)
    contour = basis @ np.diag(np.where(eigenvalues < 0, 1j, 1.))
    rotated = contour.T @ coarse @ contour
    jacobian_phase = (1j)**int(np.count_nonzero(eigenvalues < 0))
    return {
        "difference_step": step, "coarse_spectrum": expected.tolist(),
        "coarse_hessian_difference": float(np.linalg.norm(data["coarse_numeric"]-coarse)),
        "section_pullback_difference": float(np.linalg.norm(data["section"].T @ glued @ data["section"]-coarse)),
        "cycle_gluing_residual": float(np.linalg.norm(c @ v)),
        "cycle_internal_section_residual": float(np.linalg.norm((data["section"] @ data["cycle"])[data["internal"]])),
        "cycle_exact_curvature": float(expected[0]), "cycle_raw_curvature": float(v @ n @ v),
        "penalty_curvatures": [
            {"strength": k, "curvature": float(v @ (n+k*c.T @ c) @ v)} for k in (0., 1., 100., 10000.)
        ],
        "fixed_boundary_radial_curvature": float(radial @ internal @ radial),
        "fixed_boundary_exact_radial_curvature": 40*math.sqrt(5),
        "fixed_boundary_gauge_residual": float(np.linalg.norm(internal @ (np.eye(5)-np.ones((5, 5))/5))),
        "frequency_checks": frequencies,
        "memory_on_glued_residual": float(np.linalg.norm(memory @ a)),
        "noise_on_glued_residual": float(np.linalg.norm(noise @ a)),
        "mass_shift_on_glued_residual": float(np.linalg.norm(mass_shift @ a)),
        "unconstrained_flow_mixing_residual": float(np.linalg.norm((c/masses) @ n @ a)),
        "boundary_precision_threshold": threshold, "boundary_state_alternatives": alternatives,
        "rotated_curvature_residual": float(np.linalg.norm(rotated-np.diag(abs(eigenvalues)))),
        "contour_phase_in_spectral_coordinates": [jacobian_phase.real, jacobian_phase.imag],
        "conjugate_contour_phase": [jacobian_phase.conjugate().real, jacobian_phase.conjugate().imag],
        "full_nonlinear_regge_integral_tested": False,
        "lorentzian_instability_inferred": False,
    }


def run():
    files = [
        "regge_gluing_environment.py", "length_gluing_bath.py", "local_refinement_bath.py",
        "continuum_bath.py", "F-01/predict_fold_budget.py",
        "F-01/regge_one_to_five_boundary_hessian.py", "F-01/regge_one_to_five_refinement.py",
    ]
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files},
        "geometry": [certificate(h) for h in (2e-3, 1e-3)],
        "initial_slip": langevin_slip_case(),
        "scope": {
            "system_mass_supplied": True, "bath_spectrum_and_temperature_supplied": True,
            "counterterm_preserves_static_hessian": True,
            "damping_and_noise_computed_from_same_supplied_action": True,
            "gluing_environment_selects_system_kinetic_mass": False,
            "boundary_integrated_real_quadratic_measure_stabilized": False,
            "fixed_boundary_glued_gauge_quotient_positive": True,
            "boundary_state_precision_uniquely_derived": False,
            "complex_contour_is_real_length_probability": False,
            "physical_regge_time_dynamics_derived": False, "common_metric_selected": False,
            "continuum_einstein_limit_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE/"regge_gluing_environment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
