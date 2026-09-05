"""상한 없는 연속 환경의 조건부 방출과 실제 분할 상태의 잔여 상계를 검산한다.

진전 원장 §2의 준비·환경 장부에 연결되며, CE의 미시 작용을 도출하지 않는다.
에너지는 E_*로 나누고 tau=E_* t/hbar로 둔다. 한 입자 공간은
C|0> + L2([0,infinity), dx), h00=epsilon, 환경 에너지=x,
결합의 제곱 J(x)=kappa*x*exp(-x)이다. epsilon>kappa>0을 가정한다.
전체 H/E_*=dGamma(h)+u*n0*(n0-1)/2, u>=0이다.

조건부 증명의 범위:
1. 제곱 완성으로 h>=0. |0>은 순환 벡터이다. 이를 보려면 그 resolvent
   궤도에 직교하는 (a,f)를 취한다. 큰 허수 z의 극한은 a=0을 주고,
   v(x)*f(x)의 Cauchy 변환이 0이므로 v>0에서 f=0이다.
2. (z-h)^-1 관례에서 Sigma(E+i0)=kappa*(E*exp(-E)*Ei(E)-1)-i*pi*J(E).
   Ei는 양의 실축의 실수 주값이다. 모든 E>0에서 rho0(E)>0이고
   0 근처 rho0(E)~kappa*E/(epsilon-kappa)^2이다. 실축의 국소적으로
   연속인 유한 resolvent 경계값, 음의 근 부재와 영점 원자 부재가
   순수 절대연속 측도를 준다. 따라서 u=0이면 N(t)=N0*|A(t)|^2 -> 0.
3. N=2에서 자유 |2_0> 측도는 rho2=rho0*rho0>0이고 0 근처 O(E^3)이다.
   H2-H20=u|2_0><2_0|. G2(z)=integral rho2(s)/(z-s) ds의 관례에서
   새 측도는 rho2/|1-u*G2|^2이다. 양의 실축에서는 허수부가 분모의
   영점을 막고, 0에서는 G2(0)<0, 음의 축에서는 H2>=0이다.
   순환 부분공간의 국소적으로 유계인 resolvent 경계값은 특이 측도를
   배제한다. 그 직교 여공간은 자유 연산자 그대로이다.
4. Kato 1957 정리 1은 이 유계 자기수반 rank-one 차이에 적용된다.
   자유 두 입자의 n0 진화는 강하게 0으로 간다: 한 입자의 국소
   진폭은 L1 함수의 Fourier 변환이고, 텐서 곱에서는 유계 수렴을 쓴다.
   완비 파동연산자가 자유 상태에 대한 노름 근사를 주므로 상호작용
   N=2의 n0 기대값도 0이다. 생존 진폭만으로 이 결론을 내리지 않는다.
5. 실제 squeezed source의 N>=4 성분 소멸은 u>0에서 미증명이다.
   수 보존과 n0<=N을 쓰면 limsup <n0(t)> <= N0-2*p2만 보장된다.

수치 적분과 유한 별형 환경은 위 무한 시간 증명의 교차 검산이다.
초기 분할·진공 환경·J·E_*는 외부 입력이며, 준비의 자율 구현,
환경 에너지 환류, 불일치 좌표 사상 및 공통 계량 선택은 미완성이다.

반복 분할 확장(선형 환경만):
실제 원초 변의 소유자 수 n=3^D이며, 매 단계의 자식 수3과 구별한다.
모든 쌍 차이 행렬 B는 B^T B=nP이고 P는 공통 방향의 직교 여공간이다.
J_pair=κ*x*exp(-x), 시스템 항 εI+cκB^TB이면 Schur 계수는
ε+(c-1)K, K=nκ이다. c<1에서 충분큰 n은 음의 고유값을 갖는다.
c=1의 양의 완전제곱은 추가 가정이며, ε은 전체 연산자의 간격이 아니다.

c=1에서 각 유한 K의 endpoint 에너지는 ε+K, 연속 지지는 여전히 [0,∞).
국소 방출은 성립하지만 고정 tau에서 K->∞이면 |A_K(tau)|²->1이다.
quasimode_bound의 정확한 v 모멘트와 Duhamel 부등식이 이 결론을 준다.
따라서 고정 시간의 균일 방출이나 모든 단계의 정렬은 증명되지 않는다.
시간 하한은 sqrt(K) 차수 이상으로 자라며 정확한 이완시간의 예측은 아니다.

양자 초기 상태는 recursive_source_maps(3,D)를 별도로 공급한다.
B_D=nD/2-3n/16+3/(16n), N_common=(n+1/n-2)/4,
N_contrast=B_D-N_common, E_init/E_*=ε B_D+κ n N_contrast.
진공 환경에서 N_contrast(tau)=|A_K(tau)|² N_contrast(0).
이는 모든 분할 준비 뒤의 진화이며 분할과 방출을 교대로 하는 프로토콜은 미해결이다.
기하 분할 자체가 이 양자 상태나 모든 쌍 결합 작용을 유도하지는 않는다.
교대 프로토콜의 장부(추가 외부 제어 조건):
각 3진 분할을 결합 off에서 수행하고, 새 진공 환경을 공급하여 c=1 결합을
on -> tau만큼 진화 -> off한다. 그 뒤 나온 환경은 재사용하거나 초기화하지
않고 보관한다. 새 환경·외부 펌프·시계와 K/n 쌍 결합은 추가 입력이다.
off 에너지는 E_*[epsilon sum n_i + 환경 에너지]로 normal ordering하며,
준비 펌프의 일은 epsilon*(N_after_split-N_before) E_*이다.

한 차이 채널의 e0=epsilon+K, A=<0|exp(-ih tau)|0>, Z=i*dA/dtau라 두면,
i*A'=e0*A+sqrt(K)<v,f> 및 <h>=e0에서 다음이 정확히 따른다.
  interaction/C = 2 Re(A* Z)-2 e0 |A|^2
  bath/C = b = e0(1+|A|^2)-2 Re(A* Z)
  W_on/C = K, W_off/C = (2epsilon+K)|A|^2-2 Re(A* Z).
여기서 *는 복소켤레이고 모든 에너지/일은 E_*로 나눴다.
따라서 W_on+W_off=Delta E_sys+E_bath. H가 수보존 이차형이므로
실제 압축 다입자 상태에서도 이 평균 에너지는 초기 차이 점유 C만 쓴다.
이는 표준 경계 전환 일(arXiv:1610.01829v3 식44,46)을 이 상태에 적용한
장부이며, CE 자율 작용이나 장치의 최소일·구현 비용 전체를 도출하지 않는다.

공분산 축약은 일반 입력이 아니라 대칭 상태 클래스의 귀납이다.
F=I_m tensor 1_3/sqrt(3), F^T F=I, F u_m=u_3m이므로 자식 기저
[F O_parent,G_new]에서 모든 상속 block은 S V S^T,
S=diag(sqrt(3),1/sqrt(3)), 새 2m개 block은 diag(3/2,1/6)이다.
공통 block은 exp(-i epsilon tau)의 실제 위상으로 회전하며,
나머지는 R(A)V R(A)^T+(1-|A|^2)I/2로 간다. 각 block의 q-p 상관도
운반한다. tau=0일 때만 기존 recursive_source_maps의 상태와 일치한다.

T_d=(sum contrast multiplicity*V)/3^d, B=R(A)S, s=|A|^2이면
T_d=B T_(d-1) B^T/3+(1-s)(1-3^-d)I/2
    +(2/3)R(A)diag(3/2,1/6)R(A)^T.
선형항의 노름은 s<1이므로 유일한 극한 공분산이 있다. 공통 block의
R(exp(-i epsilon tau))S/sqrt(3)의 spectral radius<1일 때에만
그 공통 에너지/잎수가 0이므로 반환된 계 에너지 밀도 극한을 쓸 수 있다.

K>0,tau>0에서는 연속 지지의 순수 절대연속 측도가 |A|<1을 주고,
f는 0이 아니며 환경 에너지 x>0이므로 b>0이다. 각 단계에서 새 차이
모드가 2*3^(d-1)개, 각 점유1/3이므로 C_d>=2*3^d/9.
진공 시작에서 누적 외부일/E_*=epsilon*N_final+sum b*C_d
>=2*b*3^D/9이다. 방출을 끼워도 에너지를 환류하지 않는 이 프로토콜은
같은 유한 배터리로 모든 깊이를 구현할 수 없다. 방출된 상태를 다시
초기화하는 비용이나 병합·재사용은 계산한 것으로 취급하지 않는다.

"""

from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expi
from scipy.optimize import brentq

from split_quantum_source import frontier_resource_spectrum, mode_basis, source_dilation


def parameters(epsilon, kappa):
    epsilon, kappa = float(epsilon), float(kappa)
    if not (math.isfinite(epsilon) and math.isfinite(kappa) and epsilon > kappa > 0):
        raise ValueError("finite epsilon > kappa > 0 is required")
    return epsilon, kappa


def spectral_density(energy, epsilon, kappa):
    """(z-h)^-1의 -Im/pi. Ei는 실수 주값; 출력은 dx에 대한 밀도이다.

    정확한 밀도는 모든 E>0에서 양수이나, 큰 E의 지수 꼬리는 double에서
    0으로 반올림될 수 있다. 이는 물리적 에너지 상한이나 정확한 영점이 아니다.
    무한계 증명에는 모듈 설명의 정확한 식을 쓰며 수치 반환값을 쓰지 않는다.
    """
    epsilon, kappa = parameters(epsilon, kappa)
    x = float(energy)
    if not math.isfinite(x) or x < 0:
        raise ValueError("energy must be finite and nonnegative")
    if x == 0:
        return 0.0
    if x < 500:
        scaled_ei = math.exp(-x) * float(expi(x))
    else:
        # exp(-x)*Ei(x)의 큰 x 전개로 inf*0을 피한다.
        term = total = 1.0
        for order in range(1, 10):
            term *= order / x
            total += term
        scaled_ei = total / x
    j = kappa * (x * math.exp(-x))
    real_part = x - epsilon - kappa * (x * scaled_ei - 1)
    length = math.hypot(real_part, math.pi * j)
    return (j / length) / length


def spectral_moment(epsilon, kappa, power=0):
    epsilon, kappa = parameters(epsilon, kappa)
    if isinstance(power, bool) or power not in (0, 1, 2):
        raise ValueError("only moments zero, one and two are checked")
    edges = sorted(set((0., .5, 1., 2., 4., 8., 16., 32., epsilon)))
    edges.append(math.inf)
    values = [quad(lambda x: x**power * spectral_density(x, epsilon, kappa),
                   a, b, epsabs=1e-11, epsrel=1e-11, limit=150)
              for a, b in zip(edges, edges[1:])]
    return float(sum(v for v, _ in values)), float(sum(e for _, e in values))


def boundary_amplitude(time, epsilon, kappa):
    epsilon, kappa = parameters(epsilon, kappa)
    time = float(time)
    if not math.isfinite(time) or time < 0:
        raise ValueError("time must be finite and nonnegative")
    if time == 0:
        value, error = spectral_moment(epsilon, kappa)
        return complex(value), error
    options = dict(a=0., b=math.inf, wvar=time, epsabs=1e-11, limit=150, limlst=150)
    real, real_error = quad(lambda x: spectral_density(x, epsilon, kappa),
                            weight="cos", **options)
    imag, imag_error = quad(lambda x: spectral_density(x, epsilon, kappa),
                            weight="sin", **options)
    return complex(real, -imag), real_error + imag_error


def pair_density(energy, epsilon, kappa):
    epsilon, kappa = parameters(epsilon, kappa)
    energy = float(energy)
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("pair energy must be finite and nonnegative")
    return quad(lambda x: spectral_density(x, epsilon, kappa) *
                spectral_density(energy-x, epsilon, kappa),
                0., energy, epsabs=1e-13, epsrel=1e-11, limit=150)


def source_budget(children, epsilon, kappa, interaction):
    """실제 분할 상태의 준비 에너지를 환경과 같은 작용에서 계산한다."""
    epsilon, kappa = parameters(epsilon, kappa)
    u = float(interaction)
    if not math.isfinite(u) or u < 0:
        raise ValueError("interaction must be finite and nonnegative")
    dilation = source_dilation(children)
    basis = np.kron(mode_basis(children), np.eye(2))
    covariance = basis.T @ (.5 * dilation @ dilation.T) @ basis
    q, p = float(covariance[2, 2]), float(covariance[3, 3])
    number, anomalous = (q+p-1)/2, (q-p)/2
    squeeze = .25 * math.log(q/p)
    p2 = .5 * math.tanh(squeeze)**2 / math.cosh(squeeze)
    return {"children": children, "mean_number": number, "anomalous_moment": anomalous,
            "two_particle_probability": p2, "initial_q_variance": q,
            "initial_p_variance": p,
            "initial_energy_over_Estar": epsilon*number + .5*u*(3*number**2+number),
            "nonlinear_full_source_limsup_number_upper_bound": max(0., number-2*p2),
            "linear_asymptotic_q_variance": .5}


def _refinement_functions():
    """기존 코드의 두 조합론 함수만 읽어 기하·헤세 실행부와 분리한다."""
    path = Path(__file__).with_name("F-01") / "predict_fold_budget.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("refine", "gluing_rows")]
    if len(selected) != 2:
        raise RuntimeError("the reference topology functions are missing")
    namespace = {"np": np, "combinations": combinations}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["refine"], namespace["gluing_rows"]


def refinement_owner_counts(depth):
    """실제 1->5 분할에서 원초 변의 소유자 수를 센다. 양자 채널은 별도 가정이다."""
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 4:
        raise ValueError("direct topology check supports depths one through four")
    refine, _ = _refinement_functions()
    vertices = np.vstack((np.zeros(4), np.linalg.cholesky(np.eye(4)+np.ones((4, 4)))))
    points = {i: vertices[i] for i in range(5)}
    cells = [tuple(range(5))]
    rows = []
    for level in range(1, depth+1):
        cells = refine(cells, points)
        counts = Counter(tuple(sorted(edge)) for cell in cells for edge in combinations(cell, 2))
        if counts[(0, 1)] != 3**level or max(counts.values()) != 4*3**(level-1):
            raise RuntimeError("reference refinement disagrees with owner multiplicity")
        rows.append({"depth": level, "cells": len(cells), "old_edge_owners": counts[(0, 1)],
                     "max_owners": max(counts.values()),
                     "owner_histogram": dict(sorted(Counter(counts.values()).items()))})
    return rows


def collective_stability(owners, epsilon, pair_strength, counterterm=1.):
    if isinstance(owners, bool) or not isinstance(owners, int) or owners < 2:
        raise ValueError("owners must be an integer >= 2")
    epsilon, pair_strength, counterterm = map(float, (epsilon, pair_strength, counterterm))
    if not all(map(math.isfinite, (epsilon, pair_strength, counterterm))) or min(epsilon, pair_strength) <= 0 or counterterm < 0:
        raise ValueError("finite positive epsilon and pair strength, and nonnegative counterterm required")
    strength = owners*pair_strength
    coefficient = epsilon+(counterterm-1)*strength
    return {"owners": owners, "collective_strength": strength,
            "endpoint_energy": epsilon+counterterm*strength,
            "schur_coefficient": coefficient,
            "negative_mode_present": coefficient < 0,
            "nonnegative_for_every_owner_count": counterterm >= 1,
            "schur_coefficient_is_full_spectral_gap": False}


def quasimode_bound(strength, epsilon, time, target_survival=.5):
    """무한 환경의 노름 부등식을 대입한다. 유한 대각화에서 추정한 하한이 아니다.

    v(x)=sqrt(x exp(-x)), psi=(1,v/sqrt(K))/sqrt(1+1/K),
    lambda=K+epsilon+1. 잔차제곱=((epsilon-1)^2+2)/(K+1).
    e0와 psi의 거리 d 및 잔차 r이면 진화의 위상 제거 노름은 2d+tau*r 이하.
    따라서 |A|^2 >= max(0,1-(2d+tau*r)^2/2)^2.
    각 유한 K의 장시간 소멸과 고정 시간의 K->infinity 생존은 양립한다.
    """
    strength, epsilon, time, target_survival = map(float, (strength, epsilon, time, target_survival))
    if not all(map(math.isfinite, (strength, epsilon, time, target_survival))) or min(strength, epsilon) <= 0 or time < 0 or not 0 < target_survival < 1:
        raise ValueError("invalid quasimode parameters")
    overlap = math.sqrt(strength/(strength+1))
    distance = math.sqrt(2/((strength+1)*(1+overlap)))
    residual = math.hypot(epsilon-1, math.sqrt(2))/math.sqrt(strength+1)
    norm_bound = 2*distance+time*residual
    lower = 0. if norm_bound >= math.sqrt(2) else (1-norm_bound**2/2)**2
    target_distance = math.sqrt(2*(1-math.sqrt(target_survival)))
    return {"collective_strength": strength, "time": time,
            "quasimode_energy": strength+epsilon+1,
            "quasimode_residual_squared": residual**2,
            "initial_distance": distance, "phase_removed_norm_bound": norm_bound,
            "survival_probability_lower_bound": lower,
            "target_survival": target_survival,
            "time_to_target_lower_bound": max(0., (target_distance-2*distance)/residual)}


def recursive_preparation_energy(depth, epsilon, pair_strength):
    """원초 변에 3진 source를 D번 합성한 공급 상태. 단일 n진 source와 다르다."""
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 32:
        raise ValueError("resource evaluation supports depths one through 32")
    epsilon, pair_strength = float(epsilon), float(pair_strength)
    owners = 3**depth
    branch = collective_stability(owners, epsilon, pair_strength)
    total = frontier_resource_spectrum(3, depth)[3]
    common = (Fraction(owners)+Fraction(1, owners)-2)/4
    contrast = total-common
    return {"depth": depth, "branching_per_step": 3, "owners": owners,
            "total_number_exact": str(total), "total_number": float(total),
            "common_number_exact": str(common), "contrast_number_exact": str(contrast),
            "contrast_number": float(contrast),
            "bare_energy_over_Estar": float(epsilon*total),
            "counterterm_energy_over_Estar": float(branch["collective_strength"]*contrast),
            "total_energy_over_Estar": float(epsilon*total+branch["collective_strength"]*contrast),
            "quantum_source_derived_from_geometric_refinement": False}


def _negative_mode(epsilon, strength):
    """추가항 없는 선형 모형의 음의 고유값. strength>epsilon일 때만 호출한다."""
    if not strength > epsilon > 0:
        raise ValueError("negative-mode control needs strength > epsilon > 0")
    def denominator(z):
        if z == 0:
            return strength-epsilon
        sigma = -strength*quad(lambda x: x*math.exp(-x)/(x-z), 0., math.inf,
                               epsabs=1e-12, epsrel=1e-12)[0]
        return z-epsilon-sigma
    energy = brentq(denominator, -(epsilon+strength+1), 0., xtol=1e-13)
    return {"epsilon": epsilon, "strength": strength, "energy": energy,
            "secular_residual": abs(denominator(energy))}


def refinement_check():
    # 소유자 수와 단계당 자식 수를 구분한다. 원초 변의 실제 가지 수는 3이다.
    epsilon, pair_strength = 2., .5
    nodes, weights = np.polynomial.laguerre.laggauss(64)
    v = np.sqrt(nodes*weights)
    rows = []
    for depth in range(1, 9):
        owners = 3**depth
        strength = owners*pair_strength
        bound = quasimode_bound(strength, epsilon, 1.)
        h = np.diag(np.r_[epsilon+strength, nodes])
        h[0, 1:] = h[1:, 0] = math.sqrt(strength)*v
        psi = np.r_[1., v/math.sqrt(strength)]/math.sqrt(1+1/strength)
        residual2 = float(np.linalg.norm(h@psi-bound["quasimode_energy"]*psi)**2)
        residual_error = abs(residual2-bound["quasimode_residual_squared"])
        energies, vectors = np.linalg.eigh(h)
        survival = float(abs(np.sum(vectors[0]**2*np.exp(-1j*energies)))**2)
        if residual_error > 1e-10 or survival < bound["survival_probability_lower_bound"]-1e-10:
            raise RuntimeError("quasimode cross-check failed")
        rows.append({"depth": depth, "owners": owners, "infinite_bound": bound,
                     "finite_64_mode_survival": survival, "residual_squared_error": residual_error,
                     "recursive_source": recursive_preparation_energy(depth, epsilon, pair_strength)})
    normalized_strength = 1.5
    amplitude, error = boundary_amplitude(1., epsilon+normalized_strength, normalized_strength)
    return {"scope": "supplied linear pair baths after all recursive source preparation",
            "epsilon": epsilon, "pair_strength": pair_strength,
            "topology_source": "F-01/predict_fold_budget.py:refine",
            "topology": refinement_owner_counts(4), "rows": rows,
            "bare_negative_control": _negative_mode(epsilon, 4.5),
            "normalized_control": {"pair_strength_rule": "1.5/owners",
                "collective_strength": normalized_strength, "survival_at_time_one": abs(amplitude)**2,
                "quadrature_error_estimate": error, "normalization_physically_derived": False},
            "conditional_limits": {"strength_then_time": 1., "time_then_strength": 0.},
            "limit_order_note": "inner limit named first; survival is a fraction of initial contrast occupation",
            "complete_square_action_supplied": True,
            "full_hamiltonian_has_positive_gap": False,
            "uniform_fixed_time_emission_proved": False,
            "finite_bath_is_infinite_time_proof": False,
            "interleaved_work": interleaved_source(),
            "interleaved_splitting_and_emission_solved": False,
            "autonomous_preparation_or_recycling_derived": False,
            "common_metric_selection_proved": False}


def collision_response(epsilon=2., strength=1.5, time=1.):
    """양의 완전제곱 결합을 켠 한 충돌의 진폭·에너지·전환 일을 구한다.

    반환 에너지와 일은 E_* 및 충돌 전 차이 점유로 나눈 값이다.
    적분은 모멘트 검사를 통과한 수치 평가이며 엄밀한 오차 인증이 아니다.
    """
    epsilon, strength, time = float(epsilon), float(strength), float(time)
    if not (math.isfinite(epsilon) and epsilon > 0 and
            math.isfinite(strength) and strength > 0 and
            math.isfinite(time) and time >= 0):
        raise ValueError("finite epsilon,strength > 0 and time >= 0 are required")
    endpoint, strength = parameters(epsilon + strength, strength)
    if time == 0:
        amplitude, first, error, moment_error = 1.+0j, complex(endpoint), 0., 0.
    else:
        mass, mass_error = spectral_moment(endpoint, strength)
        mean, mean_error = spectral_moment(endpoint, strength, 1)
        moment_error = max(abs(mass-1), abs(mean/endpoint-1))
        if moment_error > 1e-9 or mass_error > 1e-8 or mean_error/endpoint > 1e-8:
            raise ArithmeticError("continuum quadrature did not resolve the spectral measure")
        amplitude, amplitude_error = boundary_amplitude(time, endpoint, strength)
        options = dict(a=0., b=math.inf, wvar=time, epsabs=1e-11, limit=150, limlst=150)
        real, real_error = quad(lambda x: x*spectral_density(x, endpoint, strength),
                                weight="cos", **options)
        imag, imag_error = quad(lambda x: x*spectral_density(x, endpoint, strength),
                                weight="sin", **options)
        first, error = complex(real, -imag), amplitude_error+real_error+imag_error
    survival = abs(amplitude)**2
    overlap = (amplitude.conjugate()*first).real
    bath = endpoint*(1+survival)-2*overlap
    interaction = 2*overlap-2*endpoint*survival
    off = (2*epsilon+strength)*survival-2*overlap
    if time > 0 and not (0 <= survival < 1 and bath > 0):
        raise ArithmeticError("finite-time bath response is not numerically resolved")
    return {"epsilon": epsilon, "collective_strength": strength, "time": time,
            "amplitude": [amplitude.real, amplitude.imag],
            "first_energy_amplitude": [first.real, first.imag],
            "survival_probability": survival,
            "bath_energy_per_initial_number": bath,
            "interaction_energy_per_initial_number": interaction,
            "switch_on_work_per_initial_number": strength,
            "switch_off_work_per_initial_number": off,
            "net_switch_work_per_initial_number": strength+off,
            "quadrature_error_estimate": error, "moment_relative_residual": moment_error}


def _quadrature_map(amplitude):
    return np.array([[amplitude.real, -amplitude.imag],
                     [amplitude.imag, amplitude.real]])


def interleaved_source(depth=12, epsilon=2., strength=1.5, time=1.):
    """실제 3진 분할과 새 진공 충돌을 교대로 실행하는 외부 제어 장부.

    모든 차이 채널이 동일하고 부모·보조 상태가 명시한 대칭 클래스에
    있을 때에만 세대별 2x2 공분산 축약이 성립한다. 쌍별 strength/n,
    고정 실험실 q축, 자유 공통 위상, 새 환경과 외부 펌프를 공급한다.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 32:
        raise ValueError("interleaved depth must be an integer from 0 to 32")
    response = collision_response(epsilon, strength, time)
    epsilon, strength, time = response["epsilon"], response["collective_strength"], response["time"]
    amplitude = complex(*response["amplitude"])
    survival = response["survival_probability"]
    rotate = _quadrature_map(amplitude)
    common_rotate = _quadrature_map(complex(np.exp(-1j*epsilon*time)))
    squeeze = np.diag([math.sqrt(3), 1/math.sqrt(3)])
    newborn = np.diag([1.5, 1/6])
    noise = (1-survival)*np.eye(2)/2
    common = np.eye(2)/2
    cohorts, rows = [], []
    previous_number = total_work = total_bath = 0.
    number = lambda covariance: float((np.trace(covariance)-1)/2)
    for level in range(1, depth+1):
        parents, owners = 3**(level-1), 3**level
        common = squeeze @ common @ squeeze.T
        cohorts = [(count, squeeze @ covariance @ squeeze.T)
                   for count, covariance in cohorts]
        cohorts.append((2*parents, newborn.copy()))
        contrast_before = sum(count*number(covariance) for count, covariance in cohorts)
        number_before = number(common)+contrast_before
        source_work = epsilon*(number_before-previous_number)
        on_work = strength*contrast_before
        off_work = response["switch_off_work_per_initial_number"]*contrast_before
        emitted = response["bath_energy_per_initial_number"]*contrast_before
        common = common_rotate @ common @ common_rotate.T
        cohorts = [(count, rotate @ covariance @ rotate.T+noise)
                   for count, covariance in cohorts]
        contrast_after = sum(count*number(covariance) for count, covariance in cohorts)
        previous_number = number(common)+contrast_after
        system_energy = epsilon*previous_number
        total_work += source_work+on_work+off_work
        total_bath += emitted
        residual = total_work-total_bath-system_energy
        rows.append({"depth": level, "owners": owners, "pair_strength": strength/owners,
                     "contrast_number_before": contrast_before,
                     "contrast_number_after": contrast_after,
                     "source_work_over_Estar": source_work,
                     "switch_on_work_over_Estar": on_work,
                     "switch_off_work_over_Estar": off_work,
                     "emitted_energy_over_Estar": emitted,
                     "system_energy_over_Estar": system_energy,
                     "system_energy_per_owner": system_energy/owners,
                     "total_external_work_over_Estar": total_work,
                     "total_emitted_energy_over_Estar": total_bath,
                     "external_work_lower_bound_over_Estar":
                         response["bath_energy_per_initial_number"]*2*owners/9,
                     "balance_relative_residual": abs(residual)/max(1., abs(total_work)),
                     "contrast_number_identity_residual":
                         abs(contrast_after-survival*contrast_before)/max(1., contrast_before)})
    limiting = None
    if survival < 1:
        inherited = rotate @ squeeze/math.sqrt(3)
        forcing = noise+(2/3)*rotate @ newborn @ rotate.T
        average = np.linalg.solve(np.eye(4)-np.kron(inherited, inherited),
                                  forcing.ravel()).reshape(2, 2)
        common_radius = float(max(abs(np.linalg.eigvals(common_rotate @ squeeze/math.sqrt(3)))))
        contrast_limit = float((np.trace(average)-1)/2)
        limiting = {"contrast_covariance_per_owner": average.tolist(),
                    "contrast_number_per_owner": contrast_limit,
                    "contraction_upper_bound": survival,
                    "common_scaled_spectral_radius": common_radius,
                    "system_energy_per_owner_if_common_vanishes": epsilon*contrast_limit,
                    "common_vanishes_per_owner": common_radius < 1-1e-12,
                    "lyapunov_residual": float(np.max(abs(
                        average-inherited @ average @ inherited.T-forcing)))}
    return {"scope": "externally driven three-way split and fresh vacuum collision",
            "coupling_rule": "pair strength=constant collective strength/owners",
            "response": response, "rows": rows, "limiting": limiting,
            "final_common_covariance": common.tolist(),
            "final_contrast_cohorts": [{"multiplicity": count, "covariance": covariance.tolist()}
                                       for count, covariance in cohorts],
            "source_axes": "fixed laboratory quadratures; free common phase is retained",
            "vacuum_reference": "normal ordered epsilon sum a_dagger a; initially all modes vacuum",
            "minimum_external_work_per_owner":
                2*response["bath_energy_per_initial_number"]/9,
            "bath_energy_recycled": False, "bath_reuse_solved": False,
            "phase_corrected_to_preserve_unattenuated_source": False,
            "pump_clock_and_reservoir_autonomous": False,
            "physical_coupling_normalization_derived": False,
            "common_metric_selection_proved": False,
            "source_reference": "Strasberg et al., arXiv:1610.01829v3, equations 44 and 46"}


def run():
    cases = []
    for kappa in (.5, 1.5):
        epsilon = 2.
        moments = [spectral_moment(epsilon, kappa, p) for p in range(3)]
        targets = (1., epsilon, epsilon**2+kappa)
        residual = max(abs(value-target) for (value, _), target in zip(moments, targets))
        if residual > 1e-9:
            raise RuntimeError("spectral moments disagree with the Hamiltonian")
        source = source_budget(3, epsilon, kappa, 0.)
        times = []
        for time in (0., .5, 1., 2., 5., 10., 20.):
            amplitude, error = boundary_amplitude(time, epsilon, kappa)
            times.append({"time_Estar_t_over_hbar": time,
                          "amplitude_real": amplitude.real, "amplitude_imag": amplitude.imag,
                          "quadrature_error_estimate": error,
                          "linear_mean_number": source["mean_number"]*abs(amplitude)**2,
                          "linear_q_variance": .5+source["mean_number"]*abs(amplitude)**2+
                          (source["anomalous_moment"]*amplitude**2).real})
        cases.append({"epsilon": epsilon, "kappa": kappa, "spectral_moments": moments,
                      "moment_residual": residual, "source_linear": source,
                      "source_nonlinear_u8": source_budget(3, epsilon, kappa, 8.),
                      "linear_time_samples": times})
    here = Path(__file__).resolve()
    dependencies = (here, here.with_name("split_quantum_source.py"),
                    here.with_name("interface_bath.py"), here.parent / "F-01/predict_fold_budget.py")
    return {"scope": "supplied positive unbounded continuum and actual squeezed contrast",
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "source_sha256": {p.relative_to(here.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
            "energy_unit": "E_star", "dimensionless_time": "E_star*t/hbar",
            "conditional_results": {
                "linear_full_source_local_number_limit": 0.,
                "repulsive_two_particle_local_number_limit": 0.,
                "k3_nonlinear_full_source_limsup_number_upper_bound": 1/3-math.sqrt(3)/8},
            "proof_dependency": "Kato 1957 Theorem 1 plus the spectral and local-decay argument in module docstring",
            "proof_reference": "https://doi.org/10.3792/pja/1195525063",
            "finite_numerics_prove_infinite_time_limit": False,
            "nonlinear_full_source_complete_emission_proved": False,
            "zero_occupation_means_zero_coordinate_variance": False,
            "source_preparation_and_energy_recycling_derived": False,
            "microscopic_coupling_derived_from_CE": False,
            "common_metric_selection_proved": False, "cases": cases,
            "refinement": refinement_check()}


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
