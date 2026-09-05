"""방출 환경의 기억과 양의 배터리로의 조건부 에너지 이동을 검산한다.

진전 원장 §2의 같은 상태·작용에서 준비·병합·환류를 계산하는 의무에 연결된다.
이 모듈은 공급된 연속 에너지 표현의 구성 정리이며 CE 물리 후보를 채택하지 않는다.
에너지 x_i,b는 E_*로, 시간 tau는 hbar/E_*로 정규화한다.

1. 같은 환경의 기억
실수 Friedrichs h에서 U=exp(-ih tau), U^T=U이고 u=Ue0=(A,f)이다.
환경에 위상 exp(2i arg A-2i arg f(x))를 곱하면 u'=exp(2i arg A)conj(u).
따라서 Uu'=exp(2i arg A)e0. 영점 위상은 임의로 정한다.
이 위상 조작은 M_x와 가환하여 결합 off의 환경 에너지를 바꾸지 않는다.
그러나 모든 주파수의 위상과 접속 시각을 공급한 반향 대조다.
수보존 Fock 확장에서 원래 압축 상태는 알려진 모드 위상까지 회복된다.
이는 계의 냉각도 되돌리므로, 같은 환경을 새 진공으로 취급할 수 없다.

2. 양의 배터리와 전 에너지 보존
배터리는 L2(R_+,db), H_bat=M_b이다. N개 환경 입자의 x_i>0에 대해
X=sum x_i, E=X+b, c=alpha E/(b+alpha X), alpha>0라 둔다.
  x_i'=c*x_i, b'=b*E/(b+alpha X).
이 변환은 E와 입자 에너지 비율을 보존하며 log(X/b)를 log(alpha)만큼 옮긴다.
양의 공간의 역변환은 alpha -> 1/alpha이고 Jacobian은
  J_N=alpha^N [E/(b+alpha X)]^(N+1).
새 좌표에서 psi'=psi/sqrt(J_N)로 정의하면 각 대칭 N입자 공간의 unitary다.
진공 N=0에서는 배터리에 항등으로 작용한다. 이들의 직합은 모든 입자 수
성분과 성분 사이의 결맞음에 작용하며 입자 수를 측정하거나 버리지 않는다.
따라서 전체 Fock 공간에서 U는 H0=dGamma(M_x)+M_b와 강하게 가환한다.

0<alpha<1이면 X'<X, b'>b이며 X'+b'=X+b. 계 S에는 작용하지 않으므로
S 주변상태는 정확히 보존한다. 반복 m회는 alpha^m을 대입한 변환이다.
유한 평균 H0에서 0<=X_m<=X+b와 지배수렴으로 <X_m>->0,
<b_m>-><X+b>이다. 배터리 에너지는 초기 총 평균에너지로 유계다.
그러나 입자 수와 S 대 (환경+배터리)의 상관은 보존된다. 진공 초기화가 아니다.
배터리와 환경의 상관은 변할 수 있으므로 저장 에너지를 재사용 가능한 일로
동일시하지 않는다. 정상화된 배터리 상태는 b>0 위에 놓이며 b=0 에너지
고유상태나 유한 차원 배터리를 공급한 구성은 아니다.

3. 시간독립 연산자의 존재와 물리 경계
theta=Arg(U), ||theta||<=pi라 하면 G=-theta/tau는 유계 자기수반이며
H0와 강하게 가환한다. H=H0+G+pi/tau>=0는 시간독립 자기수반이고
exp(-i tau H)=-exp(-i tau H0)U. 이는 스펙트럼 함수로 구성한 비국소
연산자의 존재일 뿐이다. CE의 국소 미시 작용, 인과성, 시계 반환,
배터리의 낮은 엔트로피·다음 분할 구동은 증명하지 않는다.
자유 진화를 포함하면 S에는 그 자유 진화만 남는다.

4. 실제 3진 압축 상태의 모든 입자 수 평균
한 차이 모드의 입자 수 생성함수는 G0(z)=sqrt(3)/sqrt(4-z^2).
동일한 두 차이 모드의 곱은 G0(z)^2=3/(4-z^2)이다.
한 충돌 뒤 계 잔류확률 s=|A|^2, 환경 파동묶음 확률 eta=1-s,
정규화된 환경 에너지분포 p_f(x)의 Laplace 변환 L_f를 쓰면
전체 두 환경의 X에 대해 L_X(t)=3/[4-(s+eta L_f(t))^2].
고정 배터리 에너지 b에서 Delta b=(1-alpha)bX/(b+alpha X)이므로
  <Delta b|b>=(1-alpha)b/alpha integral_0^infinity exp(-y)
                 [1-L_X(alpha*y/b)]dy.
초기 배터리 파동함수는 [1,2]에서1, 그 밖에서0으로 공급한다.
이 정상화된 순수 상태의 에너지 밀도로 b를 적분하며 입자 수를 절단하지 않는다.
에너지 관측량이 대각이므로 이 평균을 확률분포로 계산해도 결맞음을
지웠다는 뜻은 아니다. 실제 변환 뒤 상태는 일반적으로 가우시안이 아니다.

유한 별형 행렬은 파동묶음의 수치 적분과 기억 대조에만 사용한다.
위 에너지 변환은 연속 공간에 정의되며 유한 별형 스펙트럼의 unitary가 아니다.

5. 같은 자원에서 다음 분할을 준비하는 경계
새 차이 모드 T 두 개는 진공에서 시작하고 H_T=2(N_1+N_2)이다.
목표는 실제 3진 분할의 두 압축 모드이며 P(N_1+N_2=2m)=3/4^(m+1).
따라서 P(H_T>=4m)=4^(-m). 기존 계 S는 이 준비 연산에 참여하지 않는다.

자원 R은 방출 환경 전부와 [1,2] 배터리이고 H_R>=0이다. 환경 진공 확률은
p0=3/(4-s^2), s=|A|^2. 진공 환경 성분은 자원 총에너지가 2 이하이므로
P(H_R>=4)<=1-p0. [U,H_R+H_T]=0인 임의의 유니터리에 대해, 새 모드의
진공 부분공간으로 압축하면
 P_vac U^dagger Pi_(H_T>=4) U P_vac <= P_vac Pi_(H_R>=4) P_vac.
이 연산자 부등식은 자원과 기존 계의 얽힘이나 에너지 결맞음을 버리지 않는다.
따라서 D(rho_T,chi)>=p0-3/4=3s^2/[4(4-s^2)].
D는 1/2 trace norm이고 chi는 순수 목표 상태다. 회수 alpha 또는 환경 직접
접근으로 이 하한을 지울 수 없다. 추가 일, 기존 계의 에너지, 들뜬 보조계를
허용하면 자원 장부를 다시 세워야 하며 이 경계를 그대로 적용하지 않는다.

분산 Var_h(e0)=K인 스펙트럼 측도와 cos y>=1-y^2/2를 쓰면
 |A(tau)|>=max(0,1-K*tau^2/2).
K=3/2,tau=1에서는 s>=1/16, 따라서 D>=1/1364. 이 부등식은
적분 오차 추정이 필요 없는 해석적 하한이다. 실제 s의 수치 대입은 별도 산출이다.

회수 배터리만 쓰면 더해지는 경계도 있다. 유한 alpha>0에서 b'<2/alpha.
M=ceil(1/(2alpha))이면 목표 에너지 4M 이상의 확률 4^(-M)을 만들 수 없다.
따라서 D>=4^(-M), squared fidelity F<=1-4^(-M). 이 상한의 달성은 주장하지 않는다.

독립 환경 L개와 한 배터리를 공동으로 쓰면 같은 진공 경계는
D>=max(0,p0^L-3/4)다. 경계가 0이어도 준비 가능성을 증명하지 않는다.
새 압축쌍 M개를 동시에 준비하면 목표의 H_T>=4 확률은 1-(3/4)^M이다.
따라서 D>=max(0,p0^L-(3/4)^M). 정확 준비의 필요조건은
M/L<=log(p0)/log(3/4). s>0이면 우변은 1보다 작으므로 모든 유한 L에서
M=L 교체와 M=3L 분기 증식은 이 자원만으로 불가능하다.
일반 출력에는 홀수 입자도 있을 수 있으므로 출력 진공 확률을 가정하지 않고
H_T>=4 사영자로 비교한다. 이 경계는 고정 허용 오차의 점근 용량 정리가 아니다.
아래 표본 계산은 L=1,2,4에서 첫 여섯 에너지 꼬리의 필요조건만 조사한다.
입자 수는 음이항분포로 전부 포함하지만 환경 주파수는 유한 적분 격자다.
표본 오차와 격자 오차가 있으므로 이 조사는 연속 모형의 증명이나
에너지 결맞음·상관을 보존하는 준비 유니터리의 구성을 대신하지 않는다.


6. 기존 모드를 퇴역시키는 조건부 이전
기존 S가 연산에 참여하면 같은 간격 epsilon=2의 새 진공 모드 T를 둘 수 있다.
W=U D U가 e_S를 exp(i phi)e_S로 보내므로, 뒤에 SWAP_(S,T)와 T의
exp(-i phi N_T)를 붙인 수보존 연산은
 |psi>_S |0>_E |0>_T -> |0>_S |0>_E |psi>_T
를 만족한다. 단입자 열 항등식의 Fock 확장이므로 모든 입자 수와 외부
참조계의 얽힘에도 성립한다. 두 모드의 간격이 같아 SWAP과 위상 보정은
자유 에너지를 보존하며 이상적 전환 일 합은 0이다.
회수가 먼저 적용됐다면 그 전체 유니터리의 정확한 역을 먼저 적용해야 한다.
이때 배터리에 저장했던 에너지도 환경으로 돌아가며 배터리는 초기 상태가 된다.
이는 공급된 위상·스위치·역회수 제어 아래 상태를 이전하는 조건부 구성이다.
S는 진공으로 퇴역하므로 새 압축 상태의 개수가 증가하지 않는다.
자율 시계·제어기의 복귀, CE 국소 작용과 삼진 분기 증식은 여전히 미완성이다.

일과 평균 에너지의 구별 및 결맞음 조건의 배경 원전:
https://arxiv.org/html/1304.1060v3 (식 1,2 및 부록 D)
https://arxiv.org/html/1807.08656v3 (식 3).

이 코드는 계에 남은 에너지의 depth 성장, 환류 후 다음 분할,
공통 계량 또는 0D->3+1 Plebanski/Einstein을 닫지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from continuum_bath import collision_response


def positive_energy_map(energies, battery, alpha):
    """N입자 에너지 좌표의 가역 이동과 Jacobian을 반환한다."""
    x = np.asarray(energies, dtype=float)
    battery, alpha = float(battery), float(alpha)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("bath energies must be a finite positive vector")
    if not (math.isfinite(battery) and battery > 0 and math.isfinite(alpha) and alpha > 0):
        raise ValueError("battery and alpha must be finite and positive")
    total = float(np.sum(x)+battery)
    if not math.isfinite(total):
        raise ValueError("total energy must be finite")
    if not len(x):
        return {"bath": [], "battery": battery, "log_jacobian": 0.}
    bath_fraction, battery_fraction = float(np.sum(x))/total, battery/total
    denominator = battery_fraction+alpha*bath_fraction
    scale = alpha/denominator
    output, stored = scale*x, battery/denominator
    if not np.all(np.isfinite(output)) or np.any(output <= 0) or not math.isfinite(stored) or stored <= 0:
        raise ArithmeticError("energy map exceeds floating point range")
    log_jacobian = len(x)*math.log(alpha)-(len(x)+1)*math.log(denominator)
    return {"bath": output.tolist(), "battery": stored, "log_jacobian": log_jacobian}


def star_state(nodes=128, time=1.):
    """epsilon=2,K=1.5의 연속 환경을 유한 적분 격자로 근사한다."""
    if isinstance(nodes, bool) or not isinstance(nodes, int) or not 8 <= nodes <= 128:
        raise ValueError("quadrature nodes must be an integer from 8 to 128")
    time = float(time)
    if not math.isfinite(time) or time <= 0:
        raise ValueError("time must be finite and positive")
    energies, weights = np.polynomial.laguerre.laggauss(nodes)
    h = np.diag(np.r_[3.5, energies])
    h[0, 1:] = h[1:, 0] = np.sqrt(1.5*energies*weights)
    values, vectors = np.linalg.eigh(h)
    unitary = (vectors*np.exp(-1j*values*time)) @ vectors.T
    return energies, h, unitary, unitary[:, 0]


def memory_control(nodes=128):
    """계–환경 교차 공분산을 유지한 재접속과 위상 반향을 비교한다."""
    energies, h, unitary, first = star_state(nodes)
    off = np.diag(np.r_[2., energies])
    coupling = h-off
    initial = np.zeros(len(first), dtype=complex)
    initial[0] = 1.
    number = 1/3
    expected = lambda matrix, vector: float(np.vdot(vector, matrix @ vector).real)
    source_q = np.column_stack((first.real, first.imag)).ravel()
    source_p = np.column_stack((-first.imag, first.real)).ravel()
    covariance = np.eye(2*len(first))/2+np.outer(source_q, source_q)-np.outer(source_p, source_p)/3
    on1, off1 = number*expected(coupling, initial), -number*expected(coupling, first)
    rows = []
    for delay in (0., .5, 1., 2., 4.):
        remembered = np.exp(-1j*np.diag(off)*delay)*first
        final = unitary @ remembered
        on2, off2 = number*expected(coupling, remembered), -number*expected(coupling, final)
        work = on1+off1+on2+off2
        rows.append({"off_delay": delay, "system_number": number*abs(final[0])**2,
                     "second_switch_on_work_over_Estar": on2,
                     "balance_residual": abs(work-number*(expected(off, final)-2.))})
    phased = first.copy()
    phased[1:] *= np.exp(2j*(np.angle(first[0])-np.angle(first[1:])))
    echo = unitary @ phased
    echo_work = on1+off1+number*expected(coupling, phased)-number*expected(coupling, echo)
    return {"nodes": nodes, "first_system_number": number*abs(first[0])**2,
            "first_system_covariance": covariance[:2, :2].tolist(),
            "system_bath_cross_covariance_norm": float(np.linalg.norm(covariance[:2, 2:])),
            "fresh_second_system_number": number*abs(first[0])**4,
            "same_environment": rows,
            "echo_system_number": number*abs(echo[0])**2,
            "echo_remaining_bath_number": number*float(np.sum(abs(echo[1:])**2)),
            "echo_phase_free_energy_change": number*(expected(off, phased)-expected(off, first)),
            "echo_net_switch_work_over_Estar": echo_work,
            "echo_spectral_phase_is_supplied": True,
            "same_bath_is_fresh_vacuum": False}



def retired_transfer_unitary(nodes=128):
    """기존 모드를 퇴역시키며 같은 간격의 새 모드로 옮기는 공급된 제어 연산."""
    energies, _, unitary, first = star_state(nodes)
    size = len(first)
    phase = 2*np.angle(first[0])
    bath_phase = np.ones(size, dtype=complex)
    bath_phase[1:] = np.exp(1j*phase-2j*np.angle(first[1:]))
    echo = unitary @ (bath_phase[:,None]*unitary)
    transfer = np.eye(size+1, dtype=complex)
    transfer[:size,:size] = echo
    # 기존 모드와 새 모드의 행을 교환하고 목표 모드의 알려진 위상을 제거한다.
    transfer[[0,-1],:] = transfer[[-1,0],:]
    transfer[-1,:] *= np.exp(-1j*phase)
    return np.r_[2.,energies,2.], transfer


def retired_source_transfer(nodes=128):
    """단일 모드 이전의 잔차와 이상적 전환 일, 자율 구현의 미완성 범위."""
    energies, transfer = retired_transfer_unitary(nodes)
    expected = np.zeros(len(energies), dtype=complex)
    expected[-1] = 1.
    column = transfer[:,0]
    memory = memory_control(nodes)
    return {"spectral_nodes": nodes,
            "one_particle_transfer_residual": float(np.linalg.norm(column-expected)),
            "unitarity_residual": float(np.linalg.norm(transfer.conj().T @ transfer-np.eye(len(energies)))),
            "retired_source_number": float(abs(column[0])**2/3),
            "remaining_bath_number": float(np.sum(abs(column[1:-1])**2)/3),
            "new_mode_number": float(abs(column[-1])**2/3),
            "free_energy_change_over_Estar": float((np.dot(energies,abs(column)**2)-2)/3),
            "ideal_net_switch_work_over_Estar": memory["echo_net_switch_work_over_Estar"],
            "existing_system_participates": True,
            "reference_entanglement_preserved_by_exact_operator": True,
            "all_fock_sectors_transfer_by_passive_lift": True,
            "battery_return_requires_exact_inverse_if_recovery_was_applied": True,
            "spectral_phase_and_switch_control_supplied": True,
            "additional_source_copies_created": 0,
            "autonomous_clock_controller_cycle_closed": False,
            "local_causal_CE_action_derived": False}


def emitted_packet(nodes=128):
    """두 실제 차이 모드가 공유하는 한 파동묶음의 적분 가중치."""
    energies, _, _, state = star_state(nodes)
    mass = abs(state[1:])**2
    emitted_probability = float(np.sum(mass))
    survival = float(abs(state[0])**2)
    if not 0 < emitted_probability < 1:
        raise ArithmeticError("emitted packet is not resolved")
    return {"energies": energies, "probabilities": mass/emitted_probability,
            "emitted_probability": emitted_probability,
            "survival": survival,
            "norm_residual": abs(survival+emitted_probability-1),
            "two_channel_bath_energy": float((2/3)*np.sum(energies*mass))}


def recovery_energy(alpha, nodes=128, laplace_nodes=64):
    """[1,2] 양의 배터리에 저장되는 두 차이 모드의 전 입자 수 평균 에너지."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("recovery requires finite 0 < alpha <= 1")
    if isinstance(laplace_nodes, bool) or not isinstance(laplace_nodes, int) or not 16 <= laplace_nodes <= 128:
        raise ValueError("Laplace nodes must be an integer from 16 to 128")
    packet = emitted_packet(nodes)
    y, wy = np.polynomial.laguerre.laggauss(laplace_nodes)
    b_nodes, b_weights = np.polynomial.legendre.leggauss(32)
    batteries, b_weights = 1.5+b_nodes/2, b_weights/2
    times = alpha*y[None, :]/batteries[:, None]
    delta = packet["emitted_probability"]*np.sum(
        packet["probabilities"][None, None, :]*(-np.expm1(-times[:, :, None]*packet["energies"])),
        axis=2)
    # 1-G0(1-delta)^2의 안정한 식. 모든 even-N 성분을 합친다.
    one_minus_transform = delta*(2-delta)/(3+delta*(2-delta))
    conditional_gain = (1-alpha)*batteries/alpha*(one_minus_transform @ wy)
    gain = float(b_weights @ conditional_gain)
    incoming = packet["two_channel_bath_energy"]
    if gain < -1e-10 or gain > incoming+1e-8:
        raise ArithmeticError("energy integral violates its conservation bounds")
    return {"alpha": alpha, "spectral_nodes": nodes, "laplace_nodes": laplace_nodes,
            "initial_bath_energy_over_Estar": incoming,
            "battery_energy_gain_over_Estar": gain,
            "remaining_bath_energy_over_Estar": incoming-gain,
            "initial_battery_mean_over_Estar": 1.5,
            "final_battery_mean_over_Estar": 1.5+gain,
            "source_system_energy_over_Estar": 2*(1+2*packet["survival"])/3,
            "bath_vacuum_probability_preserved": 3/(4-packet["survival"]**2),
            "normalization_residual": packet["norm_residual"],
            "photon_number_truncation_used": False,
            "battery_gain_is_reusable_work_proved": False}



def next_split_bounds(alpha=.1, resource_copies=1, survival=None):
    """진공 성분과 에너지 지지집합으로 다음 분할의 필요조건을 계산한다."""
    from fractions import Fraction

    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("회수 인자는 유한한 0 < alpha <= 1이어야 합니다")
    if isinstance(resource_copies, bool) or not isinstance(resource_copies, int) or not 1 <= resource_copies <= 64:
        raise ValueError("독립 환경 개수는 1부터 64까지의 정수여야 합니다")
    survival = emitted_packet()["survival"] if survival is None else float(survival)
    if not math.isfinite(survival) or not 0 <= survival <= 1:
        raise ValueError("생존확률은 0부터 1 사이여야 합니다")
    cap = 2/alpha
    if not math.isfinite(cap):
        raise ArithmeticError("배터리 에너지 상한이 부동소수점 범위를 넘습니다")
    # 이진 부동소수점 입력을 정확한 유리수로 해석하여 천장 함수 경계를 보존한다.
    first_missing_pair = math.ceil(Fraction(1, 2)/Fraction.from_float(alpha))
    tail_log = -first_missing_pair*math.log(4)
    tail = math.exp(tail_log)
    vacuum = (3/(4-survival**2))**resource_copies
    allowed_excitation = 1-vacuum
    distance = max(0., vacuum-.75)
    fidelity = ((math.sqrt(.25*allowed_excitation)+math.sqrt(.75*vacuum))**2
                if allowed_excitation < .25 else 1.)
    return {"alpha": alpha, "independent_resource_copies": resource_copies,
            "survival_probability": survival, "joint_vacuum_probability": vacuum,
            "target_nonvacuum_probability": .25,
            "joint_resource_trace_distance_lower_bound": distance,
            "joint_resource_squared_fidelity_upper_bound": fidelity,
            "battery_energy_strict_upper_bound_over_Estar": cap,
            "battery_only_first_missing_pair": first_missing_pair,
            "battery_only_tail_exact": {"base": 4, "exponent": -first_missing_pair},
            "battery_only_trace_distance_lower_bound_float": tail,
            "battery_only_tail_underflow": tail == 0.,
            "finite_alpha_battery_only_exact_preparation_excluded": True,
            "joint_resource_exact_preparation_excluded_by_vacuum_bound": distance > 0.,
            "zero_bound_proves_preparation": False,
            "existing_system_participates": False,
            "additional_work_or_excited_auxiliaries_supplied": False}


def survival_moment_bound(strength=1.5, time=1.):
    """유한한 이차 모멘트의 코사인 부등식으로 수치 적분 없는 준비 하한을 구한다."""
    strength, time = float(strength), float(time)
    if not (math.isfinite(strength) and strength >= 0 and math.isfinite(time) and time >= 0):
        raise ValueError("결합 세기와 시간은 유한한 음이 아닌 값이어야 합니다")
    centered_variance = strength*time*time
    amplitude = max(0., 1-centered_variance/2)
    survival = amplitude**2
    distance = 3*survival**2/(4*(4-survival**2))
    return {"spectral_variance": strength, "time": time,
            "survival_probability_lower_bound": survival,
            "joint_resource_trace_distance_lower_bound": distance,
            "uses_numerical_spectral_integral": False}



def pooled_replacement_bound(resource_copies, target_copies, survival=None):
    """공동 자원이 같은 수 또는 더 많은 분할 상태를 정확히 재생산할 필요조건."""
    for count in (resource_copies, target_copies):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
            raise ValueError("자원과 목표 개수는 1부터 64까지의 정수여야 합니다")
    survival = emitted_packet()["survival"] if survival is None else float(survival)
    if not math.isfinite(survival) or not 0 <= survival <= 1:
        raise ValueError("생존확률은 0부터 1 사이여야 합니다")
    vacuum = 3/(4-survival**2)
    resource_low = vacuum**resource_copies
    target_low = .75**target_copies
    distance = max(0., resource_low-target_low)
    return {"independent_resource_copies": resource_copies,
            "independent_target_copies": target_copies,
            "survival_probability": survival,
            "resource_joint_vacuum_probability": resource_low,
            "target_joint_vacuum_probability": target_low,
            "trace_distance_lower_bound": distance,
            "exact_preparation_excluded_by_this_bound": distance > 0.,
            "exact_count_ratio_necessary_upper_bound": math.log(vacuum)/math.log(.75),
            "fixed_error_asymptotic_capacity_proved": False,
            "zero_bound_proves_preparation": False,
            "existing_system_participates": False,
            "battery_initial_energy_upper_bound_over_Estar": 2.}


def pooled_tail_probe(resource_copies, trials=250000, nodes=128):
    """독립 방출을 모은 자원의 여섯 에너지 꼬리를 전 입자 수 표본으로 조사한다."""
    if isinstance(resource_copies, bool) or not isinstance(resource_copies, int) or not 1 <= resource_copies <= 4:
        raise ValueError("표본 조사의 독립 환경 개수는 1부터 4까지여야 합니다")
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1000 <= trials <= 250000:
        raise ValueError("표본 수는 1000부터 250000까지의 정수여야 합니다")
    packet = emitted_packet(nodes)
    seed = 512700+resource_copies
    rng = np.random.default_rng(seed)
    pair_count = rng.negative_binomial(resource_copies, .75, size=trials)
    emitted = rng.binomial(2*pair_count, packet["emitted_probability"])
    energies = rng.choice(packet["energies"], size=int(emitted.sum()), p=packet["probabilities"])
    total = np.bincount(np.repeat(np.arange(trials), emitted), weights=energies, minlength=trials)
    total += rng.uniform(1., 2., size=trials)
    rows = []
    for pair_threshold in range(1, 7):
        probability = float(np.mean(total >= 4*pair_threshold))
        stderr = math.sqrt(probability*(1-probability)/trials)
        target = 4.**(-pair_threshold)
        rows.append({"pair_threshold": pair_threshold, "resource_tail": probability,
                     "target_tail": target, "target_minus_resource": target-probability,
                     "standard_error": stderr})
    return {"independent_resource_copies": resource_copies, "trials": trials, "seed": seed,
            "spectral_nodes": nodes, "mean_resource_energy_over_Estar": float(total.mean()),
            "mean_standard_error": float(total.std(ddof=1)/math.sqrt(trials)),
            "expected_mean_resource_energy_over_Estar": 1.5+resource_copies*packet["two_channel_bath_energy"],
            "joint_vacuum_frequency": float(np.mean(emitted == 0)),
            "joint_vacuum_probability": (3/(4-packet["survival"]**2))**resource_copies,
            "gates": rows, "photon_number_truncation_used": False,
            "numerical_necessary_condition_screen_only": True,
            "continuous_spectrum_tail_error_certified": False,
            "coherent_preparation_unitary_constructed": False}


def run():
    here = Path(__file__).resolve().parent
    response = collision_response()
    coarse = [recovery_energy(alpha, nodes=64) for alpha in (.5, .1, .01)]
    fine = [recovery_energy(alpha, nodes=128) for alpha in (.5, .1, .01)]
    integral_fine = recovery_energy(.5, nodes=128, laplace_nodes=128)
    dependencies = [Path(__file__).resolve(), here/"continuum_bath.py",
                    here/"split_quantum_source.py"]
    return {"scope": "conditional Fock energy transfer; supplied nonlocal spectral operator",
            "energy_unit": "E_*", "time_unit": "hbar/E_*",
            "python": platform.python_version(), "numpy": np.__version__,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
            "memory": memory_control(), "recovery": fine,
            "retired_source_transfer": retired_source_transfer(),
            "next_split": {
                "single_resource": [next_split_bounds(alpha) for alpha in (.5, .1, .01)],
                "analytic_moment_bound": survival_moment_bound(),
                "analytic_distance_exact_at_K_3_over_2_tau_1": {"numerator": 1, "denominator": 1364},
                "pooled_vacuum_bounds": [next_split_bounds(.1, copies) for copies in (1, 2, 4)],
                "pooled_tail_screen": [pooled_tail_probe(copies) for copies in (1, 2, 4)],
                "replacement_bounds": [pooled_replacement_bound(left, right)
                                       for left, right in ((1,1), (2,1), (2,2), (4,4), (4,12))]},
            "max_64_128_energy_difference": max(abs(a["battery_energy_gain_over_Estar"]-
                                                   b["battery_energy_gain_over_Estar"]) for a,b in zip(coarse,fine)),
            "laplace_64_128_difference": abs(integral_fine["battery_energy_gain_over_Estar"]-
                                            fine[0]["battery_energy_gain_over_Estar"]),
            "bath_energy_difference_from_continuum_integral": abs(
                fine[0]["initial_bath_energy_over_Estar"]-2*response["bath_energy_per_initial_number"]/3),
            "conditional_results": {
                "all_fock_sector_energy_preserving_unitary_constructed": True,
                "system_marginal_preserved_by_bath_battery_unitary": True,
                "time_independent_semibounded_generator_exists_by_spectral_calculus": True,
                "generator_is_supplied_nonlocal_operator": True,
                "bath_reset_to_vacuum": False,
                "battery_low_entropy_or_reusable_work_proved": False,
                "source_clock_controller_cycle_closed": False,
                "remaining_system_energy_growth_removed": False,
                "local_causal_CE_action_derived": False,
                "common_metric_selection_proved": False},
            "finite_spectral_grid_realizes_energy_transfer_unitary": False,
            "physical_candidate_adopted": False}


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
