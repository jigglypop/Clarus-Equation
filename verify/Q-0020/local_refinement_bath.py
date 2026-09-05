"""실제 1→5 단체 이웃에서 양의 국소 환경 결합과 정렬 시간의 경계를 검산한다.

진전 원장 §2의 같은 작용에서 준비 상태·환경·잡음을 연결하는 의무를 다룬다.
각 잎 단체에 무차원 진동자 하나를 놓는 대응은 공급된 가정이다. 실제 길이
열 개나 계량 텐서의 선택을 이 스칼라 모형과 동일시하지 않는다.
에너지는 E_*로, 시간은 hbar/E_*로 나눈다. x, epsilon, kappa, tau는 무차원이다.

1. 실제 분할의 국소 이웃
4-simplex를 내부 중심점으로 1→5 분할해도 기존 경계 사면체는 세분되지 않는다.
공유 꼭짓점 네 개가 같은 사면체를 이루는 두 잎만 이웃으로 잇는다.
깊이 D의 잎 수 n=5^D, 경계 면은 5개, 이웃 변 수는 (5n-5)/2이다.
첫 뿌리 자식의 후손 n/5개와 나머지를 잇는 변은 항상 4개이다.
이 절단의 평균을 뺀 지시함수로 Rayleigh 비를 계산하면
 lambda_2 <= 25/n, lambda_max >= trace(L)/n=5-5/n,
 R=lambda_max/lambda_2 >= max(1,(n-1)/5).
일반 Sierpinski 그래프의 스펙트럼 재귀 추측은 이 증명에 사용하지 않는다.
관련 원전: https://arxiv.org/abs/1908.04037 (일반 경우를 추측으로 구별한다).

2. 시간독립 국소 작용과 정확한 모드 분해
B는 부호 있는 변-꼭짓점 접속행렬, L=B^T B, g(x)=sqrt(x exp(-x))이다.
각 이웃 변에 독립 L2(R_+) 환경을 두고
 h = [[epsilon I+kappa L, sqrt(kappa) B^T <g|],
      [sqrt(kappa) |g>B, M_x]]
를 공급한다. epsilon,kappa>0이며 계-환경 결합은 변의 두 끝점만 쓴다.
정확한 양의 이차형식은
 epsilon||q||^2 + sum_e ||sqrt(x) f_e + sqrt(kappa) exp(-x/2)(Bq)_e||^2.
f의 정의역은 D(sqrt(M_x))이다. g/x 자체는 L2일 필요가 없고 위 가중
이차형식만 사용한다. off-diagonal은 유계이므로 h는 D(M_x)에서 자기수반이다.
이는 지정된 그래프의 시간독립 해밀토니언이며, 그래프를 자율 생성하는 작용은 아니다.

B의 특잇값 분해로 lambda>0마다 K=kappa*lambda인 기존 continuum_bath
모형과 정확히 동치다. lambda=0 공통 모드는 exp(-i epsilon tau)로 자유 회전한다.
환경 초기 진공에서 계의 진폭 행렬 A=V diag(A_K) V^T와
 X=realify(A), Y=(I-X X^T)/2
가 함께 정해진다. 따라서 이 공급 작용 안에서는 잡음을 감쇠와 독립 조정하지 않는다.

3. 고정 그래프의 장시간 수렴과 깊이에 균일하지 않은 수렴
기존 연속 환경의 순수 절대연속 스펙트럼으로 고정 유한 그래프의 모든 차이
진폭은 tau→infinity에서 0으로 간다. 유한 초기 차이 점유 N_diff에 대해
 N_out <= s_max N_diff,
 D(rho_out,rho_common(tau) tensor vacuum_diff) <= min(1,2 sqrt(N_out)).
진공 사영의 실패확률 p<=N_out, 사영 뒤 상태와의 거리<=sqrt(p), 주변상태의
거리 수축을 합한 부등식이다. 초기 공통-차이 상관도 허용한다.
공통 상태는 초기 주변상태의 자유 회전이며 유일한 공통 계량을 고른 것이 아니다.
차이 위치 분산도 0이 아니라 진공 값 1/2로 남는다.

반면 epsilon,tau를 고정하고 어느 양의 kappa_D를 골라도, R→infinity이면
가장 덜 감쇠하는 차이 모드의 생존확률은 1로 간다.
K_-=kappa_D lambda_2 <=1/sqrt(R)이면
 s_max >= max(0,1-tau^2/(2 sqrt(R)))^2.
그렇지 않으면 K_+=kappa_D lambda_max>sqrt(R)이다. 기존 quasimode_bound의
강결합 하한은 K에 따라 증가하므로 s_max>=strong(sqrt(R)).
두 경우의 작은 하한을 취하면 kappa_D 전체에 유효하고 극한은 1이다.
이는 연산자 노름에서의 균일 감쇠 경계이다. 평균 에너지의 동일한 하한이나
모든 가능한 국소 작용의 불가능성으로 확대하지 않는다.

실제 source_dilation(5)의 재귀에서 Q>=5I/4, P=Q^-1/4이므로
N=(Q+P-I)/2>=9I/40. 따라서 가장 느린/강한 모드도 초기 점유가 0인 방향은 아니다.
기하의 부모-자식 순서와 양자 source의 부모-자식 행 순서를 명시적으로 동일시한다.
이는 추가 대응 규칙이며 CE 미시 작용에서 유도된 물리 사상은 아니다.

수치 출력의 큰 시간 결과는 연속 적분을 사용한다. 유한 환경 행렬은 단일 충돌의
작용·모드 분해·잡음 일치를 검산하는 용도이며 그 행렬로 장시간 방출을 주장하지 않는다.
국소 단체 대응, 스펙트럼 g, 간격 epsilon과 kappa의 물리 선택, 자율 분할·병합,
공통 계량과 3+1 Plebanski/Einstein은 미완성이다.

4. 같은 위치 법칙의 국소 위상 자유도
표준 이차 위상 변환 U_b=exp(i b q^T L q/2)를 공급된 준비 상태에 적용한다.
원전의 한 모드 위상 변환: https://arxiv.org/abs/0903.3233, II.B(d), 식 (9).
대칭 L에 대해 U_b^dagger p U_b=p+bLq이고 q는 불변이다. L이 이웃 접속행렬에서
나오므로 생성자는 b/2 sum_edges(q_i-q_j)^2이며 서로 가환하는 국소 항의 합이다.
따라서 전체 위치 분포, 공통 모드 주변상태, 순수성, 정준성을 모두 보존한다.
하지만 같은 해밀토니언과 관측량을 고정한 실제 상태 준비에서는
 Vqq=Q, Vqp=b QL, Vpp=P+b^2 LQL,
 delta N_j=b^2 lambda_j^2 (v_j^T Q v_j)/2,
 delta E=b^2 Tr[(epsilon I+kappa L)LQL]/2
이다. 독립 진공 환경과 계의 평균 0 때문에 초기 상호작용의 기대값은 0이다.
에너지 증가는 E_* 단위이며 b도 무차원이다. 위치 법칙·국소성·정준성만으로
운동량과 준비 에너지가 유일하게 정해지지 않는 구체적인 반례다.
이는 해밀토니언까지 함께 바꾸는 정준 좌표 재명명이 아니다.
이 한 매개변수 계열에서는 주어진 작용의 준비 에너지 최소 조건이 b=0을
유일하게 고른다. 그 최소 조건 또는 시간반전 대칭은 추가 물리 입력이며,
실제 기하의 정준 축약이나 자율 분할에서 유도한 조건이 아니다.

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

from continuum_bath import collision_response, quasimode_bound
from split_quantum_source import recursive_source_maps


def _depth(depth, maximum=4):
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= maximum:
        raise ValueError(f"깊이는 1부터 {maximum}까지의 정수여야 합니다")
    return depth


def _parameters(epsilon, kappa, time=0.):
    epsilon, kappa, time = map(float, (epsilon, kappa, time))
    if not all(map(math.isfinite, (epsilon,kappa,time))) or min(epsilon,kappa) <= 0 or time < 0:
        raise ValueError("간격과 결합은 유한한 양수, 시간은 유한한 음이 아닌 값이어야 합니다")
    return epsilon,kappa,time


def refined_cells(depth):
    """기존 Regge 분할의 부모 순서·꼭짓점 생략 순서를 그대로 보존한다."""
    _depth(depth)
    cells, words, label = [tuple(range(5))], [()], 5
    for _ in range(depth):
        following, next_words = [], []
        for cell, word in zip(cells, words):
            for child, omitted in enumerate(cell):
                following.append((label,)+tuple(v for v in cell if v != omitted))
                next_words.append(word+(child,))
            label += 1
        cells, words = following, next_words
    return cells, words


def dual_graph(depth):
    """공유 사면체로만 잎 단체를 연결한다."""
    cells, words = refined_cells(depth)
    owners = {}
    for index, cell in enumerate(cells):
        for face in combinations(cell,4):
            owners.setdefault(tuple(sorted(face)),[]).append(index)
    if any(len(group)>2 for group in owners.values()):
        raise ArithmeticError("하나의 사면체에 셋 이상의 단체가 접착되었습니다")
    edges = sorted(tuple(group) for group in owners.values() if len(group)==2)
    incidence = np.zeros((len(edges),len(cells)))
    for row,(left,right) in enumerate(edges):
        incidence[row,left],incidence[row,right] = -1.,1.
    laplacian = incidence.T @ incidence
    return {"cells":cells,"words":words,"edges":edges,"incidence":incidence,
            "laplacian":laplacian,"boundary_facets":sum(len(group)==1 for group in owners.values())}


def spectrum(depth):
    """그래프 영모드 하나와 양의 차이 모드를 분리한다."""
    graph = dual_graph(depth)
    values,vectors = np.linalg.eigh(graph["laplacian"])
    if abs(values[0])>1e-10 or values[1]<=1e-10:
        raise ArithmeticError("연결 그래프의 단일 영모드가 확인되지 않았습니다")
    values[0]=0.
    return graph,values,vectors


def uniform_survival_floor(ratio, epsilon=2., time=1.):
    """한 가지 결합 세기의 모든 선택에 유효한 두 극단 모드 하한."""
    ratio,epsilon,time = map(float,(ratio,epsilon,time))
    if not all(map(math.isfinite,(ratio,epsilon,time))) or ratio<1 or epsilon<=0 or time<0:
        raise ValueError("고유값 비는 1 이상, 간격은 양수, 시간은 음이 아니어야 합니다")
    threshold=math.sqrt(ratio)
    weak=max(0.,1-time*time/(2*threshold))**2
    strong=quasimode_bound(threshold,epsilon,time)["survival_probability_lower_bound"]
    return {"spectral_ratio":ratio,"time":time,"weak_endpoint_floor":weak,
            "strong_endpoint_floor":strong,"all_scalar_kappa_worst_mode_floor":min(weak,strong)}


def refinement_floor(depth, epsilon=2., time=1.):
    """큰 그래프를 생성하지 않고 정확한 절단식에서 깊이별 하한을 얻는다."""
    _depth(depth,20)
    leaves=5**depth
    result=uniform_survival_floor(max(1.,(leaves-1)/5),epsilon,time)
    return {"depth":depth,"leaves":leaves,"gap_upper_bound":25/leaves,
            "ratio_is_proven_lower_bound":True,**result}


def graph_case(depth):
    """실제 이웃 수·병목·차이 스펙트럼을 보고한다."""
    graph,values,_=spectrum(depth)
    leaves=len(values)
    cluster=leaves//5
    cut=sum((left<cluster)!=(right<cluster) for left,right in graph["edges"])
    if cut!=4 or len(graph["edges"])!=(5*leaves-5)//2 or graph["boundary_facets"]!=5:
        raise ArithmeticError("분할 그래프의 귀납 불변식이 어긋났습니다")
    return {"depth":depth,"leaves":leaves,"edges":len(graph["edges"]),
            "boundary_facets":graph["boundary_facets"],"root_child_cut_edges":cut,
            "smallest_contrast_eigenvalue":float(values[1]),"largest_eigenvalue":float(values[-1]),
            "gap_upper_bound":25/leaves,"actual_spectral_ratio":float(values[-1]/values[1]),
            "maximum_degree":float(np.diag(graph["laplacian"]).max())}


def finite_edge_hamiltonian(depth=1, epsilon=2., kappa=.5, nodes=16):
    """이웃별 환경의 유한 적분 행렬을 검산용으로 구성한다."""
    _depth(depth,2)
    epsilon,kappa,_=_parameters(epsilon,kappa)
    if isinstance(nodes,bool) or not isinstance(nodes,int) or not 8<=nodes<=32:
        raise ValueError("유한 환경 적분점은 8부터 32까지의 정수여야 합니다")
    graph=dual_graph(depth)
    incidence=graph["incidence"]
    energies,weights=np.polynomial.laguerre.laggauss(nodes)
    coupling=math.sqrt(kappa)*np.kron(incidence,np.sqrt(energies*weights)[:,None])
    count=len(graph["cells"])
    h=np.diag(np.r_[np.zeros(count),np.tile(energies,len(incidence))])
    h[:count,:count]=epsilon*np.eye(count)+kappa*graph["laplacian"]
    h[count:,:count]=coupling
    h[:count,count:]=coupling.T
    return graph,energies,weights,h


def _scalar_finite_amplitude(strength,epsilon,time,nodes):
    if strength==0:
        return np.exp(-1j*epsilon*time)
    x,w=np.polynomial.laguerre.laggauss(nodes)
    h=np.diag(np.r_[epsilon+strength,x])
    h[0,1:]=h[1:,0]=np.sqrt(strength*x*w)
    values,vectors=np.linalg.eigh(h)
    return complex(np.sum(vectors[0,:]**2*np.exp(-1j*values*time)))


def realify(amplitude):
    """복소 수보존 진폭을 q,p 교차 순서의 실수 정준 행렬로 바꾼다."""
    n=len(amplitude)
    result=np.empty((2*n,2*n))
    result[0::2,0::2]=result[1::2,1::2]=amplitude.real
    result[0::2,1::2]=-amplitude.imag
    result[1::2,0::2]=amplitude.imag
    return result


def finite_channel_check(depth=1, epsilon=2., kappa=.5, time=1., nodes=16):
    """전체 이웃 환경 행렬과 모드별 분해가 같은 감쇠·잡음을 주는지 검사한다."""
    epsilon,kappa,time=_parameters(epsilon,kappa,time)
    graph,_,_,h=finite_edge_hamiltonian(depth,epsilon,kappa,nodes)
    count=len(graph["cells"])
    energy,basis=np.linalg.eigh(h)
    whole=(basis*np.exp(-1j*energy*time)) @ basis.T
    direct=whole[:count,:count]
    values,vectors=np.linalg.eigh(graph["laplacian"])
    values[0]=0.
    amplitudes=np.array([_scalar_finite_amplitude(kappa*float(value),epsilon,time,nodes)
                         for value in values])
    modal=(vectors*amplitudes) @ vectors.T
    x=realify(modal)
    y=(np.eye(2*count)-x@x.T)/2
    # 직사각형 환경 진폭의 복소 Gram으로 잡음을 독립 계산한다.
    noise_from_environment=realify(whole[:count,count:] @ whole[:count,count:].conj().T)/2
    omega=np.kron(np.eye(count),np.array([[0.,1.],[-1.,0.]]))
    cp=y+.5j*(omega-x@omega@x.T)
    return {"depth":depth,"nodes":nodes,"system_modes":count,
            "hamiltonian_minimum_eigenvalue":float(energy.min()),
            "amplitude_decomposition_residual":float(np.linalg.norm(modal-direct)),
            "noise_decomposition_residual":float(np.linalg.norm(y-noise_from_environment)),
            "channel_cp_minimum_eigenvalue":float(np.linalg.eigvalsh(cp).min()),
            "system_amplitude":modal,"x":x,"y":y}


def source_state(depth):
    """부모 우선 잎 순서로 공급된 실제 5진 분할 공분산을 붙인다."""
    _depth(depth,3)
    q_map,p_map=recursive_source_maps(5,depth)
    q,p=.5*q_map@q_map.T,.5*p_map@p_map.T
    count=len(q)
    covariance=np.zeros((2*count,2*count))
    covariance[0::2,0::2],covariance[1::2,1::2]=q,p
    return covariance,(q+p-np.eye(count))/2


@lru_cache(maxsize=512)
def _continuum_response(epsilon,strength,time):
    return collision_response(epsilon,strength,time)


def continuum_case(depth, time, epsilon=2., kappa=.5):
    """같은 유한 그래프의 큰 시간 방출을 연속 환경 적분으로 평가한다."""
    _depth(depth,3)
    epsilon,kappa,time=_parameters(epsilon,kappa,time)
    _,values,vectors=spectrum(depth)
    covariance,number=source_state(depth)
    occupation=np.diag(vectors.T @ number @ vectors)
    responses=[_continuum_response(epsilon,float(f"{kappa*float(value):.12g}"),time)
               for value in values[1:]]
    survival=np.array([row["survival_probability"] for row in responses])
    amplitudes=np.r_[np.exp(-1j*epsilon*time),
                     [complex(*row["amplitude"]) for row in responses]]
    system_amplitude=(vectors*amplitudes) @ vectors.T
    x=realify(system_amplitude)
    y=(np.eye(len(x))-x@x.T)/2
    output=x@covariance@x.T+y
    remaining=float(survival@occupation[1:])
    covariance_number=float(np.trace(output)/2-len(values)/2-occupation[0])
    return {"depth":depth,"time":time,"epsilon":epsilon,"edge_kappa":kappa,
            "initial_contrast_number":float(occupation[1:].sum()),
            "remaining_contrast_number":remaining,
            "worst_contrast_survival":float(survival.max()),
            "trace_distance_upper_using_numeric_number":min(1.,2*math.sqrt(max(0.,remaining))),
            "covariance_number_residual":abs(covariance_number-remaining),
            "source_number_minimum_eigenvalue":float(np.linalg.eigvalsh(number).min()),
            "slowest_graph_mode_initial_number":float(occupation[1]),
            "maximum_reported_quadrature_error":max(row["quadrature_error_estimate"] for row in responses),
            "coupling_rounding_significant_digits":12,
            "quadrature_error_is_rigorous_bound":False}



def phase_prepared_state(depth, phase):
    """위치 분포를 보존하는 이웃 위상 조작으로 실제 준비 상태를 바꾼다."""
    phase = float(phase)
    if not math.isfinite(phase):
        raise ValueError("위상 계수는 유한해야 합니다")
    covariance, _ = source_state(depth)
    graph = dual_graph(depth)
    transform = np.eye(len(covariance))
    transform[1::2, 0::2] = phase * graph["laplacian"]
    return transform @ covariance @ transform.T, transform


def phase_preparation_case(depth=2, phase=1., epsilon=2., kappa=.5, time=1.):
    """같은 위치 법칙에 허용되는 위상에 따라 준비와 방출 예산이 달라짐을 계산한다."""
    epsilon, kappa, time = _parameters(epsilon, kappa, time)
    prepared, transform = phase_prepared_state(depth, phase)
    phase = float(phase)
    initial, _ = source_state(depth)
    graph, values, vectors = spectrum(depth)
    count = len(values)
    laplacian = graph["laplacian"]
    q = initial[0::2, 0::2]
    number0 = (q + initial[1::2, 1::2] - np.eye(count)) / 2
    number1 = (q + prepared[1::2, 1::2] - np.eye(count)) / 2
    occupations0 = np.diag(vectors.T @ number0 @ vectors)
    occupations1 = np.diag(vectors.T @ number1 @ vectors)
    gain_formula = phase**2 * values**2 * np.diag(vectors.T @ q @ vectors) / 2
    system_h = epsilon * np.eye(count) + kappa * laplacian
    energy0 = float(np.trace(system_h @ number0))
    energy1 = float(np.trace(system_h @ number1))
    energy_gain = phase**2 * float(np.trace(system_h @ laplacian @ q @ laplacian)) / 2
    common = np.zeros((2, 2 * count))
    common[0, 0::2] = common[1, 1::2] = 1 / math.sqrt(count)
    omega = np.kron(np.eye(count), np.array([[0., 1.], [-1., 0.]]))
    responses = [
        _continuum_response(epsilon, float(f"{kappa*float(value):.12g}"), time)
        for value in values[1:]
    ]
    survival = np.array([row["survival_probability"] for row in responses])
    bath = np.array([row["bath_energy_per_initial_number"] for row in responses])
    return {
        "depth": depth, "phase": phase, "time": time,
        "epsilon": epsilon, "edge_kappa": kappa,
        "initial_position_covariance_residual": float(np.linalg.norm(prepared[0::2, 0::2] - q)),
        "common_marginal_covariance_residual": float(np.linalg.norm(common @ (prepared-initial) @ common.T)),
        "canonical_residual": float(np.linalg.norm(transform @ omega @ transform.T - omega)),
        "purity_residual": float(np.linalg.norm(prepared @ omega @ prepared - omega / 4)),
        "mode_number_gain_formula_residual": float(np.linalg.norm(occupations1-occupations0-gain_formula)),
        "reference_initial_energy": energy0,
        "phase_prepared_initial_energy": energy1,
        "phase_preparation_energy_gain": energy_gain,
        "energy_gain_formula_residual": abs(energy1-energy0-energy_gain),
        "reference_remaining_contrast_number": float(survival @ occupations0[1:]),
        "phase_prepared_remaining_contrast_number": float(survival @ occupations1[1:]),
        "additional_emitted_bath_energy": float(bath @ gain_formula[1:]),
        "position_joint_distribution_preserved": True,
        "same_hamiltonian_and_observables_held_fixed": True,
        "minimum_energy_selects_zero_phase_within_this_family": True,
        "physical_minimum_energy_preparation_derived": False,
    }

def run():
    finite=finite_channel_check()
    finite={key:value for key,value in finite.items() if not isinstance(value,np.ndarray)}
    here=Path(__file__).resolve().parent
    dependencies=[Path(__file__).resolve(),here/"continuum_bath.py",here/"split_quantum_source.py",
                  here/"F-01"/"predict_fold_budget.py"]
    return {"scope":"이웃 사면체에 공급된 선형 환경 작용의 조건부 동역학",
            "python":platform.python_version(),"numpy":np.__version__,
            "energy_unit":"E_*","time_unit":"hbar/E_*",
            "source_sha256":{str(path.relative_to(here)):hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in dependencies},
            "graphs":[graph_case(depth) for depth in (1,2,3,4)],
            "uniform_attenuation_floors":[refinement_floor(depth) for depth in (4,8,12,16,20)],
            "finite_local_decomposition":finite,
            "continuum_cases":[continuum_case(depth,time) for depth in (1,2,3) for time in (1.,20.,200.)],
            "phase_preparation_cases":[phase_preparation_case(phase=phase) for phase in (0.,-1.,1.)],
            "conditional_results":{
                "positive_time_independent_edge_hamiltonian_constructed":True,
                "noise_fixed_by_supplied_action_and_vacuum":True,
                "fixed_finite_graph_common_mode_limit_proved":True,
                "one_scalar_kappa_uniform_fixed_time_attenuation_excluded":True,
                "all_local_hamiltonians_excluded":False,
                "regge_cell_to_scalar_source_identification_is_supplied":True,
                "geometry_generates_quantum_source_map":False,
                "autonomous_split_merge_or_energy_recycling_closed":False,
                "common_metric_tensor_selected":False,
                "continuum_general_relativity_derived":False,
                "physical_candidate_adopted":False,
                "position_law_determines_quantum_preparation":False,
                "minimum_energy_preparation_is_supplied_condition":True},
            "spectral_decimation_conjecture_used_as_theorem":False}


if __name__=="__main__":
    result=run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
