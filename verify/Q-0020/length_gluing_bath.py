"""공유 사면체의 실제 길이 불일치와 유한 분해능의 양자 비용을 계산한다.

진전 원장 §2의 기하 변수·환경 작용 연결을 위한 조건부 검산이다.
기존 양의 국소 환경 작용을 셀당 열 개의 길이 섭동에 적용한다.
q=delta_length/length_*, p=length_* physical_momentum/hbar, [q,p]=i.
에너지와 시간은 E_*, hbar/E_* 단위이다. 정준 길이 운동량, 양의 Wick
측도, 동일 간격 epsilon과 환경 g(x)=sqrt(x exp(-x))는 공급된 입력이다.

1. 공유 사면체의 여섯 변마다 인접 셀 길이 사본의 차이를 C의 한 행으로 둔다.
n=5^D개 잎에는 10n개 사본과 E=(5n+35)/4개 전역 변이 있다.
각 기존 변의 소유자들은 그 변을 포함하는 사면체 인접 관계로 연결된다.
이는 1→5 분할에서 기존 변 소유자 하나가 서로 연결된 세 자식으로 대체되고,
새 중심점 변의 네 소유자는 서로 연결되며, 옛 접촉이 유지됨으로 귀납된다.
따라서 ker C는 전역 변별 동일 길이 사본이고 dim ker C=E,
rank C=10n-E=35(n-1)/4, 행 수는 15(n-1)이다.
내부 정점 이동은 이 kernel에 있지만, 아래 진동자 작용이 Regge의 게이지
대칭을 구현하거나 그 방향을 양자 제약으로 제거한 것은 아니다.

2. 기존 작용의 B를 C로 바꾸면 양의 이차형식
 epsilon||z||^2 + sum_a ||sqrt(x)f_a+sqrt(kappa)exp(-x/2)(Cz)_a||^2
을 얻는다. z는 한 입자 공간의 계 진폭이다. h_ss=epsilon I+kappa C^T C.
모든 양의 고유모드는 연속 환경으로 방출되고 ker C는 자유 회전한다.
이 환경은 남아 있는 전역 변 길이들 중 어느 Regge 기하를 선택할지 결정하지 않는다.
이 양성은 h>=epsilon P_system을 주므로 dGamma(h)>=epsilon N_system이다.
주어진 유한 그래프와 유한 초기 점유에서 밝은 모드의 장시간 상태는 진공이다.
그러나 위치 불일치 공분산은 C V_qq C^T -> C C^T/2이며 0이 아니다.
각 행의 노름 제곱이 2이므로 면의 변 하나당 평균 불일치 분산은 1,
물리 길이 단위에서는 length_*^2이다. 평균값의 접착과 정확한 양자 접착을
동일시하면 안 된다. 법선·외재 곡률·Lorentzian 서명·Einstein 방정식도 미유도다.

공유 사면체의 길이는 그 위 유도 계량을 정한다.
원전: https://arxiv.org/abs/0802.0864, https://arxiv.org/abs/0907.4325.
여기서는 양의 유클리드 배경 근방의 선형 길이 섭동만 다룬다.
Gaussian 꼬리를 포함한 모든 표본이 비퇴화 단체라는 주장은 하지 않는다.

3. C의 양의 고유값 lambda_j, rank r, 위치모드 분산 v_j를 쓰자.
sum lambda_j v_j <= eta^2 이면 Robertson과 Cauchy로
 E_total >= epsilon (sum sqrt(lambda_j))^2/(8 eta^2)-epsilon r/2.
실제 에너지는 음이 아니므로 이 식과 0 중 큰 것을 하한으로 쓴다.
모드 간 상관·비Gaussian 상태도 각 정준 모드의 불확정성을 피하지 못한다.
첫 분할에서는 원래 변 열 개의 소유자 그래프가 K3, 새 변 다섯 개는 K4이다.
따라서 양의 고유값은 3이 20개, 4가 15개이고 sum sqrt(lambda)=20sqrt(3)+30이다.
epsilon=2, 비교 한 개당 평균 제곱 오차 rho=eta^2/60이면 정확한 하한은
 E_total >= max(0,(35/4+5sqrt(3))/rho-35)
이다. 이 유한 모형의 조건부 부등식이며 우주론 관측량 예측은 아니다.

eta는 전체 무차원 위치 불일치의 표준편차 상한이며 평균 불일치가 0이면
이 분산이 이차 오차 전체와 같다. 평균이 0이 아니면 에너지 요구는 더 커진다.
epsilon>0과 length_*를 고정하고 eta->0이면 에너지는 발산한다.
Cq=0 위에만 지지되는 상태는 기존 L2 정준 공간에서 정규화할 수 없다.
별도 제약 Hilbert 공간을 선언하는 방식은 이 환경 감쇠로 도출하지 않았다.

4. 양의 유한 오차는 실제 압축 Gaussian으로 달성할 수 있다.
진공 환경과 독립인 평균 0 준비 상태에서는 a_j=epsilon+kappa lambda_j,
 E=sum a_j(v_j+1/(4v_j)-1)/2.
주어진 eta 이하를 만족하는 최소는
 v_j=1/[2 sqrt(1+gamma lambda_j/a_j)], gamma>=0,
 sum lambda_j v_j=eta^2 로 정해진다. 진공 오차 이상이면 gamma=0이다.
각 p 분산=1/(4v_j), 교차 공분산=0인 독립 Gaussian이 이 하한을 달성한다.
유한 분해능 압축 준비에서는 위치 측도를 새로 공급하고 전역 변의 영모드를
진공으로 둔다. 앞서 사용한 Wick 위치 측도를 보존하는 준비라는 주장은 하지 않는다.
이는 주어진 준비 상태 클래스의 변분 결과이며 환경이 그 상태를 자동 준비하는
동역학을 구성한 것은 아니다. 분해능 법칙·준비 동력·계량 선택은 미완성이다.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from continuum_bath import collision_response
from local_refinement_bath import dual_graph, realify


HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def reference():
    original = sys.path[:]
    try:
        sys.path.insert(0, str(HERE / "F-01"))
        spec = importlib.util.spec_from_file_location("length_gluing_reference", HERE / "F-01" / "predict_fold_budget.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original


def parameters(epsilon, kappa):
    epsilon, kappa = float(epsilon), float(kappa)
    if not all(map(math.isfinite, (epsilon, kappa))) or min(epsilon, kappa) <= 0:
        raise ValueError("간격과 결합은 유한한 양수여야 합니다")
    return epsilon, kappa


def geometry(depth):
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 3:
        raise ValueError("깊이는 1부터 3까지의 정수여야 합니다")
    r = reference()
    points = r.points_from_squared(np.full(10, 2.0))
    cells = [tuple(range(5))]
    for _ in range(depth):
        cells = r.refine(cells, points)
    return cells, points


def length_gluing(depth):
    cells, points = geometry(depth)
    graph = dual_graph(depth)
    if cells != graph["cells"]:
        raise ArithmeticError("기준 기하와 이웃 그래프의 셀 순서가 다릅니다")
    owners, copies = {}, []
    for i, cell in enumerate(cells):
        mapping = {}
        for j, edge in enumerate(combinations(cell, 2)):
            edge = tuple(sorted(edge))
            mapping[edge] = 10*i+j
            owners.setdefault(edge, []).append(10*i+j)
        copies.append(mapping)
    rows, faces = [], []
    for left, right in graph["edges"]:
        face = tuple(sorted(set(cells[left]) & set(cells[right])))
        faces.append((left, right, face))
        for edge in combinations(face, 2):
            rows.append((copies[left][edge], copies[right][edge]))
    c = np.zeros((len(rows), 10*len(cells)))
    for a, (left, right) in enumerate(rows):
        c[a, left], c[a, right] = 1.0, -1.0
    global_edges = sorted(owners)
    assembly = np.zeros((c.shape[1], len(global_edges)))
    for j, edge in enumerate(global_edges):
        assembly[owners[edge], j] = 1.0
    normalized = assembly / np.sqrt(np.sum(assembly, axis=0))[None, :]
    return {
        "cells": cells, "points": points, "faces": faces, "constraint": c,
        "laplacian": c.T @ c, "global_edges": global_edges,
        "assembly": assembly, "kernel_basis": normalized,
    }


def spectrum(depth):
    data = length_gluing(depth)
    values, vectors = np.linalg.eigh(data["laplacian"])
    dark = len(data["global_edges"])
    if np.max(np.abs(values[:dark])) > 1e-9 or values[dark] <= 1e-9:
        raise ArithmeticError("길이 접착 kernel의 차원이 예상과 다릅니다")
    values[:dark] = 0.0
    return data, values, vectors, dark


def face_gram_jacobian(lengths):
    """사면체의 여섯 길이에서 여섯 독립 Gram 성분으로 가는 정확한 미분."""
    edges = list(combinations(range(4), 2))
    lengths = np.asarray(lengths, dtype=float)
    if lengths.shape != (6,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("사면체 길이는 여섯 개의 유한한 양수여야 합니다")
    index = {edge: i for i, edge in enumerate(edges)}
    jacobian = np.zeros((6, 6))
    for row, (i, j) in enumerate(((1,1), (2,2), (3,3), (1,2), (1,3), (2,3))):
        if i == j:
            jacobian[row, index[(0,i)]] = 2*lengths[index[(0,i)]]
        else:
            for vertex in (i,j):
                jacobian[row, index[(0,vertex)]] = lengths[index[(0,vertex)]]
            jacobian[row, index[(i,j)]] = -lengths[index[(i,j)]]
    return jacobian


def graph_case(depth):
    data, values, _, dark = spectrum(depth)
    c, n = data["constraint"], len(data["cells"])
    gauge = reference().gauge_directions(data["cells"], data["points"], list(range(5, max(data["points"])+1)))
    return {
        "depth": depth, "cells": n, "length_copies": 10*n,
        "face_edge_comparisons": len(c), "global_edges": dark,
        "constraint_rank": len(values)-dark,
        "expected_global_edges": (5*n+35)//4,
        "kernel_assembly_residual": float(np.linalg.norm(c @ data["assembly"])),
        "kernel_orthogonality_residual": float(np.linalg.norm(data["kernel_basis"].T @ data["kernel_basis"]-np.eye(dark))),
        "internal_displacement_columns": gauge.shape[1],
        "internal_displacement_rank": int(np.linalg.matrix_rank(gauge, tol=1e-9)),
        "displacement_in_kernel_residual": float(np.linalg.norm(c @ gauge)),
        "smallest_positive_eigenvalue": float(values[dark]),
        "largest_eigenvalue": float(values[-1]),
        "vacuum_total_length_mismatch_variance": float(np.trace(c @ c.T)/2),
        "vacuum_mismatch_variance_per_comparison": float(np.trace(c @ c.T)/(2*len(c))),
    }


@lru_cache(maxsize=2)
def wick_preparation(depth=1):
    """기존의 양의 Wick 길이 측도를 평균 0 순수 Gaussian으로 실현한다."""
    if depth not in (1,2) or isinstance(depth, bool):
        raise ValueError("Wick 준비의 검산 깊이는 1 또는 2여야 합니다")
    cells, points = geometry(depth)
    r = reference()
    kappas = r.equal_split_kappas(cells, tuple(range(5)), np.full(10, np.pi))
    size = 10*len(cells)
    q, precision = np.zeros((size,size)), np.zeros((size,size))
    for i, (cell, kappa) in enumerate(zip(cells, kappas)):
        h = r.simplex_hessian(r.cell_lengths(cell, points), kappa)
        values, vectors = np.linalg.eigh(h)
        if np.min(np.abs(values)) < 1e-10:
            raise ArithmeticError("셀 Hessian의 역행렬을 안정적으로 계산할 수 없습니다")
        block = slice(10*i, 10*i+10)
        q[block,block] = (vectors / np.abs(values)) @ vectors.T
        precision[block,block] = (vectors * np.abs(values)) @ vectors.T
    covariance = np.zeros((2*size,2*size))
    covariance[0::2,0::2], covariance[1::2,1::2] = q, precision/4
    return covariance


@lru_cache(maxsize=512)
def response(epsilon, strength, time):
    return collision_response(epsilon, strength, time)


def channel(depth, time, epsilon=2., kappa=.5):
    epsilon, kappa = parameters(epsilon, kappa)
    time = float(time)
    if not math.isfinite(time) or time < 0:
        raise ValueError("시간은 유한한 음이 아닌 값이어야 합니다")
    data, values, vectors, dark = spectrum(depth)
    responses = [response(epsilon, float(f"{kappa*value:.12g}"), time) for value in values[dark:]]
    amplitudes = np.r_[
        np.full(dark, np.exp(-1j*epsilon*time)),
        [complex(*item["amplitude"]) for item in responses],
    ]
    a = (vectors * amplitudes) @ vectors.T
    x = realify(a)
    y = (np.eye(len(x))-x @ x.T)/2
    return data, values, vectors, dark, x, y, responses


def evolve_wick(depth=1, time=200., epsilon=2., kappa=.5):
    data, values, vectors, dark, x, y, responses = channel(depth,time,epsilon,kappa)
    initial = wick_preparation(depth)
    output = x @ initial @ x.T + y
    c, size = data["constraint"], len(values)
    initial_mismatch = c @ initial[0::2,0::2] @ c.T
    final_mismatch = c @ output[0::2,0::2] @ c.T
    vacuum_mismatch = c @ c.T/2
    number = (initial[0::2,0::2]+initial[1::2,1::2]-np.eye(size))/2
    occupation = np.diag(vectors.T @ number @ vectors)
    remaining = float(np.array([item["survival_probability"] for item in responses]) @ occupation[dark:])
    return {
        "depth": depth, "time": time,
        "initial_total_mismatch_variance": float(np.trace(initial_mismatch)),
        "final_total_mismatch_variance": float(np.trace(final_mismatch)),
        "vacuum_total_mismatch_variance": float(np.trace(vacuum_mismatch)),
        "vacuum_covariance_residual": float(np.linalg.norm(final_mismatch-vacuum_mismatch)),
        "remaining_bright_number": remaining,
        "distance_upper_using_numeric_number": min(1., 2*math.sqrt(max(0., remaining))),
        "maximum_quadrature_error_estimate": max(item["quadrature_error_estimate"] for item in responses),
        "quadrature_error_is_rigorous_bound": False,
    }


def resolution_preparation(depth, mean_squared_error, epsilon=2., kappa=.5):
    """유한 오차를 실현하는 최소 준비 에너지와 모든 상태에 유효한 하한."""
    epsilon, kappa = parameters(epsilon,kappa)
    error = float(mean_squared_error)
    if not math.isfinite(error) or error <= 0:
        raise ValueError("평균 제곱 오차는 유한한 양수여야 합니다")
    data, values, _, dark = spectrum(depth)
    positive = values[dark:]
    target = len(data["constraint"])*error
    vacuum = float(np.sum(positive)/2)
    energy_weights = epsilon+kappa*positive
    def variances(gamma):
        return 0.5/np.sqrt(1+gamma*positive/energy_weights)
    gamma = 0.
    if target < vacuum:
        low, high = 0., 1.
        for _ in range(1024):
            if float(positive @ variances(high)) <= target:
                break
            high *= 2
            if not math.isfinite(high):
                raise ArithmeticError("요청한 분해능은 현재 부동소수점 범위를 벗어납니다")
        else:
            raise ArithmeticError("분해능의 변분 계수를 찾지 못했습니다")
        for _ in range(120):
            middle = (low+high)/2
            if float(positive @ variances(middle)) > target:
                low = middle
            else:
                high = middle
        gamma = high
    q_variances = variances(gamma)
    energy = float(np.sum(energy_weights*(q_variances+1/(4*q_variances)-1))/2)
    universal = max(0., epsilon*float(np.sum(np.sqrt(positive)))**2/(8*target)-epsilon*len(positive)/2)
    return {
        "depth": depth, "mean_squared_error_target": error,
        "achieved_mean_squared_error": float(positive @ q_variances)/len(data["constraint"]),
        "epsilon": epsilon, "edge_kappa": kappa,
        "universal_total_energy_lower_bound": universal,
        "product_vacuum_bath_minimum_preparation_energy": energy,
        "lagrange_gamma": gamma,
        "positive_eigenvalues": positive.tolist(),
        "position_variances": q_variances.tolist(),
        "momentum_variances": (1/(4*q_variances)).tolist(),
        "independent_vacuum_bath_preparation_required": True,
        "autonomous_preparation_constructed": False,
        "wick_position_measure_preserved": False,
        "dark_modes_prepared_in_vacuum": True,
    }



def volume_preserving_shape_case(strain=.2, time=200.):
    """고정한 배치에서 부피 보존과 공유 면의 길이 일치가 별개임을 검사한다."""
    strain = float(strain)
    if not math.isfinite(strain) or abs(strain) > 1:
        raise ValueError("검산 변형 계수는 절댓값 1 이하의 유한한 수여야 합니다")
    data = length_gluing(1)
    metric = np.diag([math.exp(strain), math.exp(-strain), 1., 1.])
    cell = data["cells"][0]
    q = np.zeros(50)
    for index, (left, right) in enumerate(combinations(cell,2)):
        edge = data["points"][left]-data["points"][right]
        q[index] = math.sqrt(edge @ metric @ edge)-np.linalg.norm(edge)
    c = data["constraint"]
    mean = np.zeros(100)
    mean[0::2] = q
    _, _, _, _, x, _, _ = channel(1,time)
    evolved = x @ mean
    residual = np.linalg.norm(c @ evolved[0::2])**2+np.linalg.norm(c @ evolved[1::2])**2
    return {
        "strain": strain, "time": time,
        "cell_volume_ratio": float(math.sqrt(np.linalg.det(metric))),
        "initial_length_mismatch_norm": float(np.linalg.norm(c @ q)),
        "evolved_squared_phase_space_mismatch": float(residual),
        "all_cell_volumes_preserved": True,
        "shared_face_lengths_initially_match": bool(np.linalg.norm(c @ q)<1e-12),
        "quantum_noise_excluded_from_mean_diagnostic": True,
    }

def run():
    dependencies = [
        Path(__file__).resolve(), HERE/"local_refinement_bath.py", HERE/"continuum_bath.py",
        HERE/"F-01"/"predict_fold_budget.py",
        HERE/"F-01"/"regge_one_to_five_boundary_hessian.py",
        HERE/"F-01"/"regge_one_to_five_refinement.py",
    ]
    return {
        "scope": "실제 유클리드 길이 사본에 공급된 양의 환경 작용과 분해능 비용",
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies},
        "energy_unit": "E_*", "length_unit": "length_*", "time_unit": "hbar/E_*",
        "graphs": [graph_case(depth) for depth in (1,2)],
        "wick_evolution": [evolve_wick(time=time) for time in (0.,1.,20.,200.)],
        "volume_preserving_shape": volume_preserving_shape_case(),
        "resolution_preparations": [resolution_preparation(1,error) for error in (1.,.1,.01,.001)],
        "conditional_results": {
            "face_local_length_coupling_constructed": True,
            "global_edge_kernel_identified": True,
            "bath_selects_global_regge_geometry": False,
            "positive_action_and_noise_fixed_conditionally": True,
            "exact_quantum_length_equality_obtained": False,
            "finite_resolution_squeezed_preparation_constructed": True,
            "canonical_length_kinetic_term_derived_from_regge": False,
            "wick_measure_selection_derived": False,
            "gauge_reduction_implemented_by_bath": False,
            "all_gaussian_samples_are_valid_simplices": False,
            "autonomous_split_merge_preparation_closed": False,
            "common_metric_tensor_selected": False,
            "continuum_einstein_equations_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
