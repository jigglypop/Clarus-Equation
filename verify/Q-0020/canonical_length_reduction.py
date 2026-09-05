"""공유 길이의 정준 축약과 공급된 Regge 경계 운동량을 검산한다.

진전 원장 §2의 미시 작용·측도 연결을 점검하는 조건부 계산이다.
q=delta_length/length_*, p=length_* p_physical/hbar, [q,p]=i 이며
시간은 hbar/E_*, 운동항 W와 진동수 epsilon은 무차원으로 공급한다.

ker C=im A에서 q=A Q이고 p^T dq=(A^T p)^T dQ이므로 P=A^T p.
L=(qdot^T W qdot-q^T K q)/2, W>0를 공급하면 M=A^T W A,
p=W A M^{-1}P, K_red=A^T K A이다. 독립 행 Cbar에 대한
Cbar q=0, Cbar W^{-1}p=0는 제2종 제약(second-class)이며 Dirac 괄호는
{q,p^T}_D=I-W^{-1}Cbar^T(Cbar W^{-1}Cbar^T)^{-1}Cbar
          =A M^{-1}A^T W.
이는 내부 정점 이동의 중력 게이지를 제거하는 절차가 아니다.
제약 없는 원래 흐름의 보존 조건은 C W^{-1}K A=0이다.

K=epsilon^2 W, qp 교차항 없음, R^E 위 진동자 양자화를 공급하면
Cov(Q)=(2 epsilon M)^{-1}. 단위 사본 질량은 M_ee=m_e이고,
사본 질량 1/m_e는 M=I를 준다. 두 선택 모두 같은 정확한 접착을
만족하지만 위치 분산은 다르다. 1->5 분할은 기존 변의 각 소유자를
세 소유자로 바꾸므로 단위 사본 처방의 분산은 1/3이 된다.
각 자식에 부모 질량의 1/3을 주면 기존 변 총질량과 분산은 보존된다.
새로 생기는 변의 총질량은 이 보존 조건으로 정해지지 않는다.

별도의 Regge 계산은 양의 유클리드 단체의 기하 작용
S_hat=sum(area_hat * deficit_angle)을 쓴다. 물리적인 S/hbar에는
beta=length_*^2/(8 pi length_P^2)를 곱해야 하며 여기서는 beta=1.
보고한 gradient를 물리 운동량으로 읽을 때 hbar/length_*를 곱한다.
이 beta 선택은 입력이다. 진동자 운동항 W를 Regge에서 유도하지 않는다.

평탄 내부점 분할에서 내부 방정식이 0이므로 Schlaefli 항등식과
작용의 경계항 가법성에 의해 S_fine=S_coarse이며 경계 미분도 같다.
사본 운동량은 사슬법칙으로 A^T p_copy에 모인다. 비정규 단체와
이동한 내부점도 비교하고, 내부 길이 하나의 독립 교란을 음성대조로 쓴다.
이는 고정 경계의 유클리드 변분이며 정준 시간진화가 아니다.
표준 출처: https://arxiv.org/abs/1108.1974.

축약 Hilbert 공간 L^2(R^E)는 원래 L^2(R^copies)의 delta 지지 상태가
아니다. 기존 고정 간격의 정확 접착 에너지 발산을 피하는 유한 에너지
유니터리 준비를 구성하지 않는다. Gaussian은 선형 길이 섭동의 모형이며
모든 꼬리가 유효 단체라는 주장도 하지 않는다. 물리 측도·분해능·
자율 준비·Lorentzian 진화·공통 계량 선택·Einstein 유도는 미완성이다.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

import length_gluing_bath as gluing


HERE = Path(__file__).resolve().parent


def canonical_reduction(assembly, constraint, copy_masses):
    """중복 사본 지도와 양의 대각 운동항에 대한 정준 매입을 구성한다."""
    a, c = np.asarray(assembly, dtype=float), np.asarray(constraint, dtype=float)
    w = np.asarray(copy_masses, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0 or not np.all(np.isfinite(a)):
        raise ValueError("사본 지도는 유한한 비어 있지 않은 행렬이어야 합니다")
    n, edges = a.shape
    if not np.all((a == 0) | (a == 1)) or not np.all(a.sum(axis=1) == 1) or np.any(a.sum(axis=0) == 0):
        raise ValueError("각 사본은 정확히 한 전역 변에 속하고 모든 변에 소유자가 있어야 합니다")
    if c.ndim != 2 or c.shape[1] != n or not np.all(np.isfinite(c)):
        raise ValueError("제약 행렬의 사본 차원과 유한값을 확인하십시오")
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("사본 질량은 사본 수와 같은 길이의 유한한 양수 벡터여야 합니다")
    _, values, vt = np.linalg.svd(c, full_matrices=False)
    rank = int(np.count_nonzero(values > 1e-10 * max(1., float(values.max(initial=0.)))))
    if rank != n-edges or np.linalg.norm(c @ a) > 1e-10:
        raise ValueError("제약의 kernel과 사본 지도의 image가 일치해야 합니다")
    independent = vt[:rank]
    mass = a.T @ (w[:, None] * a)
    momentum_lift = np.linalg.solve(mass, (w[:, None] * a).T).T
    bracket = np.eye(n)
    if rank:
        normal = (independent / w) @ independent.T
        bracket -= (independent.T / w[:, None]) @ np.linalg.solve(normal, independent)
    return {
        "assembly": a, "constraint": c, "independent_constraint": independent,
        "copy_masses": w, "mass": mass, "momentum_lift": momentum_lift,
        "position_readout": momentum_lift.T, "dirac_qp": bracket,
    }


def oscillator_covariance(reduced, epsilon=2.):
    """K=epsilon^2 W인 공급된 진동자의 축약 진공 공분산."""
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("진동수는 유한한 양수여야 합니다")
    mass = reduced["mass"]
    return np.linalg.solve(mass, np.eye(len(mass))) / (2*epsilon), epsilon*mass/2


def _omega(size):
    return np.block([[np.zeros((size, size)), np.eye(size)],
                     [-np.eye(size), np.zeros((size, size))]])


def reduction_case(depth=1):
    if isinstance(depth, bool) or depth not in (1, 2):
        raise ValueError("수치 검산 깊이는 1 또는 2입니다")
    data = gluing.length_gluing(depth)
    a, c = data["assembly"], data["constraint"]
    owners = a.sum(axis=0)
    rows = {}
    for name, w in (("unit_copy_mass", np.ones(len(a))), ("unit_global_mass", 1/(a @ owners))):
        reduced = canonical_reduction(a, c, w)
        b, mass = reduced["momentum_lift"], reduced["mass"]
        zero = np.zeros_like(a)
        embedding = np.block([[a, zero], [zero, b]])
        q, p = oscillator_covariance(reduced)
        compatible_k = 4*np.diag(w)
        incompatible_k = compatible_k + np.diag(np.linspace(.1, 1., len(w)))
        rows[name] = {
            "global_mass": np.diag(mass).tolist(), "position_variance": np.diag(q).tolist(),
            "momentum_variance": np.diag(p).tolist(),
            "symplectic_residual": float(np.linalg.norm(embedding.T @ _omega(len(a)) @ embedding-_omega(a.shape[1]))),
            "secondary_constraint_residual": float(np.linalg.norm((c/w) @ b)),
            "dirac_bracket_residual": float(np.linalg.norm(reduced["dirac_qp"]-a @ b.T)),
            "compatible_flow_residual": float(np.linalg.norm((c/w) @ compatible_k @ a)),
            "incompatible_flow_residual": float(np.linalg.norm((c/w) @ incompatible_k @ a)),
        }
    return {"depth": depth, "global_edges": data["global_edges"], "owners": owners.tolist(), "choices": rows}


def refinement_case(new_edge_mass=1.):
    """실제 셀 순서에서 부모 사본 질량을 세 자식 사본에 나눈다."""
    new_edge_mass = float(new_edge_mass)
    if not math.isfinite(new_edge_mass) or new_edge_mass <= 0:
        raise ValueError("새 변의 총질량은 유한한 양수여야 합니다")
    coarse, fine = gluing.length_gluing(1), gluing.length_gluing(2)
    a0, a1 = coarse["assembly"], fine["assembly"]
    parent_weights = np.linspace(.7, 1.3, len(a0))
    from itertools import combinations
    parent_maps = [
        {tuple(sorted(e)): parent_weights[10*i+j] for j, e in enumerate(combinations(cell, 2))}
        for i, cell in enumerate(coarse["cells"])
    ]
    child_weights = []
    for i, cell in enumerate(fine["cells"]):
        parent = parent_maps[i//5]
        for e in combinations(cell, 2):
            edge = tuple(sorted(e))
            child_weights.append(parent[edge]/3 if edge in parent else new_edge_mass/4)
    old_indices = [fine["global_edges"].index(e) for e in coarse["global_edges"]]
    new_indices = [i for i, e in enumerate(fine["global_edges"]) if e not in coarse["global_edges"]]
    mass0, mass1 = a0.T @ parent_weights, a1.T @ np.asarray(child_weights)
    owners0, owners1 = a0.sum(axis=0), a1.sum(axis=0)
    return {
        "new_edge_mass_input": new_edge_mass,
        "coarse_owner_counts": owners0.tolist(),
        "fine_old_owner_counts": owners1[old_indices].tolist(),
        "unit_copy_old_variance_ratio": (owners0/owners1[old_indices]).tolist(),
        "additive_old_mass_residual": float(np.linalg.norm(mass1[old_indices]-mass0)),
        "additive_old_variance_ratio": (mass0/mass1[old_indices]).tolist(),
        "new_edge_masses": mass1[new_indices].tolist(),
        "new_edge_masses_selected_by_conservation": False,
    }


def richardson_gradient(fun, point, step=5e-4):
    point = np.asarray(point, dtype=float)
    step = float(step)
    if point.ndim != 1 or not np.all(np.isfinite(point)) or not math.isfinite(step) or step <= 0:
        raise ValueError("미분점은 유한한 벡터, 차분 간격은 유한한 양수여야 합니다")
    def central(h):
        return np.array([(fun(point+h*v)-fun(point-h*v))/(2*h) for v in np.eye(len(point))])
    return (4*central(step/2)-central(step))/3


@lru_cache(maxsize=8)
def regge_boundary_case(irregular=False, moved_center=False, relative_internal_shift=0., step=5e-4):
    """같은 기하 작용에서 사본·전역·거친 경계 미분을 직접 비교한다."""
    shift = float(relative_internal_shift)
    if not math.isfinite(shift) or abs(shift) > .05:
        raise ValueError("내부 길이 대조의 상대 교란은 절댓값 0.05 이내입니다")
    data = gluing.length_gluing(1)
    a, r = data["assembly"], gluing.reference()
    points = {key: value.copy() for key, value in data["points"].items()}
    if irregular:
        transform = np.diag([1.1, .93, 1.04, .98])
        transform[0, 1] = .08
        points = {key: transform @ value for key, value in points.items()}
    if moved_center:
        points[5] = sum(w*points[i] for i, w in enumerate((.1, .15, .2, .25, .3)))
    lengths = np.array([np.linalg.norm(points[i]-points[j]) for i, j in data["global_edges"]])
    boundary = [i for i, e in enumerate(data["global_edges"]) if max(e) < 5]
    internal = [i for i, e in enumerate(data["global_edges"]) if max(e) >= 5]
    lengths[internal[0]] *= 1+shift
    kappas = r.equal_split_kappas(data["cells"], tuple(range(5)), np.full(10, math.pi))
    def action(x, kappa):
        # 각 차분점의 단체가 양의 유클리드 Gram 행렬을 갖는지 확인한다.
        r.points_from_squared(np.asarray(x)**2)
        return r.simplex_action(x, kappa)
    def fine_action(x):
        return sum(action(row, k) for row, k in zip((a @ x).reshape(-1, 10), kappas))
    coarse_action = lambda x: action(x, np.full(10, math.pi))
    copy_lengths = (a @ lengths).reshape(-1, 10)
    copy_p = np.concatenate([
        richardson_gradient(lambda x: action(x, k), row, step)
        for row, k in zip(copy_lengths, kappas)
    ])
    assembled = a.T @ copy_p
    direct = richardson_gradient(fine_action, lengths, step)
    coarse_p = richardson_gradient(coarse_action, lengths[boundary], step)
    fine_value, coarse_value = fine_action(lengths), coarse_action(lengths[boundary])
    return {
        "irregular_boundary": bool(irregular), "moved_center": bool(moved_center),
        "relative_internal_shift": shift, "difference_step": float(step),
        "dimensionless_action_coefficient": 1.,
        "fine_action": fine_value, "coarse_action": coarse_value,
        "action_difference": fine_value-coarse_value,
        "copy_to_global_chain_residual": float(np.linalg.norm(assembled-direct)),
        "internal_gradient_norm": float(np.linalg.norm(assembled[internal])),
        "boundary_gradient_residual": float(np.linalg.norm(assembled[boundary]-coarse_p)),
        "boundary_momentum": assembled[boundary].tolist(),
        "coarse_boundary_momentum": coarse_p.tolist(),
        "finite_difference_is_proof": False,
        "canonical_time_evolution_constructed": False,
    }


def run():
    files = [
        "canonical_length_reduction.py", "length_gluing_bath.py", "local_refinement_bath.py",
        "continuum_bath.py", "F-01/predict_fold_budget.py",
        "F-01/regge_one_to_five_boundary_hessian.py", "F-01/regge_one_to_five_refinement.py",
    ]
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in files},
        "units": {
            "position": "delta_length/length_*", "momentum": "length_* p_physical/hbar",
            "time": "time_physical E_*/hbar",
            "regge_action_coefficient": "beta=length_*^2/(8 pi length_P^2); beta=1 supplied",
            "regge_momentum_physical": "hbar/length_* times reported dimensionless gradient at supplied beta",
        },
        "reduction": [reduction_case(d) for d in (1, 2)],
        "refinement": [refinement_case(m) for m in (1., 2.)],
        "regge": [
            regge_boundary_case(step=1e-3), regge_boundary_case(),
            regge_boundary_case(irregular=True, moved_center=True),
            regge_boundary_case(relative_internal_shift=.02),
        ],
        "scope": {
            "supplied_positive_diagonal_mass": True, "supplied_euclidean_regge_action": True,
            "constraint_force_needed_for_generic_stiffness": True,
            "standard_canonical_reduction_is_new_ce_result": False,
            "unique_vacuum_selected_by_length_gluing": False,
            "new_edge_mass_selected_by_old_edge_conservation": False,
            "oscillator_mass_derived_from_regge": False,
            "finite_energy_preparation_in_original_hilbert_space_constructed": False,
            "internal_vertex_gauge_removed": False, "canonical_time_evolution_constructed": False,
            "common_metric_selected": False, "lorentzian_einstein_limit_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE/"canonical_length_reduction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
