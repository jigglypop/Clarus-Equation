"""고정 경계 raw Regge Gaussian의 두 단계 합성과 측도 계수를 검산한다.

Q-0020의 전체 상위 작용 전달을 위한 조건부 검사다. 실제 전역 변 길이를
조립하므로 길이 접착은 이미 강제되어 있다. 길이 차이의 동적 선택을 유도하지 않는다.
정준 시간 진화, 진공 준비, Lorentzian 확률, 공통 계량 선택도 이 계산에서 나오지 않는다.

영역은 엄격한 내부점의 평탄 Euclidean 1->5->25 분할이다. 모든 단체는
비퇴화이고 작용은 +sum area*deficit (경계항 포함)이다.
원전 arXiv:1110.6866의 S_R과 반대 부호다. 원전 §5.2, 식 5.29-5.34의
rank-one 내부 Hessian 정리를 적용하면 각 삽입의 네 병진 게이지를 제외한
한 모드는 이 부호에서 양수다. 다섯 child를 Schur 소거하면 parent가 복원된다.
Sylvester inertia 법칙으로 고정 root boundary의 두 단계 quotient는 양의 6차원이다.
이는 표준 Regge 결과의 적용이다. child parent들은 regular일 필요가 없다.

작용의 정확한 미분은 Schlaefli를 쓴다.
grad S=sum (kappa-theta) grad A,
H_ij=partial_j(grad_i S)=sum[(kappa-theta) A_ij-A_i theta_j].
전체 합은 대칭이고, 반환할 때 roundoff 수준의 비대칭만 대칭화한다.
각도는 inverse Gram의 facet normal 내적으로 구하며 삼각형 면적은 Heron 식이다.
유한차분을 main Hessian에 사용하지 않는다.

기본 측도는 전역 내부 길이의 Euclidean 내적에 대한 orthonormal gauge quotient.
G가 gauge generator, Z가 그 직교 여공간이면 내부 covariance는
 Z (beta Z^T H Z)^(-1) Z^T.
계층 좌표 T는 일반적으로 전체 gauge에 직교하지 않으므로 R=Z^T T를 사용한다.
H_T=R^T H_Z R, dz=|det R| du이고
 log Z_direct = log Z_hierarchical_raw + log|det R|.
이 기본 측도에 vertex-coordinate gauge volume이나 FP factor를 다시 곱하지 않는다.
측도 자체의 물리적 선택은 입력이며 보편적 분할 불변성을 주장하지 않는다.
one-loop 측도의 별도 제약은 arXiv:1404.5288에 있다.

q=delta_length/length_*, beta=length_*^2/(8pi length_P^2).
regular root의 normalized collective internal length에 대해
 Var(q_root)=1/(beta*40sqrt(5)),
 Var(delta_length_root)/length_P^2=8pi/(40sqrt(5)).
두 단계에서도 child 주변화 후 root의 같은 gauge-invariant observable을 비교한다.
child 고유곡률의 역수는 부모 고정 conditional 폭이며 전체 marginal 폭과 다르다.
위 길이 폭은 Euclidean Gaussian의 조건부 결과다. 물리적 G,hbar 및 측도를
공급했으며 gluing mismatch 분해능이나 oscillator ground covariance를 유도하지 않는다.
Gaussian 꼬리 전체의 단체 유효성이나 전체 비선형 적분 수렴도 주장하지 않는다.
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
EDGES = tuple(combinations(range(5), 2))
TRIANGLES = tuple(combinations(range(5), 3))


def simplex_derivatives(lengths, kappas):
    """Return action, gradient, Hessian and the unsymmetrized roundoff residual."""
    lengths, kappas = np.asarray(lengths, dtype=float), np.asarray(kappas, dtype=float)
    if lengths.shape != (10,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("열 개의 길이는 유한한 양수여야 합니다")
    if kappas.shape != (10,) or not np.isfinite(kappas).all():
        raise ValueError("열 개의 각도 계수는 유한해야 합니다")
    squared, ds = np.zeros((5, 5)), np.zeros((10, 5, 5))
    for j, (u, v) in enumerate(EDGES):
        squared[u, v] = squared[v, u] = lengths[j]**2
        ds[j, u, v] = ds[j, v, u] = 2*lengths[j]
    gram = (squared[0, 1:, None]+squared[0, None, 1:]-squared[1:, 1:])/2
    derivative = (ds[:, 0, 1:, None]+ds[:, 0, None, 1:]-ds[:, 1:, 1:])/2
    try:
        np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as error:
        raise ValueError("양의 비퇴화 Euclidean Gram 행렬이 필요합니다") from error
    inverse = np.linalg.solve(gram, np.eye(4))
    basis = np.vstack((-np.ones(4), np.eye(4)))
    normals = basis @ inverse @ basis.T
    dn = -np.einsum("ai,kij,jb->kab", basis @ inverse, derivative, inverse @ basis.T)
    action, gradient, hessian = 0., np.zeros(10), np.zeros((10, 10))
    for triangle, kappa in zip(TRIANGLES, kappas):
        u, v = sorted(set(range(5))-set(triangle))
        denominator = math.sqrt(normals[u, u]*normals[v, v])
        cosine = -normals[u, v]/denominator
        if not -1 < cosine < 1:
            raise ValueError("각도 미분에 비퇴화 단체가 필요합니다")
        angle = math.acos(cosine)
        dc = -dn[:, u, v]/denominator+normals[u, v]/(2*denominator)*(
            dn[:, u, u]/normals[u, u]+dn[:, v, v]/normals[v, v]
        )
        angle_gradient = -dc/math.sqrt(1-cosine*cosine)
        indices = [EDGES.index(e) for e in combinations(triangle, 2)]
        local = lengths[indices]
        sq = local*local
        f = 2*(sq[0]*sq[1]+sq[0]*sq[2]+sq[1]*sq[2])-np.sum(sq*sq)
        if f <= 0:
            raise ValueError("삼각형 면적은 양수여야 합니다")
        root = math.sqrt(f)
        df = 4*local*(np.sum(sq)-2*sq)
        ddf = 8*np.outer(local, local)
        np.fill_diagonal(ddf, 4*(np.sum(sq)-4*sq))
        area_gradient = np.zeros(10)
        area_gradient[indices] = df/(8*root)
        area_hessian = ddf/(8*root)-np.outer(df, df)/(16*root**3)
        action += root*(kappa-angle)/4
        gradient += (kappa-angle)*area_gradient
        hessian[np.ix_(indices, indices)] += (kappa-angle)*area_hessian
        hessian -= np.outer(area_gradient, angle_gradient)
    return float(action), gradient, (hessian+hessian.T)/2, float(np.linalg.norm(hessian-hessian.T))


def complement(generators, expected_rank):
    u, singular, _ = np.linalg.svd(generators, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-10*max(1., float(singular.max(initial=0.)))))
    if rank != expected_rank:
        raise ArithmeticError("내부점 이동의 게이지 차원이 예상과 다릅니다")
    return u[:, rank:]


@lru_cache(maxsize=6)
def geometry(depth=2, scale=1., shear=0.):
    if isinstance(depth, bool) or depth not in (1, 2):
        raise ValueError("검산 깊이는 1 또는 2입니다")
    scale, shear = float(scale), float(shear)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shear) or abs(shear) > .3:
        raise ValueError("척도는 유한한 양수, 형상 전단은 절댓값 0.3 이내입니다")
    data, r = gluing.length_gluing(depth), gluing.reference()
    transform = np.eye(4)
    transform[0, 1] = shear
    points = {i: scale*transform @ p for i, p in data["points"].items()}
    edges, a = data["global_edges"], data["assembly"]
    index = {e: i for i, e in enumerate(edges)}
    kappas = r.equal_split_kappas(data["cells"], tuple(range(5)), np.full(10, math.pi))
    h, gradient = np.zeros((len(edges), len(edges))), np.zeros(len(edges))
    action, asymmetry = 0., 0.
    for cell, kappa in zip(data["cells"], kappas):
        local = simplex_derivatives(r.cell_lengths(cell, points), kappa)
        indices = [index[tuple(sorted(e))] for e in combinations(cell, 2)]
        h[np.ix_(indices, indices)] += local[2]
        gradient[indices] += local[1]
        action += local[0]
        asymmetry = max(asymmetry, local[3])
    vertices = list(range(5, max(points)+1))
    gauge = (a.T @ r.gauge_directions(data["cells"], points, vertices))/a.sum(axis=0)[:, None]
    boundary = [i for i, e in enumerate(edges) if max(e) < 5]
    internal = [i for i, e in enumerate(edges) if max(e) >= 5]
    coarse = simplex_derivatives(r.cell_lengths(tuple(range(5)), points), np.full(10, math.pi))
    return {
        "hessian": h, "gradient": gradient, "action": action, "gauge": gauge[internal],
        "boundary": boundary, "internal": internal, "edges": edges, "vertices": vertices,
        "coarse": coarse, "asymmetry": asymmetry,
    }


def composition(depth=2, scale=1., shear=0., beta=1.):
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("작용 계수 beta는 유한한 양수여야 합니다")
    data = geometry(depth, scale, shear)
    h, boundary, internal = data["hessian"], data["boundary"], data["internal"]
    gauge = data["gauge"]
    z = complement(gauge, 4*len(data["vertices"]))
    hi, hb = h[np.ix_(internal, internal)], h[np.ix_(boundary, internal)]
    root_rows = [i for i, j in enumerate(internal) if max(data["edges"][j]) == 5]
    root_normal = complement(gauge[root_rows, :4], 4)[:, 0]
    t = np.zeros((len(internal), len(data["vertices"])))
    t[root_rows, 0] = root_normal
    for vertex in data["vertices"][1:]:
        rows = [i for i, j in enumerate(internal) if vertex in data["edges"][j]]
        columns = list(range(4*(vertex-5), 4*(vertex-4)))
        t[rows, vertex-5] = complement(gauge[np.ix_(rows, columns)], 4)[:, 0]
    qz, qt = z.T @ hi @ z, t.T @ hi @ t
    if min(np.linalg.eigvalsh(qz)) <= 0:
        raise ArithmeticError("내부 quotient의 양성이 성립하지 않습니다")
    mixing = hb @ z
    direct = h[np.ix_(boundary, boundary)]-mixing @ np.linalg.solve(qz, mixing.T)
    full = np.block([[h[np.ix_(boundary, boundary)], hb @ t], [t.T @ hb.T, qt]])
    if len(t.T) > 1:
        child = qt[1:, 1:]
        if min(np.linalg.eigvalsh(child)) <= 0:
            raise ArithmeticError("하위 내부 모드의 양성이 성립하지 않습니다")
        middle = full[:11, :11]-full[:11, 11:] @ np.linalg.solve(child, full[11:, :11])
        child_logdet = np.linalg.slogdet(child)[1]
    else:
        child, middle, child_logdet = np.zeros((0, 0)), full, 0.
    parent_curvature = float(middle[10, 10])
    if parent_curvature <= 0:
        raise ArithmeticError("상위 Schur 곡률이 양수가 아닙니다")
    sequential = middle[:10, :10]-np.outer(middle[:10, 10], middle[10, :10])/parent_curvature
    change = z.T @ t
    sign, log_jacobian = np.linalg.slogdet(change)
    if sign == 0:
        raise ArithmeticError("계층 절단면이 게이지 quotient에 횡단하지 않습니다")
    dimension = len(qz)
    base = dimension*math.log(2*math.pi/beta)/2
    log_direct = base-np.linalg.slogdet(qz)[1]/2
    log_raw = base-(child_logdet+math.log(parent_curvature))/2
    covariance_z = np.linalg.solve(beta*qz, np.eye(dimension))
    covariance_t = np.linalg.solve(beta*qt, np.eye(dimension))
    root_readout = np.zeros(len(internal))
    root_readout[root_rows] = root_normal
    root_variance = float(root_readout @ z @ covariance_z @ z.T @ root_readout)
    prior = 10*np.eye(10)
    prior_curvature = float(np.linalg.eigvalsh(prior+beta*direct)[0])
    boundary_covariance_residual = None
    if prior_curvature > 0:
        boundary_covariance = np.linalg.solve(prior+beta*direct, np.eye(10))
        sequential_covariance = np.linalg.solve(prior+beta*sequential, np.eye(10))
        boundary_covariance_residual = float(np.linalg.norm(boundary_covariance-sequential_covariance))
    return {
        "depth": depth, "background_scale": scale, "volume_preserving_shear": shear, "beta": beta,
        "internal_lengths": len(internal), "gauge_dimension": gauge.shape[1],
        "physical_dimension": dimension, "physical_curvatures": np.linalg.eigvalsh(qz).tolist(),
        "child_conditional_curvatures": np.linalg.eigvalsh(child).tolist(),
        "parent_marginal_curvature": parent_curvature,
        "gauge_hessian_residual": float(np.linalg.norm(hi @ gauge)),
        "mixed_gauge_residual": float(np.linalg.norm(hb @ gauge)),
        "cell_hessian_asymmetry": data["asymmetry"],
        "internal_gradient_residual": float(np.linalg.norm(data["gradient"][internal])),
        "classical_action_residual": float(abs(data["action"]-data["coarse"][0])),
        "boundary_hessian_residual": float(np.linalg.norm(direct-data["coarse"][2])),
        "sequential_hessian_residual": float(np.linalg.norm(sequential-direct)),
        "orthonormal_quotient_log_integral": float(log_direct),
        "hierarchical_coordinate_log_integral": float(log_raw),
        "quotient_jacobian": float(math.exp(log_jacobian)),
        "log_normalization_residual": float(log_raw+log_jacobian-log_direct),
        "omitted_jacobian_relative_error": float(math.expm1(log_raw-log_direct)),
        "covariance_transport_residual": float(np.linalg.norm(change @ covariance_t @ change.T-covariance_z)),
        "root_observable_gauge_residual": float(np.linalg.norm(root_readout @ gauge)),
        "root_marginal_variance": root_variance,
        "root_variance_from_parent_schur": 1/(beta*parent_curvature),
        "root_collective_variance_in_planck_units": 8*math.pi*beta*root_variance,
        "conditional_boundary_prior_precision": 10., "conditional_boundary_prior_normalizable": prior_curvature > 0,
        "boundary_covariance_composition_residual": boundary_covariance_residual,
        "gaussian_integral_is_conditional_on_fixed_boundary": True,
    }


def run():
    files = [
        "fixed_boundary_composition.py", "length_gluing_bath.py", "local_refinement_bath.py",
        "continuum_bath.py", "F-01/predict_fold_budget.py",
        "F-01/regge_one_to_five_boundary_hessian.py", "F-01/regge_one_to_five_refinement.py",
    ]
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files},
        "cases": [
            composition(1), composition(2), composition(2, scale=2.),
            composition(2, shear=.2), composition(2, beta=.7),
        ],
        "regular_root_exact_variance_in_planck_units": 8*math.pi/(40*math.sqrt(5)),
        "scope": {
            "raw_regge_hessian_used": True, "global_length_gluing_imposed": True,
            "orthonormal_quotient_measure_supplied": True, "physical_G_and_hbar_supplied": True,
            "standard_regge_result_is_new_ce_discovery": False,
            "unconditional_real_boundary_measure_normalized": False,
            "physical_quotient_measure_derived": False,
            "local_triangulation_invariant_one_loop_measure_derived": False,
            "computed_internal_width_is_gluing_resolution": False,
            "canonical_quantum_vacuum_prepared": False, "lorentzian_time_evolution_derived": False,
            "common_metric_selected_dynamically": False, "continuum_einstein_limit_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE/"fixed_boundary_composition.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
