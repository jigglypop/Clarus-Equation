"""원래 레게 제약의 정준 축약과 정상점 합류에서의 계수 퇴화를 검산한다.

공유 무차원 길이, beta=1, 표준 정준 형식은 공급 전제다.
제2종 제약의 국소 리우빌 측도와 양자 구성공간 측도를 구별한다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import regge_pachner_constraints as moves

N = len(moves.FINAL.edges)
E, Y = moves.E_ID, moves.Y_ID
REST = np.array([i for i in range(N) if i not in (E, Y)])
OMEGA = np.block([[np.zeros((N, N)), np.eye(N)],
                  [-np.eye(N), np.zeros((N, N))]])


def flat_lengths(h):
    """기존 여섯 점 계열에서 네 단체 모두의 길이를 구성한다."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("높이는 유한한 양수여야 한다")
    points = moves.reference_points()
    points[0, 0], points[1, 0] = -h, h
    return moves.FINAL.lengths(points)


def constraint_jacobian(lengths, step=2e-5):
    """서로 다른 두 국소 이동의 기울기를 독립 차분한다."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("차분 간격은 유한한 양수여야 한다")

    def fields(q):
        data = moves.actions(q)
        return np.array([-data["first"]["gradient"][Y],
                         data["second"]["gradient"][E]])

    def central(delta):
        columns = []
        for i in range(N):
            shift = np.eye(N)[i]*delta
            columns.append((fields(lengths+shift)-fields(lengths-shift))/(2*delta))
        return np.column_stack(columns)

    derivative = (4*central(step/2)-central(step))/3
    jacobian = np.zeros((2, 2*N))
    jacobian[:, :N] = derivative
    jacobian[0, N+Y], jacobian[1, N+E] = 1, 1
    return jacobian


def reduction(lengths, step=2e-5, singular_tolerance=1e-7):
    """제약면 접사상과 원래 푸아송 행렬에서 축약을 각각 구성한다."""
    if not np.isfinite(singular_tolerance) or singular_tolerance <= 0:
        raise ValueError("특이성 허용 오차는 유한한 양수여야 한다")
    jac = constraint_jacobian(lengths, step)
    matrix = jac @ OMEGA @ jac.T
    a = float(matrix[0, 1])

    # 축약 좌표는 (모든 길이 q, 남은 열세 운동량 p_z)다.
    tangent = np.zeros((2*N, 2*N-2))
    tangent[:N, :N] = np.eye(N)
    tangent[N+E, :N] = -jac[1, :N]
    tangent[N+Y, :N] = -jac[0, :N]
    tangent[N+REST, N+np.arange(len(REST))] = 1
    reduced = tangent.T @ OMEGA @ tangent
    singular_values = np.linalg.svd(reduced, compute_uv=False)
    result = {
        "bracket": a,
        "constraint_rank": int(np.linalg.matrix_rank(jac)),
        "pullback_rank": int(np.sum(singular_values > singular_tolerance)),
        "smallest_singular_values": singular_values[-2:].tolist(),
        "tangent_residual": float(np.max(np.abs(jac @ tangent))),
        "second_class": bool(abs(a) > singular_tolerance),
    }
    if not result["second_class"]:
        # 계수 퇴화는 영 확률의 선언이나 전체 중력 게이지 판정이 아니다.
        result.update({"dirac_ey": None, "liouville_density": None,
                       "inverse_residual": None, "coordinate_check": None})
        return result

    projector = np.zeros((2*N-2, 2*N))
    projector[:N, :N] = np.eye(N)
    projector[N+np.arange(len(REST)), N+REST] = 1
    dirac = OMEGA-OMEGA @ jac.T @ np.linalg.solve(matrix, jac @ OMEGA)
    reduced_poisson = projector @ dirac @ projector.T
    sign, logdet = np.linalg.slogdet(reduced)
    density = float(np.exp(logdet/2))

    data = moves.actions(lengths)
    p = data["middle"]["gradient"].copy()
    p[E], p[Y] = -data["second"]["gradient"][E], data["first"]["gradient"][Y]
    canonical_change = np.eye(2*N)
    for i in (E, Y):
        canonical_change[i, i] = 1/(2*lengths[i])
        canonical_change[N+i, N+i] = 2*lengths[i]
        canonical_change[N+i, i] = p[i]/(2*lengths[i]**2)
    scales = np.array([2*lengths[Y], 2*lengths[E]])
    new_jac = (jac @ canonical_change)/scales[:, None]
    new_bracket = new_jac @ OMEGA @ new_jac.T
    reduced_change = np.eye(2*N-2)
    reduced_change[E, E] = 1/(2*lengths[E])
    reduced_change[Y, Y] = 1/(2*lengths[Y])
    new_reduced = reduced_change.T @ reduced @ reduced_change
    new_density = float(np.exp(np.linalg.slogdet(new_reduced)[1]/2))
    factor = float(4*lengths[E]*lengths[Y])
    result.update({
        "dirac_ey": float(reduced_poisson[E, Y]),
        "dirac_pair_error": float(abs(reduced_poisson[E, Y]-1/a)),
        "inverse_residual": float(np.max(np.abs(
            reduced @ reduced_poisson+np.eye(2*N-2)))),
        "liouville_density": density,
        "determinant_sign": float(sign),
        "density_error": float(abs(density-abs(a))),
        "coordinate_check": {
            "canonical_residual": float(np.max(np.abs(
                canonical_change.T @ OMEGA @ canonical_change-OMEGA))),
            "squared_bracket": float(new_bracket[0, 1]),
            "bracket_error": float(abs(new_bracket[0, 1]-a/factor)),
            "density_squared": new_density,
            "density_error": float(abs(new_density-density/factor)),
            "restored_density_error": float(abs(new_density*factor-density)),
            "jacobian_factor": factor,
        },
    })
    return result


def run():
    rows = []
    for h in (.5, .9, 1., 1.1):
        q = flat_lengths(h)
        values = moves.FINAL.evaluate(q)
        hessian, skew = moves.FINAL.hessian(q, [E, Y], step=1e-4)
        coarse = reduction(q, step=4e-5)
        fine = reduction(q, step=2e-5)
        expected = 3*math.sqrt(2)*(1-h*h)
        rows.append({
            "h": h, "e": float(q[E]), "y": float(q[Y]),
            "minimum_gram_eigenvalue": float(values["minimum_gram_eigenvalue"]),
            "equation_residual": float(abs(values["gradient"][E])),
            "curvature_residual": float(np.max(np.abs(values["deficits"][~moves.FINAL.boundary]))),
            "expected_bracket": expected,
            "bracket_formula_error": float(abs(fine["bracket"]-expected)),
            "independent_hessian_error": float(abs(fine["bracket"]+hessian[0, 1])),
            "step_difference": float(abs(coarse["bracket"]-fine["bracket"])),
            "flat_hessian_ee": float(hessian[0, 0]),
            "flat_hessian_error": float(abs(hessian[0, 0]-12*math.sqrt(3)*h*(h*h-1))),
            "hessian_skew": skew,
            "reduction": fine,
        })
    return {
        "status": "[산출]",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("regge_pachner_constraints.py", "regge_pachner_creation.py",
                         "regge_pachner_transport.py", "regge_tent_transfer.py")
        },
        "scope": "공급한 공유 길이·정준 위상공간·beta=1에서 실제 두 제약의 국소 축약",
        "exact_formulas": {
            "bracket_flat_family": "3*sqrt(2)*(1-h**2)",
            "dirac_ey": "1/a (a!=0)",
            "liouville_density": "abs(a) de dy dz dp_z (a!=0)",
            "squared_density": "abs(a)/(4*e*y) dE dY dz dp_z",
            "rank_h1": 26,
            "rank_regular": 28,
        },
        "cases": rows,
        "unfinished": [
            "제약 행렬의 계수가 변하는 영역에서의 일반 이력·양자 상태 전달",
            "정상식 밖의 물리 경로측도·내적·준비·분해능",
            "공통 계량의 동역학적 선택과 0D에서 3+1 Plebanski/Einstein 다리",
        ],
    }


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"cases": [
        {key: row[key] for key in ("h", "expected_bracket", "bracket_formula_error",
                                   "minimum_gram_eigenvalue")}
        for row in report["cases"]]}, ensure_ascii=False))

