"""실제 전방 파흐너 병합의 사전 제약과 생성 제약의 결합을 검산한다.

공유 길이와 유클리드 레게 작용을 공급한다. 직전 단체의 역제거와 다음
단체를 더하는 진화를 구분하며, 한 사후 제약만으로 물리 게이지를 선언하지
않는다. 양자 비교의 측도·순서·도메인은 별도로 공급한 조건이다.
"""

from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy.optimize import brentq

import regge_pachner_transport as transport
from regge_pachner_creation import PachnerCreation
from regge_tent_transfer import ReggeComplex


MOVE = PachnerCreation()
FUTURE_CELL = (0, 1, 3, 4, 5)
FINAL = ReggeComplex(MOVE.new.cells+(FUTURE_CELL,))
Y_ID = FINAL.edge_index[3, 4]
E_ID = FINAL.edge_index[0, 1]
OLD_IDS = FINAL.indices(MOVE.old.edges)


def reference_points():
    """내부 변 01의 연결 구가 정사면체 2345의 경계인 평탄 증인이다."""
    points = np.zeros((6, 4))
    points[0, 0], points[1, 0] = -.5, .5
    points[2:, 1:] = np.array([[1, 1, 1], [1, -1, -1],
                               [-1, 1, -1], [-1, -1, 1]])/math.sqrt(3)
    return points


def boundary_facets(complex_):
    counts = Counter(tuple(sorted(f)) for c in complex_.cells for f in combinations(c, 4))
    return {f for f, count in counts.items() if count == 1}


def local_increment(before, after, cell, lengths):
    """새 단체의 각도와 경계각 상수의 변화로 작용 증분을 직접 조립한다."""
    local = ReggeComplex((cell,))
    ids = after.indices(local.edges)
    local_lengths = np.asarray(lengths)[ids]
    data = local.evaluate(local_lengths)
    old_kappa = dict(zip(before.triangles, np.where(before.boundary, 1, 2)))
    new_kappa = dict(zip(after.triangles, np.where(after.boundary, 1, 2)))
    action, gradient = data["action"], data["gradient"].copy()
    coefficients = {}
    for tri, edge_ids, area in zip(local.triangles, local.triangle_edges, data["areas"]):
        coefficient = int(new_kappa[tri]-old_kappa.get(tri, 0))
        coefficients[str(tri)] = coefficient
        correction = math.pi*(coefficient-1)
        action += correction*area
        squared = local_lengths[edge_ids]**2
        gradient[edge_ids] += correction*local_lengths[edge_ids]*(sum(squared)-2*squared)/(8*area)
    full_gradient = np.zeros(len(after.edges))
    full_gradient[ids] = gradient
    return {"action": float(action), "gradient": full_gradient, "pi_coefficients": coefficients}


def actions(lengths):
    """두 국소 증분과 세 전체 복합체의 작용·기울기를 독립 대조한다."""
    lengths = np.asarray(lengths, dtype=float)
    old = MOVE.old.evaluate(lengths[OLD_IDS])
    middle = MOVE.new.evaluate(lengths)
    final = FINAL.evaluate(lengths)
    first = local_increment(MOVE.old, MOVE.new, (0, 1, 2, 3, 4), lengths)
    second = local_increment(MOVE.new, FINAL, FUTURE_CELL, lengths)
    old_gradient = np.zeros(len(lengths))
    old_gradient[OLD_IDS] = old["gradient"]
    return {"old": old, "middle": middle, "final": final, "first": first, "second": second,
            "action_residual": max(abs(middle["action"]-old["action"]-first["action"]),
                                   abs(final["action"]-middle["action"]-second["action"])),
            "gradient_residual": max(np.linalg.norm(middle["gradient"]-old_gradient-first["gradient"]),
                                     np.linalg.norm(final["gradient"]-middle["gradient"]-second["gradient"]))}


def forward(old_lengths, old_momenta, y, tolerance=1e-9):
    """생성 뒤 다음 단체를 붙이되 내부화되는 변의 사전 제약을 요구한다."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("허용 오차는 유한한 음이 아닌 수여야 한다")
    lengths, middle_momenta = MOVE.create(old_lengths, old_momenta, y)
    data = actions(lengths)
    final_momenta = middle_momenta+data["second"]["gradient"]
    if abs(final_momenta[E_ID]) > tolerance:
        raise ValueError("전방 병합의 내부 변 사전 제약을 만족하지 않는다")
    kept = np.array([i for i in range(len(lengths)) if i != E_ID])
    return lengths[kept], final_momenta[kept]


def mixed_derivative(lengths, step=2e-4):
    """전체 작용의 네 점 차분을 기울기 기반 혼합 미분과 대조한다."""
    def central(h):
        de, dy = np.eye(len(lengths))[E_ID]*h, np.eye(len(lengths))[Y_ID]*h
        return (FINAL.evaluate(lengths+de+dy)["action"]-FINAL.evaluate(lengths+de-dy)["action"]
                -FINAL.evaluate(lengths-de+dy)["action"]+FINAL.evaluate(lengths-de-dy)["action"])/(4*h*h)
    return float((4*central(step/2)-central(step))/3)


def exact_mixed_certificate():
    """정확한 그램 역행렬과 각도 미분으로 평탄점의 혼합 계수를 계산한다."""
    import sympy as sp

    points = sp.Matrix([[sp.Rational(-1, 2), 0, 0, 0], [sp.Rational(1, 2), 0, 0, 0],
                        [0, 1, 1, 1], [0, 1, -1, -1], [0, -1, 1, -1], [0, -1, -1, 1]])
    points[:, 1:] = points[:, 1:]/sp.sqrt(3)
    y = sp.sqrt(sp.Rational(8, 3))
    sums = {k: sp.S.Zero for k in range(2, 6)}
    contributions = {k: [] for k in range(2, 6)}
    for cell in FINAL.cells:
        distances = sp.zeros(5)
        derivative = sp.zeros(5)
        for i, j in combinations(range(5), 2):
            delta = points[cell[i], :]-points[cell[j], :]
            distances[i, j] = distances[j, i] = sp.simplify(delta.dot(delta))
            if {cell[i], cell[j]} == {3, 4}:
                derivative[i, j] = derivative[j, i] = 2*y
        gram = sp.Matrix(4, 4, lambda i, j: (distances[0, i+1]+distances[0, j+1]-distances[i+1, j+1])/2)
        dg = sp.Matrix(4, 4, lambda i, j: (derivative[0, i+1]+derivative[0, j+1]-derivative[i+1, j+1])/2)
        inverse = gram.inv()
        di = -inverse*dg*inverse

        def normals(matrix):
            n = sp.zeros(5)
            n[1:, 1:] = matrix
            for i in range(4):
                n[0, i+1] = n[i+1, 0] = -sum(matrix[i, :])
            n[0, 0] = sum(matrix)
            return n

        n, dn = normals(inverse), normals(di)
        for k in sums:
            if not {0, 1, k}.issubset(cell):
                continue
            i, j = [index for index, vertex in enumerate(cell) if vertex not in (0, 1, k)]
            denominator = sp.sqrt(n[i, i]*n[j, j])
            cosine = -n[i, j]/denominator
            dc = (-dn[i, j]+n[i, j]*(dn[i, i]/n[i, i]+dn[j, j]/n[j, j])/2)/denominator
            dtheta = sp.simplify(-dc/sp.sqrt(1-cosine*cosine))
            contributions[k].append(str(dtheta))
            sums[k] -= dtheta
    edge = sp.Symbol("edge", positive=True)
    area_derivative = sp.diff(edge*sp.sqrt(5-edge*edge)/4, edge).subs(edge, 1)
    mixed = sp.simplify(area_derivative*sum(sums.values()))
    return {"area_derivative": str(area_derivative),
            "dihedral_derivatives": contributions,
            "deficit_derivatives": {k: str(sp.simplify(value)) for k, value in sums.items()},
            "mixed": str(mixed), "poisson_bracket": str(-mixed), "value": float(mixed)}


def ordering_check(u, t=.43):
    """실제 파동함수를 미분해 순서화한 제약과 단순 미분 제약을 구분한다."""
    bounds = transport.squared_bounds(u)
    y = math.sqrt(bounds["A"]+bounds["D"]*t)
    data = transport.fields(u, y)
    center = transport.reference_coordinates()
    value = transport.wave(u, y, center, excited=False)

    def derivative(h):
        return (transport.wave(u, y+h, center, excited=False)
                -transport.wave(u, y-h, center, excited=False))/(2*h)

    h = 2e-5
    dy = (4*derivative(h/2)-derivative(h))/3
    naive = -1j*dy-data["phase_y"]*value
    expected = .5j*data["speed_y"]/data["speed"]*value
    ordered = data["speed"]*naive-.5j*data["speed_y"]*value
    return {"t": t, "y": y, "naive_residual": float(abs(naive)),
            "naive_formula_error": float(abs(naive-expected)),
            "ordered_residual": float(abs(ordered)),
            "relative_naive_residual": float(abs(naive/value))}


def run():
    lengths = FINAL.lengths(reference_points())
    old_lengths = lengths[OLD_IDS]
    old_momenta = MOVE.old.evaluate(old_lengths)["gradient"]
    rows = []
    for factor in (.85, .95, 1, 1.05, 1.1):
        x = lengths.copy()
        x[Y_ID] *= factor
        data = actions(x)
        new_lengths, p = MOVE.create(old_lengths, old_momenta, x[Y_ID])
        restored_q, restored_p = MOVE.undo(new_lengths, p)
        c = p[Y_ID]-data["first"]["gradient"][Y_ID]
        f = p[E_ID]+data["second"]["gradient"][E_ID]
        hessian, skew = FINAL.hessian(x, np.array([E_ID, Y_ID]), step=2e-5)
        try:
            forward(old_lengths, old_momenta, x[Y_ID])
            accepted = True
        except ValueError:
            accepted = False
        rows.append({"factor": factor, "postconstraint": float(c), "preconstraint": float(f),
                     "internal_equation_residual": float(abs(f-data["final"]["gradient"][E_ID])),
                     "action_residual": data["action_residual"], "gradient_residual": float(data["gradient_residual"]),
                     "undo_residual": float(np.linalg.norm(restored_q-old_lengths)+np.linalg.norm(restored_p-old_momenta)),
                     "forward_accepted": accepted, "mixed_hessian": float(hessian[0, 1]),
                     "hessian_skew": skew,
                     "curvature": data["final"]["deficits"][~FINAL.boundary].tolist()})
    def equation(y):
        x = lengths.copy()
        x[Y_ID] = y
        return FINAL.evaluate(x)["gradient"][E_ID]
    root = brentq(equation, .95*lengths[Y_ID], 1.05*lengths[Y_ID], xtol=1e-13)
    exact = exact_mixed_certificate()
    coarse = mixed_derivative(lengths)
    fine = mixed_derivative(lengths, step=1e-4)
    b0, b1, b2 = (boundary_facets(item) for item in (MOVE.old, MOVE.new, FINAL))
    boundary_edges = {tuple(sorted(e)) for facet in b2 for e in combinations(facet, 2)}
    u = transport.reference_coordinates()
    length = math.sqrt(transport.squared_bounds(u)["B"])
    return {"status": "[산출]", "scope": "공급한 실제 경계 2→3 뒤 전방 3→2의 제약 결합과 양자 순서·도메인 대조",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("regge_pachner_creation.py", "regge_pachner_transport.py", "regge_tent_transfer.py")},
            "reference_points": reference_points().tolist(),
            "boundary_moves": {"first_removed": sorted(b0-b1), "first_added": sorted(b1-b0),
                               "second_removed": sorted(b1-b2), "second_added": sorted(b2-b1),
                               "final_internal_edges": sorted(set(FINAL.edges)-boundary_edges)},
            "cases": rows, "exact_certificate": exact,
            "action_mixed_finite_difference": [coarse, fine],
            "action_mixed_error": max(abs(coarse-exact["value"]), abs(fine-exact["value"])),
            "root": float(root), "reference_new_length": float(lengths[Y_ID]),
            "root_residual": float(abs(equation(root))),
            "local_uniqueness": "기준점의 비영 혼합 미분에 따른 국소 유일성만 주장한다",
            "ordering": [ordering_check(u), ordering_check(u+np.linspace(-.13, .17, 14))],
            "naive_domain": {"cutoffs": [1e-1, 1e-3, 1e-6],
                             "exact_truncated_squared_norms": [math.log(1/eta)/(2*length*length) for eta in (1e-1, 1e-3, 1e-6)],
                             "endpoint_behavior": "단위 기준 섬유에서 단순 미분 제약의 제곱노름은 log(1/eta)/(2L^2)로 발산한다",
                             "domain_boundary": "대칭화 미분식은 내부 코어에서만 같고 준비 상태는 단순 미분 연산자의 도메인에 없다"},
            "assumptions": ["공유 길이·유클리드 레게 작용 beta=1과 평탄 여섯 점을 공급한다",
                            "초기 경계 운동량은 실제 이전 복합체 작용의 기울기로 공급한다",
                            "양자 순서 비교는 이전에 공급한 du dy·주기 섬유 도메인의 범위다"],
            "unfinished": ["비평탄 일반 이력의 전체 제약·물리 양자측도·내적·초기 준비·방출·환류",
                           "공통 계량의 동역학적 선택과 0D에서 3+1 Plebanski/Einstein으로 가는 전체 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
