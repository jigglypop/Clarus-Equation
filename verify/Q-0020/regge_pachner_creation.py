"""경계 파흐너 2→3 이동의 새 곡률 모드와 준비·역합성 조건을 검산한다.

길이 x=ell/ell_star와 작용 s=S/hbar는 무차원이다. 공유 길이·유클리드
레게 작용·길이 측도는 공급한다. 이미 접착한 기하에서 준비 선택을 대조하며,
공통 계량 선택이나 물리 에너지·새 상태의 미시적 준비를 유도하지 않는다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from regge_tent_transfer import ReggeComplex


SIMPLEX = ReggeComplex(((0, 1, 2, 3, 4),))
OLD_LOCAL_EDGES = tuple(e for e in SIMPLEX.edges if e != (3, 4))
PI_TRIANGLES = {(0, 1, 2), (0, 3, 4), (1, 3, 4), (2, 3, 4)}
LENGTH_LIMIT = math.sqrt(8/3)


def reference_points():
    """단위 밑면과 두 직교 꼭짓점, 반대편의 이전 내부 꼭짓점을 준다."""
    h = math.sqrt(2/3)
    center = (.5, math.sqrt(3)/6)
    return np.array([[0, 0, 0, 0], [1, 0, 0, 0], [.5, math.sqrt(3)/2, 0, 0],
                     [*center, h, 0], [*center, 0, h], [*center, -.8*h, -.9*h]])


def admissible_interval(old_local_lengths):
    """국소 기존 길이 9개로부터 새 길이의 열린 허용 구간을 계산한다."""
    lengths = np.asarray(old_local_lengths, dtype=float)
    if lengths.shape != (9,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("기존 국소 길이 아홉 개가 유한한 양수여야 한다")
    squared = dict(zip(OLD_LOCAL_EDGES, lengths**2))
    g = np.array([[squared[0, 1], (squared[0, 1]+squared[0, 2]-squared[1, 2])/2],
                  [(squared[0, 1]+squared[0, 2]-squared[1, 2])/2, squared[0, 2]]])
    if np.linalg.eigvalsh(g)[0] <= 0:
        raise ValueError("밑면 삼각형이 비퇴화 유클리드 기하가 아니다")
    projections, heights = [], []
    for j in (3, 4):
        r = np.array([(squared[0, j]+squared[0, i]-squared[i, j])/2 for i in (1, 2)])
        height_squared = squared[0, j]-r @ np.linalg.solve(g, r)
        if height_squared <= 0:
            raise ValueError("접착 사면체의 높이가 양수가 아니다")
        projections.append(r)
        heights.append(math.sqrt(height_squared))
    difference = projections[0]-projections[1]
    parallel_squared = float(difference @ np.linalg.solve(g, difference))
    return (math.sqrt(max(0.0, parallel_squared+(heights[0]-heights[1])**2)),
            math.sqrt(parallel_squared+(heights[0]+heights[1])**2))


def increment(local_lengths, beta=1.0):
    """새 단체의 경계 작용에서 기존 경계 삼각형 여섯 개의 모서리 항을 뺀다."""
    lengths = np.asarray(local_lengths, dtype=float)
    data = SIMPLEX.evaluate(lengths, beta)
    action = data["action"]
    gradient = data["gradient"].copy()
    for triangle, ids, area in zip(SIMPLEX.triangles, SIMPLEX.triangle_edges, data["areas"]):
        if triangle in PI_TRIANGLES:
            continue
        action -= beta*math.pi*area
        squared = lengths[ids]**2
        gradient[ids] -= beta*math.pi*lengths[ids]*(sum(squared)-2*squared)/(8*area)
    return {"action": float(action), "gradient": gradient}


class PachnerCreation:
    """두 접착 사면체를 가진 내부 복합체에 단체 하나를 추가한다."""

    def __init__(self):
        self.old = ReggeComplex(((5, 0, 1, 2, 3), (5, 0, 1, 2, 4)))
        self.new = ReggeComplex(self.old.cells+((0, 1, 2, 3, 4),))
        self.old_ids = self.new.indices(self.old.edges)
        self.local_ids = self.new.indices(SIMPLEX.edges)
        self.local_old_ids = self.old.indices(OLD_LOCAL_EDGES)
        self.new_id = self.new.edge_index[3, 4]

    def lengths_after(self, old_lengths, y):
        old_lengths = np.asarray(old_lengths, dtype=float)
        self.old.evaluate(old_lengths)
        low, high = admissible_interval(old_lengths[self.local_old_ids])
        if not np.isfinite(y) or not low < y < high:
            raise ValueError("새 길이는 비퇴화 열린 구간 안에 있어야 한다")
        lengths = np.empty(len(self.new.edges))
        lengths[self.old_ids], lengths[self.new_id] = old_lengths, y
        return lengths

    def evaluate(self, old_lengths, y, beta=1.0):
        lengths = self.lengths_after(old_lengths, y)
        before = self.old.evaluate(old_lengths, beta)
        after = self.new.evaluate(lengths, beta)
        change = increment(lengths[self.local_ids], beta)
        gradient = np.zeros(len(self.new.edges))
        gradient[self.local_ids] = change["gradient"]
        direct_gradient = after["gradient"].copy()
        direct_gradient[self.old_ids] -= before["gradient"]
        return {"lengths": lengths, "increment": change["action"], "gradient": gradient,
                "action_residual": after["action"]-before["action"]-change["action"],
                "gradient_residual": float(np.linalg.norm(direct_gradient-gradient)),
                "curvature": float(after["deficits"][self.new.triangles.index((0, 1, 2))]),
                "minimum_gram_eigenvalue": after["minimum_gram_eigenvalue"]}

    def create(self, old_lengths, old_momenta, y, beta=1.0):
        momenta = np.asarray(old_momenta, dtype=float)
        if momenta.shape != (len(self.old.edges),) or not np.all(np.isfinite(momenta)):
            raise ValueError("이전 운동량 열네 개가 유한해야 한다")
        data = self.evaluate(old_lengths, y, beta)
        updated = data["gradient"].copy()
        updated[self.old_ids] += momenta
        return data["lengths"], updated

    def undo(self, new_lengths, new_momenta, beta=1.0, tolerance=1e-10):
        """직전 생성을 되돌리며 새 운동량 제약 밖의 입력은 거부한다."""
        self.new.evaluate(new_lengths, beta)
        lengths, momenta = np.asarray(new_lengths, dtype=float), np.asarray(new_momenta, dtype=float)
        if momenta.shape != (len(self.new.edges),) or not np.all(np.isfinite(momenta)):
            raise ValueError("새 운동량 열다섯 개가 유한해야 한다")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("허용 오차는 유한한 음이 아닌 수여야 한다")
        old_lengths = lengths[self.old_ids]
        data = self.evaluate(old_lengths, lengths[self.new_id], beta)
        residual = momenta-data["gradient"]
        if abs(residual[self.new_id]) > tolerance:
            raise ValueError("새 모서리의 운동량 제약을 만족하지 않아 역제거할 수 없다")
        return old_lengths.copy(), residual[self.old_ids]


def preparation(kind, y, length=LENGTH_LIMIT):
    """길이 측도에서 정규화한 세 준비와 그 길이 미분을 준다."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(length) or length <= 0 or not np.all(np.isfinite(y)) or np.any((y < 0) | (y > length)):
        raise ValueError("준비의 길이는 유한한 구간 안에 있어야 한다")
    z = y/length
    if kind == "uniform":
        return np.ones_like(z)/math.sqrt(length), np.zeros_like(z)
    if kind == "first":
        c = math.sqrt(30/length)
        return c*z*(1-z), c*(1-2*z)/length
    if kind == "second":
        c = math.sqrt(105/length)
        return c*z*z*(1-z), c*(2*z-3*z*z)/length
    raise ValueError("알 수 없는 준비 종류다")


def creation_fibers(order=40, beta=1.0):
    """같은 실제 작용 위상과 세 준비를 길이 적분의 직교 구적 벡터로 만든다."""
    if not isinstance(order, int) or not 8 <= order <= 128:
        raise ValueError("구적 차수는 8 이상 128 이하 정수여야 한다")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angle = (nodes+1)*math.pi/4
    y = LENGTH_LIMIT*np.sin(angle)
    weights = weights*(math.pi/4)*LENGTH_LIMIT*np.cos(angle)
    move = PachnerCreation()
    old_lengths = move.old.lengths(reference_points())
    geometry = [move.evaluate(old_lengths, value, beta) for value in y]
    phase = np.exp(1j*np.array([row["increment"] for row in geometry]))
    curvature = np.array([row["curvature"] for row in geometry])
    fibers, records = [], {}
    for kind in ("uniform", "first", "second"):
        chi, derivative = preparation(kind, y)
        fiber = np.sqrt(weights)*chi*phase
        fibers.append(fiber)
        records[kind] = {"norm_squared": float(np.vdot(fiber, fiber).real),
                         "curvature_mean": float(np.dot(weights*chi**2, curvature)),
                         "constraint_residual_squared": float(np.dot(weights, derivative**2))}
    return {"vectors": np.column_stack(fibers), "records": records, "y": y, "weights": weights,
            "curvature": curvature, "max_action_residual": max(abs(row["action_residual"]) for row in geometry)}


def run():
    move = PachnerCreation()
    old_lengths = move.old.lengths(reference_points())
    momenta = np.linspace(-.4, .8, len(old_lengths))
    rows = []
    for z in (.1, .3, 1/math.sqrt(2), .8, .95):
        data = move.evaluate(old_lengths, z*LENGTH_LIMIT)
        after_lengths, after_momenta = move.create(old_lengths, momenta, z*LENGTH_LIMIT)
        restored_lengths, restored_momenta = move.undo(after_lengths, after_momenta)
        rows.append({"z": z, "curvature": data["curvature"], "increment": data["increment"],
                     "new_momentum": float(after_momenta[move.new_id]),
                     "action_residual": data["action_residual"], "gradient_residual": data["gradient_residual"],
                     "curvature_formula_residual": data["curvature"]-(math.pi/2-2*math.asin(z)),
                     "inverse_residual": float(np.linalg.norm(restored_lengths-old_lengths)+np.linalg.norm(restored_momenta-momenta))})
    fibers = creation_fibers()
    finer = creation_fibers(56)
    vectors = fibers["vectors"]
    overlap = np.vdot(vectors[:, 1], vectors[:, 2])
    projector = np.outer(vectors[:, 1], vectors[:, 1].conj())
    projection_loss = np.linalg.norm(vectors[:, 2]-projector @ vectors[:, 2])**2
    exact_curvatures = {"uniform": 2-math.pi/2, "first": 296/15-49*math.pi/8,
                        "second": 1264/35-183*math.pi/16}
    report = {
        "status": "[산출]", "scope": "공급한 경계 2→3 기하와 길이 측도의 조건부 준비 대조",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry_source_sha256": hashlib.sha256(Path(__file__).with_name("regge_tent_transfer.py").read_bytes()).hexdigest(),
        "counts": {"old_edges": len(move.old.edges), "new_edges": len(move.new.edges),
                   "old_triangles": len(move.old.triangles), "new_triangles": len(move.new.triangles)},
        "new_edge_interval": list(admissible_interval(old_lengths[move.local_old_ids])),
        "classical_cases": rows, "preparations": fibers["records"],
        "exact_curvature_means": exact_curvatures,
        "preparation_overlap": [float(overlap.real), float(overlap.imag)],
        "mismatched_inverse_probability": float(abs(overlap)**2),
        "mismatched_projection_loss": float(projection_loss),
        "projector_idempotence_residual": float(np.linalg.norm(projector @ projector-projector)),
        "quadrature_order_convergence": max(abs(fibers["records"][k][f]-finer["records"][k][f])
                                            for k in exact_curvatures for f in fibers["records"][k]),
        "curvature_exact_residual": max(abs(fibers["records"][k]["curvature_mean"]-v) for k, v in exact_curvatures.items()),
        "alternatives": {
            "phase_only": "유한 구간의 균등 준비는 정규화되고 내부 제약 잔차가 0이다. 위상으로 켤레화한 주기 운동량의 자기수반 도메인을 별도로 택하면 허용할 수 있다.",
            "dirichlet_preparations": "두 다항 준비는 각자 등거리 삽입과 영상 위 역을 갖지만 같은 정확한 양자 제약의 해가 아니다. 도함수 노름은 제약 잔차이며 물리 에너지가 아니다.",
            "coarse_graining": "J†J=I는 준비된 영상에서의 복원이다. JJ†는 사영이며 임의 새 상태 전체의 항등연산자가 아니다. 환류·환경 초기화 장치는 아직 공급되지 않았다."},
        "unfinished": ["일반 기존 길이 변동에서 구간·정규화의 의존성과 이전 운동량 제약의 양자 연산자 처리",
                       "물리 측도·경계조건·초기 준비·방출·환류 및 질량·시간의 같은 작용 유도",
                       "공통 계량의 동역학적 선택과 0D에서 3+1 Plebanski/Einstein으로 가는 전체 다리"],
    }
    return report


if __name__ == "__main__":
    output = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True))
