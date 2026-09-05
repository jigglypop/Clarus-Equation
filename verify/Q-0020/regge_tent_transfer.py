"""평탄한 천막 이동의 실제 레게 작용에서 경계 정준 전달과 두 단계 합성을 검산한다.

모든 길이는 기준 길이로 나눈 값이다. 작용의 계수 beta와 기존 공통 길이,
유클리드 가지, 진동 적분에 의한 양자화는 공급 조건이다. 공통 계량 선택이나
기존 분할 V, 로런츠 시간, 양의 유클리드 확률 측도를 유도하지 않는다.
"""

from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

import numpy as np


LINK = tuple((i, j, p) for i, j in ((0, 1), (1, 2), (0, 2)) for p in (3, 4))
LOCAL_EDGES = tuple(combinations(range(5), 2))
LOCAL_TRIANGLES = tuple(combinations(range(5), 3))
COMPLEMENTS = tuple(tuple(i for i in range(5) if i not in t) for t in LOCAL_TRIANGLES)
RTOL = 1e-8


def reference_points(bend=3.0):
    """마지막 세 점은 이전·중간·다음 정점이며 bend는 이웃 경계의 굽음이다."""
    if not np.isfinite(bend):
        raise ValueError("경계 굽음은 유한해야 한다")
    points = np.array([
        [1, 0, 0, .03], [-.5, math.sqrt(3)/2, 0, -.02],
        [-.5, -math.sqrt(3)/2, 0, .04], [0, 0, 1, .01], [0, 0, -1, -.03],
        [.08, -.04, .03, 0], [.1, .02, -.02, .5], [.05, .05, .01, 1.0],
    ])
    points[:5, 3] *= bend
    return points


def tent_cells(old=5, new=6):
    return tuple((old, new, *face) for face in LINK)


class ReggeComplex:
    """주어진 단체 복합체에서 내부·경계 삼각형을 직접 세어 작용을 계산한다."""

    def __init__(self, cells):
        self.cells = tuple(tuple(c) for c in cells)
        if not self.cells or any(len(c) != 5 or len(set(c)) != 5 for c in self.cells):
            raise ValueError("각 단체에는 서로 다른 꼭짓점 다섯 개가 필요하다")
        facets = Counter(tuple(sorted(f)) for c in self.cells for f in combinations(c, 4))
        if max(facets.values()) > 2:
            raise ValueError("하나의 사면체를 셋 이상의 단체가 공유한다")
        self.edges = sorted({tuple(sorted(e)) for c in self.cells for e in combinations(c, 2)})
        self.triangles = sorted({tuple(sorted(t)) for c in self.cells for t in combinations(c, 3)})
        boundary = {tuple(sorted(t)) for f, n in facets.items() if n == 1 for t in combinations(f, 3)}
        self.boundary = np.array([t in boundary for t in self.triangles])
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        triangle_index = {t: i for i, t in enumerate(self.triangles)}
        self.cell_edges = [self.indices(combinations(c, 2)) for c in self.cells]
        self.cell_triangles = [np.array([triangle_index[tuple(sorted(t))] for t in combinations(c, 3)]) for c in self.cells]
        self.triangle_edges = [self.indices(combinations(t, 2)) for t in self.triangles]

    def indices(self, edges):
        return np.array([self.edge_index[tuple(sorted(e))] for e in edges], dtype=int)

    def lengths(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 4 or not np.all(np.isfinite(points)):
            raise ValueError("꼭짓점은 유한한 4차원 좌표여야 한다")
        return np.array([np.linalg.norm(points[i]-points[j]) for i, j in self.edges])

    def evaluate(self, lengths, beta=1.0):
        lengths = np.asarray(lengths, dtype=float)
        if lengths.shape != (len(self.edges),) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("모든 모서리 길이는 유한한 양수여야 한다")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("작용 계수는 유한한 양수여야 한다")
        deficits = np.where(self.boundary, math.pi, 2*math.pi)
        minimum = math.inf
        dihedrals = []
        for edge_ids, triangle_ids in zip(self.cell_edges, self.cell_triangles):
            squared_distances = np.zeros((5, 5))
            for (i, j), length in zip(LOCAL_EDGES, lengths[edge_ids]):
                squared_distances[i, j] = squared_distances[j, i] = length*length
            gram = (squared_distances[0, 1:, None]+squared_distances[None, 0, 1:]-squared_distances[1:, 1:])/2
            minimum = min(minimum, float(np.linalg.eigvalsh(gram)[0]))
            if minimum <= 0:
                raise ValueError("비퇴화 유클리드 단체가 아니다")
            inverse = np.linalg.inv(gram)
            normals = np.empty((5, 5))
            normals[1:, 1:] = inverse
            normals[0, 1:] = normals[1:, 0] = -inverse.sum(axis=0)
            normals[0, 0] = inverse.sum()
            angles = np.array([math.acos(float(np.clip(-normals[i, j]/math.sqrt(normals[i, i]*normals[j, j]), -1, 1))) for i, j in COMPLEMENTS])
            deficits[triangle_ids] -= angles
            dihedrals.append(angles)
        gradient = np.zeros(len(lengths))
        areas = []
        for ids, deficit in zip(self.triangle_edges, deficits):
            squared = lengths[ids]**2
            area_squared = (2*(squared[0]*squared[1]+squared[0]*squared[2]+squared[1]*squared[2])-np.dot(squared, squared))/16
            if area_squared <= 0:
                raise ValueError("삼각형 면적이 양수가 아니다")
            area = math.sqrt(area_squared)
            areas.append(area)
            # 슐레플리 항등식으로 각도의 미분 항이 단체마다 상쇄된다.
            gradient[ids] += beta*deficit*lengths[ids]*(sum(squared)-2*squared)/(8*area)
        return {"action": beta*float(np.dot(areas, deficits)), "gradient": gradient,
                "deficits": deficits, "areas": np.array(areas), "minimum_gram_eigenvalue": minimum,
                "dihedrals": np.array(dihedrals)}

    def hessian(self, lengths, ids, step=1e-4, beta=1.0):
        """해석적 1차 미분을 두 간격에서 차분하고 리처드슨 보정한다."""
        if not np.isfinite(step) or step <= 0:
            raise ValueError("차분 간격은 유한한 양수여야 한다")
        lengths, ids = np.asarray(lengths, dtype=float), np.asarray(ids, dtype=int)
        def central(delta):
            columns = []
            for index in ids:
                displacement = np.zeros(len(lengths))
                displacement[index] = delta
                columns.append((self.evaluate(lengths+displacement, beta)["gradient"][ids]-self.evaluate(lengths-displacement, beta)["gradient"][ids])/(2*delta))
            return np.column_stack(columns)
        raw = (4*central(step/2)-central(step))/3
        return (raw+raw.T)/2, float(np.linalg.norm(raw-raw.T))


def complement(gauge):
    """실제 정점 변위의 열공간을 제거하며 특이 배경은 거부한다."""
    u, singular, _ = np.linalg.svd(gauge, full_matrices=True)
    if len(singular) != 4 or singular[-1] <= RTOL*singular[0]:
        raise ValueError("정점 변위의 계수가 4보다 작다")
    basis = u[:, 4:]
    for col in range(basis.shape[1]):
        pivot = np.argmax(np.abs(basis[:, col]))
        if basis[pivot, col] < 0:
            basis[:, col] *= -1
    return basis, singular


def scalar_map(coefficients):
    a, b, c = np.asarray(coefficients, dtype=float)
    if not np.all(np.isfinite([a, b, c])) or abs(b) <= RTOL*max(abs(a), abs(b), abs(c)):
        raise ValueError("혼합 미분이 사라진 가지에는 이 전달식을 쓸 수 없다")
    return np.array([[-a/b, -1/b], [b-c*a/b, -c/b]])


def compose_coefficients(first, second):
    a, b, c = first
    aa, bb, cc = second
    d = c+aa
    if abs(d) <= RTOL*max(abs(c), abs(aa), abs(b), abs(bb)):
        raise ValueError("중간 물리 방향이 특이하여 제거할 수 없다")
    return np.array([a-b*b/d, -b*bb/d, cc-bb*bb/d]), float(d)


def gaussian_transfer(coefficients, width, amplitude):
    """별도 채택한 진동 커널을 정규화 가능한 가우스 입력에 작용시킨다."""
    a, b, c = np.asarray(coefficients, dtype=float)
    scalar_map(coefficients)
    if not np.isfinite(width) or complex(width).real <= 0 or not np.isfinite(amplitude):
        raise ValueError("가우스 폭의 실수부는 양수이고 진폭은 유한해야 한다")
    denominator = complex(width)-1j*a
    return b*b/denominator-1j*c, amplitude*math.sqrt(abs(b))/np.sqrt(denominator)


def one_step(points=None, old=5, new=6, step=1e-4, beta=1.0, lengths=None):
    points = reference_points() if points is None else np.asarray(points, dtype=float)
    complex_ = ReggeComplex(tent_cells(old, new))
    embedded = complex_.lengths(points)
    lengths = embedded if lengths is None else np.asarray(lengths, dtype=float)
    evaluation = complex_.evaluate(lengths, beta)
    ids = complex_.indices([(i, old) for i in range(5)]+[(i, new) for i in range(5)]+[(old, new)])
    flat = float(np.max(np.abs(evaluation["deficits"][~complex_.boundary])))
    pole_gradient = float(evaluation["gradient"][ids[-1]])
    if flat > 1e-9 or abs(pole_gradient) > 1e-9*beta:
        raise ValueError("평탄한 내부와 천막변 정상 조건이 성립하지 않는다")
    if not np.allclose(lengths, embedded, rtol=1e-12, atol=1e-12):
        raise ValueError("변위 게이지와 길이는 같은 매장 배경에서 와야 한다")
    hessian, skew = complex_.hessian(lengths, ids, step, beta)
    if abs(hessian[-1, -1]) <= RTOL*np.linalg.norm(hessian):
        raise ValueError("천막변의 2차 미분이 특이하여 제거할 수 없다")
    effective = hessian[:-1, :-1]-np.outer(hessian[:-1, -1], hessian[-1, :-1])/hessian[-1, -1]
    yin = np.array([(points[old]-points[i])/lengths[complex_.edge_index[tuple(sorted((i, old)))]] for i in range(5)])
    yout = np.array([(points[new]-points[i])/lengths[complex_.edge_index[tuple(sorted((i, new)))]] for i in range(5)])
    ein, sin = complement(yin)
    eout, sout = complement(yout)
    ein, eout = ein[:, 0], eout[:, 0]
    a, b, c = effective[:5, :5], effective[:5, 5:], effective[5:, 5:]
    mixed_singular = np.linalg.svd(b, compute_uv=False)
    residual = max(np.linalg.norm(yin.T @ b), np.linalg.norm(b @ yout))
    if residual > RTOL*max(1, np.linalg.norm(b)) or mixed_singular[1] > RTOL*mixed_singular[0]:
        raise ArithmeticError("혼합 경계 미분의 네 게이지 방향이 소거되지 않았다")
    coefficients = np.array([ein @ a @ ein, ein @ b @ eout, eout @ c @ eout])
    return {"complex": complex_, "lengths": lengths, "ids": ids, "evaluation": evaluation,
            "hessian": hessian, "effective": effective, "hessian_skew": skew,
            "yin": yin, "yout": yout, "ein": ein, "eout": eout,
            "gauge_singular_values": [sin, sout], "mixed_singular_values": mixed_singular,
            "mixed_gauge_residual": float(residual), "bulk_deficit_residual": flat,
            "pole_gradient": pole_gradient, "coefficients": coefficients, "map": scalar_map(coefficients)}


def two_step(points=None, step=1e-4, beta=1.0):
    """12단체의 직접 작용과 17차 미분을 단계별 제거와 독립 대조한다."""
    points = reference_points() if points is None else np.asarray(points, dtype=float)
    first = one_step(points, 5, 6, step, beta)
    second = one_step(points, 6, 7, step, beta)
    complex_ = ReggeComplex(tent_cells(5, 6)+tent_cells(6, 7))
    lengths = complex_.lengths(points)
    edges = [(i, vertex) for vertex in (5, 6, 7) for i in range(5)]+[(5, 6), (6, 7)]
    ids = complex_.indices(edges)
    evaluation = complex_.evaluate(lengths, beta)
    full, skew = complex_.hessian(lengths, ids, step, beta)
    assembled = np.zeros((17, 17))
    first_ids = [*range(10), 15]
    second_ids = [*range(5, 15), 16]
    assembled[np.ix_(first_ids, first_ids)] += first["hessian"]
    assembled[np.ix_(second_ids, second_ids)] += second["hessian"]
    corners = [complex_.triangles.index(tuple(sorted(face))) for face in LINK]
    corner = beta*math.pi*sum(evaluation["areas"][corners])
    action_residual = first["evaluation"]["action"]+second["evaluation"]["action"]-corner-evaluation["action"]

    outer = np.array([*range(5), *range(10, 15)])
    inner = np.array([*range(5, 10), 15, 16])
    hin = full[np.ix_(inner, inner)]
    cross = full[np.ix_(outer, inner)]
    gauge = np.vstack([first["yout"],
                       (points[6]-points[5])/np.linalg.norm(points[6]-points[5]),
                       (points[6]-points[7])/np.linalg.norm(points[6]-points[7])])
    basis, singular = complement(gauge)
    physical = basis.T @ hin @ basis
    eigenvalues = np.linalg.eigvalsh(physical)
    if min(abs(eigenvalues)) <= RTOL*np.linalg.norm(hin):
        raise ValueError("내부 물리 부분공간이 특이하여 제거할 수 없다")
    gauge_residual = max(np.linalg.norm(hin @ gauge), np.linalg.norm(cross @ gauge))
    if gauge_residual > RTOL*np.linalg.norm(full):
        raise ArithmeticError("합친 내부 정점의 게이지 방향이 소거되지 않았다")
    projected_cross = cross @ basis
    direct_outer = full[np.ix_(outer, outer)]-projected_cross @ np.linalg.solve(physical, projected_cross.T)

    # 천막변을 먼저 제거하고 중간 별의 물리 방향 하나만 제거한다.
    reduced = np.zeros((15, 15))
    reduced[:10, :10] += first["effective"]
    reduced[5:, 5:] += second["effective"]
    middle = reduced[5:10, 5:10]
    emid = first["eout"]
    d = float(emid @ middle @ emid)
    coefficients, denominator = compose_coefficients(first["coefficients"], second["coefficients"])
    middle_cross = reduced[np.ix_(outer, range(5, 10))] @ emid
    sequential_outer = reduced[np.ix_(outer, outer)]-np.outer(middle_cross, middle_cross)/d
    quotient = np.zeros((10, 2))
    quotient[:5, 0], quotient[5:, 1] = first["ein"], second["eout"]
    direct_scalar = quotient.T @ direct_outer @ quotient
    direct_coefficients = np.array([direct_scalar[0, 0], direct_scalar[0, 1], direct_scalar[1, 1]])
    maslov = np.exp(1j*math.pi*np.sign(d)/4)
    width0, amplitude0 = .8+.3j, (.8/math.pi)**.25
    width1, amplitude1 = gaussian_transfer(first["coefficients"], width0, amplitude0)
    width2, amplitude2 = gaussian_transfer(second["coefficients"], width1, amplitude1)
    width_direct, amplitude_direct = gaussian_transfer(coefficients, width0, amplitude0)
    sample = np.linspace(-2, 2, 17)
    gaussian_residual = np.max(abs(amplitude2*np.exp(-width2*sample**2/2)-maslov*amplitude_direct*np.exp(-width_direct*sample**2/2)))
    return {"first": first, "second": second, "complex": complex_, "lengths": lengths,
            "evaluation": evaluation, "ids": ids, "full_hessian": full,
            "direct_outer": direct_outer, "sequential_outer": sequential_outer,
            "internal_gauge": gauge, "internal_physical_eigenvalues": eigenvalues,
            "internal_gauge_singular_values": singular, "corner": float(corner),
            "coefficients": coefficients, "direct_coefficients": direct_coefficients,
            "middle_denominator": denominator, "middle_eigenvalues": np.linalg.eigvalsh(middle),
            "maslov_phase": maslov,
            "residuals": {
                "action_with_corner": abs(float(action_residual)),
                "assembled_hessian": float(np.linalg.norm(full-assembled)),
                "global_hessian_skew": skew,
                "global_bulk_deficits": float(max(abs(evaluation["deficits"][~complex_.boundary]))),
                "internal_stationarity": float(np.linalg.norm(evaluation["gradient"][ids[inner]])),
                "internal_gauge": float(gauge_residual),
                "middle_gauge": float(np.linalg.norm(middle @ first["yout"])),
                "schur_order": float(np.linalg.norm(direct_outer-sequential_outer)),
                "coefficients": float(np.linalg.norm(direct_coefficients-coefficients)),
                "canonical_composition": float(np.linalg.norm(scalar_map(coefficients)-second["map"] @ first["map"])),
                "gaussian_with_maslov": float(gaussian_residual),
                "gaussian_norm": abs(float(abs(amplitude2)**2*math.sqrt(math.pi/width2.real))-1),
            }}


def _step_report(result):
    return {"coefficients": result["coefficients"].tolist(), "canonical_map": result["map"].tolist(),
            "minimum_gram_eigenvalue": result["evaluation"]["minimum_gram_eigenvalue"],
            "bulk_deficit_residual": result["bulk_deficit_residual"],
            "pole_gradient": result["pole_gradient"], "pole_hessian": float(result["hessian"][-1, -1]),
            "mixed_singular_values": result["mixed_singular_values"].tolist(),
            "gauge_singular_values": [value.tolist() for value in result["gauge_singular_values"]],
            "mixed_gauge_residual": result["mixed_gauge_residual"],
            "hessian_skew": result["hessian_skew"],
            "raw_boundary_gradient_contractions": [float(np.linalg.norm(result["yin"].T @ result["evaluation"]["gradient"][result["ids"][:5]])),
                                                   float(np.linalg.norm(result["yout"].T @ result["evaluation"]["gradient"][result["ids"]][5:10]))]}


def run():
    result = two_step()
    finer = one_step(step=5e-5)
    convergence = float(np.linalg.norm(finer["coefficients"]-result["first"]["coefficients"]))
    if max(result["residuals"].values()) > 1e-7 or convergence > 1e-7:
        raise ArithmeticError("참조 배경의 합성 또는 차분 수렴 오차가 허용 범위를 넘었다")
    planar = ReggeComplex(tent_cells())
    planar_lengths = planar.lengths(reference_points(0))
    planar_evaluation = planar.evaluate(planar_lengths)
    try:
        one_step(reference_points(0))
    except ValueError as error:
        planar_rejection = str(error)
    else:
        raise AssertionError("특이 천막변 대조군을 거부하지 않았다")
    offshell = result["first"]["lengths"].copy()
    offshell[result["first"]["ids"][-1]] += .001
    try:
        one_step(lengths=offshell)
    except ValueError as error:
        offshell_rejection = str(error)
    else:
        raise AssertionError("정상 조건 밖의 천막변 대조군을 거부하지 않았다")
    source = Path(__file__)
    return {
        "scope": {
            "supplied_flat_euclidean_regge_action": True,
            "one_physical_boundary_direction_numerically_verified": True,
            "two_step_global_action_and_quotient_composition": True,
            "oscillatory_quantization_separately_supplied": True,
            "euclidean_middle_gaussian_converges": bool(result["middle_denominator"] > 0),
            "existing_split_V_derived": False, "physical_clock_or_mass_derived": False,
            "common_metric_selected": False, "lorentzian_einstein_limit_derived": False,
        },
        "conventions": {"lengths": "ell/ell_star", "action": "S/hbar = beta * sum(area*deficit)",
                        "beta": 1.0, "physical_beta": "ell_star^2/(8*pi*ell_P^2)",
                        "boundary_angle": "pi-sum(theta)", "bulk_angle": "2*pi-sum(theta)",
                        "basis_sign": "largest absolute component positive",
                        "kernel": "sqrt(abs(b)/(2*pi))*exp(i*(a*Q0^2+2*b*Q0*Q1+c*Q1^2)/2)",
                        "background_action_corner": "Sglobal=S1+S2-beta*pi*sum(fixed_link_areas)",
                        "kernel_background": "quadratic perturbations; background action and linear momenta removed"},
        "geometry": {"points": reference_points().tolist(), "link": LINK,
                     "one_step_counts": [6, len(result["first"]["complex"].edges), len(result["first"]["complex"].triangles)],
                     "two_step_counts": [12, len(result["complex"].edges), len(result["complex"].triangles)],
                     "one_step_dynamic_variables": 11, "two_step_dynamic_variables": 17},
        "first": _step_report(result["first"]), "second": _step_report(result["second"]),
        "composition": {"coefficients": result["coefficients"].tolist(),
                        "middle_denominator": result["middle_denominator"],
                        "middle_eigenvalues": result["middle_eigenvalues"].tolist(),
                        "internal_physical_eigenvalues": result["internal_physical_eigenvalues"].tolist(),
                        "maslov_phase": [float(result["maslov_phase"].real), float(result["maslov_phase"].imag)],
                        "fixed_corner": result["corner"], "residuals": result["residuals"]},
        "hessian_step_convergence": convergence,
        "controls": {"planar_minimum_gram_eigenvalue": planar_evaluation["minimum_gram_eigenvalue"],
                     "planar_rejected": planar_rejection, "offshell_rejected": offshell_rejection},
        "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "sources": ["https://arxiv.org/abs/1108.1974v2", "https://arxiv.org/abs/1411.5672v2"],
    }


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
