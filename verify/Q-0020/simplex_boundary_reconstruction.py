"""경계 사면체의 면적·각 자료에서 실제 4단체와 계량 부호를 복원한다.

차원, 경계 자료, 면적–각 폐합/접착과 유클리드 또는 로런츠 부문은 공급한다.
기하적 복원은 부호의 동역학적 선택이나 0D 물리 시간 발생의 증명이 아니다.
"""
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import root
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from regge_tent_transfer import ReggeComplex
import regge_lorentz_clock as lorentz

HERE = Path(__file__).resolve().parent
VERTICES = tuple(range(5))
EDGES = tuple(combinations(VERTICES, 2))
EDGE = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = tuple(tuple(v for v in VERTICES if v not in e) for e in EDGES)
TRIANGLE = {t: i for i, t in enumerate(TRIANGLES)}
FACETS = tuple(tuple(v for v in VERTICES if v != k) for k in VERTICES)
ANGLES = tuple((k, i, j) for k in VERTICES for i, j in combinations(FACETS[k], 2))
ANGLE = {key: i for i, key in enumerate(ANGLES)}
AREA0 = math.sqrt(3) / 4
SIMPLEX = ReggeComplex((VERTICES,))


def _squared(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (10,) or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("양의 유한한 변제곱 열 개가 필요하다")
    return x


def gram(values, vertices=VERTICES):
    x = _squared(values)
    vertices = tuple(vertices)
    distances = np.zeros((len(vertices), len(vertices)))
    for i, j in combinations(range(len(vertices)), 2):
        distances[i, j] = distances[j, i] = x[EDGE[tuple(sorted((vertices[i], vertices[j])))]]
    return (distances[0, 1:, None] + distances[None, 0, 1:] - distances[1:, 1:]) / 2


def areas_squared(values):
    x = _squared(values)
    out = []
    for triangle in TRIANGLES:
        sides = x[[EDGE[e] for e in combinations(triangle, 2)]]
        out.append((sides.sum()**2 - 2*sides@sides) / 16)
    return np.array(out)


def area_jacobian(values):
    x = _squared(values)
    jac = np.zeros((10, 10))
    for n, triangle in enumerate(TRIANGLES):
        ids = [EDGE[e] for e in combinations(triangle, 2)]
        sides = x[ids]
        jac[n, ids] = (sides.sum() - 2*sides) / 8
    return jac


def normal_gram(metric):
    inverse = np.linalg.inv(metric)
    n = len(metric)
    normal = np.zeros((n+1, n+1))
    normal[1:, 1:] = inverse
    normal[0, 1:] = normal[1:, 0] = -inverse.sum(axis=0)
    normal[0, 0] = inverse.sum()
    return normal


def intrinsic(values):
    x = _squared(values)
    a2 = areas_squared(x)
    if np.any(a2 <= 0):
        raise ValueError("삼각형 면적은 양수여야 한다")
    cosines = np.zeros(30)
    for k, facet in enumerate(FACETS):
        g = gram(x, facet)
        if np.linalg.eigvalsh(g)[0] <= 0:
            raise ValueError("모든 경계 사면체는 비퇴화 유클리드여야 한다")
        normal = normal_gram(g)
        for i, j in combinations(facet, 2):
            p, q = facet.index(i), facet.index(j)
            cosines[ANGLE[k, i, j]] = -normal[p, q]/math.sqrt(normal[p, p]*normal[q, q])
    return np.r_[np.sqrt(a2)/AREA0, cosines]


def _coordinates(values):
    w = np.asarray(values, dtype=float)
    if w.shape != (40,) or not np.all(np.isfinite(w)):
        raise ValueError("면적과 각 코사인 좌표 40개가 필요하다")
    if np.any(w[:10] <= 0) or np.any(abs(w[10:]) >= 1):
        raise ValueError("양의 면적과 비퇴화 내부 각이 필요하다")
    return w


def angle_value(w, k, i, j):
    return w[10 + ANGLE[k, min(i, j), max(i, j)]]


def triangle_cosine(w, k, i, j, l):
    a = angle_value(w, k, i, j)
    b = angle_value(w, k, i, l)
    c = angle_value(w, k, j, l)
    return (a+b*c)/math.sqrt((1-b*b)*(1-c*c))


def constraints(values):
    w = _coordinates(values)
    closure, matching = [], []
    for k in VERTICES:
        for i in FACETS[k]:
            value = w[EDGE[tuple(sorted((k, i)))]]
            for j in FACETS[k]:
                if j != i:
                    value -= w[EDGE[tuple(sorted((k, j)))]] * angle_value(w, k, i, j)
            closure.append(value)
    for k, l in EDGES:
        rest = [v for v in VERTICES if v not in (k, l)]
        for i, j in combinations(rest, 2):
            matching.append(triangle_cosine(w, k, i, j, l) -
                            triangle_cosine(w, l, i, j, k))
    return np.r_[closure, matching]


def reconstructed_edges(values):
    """면적 법선의 그램과 삼중곱으로 각 사면체의 여섯 길이를 독립 복원한다."""
    w = _coordinates(values)
    occurrences = {edge: [] for edge in EDGES}
    volumes = []
    for k, facet in enumerate(FACETS):
        face_areas = AREA0*np.array([w[EDGE[tuple(sorted((k, i)))]] for i in facet])
        normal = np.diag(face_areas**2)
        for p, q in combinations(range(4), 2):
            normal[p, q] = normal[q, p] = (
                -face_areas[p]*face_areas[q]*angle_value(w, k, facet[p], facet[q]))
        eigenvalues = np.linalg.eigvalsh(normal)
        if eigenvalues[0] < -1e-9 or eigenvalues[1] <= 1e-12:
            raise ValueError("면적 법선은 계수3 양의 반정부호 그램이어야 한다")
        if np.linalg.norm(normal @ np.ones(4)) > 1e-8:
            raise ValueError("면적 법선의 폐합 조건이 필요하다")
        determinant = np.linalg.det(normal[1:, 1:])
        if determinant <= 0:
            raise ValueError("부피가 영인 사면체는 제외한다")
        volume2 = (2/9)*math.sqrt(determinant)
        volumes.append(math.sqrt(volume2))
        for i, j in combinations(facet, 2):
            other = [v for v in facet if v not in (i, j)]
            p, q = [facet.index(v) for v in other]
            cross2 = normal[p, p]*normal[q, q]-normal[p, q]**2
            edge2 = 4*cross2/(9*volume2)
            occurrences[i, j].append(float(edge2))
    mean = np.array([np.mean(occurrences[e]) for e in EDGES])
    mismatch = max(max(v)-min(v) for v in occurrences.values())
    return mean, float(mismatch), volumes


def four_cosines(values):
    w = _coordinates(values)
    candidates = []
    for i, j in EDGES:
        estimates = []
        for k in VERTICES:
            if k in (i, j):
                continue
            a = angle_value(w, k, i, j)
            b = angle_value(w, j, i, k)
            c = angle_value(w, i, j, k)
            estimates.append((a-b*c)/math.sqrt((1-b*b)*(1-c*c)))
        candidates.append(estimates)
    return np.array(candidates)


def boundary_action(values):
    w = _coordinates(values)
    lengths, mismatch, _ = reconstructed_edges(w)
    metric = gram(lengths)
    if mismatch > 1e-8 or np.linalg.eigvalsh(metric)[0] <= 1e-10*np.linalg.norm(metric,2):
        raise ValueError("접착된 비퇴화 유클리드 4단체가 필요하다")
    c = four_cosines(w)
    if np.max(np.ptp(c, axis=1)) > 1e-8 or np.any(abs(c) >= 1):
        raise ValueError("실수 유클리드 4차원 이면각 가지가 필요하다")
    return float(np.dot(AREA0*w[:10], math.pi-np.arccos(c.mean(axis=1))))


def regular_linear_certificate():
    """정규 자료의 유리 계수 행렬을 기호 계산하여 영방향을 구별한다."""
    p = sp.Matrix([[int(set(e).isdisjoint(f)) for f in EDGES] for e in EDGES])
    jac = sp.zeros(50, 40)
    row = 0
    for k in VERTICES:
        for i in FACETS[k]:
            jac[row, EDGE[tuple(sorted((k, i)))]] = 1
            for j in FACETS[k]:
                if j != i:
                    jac[row, EDGE[tuple(sorted((k, j)))]] = -sp.Rational(1, 3)
                    jac[row, 10+ANGLE[k, min(i, j), max(i, j)]] = -1
            row += 1
    for k, l in EDGES:
        rest = [v for v in VERTICES if v not in (k, l)]
        for i, j in combinations(rest, 2):
            for owner, other, sign in ((k, l, 1), (l, k, -1)):
                for a, b, weight in ((i, j, sp.Rational(9, 8)),
                                     (i, other, sp.Rational(9, 16)),
                                     (j, other, sp.Rational(9, 16))):
                    jac[row, 10+ANGLE[owner, min(a,b), max(a,b)]] += sign*weight
            row += 1
    return p, jac


def central_jacobian(function, point, step=1e-5):
    point = np.asarray(point, dtype=float)
    out = []
    for direction in np.eye(len(point)):
        coarse = (function(point+step*direction)-function(point-step*direction))/(2*step)
        fine = (function(point+step*direction/2)-function(point-step*direction/2))/step
        out.append((4*fine-coarse)/3)
    return np.array(out).T


def linear_audit():
    p, exact = regular_linear_certificate()
    w = intrinsic(np.ones(10))
    direct = central_jacobian(constraints, w)
    tangent = central_jacobian(intrinsic, np.ones(10))
    matrix = np.array(exact, dtype=float)
    rank = lambda m: len(DomainMatrix.from_Matrix(m).convert_to(sp.QQ).rref()[1])
    return {
        "petersen_spectrum": {str(k): v for k,v in p.eigenvals().items()},
        "area_squared_jacobian_determinant": str((p/8).det()),
        "closure_rank": rank(exact[:20, :]),
        "matching_rank": rank(exact[20:, :]),
        "combined_rank": rank(exact), "fixed_area_rank": rank(exact[:, 10:]),
        "geometry_tangent_rank": int(np.linalg.matrix_rank(tangent)),
        "constraint_jacobian_error": float(np.max(abs(direct-matrix))),
        "geometry_tangent_constraint_error": float(np.max(abs(matrix@tangent))),
        "area_squared_jacobian_error": float(np.max(abs(area_jacobian(np.ones(10))-np.array(p,dtype=float)/8))),
        "fixed_area_smallest_singular": float(np.linalg.svd(matrix[:,10:], compute_uv=False)[-1]),
    }


def reconstruction_case(squared, label):
    w = intrinsic(squared)
    restored, mismatch, volumes = reconstructed_edges(w)
    spectrum = np.linalg.eigvalsh(gram(squared))
    cosines = four_cosines(w)
    euclidean = spectrum[0] > 1e-10
    action_error = gradient_error = None
    if euclidean:
        original = SIMPLEX.evaluate(np.sqrt(squared))
        action_error = abs(boundary_action(w)-original["action"])
        derivative = central_jacobian(lambda x: np.array([boundary_action(intrinsic(x))]),
                                      squared).reshape(-1)
        expected = original["gradient"]/(2*np.sqrt(squared))
        gradient_error = float(np.max(abs(derivative-expected)))
    return {
        "label": label, "squared": np.asarray(squared).tolist(),
        "constraints_error": float(np.max(abs(constraints(w)))),
        "restored_edge_error": float(np.max(abs(restored-squared))),
        "owner_edge_mismatch": mismatch, "tetrahedron_volumes": volumes,
        "gram_spectrum": spectrum.tolist(), "gram_determinant": float(np.linalg.det(gram(squared))),
        "inertia": [int(np.sum(spectrum < -1e-10)), int(np.sum(abs(spectrum) <= 1e-10)), int(np.sum(spectrum > 1e-10))],
        "four_angle_choice_error": float(np.max(np.ptp(cosines,axis=1))),
        "four_cosines": cosines.mean(axis=1).tolist(),
        "euclidean_action_error": action_error, "euclidean_gradient_error": gradient_error,
    }


def apex_family(r2):
    if not math.isfinite(r2) or r2 <= 1/3:
        raise ValueError("모든 경계 사면체의 양성을 위해 r제곱>1/3이 필요하다")
    return np.array([r2 if 4 in e else 1. for e in EDGES])


def signature_case(r2):
    x = apex_family(r2)
    row = reconstruction_case(x, "apex_"+str(r2))
    expected_side = .75*(r2-1/3)
    side = [np.linalg.det(gram(x, FACETS[k])) for k in range(4)]
    height2 = r2-3/8
    a = 1/(2*math.sqrt(2))
    spatial = a*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[0,0,0]])
    error = orientation_error = None
    if abs(height2) > 1e-12:
        coordinate = np.c_[np.r_[np.zeros(4),math.sqrt(abs(height2))], spatial]
        metric = np.diag([1. if height2>0 else -1.,1.,1.,1.])
        actual = np.array([(coordinate[i]-coordinate[j])@metric@(coordinate[i]-coordinate[j]) for i,j in EDGES])
        reversed_coordinate = coordinate.copy();reversed_coordinate[:,0] *= -1
        reversed_squared = np.array([(reversed_coordinate[i]-reversed_coordinate[j])@metric@(reversed_coordinate[i]-reversed_coordinate[j]) for i,j in EDGES])
        error = float(np.max(abs(actual-x)))
        orientation_error = float(np.max(abs(actual-reversed_squared)))
    row.update({"r2": r2, "height_squared": height2,
                "side_determinant_error": float(max(abs(np.array(side)-expected_side))),
                "full_determinant_error": abs(row["gram_determinant"]-.5*height2),
                "coordinate_reconstruction_error": error,
                "time_reflection_edge_error": orientation_error,
                "all_edges_spacelike": bool(height2 < 0 and np.all(x>0))})
    return row


def unglued_shape_control(eta=.2):
    """한 사면체의 네 면적과 폐합을 유지하면서 공유 삼각형의 모양을 바꾼다."""
    w = intrinsic(np.ones(10))
    axes = np.exp([eta, -eta, 0.])/(2*math.sqrt(2))
    points = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*axes
    face = points[[1,2,3]]
    area = np.linalg.norm(np.cross(face[1]-face[0],face[2]-face[0]))/2
    points *= math.sqrt(AREA0/area)
    x = np.ones(10)
    facet = FACETS[0]
    for a,b in combinations(range(4),2):
        x[EDGE[facet[a],facet[b]]] = np.sum((points[a]-points[b])**2)
    local = gram(x,facet)
    normal = normal_gram(local)
    for a,b in combinations(range(4),2):
        w[10+ANGLE[0,facet[a],facet[b]]] = -normal[a,b]/math.sqrt(normal[a,a]*normal[b,b])
    residual = constraints(w)
    _, mismatch, _ = reconstructed_edges(w)
    return {"eta": eta, "area_change": float(np.max(abs(w[:10]-1))),
            "closure_error": float(np.max(abs(residual[:20]))),
            "matching_defect": float(np.max(abs(residual[20:]))),
            "owner_edge_mismatch": mismatch,
            "changed_tetrahedron_gram_minimum": float(np.linalg.eigvalsh(local)[0])}


def inverse_area_case():
    target = 1+.015*np.sin(np.arange(10)+.3)
    wanted = areas_squared(target)
    solved = root(lambda x: areas_squared(x)-wanted, np.ones(10), jac=area_jacobian, tol=1e-11)
    return {"success": bool(solved.success), "length_error": float(np.max(abs(solved.x-target))),
            "area_error": float(np.max(abs(areas_squared(solved.x)-wanted))),
            "gram_minimum": float(np.linalg.eigvalsh(gram(solved.x))[0])}



def lorentz_action_case(T, v=1., beta=1., coupling=1.):
    """같은 고정 복소 가지의 경계 작용과 시간 크기의 변분을 대조한다."""
    if not all(math.isfinite(z) for z in (T,v,beta,coupling)):
        raise ValueError("유한한 매개변수가 필요하다")
    if not 0 < T < 1/math.sqrt(24) or beta <= 0 or coupling <= 0:
        raise ValueError("0<T<1/sqrt(24), beta>0, coupling>0이 필요하다")
    x = apex_family(3/8-T*T)
    supplied = np.ones(lorentz.N)
    direction = np.zeros(lorentz.N)
    for edge, value in zip(EDGES,x):
        index = lorentz.INDEX[edge]
        supplied[index] = value
        if 4 in edge:
            direction[index] = -2*T
    phi = np.zeros(6)
    phi[4] = v
    native = lorentz.regge(supplied,(VERTICES,))
    scalar = lorentz.scalar(supplied,phi,(VERTICES,))
    alpha = math.acosh(1/math.sqrt(1-24*T*T))
    b = math.acosh((1+8*T*T)/(1-24*T*T))
    side_area = math.sqrt(1-8*T*T)/(4*math.sqrt(2))
    expected = -4*AREA0*alpha+6*side_area*b
    derivative = -6*math.sqrt(2)*T*b/math.sqrt(1-8*T*T)
    scalar_expected = v*v/(48*math.sqrt(2)*T)
    scalar_derivative = -v*v/(48*math.sqrt(2)*T*T)
    actual_derivative = float(native["gradient"].real@direction)
    actual_scalar_derivative = float(scalar["gradient"]@direction)
    def action_at(t):
        values = supplied.copy()
        for edge in EDGES:
            if 4 in edge:
                values[lorentz.INDEX[edge]] = 3/8-t*t
        return np.array([lorentz.regge(values,(VERTICES,))["action"].real])
    difference = float(central_jacobian(lambda z: action_at(z[0]), [T], step=1e-6)[0,0])
    return {
        "T":T, "v":v, "beta":beta, "coupling":coupling,
        "regge_real":float(native["action"].real),
        "regge_imaginary":float(native["action"].imag),
        "regge_action_error":abs(native["action"].real-expected),
        "regge_gradient_error":abs(actual_derivative-derivative),
        "regge_difference_error":abs(difference-derivative),
        "scalar_action_error":abs(scalar["action"]-scalar_expected),
        "scalar_gradient_error":abs(actual_scalar_derivative-scalar_derivative),
        "scalar_volume_error":abs(scalar["volumes"][0]-T/(24*math.sqrt(2))),
        "scalar_norm_error":abs(scalar["norms"][0]+v*v/(T*T)),
        "scalar_norm_relative_error":abs(scalar["norms"][0]+v*v/(T*T))/max(1.,v*v/(T*T)),
        "total_derivative":beta*actual_derivative+coupling*actual_scalar_derivative,
        "derivative_formula":beta*derivative+coupling*scalar_derivative,
    }

def run():
    return {
        "status": "면적–각 복원과 부호의 조건부 판정; 동역학적 선택은 미완성",
        "linear": linear_audit(),
        "reconstruction": [reconstruction_case(x,label) for x,label in (
            (np.ones(10),"regular"),(1+.02*np.sin(np.arange(10)+.2),"perturbed"))],
        "signature": [signature_case(r2) for r2 in (17/48, 3/8, .4, 1.)],
        "unglued": [unglued_shape_control(e) for e in (.05,.2)],
        "inverse_area": inverse_area_case(),
        "lorentz_action": [lorentz_action_case(t,v) for t in (.03,.1,1/math.sqrt(48),.19) for v in (0.,1.)],
        "dependencies": {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in (
            "simplex_boundary_reconstruction.py","regge_tent_transfer.py","regge_lorentz_clock.py")},
        "sources": ["https://arxiv.org/abs/0802.0864v3","https://arxiv.org/abs/0907.2440"],
    }


if __name__ == "__main__":
    print(json.dumps(run(),ensure_ascii=False,indent=2))
