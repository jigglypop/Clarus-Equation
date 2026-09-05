"""실제 네 단체의 내부 변을 적분해 경계 작용과 두 공급 측도를 대조한다.

길이 e=ell/ell_star와 단위 계수 작용 s는 무차원이다. beta와 진동 부호,
공유 길이 및 정규화 측도는 공급 조건이다. 전체 구간 적분과 정상위상
근사는 구별하며, 평탄 해를 물리 측도나 공통 계량 선택으로 승격하지 않는다.
"""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre

from regge_pachner_constraints import FINAL, E_ID, reference_points
from regge_pachner_creation import OLD_LOCAL_EDGES, admissible_interval
from regge_tent_transfer import ReggeComplex


COARSE = ReggeComplex(((0, 2, 3, 4, 5), (1, 2, 3, 4, 5)))
BOUNDARY_IDS = FINAL.indices(COARSE.edges)
LIMIT = math.sqrt(13)/3
INTERNAL_HESSIAN = -9*math.sqrt(3)/2


def internal_interval(boundary):
    """네 단체의 독립 열린 구간을 교차해 실제 내부 변의 범위를 구한다."""
    COARSE.evaluate(boundary)
    values = dict(zip(COARSE.edges, boundary))
    intervals = []
    for cell in FINAL.cells:
        rim = sorted(set(cell)-{0, 1})
        vertices = (*rim, 0, 1)
        lengths = [values[tuple(sorted((vertices[i], vertices[j])))]
                   for i, j in OLD_LOCAL_EDGES]
        intervals.append(admissible_interval(lengths))
    lower, upper = max(a for a, _ in intervals), min(b for _, b in intervals)
    if not lower < upper:
        raise ValueError("네 단체가 공유하는 내부 길이 구간이 없다")
    return lower, upper


def flat_completion(boundary):
    """공유 사면체의 반대편 두 꼭짓점으로 평탄 내부 길이를 복원한다."""
    boundary = np.asarray(boundary, dtype=float)
    COARSE.evaluate(boundary)
    squared = dict(zip(COARSE.edges, boundary**2))
    def distance(i, j):
        return 0 if i == j else squared[tuple(sorted((i, j)))]
    vertices = (3, 4, 5)
    gram = np.array([[(distance(2, i)+distance(2, j)-distance(i, j))/2
                      for j in vertices] for i in vertices])
    lower = np.linalg.cholesky(gram)
    projections, heights = [], []
    for apex in (0, 1):
        inner = np.array([(distance(2, apex)+distance(2, i)-distance(apex, i))/2
                          for i in vertices])
        projection = np.linalg.solve(lower, inner)
        height_squared = distance(2, apex)-projection @ projection
        if height_squared <= 0:
            raise ValueError("공유 사면체에 대한 꼭짓점 높이가 양수가 아니다")
        projections.append(projection)
        heights.append(math.sqrt(height_squared))
    crossing = (heights[1]*projections[0]+heights[0]*projections[1])/sum(heights)
    coordinates = np.linalg.solve(lower.T, crossing)
    barycentric = np.r_[1-sum(coordinates), coordinates]
    if np.min(barycentric) <= 0:
        raise ValueError("내부 변이 공유 사면체의 내부를 통과하지 않는다")
    edge = math.sqrt(np.linalg.norm(projections[0]-projections[1])**2+sum(heights)**2)
    lengths = np.empty(len(FINAL.edges))
    lengths[BOUNDARY_IDS], lengths[E_ID] = boundary, edge
    FINAL.evaluate(lengths)
    return lengths, barycentric


def symmetric_action(edge):
    """고정 대칭 경계의 작용을 세 종류의 면적·이면각으로 계산한다."""
    edge = np.asarray(edge, dtype=float)
    if not np.all(np.isfinite(edge)) or np.any((edge < 0) | (edge > LIMIT)):
        raise ValueError("대칭 내부 길이는 닫힌 허용 구간 안에 있어야 한다")
    remainder = np.sqrt(np.maximum(13-9*edge**2, 0))
    internal_angle = np.arctan2(math.sqrt(3)*remainder*np.sqrt(5-edge**2), -(3*edge**2+1))
    spoke_angle = np.arctan2(math.sqrt(7)*remainder, 2*math.sqrt(6)*edge)
    rim_angle = 2*np.arctan2(3*edge, remainder)
    area = edge*np.sqrt(5-edge**2)/4
    return (4*area*(2*math.pi-3*internal_angle)
            +2*math.sqrt(14)*(math.pi-2*spoke_angle)
            +8*math.sqrt(3)/3*(math.pi-rim_angle))


def symmetric_gradient(edge):
    symmetric_action(edge)
    if edge == LIMIT:
        # 퇴화 끝점의 역코사인 반올림 오차 대신 해석적 극한을 사용한다.
        return -19*math.pi/(12*math.sqrt(2))
    angle = math.acos(float(np.clip(-(3*edge**2+1)/(2*(7-3*edge**2)), -1, 1)))
    return (5-2*edge**2)/math.sqrt(5-edge**2)*(2*math.pi-3*angle)


def density(edge, kind):
    if kind == "length":
        return np.ones_like(np.asarray(edge))/LIMIT
    if kind == "squared":
        return 2*np.asarray(edge)/LIMIT**2
    raise ValueError("길이 또는 길이제곱 측도만 지원한다")


@lru_cache(maxsize=4)
def angular_rule(order):
    nodes, weights = roots_legendre(order)
    return (nodes+1)*math.pi/4, weights*math.pi/4


def integral(beta, kind, *, order=None, decaying=False):
    """전체 유한 구간 적분이다. 감쇠 대조만 좌끝 지수를 분리해 반환한다."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta는 유한한 음이 아닌 수여야 한다")
    density(1.0, kind)
    if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 16):
        raise ValueError("구적 차수는 16 이상 정수여야 한다")
    reference = float(symmetric_action(0 if decaying else 1))
    def integrand(angle):
        edge = LIMIT*np.sin(angle)
        shift = symmetric_action(edge)-reference
        weight = density(edge, kind)*LIMIT*np.cos(angle)
        return weight*np.exp(-beta*shift if decaying else 1j*beta*shift)
    if order is not None:
        angles, weights = angular_rule(order)
        reduced, error = np.dot(weights, integrand(angles)), None
    else:
        options = {"epsabs": 2e-11, "epsrel": 2e-11, "limit": 1500,
                   "points": [math.asin(1/LIMIT)]}
        real, er = quad(lambda t: float(np.real(integrand(t))), 0, math.pi/2, **options)
        imag, ei = (0.0, 0.0) if decaying else quad(
            lambda t: float(np.imag(integrand(t))), 0, math.pi/2, **options)
        reduced, error = real+1j*imag, er+ei
    # 감쇠 적분은 exp(beta*s(0))를 곱한 값이다. 진동 적분은 전체 위상을 복구한다.
    return complex(reduced if decaying else reduced*np.exp(1j*beta*reference)), error


def stationary_terms(beta, kind):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("정상위상 근사는 양의 beta가 필요하다")
    saddle = (density(1.0, kind)*np.exp(1j*beta*symmetric_action(1)-1j*math.pi/4)
              *math.sqrt(2*math.pi/(beta*abs(INTERNAL_HESSIAN))))
    endpoints = ((density(LIMIT, kind)*np.exp(1j*beta*symmetric_action(LIMIT))/symmetric_gradient(LIMIT)
                  -density(0.0, kind)*np.exp(1j*beta*symmetric_action(0))/symmetric_gradient(0))/(1j*beta))
    return complex(saddle), complex(endpoints)


def passive_squared_integral(beta):
    """같은 길이 측도를 z=e²로 옮기면 야코비안이 적분을 보존한다."""
    def integrand(z):
        return np.exp(1j*beta*symmetric_action(math.sqrt(z)))/(2*LIMIT*math.sqrt(z))
    options = {"epsabs": 2e-10, "epsrel": 2e-10, "limit": 600}
    real = quad(lambda z: float(integrand(z).real), 0, LIMIT**2, **options)[0]
    imag = quad(lambda z: float(integrand(z).imag), 0, LIMIT**2, **options)[0]
    return complex(real, imag)


def classical_case(boundary):
    lengths, barycentric = flat_completion(boundary)
    fine, coarse = FINAL.evaluate(lengths), COARSE.evaluate(boundary)
    hf, skew_f = FINAL.hessian(lengths, np.arange(len(lengths)), step=2e-5)
    if abs(hf[E_ID, E_ID]) <= 1e-8*np.linalg.norm(hf):
        raise ValueError("내부 헤시안이 특이하여 슈어 제거를 적용할 수 없다")
    hc, skew_c = COARSE.hessian(boundary, np.arange(len(boundary)), step=2e-5)
    mixed = hf[BOUNDARY_IDS, E_ID]
    schur = hf[np.ix_(BOUNDARY_IDS, BOUNDARY_IDS)]-np.outer(mixed, mixed)/hf[E_ID, E_ID]
    return {"internal_length": float(lengths[E_ID]), "interval": list(internal_interval(boundary)),
            "minimum_crossing_barycentric": float(min(barycentric)),
            "action": float(fine["action"]), "action_residual": float(abs(fine["action"]-coarse["action"])),
            "gradient_residual": float(np.linalg.norm(fine["gradient"][BOUNDARY_IDS]-coarse["gradient"])),
            "internal_equation_residual": float(abs(fine["gradient"][E_ID])),
            "flatness_residual": float(np.max(np.abs(fine["deficits"][~FINAL.boundary]))),
            "internal_hessian": float(hf[E_ID, E_ID]),
            "boundary_schur_residual": float(np.linalg.norm(schur-hc)),
            "hessian_skew": max(skew_f, skew_c)}


def exact_certificate():
    import sympy as sp
    edge = sp.Symbol("e", positive=True)
    square = edge**2
    gram = sp.Matrix([[square, square/2, square/2, square/2],
                      [square/2, sp.Rational(5, 4), -sp.Rational(1, 12), -sp.Rational(1, 12)],
                      [square/2, -sp.Rational(1, 12), sp.Rational(5, 4), -sp.Rational(1, 12)],
                      [square/2, -sp.Rational(1, 12), -sp.Rational(1, 12), sp.Rational(5, 4)]])
    cosine = -(3*square+1)/(2*(7-3*square))
    gradient = (5-2*square)/sp.sqrt(5-square)*(2*sp.pi-3*sp.acos(cosine))
    return {"gram_determinant": str(sp.factor(gram.det())),
            "internal_angle_cosine_derivative": str(sp.factor(sp.diff(cosine, edge))),
            "stationary_gradient": str(sp.simplify(gradient.subs(edge, 1))),
            "internal_hessian": str(sp.simplify(sp.diff(gradient, edge).subs(edge, 1))),
            "cospherical_area_derivative": str(sp.diff(edge*sp.sqrt(8-edge**2)/4, edge).subs(edge, 2)),
            "global_uniqueness": "theta_i는 열린 구간에서 엄격히 증가하고 면적 미분은 양수다. e=1만 정상점이며 엄격한 전역 최대다."}


def pair(value):
    return [float(value.real), float(value.imag)]


def run():
    boundary = COARSE.lengths(reference_points())
    classical = [classical_case(boundary), classical_case(boundary*1.1),
                 classical_case(boundary*(1+np.linspace(-.002, .003, len(boundary))))]
    checks = []
    full = FINAL.lengths(reference_points())
    for edge in np.linspace(.05, LIMIT-.02, 9):
        full[E_ID] = edge
        actual = FINAL.evaluate(full)
        checks.append([float(abs(actual["action"]-symmetric_action(edge))),
                       float(abs(actual["gradient"][E_ID]-symmetric_gradient(edge)))])
    rows = []
    for beta in (0, 10, 40, 160, 640):
        for kind in ("length", "squared"):
            value, error = integral(beta, kind)
            independent, _ = integral(beta, kind, order=1024)
            row = {"beta": beta, "measure": kind, "kernel": pair(value),
                   "adaptive_error_estimate": error, "quadrature_difference": abs(value-independent)}
            if beta:
                saddle, endpoints = stationary_terms(beta, kind)
                decay, _ = integral(beta, kind, decaying=True)
                slope = symmetric_gradient(0)
                leading = (1/(LIMIT*beta*slope) if kind == "length" else 2/(LIMIT**2*beta**2*slope**2))
                row.update({"saddle": pair(saddle), "endpoints": pair(endpoints),
                            "saddle_only_error": abs(value-saddle),
                            "saddle_and_endpoints_error": abs(value-saddle-endpoints),
                            "scaled_decaying_kernel": decay.real, "decaying_endpoint_leading": leading,
                            "decaying_ratio_to_leading": decay.real/leading})
            rows.append(row)
    passive = abs(passive_squared_integral(10)-integral(10, "length")[0])
    singular_points = reference_points()
    singular_points[:2, 0] *= 2
    singular_boundary = COARSE.lengths(singular_points)
    singular_lengths, _ = flat_completion(singular_boundary)
    singular_hessian, _ = FINAL.hessian(singular_lengths, [E_ID], step=2e-5)
    try:
        classical_case(singular_boundary)
        singular_rejected = False
    except ValueError:
        singular_rejected = True
    return {"status": "[산출]", "scope": "실제 내부 변의 고전 경계 제거와 두 공급 측도의 전체 유한 구간 적분",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("regge_pachner_constraints.py", "regge_pachner_creation.py",
                                          "regge_pachner_transport.py", "regge_tent_transfer.py")},
            "interval": [0, LIMIT], "symmetric_endpoint_actions": [float(symmetric_action(0)), float(symmetric_action(LIMIT))],
            "symmetric_endpoint_gradients": [symmetric_gradient(0), symmetric_gradient(LIMIT)],
            "exact_certificate": exact_certificate(), "classical_cases": classical,
            "closed_formula_errors": np.max(checks, axis=0).tolist(), "integrals": rows,
            "passive_coordinate_change_residual": passive,
            "stationary_density_ratio": 2/LIMIT,
            "cospherical_control": {"points": singular_points.tolist(), "internal_length": float(singular_lengths[E_ID]),
                                     "internal_hessian_numeric": float(singular_hessian[0, 0]),
                                     "minimum_gram_eigenvalue": FINAL.evaluate(singular_lengths)["minimum_gram_eigenvalue"],
                                     "schur_rejected": singular_rejected,
                                     "exact_reason": "여섯 꼭짓점이 단위 구면 위에 있다. e=2의 내부 면적 미분과 평탄 결손각이 모두 0이므로 H_ee=0이다."},
            "assumptions": ["공유 길이와 유클리드 레게 단체를 공급한다. 14개 경계 길이는 동일하다.",
                            "무차원 beta와 진동 부호, 정규화 de/L 및 2e de/L² 측도를 각각 공급한다.",
                            "감쇠 대조는 양의 s 부호에 exp(-beta*s)를 적용한 별도 유한 적분이다."],
            "unfinished": ["일반 비평탄 이력의 전체 상위 커널과 물리 측도·내적·초기 준비·분해능·환경",
                           "반복 합성의 비국소 경계 인자와 공통 계량의 동역학적 선택",
                           "0D에서 3+1 Plebanski/Einstein으로 가는 전체 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
