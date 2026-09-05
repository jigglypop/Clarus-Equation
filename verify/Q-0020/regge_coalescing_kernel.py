"""경계 높이에 따른 실제 내부 정상점 합류와 전체 진동 커널을 검산한다.

공유 길이와 무차원 h,e,beta 및 두 정규화 측도를 공급한 검증 모형이다.
유한 적분을 가우스 또는 에어리 근사와 구별하며 물리 측도를 유도하지 않는다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import airy

from regge_internal_marginal import (
    BOUNDARY_IDS, COARSE, E_ID, FINAL, angular_rule, internal_interval,
    pair, reference_points, symmetric_action,
)


CUBIC = 48*math.sqrt(3)
QUARTIC = 720*math.sqrt(3)
KAPPA = (CUBIC/2)**(1/3)


def limit(h):
    if not np.isfinite(h) or h <= 0:
        raise ValueError("경계 높이는 유한한 양수여야 한다")
    value = 2*math.hypot(h, 1/3)
    if not np.isfinite(value):
        raise ValueError("경계 높이의 허용상한이 유한하지 않다")
    return value


def geometry(edge, h):
    upper = limit(h)
    edge = np.asarray(edge, dtype=float)
    if not np.all(np.isfinite(edge)) or np.any((edge < 0) | (edge > upper)):
        raise ValueError("내부 길이는 닫힌 허용 구간 안에 있어야 한다")
    remainder = np.sqrt(np.maximum((upper-edge)*(upper+edge), 0))
    internal = np.arctan2(3*math.sqrt(3)*remainder*np.sqrt(9*remainder**2+32),
                          9*remainder**2-16)
    spoke = np.arctan2(remainder*math.sqrt(9*upper**2+8), 2*math.sqrt(2)*edge)
    rim = 2*np.arctan2(edge, remainder)
    return edge, remainder, internal, spoke, rim


def action(edge, h):
    edge, _, internal, spoke, rim = geometry(edge, h)
    area = edge*np.sqrt(4*(h*h+1)-edge**2)/4
    boundary_area = math.sqrt(2*(3*h*h+1))/3
    return (4*area*(2*math.pi-3*internal)
            +12*boundary_area*(math.pi-2*spoke)
            +8*math.sqrt(3)/3*(math.pi-rim))


def deficit(edge, h):
    edge, remainder, internal, _, _ = geometry(edge, h)
    # cos(theta)+1/2를 인수분해해 평탄 근 부근의 큰 수 뺄셈을 피한다.
    cosine_difference = 9*(2*h-edge)*(2*h+edge)/(9*remainder**2+8)
    ratio = cosine_difference/(2*np.sin((internal+2*math.pi/3)/2))
    return 6*np.arcsin(np.clip(ratio, -1, 1))


def gradient(edge, h):
    edge, _, _, _, _ = geometry(edge, h)
    area_root = math.sqrt(2*(h*h+1))
    factor = 2*(area_root-edge)*(area_root+edge)/np.sqrt(4*(h*h+1)-edge**2)
    return factor*deficit(edge, h)


def hessian(edge, h):
    edge, _, internal, _, _ = geometry(edge, h)
    if np.any((edge <= 0) | (edge >= limit(h))):
        raise ValueError("헤시안은 열린 구간에서만 계산한다")
    width = 4*(h*h+1)-edge**2
    prefactor = (4*(h*h+1)-2*edge**2)/np.sqrt(width)
    prefactor_derivative = -4*edge/np.sqrt(width)+(4*(h*h+1)-2*edge**2)*edge/width**1.5
    angle_denominator = 12*h*h+4-3*edge**2
    deficit_derivative = -72*edge/(angle_denominator**2*np.sin(internal))
    return prefactor_derivative*deficit(edge, h)+prefactor*deficit_derivative


def stationary_points(h):
    limit(h)
    roots = [2*h]
    if h > math.sqrt(7)/3 and h != 1:
        roots.append(math.sqrt(2*(h*h+1)))
    return sorted(roots)


def density(edge, h, kind):
    upper = limit(h)
    if kind == "length":
        return np.ones_like(np.asarray(edge))/upper
    if kind == "squared":
        return 2*np.asarray(edge)/upper**2
    raise ValueError("길이 또는 길이제곱 측도만 지원한다")


def integral(beta, h, kind, *, order=None):
    """e=L(h)sin(theta)로 움직이는 상한을 고정한 전체 유한 적분."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta는 유한한 음이 아닌 수여야 한다")
    upper = limit(h)
    density(0, h, kind)
    if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 16):
        raise ValueError("구적 차수는 16 이상 정수여야 한다")
    reference = float(action(2*h, h))
    def integrand(angle):
        edge = upper*np.sin(angle)
        weight = density(edge, h, kind)*upper*np.cos(angle)
        return weight*np.exp(1j*beta*(action(edge, h)-reference))
    if order is not None:
        angles, weights = angular_rule(order)
        reduced, error = np.dot(weights, integrand(angles)), None
    else:
        options = {"epsabs": 2e-10, "epsrel": 2e-10, "limit": 4000,
                   "points": [math.asin(e/upper) for e in stationary_points(h)]}
        real, er = quad(lambda t: float(integrand(t).real), 0, math.pi/2, **options)
        imag, ei = quad(lambda t: float(integrand(t).imag), 0, math.pi/2, **options)
        reduced, error = real+1j*imag, er+ei
    return complex(reduced*np.exp(1j*beta*reference)), error


def endpoint_term(beta, h, kind):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("끝점 근사는 양의 beta가 필요하다")
    upper = limit(h)
    slopes = [float(gradient(e, h)) for e in (0, upper)]
    if min(abs(slope) for slope in slopes) < 1e-10:
        raise ValueError("정상 끝점에는 이 근사를 적용할 수 없다")
    return complex((density(upper, h, kind)*np.exp(1j*beta*action(upper, h))/slopes[1]
                    -density(0, h, kind)*np.exp(1j*beta*action(0, h))/slopes[0])/(1j*beta))


def gaussian_term(beta, h, kind):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("가우스 근사는 양의 beta가 필요하다")
    density(0, h, kind)
    if h == 1:
        raise ValueError("합류점의 헤시안은 0이므로 개별 가우스 근사를 적용할 수 없다")
    result = 0j
    for edge in stationary_points(h):
        curvature = float(hessian(edge, h))
        result += (density(edge, h, kind)*math.sqrt(2*math.pi/(beta*abs(curvature)))
                   *np.exp(1j*beta*action(edge, h)+1j*np.sign(curvature)*math.pi/4))
    return complex(result)


def airy_parameters(h, kind):
    """두 정상값에 맞춘 국소 삼차 정규형의 계수. 끝점은 포함하지 않는다.

    정확한 h=1은 해석적 극한을 쓴다. 서로 다른 두 근을 부동소수점으로
    분리할 수 없으면 ValueError로 거부하며 임의로 합류점에 합치지 않는다.
    """
    if not .9 <= h <= 1.1:
        raise ValueError("균일 근사는 확인한 경계 높이 구간에서만 제공한다")
    if h == 1:
        mu = float(density(2, h, kind))
        derivative = 0 if kind == "length" else 2/limit(h)**2
        return {"phase": float(action(2, h)), "delta": 0.0,
                "a0": mu/KAPPA, "a1": (derivative-mu*QUARTIC/(6*CUBIC))/KAPPA**2}
    left, right = stationary_points(h)
    # 정상값의 차이는 O((h-1)^3)이다. 작용 두 개를 빼지 않고 미분을 적분한다.
    difference = -quad(lambda e: float(gradient(e, h)), left, right,
                       epsabs=1e-30, epsrel=1e-10)[0]
    if difference <= 0:
        raise ValueError("두 정상값의 순서를 수치적으로 분리할 수 없다")
    delta = (3*difference/4)**(2/3)
    lower_weight = float(density(left, h, kind))*math.sqrt(2*math.sqrt(delta)/abs(hessian(left, h)))
    upper_weight = float(density(right, h, kind))*math.sqrt(2*math.sqrt(delta)/abs(hessian(right, h)))
    return {"phase": float(action(left, h))-difference/2, "delta": delta,
            "a0": (lower_weight+upper_weight)/2,
            "a1": (upper_weight-lower_weight)/(2*math.sqrt(delta))}


def airy_term(beta, h, kind):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("에어리 근사는 양의 beta가 필요하다")
    pars = airy_parameters(h, kind)
    ai, aip, _, _ = airy(-beta**(2/3)*pars["delta"])
    value = 2*math.pi*np.exp(1j*beta*pars["phase"])*(
        pars["a0"]*ai/beta**(1/3)-1j*pars["a1"]*aip/beta**(2/3))
    return complex(value)


def exact_certificate():
    import sympy as sp
    e = sp.Symbol("e", positive=True)
    q = sp.Symbol("q", positive=True)
    gram = sp.Matrix([[e**2, e**2/2, e**2/2, e**2/2],
                      [e**2/2, q, q-sp.Rational(4, 3), q-sp.Rational(4, 3)],
                      [e**2/2, q-sp.Rational(4, 3), q, q-sp.Rational(4, 3)],
                      [e**2/2, q-sp.Rational(4, 3), q-sp.Rational(4, 3), q]])
    cosine = (8-3*e**2)/(2*(16-3*e**2))
    derivative = (8-2*e**2)/sp.sqrt(8-e**2)*(2*sp.pi-3*sp.acos(cosine))
    return {"gram_determinant_q_h2_plus_1": str(sp.factor(gram.det())),
            "critical_derivatives_2_3_4": [str(sp.simplify(sp.diff(derivative, e, n).subs(e, 2)))
                                           for n in (1, 2, 3)],
            "flat_hessian": "12*sqrt(3)*h*(h**2-1)",
            "roots": "2h; sqrt(2(h**2+1))는 h>sqrt(7)/3일 때만 내부에 존재한다.",
            "critical_leading": "2*pi*mu(2,1)*Ai(0)/(24*sqrt(3)*beta)**(1/3)",
            "uniform_window": "delta~(24*sqrt(3))**(2/3)*(h-1)**2/4; |h-1|=O(beta**(-1/3))"}


def run():
    checks, branches = [], []
    for h in (.5, .9, .99, 1, 1.01, 1.1):
        points = reference_points()
        points[:2, 0] = [-h, h]
        lengths = FINAL.lengths(points)
        actual_interval = internal_interval(lengths[BOUNDARY_IDS])
        for edge in np.linspace(.08*limit(h), .96*limit(h), 7):
            lengths[E_ID] = edge
            actual = FINAL.evaluate(lengths)
            checks.append([abs(float(action(edge, h))-actual["action"]),
                           abs(float(gradient(edge, h))-actual["gradient"][E_ID])])
        for edge in stationary_points(h):
            lengths[E_ID] = edge
            actual = FINAL.evaluate(lengths)
            hh, _ = FINAL.hessian(lengths, [E_ID], step=2e-5)
            branches.append({"h": h, "edge": edge, "interval": list(actual_interval),
                             "stationary_residual": float(abs(actual["gradient"][E_ID])),
                             "internal_deficit": float(np.max(np.abs(actual["deficits"][~FINAL.boundary]))),
                             "hessian": float(hessian(edge, h)), "hessian_difference": float(abs(hh[0, 0]-hessian(edge, h))),
                             "minimum_gram_eigenvalue": actual["minimum_gram_eigenvalue"]})
    rows = []
    for h in (.9, .99, 1, 1.01, 1.1):
        for beta in (0, 20, 80, 320):
            for kind in ("length", "squared"):
                value, error = integral(beta, h, kind)
                other, _ = integral(beta, h, kind, order=4096)
                row = {"h": h, "beta": beta, "measure": kind, "kernel": pair(value),
                       "quadrature_difference": abs(value-other), "adaptive_error_estimate": error}
                if beta:
                    uniform = airy_term(beta, h, kind)
                    end = endpoint_term(beta, h, kind)
                    row.update({"airy_and_endpoints": pair(uniform+end),
                                "airy_and_endpoints_error": abs(value-uniform-end)})
                    if h != 1:
                        row["gaussian_and_endpoints_error"] = abs(value-gaussian_term(beta, h, kind)-end)
                rows.append(row)
    critical = []
    for beta in (80, 320, 1280):
        for kind in ("length", "squared"):
            value, error = integral(beta, 1, kind)
            other, _ = integral(beta, 1, kind, order=8192)
            scaled = value*np.exp(-1j*beta*action(2, 1))*beta**(1/3)
            leading = 2*math.pi*density(2, 1, kind)*airy(0)[0]/KAPPA
            critical.append({"beta": beta, "measure": kind, "scaled_kernel": pair(scaled),
                             "leading": float(leading), "leading_error": float(abs(scaled-leading)),
                             "airy_and_endpoints_error": abs(value-airy_term(beta, 1, kind)-endpoint_term(beta, 1, kind)),
                             "quadrature_difference": abs(value-other), "adaptive_error_estimate": error})
    continuity = []
    for offset in (1e-2, 1e-3, 1e-4, 1e-6):
        center, _ = integral(20, 1, "length")
        below, _ = integral(20, 1-offset, "length")
        above, _ = integral(20, 1+offset, "length")
        continuity.append({"offset": offset, "maximum_kernel_difference": max(abs(below-center), abs(above-center)),
                           "maximum_gaussian_modulus": max(abs(gaussian_term(20, 1-offset, "length")),
                                                           abs(gaussian_term(20, 1+offset, "length")))})
    return {"status": "[산출]", "scope": "같은 실제 네 단체의 대칭 경계 높이 변화와 내부 정상점 합류",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("regge_internal_marginal.py", "regge_pachner_constraints.py",
                                          "regge_pachner_creation.py", "regge_tent_transfer.py")},
            "exact_certificate": exact_certificate(), "branches": branches,
            "closed_formula_errors": np.max(checks, axis=0).tolist(),
            "old_action_regression": float(max(abs(action(e, .5)-symmetric_action(e))
                                               for e in np.linspace(0, limit(.5), 17))),
            "integrals": rows, "critical_asymptotic": critical, "continuity": continuity,
            "assumptions": ["모든 길이·h·beta는 무차원이고 공유 경계 기하와 유클리드 작용을 공급한다.",
                            "정규화 de/L(h),2e de/L(h)^2와 실수 유한 구간을 별도로 공급한다.",
                            "에어리 근사는 내부 두 정상점용이며 퇴화 끝점의 항은 따로 더한다."],
            "unfinished": ["물리 측도·내적·준비·분해능의 독립 고정과 일반 경계의 실제 반복 합성",
                           "공통 계량의 동역학적 선택과 0D에서 3+1 Plebanski/Einstein 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
