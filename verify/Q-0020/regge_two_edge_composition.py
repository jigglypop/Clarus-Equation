"""일곱 실제 레게 단체의 두 내부 변 전체 적분과 측도 전달을 검산한다.

모든 길이·작용 계수는 무차원이며 공유 기하·유클리드 가지와 두 결합측도는
공급 조건이다. 단계마다 독립 정규화를 재대입하는 대조를 정확한 합성과 구별한다.
"""

from functools import lru_cache
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import roots_legendre
from numpy.polynomial import Polynomial

from regge_pachner_constraints import FINAL as OLD
from regge_pachner_constraints import reference_points as old_points
from regge_pachner_creation import OLD_LOCAL_EDGES, admissible_interval
from regge_tent_transfer import ReggeComplex, LOCAL_EDGES, COMPLEMENTS


ADDED = ((0, 2, 3, 4, 6), (0, 2, 3, 5, 6), (0, 2, 4, 5, 6))
WHOLE = ReggeComplex(OLD.cells+ADDED)
SHELL = ReggeComplex(ADDED)
COARSE = ReggeComplex(((0, 2, 3, 4, 5), (1, 2, 3, 4, 5)))
INTERNAL = ((0, 1), (0, 2))
INTERNAL_IDS = WHOLE.indices(INTERNAL)
BOUNDARY_EDGES = [edge for edge in WHOLE.edges if edge not in INTERNAL]
BOUNDARY_IDS = WHOLE.indices(BOUNDARY_EDGES)


def reference_points():
    points = old_points()
    return np.vstack([points, [-1, *points[2, 1:]]])


class Domain:
    """두 길이제곱의 그램 양성 영역을 세 타원과 두 띠의 교집합으로 보존한다."""

    def __init__(self, boundary):
        boundary = np.asarray(boundary, dtype=float)
        if boundary.shape != (18,) or not np.all(np.isfinite(boundary)) or np.any(boundary <= 0):
            raise ValueError("경계 길이 열여덟 개는 유한한 양수여야 한다")
        self.boundary = boundary.copy()
        self.squared = dict(zip(BOUNDARY_EDGES, boundary**2))
        self.strips = np.array([[0.0, math.inf], [0.0, math.inf]])
        self.ellipses = []
        for cell in WHOLE.cells:
            active = [i for i, edge in enumerate(INTERNAL) if set(edge).issubset(cell)]
            if len(active) == 1:
                axis = active[0]
                edge = INTERNAL[axis]
                vertices = (*sorted(set(cell)-set(edge)), *edge)
                values = [math.sqrt(self.distance(vertices[i], vertices[j])) for i, j in OLD_LOCAL_EDGES]
                lower, upper = admissible_interval(values)
                self.strips[axis, 0] = max(self.strips[axis, 0], lower**2)
                self.strips[axis, 1] = min(self.strips[axis, 1], upper**2)
            elif len(active) == 2:
                others = sorted(set(cell)-{0})
                base = others[-1]
                vertices = others[:-1]
                gram = np.array([[(self.distance(base, i)+self.distance(base, j)-self.distance(i, j))/2
                                  for j in vertices] for i in vertices])
                if np.linalg.eigvalsh(gram)[0] <= 0:
                    raise ValueError("고정 경계 사면체가 비퇴화하지 않는다")
                radius = self.distance(0, base)
                constant, linear = [], np.zeros((3, 2))
                for row, vertex in enumerate(vertices):
                    known = 0 if vertex in (1, 2) else self.distance(0, vertex)
                    constant.append((radius+self.distance(base, vertex)-known)/2)
                    if vertex in (1, 2):
                        linear[row, vertex-1] = -.5
                constant = np.asarray(constant)
                inverse = np.linalg.inv(gram)
                quadratic = linear.T @ inverse @ linear
                linear_term = linear.T @ inverse @ constant
                center = -np.linalg.solve(quadratic, linear_term)
                remaining = radius-constant @ inverse @ constant+linear_term @ np.linalg.solve(quadratic, linear_term)
                if remaining <= 0 or np.linalg.eigvalsh(quadratic)[0] <= 0:
                    raise ValueError("두 내부 길이에 대한 그램 타원이 없다")
                self.ellipses.append((center, quadratic/remaining))
            else:
                raise ValueError("확인한 일곱 단체 이외의 변 구조다")
        self.raw = self.strips.copy()
        for center, metric in self.ellipses:
            radii = np.sqrt(np.diag(np.linalg.inv(metric)))
            self.raw[:, 0] = np.maximum(self.raw[:, 0], center-radii)
            self.raw[:, 1] = np.minimum(self.raw[:, 1], center+radii)
        if np.any(self.raw[:, 0] >= self.raw[:, 1]) or not np.all(np.isfinite(self.raw)):
            raise ValueError("공통 그램 영역이 비어 있다")
        self.projections = np.array([self._projection(axis) for axis in (0, 1)])

    def distance(self, i, j):
        return 0 if i == j else self.squared[tuple(sorted((i, j)))]

    def fiber_squared(self, outer, outer_axis):
        outer = np.asarray(outer, dtype=float)
        inner = 1-outer_axis
        lower = np.full(outer.shape, self.strips[inner, 0])
        upper = np.full(outer.shape, self.strips[inner, 1])
        for center, metric in self.ellipses:
            displacement = outer-center[outer_axis]
            middle = center[inner]-metric[inner, outer_axis]/metric[inner, inner]*displacement
            room = 1-(metric[outer_axis, outer_axis]-metric[inner, outer_axis]**2/metric[inner, inner])*displacement**2
            half = np.sqrt(np.maximum(room, 0)/metric[inner, inner])
            lower = np.maximum(lower, middle-half)
            upper = np.minimum(upper, middle+half)
        return lower, upper

    def _projection(self, axis):
        lo, hi = self.raw[axis]
        def gap(value):
            a, b = self.fiber_squared(value, axis)
            return float(b-a)
        result = minimize_scalar(lambda x: -gap(x), bounds=(lo, hi), method="bounded",
                                 options={"xatol": 1e-14})
        if not result.success or gap(result.x) <= 1e-12:
            raise ValueError("양의 폭을 가진 결합 그램 영역이 없다")
        left = lo if gap(lo) >= -1e-12 else brentq(gap, lo, result.x, xtol=1e-13)
        right = hi if gap(hi) >= -1e-12 else brentq(gap, result.x, hi, xtol=1e-13)
        return left, right

    def fiber(self, outer, outer_axis):
        if outer_axis not in (0, 1):
            raise ValueError("적분 순서는 내부 변 0 또는 1이어야 한다")
        outer = np.asarray(outer, dtype=float)
        lo, hi = np.sqrt(self.projections[outer_axis])
        if not np.all(np.isfinite(outer)) or np.any((outer < lo) | (outer > hi)):
            raise ValueError("바깥 내부 길이가 결합 영역의 사영 밖이다")
        lower, upper = self.fiber_squared(outer**2, outer_axis)
        if np.any(upper < lower-1e-10):
            raise ValueError("양의 폭의 내부 절단 구간이 없다")
        return np.sqrt(np.maximum(lower, 0)), np.sqrt(np.maximum(upper, 0))

    def lengths(self, e, f):
        e, f = np.broadcast_arrays(e, f)
        values = np.empty(e.shape+(len(WHOLE.edges),))
        values[..., BOUNDARY_IDS] = self.boundary
        values[..., INTERNAL_IDS[0]], values[..., INTERNAL_IDS[1]] = e, f
        return values

    def contains(self, e, f):
        point = np.array([e*e, f*f])
        return bool(e > 0 and f > 0 and np.all(point > self.strips[:, 0])
                    and np.all(point < self.strips[:, 1])
                    and all((point-center) @ metric @ (point-center) < 1
                            for center, metric in self.ellipses))

    def breakpoints(self, outer_axis):
        """타원끼리 또는 띠와 만나는 좌표에서 구적 구간을 나눈다.

        수치 다항식의 실근 후보를 분할에만 쓴다. 영역 판정은 원래 그램
        타원으로 수행하므로 여분의 분할점은 적분 영역을 바꾸지 않는다.
        """
        inner = 1-outer_axis
        variable = Polynomial([0, 1])
        polynomials, candidates = [], list(self.projections[outer_axis])
        def collect(polynomial):
            scale = max(abs(polynomial.coef))
            if scale < 1e-24:
                return
            for root in (polynomial/scale).trim(1e-12).roots():
                # 중근의 반올림 허수부는 여분의 분할만 허용한다.
                if abs(np.imag(root)) < 1e-5:
                    candidates.append(float(np.real(root)))
        for center, metric in self.ellipses:
            shift = variable-center[outer_axis]
            a = metric[inner, inner]
            b = (2*metric[inner, outer_axis]*shift-2*a*center[inner])/a
            c = (a*center[inner]**2-2*metric[inner, outer_axis]*center[inner]*shift
                 +metric[outer_axis, outer_axis]*shift**2-1)/a
            for endpoint in self.strips[inner]:
                collect(Polynomial([endpoint**2])+b*endpoint+c)
            polynomials.append((b, c))
        for (b1, c1), (b2, c2) in combinations(polynomials, 2):
            d, k = b1-b2, c1-c2
            if max(np.linalg.norm(d.coef), np.linalg.norm(k.coef)) < 1e-12:
                continue
            collect(k*k-b1*k*d+c1*d*d)
        lo, hi = self.projections[outer_axis]
        result = [lo]
        for value in sorted(candidates):
            if lo+1e-8 < value < hi-1e-8 and value-result[-1] > 1e-8:
                result.append(value)
        return np.sqrt(np.r_[result, hi])


class BatchAction:
    """기존 스칼라 기하와 대조하는 같은 면적·법선 그램의 일괄 계산."""

    def __init__(self, complex_):
        self.complex = complex_
        self.ids = WHOLE.indices(complex_.edges)

    def data(self, whole_lengths):
        lengths = np.asarray(whole_lengths, dtype=float)[..., self.ids]
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("모든 모서리 길이는 유한한 양수여야 한다")
        shape = lengths.shape[:-1]
        deficits = np.broadcast_to(np.where(self.complex.boundary, math.pi, 2*math.pi),
                                   shape+(len(self.complex.triangles),)).copy()
        for edge_ids, triangle_ids in zip(self.complex.cell_edges, self.complex.cell_triangles):
            distances = np.zeros(shape+(5, 5))
            for k, (i, j) in enumerate(LOCAL_EDGES):
                distances[..., i, j] = distances[..., j, i] = lengths[..., edge_ids[k]]**2
            gram = (distances[..., 0, 1:, None]+distances[..., None, 0, 1:]-distances[..., 1:, 1:])/2
            if np.min(np.linalg.eigvalsh(gram)[..., 0]) <= 0:
                raise ValueError("비퇴화 유클리드 단체가 아니다")
            inverse = np.linalg.inv(gram)
            normals = np.empty(shape+(5, 5))
            normals[..., 1:, 1:] = inverse
            normals[..., 0, 1:] = normals[..., 1:, 0] = -inverse.sum(axis=-1)
            normals[..., 0, 0] = inverse.sum(axis=(-2, -1))
            angles = np.stack([np.arccos(np.clip(-normals[..., i, j]/np.sqrt(normals[..., i, i]*normals[..., j, j]), -1, 1))
                               for i, j in COMPLEMENTS], axis=-1)
            deficits[..., triangle_ids] -= angles
        sides = lengths[..., np.asarray(self.complex.triangle_edges)]**2
        area_squared = (2*(sides[..., 0]*sides[..., 1]+sides[..., 0]*sides[..., 2]+sides[..., 1]*sides[..., 2])
                        -np.sum(sides*sides, axis=-1))/16
        if np.any(area_squared <= 0):
            raise ValueError("삼각형 면적이 양수가 아니다")
        areas = np.sqrt(area_squared)
        return np.sum(areas*deficits, axis=-1), areas

    def __call__(self, whole_lengths):
        return self.data(whole_lengths)[0]


WHOLE_ACTION, OLD_ACTION, SHELL_ACTION = (BatchAction(c) for c in (WHOLE, OLD, SHELL))
old_kappa = dict(zip(OLD.triangles, np.where(OLD.boundary, 1, 2)))
whole_kappa = dict(zip(WHOLE.triangles, np.where(WHOLE.boundary, 1, 2)))
CORNER = np.array([whole_kappa[t]-old_kappa.get(t, 0)-(1 if boundary else 2)
                   for t, boundary in zip(SHELL.triangles, SHELL.boundary)])


def increment(domain, f):
    """경계각 보정까지 합친 실제 세 단체 증분이며 기존 내부 변 e와 무관하다."""
    f = np.asarray(f)
    unique, ids = np.unique(f, return_inverse=True)
    values, areas = SHELL_ACTION.data(domain.lengths(np.ones_like(unique), unique))
    return (values+math.pi*(areas @ CORNER))[ids].reshape(f.shape)


@lru_cache(maxsize=4)
def rule(order):
    if not isinstance(order, int) or isinstance(order, bool) or order < 16:
        raise ValueError("구적 차수는 16 이상 정수여야 한다")
    return roots_legendre(order)


def integrate(domain, beta_values=(0, 1, 5, 20), *, order=96, outer_axis=1, staged=True):
    betas = np.asarray(beta_values, dtype=float)
    if betas.ndim != 1 or not len(betas) or not np.all(np.isfinite(betas)):
        raise ValueError("위상 계수는 비어 있지 않은 유한 실수 열이어야 한다")
    if outer_axis not in (0, 1):
        raise ValueError("적분 순서는 내부 변 0 또는 1이어야 한다")
    nodes, weights = rule(order)
    angles = (nodes+1)*math.pi/4
    breaks = domain.breakpoints(outer_axis)
    outer_parts, weight_parts = [], []
    for lower, upper in zip(breaks, breaks[1:]):
        values = (lower+upper)/2-(upper-lower)*np.cos(2*angles)/2
        quadrature_weights = weights*math.pi/4*(upper-lower)*np.sin(2*angles)
        if lower < 1e-8*upper:
            # 길이 0에서는 제곱 길이의 추가 압축을 피한다. 적분 범위는 그대로다.
            values = (lower+upper)/2+(upper-lower)*nodes/2
            quadrature_weights = weights*(upper-lower)/2
        outer_parts.append(values)
        weight_parts.append(quadrature_weights)
    outer, outer_weights = np.concatenate(outer_parts), np.concatenate(weight_parts)
    left, right = domain.fiber(outer, outer_axis)
    inner = left[:, None]+(right-left)[:, None]*(nodes[None, :]+1)/2
    outer = np.broadcast_to(outer[:, None], inner.shape)
    e, f = (inner, outer) if outer_axis == 1 else (outer, inner)
    lengths = domain.lengths(e, f)
    direct = WHOLE_ACTION(lengths)
    old_action = OLD_ACTION(lengths) if staged else None
    delta = increment(domain, f) if staged else None
    staged_action = old_action+delta if staged else direct
    inner_weights = (right-left)[:, None]*weights[None, :]/2
    phases = np.exp(1j*direct[..., None]*betas)
    result = {}
    for name in ("length", "squared"):
        inner_density = np.ones_like(inner) if name == "length" else 2*inner
        outer_density = np.ones(len(outer)) if name == "length" else 2*outer[:, 0]
        joint = outer_weights[:, None]*outer_density[:, None]*inner_weights*inner_density
        volume = float(joint.sum())
        inner_mass = right-left if name == "length" else right**2-left**2
        conditional_weights = inner_weights*inner_density/inner_mass[:, None]
        marginal_weights = outer_weights*outer_density*inner_mass/volume
        wrong = joint/inner_mass[:, None]
        wrong /= wrong.sum()
        values = np.sum(joint[..., None]*phases, axis=(0, 1))/volume
        stage_values = values.copy()
        transport = None
        if staged:
            old_phases = np.exp(1j*old_action[..., None]*betas)
            delta_phases = np.exp(1j*delta[..., None]*betas)
            old_kernel = np.sum(conditional_weights[..., None]*old_phases, axis=1)
            if outer_axis == 1:
                # 먼저 얻은 복소 커널에 다음 작용을 곱하고 유도한 주변측도로 적분한다.
                next_kernel = old_kernel*delta_phases[:, 0, :]
                transport = {"outer_lengths": outer[:, 0], "inner_mass": inner_mass,
                             "old_kernel": old_kernel, "increment_phase": delta_phases[:, 0, :],
                             "marginal_weights": marginal_weights}
            else:
                # 역순에서는 증분이 내부 변수 f에 의존하므로 내부 적분 안에 둔다.
                next_kernel = np.sum(conditional_weights[..., None]*old_phases*delta_phases, axis=1)
            stage_values = np.sum(marginal_weights[:, None]*next_kernel, axis=0)
        wrong_values = np.sum(wrong[..., None]*phases, axis=(0, 1))
        result[name] = {"volume": volume, "kernel": values, "staged_kernel": stage_values,
                        "wrong_reset_kernel": wrong_values, "transport": transport}
    result["action_identity_error"] = float(np.max(np.abs(direct-staged_action)))
    return result


def exact_certificate():
    import sympy as sp
    E, F = sp.symbols("E F", positive=True)
    points = sp.Matrix([[sp.Rational(-1, 2), 0, 0, 0], [sp.Rational(1, 2), 0, 0, 0],
                        [0, 1, 1, 1], [0, 1, -1, -1], [0, -1, 1, -1], [0, -1, -1, 1],
                        [-1, 1, 1, 1]])
    points[:, 1:] = points[:, 1:]/sp.sqrt(3)
    def gram_matrix(cell):
        def distance(i, j):
            if {i, j} == {0, 1}: return E
            if {i, j} == {0, 2}: return F
            delta = points[i, :]-points[j, :]
            return sp.simplify(delta.dot(delta))
        base = cell[0]
        return sp.Matrix(4, 4, lambda i, j: (distance(base, cell[i+1])+distance(base, cell[j+1])
                                            -distance(cell[i+1], cell[j+1]))/2)
    def determinant(cell):
        return str(sp.factor(gram_matrix(cell).det()))
    witness = {E: sp.Rational(4, 9), F: sp.Rational(49, 100)}
    fine_minors = []
    for cell in WHOLE.cells:
        gram = gram_matrix(cell).subs(witness)
        fine_minors.append([str(sp.factor(gram[:i, :i].det())) for i in range(1, 5)])
    return {"E_only_determinant": determinant((0, 1, 3, 4, 5)),
            "coupled_determinant": determinant((0, 1, 2, 3, 4)),
            "F_only_determinant": determinant((0, 2, 3, 4, 6)),
            "coarse_determinant": determinant(COARSE.cells[0]),
            "support_witness": {"e": "2/3", "f": "7/10", "fine_leading_minors": fine_minors,
                                "coarse_determinant": str(sp.factor(gram_matrix(COARSE.cells[0]).subs(witness).det()))},
            "corner_pi_coefficients": dict(zip(map(str, SHELL.triangles), map(int, CORNER)))}


def pair(value):
    return [float(value.real), float(value.imag)]


def run():
    base = WHOLE.lengths(reference_points())[BOUNDARY_IDS]
    boundaries = [base, base*(1+.002*np.linspace(-1, 1, len(base))),
                  base*(1+.0015*np.cos(np.arange(len(base))))]
    cases = []
    for boundary in boundaries:
        domain = Domain(boundary)
        forward = integrate(domain, order=64)
        reverse = integrate(domain, order=128, outer_axis=0, staged=False)
        refined = integrate(domain, order=128, staged=False)
        points = []
        for outer in np.linspace(*np.sqrt(domain.projections[1]), 9)[1:-1]:
            a, b = domain.fiber(outer, 1)
            for e in (a+.2*(b-a), a+.7*(b-a)):
                lengths = domain.lengths(e, outer)
                actual = WHOLE.evaluate(lengths)
                points.append(abs(float(WHOLE_ACTION(lengths))-actual["action"]))
        rows = []
        for name in ("length", "squared"):
            for i, beta in enumerate((0, 1, 5, 20)):
                value = refined[name]["kernel"][i]
                rows.append({"measure": name, "beta": beta, "kernel": pair(value),
                             "order_difference": float(abs(value-forward[name]["kernel"][i])),
                             "elimination_order_difference": float(abs(value-reverse[name]["kernel"][i])),
                             "staged_direct_difference": float(abs(forward[name]["kernel"][i]-forward[name]["staged_kernel"][i])),
                             "wrong_forward_kernel": pair(forward[name]["wrong_reset_kernel"][i]),
                             "wrong_reverse_kernel": pair(reverse[name]["wrong_reset_kernel"][i])})
        cases.append({"boundary": boundary.tolist(), "squared_projections": domain.projections.tolist(),
                      "quadrature_breakpoints": [domain.breakpoints(i).tolist() for i in (0, 1)],
                      "ellipses": [{"center": c.tolist(), "metric": m.tolist()} for c, m in domain.ellipses],
                      "strips": domain.strips.tolist(), "batch_scalar_error": float(max(points)),
                      "action_identity_error": forward["action_identity_error"],
                      "volumes": {name: refined[name]["volume"] for name in ("length", "squared")},
                      "transport": {name: {
                          "marginal_mass": float(forward[name]["transport"]["marginal_weights"].sum()),
                          "maximum_conditional_modulus": float(np.max(abs(forward[name]["transport"]["old_kernel"]))),
                          "sample": [{"f": float(forward[name]["transport"]["outer_lengths"][i]),
                                      "inner_mass": float(forward[name]["transport"]["inner_mass"][i]),
                                      "old_kernel_beta5": pair(forward[name]["transport"]["old_kernel"][i, 2])}
                                     for i in np.linspace(0, len(forward[name]["transport"]["outer_lengths"])-1, 5, dtype=int)]}
                                    for name in ("length", "squared")},
                      "rows": rows})
    return {"status": "[산출]", "scope": "실제 일곱 단체의 두 내부 변 전체 적분과 같은 결합측도의 조건부 전달",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("regge_pachner_constraints.py", "regge_pachner_creation.py", "regge_tent_transfer.py")},
            "cells": WHOLE.cells, "internal_edges": INTERNAL, "boundary_edges": BOUNDARY_EDGES,
            "exact_certificate": exact_certificate(), "cases": cases,
            "assumptions": ["공유 길이와 유클리드 레게 작용, 최종 그램 양성 영역을 공급한다.",
                            "최종 영역 위 de df 및 d(e^2)d(f^2)의 전역 정규화 측도를 각각 공급한다.",
                            "같은 결합측도에서 주변·조건부 밀도를 유도하고 전체 복소 커널을 전달한다."],
            "unfinished": ["일반 세분화의 물리 측도·내적·초기 준비·분해능을 같은 미시 작용에서 고정",
                           "공통 계량의 동역학적 선택과 0D에서 3+1 Plebanski/Einstein 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
