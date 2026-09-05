"""지지 전체의 경계 질량·진폭 전달과 단체 충전의 선택 범위를 검산한다.

공유 유클리드 기하와 두 기준측도는 공급 조건이다. 원뿔은 기하의 존재
증인이며, 기존 복소 진폭을 새 원뿔 작용으로 대체하지 않는다.
"""

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval

from regge_pachner_constraints import boundary_facets
from regge_pachner_creation import OLD_LOCAL_EDGES, admissible_interval
import regge_two_edge_composition as base


FACETS = sorted(boundary_facets(base.OLD))
EDGES = sorted({edge for face in FACETS for edge in combinations(face, 2)})
CONE = base.ReggeComplex([(7, *face) for face in FACETS])


def phase_values(betas):
    values = np.asarray(betas, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("위상 계수는 비어 있지 않은 유한 실수 열이어야 한다")
    return values


def pair_at(domain, f, betas=(0, 1, 5, 20), *, order=128):
    """내부 e 적분을 정규화하지 않고 질량과 진폭을 함께 반환한다."""
    betas = phase_values(betas)
    f = np.asarray(f, dtype=float)
    left, right = domain.fiber(f, 1)
    if np.any(right <= left):
        raise ValueError("내부 적분은 양의 폭을 가진 열린 절단에서 계산한다")
    nodes, weights = base.rule(order)
    e = left[..., None]+(right-left)[..., None]*(nodes+1)/2
    lengths = domain.lengths(e, f[..., None])
    phase = np.exp(1j*base.OLD_ACTION(lengths)[..., None]*betas)
    result = {}
    for name, density in (("length", np.ones_like(e)), ("squared", 2*e)):
        w = (right-left)[..., None]*weights*density/2
        mass = right-left if name == "length" else right**2-left**2
        result[name] = {"mass": mass, "amplitude": np.sum(w[..., None]*phase, axis=-2)}
    return result


def coarse_interval(domain):
    """제거변을 되살리지 않는 두 단체 치환의 가능한 f 구간을 구한다."""
    vertices = (3, 4, 5, 0, 2)
    lengths = [math.sqrt(domain.distance(vertices[i], vertices[j])) for i, j in OLD_LOCAL_EDGES]
    lo, hi = admissible_interval(lengths)
    values = domain.lengths(1, (lo+hi)/2)[base.WHOLE.indices(base.COARSE.edges)]
    # f와 무관한 두 번째 상위 단체의 적합성도 함께 확인한다.
    base.COARSE.evaluate(values)
    return lo, hi


def cone_completion(domain, f, clearance=1):
    if not np.isfinite(clearance) or clearance <= 0:
        raise ValueError("원뿔 높이 제곱의 여유는 유한한 양수여야 한다")
    values = domain.lengths(1, f)
    distances = {edge: values[i]**2 for i, edge in enumerate(base.WHOLE.edges)}
    radii = []
    for face in FACETS:
        p, *rim = face
        gram = np.array([[(distances[tuple(sorted((p, i)))]+distances[tuple(sorted((p, j)))]
                           -(0 if i == j else distances[tuple(sorted((i, j)))]))/2
                          for j in rim] for i in rim])
        if np.linalg.eigvalsh(gram)[0] <= 0:
            raise ValueError("경계 사면체가 비퇴화 유클리드 기하가 아니다")
        h = np.diag(gram)/2
        radii.append(float(h @ np.linalg.solve(gram, h)))
    radius_squared = max(radii)+clearance
    cone_lengths = np.array([math.sqrt(radius_squared if 7 in edge else distances[edge])
                             for edge in CONE.edges])
    actual = CONE.evaluate(cone_lengths)
    return {"clearance": clearance, "radius_squared": radius_squared,
            "maximum_circumradius_squared": max(radii), "action": actual["action"],
            "minimum_gram_eigenvalue": actual["minimum_gram_eigenvalue"],
            "boundary_preserved": boundary_facets(CONE) == set(FACETS)}


@dataclass
class PairTable:
    boundary: list
    betas: list
    segments: list
    degree: int
    inner_order: int

    @classmethod
    def build(cls, domain, betas=(0, 1, 5, 20), *, degree=64, inner_order=128):
        betas = phase_values(betas)
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 16:
            raise ValueError("보간 차수는 16 이상 정수여야 한다")
        breaks = list(domain.breakpoints(1))
        coarse_lo, coarse_hi = coarse_interval(domain)
        breaks += [x for x in (coarse_lo, coarse_hi) if breaks[0] < x < breaks[-1]]
        breaks = np.unique(breaks)
        nodes = np.cos((np.arange(degree+1)+.5)*math.pi/(degree+1))
        angle = (nodes+1)*math.pi/4
        segments = []
        for lower, upper in zip(breaks, breaks[1:]):
            f = lower+(upper-lower)*np.sin(angle)**2
            pairs = pair_at(domain, f, betas, order=inner_order)
            entry = {"lower": float(lower), "upper": float(upper), "coefficients": {}}
            for name in ("length", "squared"):
                p = pairs[name]
                data = np.column_stack((p["mass"], p["amplitude"].real, p["amplitude"].imag))
                entry["coefficients"][name] = chebfit(nodes, data, degree).tolist()
            segments.append(entry)
        return cls(domain.boundary.tolist(), betas.tolist(), segments, degree, inner_order)

    def to_json(self):
        return json.dumps(vars(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, encoded):
        return cls(**json.loads(encoded))

    def evaluate_segment(self, index, x, name):
        if name not in ("length", "squared"):
            raise ValueError("확인한 두 기준측도 중 하나여야 한다")
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(abs(x) > 1):
            raise ValueError("보간 좌표는 유한한 [-1,1] 범위여야 한다")
        values = np.moveaxis(chebval(x, np.asarray(self.segments[index]["coefficients"][name])), 0, -1)
        n = len(self.betas)
        return values[..., 0], values[..., 1:n+1]+1j*values[..., n+1:]

    def integrate(self, *, order=128):
        domain = base.Domain(self.boundary)
        betas = phase_values(self.betas)
        nodes, weights = base.rule(order)
        angle = (nodes+1)*math.pi/4
        co_lo, co_hi = coarse_interval(domain)
        result = {name: {"mass": 0.0, "amplitude": np.zeros(len(betas), dtype=complex),
                         "excluded_mass": 0.0, "excluded_amplitude": np.zeros(len(betas), dtype=complex),
                         "dominance_error": 0.0}
                  for name in ("length", "squared")}
        for i, segment in enumerate(self.segments):
            lo, hi = segment["lower"], segment["upper"]
            f = lo+(hi-lo)*np.sin(angle)**2
            jac = (hi-lo)*np.sin(2*angle)*math.pi/4
            phase = np.exp(1j*base.increment(domain, f)[:, None]*betas)
            excluded = (f < co_lo) | (f > co_hi)
            for name in ("length", "squared"):
                mass, amplitude = self.evaluate_segment(i, nodes, name)
                w = weights*jac*(1 if name == "length" else 2*f)
                row = result[name]
                row["mass"] += float(w @ mass)
                row["amplitude"] += np.sum(w[:, None]*phase*amplitude, axis=0)
                row["excluded_mass"] += float(w[excluded] @ mass[excluded])
                row["excluded_amplitude"] += np.sum(w[excluded, None]*phase[excluded]*amplitude[excluded], axis=0)
                row["dominance_error"] = max(row["dominance_error"],
                                            float(np.max(abs(amplitude)-mass[:, None])))
        for row in result.values():
            row["kernel"] = row["amplitude"]/row["mass"]
            row["cut_kernel"] = ((row["amplitude"]-row["excluded_amplitude"])
                                 /(row["mass"]-row["excluded_mass"]))
            row["excluded_fraction"] = row["excluded_mass"]/row["mass"]
        return result


def complex_list(values):
    return [[float(z.real), float(z.imag)] for z in np.asarray(values).ravel()]


def run():
    previous = json.loads(Path(base.__file__).with_suffix(".json").read_text(encoding="utf-8"))
    if previous["source_sha256"] != hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest():
        raise ValueError("입력 기하 검산의 소스 해시가 현재 코드와 다르다")
    cases = []
    for case in previous["cases"]:
        domain = base.Domain(case["boundary"])
        table = PairTable.build(domain)
        saved = PairTable.from_json(table.to_json())
        current = saved.integrate()
        lower = PairTable.build(domain, degree=32, inner_order=64).integrate()
        interpolation = PairTable.build(domain, degree=32, inner_order=128).integrate()
        independent = base.integrate(domain, order=128, staged=False)
        rows = {}
        for name in ("length", "squared"):
            r = current[name]
            rows[name] = {"mass": r["mass"], "kernel": complex_list(r["kernel"]),
                          "coarse_cut_fraction": r["excluded_fraction"],
                          "coarse_cut_amplitude": complex_list(r["excluded_amplitude"]/r["mass"]),
                          "coarse_cut_kernel": complex_list(r["cut_kernel"]),
                          "direct_error": float(np.max(abs(r["kernel"]-independent[name]["kernel"]))),
                          "table_order_error": float(np.max(abs(r["kernel"]-lower[name]["kernel"]))),
                          "interpolation_order_error": float(np.max(abs(r["kernel"]-interpolation[name]["kernel"]))),
                          "inner_order_error": float(np.max(abs(interpolation[name]["kernel"]-lower[name]["kernel"]))),
                          "mass_error": abs(r["mass"]-independent[name]["volume"]),
                          "dominance_error": r["dominance_error"]}
        f = .7
        raw = pair_at(domain, f)
        cases.append({"boundary": case["boundary"], "coarse_interval": coarse_interval(domain),
                      "results": rows, "table": vars(saved),
                      "invalid_coarse_sample": {"f": f, "pairs": {
                          name: {"mass": float(raw[name]["mass"]),
                                 "amplitude": complex_list(raw[name]["amplitude"])}
                          for name in ("length", "squared")},
                          "cones": [cone_completion(domain, f, c) for c in (1, 4)]}})
    return {"status": "[산출]", "scope": "동일 전역 측도에서 유도한 경계 질량·진폭의 전체 지지 전달",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("regge_two_edge_composition.py", "regge_pachner_constraints.py",
                                          "regge_pachner_creation.py", "regge_tent_transfer.py",
                                          "regge_two_edge_composition.json")},
            "boundary_tetrahedra": FACETS, "boundary_edges": EDGES, "cone_cells": CONE.cells,
            "cases": cases, "unfinished": ["물리 측도·경계 삽입 사상·초기 준비와 분해능의 독립 고정",
                                          "모든 세분화에서 양립하는 동역학과 공통 계량 선택·전체 GR 다리"]}


if __name__ == "__main__":
    report = run()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "scope", "source_sha256")}, ensure_ascii=True))
