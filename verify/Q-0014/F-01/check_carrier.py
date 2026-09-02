"""Q-0014 F-01 카드 K1 smoke: split/merge 2-complex의 첫 곡률 carrier 차수.

카드가 사전등록한 값(사후 변경 금지):
  k_carrier = 2  (n in {50,100,200})
  unbounded_min_cycles = 0
  tree_b1 = 0
  flat_fraction_M1 = 1.0
  flat_fraction_all < 1.0
  P(F_b>=2) = 0.6399752045 / 0.9312576369 / 0.9890448472 / 0.9984000309  (b=1..4)
  b_95^curv = 3, b_99^curv = 4,  12.2의 F>=4 표와 b_95=4 / b_99=5 재현

carrier 차수의 정의(카드 scope):
  k = 경계가 0이 아닌 닫힌 (k-1)-cycle이고 그 loop의 holonomy가 구조적으로
  항등이 아닌 최소 cell 차수. 1-cell의 경계는 0-chain이라 닫힌 loop이 아니므로 k>=2.

씨앗 20260902. 표준 라이브러리 + numpy만 쓴다.
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.physics.causal_face_simplicity import composition_faces  # noqa: E402

SEED = 20260902
SIZES = (50, 100, 200)
PARENT_WINDOW = 8
MAX_PARENTS = 3
FLAT_TOL = 1.0e-12
MU = 3.1777584234 - 1.0  # 12.2: epoch당 face 강도


# ------------------------------------------------------------------ 인과 DAG

def split_merge_dag(n: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    """split/merge 인과 DAG: 새 정점은 최근 창 안의 기존 정점에서 1..3개 부모를 뽑는다.

    한 부모의 여러 자식 = split, 한 자식의 여러 부모 = merge. 위상 순서가 곧
    인과 순서이므로 acyclic이 보장된다.
    """
    edges: set[tuple[int, int]] = set()
    for v in range(1, n):
        low = max(0, v - PARENT_WINDOW)
        pool = list(range(low, v))
        count = min(len(pool), int(rng.integers(1, MAX_PARENTS + 1)))
        parents = rng.choice(np.array(pool), size=count, replace=False)
        for u in parents:
            edges.add((int(u), v))
    return edges


def split_only_tree(n: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    """merge를 끈 대조군: 모든 정점이 부모 하나(=tree, split만)."""
    edges: set[tuple[int, int]] = set()
    for v in range(1, n):
        low = max(0, v - PARENT_WINDOW)
        u = int(rng.integers(low, v))
        edges.add((u, v))
    return edges


def two_step_coarse(fine: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """블록 깊이 b=1: 정확히 2 fine step으로 도달하는 쌍을 coarse continuation으로 선언."""
    out: dict[int, set[int]] = {}
    for u, v in fine:
        out.setdefault(u, set()).add(v)
    coarse: set[tuple[int, int]] = set()
    for u, mids in out.items():
        for m in mids:
            for v in out.get(m, ()):  # u -> m -> v
                if v != u:
                    coarse.add((u, v))
    return coarse


def betti_one(edges: set[tuple[int, int]], vertices: set[int]) -> int:
    """b_1 = E - V + C (무향 1-skeleton)."""
    parent = {v: v for v in vertices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    undirected = {(min(u, v), max(u, v)) for u, v in edges}
    for u, v in undirected:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    components = len({find(v) for v in vertices})
    return len(undirected) - len(vertices) + components


# ------------------------------------------------------------------ holonomy

def random_su2(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]], dtype=complex)


def face_holonomies(
    fine: set[tuple[int, int]],
    faces: tuple,
    rng: np.random.Generator,
) -> tuple[float, float, dict[int, int]]:
    """블록 렌더링 규칙(카드 scope 공리): coarse holonomy := 사전순 첫 factorization의 곱.

    반환: (M=1 coarse edge 위 face의 flat 비율, 전체 face의 flat 비율, M 분포)
    """
    edge_hol = {edge: random_su2(rng) for edge in sorted(fine)}
    by_coarse: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        by_coarse.setdefault((face.source, face.target), []).append(face.middle)

    flat_m1 = total_m1 = flat_all = total_all = 0
    m_hist: dict[int, int] = {}
    identity = np.eye(2, dtype=complex)
    for (u, v), middles in by_coarse.items():
        middles = sorted(middles)
        m_hist[len(middles)] = m_hist.get(len(middles), 0) + 1
        chosen = middles[0]
        u_coarse = edge_hol[(u, chosen)] @ edge_hol[(chosen, v)]
        for m in middles:
            u_face = edge_hol[(u, m)] @ edge_hol[(m, v)] @ np.linalg.inv(u_coarse)
            is_flat = float(np.linalg.norm(u_face - identity)) < FLAT_TOL
            total_all += 1
            flat_all += int(is_flat)
            if len(middles) == 1:
                total_m1 += 1
                flat_m1 += int(is_flat)
    frac_m1 = flat_m1 / total_m1 if total_m1 else float("nan")
    frac_all = flat_all / total_all if total_all else float("nan")
    return frac_m1, frac_all, m_hist


# ------------------------------------------------------------------ 최소 cycle

def unbounded_triangles(edges: set[tuple[int, int]], faces: tuple) -> int:
    """길이 3의 무향 cycle 중 합성 face가 붙지 않은 것의 수."""
    face_keys = {frozenset((f.source, f.middle, f.target)) for f in faces}
    adjacency: dict[int, set[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    seen: set[frozenset[int]] = set()
    unbounded = 0
    for x, neighbours in adjacency.items():
        for y, z in combinations(sorted(neighbours), 2):
            if z in adjacency.get(y, ()):
                key = frozenset((x, y, z))
                if key in seen:
                    continue
                seen.add(key)
                if key not in face_keys:
                    unbounded += 1
    return unbounded


# ------------------------------------------------------------------ Poisson

def poisson_tail(threshold: int, lam: float) -> float:
    return 1.0 - sum(math.exp(-lam) * lam**j / math.factorial(j) for j in range(threshold))


def first_depth(threshold: int, level: float) -> int:
    b = 1
    while poisson_tail(threshold, b * MU) < level:
        b += 1
    return b


# ------------------------------------------------------------------ main

def main() -> int:
    rng = np.random.default_rng(SEED)
    per_size = []
    for n in SIZES:
        fine = split_merge_dag(n, rng)
        coarse = two_step_coarse(fine)
        faces = composition_faces(fine, coarse)
        skeleton = fine | coarse
        vertices = {v for edge in skeleton for v in edge}

        frac_m1, frac_all, m_hist = face_holonomies(fine, faces, rng)
        # dim 1: edge의 경계는 0-chain(v-u)이라 닫힌 loop이 아니다 -> 1-carrier 없음.
        carrier_at_1 = False
        # dim 2: 경계가 0이 아닌 삼각 1-cycle이면서 holonomy가 항등이 아닌 face가 있는가.
        carrier_at_2 = bool(faces) and frac_all < 1.0
        k_carrier = 1 if carrier_at_1 else (2 if carrier_at_2 else 3)

        tree = split_only_tree(n, rng)
        tree_vertices = {v for edge in tree for v in edge}

        per_size.append(
            {
                "n": n,
                "fine_edges": len(fine),
                "coarse_edges": len(coarse),
                "faces": len(faces),
                "b1_skeleton": betti_one(skeleton, vertices),
                "k_carrier": k_carrier,
                "unbounded_min_cycles": unbounded_triangles(skeleton, faces),
                "tree_b1": betti_one(tree, tree_vertices),
                "flat_fraction_M1": frac_m1,
                "flat_fraction_all": frac_all,
                "M_histogram": {str(key): m_hist[key] for key in sorted(m_hist)},
                "mean_M": sum(k * c for k, c in m_hist.items()) / max(sum(m_hist.values()), 1),
            }
        )

    poisson = {
        "P_F_ge2": [poisson_tail(2, b * MU) for b in (1, 2, 3, 4, 5)],
        "P_F_ge4": [poisson_tail(4, b * MU) for b in (1, 2, 3, 4, 5)],
        "b95_curv": first_depth(2, 0.95),
        "b99_curv": first_depth(2, 0.99),
        "b95_F4": first_depth(4, 0.95),
        "b99_F4": first_depth(4, 0.99),
    }
    poisson["gap_b95"] = poisson["b95_F4"] - poisson["b95_curv"]

    stats = {
        "k_carrier": sorted({row["k_carrier"] for row in per_size}),
        "unbounded_min_cycles": max(row["unbounded_min_cycles"] for row in per_size),
        "tree_b1": max(row["tree_b1"] for row in per_size),
        "flat_fraction_M1": min(row["flat_fraction_M1"] for row in per_size),
        "flat_fraction_all": max(row["flat_fraction_all"] for row in per_size),
    }
    result = {
        "card": "derivations/Q-0014/F-01.formula.md",
        "kill": "K1",
        "seed": SEED,
        "sizes": list(SIZES),
        "block_depth": 1,
        "stats": stats,
        "per_size": per_size,
        "poisson": poisson,
        "prereg": {
            "k_carrier": 2,
            "unbounded_min_cycles": 0,
            "tree_b1": 0,
            "flat_fraction_M1": 1.0,
            "flat_fraction_all_lt": 1.0,
            "b95_curv": 3,
            "b99_curv": 4,
            "gap_b95": 1,
            "P_F_ge2_b1_to_b4": [0.6399752045, 0.9312576369, 0.9890448472, 0.9984000309],
        },
    }
    verdict = (
        stats["k_carrier"] == [2]
        and stats["unbounded_min_cycles"] == 0
        and stats["tree_b1"] == 0
        and stats["flat_fraction_M1"] == 1.0
        and stats["flat_fraction_all"] < 1.0
        and poisson["b95_curv"] == 3
        and poisson["b99_curv"] == 4
        and poisson["gap_b95"] == 1
    )
    result["kill_fired"] = not verdict
    out = Path(__file__).resolve().parent / "result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stats": stats, "poisson": poisson, "kill_fired": result["kill_fired"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
