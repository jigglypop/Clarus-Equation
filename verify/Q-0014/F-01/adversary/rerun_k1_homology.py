"""adversary: Q-0014 F-01 K1 독립 재현 + 2-complex 호몰로지.

목적 세 가지.
  (1) prover가 보고한 K1 발동(unbounded_min_cycles != 0)이 같은 seed/격자에서
      재현되는지 독립 구현으로 확인한다.
  (2) 발동 원인이 "혼합 삼각형"인지 같은 복합체 위에서 분류한다
      (prover의 diag_triangle_types.py는 rng 소비 순서가 달라 n=100/200에서
       다른 DAG를 본다 -- 그것도 여기서 확인한다).
  (3) 카드 사다리 2단 (iii) "모든 독립 1-cycle이 2-cell로 bound"를
      실제 복합체에서 H_1 = ker d1 / im d2 로 검사하고, ker d2 (= H_2)도 잰다.

카드의 어떤 숫자도 바꾸지 않는다. 읽기 전용.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from examples.physics.gravity.causal_face_simplicity import composition_faces  # noqa: E402
import check_carrier as CC  # noqa: E402


def undirected(edges):
    return {(min(u, v), max(u, v)) for u, v in edges}


def betti1_independent(edges, vertices):
    """b_1 = E - V + C, union-find 없이 numpy 인접행렬 BFS로 독립 구현."""
    adj = {}
    for u, v in undirected(edges):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen = set()
    comps = 0
    for s in vertices:
        if s in seen:
            continue
        comps += 1
        stack = [s]
        seen.add(s)
        while stack:
            x = stack.pop()
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return len(undirected(edges)) - len(vertices) + comps, comps


def triangle_census(fine, coarse, faces):
    """skeleton(=fine|coarse)의 모든 무향 삼각형을 분류."""
    skeleton = fine | coarse
    face_keys = {frozenset((f.source, f.middle, f.target)) for f in faces}
    adj = {}
    for u, v in skeleton:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    seen = set()
    kinds = {}
    pure_fine_total = 0
    pure_fine_unbounded = 0
    unbounded = 0
    for x, nb in adj.items():
        for y, z in combinations(sorted(nb), 2):
            if z not in adj.get(y, ()):
                continue
            key = frozenset((x, y, z))
            if key in seen:
                continue
            seen.add(key)
            a, b, c = sorted(key)
            is_pure_fine = all(p in fine for p in ((a, b), (b, c), (a, c)))
            if is_pure_fine:
                pure_fine_total += 1
            if key in face_keys:
                continue
            unbounded += 1
            if is_pure_fine:
                pure_fine_unbounded += 1
            tags = []
            for p, q in ((a, b), (b, c), (a, c)):
                tag = ("F" if (p, q) in fine else "") + ("C" if (p, q) in coarse else "")
                tags.append(tag or "-")
            label = "".join(sorted(tags))
            kinds[label] = kinds.get(label, 0) + 1
    return {
        "triangles_total": len(seen),
        "unbounded": unbounded,
        "pure_fine_total": pure_fine_total,
        "pure_fine_unbounded": pure_fine_unbounded,
        "kinds": dict(sorted(kinds.items())),
    }


def boundary2_ranks(fine, coarse, faces):
    """d2: C_2 -> C_1 의 계수를 Q(수치 SVD)와 GF(2)에서 잰다."""
    skeleton = fine | coarse
    edge_list = sorted(undirected(skeleton))
    eidx = {e: i for i, e in enumerate(edge_list)}
    E = len(edge_list)
    F = len(faces)
    mat = np.zeros((E, F), dtype=float)
    bits = [0] * F  # GF(2) 열
    for j, f in enumerate(faces):
        u, m, v = f.source, f.middle, f.target
        for (p, q), sign in (((u, m), 1.0), ((m, v), 1.0), ((u, v), -1.0)):
            key = (min(p, q), max(p, q))
            i = eidx[key]
            # 무향 기저: 방향 부호를 (작은쪽 -> 큰쪽) 기준으로 맞춘다
            s = sign if p < q else -sign
            mat[i, j] += s
            bits[j] ^= 1 << i
    rank_q = int(np.linalg.matrix_rank(mat, tol=1e-8)) if F else 0
    # GF(2) 소거
    pivots = []
    cols = list(bits)
    rank_2 = 0
    used = {}
    for col in cols:
        cur = col
        while cur:
            top = cur.bit_length() - 1
            if top in used:
                cur ^= used[top]
            else:
                used[top] = cur
                rank_2 += 1
                break
    return {"E_undirected": E, "F": F, "rank_d2_Q": rank_q, "rank_d2_GF2": rank_2}


def main() -> int:
    rng = np.random.default_rng(CC.SEED)
    rows = []
    for n in CC.SIZES:
        fine = CC.split_merge_dag(n, rng)
        coarse = CC.two_step_coarse(fine)
        faces = composition_faces(fine, coarse)
        skeleton = fine | coarse
        vertices = {v for e in skeleton for v in e}

        frac_m1, frac_all, m_hist = CC.face_holonomies(fine, faces, rng)
        tree = CC.split_only_tree(n, rng)
        tree_vertices = {v for e in tree for v in e}

        b1, comps = betti1_independent(skeleton, vertices)
        b1_tree, _ = betti1_independent(tree, tree_vertices)
        fine_only_b1, _ = betti1_independent(fine, {v for e in fine for v in e})
        census = triangle_census(fine, coarse, faces)
        ranks = boundary2_ranks(fine, coarse, faces)
        h1 = b1 - ranks["rank_d2_Q"]
        h2 = ranks["F"] - ranks["rank_d2_Q"]

        rows.append({
            "n": n,
            "fine_edges": len(fine),
            "coarse_edges": len(coarse),
            "faces": len(faces),
            "components": comps,
            "b1_skeleton": b1,
            "b1_fine_only": fine_only_b1,
            "tree_b1": b1_tree,
            "flat_fraction_M1": frac_m1,
            "flat_fraction_all": frac_all,
            "M_histogram": {str(k): m_hist[k] for k in sorted(m_hist)},
            "triangle_census": census,
            "boundary2": ranks,
            "H1_dim": h1,
            "H2_dim_ker_d2": h2,
        })

    stats = {
        "unbounded_min_cycles_max": max(r["triangle_census"]["unbounded"] for r in rows),
        "unbounded_per_size": [r["triangle_census"]["unbounded"] for r in rows],
        "pure_fine_unbounded_per_size": [r["triangle_census"]["pure_fine_unbounded"] for r in rows],
        "tree_b1_max": max(r["tree_b1"] for r in rows),
        "flat_fraction_M1_min": min(r["flat_fraction_M1"] for r in rows),
        "flat_fraction_all_max": max(r["flat_fraction_all"] for r in rows),
        "H1_per_size": [r["H1_dim"] for r in rows],
        "H2_per_size": [r["H2_dim_ker_d2"] for r in rows],
    }
    out = {
        "what": "adversary independent rerun of Q-0014 F-01 K1 (seed/grid unchanged)",
        "seed": CC.SEED,
        "sizes": list(CC.SIZES),
        "stats": stats,
        "per_size": rows,
        "kill_b_fires": stats["unbounded_min_cycles_max"] != 0,
    }
    (HERE / "rerun_k1_homology.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
