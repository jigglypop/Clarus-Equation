"""adversary(정정판): check_carrier 의 실제 holonomy 규약을 그대로 옮겨
2-cycle 위 3-곡률을 잰다. degree3_curvature.py 의 규약 I 구현은 긴 변이
fine 이기도 하면 fine 값을 썼는데, check_carrier 는 긴 변에 대해 항상
factorization 곱(u_coarse)을 쓴다. 그 차이를 바로잡았다.

규약
  (I)  코드 규약: 짧은 두 변 = fine 독립 각도, 긴 변 = 사전순 첫 factorization 의 합.
       => fine ∩ coarse 인 1-cell 은 두 값을 갖는다(위상은 동일시, holonomy 는 분리).
  (II) 카드 scope 공리를 정합적으로: 동일시된 1-cell 하나에 값 하나 (재귀 치환).

측정
  A. 동일시 복합체의 b1 / rank d2 / H1 / H2, 그리고 ker d2 위 3-곡률 계수 rank.
  B. fine 과 coarse 를 서로 다른 1-cell 로 둔 복합체의 같은 값들.
  C. 각 규약에서 face flat 비율 (K1(d),(e) 가 구조적으로 결정되는지).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from examples.physics.gravity.causal_face_simplicity import composition_faces  # noqa: E402
import check_carrier as CC  # noqa: E402


def analyse(fine, coarse, faces):
    fine_list = sorted(fine)
    fidx = {e: i for i, e in enumerate(fine_list)}
    first_middle = {}
    for f in faces:
        key = (f.source, f.target)
        m = first_middle.get(key)
        if m is None or f.middle < m:
            first_middle[key] = f.middle

    # ---- A. 동일시 복합체
    ident_edges = sorted({(min(u, v), max(u, v)) for u, v in (fine | coarse)})
    ie = {e: i for i, e in enumerate(ident_edges)}
    d2_ident = np.zeros((len(ident_edges), len(faces)))
    for j, f in enumerate(faces):
        u, m, v = f.source, f.middle, f.target
        for (p, q), sg in (((u, m), 1.0), ((m, v), 1.0), ((u, v), -1.0)):
            key = (min(p, q), max(p, q))
            d2_ident[ie[key], j] += sg if p < q else -sg

    # ---- B. 분리 복합체 (fine 사본 / coarse 사본)
    dist_edges = [("f", e) for e in fine_list] + [("c", e) for e in sorted(coarse)]
    de = {e: i for i, e in enumerate(dist_edges)}
    d2_dist = np.zeros((len(dist_edges), len(faces)))
    for j, f in enumerate(faces):
        u, m, v = f.source, f.middle, f.target
        d2_dist[de[("f", (u, m))], j] += 1.0
        d2_dist[de[("f", (m, v))], j] += 1.0
        d2_dist[de[("c", (u, v))], j] -= 1.0

    def nullspace(mat):
        if mat.shape[1] == 0:
            return 0, np.zeros((0, 0))
        _, s, vt = np.linalg.svd(mat)
        tol = max(mat.shape) * (s[0] if s.size else 0.0) * 1e-12
        r = int((s > max(tol, 1e-9)).sum())
        return r, vt[r:].T

    rank_i, null_i = nullspace(d2_ident)
    rank_d, null_d = nullspace(d2_dist)

    # ---- holonomy 규약별 face defect 를 fine-edge 계수 벡터로
    @lru_cache(maxsize=None)
    def theta_II(u, v):
        if (u, v) in first_middle:
            m = first_middle[(u, v)]
            return tuple(np.array(theta_II(u, m)) + np.array(theta_II(m, v)))
        vec = np.zeros(len(fine_list)); vec[fidx[(u, v)]] = 1.0
        return tuple(vec)

    W = {}
    for conv in ("I", "II"):
        mat = np.zeros((len(fine_list), len(faces)))
        for j, f in enumerate(faces):
            u, m, v = f.source, f.middle, f.target
            if conv == "I":
                m0 = first_middle[(u, v)]
                col = np.zeros(len(fine_list))
                col[fidx[(u, m)]] += 1.0
                col[fidx[(m, v)]] += 1.0
                col[fidx[(u, m0)]] -= 1.0
                col[fidx[(m0, v)]] -= 1.0
            else:
                col = (np.array(theta_II(u, m)) + np.array(theta_II(m, v))
                       - np.array(theta_II(u, v)))
            mat[:, j] = col
        W[conv] = mat

    res = {
        "V": len({x for e in (fine | coarse) for x in e}),
        "fine": len(fine), "coarse": len(coarse), "fine_cap_coarse": len(fine & coarse),
        "faces": len(faces),
        "identified": {"E": len(ident_edges), "rank_d2": rank_i,
                       "H2_ker_d2": len(faces) - rank_i},
        "distinct": {"E": len(dist_edges), "rank_d2": rank_d,
                     "H2_ker_d2": len(faces) - rank_d},
    }
    for conv in ("I", "II"):
        flat = int(np.sum(np.all(np.abs(W[conv]) < 1e-12, axis=0)))
        img_i = W[conv] @ null_i if null_i.size else np.zeros((len(fine_list), 0))
        img_d = W[conv] @ null_d if null_d.size else np.zeros((len(fine_list), 0))
        res[f"conv_{conv}"] = {
            "flat_faces": flat, "flat_fraction_all": flat / max(len(faces), 1),
            "rank_3curv_on_ker_d2_identified": int(np.linalg.matrix_rank(img_i, tol=1e-8))
            if img_i.size else 0,
            "max_abs_3curv_identified": float(np.max(np.abs(img_i))) if img_i.size else 0.0,
            "rank_3curv_on_ker_d2_distinct": int(np.linalg.matrix_rank(img_d, tol=1e-8))
            if img_d.size else 0,
        }
    return res


def main() -> int:
    out = {}
    fine = {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}
    coarse = CC.two_step_coarse(fine)
    faces = composition_faces(fine, coarse)
    out["tetrahedron"] = analyse(fine, coarse, faces)

    rng = np.random.default_rng(CC.SEED)
    grids = []
    for n in CC.SIZES:
        fn = CC.split_merge_dag(n, rng)
        cn = CC.two_step_coarse(fn)
        fa = composition_faces(fn, cn)
        r = analyse(fn, cn, fa); r["n"] = n
        grids.append(r)
        CC.face_holonomies(fn, fa, rng)
        CC.split_only_tree(n, rng)
    out["k1_grids"] = grids
    (HERE / "degree3_curvature_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
