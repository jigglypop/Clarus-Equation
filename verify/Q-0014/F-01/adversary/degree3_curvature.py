"""adversary: 2-cycle(ker d2) 위의 3-곡률이 0인가 — k=3 분기가 살아 있는가.

아벨(U(1)) 판본에서 surface holonomy 는 모호성 없이 정의된다.
face f=(u,m,v) 의 defect  D_f = Theta(u,m) + Theta(m,v) - Theta(u,v).
2-cycle z (d2 z = 0) 위의 3-곡률 = sum_f z_f D_f, 이는 fine edge 각도들의
정수 선형결합이므로 계수벡터가 항등적으로 0인지 아닌지가 theta 무관하게 판정된다.

두 규약을 비교한다.
  (I)  코드 규약 (check_carrier.py): fine edge 는 독립 각도를 갖고,
       coarse edge 는 사전순 첫 factorization 의 합. fine ∩ coarse 는 두 값을 갖는다.
  (II) 카드 scope 공리를 정합적으로 강제: 동일시된 1-cell 하나에 값 하나.
       (u,v) 가 coarse 이면 Theta(u,v) := Theta(u,m0) + Theta(m0,v) 로 재귀 정의.

(I)에서 3-곡률이 0이 아니면 degree-3 carrier 가 존재하고, 카드의 K4가 스스로
말한 대로 k=min 이 아니라 readout 규칙이 미정이 된다.
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

from examples.physics.causal_face_simplicity import composition_faces  # noqa: E402
import check_carrier as CC  # noqa: E402


def build(fine, coarse, faces):
    edge_list = sorted({(min(u, v), max(u, v)) for u, v in (fine | coarse)})
    eidx = {e: i for i, e in enumerate(edge_list)}
    fine_list = sorted(fine)
    fidx = {e: i for i, e in enumerate(fine_list)}
    first_middle = {}
    for f in faces:
        key = (f.source, f.target)
        m = first_middle.get(key)
        if m is None or f.middle < m:
            first_middle[key] = f.middle
    d2 = np.zeros((len(edge_list), len(faces)))
    for j, f in enumerate(faces):
        u, m, v = f.source, f.middle, f.target
        for (p, q), sg in (((u, m), 1.0), ((m, v), 1.0), ((u, v), -1.0)):
            key = (min(p, q), max(p, q))
            d2[eidx[key], j] += sg if p < q else -sg
    return edge_list, eidx, fine_list, fidx, first_middle, d2


def theta_vectors(fine_list, fidx, first_middle, conv):
    """Theta(u,v) 를 fine edge 기저 위의 정수 벡터로 반환하는 함수."""
    n = len(fine_list)

    @lru_cache(maxsize=None)
    def theta(u, v):
        vec = np.zeros(n)
        if conv == "I":
            if (u, v) in fidx:
                vec[fidx[(u, v)]] += 1.0
                return tuple(vec)
            m = first_middle[(u, v)]
            a = np.array(theta(u, m)); b = np.array(theta(m, v))
            return tuple(a + b)
        # conv II: coarse 이면 무조건 재귀 치환 (동일시된 1-cell 에 값 하나)
        if (u, v) in first_middle:
            m = first_middle[(u, v)]
            a = np.array(theta(u, m)); b = np.array(theta(m, v))
            return tuple(a + b)
        vec[fidx[(u, v)]] += 1.0
        return tuple(vec)

    return theta


def curvature3_rank(fine, coarse, faces):
    edge_list, eidx, fine_list, fidx, first_middle, d2 = build(fine, coarse, faces)
    F = len(faces)
    if F == 0:
        return None
    # ker d2 기저 (SVD 영공간)
    u_, s_, vt_ = np.linalg.svd(d2)
    tol = max(d2.shape) * (s_[0] if s_.size else 0.0) * 1e-12
    rank = int((s_ > max(tol, 1e-9)).sum())
    null = vt_[rank:].T  # F x nullity
    result = {}
    for conv in ("I", "II"):
        theta = theta_vectors(fine_list, fidx, first_middle, conv)
        W = np.zeros((len(fine_list), F))
        for j, f in enumerate(faces):
            u, m, v = f.source, f.middle, f.target
            W[:, j] = (np.array(theta(u, m)) + np.array(theta(m, v))
                       - np.array(theta(u, v)))
        img = W @ null                      # fine_edges x nullity
        r = int(np.linalg.matrix_rank(img, tol=1e-8)) if img.size else 0
        result[conv] = {
            "rank_d2": rank, "nullity_ker_d2": F - rank,
            "rank_of_3curvature_on_ker_d2": r,
            "max_abs_3curvature_coeff": float(np.max(np.abs(img))) if img.size else 0.0,
            "degree3_carrier_present": r > 0,
        }
    return result


def main() -> int:
    out = {}
    # 최소 반례: 사면체
    fine = {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}
    coarse = CC.two_step_coarse(fine)
    faces = composition_faces(fine, coarse)
    out["tetrahedron"] = curvature3_rank(fine, coarse, faces)

    # K1 사전등록 격자
    rng = np.random.default_rng(CC.SEED)
    grid = []
    for n in CC.SIZES:
        f = CC.two_step_coarse
        fn = CC.split_merge_dag(n, rng)
        cn = f(fn)
        fa = composition_faces(fn, cn)
        res = curvature3_rank(fn, cn, fa)
        res["n"] = n
        grid.append(res)
        CC.face_holonomies(fn, fa, rng)
        CC.split_only_tree(n, rng)
    out["k1_grids"] = grid
    (HERE / "degree3_curvature.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
