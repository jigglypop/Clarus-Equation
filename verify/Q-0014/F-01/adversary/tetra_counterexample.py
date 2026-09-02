"""adversary: CE 12.1 규칙만으로 2-cycle(S^2)이 닫히는 최소 인과 DAG.

주장 검사: 카드 사다리 2단 (iii) "fan에서 b_1=M=F이므로 모든 독립 1-cycle이
2-cell로 bound되어 3-cell이 불필요".

반례 구성: fine 인과 edge 다섯 개
    0->1, 1->2, 2->3, 0->2, 1->3      (acyclic, split/merge 규칙 안)
두 단계 coarse = {(0,2),(1,3),(0,3)}, 12.1 합성 face = 정확히 K4의 네 삼각형.
즉 block quotient 2-complex가 사면체의 경계 = 2-sphere이고 ker d2 != 0이다.
그 2-cycle을 bound하려면 3-cell이 필요하다.

부수 확인: fine ∩ coarse != 0 일 때 check_carrier의 holonomy 규약이
같은 1-cell에 서로 다른 두 holonomy를 준다(위상은 동일시, holonomy는 분리).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from examples.physics.causal_face_simplicity import composition_faces  # noqa: E402
import check_carrier as CC  # noqa: E402
from rerun_k1_homology import boundary2_ranks, betti1_independent, triangle_census, undirected  # noqa: E402


def main() -> int:
    fine = {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}
    coarse = CC.two_step_coarse(fine)
    faces = composition_faces(fine, coarse)
    skeleton = fine | coarse
    vertices = {v for e in skeleton for v in e}
    b1, comps = betti1_independent(skeleton, vertices)
    ranks = boundary2_ranks(fine, coarse, faces)
    census = triangle_census(fine, coarse, faces)
    h1 = b1 - ranks["rank_d2_Q"]
    h2 = ranks["F"] - ranks["rank_d2_Q"]

    # 명시 2-cycle: 사면체 경계 계수 (+1,-1,+1,-1)
    face_list = [(f.source, f.middle, f.target) for f in faces]
    eidx = {e: i for i, e in enumerate(sorted(undirected(skeleton)))}
    coeffs = {}
    signs = {(0, 1, 2): 1, (0, 1, 3): -1, (0, 2, 3): 1, (1, 2, 3): -1}
    residual = np.zeros(len(eidx))
    for (u, m, v) in face_list:
        s = signs.get((u, m, v), 0)
        coeffs[(u, m, v)] = s
        for (p, q), sg in (((u, m), 1.0), ((m, v), 1.0), ((u, v), -1.0)):
            key = (min(p, q), max(p, q))
            residual[eidx[key]] += s * (sg if p < q else -sg)

    # fine 과 coarse 가 같은 쌍을 공유하는가 (holonomy 이중정의)
    overlap = sorted(fine & coarse)

    # 큰 격자에서도 같은 중복이 있는지
    rng = np.random.default_rng(CC.SEED)
    overlaps = []
    for n in CC.SIZES:
        f = CC.split_merge_dag(n, rng)
        c = CC.two_step_coarse(f)
        faces_n = composition_faces(f, c)
        CC.face_holonomies(f, faces_n, rng)
        CC.split_only_tree(n, rng)
        overlaps.append({"n": n, "fine_and_coarse": len(f & c), "fine": len(f), "coarse": len(c)})

    out = {
        "minimal_counterexample": {
            "fine_edges": sorted(fine),
            "coarse_edges": sorted(coarse),
            "faces": face_list,
            "V": len(vertices), "E_undirected": ranks["E_undirected"], "F": ranks["F"],
            "b1": b1, "rank_d2_Q": ranks["rank_d2_Q"],
            "H1": h1, "H2_ker_d2": h2,
            "unbounded_triangles": census["unbounded"],
            "explicit_2cycle_coefficients": {str(k): v for k, v in coeffs.items()},
            "boundary_of_that_2cycle_linf": float(np.max(np.abs(residual))),
            "fine_cap_coarse": overlap,
        },
        "fine_cap_coarse_in_K1_grids": overlaps,
    }
    (HERE / "tetra_counterexample.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
