"""adversary: K1 하위조건 (a)~(e)가 씨앗을 바꿔도 값이 바뀔 수 있는가.

카드는 다섯 조건을 kill 로 적었다. 실제로 코드 위에서 반증 가능한 것은 몇 개인가를
씨앗 200개로 잰다. 값이 상수이면 그 조건은 '시험'이 아니라 코드의 정리다.
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

from examples.physics.gravity.causal_face_simplicity import composition_faces  # noqa: E402
import check_carrier as CC  # noqa: E402


def main() -> int:
    n = 50
    tree_b1, fm1, fall, unb, kcar = [], [], [], [], []
    for s in range(200):
        rng = np.random.default_rng(20260902 + s)
        fine = CC.split_merge_dag(n, rng)
        coarse = CC.two_step_coarse(fine)
        faces = composition_faces(fine, coarse)
        skeleton = fine | coarse
        a, b, _ = CC.face_holonomies(fine, faces, rng)
        tree = CC.split_only_tree(n, rng)
        tv = {x for e in tree for x in e}
        tree_b1.append(CC.betti_one(tree, tv))
        fm1.append(a)
        fall.append(b)
        unb.append(CC.unbounded_triangles(skeleton, faces))
        kcar.append(1 if False else (2 if (bool(faces) and b < 1.0) else 3))

    out = {
        "seeds": 200, "n": n,
        "K1a_k_carrier": {"distinct_values": sorted(set(kcar)),
                          "constant": len(set(kcar)) == 1,
                          "note": "코드에서 carrier_at_1=False 하드코딩; 값은 (e)와 동치"},
        "K1b_unbounded_min_cycles": {"min": int(min(unb)), "max": int(max(unb)),
                                     "zero_count": int(sum(1 for x in unb if x == 0)),
                                     "constant": len(set(unb)) == 1},
        "K1c_tree_b1": {"distinct_values": sorted(set(tree_b1)),
                        "constant": len(set(tree_b1)) == 1,
                        "note": "split_only_tree 는 정점마다 부모 1개 => 항상 forest"},
        "K1d_flat_fraction_M1": {"distinct_values": sorted(set(fm1)),
                                 "constant": len(set(fm1)) == 1,
                                 "note": "M=1 이면 chosen=m 이므로 U_face=U U^{-1}=I, 정의상 항등"},
        "K1e_flat_fraction_all": {"min": min(fall), "max": max(fall),
                                  "hit_one_count": int(sum(1 for x in fall if x == 1.0))},
    }
    out["falsifiable_subconditions"] = [
        key for key, val in out.items()
        if isinstance(val, dict) and not val.get("constant", False) and key.startswith("K1")
    ]
    (HERE / "kill_falsifiability.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
