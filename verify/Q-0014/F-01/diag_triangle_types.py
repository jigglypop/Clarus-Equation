"""Q-0014 F-01 사후 진단 (kill 아님, 사전등록 값 변경 없음).

check_carrier.py가 K1 하위조건 (b) `unbounded_min_cycles != 0`으로 발동한 뒤,
발동 원인을 분류하기 위해 돌린 진단이다. 이 파일의 어떤 숫자도 카드의
예측·창·tolerance를 바꾸지 않는다.

결과 요약(seed 20260902, n=50/100/200):
  - 세 변이 모두 fine인 삼각형은 21/32/62개이고 **전부** 합성 face로 덮인다
    (assert 통과). 즉 사다리 2단의 의도한 주장 "acyclic 순서의 fine 3-cycle은
    transitive triangle이므로 반드시 face가 붙는다"는 참이다.
  - unbounded 807개(211/293/589)는 모두 coarse edge를 짧은 변으로 포함하는
    혼합 삼각형이다(CCF 671, CCC 243, CFF 51, CCFC 89, CFFC 35, CFCFC 4).
    카드가 kill (b)를 fine 1-skeleton이 아니라 fine∪coarse 전체 1-skeleton 위에
    적었기 때문에 발동했다.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from examples.physics.causal_face_simplicity import composition_faces  # noqa: E402
from check_carrier import SEED, SIZES, split_merge_dag, two_step_coarse  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(SEED)
    for n in SIZES:
        fine = split_merge_dag(n, rng)
        coarse = two_step_coarse(fine)
        faces = composition_faces(fine, coarse)
        face_keys = {frozenset((f.source, f.middle, f.target)) for f in faces}
        skeleton = fine | coarse
        adjacency: dict[int, set[int]] = {}
        for u, v in skeleton:
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

        seen: set[frozenset[int]] = set()
        kinds: dict[str, int] = {}
        for x, neighbours in adjacency.items():
            for y, z in combinations(sorted(neighbours), 2):
                if z not in adjacency.get(y, ()):
                    continue
                key = frozenset((x, y, z))
                if key in seen:
                    continue
                seen.add(key)
                if key in face_keys:
                    continue
                a, b, c = sorted(key)
                tags = []
                for p, q in ((a, b), (b, c), (a, c)):
                    tag = ("F" if (p, q) in fine else "") + ("C" if (p, q) in coarse else "")
                    tags.append(tag or "-")
                label = "".join(sorted(tags))
                kinds[label] = kinds.get(label, 0) + 1

        pure_fine = 0
        for key in seen:
            a, b, c = sorted(key)
            if (a, b) in fine and (b, c) in fine and (a, c) in fine:
                pure_fine += 1
                assert key in face_keys, ("pure fine triangle without face", key)

        print(
            f"n={n} triangles={len(seen)} faces={len(face_keys)} "
            f"unbounded={sum(kinds.values())} pure_fine={pure_fine} (all bounded) kinds={kinds}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
