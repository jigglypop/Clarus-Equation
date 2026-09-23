"""Bool 사전 — 참과 거짓, 나와 아닌 나로 본 통로 계수 전체. 원장: 43장 §43.44. 예측값을 바꾸지 않는다.

계산 전에 적은 규칙:

어휘. 명제 n개(렌더링된 축)의 세계 = 참·거짓 배정 = 부분집합. 무차별(Ind): 모든 세계의 무게가 같다.
   사건은 여섯 가지로 제한한다: V(전부 거짓, 진공 세계), A(전부 참), M_k(k번째 = 나가 참), ODD(참의 수가 홀수),
   EXACT_j(정확히 j개 참), MAJ(과반 참). 여집합 C(S) = Sᶜ은 나 ↔ 아닌 나.
구조 정리(참이어야 함). (i) P(M_k) = P(ODD) = P(MAJ, n 홀수) = 1/2 (여집합 대칭).
   (ii) {V, EXACT₁, MAJ}는 B₃의 분할 → 각 π·(1/8, 3/8, 4/8), 합 π, 직각 강제(U1).
   (iii) Λ(C⁵)에서 C는 Q → −Q, 짝 ↔ 홀(입자 ↔ 반입자).
정직성. 어휘가 k/8(k = 1..7)을 얼마나 덮는지 센다. 거의 다 덮으면 사전은 읽기이며 증거로 세지 않는다.

python -B -m examples.physics.rendering.ce_rendering_bool
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from examples.physics.rendering import ce_rendering_e4_trace as E4


def worlds(n: int) -> list[frozenset[int]]:
    return [frozenset(s) for k in range(n + 1) for s in itertools.combinations(range(n), k)]


def vocabulary(n: int) -> dict[str, callable]:
    ev = {"V": lambda s: len(s) == 0, "A": lambda s: len(s) == n, "ODD": lambda s: len(s) % 2 == 1,
          "MAJ": lambda s: 2 * len(s) > n}
    ev.update({f"M{k + 1}": (lambda s, k=k: k in s) for k in range(n)})
    ev.update({f"EXACT{j}": (lambda s, j=j: len(s) == j) for j in range(n + 1)})
    return ev


def prob(n: int, event) -> Fraction:
    w = worlds(n)
    return Fraction(sum(1 for s in w if event(s)), len(w))


def complement_symmetry(n: int = 3) -> dict:
    """(i) 여집합 대칭으로 1/2이 되는 사건들, 그리고 C가 사건을 어떻게 옮기는지."""
    voc = vocabulary(n)
    full = frozenset(range(n))
    halves = {k: prob(n, voc[k]) for k in ("M1", "M2", "M3", "ODD", "MAJ")}
    maps = {}
    for a, b in (("V", "A"), ("MAJ", "EXACT0"), ("EXACT1", "EXACT2")):
        maps[f"C({a})"] = all(voc[a](s) == (voc[b](full - s) if b != "EXACT0" else len(full - s) < 2) for s in worlds(n)) \
            if a == "MAJ" else all(voc[a](s) == voc[b](full - s) for s in worlds(n))
    return {"halves": halves, "maps": maps}


def ckm_triangle_from_partition() -> dict:
    """(ii) {V, EXACT₁, MAJ}의 확률 × π = (β, γ, α)."""
    voc = vocabulary(3)
    parts = [voc["V"], voc["EXACT1"], voc["MAJ"]]
    disjoint_cover = all(sum(1 for e in parts if e(s)) == 1 for s in worlds(3))
    beta, gamma, alpha = (prob(3, e) for e in parts)
    return {"partition": disjoint_cover, "beta": beta, "gamma": gamma, "alpha": alpha,
            "deg": tuple(float(x) * 180 for x in (beta, gamma, alpha))}


def charge_conjugation_on_full_space() -> dict:
    """(iii) Λ(C⁵): 색 {0,1,2}, 약 {3,4}. C(S) = Sᶜ에서 Y → −Y, T₃ → −T₃, 짝 ↔ 홀."""
    full = frozenset(range(5))

    def charges(s: frozenset[int]) -> tuple[Fraction, Fraction]:
        c = sum(i in (0, 1, 2) for i in s)
        w = sum(i in (3, 4) for i in s)
        y = Fraction(w, 2) - Fraction(c, 3)
        t3 = (Fraction(1, 2) if 3 in s else Fraction(-1, 2)) if w == 1 else Fraction(0)
        return y, t3 + y

    ok_q = all(charges(full - s)[1] == -charges(s)[1] for s in worlds(5))
    ok_y = all(charges(full - s)[0] == -charges(s)[0] for s in worlds(5))
    parity = all((len(s) % 2) != (len(full - s) % 2) for s in worlds(5))
    even = [s for s in worlds(5) if len(s) % 2 == 0]
    return {"Q_flips": ok_q, "Y_flips": ok_y, "parity_flips": parity, "even_states": len(even),
            "even_is_generation": sorted(charges(s)[1] for s in even) == sorted(st["Q"] for st in E4.generation())}


def dictionary() -> list[dict]:
    """43장의 통로 계수와 Bool 사건(읽기). kind: 구조(여집합·분할이 강제) / 읽기(사건 선택)."""
    p3 = lambda e: prob(3, vocabulary(3)[e])
    return [
        {"coef": "s13^2 / delta", "value": Fraction(1, 8), "event": "M1 at stage 1, over 8", "bool": Fraction(1, 8), "kind": "structure (staircase)"},
        {"coef": "s12^2 shift / delta", "value": Fraction(2, 8), "event": "M2 at stage 2, over 8", "bool": Fraction(2, 8), "kind": "structure (staircase)"},
        {"coef": "s23^2 shift / delta (octant)", "value": Fraction(4, 8), "event": "M3 (me) at stage 3", "bool": p3("M3"), "kind": "structure (complement)"},
        {"coef": "CKM alpha / pi", "value": Fraction(1, 2), "event": "MAJ", "bool": p3("MAJ"), "kind": "structure (complement)"},
        {"coef": "CKM beta / pi = O1 tilt / pi", "value": Fraction(1, 8), "event": "V (vacuum world)", "bool": p3("V"), "kind": "reading"},
        {"coef": "CKM gamma / pi", "value": Fraction(3, 8), "event": "EXACT1", "bool": p3("EXACT1"), "kind": "structure (partition, given beta)"},
        {"coef": "R-Pl, v/M_Pl loop / (alpha_s/2pi)", "value": Fraction(4, 8), "event": "ODD", "bool": p3("ODD"), "kind": "structure (complement)"},
        {"coef": "sum-rule loop (5) / (alpha_s/2pi)", "value": Fraction(1, 8), "event": "V (vacuum world)", "bool": p3("V"), "kind": "reading"},
    ]


def coverage(n: int = 3) -> dict:
    """정직성: 어휘가 주는 서로 다른 확률과 k/8 덮개 비율."""
    vals = sorted({prob(n, e) for e in vocabulary(n).values()})
    targets = [Fraction(k, 2 ** n) for k in range(1, 2 ** n)]
    covered = [t for t in targets if t in vals]
    return {"distinct": [str(v) for v in vals], "covered": f"{len(covered)}/{len(targets)}"}


def vacuum_chain() -> dict:
    """읽기: P(V₃) = P(V₂)·P(¬나) = ¼·½ — 한계 π/4(= π·P(V₂))의 절반이 π/8(D′)이고 진공 세계 π·P(V₃)(U1)와 같다."""
    pv2 = prob(2, vocabulary(2)["V"])
    pv3 = prob(3, vocabulary(3)["V"])
    return {"P(V2)": pv2, "P(V3)": pv3, "P(V3) == P(V2)/2": pv3 == pv2 / 2}


def main() -> None:
    print("(i) complement:", {k: {kk: str(vv) for kk, vv in v.items()} for k, v in complement_symmetry().items()})
    print("(ii) CKM triangle from partition:", {k: (str(v) if isinstance(v, Fraction) else v) for k, v in ckm_triangle_from_partition().items()})
    print("(iii) Lambda(C^5) complement:", charge_conjugation_on_full_space())
    for d in dictionary():
        print(f"  {d['coef']:36s} {str(d['value']):5s} = P({d['event']}) = {str(d['bool']):5s} "
              f"{'OK' if d['value'] == d['bool'] else 'MISMATCH'}  [{d['kind']}]")
    print("coverage:", coverage(), "| vacuum chain:", {k: str(v) for k, v in vacuum_chain().items()})


if __name__ == "__main__":
    main()
