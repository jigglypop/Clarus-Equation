"""E4의 식별 (ii) — 섞임각은 단일항 구조로 강제되지 않는다. 원장: 43장 §43.79. 예측값을 바꾸지 않는다.

(i) α_s = a³은 E1(렌더링은 색 축에만) 아래 외대수 단일항의 유일한 자명하지 않은 무게로 강제되었다(§43.78).
계산 전에 적은 판본과 판정:

(가) 교환 관계. 동전의 범위를 E1(색 축만)과 5축 전부 두 가지로 두고 단일항 무게를 센다.
    (i) 강제: 색 가득 단일항 후보 무게가 하나인가. (ii) 단일항 경로: 약 단일항 Λ²V₂가 자명하지 않은 무게를 갖는가.
(나) E1 안의 후보 12개: 대상 {선 Λ¹V₃, 면 Λ²V₃, 부피 Λ³V₃} × 읽기 {det(확률), tr·det(진폭)} × 과녁 {sin θ_W, ŝ²}.
    각각 ŝ²에서 α_s를 예측해 세계 평균 0.1180 ± 0.0009와 1σ 안인지 본다(E4는 자료로 찾은 식이라 적중은 증거 아님).
판정. 두 식별이 동시에 강제되는 동전 범위가 없으면 (ii)는 이 구조로 유도되지 않는다. E4의 남은 공리 내용은
    “섞임각 = 색 2-평면의 회전 진폭” 한 줄이다.

python -B -m examples.physics.rendering.ce_rendering_e4_ii
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_singlets as SG

AS_WORLD = (0.1180, 0.0009)


def tradeoff() -> dict:
    out = {}
    for scope, weak_coin in (("E1 (colour only)", False), ("all five axes", True)):
        rows = {}
        for s in SG.singlets():
            power = s["k"] + (s["j"] if weak_coin else 0)
            rows[(s["k"], s["j"])] = power
        colour_full = sorted({p for (k, j), p in rows.items() if k == 3})
        weak_singlet = rows[(0, 2)]
        out[scope] = {"colour_full_powers": colour_full, "i_forced": len(colour_full) == 1,
                      "weak_singlet_power": weak_singlet, "ii_from_singlet": weak_singlet > 0}
    out["both_forced_somewhere"] = any(v["i_forced"] and v["ii_from_singlet"] for v in out.values())
    return out


def family() -> dict:
    s2 = R.SZ2
    s = math.sqrt(s2)
    rows = {}
    for m, obj in ((1, "line"), (2, "plane"), (3, "volume")):
        for reading, coef in (("det", 1), ("tr.det", m)):
            for target, value in (("sin", s), ("s2", s2)):
                a = (value / coef) ** (1 / m)
                alpha = a ** 3
                rows[f"{obj}|{reading}|{target}"] = {"alpha_s": alpha, "pull": (alpha - AS_WORLD[0]) / AS_WORLD[1]}
    hits = sorted(k for k, v in rows.items() if abs(v["pull"]) <= 1)
    return {"rows": rows, "hits": hits}


def main() -> None:
    for k, v in tradeoff().items():
        print(k, v)
    f = family()
    for k, v in f["rows"].items():
        print(f"  {k:22s} alpha_s={v['alpha_s']:.5f} pull={v['pull']:+9.2f}")
    print("hits:", f["hits"])


if __name__ == "__main__":
    main()
