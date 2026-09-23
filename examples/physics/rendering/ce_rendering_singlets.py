"""가둠 ⇔ 붙잡힘 — 외대수의 단일항과 렌더링 무게. 원장: 43장 §43.78. 예측값을 바꾸지 않는다.

한 세대 = Λ(V₃ ⊕ V₂)의 32 상태(§43.20). 색 축의 초전하 −1/3, 약 축 +1/2(Higgs 통로 (1,2,+½), §43.22).
E1: 렌더링 연산 R = a·I는 색 축에만 작용한다. 계산 전에 적은 판본과 kill:

(가) SU(3)×SU(2) 단일항 줄(색 점유 k ∈ {0, 3}, 약 점유 j ∈ {0, 2})과 전하 Q = Y(T3 = 0).
(나) 단일항 줄의 R 무게 a^k. 주장: 자명하지 않은 무게는 a³ 하나다 → 가둔 기록의 무게는 고를 여지 없이 α_s = a³.
   kill: 자명하지 않은 단일항 무게가 a³ 외에 더 있으면 “고를 여지 없음” 기각.
(다) 범위: 같은 논리를 SU(2) 단일항에 쓰면(R을 약 축에 두면) 무게 a² = α₂? 약력은 가두지 않으므로 틀려야 일관(§43.70의 ×7).
목격(읽기로만): 중성 단일항 Λ⁰(무게 1)과 Λ⁵(무게 a³)가 경주의 자유(진공)와 붙잡힘(암흑물질)에 짝지어진다.

python -B -m examples.physics.rendering.ce_rendering_singlets
"""

from __future__ import annotations

import itertools
from fractions import Fraction

from examples.physics.rendering import ce_rendering_registry as R

COLOR, WEAK = (0, 1, 2), (3, 4)
Y_AXIS = {0: Fraction(-1, 3), 1: Fraction(-1, 3), 2: Fraction(-1, 3), 3: Fraction(1, 2), 4: Fraction(1, 2)}


def states() -> list[dict]:
    out = []
    for n in range(6):
        for s in itertools.combinations(range(5), n):
            k = sum(1 for i in s if i in COLOR)
            j = sum(1 for i in s if i in WEAK)
            y = sum((Y_AXIS[i] for i in s), Fraction(0))
            out.append({"axes": s, "k": k, "j": j, "Y": y})
    return out


def singlets() -> list[dict]:
    """색 단일항 k ∈ {0,3}, 약 단일항 j ∈ {0,2}(외대수의 불변 줄)."""
    return [s for s in states() if s["k"] in (0, 3) and s["j"] in (0, 2)]


def weights() -> dict:
    a = R.calibrated_alpha_s()[0] ** (1 / 3)
    rows = {}
    for s in singlets():
        name = {(0, 0): "L0 (empty)", (3, 0): "L3 V3 (colour full)", (0, 2): "L2 V2 (weak full)",
                (3, 2): "L5 (all full)"}[(s["k"], s["j"])]
        rows[name] = {"Q": str(s["Y"]), "weight_power": s["k"], "weight": a ** s["k"]}
    nontrivial = sorted({v["weight_power"] for v in rows.values() if v["weight_power"] > 0})
    return {"rows": rows, "nontrivial_powers": nontrivial, "forced_alpha_s": nontrivial == [3],
            "alpha_s": R.calibrated_alpha_s()[0], "a_cubed": a ** 3}


def scope_su2() -> dict:
    a = R.calibrated_alpha_s()[0] ** (1 / 3)
    alpha2 = (1 / 127.951) / R.SZ2
    return {"weight_if_R_on_V2": a ** 2, "alpha2_obs": alpha2, "ratio": a ** 2 / alpha2}


def neutral_pairing() -> dict:
    """목격 후 읽기: 중성 단일항의 무게 비 = 경주의 붙잡힘:자유(통로 하나당)."""
    w = weights()["rows"]
    return {"neutral_singlets": [k for k, v in w.items() if v["Q"] == "0"],
            "capture_over_free_per_channel": w["L5 (all full)"]["weight"] / w["L0 (empty)"]["weight"],
            "alpha_s": R.calibrated_alpha_s()[0]}


def main() -> None:
    print("states:", len(states()), "singlets:", len(singlets()))
    w = weights()
    for k, v in w["rows"].items():
        print(f"  {k:22s} Q={v['Q']:>3s} weight=a^{v['weight_power']} = {v['weight']:.6f}")
    print("nontrivial powers:", w["nontrivial_powers"], "forced:", w["forced_alpha_s"])
    print("scope SU(2):", {k: round(v, 4) for k, v in scope_su2().items()})
    print("neutral pairing:", neutral_pairing())


if __name__ == "__main__":
    main()
