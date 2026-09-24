"""최종 지평선의 O(1) 상수 — 자연 기제는 주지 않고, 단순 상수는 가려낼 수 없다. 원장: 43장 §43.98. 예측값을 바꾸지 않는다.

§43.97의 원리 PL(원 없는 임계 잠김)은 선도항 (π²/2) N_e를 주지만 최종 지평선의 결손은 0이다. 필요한 상수는
C_final = ln S_dS − (π²/2) N_e = −0.1799 ± 0.0026(§43.94)이다. 오늘 지평선은 ln S_H0 = ln S_dS + ln Ω_Λ(+R-Pl)의
운동학 항등식으로 따라 나오므로, 원리가 설명할 상수는 최종 지평선 하나다(TT: 물리적 지평선 = 최종 지평선).

계산 전에 적은 규칙. 사전 목격: −δ(+0.8σ)와 −(3/2)α_s(+1.1σ)가 창 근처에 있다는 것을 암산으로 먼저 보았다.
그래서 (다)의 적중 목록은 증거로 세지 않고, 판정은 적중 수와 우연 기대 수로만 한다.

(나) PL 안의 자연 기제(맞출 수 있는 수 없음). 2σ 안이면 적중.
    N0 기록 탄생이 연속(기대 개수 N_e): 0
    N1 탄생이 e-fold 1, …, ⌊N_e⌋: 3ζ(2)(⌊N_e⌋ − N_e)
    N2 탄생이 e-fold 0, …, ⌊N_e⌋: 3ζ(2)(⌊N_e⌋ + 1 − N_e)
    N3 축마다 첫 기록이 기록 틀(F = 2)에서 태어남: 첫 항 1씩 빠짐, −3
    N4 “나” 축(마지막 렌더링, §43.43)이 한 e-fold 늦게 시작: −ζ(2)
    N5 나/아닌 나 대칭 인자 2로 나눔: −ln 2
    N6 세 축 순열 대칭 3!로 나눔: −ln 6
(다) 식별 가능성. 단순 상수 가족: 기본량 {δ, q, α_s, ŝ², α_sD, ln F, −ln(1−q)}의 하나 또는 두 개의 곱에
    유리수 {1, 2, 3, 4, 6, 8, 1/2, 1/3, 1/4, 1/6, 1/8, 2/3, 3/2, 3/4, 4/3}와 π 인자 {1, π, 1/π, π/2, 2/π, 2π, 1/2π, π², 1/π²}를
    곱한 값(순수 유리수 × π 인자 포함) 가운데 크기 0.01–3인 것. |C_final|의 2σ 창에 드는 수와, 창 주변(±0.05)의 밀도로 잰
    우연 기대 수를 낸다. 기대 수가 1 이상이면 “가려낼 수 없음”. 적중 가운데 가장 가까운 두 값을 3σ로 가르는 데 필요한 σ도 낸다.

python -B -m examples.physics.rendering.ce_rendering_horizon_constant
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from examples.physics.rendering import ce_rendering_horizon_towers as HTW
from examples.physics.rendering import ce_rendering_registry as R

PI = math.pi
ZETA2 = PI ** 2 / 6.0
AXES = 3
RATIONALS = [Fraction(1), Fraction(2), Fraction(3), Fraction(4), Fraction(6), Fraction(8), Fraction(1, 2), Fraction(1, 3),
             Fraction(1, 4), Fraction(1, 6), Fraction(1, 8), Fraction(2, 3), Fraction(3, 2), Fraction(3, 4), Fraction(4, 3)]
PI_FACTORS = {"1": 1.0, "π": PI, "1/π": 1 / PI, "π/2": PI / 2, "2/π": 2 / PI, "2π": 2 * PI, "1/2π": 1 / (2 * PI),
              "π²": PI ** 2, "1/π²": 1 / PI ** 2}
RANGE = (0.01, 3.0)
NEIGHBOURHOOD = 0.05


def _core() -> dict:
    return R.core(R.calibrated_alpha_s()[0])


def primitives() -> dict:
    c = _core()
    return {"δ": c["d"], "q": c["q"], "α_s": c["a"], "ŝ²": c["s2"], "α_sD": c["a"] * c["D"],
            "ln F": math.log(c["F"]), "−ln(1−q)": -math.log(1.0 - c["q"])}


# ------------------------------------------------------------------ (나) 자연 기제
def natural_mechanisms() -> dict:
    t = HTW.target()
    ne = t["N_e"]
    cands = {
        "N0 연속 탄생": 0.0,
        "N1 탄생 1…⌊N_e⌋": AXES * ZETA2 * (math.floor(ne) - ne),
        "N2 탄생 0…⌊N_e⌋": AXES * ZETA2 * (math.floor(ne) + 1 - ne),
        "N3 첫 기록 기록 틀에서": -3.0,
        "N4 나 축 한 e-fold 늦음": -ZETA2,
        "N5 나/아닌 나 대칭 2": -math.log(2.0),
        "N6 축 순열 3!": -math.log(6.0),
    }
    out = {k: {"value": v, "pull": (v - t["C_final"]) / t["sigma_S"]} for k, v in cands.items()}
    return {"target": t["C_final"], "sigma": t["sigma_S"], "members": out,
            "hits": [k for k, d in out.items() if abs(d["pull"]) <= 2.0]}


# ------------------------------------------------------------------ (다) 단순 상수 가족
def constant_family() -> list[tuple[str, float]]:
    prim = primitives()
    members = []
    for r in RATIONALS:
        for pn, pv in PI_FACTORS.items():
            base = float(r) * pv
            members.append((f"{r}·{pn}", base))
            for xn, xv in prim.items():
                members.append((f"{r}·{pn}·{xn}", base * xv))
            for (xn, xv), (yn, yv) in itertools.combinations_with_replacement(prim.items(), 2):
                members.append((f"{r}·{pn}·{xn}·{yn}", base * xv * yv))
    seen, out = set(), []
    for name, v in members:
        key = round(v, 12)
        if RANGE[0] <= v <= RANGE[1] and key not in seen:
            seen.add(key)
            out.append((name, v))
    return out


def identifiability() -> dict:
    t = HTW.target()
    target, sig = -t["C_final"], t["sigma_S"]
    fam = constant_family()
    window = 2.0 * sig
    hits = sorted(((n, v, (v - target) / sig) for n, v in fam if abs(v - target) <= window), key=lambda x: abs(x[2]))
    local = sum(1 for _, v in fam if abs(v - target) <= NEIGHBOURHOOD)
    expected = local / (2.0 * NEIGHBOURHOOD) * (2.0 * window)
    values = sorted({round(v, 12) for _, v, _ in hits})
    gaps = [b - a for a, b in zip(values, values[1:])]
    min_gap = min(gaps) if gaps else None
    return {"members": len(fam), "window": window, "hits": hits, "n_hits": len(hits), "local_density_count": local,
            "expected_chance_hits": expected, "unidentifiable": expected >= 1.0,
            "min_gap": min_gap, "sigma_needed_3sigma": (min_gap / 3.0) if min_gap else None,
            "improvement_needed": (sig / (min_gap / 3.0)) if min_gap else None}


def main() -> None:
    n = natural_mechanisms()
    print("(나) target C_final =", round(n["target"], 4), "σ =", round(n["sigma"], 4))
    for k, d in n["members"].items():
        print(f"     {k:22s} {d['value']:+.4f}  pull {d['pull']:+9.1f}")
    print("     hits:", n["hits"])
    i = identifiability()
    print("(다) members", i["members"], "| hits", i["n_hits"], "| expected chance hits", round(i["expected_chance_hits"], 2),
          "| unidentifiable", i["unidentifiable"])
    for name, v, p in i["hits"][:12]:
        print(f"     {name:28s} {v:.5f}  pull {p:+.2f}")
    print("     min gap", i["min_gap"], "| σ needed", i["sigma_needed_3sigma"], "| improvement ×", i["improvement_needed"])


if __name__ == "__main__":
    main()
