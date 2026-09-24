"""지평선 계수 π²/2의 원리 — 세 축의 열적 기록 탑. 원장: 43장 §43.94. 예측값을 바꾸지 않는다.

H₀ 지평선 식 ln S_H = (π²/2) N_e − π δ (1−q)(§43.93 (가)에서 √π e^{−S/2}는 정의로 떼어 냄)의 선도 계수 π²/2에 원리를 세운다.
계산 전에 적은 규칙. 사전 목격: 탑 가족의 O(1) 상수(NS 0, R −(3/2)ln2, 보손 −(3/2)ln(2πN_e))와 과녁 상수의 대략값
(오늘 −0.53, 최종 지평선 −0.18)을 암산으로 먼저 보았다. 그래서 (나)의 판정은 보고로만 쓰고 증거로 세지 않는다.

(가) 과녁. θ*에 맞춘 h(DV.h_from_theta, CE의 Ω 비율 고정)에서 필요한 S와 그 σ를 구한다. σ는 θ*의 σ(100θ*)를 h로 옮긴 몫과
    α_s 전파 몫의 합이다. TT(§43.53)가 물리적 지평선을 최종 지평선으로 두므로 주 과녁은 최종 지평선 ln S_dS = ln π − 2 ln(H_Λ t_Pl),
    H_Λ = H₀√Ω_Λ다. 오늘의 허블 지평선은 보고한다. 상수 C = (필요한 ln S) − (π²/2) N_e.
(나) 원리 PH(열적 기록 탑):
    PH1 렌더링된 공간 축마다 기록 모드 탑이 하나씩 있다(n = 1, 2, …, 간격 ω).
    PH2 기록 하나는 Bool 비트(점유 0/1)이므로 페르미 통계다.
    PH3 기록은 열적 원(TT) 위에 놓이므로 페르미온은 반주기(NS)다. 열적 장론의 경계 조건이라 선택이 아니다.
    PH4 기록은 실수다(한 비트는 제 거울, §43.44). 축마다 키랄 마요라나 탑 하나다.
    PH5 T/ω = N_e(지평선은 급팽창 기록 전체를 담는다). 새 전제다.
    주장: ln(화소 수) = 탑의 열적 엔트로피 = 3 × π²/(6x), x = 1/N_e → (π²/2) N_e(지수적으로 작은 항 빼고 정확).
    가족: 통계 {보손(주기), NS, R} × 양 {ln Z, S} × {실수, 복소} × 탑 수 {1, 2, 3, 6}. 값은 x = 1/N_e에서 직접 합으로 계산.
    판정: (L) 선도 계수가 정확히 π²/2인가, (C) O(1) 상수가 과녁 C와 2σ 안인가. (L)과 (C)를 모두 통과하는 원소가 있으면
    “유도(상수 포함)”, (L)만이면 “선도 계수의 읽기”, 없으면 “실패”.
(다) 경쟁 읽기: §43.11의 (2π)²/8(8통로)도 π²/2를 정확히 준다. 계수 하나에 읽기가 여럿이면 증거가 약하다는 것을 적는다.
(라) 설명 길이의 결과: PH를 공짜 공리로 인정해도(가장 관대) 남는 기호(보정항 7, R-Pl 7, 연산자 1)의 잎 부호 비용과
    h 매개변수 비용(§43.93)을 비교한다. 매개변수가 더 싸면 지평선 식은 내린 상태로 남는다.

(마) 목격 후 부산물(증거 아님): (나)의 과녁을 암산하다가 최종 지평선 상수 C_final ≈ −0.18이 −δ와 가깝다는 것을 보았다.
    그 뒤에 적은 규칙: CE 기본량으로 된 단순 O(1) 상수 가족 CONSTANTS(부호 ±, 0 포함)를 두 과녁(오늘·최종)에 대어 2σ 적중과
    우연 확률(과녁을 가족 범위에서 균등하게 뽑을 때 하나 이상 맞을 확률)을 보고한다. 적중한 새 판본 HT-δ(ln S_dS = (π²/2)N_e − δ,
    R-Pl 없음)는 채택하지 않고 경쟁 판본으로 두며, 기존 식과 갈리는 h와 필요한 정밀도를 적는다.

python -B -m examples.physics.rendering.ce_rendering_horizon_towers
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_audit as AUD
from examples.physics.rendering import ce_rendering_core_four as CF4
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_registry as R

PI = math.pi
LEAD = PI ** 2 / 2.0
TOWERS = (1, 2, 3, 6)
NMAX = 20000


def _core(a: float | None = None) -> dict:
    return R.core(R.calibrated_alpha_s()[0] if a is None else a)


def ce_S(a: float | None = None) -> float:
    c = _core(a)
    return (PI ** 2 / 2.0) * c["Ne"] - PI * c["d"] * (1.0 - c["q"])


# ------------------------------------------------------------------ (가) 과녁
def target() -> dict:
    a0, sa = R.calibrated_alpha_s()
    c = _core()
    loop = 1.0 + c["a"] / (4.0 * PI)
    h_t = DV.h_from_theta(a0)
    t_h = lambda h: DV.theta_star_100(c["q"] * h * h, (c["Om"] - c["q"]) * h * h, h)
    dth = (t_h(h_t * (1 + 1e-5)) - t_h(h_t * (1 - 1e-5))) / (2e-5 * h_t)
    sig_h = DV.THETA_SIGMA / abs(dth)
    ht_pl = 100.0 * h_t / R.KM_PER_MPC * R.T_PLANCK_S
    s_need = math.log(PI) + 2.0 * math.log(loop) - 2.0 * math.log(ht_pl)       # H₀ = √π e^{−S/2}(1+α_s/4π)/t_Pl
    om_l = 1.0 - c["Om"]
    lnS_dS_need = s_need - 2.0 * math.log(loop) - math.log(om_l)
    lead = LEAD * c["Ne"]
    s_theory = abs(ce_S(a0 + sa) - ce_S(a0 - sa)) / 2.0
    sigma = math.hypot(2.0 * sig_h / h_t, s_theory)
    return {"h_theta": h_t, "S_needed": s_need, "S_CE": ce_S(), "CE_minus_needed_sigma": (ce_S() - s_need) / sigma,
            "lead": lead, "C_today": s_need - lead, "C_final": lnS_dS_need - lead, "C_CE_today": -PI * c["d"] * (1 - c["q"]),
            "sigma_S": sigma, "sigma_theta_part": 2.0 * sig_h / h_t, "sigma_alpha_part": s_theory, "N_e": c["Ne"]}


# ------------------------------------------------------------------ (나) 탑 가족
def tower(stat: str, x: float, nmax: int = NMAX) -> tuple[float, float]:
    """키랄 실수 탑 하나의 (ln Z, βE). 직접 합."""
    n = np.arange(1, nmax + 1, dtype=float)
    e = (n - 0.5) * x if stat == "NS" else n * x
    w = np.exp(-e)                                   # 큰 e에서 넘치지 않도록 e^{−e}로 쓴다
    if stat == "B":
        return float(-np.sum(np.log1p(-w))), float(np.sum(e * w / -np.expm1(-e)))
    return float(np.sum(np.log1p(w))), float(np.sum(e * w / (1.0 + w)))


PER_TOWER_COEF = {("B", "lnZ"): PI ** 2 / 6, ("B", "S"): PI ** 2 / 3, ("NS", "lnZ"): PI ** 2 / 12,
                  ("NS", "S"): PI ** 2 / 6, ("R", "lnZ"): PI ** 2 / 12, ("R", "S"): PI ** 2 / 6}


def family() -> list[dict]:
    t = target()
    x = 1.0 / t["N_e"]
    out = []
    for stat in ("B", "NS", "R"):
        lnz, be = tower(stat, x)
        for qty, v1 in (("lnZ", lnz), ("S", lnz + be)):
            for real, mult in (("real", 1), ("complex", 2)):
                for k in TOWERS:
                    coef = PER_TOWER_COEF[(stat, qty)] * mult * k
                    value = v1 * mult * k
                    lead_ok = abs(coef - LEAD) < 1e-12
                    const = value - coef * t["N_e"]
                    out.append({"stat": stat, "qty": qty, "real": real, "towers": k, "coef": coef, "lead_ok": lead_ok,
                                "const": const,
                                "pull_final": (const - t["C_final"]) / t["sigma_S"] if lead_ok else None,
                                "pull_today": (const - t["C_today"]) / t["sigma_S"] if lead_ok else None})
    return out


def verdict() -> dict:
    fam = family()
    lead = [m for m in fam if m["lead_ok"]]
    both_final = [m for m in lead if abs(m["pull_final"]) <= 2.0]
    both_today = [m for m in lead if abs(m["pull_today"]) <= 2.0]
    status = "유도(상수 포함)" if (both_final or both_today) else ("선도 계수의 읽기" if lead else "실패")
    return {"lead_members": len(lead), "pass_final": both_final, "pass_today": both_today, "status": status}


# ------------------------------------------------------------------ (다) 경쟁 읽기
def competing_readings() -> dict:
    return {"3 zeta(2) (towers)": 3 * PI ** 2 / 6, "(2 pi)^2 / 8 (8 channels, §43.11)": (2 * PI) ** 2 / 8, "target": LEAD}


# ------------------------------------------------------------------ (라) 설명 길이
RESIDUAL_WITH_PH_FREE = {"보정항 −πδ(1−q)": "* * pi d - 1 q", "R-Pl 고리": "+ 1 / a * 4 pi", "연산자": "-"}
RESIDUAL_HT_DELTA_WITH_PH_FREE = {"−δ": "- d"}


def mdl_with_ph_free(minimal_prior: bool = False, residual: dict | None = None) -> dict:
    g2_leaf = math.log2(AUD.token_counts()["leaf_kinds"])
    toks = [t for f in (residual or RESIDUAL_WITH_PH_FREE).values() for t in f.split()]
    leaves = sum(1 for t in toks if t not in AUD.OPERATORS)
    param = CF4.per_formula(minimal_prior)["H₀ 지평선"]["param_bits"]
    return {"residual_tokens": len(toks), "G2_bits": leaves * g2_leaf, "G1_bits": len(toks) * math.log2(AUD.MAIN_ALPHABET),
            "param_bits": param, "cheaper": "parameter" if param < leaves * g2_leaf else "formula"}


# ------------------------------------------------------------------ (마) 목격 후 부산물
def _constants(c: dict) -> dict:
    d, q, a, s2, F = c["d"], c["q"], c["a"], c["s2"], c["F"]
    base = {"δ": d, "δ/2": d / 2, "2δ": 2 * d, "δ(1−q)": d * (1 - q), "πδ": PI * d, "πδ(1−q)": PI * d * (1 - q),
            "q": q, "ln F": math.log(F), "−ln(1−q)": -math.log(1 - q), "½ln2": 0.5 * math.log(2), "ln2": math.log(2),
            "(3/2)ln2": 1.5 * math.log(2), "α_s": a, "ŝ²": s2, "1/8": 0.125, "π/8": PI / 8}
    out = {"0": 0.0}
    for k, v in base.items():
        out["+" + k], out["−" + k] = v, -v
    return out


def constant_family() -> dict:
    t = target()
    cons = _constants(_core())
    lo, hi = min(cons.values()), max(cons.values())
    w = 2.0 * t["sigma_S"]
    chance = min(1.0, len(cons) * 2 * w / (hi - lo))
    hits = {tg: [(k, round((v - t[tg]) / t["sigma_S"], 2)) for k, v in cons.items() if abs(v - t[tg]) <= w]
            for tg in ("C_today", "C_final")}
    return {"members": len(cons), "hits": hits, "chance_per_target_upper": chance}


def ht_delta_branch() -> dict:
    """HT-δ: ln S_dS = (π²/2)N_e − δ, H₀ = H_Λ/√Ω_Λ, R-Pl 없음. 기존 식(오늘, R-Pl)과 갈리는 h."""
    t = target()
    c = _core()
    ln_sds = LEAD * c["Ne"] - c["d"]
    hl_tpl = math.sqrt(PI / math.exp(ln_sds))
    h_cand = hl_tpl / math.sqrt(1.0 - c["Om"]) / R.T_PLANCK_S * R.KM_PER_MPC / 100.0
    h_ce = R.hubble_readout(c, False) / 100.0 * (1.0 + c["a"] / (4.0 * PI))
    s_cand_today = ln_sds + math.log(1.0 - c["Om"]) + 2.0 * math.log(1.0 + c["a"] / (4.0 * PI))
    return {"h_candidate": h_cand, "h_CE": h_ce, "h_theta": t["h_theta"], "rel_split": h_cand / h_ce - 1.0,
            "pull_vs_theta_needed": (s_cand_today - t["S_needed"]) / t["sigma_S"],
            "split_in_sigma_S": (s_cand_today - t["S_CE"]) / t["sigma_S"]}


def main() -> None:
    t = target()
    print("(가) target:", {k: round(v, 5) for k, v in t.items()})
    print("(나) members with lead π²/2:")
    for m in family():
        if m["lead_ok"]:
            print(f"     {m['stat']:2s} {m['qty']:3s} {m['real']:7s} x{m['towers']}  const {m['const']:+.4f}  "
                  f"pull final {m['pull_final']:+8.1f}  today {m['pull_today']:+8.1f}")
    print("    verdict:", verdict())
    print("(다)", competing_readings())
    for mp in (False, True):
        print(f"(라) minimal_prior={mp}:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in mdl_with_ph_free(mp).items()})
    print("(마) constant family:", constant_family())
    print("    HT-δ branch:", {k: round(v, 5) for k, v in ht_delta_branch().items()})
    for mp in (False, True):
        print(f"    HT-δ with PH free, minimal_prior={mp}:",
              {k: (round(v, 1) if isinstance(v, float) else v)
               for k, v in mdl_with_ph_free(mp, RESIDUAL_HT_DELTA_WITH_PH_FREE).items()})


if __name__ == "__main__":
    main()
