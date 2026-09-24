"""코어 네 식의 공략 — 유도할 수 있는가, 없으면 매개변수로 내린다. 원장: 43장 §43.93. 예측값을 바꾸지 않는다.

코어 네 식: α_em 합규칙, 계층 식 v/M_Pl, A_s, H₀ 지평선 식. §43.91에서 설명 길이 부담의 54–74%가 이 넷에 있었다.
계산 전에 적은 규칙. 사전 목격: 계층 식이 “12 통로가 모두 조용할 포아송 확률”로 읽힌다는 것, 그리고 (다)의 손익분기
(기호당 약 3.5 bit)를 암산으로 먼저 보았다. 그래서 (나)-계층의 유일성과 (다)의 판정은 보고로만 쓴다.

(가) 강제 부분(표준 항등식)을 떼어 낸 잔여 내용.
  - 지평선 식: H₀t_Pl = √π e^{−S/2}는 S = ln S_H(오늘의 허블 지평선 넓이 엔트로피, HP §43.71)의 정의 S_H = π/(H₀t_Pl)²와 같다.
    잔여: ln S_H = (π²/2) N_e − π δ (1−q), R-Pl 고리.
  - 합규칙: α_em = ŝ² α₂는 MS-bar의 정의 ê = ĝ ŝ다. 잔여: α_s + α₂ + α_em = (1/2π)(1 + α_s/16π).
  - 계층 식, A_s: 강제 부분 없음.
(나) 유도 판정 기준: 기존 공리·정리의 조합만으로 잔여 내용이 나오고 새 선택이 없으면 “유도”, 새 선택이 하나라도 있으면
    “읽기”, 조합이 잔여 내용을 주지 못하면 “실패”.
  - 계층 SIL: v/M_Pl = P(12 게이지 생성자 통로가 모두 조용) × P(자기 먼저) × ODD 고리 = e^{−12D}(1/F)(1 + α_s/4π).
    부품 BR1(통로마다 포아송 D), N_gauge = 12(§43.23), 경주(§43.67), R-Pl. 새 선택은 “조용할 확률이 척도의 비”라는 사상.
    유일성: k ∈ {1..24} × 경주 인자 {1, 1/F, F} × 고리 {1, ODD, V} 216개에서 1σ 적중 수와 우연 확률.
  - 교차 검사: 같은 SIL이 지평선 계층 ln(1/H₀t_Pl)(Planck 67.36 ± 0.54)을 구조 정수 k ∈ STRUCTURAL로 1σ 안에 주면
    통합 성공, 아니면 실패.
  - 지평선 계수 π²/2: 후보 h1(e-fold마다 넓이 e² → 계수 2), h2(SIL, 계수 2k/18, k 구조 정수). 1% 안이면 적중.
  - 합규칙: s1(세 게이지 통로 기록 확률 2πα_i의 합 = 1, ⑤ 고리 포함 — 2π가 새 선택), s2(사전 D: Σ det(R|V_k) = a³ + a²).
    s2가 관측 Σα와 3σ 안이면 유도, 아니면 실패.
  - A_s: 후보 원리 없음. 느린 굴림 일관성(r = 12/N_e²와 함께 H_inf, Starobinsky 질량)만 보고한다.
(다) 설명 길이 최적 부호: 네 식마다 식 비용(G1 A=32, G2 잎)과 그 식이 대신하는 매개변수 비용(§43.47의 parameter_bits,
    20자릿수·좁은 사전)을 비교한다. 네 식을 매개변수로 내린 CE의 L과 손익분기 b*_after를 낸다. 내린 식의 행은 pull 0,
    H₀ 지평선 식을 내리면 h는 θ* 행으로 맞춘 매개변수(θ* pull 0, 다른 h 의존 행은 그대로 — 보수적).

python -B -m examples.physics.rendering.ce_rendering_core_four
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_audit as AUD
from examples.physics.rendering import ce_rendering_formula_status as FST
from examples.physics.rendering import ce_rendering_ledger as L
from examples.physics.rendering import ce_rendering_registry as R

PI = math.pi
STRUCTURAL = {1, 2, 3, 4, 5, 8, 9, 12, 16, 18, 24, 32, 36, 48}
H0_PLANCK = (67.36, 0.54)


def _core() -> dict:
    return R.core(R.calibrated_alpha_s()[0])


# ------------------------------------------------------------------ (가) 강제 부분
def horizon_identity() -> dict:
    c = _core()
    log_s = (PI ** 2 / 2.0) * c["Ne"] - PI * c["d"] * (1.0 - c["q"])
    via_formula = R.hubble_kms(c)
    via_entropy = math.sqrt(PI / math.exp(log_s)) / R.T_PLANCK_S * R.KM_PER_MPC
    return {"lnS_H": log_s, "rel_diff": abs(via_formula - via_entropy) / via_formula}


def ew_identity() -> dict:
    c = _core()
    inv = R.alpha_em_inv(c)
    total = (1.0 / (2.0 * PI)) * (1.0 + c["a"] / (16.0 * PI))
    alpha2 = (total - c["a"]) / (1.0 + c["s2"])
    return {"rel_diff": abs(1.0 / inv - c["s2"] * alpha2) * inv}


# ------------------------------------------------------------------ (나) 유도 시도
RACE = {"1": lambda c: 1.0, "1/F": lambda c: 1.0 / c["F"], "F": lambda c: c["F"]}
LOOP = {"1": lambda c: 1.0, "ODD": lambda c: 1.0 + c["a"] / (4.0 * PI), "V": lambda c: 1.0 + c["a"] / (16.0 * PI)}


def _ln_hierarchy(k: int, race: str, loop: str, a: float) -> float:
    c = R.core(a)
    return k * c["D"] - math.log(RACE[race](c)) - math.log(LOOP[loop](c))


def sil_scan(target_ln: float, target_sigma: float, k_range=range(1, 25)) -> dict:
    """ln(M_Pl/X) = kD − ln(경주) − ln(고리). σ는 목표 σ와 α_s 전파 이론 오차의 합."""
    a0, sa = R.calibrated_alpha_s()
    members, hits = [], []
    for k in k_range:
        for race in RACE:
            for loop in LOOP:
                v = _ln_hierarchy(k, race, loop, a0)
                st = abs(_ln_hierarchy(k, race, loop, a0 + sa) - _ln_hierarchy(k, race, loop, a0 - sa)) / 2
                s = math.hypot(target_sigma, st)
                members.append((v, s))
                if abs(v - target_ln) <= s:
                    hits.append({"k": k, "race": race, "loop": loop, "pull": (v - target_ln) / s})
    lo, hi = min(v for v, _ in members), max(v for v, _ in members)
    cover = sum(2 * s for _, s in members)                        # 겹침을 무시한 상한
    return {"hits": hits, "n_members": len(members), "chance_upper": min(1.0, cover / (hi - lo))}


def ew_sil() -> dict:
    target = math.log(R.M_PLANCK / R.V_EW)
    out = sil_scan(target, 1.1e-5)
    formula = -math.log(R.v_over_mpl(_core()))
    out.update({"target_ln": target, "formula_ln": formula,
                "formula_is_sil": abs(formula - _ln_hierarchy(12, "1/F", "ODD", R.calibrated_alpha_s()[0])) < 1e-12})
    return out


def horizon_sil_crosscheck() -> dict:
    h0, sh = H0_PLANCK
    target = -math.log(h0 / R.KM_PER_MPC * R.T_PLANCK_S)
    scan = sil_scan(target, sh / h0, range(1, 61))
    structural_hits = [h for h in scan["hits"] if h["k"] in STRUCTURAL]
    return {"target_ln": target, "k_star": target / _core()["D"], "hits": scan["hits"],
            "structural_hits": structural_hits, "unified": bool(structural_hits)}


def horizon_coefficient() -> dict:
    want = PI ** 2 / 2.0
    cands = {"h1 (e^2 per e-fold)": 2.0}
    cands.update({f"h2 SIL k={k}": 2.0 * k / 18.0 for k in sorted(STRUCTURAL)})
    hits = [n for n, v in cands.items() if abs(v / want - 1.0) < 0.01]
    return {"want": want, "nearest": min(cands.items(), key=lambda kv: abs(kv[1] - want)), "hits": hits}


def sum_rule() -> dict:
    c = _core()
    alpha_em = 1.0 / R.AEM_INV_MZ
    alpha2 = alpha_em / R.SZ2
    total_obs = R.A_WORLD + alpha2 + alpha_em
    total_err = math.hypot(R.A_WORLD_ERR, alpha2 * 0.0002)
    a = (R.SZ2 / 4.0) ** 0.25
    s2_value = a ** 3 + a ** 2
    return {"sum_obs": total_obs, "s1_value": (1.0 / (2 * PI)) * (1 + c["a"] / (16 * PI)),
            "s2_value": s2_value, "s2_pull": (s2_value - total_obs) / total_err, "s2_derived": abs(s2_value - total_obs) < 3 * total_err}


def a_s_slow_roll() -> dict:
    c = _core()
    a_s = R.scalar_amplitude(c) * 1e-9
    r = 12.0 / c["Ne"] ** 2
    m_red = 2.435e18
    h_inf = PI * m_red * math.sqrt(r * a_s / 2.0)
    m_star = PI * math.sqrt(24.0 * a_s) / c["Ne"] * m_red
    return {"A_s": a_s, "r": r, "H_inf_GeV": h_inf, "starobinsky_M_GeV": m_star, "principle": None}


# ------------------------------------------------------------------ (다) 설명 길이 최적 부호
DEMOTE = {  # 식 이름(§43.87 FORMULAS) → 대신하는 행
    "α_em⁻¹: 합규칙 + 한 통로 고리": "alpha_em^-1(M_Z)",
    "v/M_Pl = e^{−12D}(1+α_s/4π)/F": "v/M_Pl",
    "A_s": "A_s x1e9",
    "H₀ 지평선": "100 theta*",
}
FORMULA_GROUPS = {
    "α_em⁻¹: 합규칙 + 한 통로 고리": ["α_em⁻¹: 합규칙 + 한 통로 고리"],
    "v/M_Pl = e^{−12D}(1+α_s/4π)/F": ["v/M_Pl = e^{−12D}(1+α_s/4π)/F"],
    "A_s": ["A_s: Q_s", "A_s = Q_s² q/[(1−q)² 2π N_e²]"],
    "H₀ 지평선": ["S = (π²/2) N_e − π δ (1−q)", "H₀ t_Pl = √π e^{−S/2}(1+α_s/4π)"],
}
BLOCK = {"α_em⁻¹: 합규칙 + 한 통로 고리": "Q", "v/M_Pl = e^{−12D}(1+α_s/4π)/F": "Q", "A_s": "M", "H₀ 지평선": "M"}


def _tokens(names: list[str]) -> tuple[int, int]:
    toks = [t for n in names for blk in AUD.FORMULAS.values() for k, f in blk.items() if k == n for t in f.split()]
    return len(toks), sum(1 for t in toks if t not in AUD.OPERATORS)


def per_formula(minimal_prior: bool = False) -> dict:
    rows = {o["key"]: o for o in L.ce_rows("IV")[0]}
    g2_leaf = math.log2(AUD.token_counts()["leaf_kinds"])
    out = {}
    for name, row in DEMOTE.items():
        n, nl = _tokens(FORMULA_GROUPS[name])
        o = rows[row]
        pb = L.parameter_bits(row, o["obs"], o["sigma"], minimal_prior)
        out[name] = {"row": row, "tokens": n, "G1_bits": n * math.log2(AUD.MAIN_ALPHABET), "G2_bits": nl * g2_leaf,
                     "param_bits": pb, "pull": o["pull"], "cheaper": "parameter" if pb < nl * g2_leaf else "formula"}
    return out


def demoted_mdl(minimal_prior: bool = False) -> dict:
    g = AUD.grammar_mdl(minimal_prior)
    split = FST.split()
    pf = per_formula(minimal_prior)
    g2_leaf = math.log2(AUD.token_counts()["leaf_kinds"])
    out = {}
    for blk in ("Q", "M"):
        v = g[blk]
        fixed = v["L_base"] - v["break_even_bits_per_token"] * v["tokens"]
        mine = {k: d for k, d in pf.items() if BLOCK[k] == blk}
        chi_drop = sum(d["pull"] ** 2 for d in mine.values()) / L.NATS_PER_BIT_CHI2
        params = sum(d["param_bits"] for d in mine.values())
        core_tokens = sum(d["tokens"] for d in mine.values())
        remaining = split[blk]["empirical_tokens"] - core_tokens
        rem_leaves = sum(1 for name, _ in split[blk]["empirical_formulas"]
                         for f in [next((ff for t in AUD.FORMULAS.values() for kk, ff in t.items()
                                         if kk == name.replace(" [진폭]", "")), "")]
                         for t in (f.split() if not name.endswith("[진폭]") else FST.W2_AMPLITUDE_TOKENS)
                         if t not in AUD.OPERATORS
                         and not any(name == gname for grp in FORMULA_GROUPS.values() for gname in grp))
        fixed_after = fixed - chi_drop + params
        out[blk] = {"L_base": v["L_base"], "params_added": params, "remaining_tokens": remaining,
                    "b_star_after": (v["L_base"] - fixed_after) / remaining,
                    "L_CE_after_G2": fixed_after + rem_leaves * g2_leaf,
                    "L_CE_after_G1": fixed_after + remaining * math.log2(AUD.MAIN_ALPHABET)}
    return out


def main() -> None:
    print("(가) horizon identity:", {k: f"{v:.3g}" for k, v in horizon_identity().items()},
          "| EW identity:", {k: f"{v:.1e}" for k, v in ew_identity().items()})
    e = ew_sil()
    print("(나) EW SIL: formula is SIL:", e["formula_is_sil"], "| hits:", e["hits"], "| members", e["n_members"],
          "| chance upper", f"{e['chance_upper']:.2e}")
    h = horizon_sil_crosscheck()
    print("    horizon cross-check: target", round(h["target_ln"], 4), "k*", round(h["k_star"], 3),
          "| hits", h["hits"], "| unified", h["unified"])
    print("    horizon coefficient:", horizon_coefficient())
    print("    sum rule:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in sum_rule().items()})
    print("    A_s slow roll:", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in a_s_slow_roll().items()})
    for mp in (False, True):
        print(f"(다) minimal_prior={mp}")
        for name, d in per_formula(mp).items():
            print(f"     {name:34s} tokens {d['tokens']:2d} G1 {d['G1_bits']:6.1f} G2 {d['G2_bits']:6.1f} "
                  f"param {d['param_bits']:5.1f} -> {d['cheaper']}")
        print("     demoted:", {b: {k: round(x, 2) for k, x in v.items()} for b, v in demoted_mdl(mp).items()})


if __name__ == "__main__":
    main()
