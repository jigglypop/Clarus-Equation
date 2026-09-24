"""식 구조의 유도·경험 분리 — 설명 길이의 부담이 어디에 몰리는가. 원장: 43장 §43.91. 예측값을 바꾸지 않는다.

§43.87 (다)는 CE 식 32개의 기호 비용을 모두 세어 손익분기를 기호당 약 1 bit로 얻었다. 그러나 일부 식은 구조가 공리에서
유도되었다. 계산 전에 적은 분류와 규칙:

유도(구조가 적힌 공리·정리에서 정해짐, 기호 비용 0, 공리의 비용은 세지 않음 — CE에 가장 관대):
  E4(§43.88: C6 + E1′ + D + CC), δ = ŝ²(1−ŝ²)(분별량), D = 3 + δ(BR1), F(경주 RC + MR, §43.67), U1 삼각형(Bool 분할, §43.44),
  s13²·s12²·s23²(Bool·Ind·SA, §43.43), T1(공리 B, §43.43), q(Galton–Watson + BR1, §43.54), Ω_m 분할(경주, §43.67),
  O1 기울기(TT + H1, §43.89), ω_b·ω_c(정의), W2의 꼴(정리 B), G1m(정리 E)
경험(조립만 있고 유도 없음, 기호 비용을 진다):
  u, ε, w와 |V_us|·|V_cb|·m_μ/m_τ 조립(§43.5 ②), Koide 2/3, α_em 합규칙, v/M_Pl 계층 식, N_e = 18D, H₀ 지평선 식(S와 H),
  n_s, A_s, dn_s/dlnk, W2의 진폭 ξ² = α_s^{2/3}
보고: 블록별 경험 기호 수, 경험 기호만 비용을 질 때의 손익분기 b*_emp, 식별 기호 수 순위(유도가 가장 필요한 식).
판정 규칙은 없다. b*_emp가 G2의 기호당 평균 비용보다 크면 “유도된 부분을 인정하면 우위가 개방 탐색의 잎 부호까지 견딘다”고 적는다.

python -B -m examples.physics.rendering.ce_rendering_formula_status
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_audit as AUD

DERIVED = {
    "E4: ŝ² = 4 α_s^{4/3}", "δ = ŝ²(1−ŝ²)", "D = 3 + δ", "F = 1 + α_s D (M_H/M_Z)", "U1: (β, α) = (π/8, π/2)",
    "s13² = δ/8", "s12² = (1 − 2δ/8)/3", "s23² = (1 − 4δ/8)/2", "T1: |U_μ1| = |U_τ1|",
    "q = e^{−D(1−q)}", "Ω_m = q + (1−q) α_s D/F", "O1: 직접 판독 = H/cos(π/8)", "ω_b = q h²", "ω_c = (Ω_m − q) h²",
    "G1m: 1 − Ω_m(1 − cos ψ/2)",
}
W2_KEY = "W2: 1 + α_s^{2/3} cos(H_Λ t/2)"
W2_AMPLITUDE_TOKENS = "^ a / 2 3".split()          # 진폭 ξ² = α_s^{2/3}만 경험


def split() -> dict:
    out = {}
    for blk, table in AUD.FORMULAS.items():
        emp, der, per = 0, 0, []
        for name, f in table.items():
            n = len(f.split())
            if name == W2_KEY:
                e = len(W2_AMPLITUDE_TOKENS)
                emp += e
                der += n - e
                per.append((name + " [진폭]", e))
            elif name in DERIVED:
                der += n
            else:
                emp += n
                per.append((name, n))
        out[blk] = {"empirical_tokens": emp, "derived_tokens": der, "empirical_formulas": sorted(per, key=lambda x: -x[1])}
    return out


def break_even(minimal_prior: bool = False) -> dict:
    g = AUD.grammar_mdl(minimal_prior)
    s = split()
    g2_per_leaf = math.log2(AUD.token_counts()["leaf_kinds"])
    out = {}
    for blk in ("Q", "M"):
        v = g[blk]
        fixed = v["L_base"] - v["break_even_bits_per_token"] * v["tokens"]      # 매개변수 + 부호 + χ² 몫
        n_emp = s[blk]["empirical_tokens"]
        leaf_share = v["leaves"] / v["tokens"]
        out[blk] = {"b_star_all": v["break_even_bits_per_token"], "b_star_empirical": (v["L_base"] - fixed) / n_emp,
                    "empirical_tokens": n_emp, "g2_bits_per_token": leaf_share * g2_per_leaf}
    return out


CORE_FOUR = {"α_em⁻¹: 합규칙 + 한 통로 고리", "v/M_Pl = e^{−12D}(1+α_s/4π)/F", "A_s: Q_s",
             "A_s = Q_s² q/[(1−q)² 2π N_e²]", "S = (π²/2) N_e − π δ (1−q)", "H₀ t_Pl = √π e^{−S/2}(1+α_s/4π)"}


def if_core_four_derived(minimal_prior: bool = False) -> dict:
    """저장소 코어의 네 식(합규칙, 계층 식, A_s, H₀ 지평선)이 유도된다면 남는 경험 기호와 손익분기."""
    s = split()
    b = break_even(minimal_prior)
    out = {}
    for blk in ("Q", "M"):
        core = sum(n for name, n in s[blk]["empirical_formulas"] if name in CORE_FOUR)
        rest = s[blk]["empirical_tokens"] - core
        budget = b[blk]["b_star_empirical"] * s[blk]["empirical_tokens"]
        out[blk] = {"core_tokens": core, "core_share": core / s[blk]["empirical_tokens"], "remaining_tokens": rest,
                    "b_star_remaining": budget / rest, "g2_bits_per_token": b[blk]["g2_bits_per_token"]}
    return out


def main() -> None:
    s = split()
    for blk, d in s.items():
        print(f"[{blk}] empirical {d['empirical_tokens']} / derived {d['derived_tokens']} tokens")
        for name, n in d["empirical_formulas"]:
            print(f"     {n:3d}  {name}")
    for mp in (False, True):
        print(f"minimal_prior={mp}:", {b: {k: round(x, 2) for k, x in v.items()} for b, v in break_even(mp).items()})


if __name__ == "__main__":
    main()
