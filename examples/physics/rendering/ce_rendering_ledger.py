"""전체 가설 장부 재집계와 VAL-01 기준 모형 비교. 원장: 43장 §43.47. 예측값을 바꾸지 않는다.

계산 전에 적은 규칙:

장부. 사후에 세운 공리가 규칙에 이름만 붙이면 bit를 줄이지 않는다. 여러 규칙을 한 메뉴로 압축할 때만 줄인다.
   장부 밖에 있던 식은 감사한 것만 넣는다(중성미자 13.7 bit, E4 지수 사전 5.1 bit). 감사하지 않은 코어 식
   (q 방정식, F, N_e = 18D, A_s)은 “미감사”로 따로 적고 합계에 넣지 않는다.
기준. SM + ΛCDM(Planck 2018). 맛깔·결합 14행과 ΛCDM 5행은 자유 매개변수로 정확히 맞춘다(pull 0).
   m_W/m_Z, Δa_μ, dn_s/dlnk, 직접 판독 H₀ 3행, BAO 13성분은 예측으로 채점한다.
비교. VAL-01: 날 RMSE와 블록(Q, M). VAL-02: 점수 S = χ² + 2k + 2 ln2 · bit(작을수록 좋음).

python -B -m examples.physics.rendering.ce_rendering_ledger
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_open_checks as OC
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

NATS_PER_BIT_CHI2 = 2 * math.log(2)
PLANCK_OM = 0.3153
PLANCK_H0 = 67.36

# (항목, 이전 bit, 현재 bit, 근거, 적용 판본)
LEDGER = [
    ("|V_us| 조립(G1)", 1.0, 1.0, "레지스트리", "39"),
    ("|V_cb| 조립(TC)", 1.0, 1.0, "레지스트리", "39"),
    ("U1 분할 + 화살표", 3.0, 2.6, "Bool: 직각 강제, 분할 3가지 중 하나(1.6) + 화살표 1", "39"),
    ("m_mu/m_tau 조립", 1.0, 1.0, "레지스트리", "39"),
    ("Koide 2/3", 1.0, 1.0, "레지스트리", "39"),
    ("R-ch: ⑤ 합규칙 고리", 3.0, 1.6, "Bool 세 값 메뉴(1/8, 3/8, 1/2)에서 하나", "39"),
    ("R-ch: v/M_Pl 고리", 3.0, 1.6, "Bool 세 값 메뉴에서 하나", "39"),
    ("R-Pl 플랑크 판독", 1.6, 0.0, "v/M_Pl과 같은 사건(ODD)의 전이", "39"),
    ("옥탄트 부호(S2 → SA)", 1.0, 1.0, "사후 공리 SA가 대체(ν₂ 교차 확인 있음), 비용 유지", "39"),
    ("δ_PMNS(T1 + M1 → B + M1)", 2.0, 2.0, "사후 공리 B가 대체, 비용 유지", "39"),
    ("O1 직접 판독(4행 × 0.5)", 2.0, 2.0, "레지스트리; 방향은 자료가 정함(시계 해석 −12σ)", "39"),
    ("W2 주파수·부호", 1.6, 0.0, "C6 최저 조화 + NEC(정리 B)로 흡수", "39"),
    ("G1m → 정리 E 무게 시점", 2.0, 1.0, "1차 투영은 정리, 무게 시점(오늘) 1 bit", "39"),
    ("E4 지수 사전(4/3)", 0.0, 5.1, "장부 밖 → 편입(§43.3 사전의 정보량)", "39"),
    ("중성미자 질량식", 0.0, 13.7, "장부 밖 → 편입(§43.46)", "43"),
]
UNAUDITED = ["q = e^{-D(1-q)} → Ω_b", "F = 1 + α_s D", "N_e = 18D", "A_s 식", "합규칙 α_s + α_2 + α_em = 1/2π"]

FITTED_IN_BASELINE = {"delta_PMNS", "M_H/M_Z", "delta_CKM", "s23^2", "s12^2", "|V_ub|", "alpha_em^-1(M_Z)", "|V_cb|",
                      "m_mu/m_tau", "m_e/m_mu", "v/M_Pl", "alpha_s world", "s13^2", "|V_us|",
                      "omega_b h^2", "omega_c h^2", "100 theta*", "A_s x1e9", "n_s", "Delta m^2_21", "Delta m^2_31"}


def ledger_totals() -> dict:
    t = {"39": {"before": 0.0, "now": 0.0}, "43": {"before": 0.0, "now": 0.0}}
    for _, b, n, _, scope in LEDGER:
        for key in ("39", "43"):
            if scope == "39" or key == "43":
                t[key]["before"] += b
                t[key]["now"] += n
    return t


def planck_bao_chi2() -> float:
    wb, wc, h = DV.PLANCK_BEST
    b = VT.bao_vectors(PLANCK_OM, VT.density(None, None))
    r = GD.C_KM_S / (100 * h * BR.r_drag(wb, wc, h)) * b - R.BAO_Y
    return float(r @ R.BAO_CINV @ r)


def ce_rows(rows_from: str) -> tuple[list[dict], float]:
    base = VT.score(VT.ADOPTED_NU, rows_from, "fixed")
    return [dict(o) for o in base["rows"]], GD.bao_rows("G1m")["bao_chi2_fixed"]


def baseline_pulls(rows: list[dict]) -> list[dict]:
    """기준 모형의 행별 pull: 맞춘 행 0, 예측 행은 SM/ΛCDM 값."""
    out = []
    for o in rows:
        k = o["key"]
        if k in FITTED_IN_BASELINE:
            pull, fitted = 0.0, True
        elif k.startswith("H0 "):
            pull, fitted = (PLANCK_H0 - o["obs"]) / o["sigma"], False
        elif k == "dn_s/dlnk":
            pull, fitted = (0.0 - o["obs"]) / o["sigma"], False
        elif k.startswith("S8"):
            pull, fitted = (0.832 - o["obs"]) / o["sigma"], False       # Planck ΛCDM S8
        else:                                                              # m_W/m_Z(SM 식 동일), Δa_μ(SM = 0)
            pull, fitted = o["pull"], False
        out.append({"key": k, "block": o["block"], "pull": pull, "fitted": fitted})
    return out


def compare(rows_from: str = "IV") -> dict:
    rows, bao_ce = ce_rows(rows_from)
    label = "39" if rows_from == "IV" else "43"
    bits = ledger_totals()[label]["now"]
    base = baseline_pulls(rows)
    bao_b = planck_bao_chi2()
    n = len(rows) + 13

    def block(pulls, bao, blk):
        chi = sum(p ** 2 for p, o in zip(pulls, rows) if o["block"] == blk) + (bao if blk == "M" else 0.0)
        cnt = sum(1 for o in rows if o["block"] == blk) + (13 if blk == "M" else 0)
        return math.sqrt(chi / cnt)

    ce_p = [o["pull"] for o in rows]
    b_p = [o["pull"] for o in base]
    chi_ce = sum(p ** 2 for p in ce_p) + bao_ce
    chi_b = sum(p ** 2 for p in b_p) + bao_b
    k_b = sum(1 for o in base if o["fitted"])
    return {
        "N": n, "bits": bits,
        "CE": {"chi2": chi_ce, "rmse": math.sqrt(chi_ce / n), "Q": block(ce_p, bao_ce, "Q"), "M": block(ce_p, bao_ce, "M"),
               "k": 1, "S": chi_ce + 2 * 1 + NATS_PER_BIT_CHI2 * bits},
        "baseline": {"chi2": chi_b, "rmse": math.sqrt(chi_b / n), "Q": block(b_p, bao_b, "Q"), "M": block(b_p, bao_b, "M"),
                     "k": k_b, "S": chi_b + 2 * k_b, "bao_chi2": bao_b},
        "baseline_largest": sorted(((o["key"], round(o["pull"], 2)) for o in base if not o["fitted"]),
                                   key=lambda x: -abs(x[1]))[:4],
    }


M_ITEMS = {"R-Pl 플랑크 판독", "O1 직접 판독(4행 × 0.5)", "W2 주파수·부호", "G1m → 정리 E 무게 시점"}


def block_scores(rows_from: str = "IV") -> dict:
    """블록별 점수. AIC: 매개변수당 2, BIC: 매개변수당 ln N. bit는 둘 다 2 ln2. CE의 α_s는 Q 블록."""
    rows, bao_ce = ce_rows(rows_from)
    label = "39" if rows_from == "IV" else "43"
    base = baseline_pulls(rows)
    bao_b = planck_bao_chi2()
    n = len(rows) + 13
    bits = {"Q": 0.0, "M": 0.0}
    for item, _, now, _, scope in LEDGER:
        if scope == "39" or label == "43":
            bits["M" if item in M_ITEMS else "Q"] += now
    out = {}
    for blk in ("Q", "M"):
        chi_ce = sum(o["pull"] ** 2 for o in rows if o["block"] == blk) + (bao_ce if blk == "M" else 0.0)
        chi_b = sum(o["pull"] ** 2 for o in base if o["block"] == blk) + (bao_b if blk == "M" else 0.0)
        k_ce = 1 if blk == "Q" else 0
        k_b = sum(1 for o in base if o["block"] == blk and o["fitted"])
        desc = NATS_PER_BIT_CHI2 * bits[blk]
        out[blk] = {"bits": bits[blk], "k_base": k_b,
                    "AIC_CE": chi_ce + 2 * k_ce + desc, "AIC_base": chi_b + 2 * k_b,
                    "BIC_CE": chi_ce + math.log(n) * k_ce + desc, "BIC_base": chi_b + math.log(n) * k_b}
    return out


def flexible_baseline_m(rows_from: str = "IV") -> dict:
    """B2: 긴장을 자유 매개변수로 흡수하는 거시 기준. CMB 5행 적합 + BAO(Ω_m, 눈금 2) + 국소 H₀(1) (+ S8 1)."""
    from scipy.optimize import minimize_scalar

    rows, bao_ce = ce_rows(rows_from)
    n = len(rows) + 13
    cinv, y = R.BAO_CINV, R.BAO_Y

    def bao_free(om: float) -> float:
        b = VT.bao_vectors(om, VT.density(None, None))
        a = float(b @ cinv @ y) / float(b @ cinv @ b)
        r = a * b - y
        return float(r @ cinv @ r)

    fit = minimize_scalar(bao_free, bounds=(0.22, 0.40), method="bounded")
    h0 = [(o["obs"], o["sigma"]) for o in rows if o["key"].startswith("H0 ")]
    w = [1 / s ** 2 for _, s in h0]
    h0_loc = sum(v * wi for (v, _), wi in zip(h0, w)) / sum(w)
    chi_h0 = sum(((v - h0_loc) / s) ** 2 for v, s in h0)
    s8 = [(o["obs"], o["sigma"]) for o in rows if o["key"].startswith("S8")]
    chi_s8, k_s8 = 0.0, 0
    if s8:
        ws = [1 / s ** 2 for _, s in s8]
        m = sum(v * wi for (v, _), wi in zip(s8, ws)) / sum(ws)
        chi_s8, k_s8 = sum(((v - m) / s) ** 2 for v, s in s8), 1
    chi_dn = next(((0.0 - o["obs"]) / o["sigma"]) ** 2 for o in rows if o["key"] == "dn_s/dlnk")
    chi = fit.fun + chi_h0 + chi_s8 + chi_dn
    k = 5 + 2 + 1 + k_s8
    ce_m = block_scores(rows_from)["M"]
    return {"bao_om": float(fit.x), "bao_chi2": float(fit.fun), "H0_local": h0_loc, "chi2_M": chi, "k_M": k,
            "AIC_B2": chi + 2 * k, "BIC_B2": chi + math.log(n) * k, "AIC_CE": ce_m["AIC_CE"], "BIC_CE": ce_m["BIC_CE"]}


UNIFORM_RANGE = {"s12^2": 1.0, "s13^2": 1.0, "s23^2": 1.0, "delta_CKM": 2 * math.pi, "delta_PMNS": 360.0, "n_s": 2.0}
LOG_DECADES = 20


def parameter_bits(key: str, value: float, sigma: float, minimal_prior: bool = False) -> float:
    """2부 부호: 자유 매개변수를 측정 정밀도까지 적는 bit. 균등 범위 W 또는 20자릿수 로그 균등(W_eff = x ln 10²⁰).

    minimal_prior=True이면 로그 균등의 범위를 값 자체(e-배 한 칸)로 좁혀 사전 가정의 영향을 본다."""
    if key in UNIFORM_RANGE:
        w = UNIFORM_RANGE[key]
    else:
        w = abs(value) * (1.0 if minimal_prior else LOG_DECADES * math.log(10))
    return max(0.0, math.log2(w / sigma))


def mdl_blocks(rows_from: str = "IV", minimal_prior: bool = False) -> dict:
    """같은 화폐(bit)로 본 블록별 설명 길이 L = 매개변수 bit + 가설 bit + χ²/(2 ln 2). 작을수록 좋다."""
    rows, bao_ce = ce_rows(rows_from)
    label = "39" if rows_from == "IV" else "43"
    base = baseline_pulls(rows)
    scores = block_scores(rows_from)
    alpha = next(o for o in rows if o["key"] == "alpha_s world")
    ce_param_q = parameter_bits("alpha_s world", alpha["obs"], alpha["sigma"], minimal_prior)
    out = {}
    for blk in ("Q", "M"):
        chi_ce = sum(o["pull"] ** 2 for o in rows if o["block"] == blk) + (bao_ce if blk == "M" else 0.0)
        chi_b = sum(o["pull"] ** 2 for o in base if o["block"] == blk) + (planck_bao_chi2() if blk == "M" else 0.0)
        pb = sum(parameter_bits(o["key"], o["obs"], o["sigma"], minimal_prior)
                 for o, b in zip(rows, base) if b["block"] == blk and b["fitted"])
        ce = (ce_param_q if blk == "Q" else 0.0) + scores[blk]["bits"] + chi_ce / NATS_PER_BIT_CHI2
        out[blk] = {"L_CE": ce, "L_base": pb + chi_b / NATS_PER_BIT_CHI2, "base_param_bits": pb}
    return out


def break_even_bits(cmp: dict) -> float:
    """CE 점수가 기준과 같아지는 bit(이보다 적으면 CE가 앞선다)."""
    return (cmp["baseline"]["S"] - cmp["CE"]["chi2"] - 2) / NATS_PER_BIT_CHI2


def main() -> None:
    for item, b, n, why, scope in LEDGER:
        print(f"  {item:28s} {b:4.1f} -> {n:4.1f}  [{scope}]  {why}")
    print("totals:", ledger_totals(), "| unaudited:", UNAUDITED)
    for rf in ("IV", "full"):
        c = compare(rf)
        print(f"[{c['N']} rows] bits={c['bits']:.1f}")
        for side in ("CE", "baseline"):
            print(f"   {side:8s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c[side].items()})
        print("   baseline largest predicted pulls:", c["baseline_largest"], f"| break-even bits {break_even_bits(c):.1f}")
        print("   blocks:", {b: {k: round(v, 1) for k, v in d.items()} for b, d in block_scores(rf).items()})
        print("   flexible M baseline B2:", {k: round(v, 2) for k, v in flexible_baseline_m(rf).items()})
        for mp in (False, True):
            print(f"   MDL bits (minimal_prior={mp}):",
                  {b: {k: round(v, 1) for k, v in d.items()} for b, d in mdl_blocks(rf, mp).items()})
    s = OC.rescore_2026()
    print("2026 data CE rmse:", {k: round(v["after"], 3) for k, v in s.items()})


if __name__ == "__main__":
    main()
