"""전체 감사 — 우위의 출처, 등록 시점, 식 구조의 설명 길이, E4 지위. 원장: 43장 §43.87. 예측값을 바꾸지 않는다.

계산 전에 적은 규칙. 사전 목격: (다)의 기호 표를 짜는 동안 기호 수와 대략의 bit를 암산으로 먼저 보았다. 그래서 (다)는
판정이 아니라 민감도 보고로만 쓰고, 결론은 손익분기 b*(알파벳 선택과 무관한 양)로 적는다.

(가) VAL-01 분해. §43.47의 판본 V(39행)에서 CE와 기준(SM+ΛCDM)의 χ²를 네 몫으로 나눈다: SH0ES 행, BAO 13성분,
     기준이 자유 매개변수로 맞춘 행, 양쪽이 모두 예측하는 나머지 행. 두 긴장(SH0ES, BAO)을 뺀 비교를 함께 적는다.
     판정 규칙은 없다(보고).
(나) 등록 시점. 판정에 쓴 자료의 공개 연월(arXiv 번호 또는 출판 연도)과, 그 예측이 처음 들어간 사전 등록 판본의
     동결일을 비교한다. 자료가 동결일보다 먼저 공개되었으면 “동결 후 판정”으로 세지 않는다.
(다) 식 구조의 문법 부호. CE 식마다 전위(폴란드) 표기의 기호를 센다. 공통 정의(ŝ², δ, D, F, u, ε, w, q, N_e, Ω_m,
     S, H)는 한 번만 센다. 표준 물리(유니타리성, 볼츠만·거리 계산, ν 페르미–디랙)와 SM 상속 행(m_W/m_Z, Δa_μ)은 세지 않는다.
     - G1(개방 탐색): 기호마다 log2 A bit, 주 A = 32, 민감도 A ∈ {16, 64}.
     - G2(잎만): 연산자는 공짜, 잎 기호(기본량·정수)마다 log2(사용한 잎 종류 수) bit.
     - 부호 선택(T1의 방향)은 1 bit를 더한다. 장부(§43.47)의 가설 bit는 기호 비용으로 대체한다(이중 계산 방지).
     - L = 매개변수 bit + 식 bit + χ²/(2 ln 2). 기준은 §43.47 그대로(20자릿수 사전과 가장 좁은 사전).
     - 손익분기 b*: 블록마다 CE와 기준의 L이 같아지는 기호당 bit.
(라) E4 보른 배가(§43.82). 조건 (가) m² = 2m과 (다) m + 2 = 2m이 m ≠ 0에서 같은 해집합인지, E3의 진폭 A_m이
     V₃ = ℂ³에서 m ≤ 3에만 정의되는지 확인한다.

python -B -m examples.physics.rendering.ce_rendering_audit
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from examples.physics.rendering import ce_rendering_born_doubling as BDB
from examples.physics.rendering import ce_rendering_ledger as L

PREREGISTRATION_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "preregistration"

# ------------------------------------------------------------------ (가) VAL-01 분해
TENSION_ROW = "H0 SH0ES"


def val01_decomposition(rows_from: str = "IV") -> dict:
    rows, bao_ce = L.ce_rows(rows_from)
    base = L.baseline_pulls(rows)
    bao_b = L.planck_bao_chi2()
    parts = {k: {"CE": 0.0, "base": 0.0, "n": 0} for k in ("SH0ES", "BAO", "fitted_in_base", "predicted_by_both")}
    parts["BAO"] = {"CE": bao_ce, "base": bao_b, "n": 13}
    for o, b in zip(rows, base):
        key = "SH0ES" if o["key"] == TENSION_ROW else ("fitted_in_base" if b["fitted"] else "predicted_by_both")
        parts[key]["CE"] += o["pull"] ** 2
        parts[key]["base"] += b["pull"] ** 2
        parts[key]["n"] += 1
    tot = {s: sum(p[s] for p in parts.values()) for s in ("CE", "base")}
    n = sum(p["n"] for p in parts.values())
    rest = {s: tot[s] - parts["SH0ES"][s] - parts["BAO"][s] for s in ("CE", "base")}
    n_rest = n - parts["SH0ES"]["n"] - parts["BAO"]["n"]
    gain = tot["base"] - tot["CE"]
    return {
        "parts": parts, "total": tot, "N": n,
        "rmse": {s: math.sqrt(tot[s] / n) for s in tot},
        "gain": gain,
        "gain_share": {k: (p["base"] - p["CE"]) / gain for k, p in parts.items()},
        "base_share_of_two_tensions": (parts["SH0ES"]["base"] + parts["BAO"]["base"]) / tot["base"],
        "without_two_tensions": {"N": n_rest, "chi2": rest, "rmse": {s: math.sqrt(rest[s] / n_rest) for s in rest}},
    }


# ------------------------------------------------------------------ (나) 등록 시점
# (예측, 판정 자료, 자료 공개 연월). 모두 43장 본문에 인용된 자료다.
EVIDENCE = (
    ("P17", "arXiv:2606.17994 (CMB-SPA + DESI DR2 + DES Y5, 52 meV)", "2026-06"),
    ("P19", "arXiv:2511.14593 (JUNO 첫 결과)", "2025-11"),
    ("P03", "arXiv:2511.14593 (JUNO 첫 결과)", "2025-11"),
    ("P07", "arXiv:2510.23823 (H0DN)", "2025-10"),
    ("P37", "arXiv:2502.05259 (JAGB 2.0)", "2025-02"),
    ("P32", "arXiv:2408.06153 (CCHP JWST)", "2024-08"),
    ("P16", "arXiv:2503.19441 (KiDS-Legacy)", "2025-03"),
    ("P40", "Laghi 외 2021 (CQG 38, 095005)", "2021-01"),
    ("P41", "arXiv:2304.05202 (ACT DR6 렌즈) + DESI DR2", "2025-03"),
)


def _manifests() -> list[tuple[int, dict]]:
    out = []
    for path in PREREGISTRATION_ROOT.glob("rendering_predictions_v*.json"):
        v = int(path.stem.rsplit("_v", 1)[1])
        out.append((v, json.loads(path.read_text(encoding="utf-8"))))
    return sorted(out)


def first_registration(pid: str) -> tuple[int, str]:
    for v, m in _manifests():
        if any(p.get("id") == pid for p in m["predictions"]):
            return v, m["freeze"]["freeze_date"]
    raise KeyError(pid)


def registration_timing() -> dict:
    rows = []
    for pid, source, published in EVIDENCE:
        v, frozen = first_registration(pid)
        rows.append({"id": pid, "version": v, "frozen": frozen, "source": source, "published": published,
                     "after_freeze": published > frozen[:7]})
    versions = _manifests()
    return {"rows": rows, "post_freeze_verdicts": sum(r["after_freeze"] for r in rows),
            "versions": len(versions), "first_freeze": versions[0][1]["freeze"]["freeze_date"],
            "last_freeze": versions[-1][1]["freeze"]["freeze_date"]}


# ------------------------------------------------------------------ (다) 식 구조의 문법 부호
# 전위 표기. 공통 정의는 한 번만 적는다. 블록은 §43.47과 같다(α_s와 맛깔·결합은 Q, 우주 성분·급팽창·H₀는 M).
FORMULAS = {
    "Q": {
        "E4: ŝ² = 4 α_s^{4/3}": "* 4 ^ a / 4 3",
        "δ = ŝ²(1−ŝ²)": "* s2 - 1 s2",
        "D = 3 + δ": "+ 3 d",
        "F = 1 + α_s D (M_H/M_Z)": "+ 1 * a D",
        "u = ŝ²/4": "/ s2 4",
        "ε = α_s^{1/6}": "^ a / 1 6",
        "w = 1 + δ/2π": "+ 1 / d * 2 pi",
        "|V_us| = 4u/w": "/ * 4 u w",
        "|V_cb| = u ε w": "* * u eps w",
        "U1: (β, α) = (π/8, π/2)": "/ pi 8 / pi 2",
        "m_μ/m_τ = u w": "* u w",
        "Koide 2/3": "koide / 2 3",
        "α_em⁻¹: 합규칙 + 한 통로 고리": "/ 1 * s2 / - * / 1 * 2 pi + 1 / a * 16 pi a + 1 s2",
        "v/M_Pl = e^{−12D}(1+α_s/4π)/F": "/ + 1 / a * 4 pi * F exp * 12 D",
        "s13² = δ/8": "/ d 8",
        "s12² = (1 − 2δ/8)/3": "* / 1 3 - 1 / * 2 d 8",
        "s23² = (1 − 4δ/8)/2": "* / 1 2 - 1 / * 4 d 8",
        "T1: |U_μ1| = |U_τ1|": "= abs U 2 1 abs U 3 1",
    },
    "M": {
        "q = e^{−D(1−q)}": "fix exp * D - x 1",
        "N_e = (3/2)·12·D": "* * / 3 2 12 D",
        "Ω_m = q + (1−q) α_s D/F": "+ q * - 1 q / * a D F",
        "S = (π²/2) N_e − π δ (1−q)": "- * / ^ pi 2 2 Ne * * pi d - 1 q",
        "H₀ t_Pl = √π e^{−S/2}(1+α_s/4π)": "/ * sqrt pi + 1 / a * 4 pi exp / S 2",
        "O1: 직접 판독 = H/cos(π/8)": "/ H cos / pi 8",
        "n_s = 1 − 2/N_e": "- 1 / 2 Ne",
        "A_s: Q_s": "* * * / 2 pi ^ - 1 q / D + D 1 q - 1 q",
        "A_s = Q_s² q/[(1−q)² 2π N_e²]": "/ * ^ Qs 2 q * * ^ - 1 q 2 * 2 pi ^ Ne 2",
        "dn_s/dlnk = −2/N_e²": "- 0 / 2 ^ Ne 2",
        "ω_b = q h²": "* q ^ h 2",
        "ω_c = (Ω_m − q) h²": "* - Om q ^ h 2",
        "W2: 1 + α_s^{2/3} cos(H_Λ t/2)": "+ 1 * ^ a / 2 3 cos / * HL t 2",
        "G1m: 1 − Ω_m(1 − cos ψ/2)": "- 1 * Om - 1 cos / psi 2",
    },
}
SIGN_BITS = {"Q": 1.0, "M": 0.0}            # T1의 원 방향(M1)
OPERATORS = {"*", "/", "+", "-", "^", "exp", "sqrt", "cos", "fix", "abs", "U", "=", "koide"}
ALPHABETS = (16, 32, 64)
MAIN_ALPHABET = 32


def token_counts() -> dict:
    out = {}
    for blk, table in FORMULAS.items():
        toks = [t for f in table.values() for t in f.split()]
        leaves = [t for t in toks if t not in OPERATORS]
        out[blk] = {"tokens": len(toks), "leaves": len(leaves), "formulas": len(table)}
    all_toks = [t for table in FORMULAS.values() for f in table.values() for t in f.split()]
    out["leaf_kinds"] = len({t for t in all_toks if t not in OPERATORS})
    out["symbol_kinds"] = len(set(all_toks))
    return out


def grammar_mdl(minimal_prior: bool = False) -> dict:
    """블록별 L_CE(장부 부호, G1, G2)와 L_기준, 손익분기 기호당 bit."""
    rows, bao_ce = L.ce_rows("IV")
    ledger = L.mdl_blocks("IV", minimal_prior)
    alpha = next(o for o in rows if o["key"] == "alpha_s world")
    a_bits = L.parameter_bits("alpha_s world", alpha["obs"], alpha["sigma"], minimal_prior)
    counts = token_counts()
    g2_per_leaf = math.log2(counts["leaf_kinds"])
    out = {}
    for blk in ("Q", "M"):
        chi = sum(o["pull"] ** 2 for o in rows if o["block"] == blk) + (bao_ce if blk == "M" else 0.0)
        fixed = (a_bits if blk == "Q" else 0.0) + SIGN_BITS[blk] + chi / L.NATS_PER_BIT_CHI2
        n, nl = counts[blk]["tokens"], counts[blk]["leaves"]
        out[blk] = {
            "L_base": ledger[blk]["L_base"],
            "L_CE_ledger": ledger[blk]["L_CE"],
            "L_CE_G1": {A: fixed + n * math.log2(A) for A in ALPHABETS},
            "L_CE_G2": fixed + nl * g2_per_leaf,
            "break_even_bits_per_token": (ledger[blk]["L_base"] - fixed) / n,
            "tokens": n, "leaves": nl,
        }
    return out


# ------------------------------------------------------------------ (라) E4 보른 배가
E3_DIMENSION = 3                             # E1: V₃ = ℂ³, A_m(W)는 W ⊂ V₃에서만 정의된다


def born_doubling_audit(m_max: int = 12) -> dict:
    ms = range(1, m_max + 1)
    born = {m for m in ms if BDB.born_consistent(m)}
    jump = {m for m in ms if BDB.user_jump(m)}
    return {
        "born_equals_user_jump": born == jump,
        "solution": sorted(born),
        "needs_A_of": sorted(2 * m for m in born),
        "A_defined_up_to": E3_DIMENSION,
        "uses_undefined_amplitude": any(2 * m > E3_DIMENSION for m in born),
    }


def main() -> None:
    d = val01_decomposition()
    print("(가) VAL-01:", {k: round(v, 2) for k, v in d["total"].items()}, "gain", round(d["gain"], 2),
          "| base share of SH0ES+BAO", round(d["base_share_of_two_tensions"], 3))
    for k, p in d["parts"].items():
        print(f"   {k:18s} n={p['n']:2d} CE={p['CE']:6.2f} base={p['base']:6.2f} share_of_gain={d['gain_share'][k]:+.3f}")
    print("   without SH0ES+BAO:", d["without_two_tensions"])
    t = registration_timing()
    print("(나) versions", t["versions"], t["first_freeze"], "→", t["last_freeze"],
          "| post-freeze verdicts:", t["post_freeze_verdicts"])
    for r in t["rows"]:
        print(f"   {r['id']:4s} v{r['version']:<2d} frozen {r['frozen']}  data {r['published']}  {r['source']}")
    print("(다) tokens:", token_counts())
    for mp in (False, True):
        print(f"   minimal_prior={mp}:")
        for blk, v in grammar_mdl(mp).items():
            print(f"     {blk}: L_base={v['L_base']:.1f} ledger={v['L_CE_ledger']:.1f} "
                  f"G1={ {A: round(x, 1) for A, x in v['L_CE_G1'].items()} } G2={v['L_CE_G2']:.1f} "
                  f"b*={v['break_even_bits_per_token']:.2f} bit/token (n={v['tokens']})")
    print("(라) born doubling:", born_doubling_audit())


if __name__ == "__main__":
    main()
