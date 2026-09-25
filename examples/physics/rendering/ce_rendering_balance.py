"""균형 원리 PB — 과거·현재·미래의 균형, 현실에 나오는 최소작용점. 원장: 43장 §43.99. 예측값을 바꾸지 않는다.

사용자 가설(2026-09-25): “E4는 무언가의 균형추다.” → “과거와 현재 그리고 미래의 균형. 현실에 나오기 위한 최소작용점.”
PB: 현재(기록)는 과거에서 오는 몫과 미래에서 오는 몫이 같아지는 최소작용점에서만 생긴다.
사용자 요청 “전부 실험해보자”에 따라 세 가지 읽기와 숫자의 세 판본을 모두 시험한다. 계산 전에 적은 판본과 kill:

(가) 전파 인자. D(s) = 1/(s − M² + iMΓ). 과거 몫 = 지연 G_ret = D, 미래 몫 = 선행 G_adv = conj(D)(실수 s).
    시간대칭 몫 ½(G_ret + G_adv) = Re D, 현재(기록) 몫 = Im D(흡수부 = 붕괴). 작용 비용 = Re D⁻¹ = s − M².
    PB → 사건점 = 과거·미래 몫이 상쇄되고 작용 비용이 0인 점.
    K1: Re D의 영점이 유일하지 않거나 극이 아니면, 또는 |Im D|의 최대가 극이 아니면 (가) 기각.
    K3(R1): 사건점에서 시간대칭 몫이 남으면(미래 몫이 기록에 섞이면) 기각.
(자) 현실에 나온 숫자. E4의 α = (s²/4)^(3/4)를 어떤 숫자로 읽는가.
    S-a  척도만 정함: s² = v17 MS-bar, α = MS-bar α_s.
    S-b1 극의 기록에서 뽑은 결합: s² = 유효각, α = Z 극 기록에서 뽑은 MS-bar α_s.
    S-b2 기록 그 자체: s² = 유효각, α = Z 하드론 폭의 유효 결합 α_R(Γ_had/Γ_had⁰ − 1 = α_R/π).
    민감도: 유효각의 충돌기 직접 평균.
    K2(§43.63 RS가 값을 보기 전에 정한 비교): Z 극 기록 대비 |pull| ≥ 3이면 그 판본 기각. 세계 평균 대비는 민감도 행이다.
    공개: S-b2의 세계 평균 대비 값(약 −3.9σ)은 이 설계 전에 대화의 어림에서 이미 봤다.
(나) 세 몫. 과거 = 기록된 몫 p, 미래 = 기록 안 된 몫 f, 현재 = 지금 기록 중인 몫 c, p + c + f = 1.
    PB(p = f) → 동전 a = f = ½(1 − c). c = (α_s/2π)(1 + δ/2π)이면 FP와 같은 식이다(목격 후 읽기, 새 증거 없음).
    K4: FP의 s²가 판본의 s²와 3σ 넘게 어긋나면 (나)와 그 판본은 양립하지 않는다. 유효각 판본에는 §43.55 가족(24개)을
        다시 대고 적중 수와 우연 기대를 센다.
(다) 우주. 과거 = CL(과거가 끎), 미래 = TT(최종 지평선이 시계를 돌림), 현재 = 지금. PB(지금 = 정확한 절반) → H_Λ t₀ = π/4.
    §43.14에서 이미 반례(음향 각도 +14σ). K5: 요구되는 Ω_m이 자료와 3σ 넘게 어긋나면 기각을 확인한다.

python -B -m examples.physics.rendering.ce_rendering_balance
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from examples.physics.rendering import ce_rendering_boundary_loop as BLP
from examples.physics.rendering import ce_rendering_causal_lock as CLK
from examples.physics.rendering import ce_rendering_distinction as DST
from examples.physics.rendering import ce_rendering_e4_anchor as E4A
from examples.physics.rendering import ce_rendering_pole as POL
from examples.physics.rendering import ce_rendering_thermal_time as TTM

M_Z, GAMMA_Z = POL.M_Z, POL.GAMMA_Z
R_NS = (1.0, 1.409, -12.77, -79.98)            # R(s) 비단일항 QCD 급수, n_f = 5 (Baikov–Chetyrkin–Kühn 2008)
Z_POLE = POL.ALPHA_S_RECORDS["Z pole (R_l, Gamma_Z, sigma_had)"]
WORLD = POL.ALPHA_S_RECORDS["world average"]
S2 = {"MS-bar": DST.SCHEMES["MS-bar"], "effective": DST.SCHEMES["effective"],
      "effective (direct avg)": DST.EFFECTIVE_DIRECT}
READINGS = {"S-a": ("MS-bar", "msbar"), "S-b1": ("effective", "msbar"), "S-b2": ("effective", "record"),
            "S-b1 (direct avg)": ("effective (direct avg)", "msbar"),
            "S-b2 (direct avg)": ("effective (direct avg)", "record")}


def _zeros(x: np.ndarray, y: np.ndarray) -> list[float]:
    s = np.sign(y)
    out = [float(x[i]) for i in np.flatnonzero(s == 0)]
    idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    out += [float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])) for i in idx]
    return sorted(out)


def pole_balance(n: int = 40001, span: float = 10.0) -> dict:
    """(가): 과거(지연)·미래(선행) 몫, 시간대칭 몫, 기록 몫, 작용 비용을 Z 극 주변 ±10Γ에서 훑는다."""
    rs = np.linspace(M_Z - span * GAMMA_Z, M_Z + span * GAMMA_Z, n)
    d = np.array([POL.propagator(r * r) for r in rs])
    zeros = _zeros(rs, d.real)                     # ½(G_ret + G_adv) = Re D
    at = POL.propagator(M_Z ** 2)
    return {"sym_zero_count": len(zeros), "sym_zero_at": zeros[0] if zeros else float("nan"),
            "record_peak_at": float(rs[np.argmax(np.abs(d.imag))]),
            "cost_at_pole": (1 / at).real,               # 운동 항 − 질량 항
            "record_rate_at_pole": (1 / at).imag,        # MΓ
            "past_plus_future_at_pole": abs(at + at.conjugate()),
            "sym_over_record_at_pole": abs(at.real) / abs(at.imag),
            "phase_deg_at_pole": math.degrees(cmath.phase(at)),
            "grid_step": float(rs[1] - rs[0])}


def alpha_record(alpha_s: float) -> float:
    """MS-bar α_s → Z 하드론 폭의 유효 결합 α_R(Γ_had/Γ_had⁰ − 1 = α_R/π)."""
    a = alpha_s / math.pi
    return math.pi * sum(c * a ** (k + 1) for k, c in enumerate(R_NS))


def d_alpha_record(alpha_s: float) -> float:
    a = alpha_s / math.pi
    return sum((k + 1) * c * a ** k for k, c in enumerate(R_NS))


def _data(kind: str) -> dict:
    """비교 자료를 판본의 숫자(MS-bar 또는 기록 그 자체)로 옮긴다."""
    out = {}
    for name, (v, s) in (("Z pole", Z_POLE), ("world", WORLD)):
        out[name] = (v, s) if kind == "msbar" else (alpha_record(v), s * d_alpha_record(v))
    return out


def scheme_readings() -> dict:
    """(자): 판본마다 E4의 α = (s²/4)^(3/4)를 Z 극 기록(주 비교)·세계 평균(민감도)과 비교한다."""
    out = {}
    for key, (s2_key, kind) in READINGS.items():
        s2, s2_err = S2[s2_key]
        x = (s2 / 4) ** 0.75
        dx = 0.75 * x * s2_err / s2
        data = _data(kind)
        pulls = {k: (x - v) / math.hypot(e, dx) for k, (v, e) in data.items()}
        out[key] = {"s2": s2, "alpha": x, "Z_pole_data": data["Z pole"][0], "pull_Z_pole": pulls["Z pole"],
                    "pull_world": pulls["world"], "killed": abs(pulls["Z pole"]) >= 3,
                    "warn_world": abs(pulls["world"]) >= 3}
    t = DST.scheme_test({k: S2[k] for k in ("MS-bar", "effective")})
    out["P36_FP"] = {k: {"P36": v["P36"], "FP": v["FP"]} for k, v in t.items()}
    out["lever"] = {"s2_side": out["S-b1"]["pull_Z_pole"] - out["S-a"]["pull_Z_pole"],
                    "alpha_side": out["S-b2"]["pull_Z_pole"] - out["S-b1"]["pull_Z_pole"]}
    return out


def three_shares() -> dict:
    """(나): FP를 세 몫의 균형으로 읽고, §43.55 가족을 판본마다 다시 댄다."""
    fp = E4A.fixed_point(*E4A.ADOPTED)
    shares = {"past": fp["r"], "future": fp["r"], "present": fp["eps"]}
    fam = {f"{l}|{c}": E4A.fixed_point(l, c)["s2"] for l in E4A.L_TERMS for c in E4A.C_TERMS}
    lo, hi = min(fam.values()), max(fam.values())
    pulls_fp, hits = {}, {}
    for k, (v, e) in S2.items():
        pulls_fp[k] = (fp["s2"] - v) / e
        p = {name: (s - v) / e for name, s in fam.items()}
        hits[k] = {"1sigma": sorted(n for n, x in p.items() if abs(x) <= 1),
                   "3sigma": sorted(n for n, x in p.items() if abs(x) <= 3),
                   "chance_3sigma": len(fam) * min(1.0, 6 * e / (hi - lo))}
    return {"shares": shares, "sum": sum(shares.values()), "fp_alpha_s": fp["alpha_s"], "fp_s2": fp["s2"],
            "pull_s2": pulls_fp, "pull_world": fp["pull_as_world"],
            "fp_vs_lepton": BLP.alpha_from_lepton_rule()["FP_vs_lepton"],
            "family_n": len(fam), "family_s2_range": (lo, hi), "hits": hits,
            "killed": {k: abs(v) >= 3 for k, v in pulls_fp.items()}}


def cosmic_now() -> dict:
    """(다): 지금 = 정확한 절반(H_Λ t₀ = π/4)이 요구하는 Ω_m. §43.14에서 이미 반례."""
    h = TTM.exact_half_now()
    pulls = {k: v for k, v in h.items() if k not in ("Om_star", "CE_core_Om")}
    return {"Om_star": h["Om_star"], "CE_core_Om": h["CE_core_Om"], "CE_gap_rel": h["Om_star"] / h["CE_core_Om"] - 1,
            "phase_gap_rel": CLK.why_now()["rel_gap"], "pulls": pulls,
            "killed": any(abs(v) >= 3 for v in pulls.values()), "killed_43_14": True}


def verdict() -> dict:
    p, s, t, c = pole_balance(), scheme_readings(), three_shares(), cosmic_now()
    return {"K1_pole_unique": p["sym_zero_count"] == 1 and abs(p["sym_zero_at"] - M_Z) < p["grid_step"]
            and abs(p["record_peak_at"] - M_Z) < p["grid_step"],
            "K3_no_future_in_record": p["sym_over_record_at_pole"] < 1e-12,
            "K2_killed": {k: v["killed"] for k, v in s.items() if k.startswith("S-")},
            "K2_warn_world": {k: v["warn_world"] for k, v in s.items() if k.startswith("S-")},
            "K4_killed": t["killed"], "K5_killed": c["killed"]}


def _r(x, n: int = 4):
    if isinstance(x, float):
        return round(x, n)
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_r(v, n) for v in x)
    return x


def main() -> None:
    print("(가) pole balance:", _r(pole_balance(), 8))
    for k, v in scheme_readings().items():
        print("(자)", k, _r(v))
    for k, v in three_shares().items():
        print("(나)", k, _r(v, 6))
    print("(다) cosmic now:", _r(cosmic_now()))
    print("verdict:", verdict())


if __name__ == "__main__":
    main()
