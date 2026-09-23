"""Higgs의 무게 — F는 살아남은 계보의 분배함수다. 원장: 43장 §43.65. 예측값을 바꾸지 않는다.

코어에서 Ω_m = q + (1 − q)(F − 1)/F이고 F = 1 + α_s D는 살아남은 계보의 분배함수(자유 1, 묶임 α_s D)다.
Higgs 행 M_H/M_Z = F는 그래서 “Higgs/Z = 살아남은 계보 전체의 무게 ÷ 자유 계보의 무게”다. 계산 전에 적은 판본과 규칙:

HB(읽기). Z는 전하에 붙어 계보의 머리(자유)만, Higgs는 질량에 붙어 머리 + 강하게 붙잡힌 자식 한 세대를 본다:
   M_H/M_Z = E[1 + N], N ~ Poisson(α_s D).
한 걸음 시험(사용자 “부트스트랩 자기재귀”). 끝없는 재귀(총 후손 1/(1 − α_s D)), 복리(e^{α_s D})를 선형과 비교.
   kill: 선형보다 Δχ² > 9. 선형은 M_H에 맞춰 고른 꼴이므로 구성상 결과이며 증거가 아니라 패턴(§43.55·§43.58과 같은 한 걸음)이다.
계층 교차 확인. v/M_Pl = e^{−12D}(1 + α_s/4π)/F에서 F를 역산해 코어 F·Higgs 측정 F와 비교(규칙이 코어 F로 만들어졌으므로 구성상).
   이론 오차는 ŝ² → δ → 12D 경로(상대 12(1 − 2ŝ²)σ_ŝ²).
Q2(탐색, 암산으로 먼저 봄). §43.64의 (1 − q²)(나와 아닌 나 두 계보가 모두 끊기면 기록 없음)를 보편 인자로 두고
   FP의 λ와 α_s D(Higgs F, 우주 성분, 계층 행) 네 곳에 함께 적용한다. 채택: 네 곳의 공동 χ²가 줄고 어느 곳도 2σ를 넘지 않을 때.

python -B -m examples.physics.rendering.ce_rendering_higgs_weight
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_boundary_loop as BLP
from examples.physics.rendering import ce_rendering_fp_ladder as FPL
from examples.physics.rendering import ce_rendering_registry as R

M_Z = 91.1876
OM_PLANCK = (0.3153, 0.0073)


def _rows() -> dict:
    return {r.key: r for r in R.rows("SK") if r.key in ("M_H/M_Z", "v/M_Pl")}


def _core() -> dict:
    return R.core(R.calibrated_alpha_s()[0])


def partition_identity() -> dict:
    c = _core()
    m = c["a"] * c["D"]
    return {"F_minus_1_minus_aD": c["F"] - 1 - m,
            "Om_from_partition_minus_core": c["q"] + (1 - c["q"]) * (c["F"] - 1) / c["F"] - c["Om"]}


def step_forms() -> dict:
    c = _core()
    m = c["a"] * c["D"]
    row = _rows()["M_H/M_Z"]
    forms = {"one step 1+m": 1 + m, "compound e^m": math.exp(m), "full recursion 1/(1-m)": 1 / (1 - m)}
    out = {k: {"F": f, "M_H": M_Z * f, "pull": (f - row.obs) / row.sig_up} for k, f in forms.items()}
    base = out["one step 1+m"]["pull"] ** 2
    for v in out.values():
        v["d_chi2"] = v["pull"] ** 2 - base
        v["killed"] = v["d_chi2"] > 9
    return out


def _hier_sigma_rel(c: dict) -> float:
    row = _rows()["v/M_Pl"]
    return math.hypot(12 * (1 - 2 * c["s2"]) * R.SZ2_ERR, row.sig_up / row.obs)


def hierarchy_cross_check() -> dict:
    c = _core()
    row = _rows()["v/M_Pl"]
    f_impl = math.exp(-12 * c["D"]) * (1 + c["a"] / (4 * math.pi)) / row.obs
    s = f_impl * _hier_sigma_rel(c)
    h = _rows()["M_H/M_Z"]
    return {"F_implied": f_impl, "sigma": s, "F_core": c["F"], "F_higgs_obs": h.obs,
            "implied_vs_core": (f_impl - c["F"]) / s,
            "implied_vs_higgs": (f_impl - h.obs) / math.hypot(s, h.sig_up),
            "M_H_required": M_Z * f_impl, "M_H_required_sigma": M_Z * s}


def q2_universality() -> dict:
    """Q2: (1 − q²)를 λ(FP)와 α_s D(Higgs, Ω_m, 계층)에 함께 적용했을 때 네 곳의 pull."""
    c = _core()
    q, m = c["q"], c["a"] * c["D"]
    rows = _rows()
    fp_plain = FPL.solve(FPL._series(1))["pull_lepton"]
    fp_q2 = FPL.seen_before()["lam q^2"]["pull_lepton"]
    out = {}
    for tag, mm, fp in (("plain", m, fp_plain), ("q2", m * (1 - q * q), fp_q2)):
        f = 1 + mm
        om = q + (1 - q) * mm / f
        hier = math.exp(-12 * c["D"]) * (1 + c["a"] / (4 * math.pi)) / f
        pulls = {"FP vs lepton rule": fp,
                 "M_H/M_Z": (f - rows["M_H/M_Z"].obs) / rows["M_H/M_Z"].sig_up,
                 "Omega_m (Planck)": (om - OM_PLANCK[0]) / OM_PLANCK[1],
                 "v/M_Pl (th+exp)": (hier / rows["v/M_Pl"].obs - 1) / _hier_sigma_rel(c)}
        out[tag] = {"pulls": pulls, "chi2": sum(p * p for p in pulls.values())}
    out["adopt"] = (out["q2"]["chi2"] < out["plain"]["chi2"]
                    and all(abs(p) <= 2 for p in out["q2"]["pulls"].values()))
    return out


def main() -> None:
    print("partition identity:", partition_identity())
    for k, v in step_forms().items():
        print(f"{k:24s}", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("hierarchy:", {k: round(v, 6) for k, v in hierarchy_cross_check().items()})
    q = q2_universality()
    for tag in ("plain", "q2"):
        print(tag, {k: round(v, 2) for k, v in q[tag]["pulls"].items()}, "chi2", round(q[tag]["chi2"], 2))
    print("adopt Q2:", q["adopt"])


if __name__ == "__main__":
    main()
