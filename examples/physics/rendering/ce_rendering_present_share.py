"""현재 몫의 꼬리 — CE의 날카로운 식들을 동전의 자로 쓴다. 원장: 43장 §43.101. 예측값을 바꾸지 않는다.

문제(§43.64). 입력 없는 고정점 FP r = ½[1 − λ − ε], λ = (α_s/2π)(1 + δ/2π)에 필요한 보정 ε = −(4.94 ± 1.32)×10⁻⁵는
m_τ 하나로 재서 폭이 27%다. 네 꼴(λq², λ²/8, λe⁻⁶, λ²e⁻²)이 동률이다. §43.99 (나)에서 λ는 현재가 가져가는 몫이다.
사용자 요청(2026-09-25, “저기로 가보자”): 현재가 가져가는 한 고리의 정확한 크기를 겨냥한다.
발상: 과녁을 좁힌다. CE 안에서 동전 값을 재는 날카로운 식 다섯을 자로 쓰고, 각 식을 관측값에 맞추는 동전 α와 오차(관측 오차만)를 낸다.
    M1 렙톤 규칙    m_μ/m_τ = α^{4/3}(1 + δ/2π)              (m_τ 오차)
    M2 Koide 자     m_e/m_μ = Koide(M1의 비), Q = 2/3        (m_e·m_μ 오차; Koide를 정확하다고 둔다)
    M3 E4           ŝ² = 4α^{4/3}                            (ŝ² 오차)
    M4 합규칙       α_em⁻¹(M_Z)                              (코어 식, §43.93)
    M5 계층 식      v/M_Pl                                   (코어 식, §43.93)

계산 전에 적은 판정:
K1(자들의 일치). 다섯 자의 가중 평균 둘레 χ². 자유도 4에서 χ² > 18.47(p < 0.001)이면 자들이 서로 맞지 않는다.
    그러면 CE 식들은 그 정밀도에서 정확하지 않은 것이고, FP–렙톤 차이를 자들의 흩어짐(이론 해상도)과 견준다.
K2(자들이 맞으면). 합친 동전으로 필요한 ε를 다시 구하고 §43.64의 네 꼴과 PB 꼴 λ²/8(§43.99 읽기)을 |pull| > 3이면 기각한다.
공개. 네 꼴이 m_τ 과녁에 맞는다는 것은 §43.64에서 안다. M3–M5 행의 pull(−0.60, +0.10 등)도 이미 보았다.

python -B -m examples.physics.rendering.ce_rendering_present_share
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_boundary_loop as BLP
from examples.physics.rendering import ce_rendering_registry as R

CHI2_P001_DOF4 = 18.47
REL_ME_MMU = math.hypot(0.00000000015 / 0.51099895, 0.0000023 / 105.6583755)   # CODATA 2022
FORMS = {"lambda q^2": lambda lam, c: lam * c["q"] ** 2, "lambda^2/8": lambda lam, c: lam * lam / 8,
         "lambda e^-6": lambda lam, c: lam * math.exp(-6), "lambda^2 e^-2": lambda lam, c: lam * lam * math.exp(-2)}


def _lepton_ratio(a: float) -> float:
    return R.flavour_words(R.core(a))["m_mu/m_tau"]


METERS = {
    "M1 lepton rule": (_lepton_ratio, R.M_MU / R.M_TAU, R.M_MU / R.M_TAU * R.M_TAU_ERR / R.M_TAU),
    "M2 Koide": (lambda a: R.koide_me_over_mmu(_lepton_ratio(a)), R.M_E / R.M_MU, R.M_E / R.M_MU * REL_ME_MMU),
    "M3 E4 s^2": (lambda a: 4 * a ** (4 / 3), R.SZ2, R.SZ2_ERR),
    "M4 sum rule": (lambda a: R.alpha_em_inv(R.core(a)), R.AEM_INV_MZ, R.AEM_INV_MZ_ERR),
    "M5 hierarchy": (lambda a: R.v_over_mpl(R.core(a)), R.V_EW / R.M_PLANCK, R.V_EW / R.M_PLANCK * 1.1e-5),
}


def meter(name: str) -> dict:
    f, obs, sig = METERS[name]
    a = brentq(lambda x: f(x) - obs, 0.10, 0.13, xtol=1e-15)
    h = 1e-7
    slope = (f(a + h) - f(a - h)) / (2 * h)
    return {"alpha": a, "sigma": abs(sig / slope)}


def meters() -> dict:
    """K1: 다섯 자의 동전 값, 가중 평균, χ², 흩어짐."""
    m = {k: meter(k) for k in METERS}
    w = {k: 1 / v["sigma"] ** 2 for k, v in m.items()}
    mean = sum(w[k] * m[k]["alpha"] for k in m) / sum(w.values())
    chi2 = sum(((v["alpha"] - mean) / v["sigma"]) ** 2 for v in m.values())
    vals = [v["alpha"] for v in m.values()]
    return {"meters": m, "mean": mean, "mean_sigma": sum(w.values()) ** -0.5, "chi2": chi2,
            "consistent": chi2 <= CHI2_P001_DOF4, "spread": max(vals) - min(vals),
            "pulls_vs_mean": {k: (v["alpha"] - mean) / v["sigma"] for k, v in m.items()}}


def corrected_fp(form: str | None) -> float:
    """FP r = ½[1 − λ − ε], ε = −form(λ) (§43.64의 부호: 기울기를 줄인다). 자기일관 고정점의 α = r³."""
    def g(r):
        a = r ** 3
        c = R.core(a)
        lam = a / (2 * math.pi) * (1 + c["d"] / (2 * math.pi))
        eps = 0.0 if form is None else -FORMS[form](lam, c)
        return 0.5 * (1 - lam - eps)
    r = brentq(lambda x: x - g(x), 0.3, 0.5, xtol=1e-15)
    return r ** 3


def against_meters() -> dict:
    """FP와 보정된 FP를 자마다 비교한다(pull은 그 자의 관측 오차 단위)."""
    m = meters()
    cands = {"FP": corrected_fp(None), **{k: corrected_fp(k) for k in FORMS}}
    table = {c: {k: (a - v["alpha"]) / v["sigma"] for k, v in m["meters"].items()} for c, a in cands.items()}
    return {"candidates": cands, "pulls": table,
            "fp_gap": cands["FP"] - m["meters"]["M1 lepton rule"]["alpha"],
            "candidate_spread": max(cands[k] for k in FORMS) - min(cands[k] for k in FORMS)}


def _fp_with(eps_fn) -> float:
    def g(r):
        a = r ** 3
        c = R.core(a)
        lam = a / (2 * math.pi) * (1 + c["d"] / (2 * math.pi))
        return 0.5 * (1 - lam - eps_fn(lam, c))
    r = brentq(lambda x: x - g(x), 0.3, 0.5, xtol=1e-16)
    return r ** 3


def exploration() -> dict:
    """목격 후 탐색(증거 아님): 자마다 필요한 상수 보정 ε*, ε*/λ², 사후 가족 28개의 적중."""
    m = meters()
    eps_for = lambda target: brentq(lambda e: _fp_with(lambda lam, c: e) - target, -2e-4, 2e-4, xtol=1e-16)
    eps = {k: eps_for(v["alpha"]) for k, v in m["meters"].items()}
    eps_err = {k: abs(eps_for(v["alpha"] + v["sigma"]) - eps[k]) for k, v in m["meters"].items()}
    c = R.core(m["mean"])
    lam = m["mean"] / (2 * math.pi) * (1 + c["d"] / (2 * math.pi))
    fam = {f"lambda^2/{k}": (lambda k: lambda L, cc: -L * L / k)(k) for k in range(1, 21)}
    fam.update({f"lambda q^{j}": (lambda j: lambda L, cc: -L * cc["q"] ** j)(j) for j in (1, 2, 3)})
    fam.update({f"lambda e^-{n}": (lambda n: lambda L, cc: -L * math.exp(-n))(n) for n in range(4, 9)})
    m5 = m["meters"]["M5 hierarchy"]
    pulls = {k: (_fp_with(f) - m5["alpha"]) / m5["sigma"] for k, f in fam.items()}
    return {"eps_star": eps, "eps_err": eps_err, "lambda": lam, "eps_over_lambda2": eps["M2 Koide"] / lam ** 2,
            "present_share": lam + eps["M2 Koide"], "present_over_lambda": 1 + eps["M2 Koide"] / lam,
            "family_n": len(fam), "hits_M5": sorted(k for k, p in pulls.items() if abs(p) <= 3),
            "pulls_M5": pulls}


def tau_mass(meter_name: str) -> dict:
    """부산물: 자가 가리키는 동전으로 렙톤 규칙을 거꾸로 써서 m_τ를 낸다."""
    v = meters()["meters"][meter_name]
    mt = R.M_MU / _lepton_ratio(v["alpha"])
    dm = abs(R.M_MU / _lepton_ratio(v["alpha"] + v["sigma"]) - mt)
    return {"m_tau": mt, "sigma": dm, "pull_vs_measured": (mt - R.M_TAU) / R.M_TAU_ERR}


def verdict() -> dict:
    m, a = meters(), against_meters()
    out = {"K1_consistent": m["consistent"], "chi2": m["chi2"], "meter_spread": m["spread"],
           "fp_gap_vs_lepton": a["fp_gap"], "gap_over_spread": abs(a["fp_gap"]) / m["spread"]}
    if m["consistent"]:
        out["K2_killed"] = {k: abs((v - m["mean"]) / m["mean_sigma"]) > 3 for k, v in a["candidates"].items()}
    return out


def _r(x, n: int = 7):
    if isinstance(x, float):
        return float(f"{x:.{n}g}")
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    return x


def main() -> None:
    m = meters()
    for k, v in m["meters"].items():
        print(f"{k:16s} alpha = {v['alpha']:.8f} +- {v['sigma']:.2e}   pull vs mean {m['pulls_vs_mean'][k]:+.1f}")
    print("mean", _r(m["mean"], 9), "+-", _r(m["mean_sigma"], 3), "chi2", _r(m["chi2"], 4), "spread", _r(m["spread"], 3))
    a = against_meters()
    print("candidates:", _r(a["candidates"], 9))
    for c, row in a["pulls"].items():
        print(f"  {c:14s}", {k: round(v, 2) for k, v in row.items()})
    print("lepton-rule check:", _r(BLP.alpha_from_lepton_rule(), 8))
    print("verdict:", _r(verdict(), 4))
    e = exploration()
    print("exploration (post hoc):", _r({k: v for k, v in e.items() if k != "pulls_M5"}, 6))
    for k in ("M1 lepton rule", "M2 Koide", "M5 hierarchy"):
        print("tau mass from", k, _r(tau_mass(k), 8))


if __name__ == "__main__":
    main()
