"""경계 한 고리 인자 w의 통일과 형태. 원장: 43장 §43.58. 예측값을 바꾸지 않는다.

w = 1 + δ/2π는 세대 전이(R-δ: |V_us| = 4u/w, |V_cb| = u ε w, m_μ/m_τ = u w)와 고정점의 기운 동전
(a = (α_s/2π) w, §43.57)에 같은 모양으로 나온다. 계산 전에 적은 판본과 kill:

통일 가설 UB. 나/아닌 나 경계를 건너는 모든 양은 같은 경계 한 고리 인자 w로 치장된다.
형태. x = δ/2π에서 선형 1 + x, 복리 e^x, 기하 1/(1 − x). 각 형태를 세대 행과 고정점(ŝ²)에 동시에 적용.
   α_s는 두 방식: (i) E4 보정(ŝ²에서 역산, 기존 채점), (ii) 같은 형태의 고정점(입력 없음).
kill. 선형보다 공동 Δχ² > 9인 형태는 기각.
장부. UB가 서면 고정점의 안쪽 인자는 새 선택이 아니라 R-δ(이미 1 bit)이므로 FP 비용은 log₂6 = 2.6 bit.

python -B -m examples.physics.rendering.ce_rendering_boundary_loop
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_registry as R

FORMS = {"linear": lambda x: 1 + x, "compound": math.exp, "geometric": lambda x: 1 / (1 - x)}
S2_OBS = (0.23129, 0.00004)
KEYS = ("|V_us|", "|V_cb|", "m_mu/m_tau")


def _rows() -> dict:
    return {r.key: r for r in R.rows("SK") if r.key in KEYS}


def _pull(pred: float, row) -> float:
    return (pred - row.obs) / (row.sig_up if pred > row.obs else row.sig_dn)


def fixed_point_alpha(form: str) -> float:
    w = FORMS[form]

    def g(r):
        s2 = 4 * r ** 4
        d = s2 * (1 - s2)
        return 0.5 * (1 - r ** 3 / (2 * math.pi) * w(d / (2 * math.pi)))
    r = brentq(lambda x: x - g(x), 0.3, 0.5)
    return r ** 3


def flavour_predictions(alpha_s: float, form: str) -> dict:
    s2 = 4 * alpha_s ** (4 / 3)
    d = s2 * (1 - s2)
    w = FORMS[form](d / (2 * math.pi))
    u, eps = alpha_s ** (4 / 3), alpha_s ** (1 / 6)
    return {"|V_us|": 4 * u / w, "|V_cb|": u * eps * w, "m_mu/m_tau": u * w, "s2": s2}


def score(form: str, alpha_source: str) -> dict:
    a = R.calibrated_alpha_s()[0] if alpha_source == "E4" else fixed_point_alpha(form)
    p = flavour_predictions(a, form)
    rows = _rows()
    pulls = {k: _pull(p[k], rows[k]) for k in KEYS}
    if alpha_source == "FP":
        pulls["s2"] = (p["s2"] - S2_OBS[0]) / S2_OBS[1]
    return {"alpha_s": a, "pulls": pulls, "chi2": sum(v * v for v in pulls.values())}


def alpha_free_relation() -> dict:
    """α_s가 없는 관계 m_μ/m_τ = (ŝ²/4)·w(δ(ŝ²)). 실험 오차만(m_μ/m_τ, ŝ²)으로 형태별 pull."""
    row = _rows()["m_mu/m_tau"]
    s2 = S2_OBS[0]
    d = s2 * (1 - s2)
    sig_rel = math.hypot(row.sig_up / row.obs, S2_OBS[1] / s2)
    out = {}
    for f, w in FORMS.items():
        pred = s2 / 4 * w(d / (2 * math.pi))
        out[f] = {"pred": pred, "rel_dev": pred / row.obs - 1, "pull": (pred / row.obs - 1) / sig_rel}
    return out


def alpha_from_lepton_rule(form: str = "linear") -> dict:
    """렙톤 규칙 m_μ/m_τ = α_s^{4/3} w(δ(α_s))를 풀어 얻은 α_s와 실험 오차, 고정점 α_s와의 pull."""
    row = _rows()["m_mu/m_tau"]

    def f(a):
        s2 = 4 * a ** (4 / 3)
        d = s2 * (1 - s2)
        return a ** (4 / 3) * FORMS[form](d / (2 * math.pi)) - row.obs
    a = brentq(f, 0.10, 0.13)
    sig = a * 0.75 * row.sig_up / row.obs
    fp = fixed_point_alpha(form)
    cal = R.calibrated_alpha_s()
    return {"alpha_s_lepton": a, "sigma": sig, "alpha_s_E4": cal[0], "sigma_E4": cal[1],
            "E4_vs_lepton": (cal[0] - a) / math.hypot(sig, cal[1]), "alpha_s_FP": fp, "FP_vs_lepton": (fp - a) / sig}


def verdict() -> dict:
    out = {}
    for src in ("E4", "FP"):
        base = score("linear", src)["chi2"]
        out[src] = {f: {"chi2": score(f, src)["chi2"], "d_chi2": score(f, src)["chi2"] - base,
                        "killed": score(f, src)["chi2"] - base > 9} for f in FORMS}
    return out


def main() -> None:
    for src in ("E4", "FP"):
        for f in FORMS:
            s = score(f, src)
            print(f"[{src}] {f:9s} alpha_s={s['alpha_s']:.6f} chi2={s['chi2']:.2f}",
                  {k: round(v, 2) for k, v in s["pulls"].items()})
    print("verdict:", {src: {f: (round(v["d_chi2"], 2), v["killed"]) for f, v in d.items()} for src, d in verdict().items()})
    print("alpha-free relation:", {f: {k: (f"{v:.3e}" if k == "rel_dev" else round(v, 3)) for k, v in d.items()}
                                   for f, d in alpha_free_relation().items()})
    print("alpha_s via lepton rule:", {k: (f"{v:.7f}" if k.startswith(("alpha", "sigma")) else round(v, 2))
                                       for k, v in alpha_from_lepton_rule().items()})


if __name__ == "__main__":
    main()
