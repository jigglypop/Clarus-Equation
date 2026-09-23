"""중성미자 질량 행과 43행 공동 채점. 원장: 43장 §43.19, 사전 등록 v7.

잠긴 모듈은 import만 한다.

저장소의 경험식(paper/참조/2_경로적분과_응용/07_중성미자_질량.md §7.3)을 현재 코어로 계산한다:
    m_{ν_l} = δ⁴(1 − α_s/π) / [(16π²)² 32π³ (1 + R)] · m_l^{5/8} m_τ^{3/8},  R = α_s D (현재 코어).
지수 5/8, 3/8은 8통로 비율로 읽을 수 있으나 유도가 아니며, 이 식의 과거 선택은 bit 장부 밖이다.

python -B -m examples.physics.rendering.ce_rendering_neutrino
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_registry as R

DM21_OBS = (7.49e-5, 0.19e-5, 0.19e-5)       # NuFIT 6.0 NO
DM31_OBS = (2.513e-3, 0.021e-3, 0.019e-3)    # NuFIT 6.0 NO, with SK


def neutrino_masses_mev(c: dict) -> tuple[float, float, float]:
    """(m1, m2, m3) in meV, normal ordering."""
    a, d = c["a"], c["d"]
    pref = d ** 4 * (1 - a / math.pi) / ((16 * math.pi ** 2) ** 2 * 32 * math.pi ** 3 * (1 + a * c["D"]))
    return tuple(pref * ml ** (5 / 8) * R.M_TAU ** (3 / 8) * 1e9 for ml in (R.M_E, R.M_MU, R.M_TAU))


def splittings_ev2(c: dict) -> tuple[float, float]:
    m1, m2, m3 = neutrino_masses_mev(c)
    return (m2 ** 2 - m1 ** 2) * 1e-6, (m3 ** 2 - m1 ** 2) * 1e-6


def score_full(pmns: str = "SK") -> dict:
    """판본 IV + 약한 렌즈 2행 + 중성미자 질량 분리 2행 = 43행."""
    base = GR.score_variant_iv_with_lensing(pmns)
    a0, sa = R.calibrated_alpha_s()
    rows = list(base["rows"])
    for idx, (name, obs) in enumerate((("Delta m^2_21", DM21_OBS), ("Delta m^2_31", DM31_OBS))):
        f = lambda a: splittings_ev2(R.core(a))[idx]
        pred = f(a0)
        st = abs(f(a0 + sa) - f(a0 - sa)) / 2
        sig = math.hypot(obs[1] if pred >= obs[0] else obs[2], st)
        rows.append({"key": name, "block": "Q", "pred": pred, "obs": obs[0], "sigma": sig,
                     "pull": (pred - obs[0]) / sig, "status": "경험식", "bits": 0.0, "note": "저장소 식, 과거 선택 장부 밖"})
    q = [o for o in rows if o["block"] == "Q"]
    m = [o for o in rows if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + R.bao_chi2(R.core(a0)["Om"])
    nq, nm = len(q), len(m) + 13
    return {"rows": rows, "rmse_Q": math.sqrt(cq / nq), "rmse_M": math.sqrt(cm / nm),
            "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "N": nq + nm, "k_continuous": base["k_continuous"]}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    m = neutrino_masses_mev(c)
    print(f"masses {tuple(round(x, 3) for x in m)} meV, sum {sum(m):.2f} meV; splittings {splittings_ev2(c)}")
    res = score_full("SK")
    big = [(o["key"], round(o["pull"], 2)) for o in res["rows"] if abs(o["pull"]) > 1.5]
    print(f"[43 rows SK] RMSE Q={res['rmse_Q']:.3f} M={res['rmse_M']:.3f} ALL={res['rmse_all']:.3f} N={res['N']} {big}")


if __name__ == "__main__":
    main()
