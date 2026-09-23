"""동결된 v1 레지스트리 위의 후속 검사와 교정 판본. 원장: 43장 §43.11.

v1(`ce_rendering_registry.py`)은 사전 등록 해시로 잠겨 있으므로 수정하지 않고 import만 한다.

1. 음향 각도 θ* = r_s(z*)/D_M(z*): CE의 (Ω_b, Ω_m, H_CE)를 Planck 최적값과 같은 코드로 비교한다.
   z*는 Hu–Sugiyama 적합식이며 절대값이 아니라 상대 차이만 판정에 쓴다.
2. 판본 III: 독립 행으로 두었던 Planck Ω_b, Ω_m, H0를 θ*에 맞춘 h 하나로 대체한다
   (연속 적합 +1). 직접 판독은 h_θ/cos(π/8), 비율 1/cos(π/8)는 적합 없이 예측된다.

python -B -m examples.physics.rendering.ce_rendering_derivations
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_registry as R

C_KM_S = 299792.458
OMEGA_GAMMA_H2 = 2.469e-5
N_EFF = 3.044
OMEGA_R_H2 = OMEGA_GAMMA_H2 * (1.0 + 0.2271 * N_EFF)
PLANCK_BEST = (0.02237, 0.1200, 0.6736)   # Planck 2018 TT,TE,EE+lowE+lensing
THETA_SIGMA = 0.00031                      # σ(100 θ*)
OMEGA_B_H2_OBS, OMEGA_B_H2_ERR = 0.02237, 0.00015
OMEGA_C_H2_OBS, OMEGA_C_H2_ERR = 0.1200, 0.0012


def z_star(wb: float, wm: float) -> float:
    g1 = 0.0783 * wb ** -0.238 / (1 + 39.5 * wb ** 0.763)
    g2 = 0.560 / (1 + 21.1 * wb ** 1.81)
    return 1048 * (1 + 0.00124 * wb ** -0.738) * (1 + g1 * wm ** g2)


@lru_cache(maxsize=4096)
def theta_star_100(wb: float, wc: float, h: float) -> float:
    wm = wb + wc
    ol = 1.0 - (wm + OMEGA_R_H2) / h ** 2

    def hub(z: float) -> float:
        return 100 * h * math.sqrt((wm * (1 + z) ** 3 + OMEGA_R_H2 * (1 + z) ** 4) / h ** 2 + ol)

    zs = z_star(wb, wm)
    cs = lambda z: C_KM_S / math.sqrt(3 * (1 + 3 * wb / (4 * OMEGA_GAMMA_H2) / (1 + z)))
    rs = quad(lambda z: cs(z) / hub(z), zs, np.inf, limit=400)[0]
    dm = quad(lambda z: C_KM_S / hub(z), 0, zs, limit=400)[0]
    return 100 * rs / dm


def ce_theta_pull(c: dict, h: float | None = None) -> float:
    """CE의 Ω 비율과 h(기본: 지평선 판독 H_CE)로 계산한 θ*의 Planck 대비 잔차(σ)."""
    hh = R.hubble_readout(c, False) / 100 if h is None else h
    wb, wc = c["q"] * hh * hh, (c["Om"] - c["q"]) * hh * hh
    return (theta_star_100(wb, wc, hh) - theta_star_100(*PLANCK_BEST)) / THETA_SIGMA


@lru_cache(maxsize=4096)
def h_from_theta(alpha_s: float) -> float:
    """CE의 Ω 비율을 고정하고 θ*를 Planck 값에 맞추는 h(연속 적합 1개)."""
    c = R.core(alpha_s)
    target = theta_star_100(*PLANCK_BEST)
    return brentq(lambda hh: theta_star_100(c["q"] * hh * hh, (c["Om"] - c["q"]) * hh * hh, hh) - target,
                  0.55, 0.85, xtol=1e-12)


def score_variant_iii(pmns: str = "SK") -> dict:
    """판본 I의 행에서 Planck Ω_b·Ω_m·H0를 θ*-교정 ω_b·ω_c로 바꾸고 직접 판독을 h_θ로 옮긴다."""
    base = R.score("I", pmns)
    a0, sa = R.calibrated_alpha_s()
    drop = {"Omega_b", "Omega_m", "H0 Planck", "H0 TDCOSMO", "H0 TRGB", "H0 SH0ES"}
    rows = [o for o in base["rows"] if o["key"] not in drop]

    def h_of(a: float) -> float:
        return h_from_theta(a)

    def add(key, f, obs, up, dn, note):
        pred = f(a0)
        st = abs(f(a0 + sa) - f(a0 - sa)) / 2
        sig = math.hypot(up if pred >= obs else dn, st)
        rows.append({"key": key, "block": "M", "pred": pred, "obs": obs, "sigma": sig,
                     "pull": (pred - obs) / sig, "status": "산출", "bits": 0.0, "note": note})

    add("omega_b h^2", lambda a: R.core(a)["q"] * h_of(a) ** 2, OMEGA_B_H2_OBS, OMEGA_B_H2_ERR, OMEGA_B_H2_ERR,
        "q h_θ^2")
    add("omega_c h^2", lambda a: (R.core(a)["Om"] - R.core(a)["q"]) * h_of(a) ** 2, OMEGA_C_H2_OBS,
        OMEGA_C_H2_ERR, OMEGA_C_H2_ERR, "Ω_DM h_θ^2")
    for name, direct, v, up, dn in R.H0_READOUTS:
        if not direct:
            continue
        add(name, lambda a: 100 * h_of(a) / math.cos(R.TIME_AXIS_TILT), v, up, dn, "h_θ/cos(π/8)")
    q = [o for o in rows if o["block"] == "Q"]
    m = [o for o in rows if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + base["bao_chi2"]
    nq, nm = len(q), len(m) + 13
    return {"variant": "III", "pmns": pmns, "rows": rows, "h_theta": h_from_theta(a0),
            "rmse_Q": math.sqrt(cq / nq), "rmse_M": math.sqrt(cm / nm),
            "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "N": nq + nm,
            "k_continuous": base["k_continuous"] + 1, "bits": sum(o["bits"] for o in rows)}


CMB_DIPOLE_BETA = 369.82 / C_KM_S   # 태양계의 CMB 대비 속도 / c


def boost_equivalent_beta(tilt: float = R.TIME_AXIS_TILT) -> float:
    """C2 반례 검산: 판독비 1/cos(tilt)를 로렌츠 부스트(cosh η)로 만들 때 필요한 속도 β.

    결과(약 0.38)는 CMB 쌍극자 β≈0.0012와 맞지 않으므로 O1의 기울기는 관측자의 운동이 아니라
    관계공간(허수 시간) 안의 회전이어야 한다.
    """
    eta = math.acosh(1.0 / math.cos(tilt))
    return math.tanh(eta)


def main() -> None:
    a0 = R.calibrated_alpha_s()[0]
    c = R.core(a0)
    print(f"θ* pull with horizon H_CE = {R.hubble_readout(c, False):.3f}: {ce_theta_pull(c):+.1f} σ")
    hth = h_from_theta(a0)
    print(f"h from θ* = {hth:.5f} -> rings H0 {100 * hth:.2f}, direct H0 {100 * hth / math.cos(R.TIME_AXIS_TILT):.2f}")
    for pm in ("SK", "noSK"):
        res = score_variant_iii(pm)
        big = [(o["key"], round(o["pull"], 2)) for o in res["rows"] if abs(o["pull"]) > 1.5]
        print(f"[III {pm}] RMSE Q={res['rmse_Q']:.3f} M={res['rmse_M']:.3f} ALL={res['rmse_all']:.3f} "
              f"N={res['N']} k={res['k_continuous']} bits={res['bits']:.1f} {big}")


if __name__ == "__main__":
    main()
