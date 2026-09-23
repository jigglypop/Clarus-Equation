"""열적 시간과 우주 순환 — 인과가 지평선의 온도로 쌓이다가 흐려지고 다시 시작한다. 원장: 43장 §43.53.

사용자 가설: “우주가 점점 인과율이 쌓이다가 어느 순간 다시 흐려져 버리고 다시 시작되는 우주 순환.” 예측값을 바꾸지 않는다.
계산 전에 적은 판본과 kill:

전제 TT(열적 시간, Connes–Rovelli). 기록 위상 Φ는 렌더링 지평선 상태의 모듈러 흐름으로 쌓인다:
   dΦ/dt = 2π k_B T_H / ħ. H1(§43.51): 기울기 θ = Φ/2(진폭 = 확률 위상의 절반).
V-dS  T_H = 최종(드 시터) 지평선: dΦ/dt = H_Λ → Φ = H_Λ t. 오늘의 직접 판독도 실제 위상 Φ₀/2로(완전 통일판).
V-EH  T_H = 실제 사건 지평선: dΦ/dt = c / r_e(t), r_e = a ∫_t^∞ c dt'/a.
V-AH  T_H = 겉보기 지평선(dΦ/dt = H(t)): 탄생에서 발산하므로 구성상 제외.
kill. 판본 V > 0.909 또는 H₀ 행 |pull| > 3.

python -B -m examples.physics.rendering.ce_rendering_thermal_time
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

GYR_PER_INV_KMS_MPC = 977.79
HBAR_S = 1.054571817e-34
K_B = 1.380649e-23


def _cosmo():
    c = R.core(R.calibrated_alpha_s()[0])
    h = NL.early_densities(c)[2]
    om = c["Om"]
    h0 = 100 * h / GYR_PER_INV_KMS_MPC                     # 1/Gyr
    return c, om, h0, h0 * math.sqrt(1 - om)


def phase_histories(n: int = 20001) -> dict:
    """a 격자에서 t(a), Φ_dS(a) = H_Λ t, Φ_EH(a) = ∫ c/r_e dt. 단위 Gyr, c = 1 (Gly/Gyr)."""
    _, om, h0, hl = _cosmo()
    hub = lambda a: h0 * math.sqrt(om / a ** 3 + 1 - om)
    a = np.logspace(-6, 0, n)
    t = cumulative_trapezoid(1 / (a * np.array([hub(x) for x in a])), a, initial=0.0)
    t += quad(lambda x: 1 / (x * hub(x)), 0, a[0])[0]
    chi_e = np.array([quad(lambda x: 1 / (x * x * hub(x)), ai, np.inf, limit=200)[0] for ai in a])
    r_e = a * chi_e
    dphi_dt = 1 / r_e
    phi_eh = cumulative_trapezoid(dphi_dt, t, initial=0.0)
    phi_eh += dphi_dt[0] * t[0] * 2                        # 앞쪽 ∝ t^{-1/3}에 가까운 적분의 근사 보정(무시 가능)
    return {"a": a, "t": t, "phi_dS": hl * t, "phi_EH": phi_eh, "r_e_today": float(r_e[-1]), "c_over_HL": 1 / hl}


def score(variant: str) -> dict:
    c, om, _, hl = _cosmo()
    ring = 100 * NL.early_densities(c)[2]
    hist = phase_histories()
    a_bao = 1 / (1 + R.BAO_Z)
    phi = hist["phi_dS"] if variant == "V-dS" else hist["phi_EH"]
    th_z = np.interp(a_bao, hist["a"], phi) / 2
    th0 = float(phi[-1]) / 2
    direct = ring / math.cos(th0)
    factor = 1 - om * (1 - np.cos(th_z))
    rd, h = NL.rd_and_h(c)
    b = VT.bao_vectors(om, VT.density(c, VT.ADOPTED_NU))
    r = GD.C_KM_S / (100 * h * rd) * b * factor - R.BAO_Y
    bao = float(r @ R.BAO_CINV @ r)
    base = VT.score(VT.ADOPTED_NU)
    chi, pulls = 0.0, {}
    for o in base["rows"]:
        p = o["pull"]
        if o["key"].startswith("H0 "):
            p = (direct - o["obs"]) / o["sigma"]
            pulls[o["key"]] = round(p, 2)
        chi += p ** 2
    v39 = math.sqrt((chi + bao) / base["N"])
    return {"phi_today": 2 * th0, "theta_today": th0, "theta_bao": (float(th_z.min()), float(th_z.max())),
            "direct_H0": direct, "h0_pulls": pulls, "bao_chi2": bao, "V39": v39,
            "kill": v39 > 0.909 or any(abs(v) > 3 for v in pulls.values())}


def cycle_timetable() -> dict:
    """V-dS의 순환: 렌더링(Φ < π/2), 흐려짐(π/2–3π/2), 재생, 재시작(Φ = 2π = 지평선 열적 주기)."""
    c, om, _, hl = _cosmo()
    t_half = (math.pi / 4) / hl
    temp = HBAR_S * (hl / (1e9 * 365.25 * 86400)) / (2 * math.pi * K_B)
    hl_t0 = 2 / 3 * math.asinh(math.sqrt((1 - om) / om))
    return {"T_horizon_K": temp, "half_point_Gyr": t_half, "today_Gyr": hl_t0 / hl, "blur_Gyr": (math.pi / 2) / hl,
            "reemerge_Gyr": (3 * math.pi / 2) / hl, "restart_Gyr": 2 * math.pi / hl}


def exact_half_now() -> dict:
    """읽기: H_Λ t₀ = π/4(지금이 정확히 절반 지점)가 요구하는 ΛCDM Ω_m과 여러 측정."""
    f = lambda om: 2 / 3 * math.asinh(math.sqrt((1 - om) / om)) - math.pi / 4
    from scipy.optimize import brentq
    om_star = brentq(f, 0.2, 0.4)
    obs = {"Planck 2018": (0.3153, 0.0073), "DESI DR2 + CMB": (0.3027, 0.0036), "DESI DR2 BAO": (0.2975, 0.0086)}
    c = R.core(R.calibrated_alpha_s()[0])
    return {"Om_star": om_star, "CE_core_Om": c["Om"],
            **{k: round((om_star - v) / s, 2) for k, (v, s) in obs.items()}}


def main() -> None:
    hist = phase_histories()
    print("event horizon today r_e =", round(hist["r_e_today"], 3), "Gly; c/H_L =", round(hist["c_over_HL"], 3), "Gly")
    for v in ("V-dS", "V-EH"):
        print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in score(v).items()})
    print("cycle:", {k: (f"{v:.3e}" if k.startswith("T_") else round(v, 2)) for k, v in cycle_timetable().items()})
    print("exact half now:", {k: round(v, 4) for k, v in exact_half_now().items()})


if __name__ == "__main__":
    main()
