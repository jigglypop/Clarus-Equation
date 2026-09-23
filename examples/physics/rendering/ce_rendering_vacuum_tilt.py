"""W2: 진공 판독이 기록 틀과 현재 틀 사이의 기울기를 따라 변한다. 원장: 43장 §43.26, 사전 등록 v9.

잠긴 모듈(v1–v8)은 import만 한다.

사전 규칙(계산 전 고정): ρ_DE(t)/ρ_DE(t0) = [1 + ξ² cos(ν H_Λ t)] / [1 + ξ² cos(ν H_Λ t0)],
ξ² = α_s^{2/3}(W1과 같은 진폭), 탄생 t = 0에서 위상 0, ν ∈ {1/2(C3 기울기 속도), 1(순환 원), 3(세 축)}.
연속 매개변수 0, ν 선택 1.6 bit. 채택 조건은 §43.25와 같다(판본 V RMSE 감소, 새 3σ 행 없음, CMB–BAO 감소).
ν = 1/2이면 각 ν H_Λ t = H_Λ t / 2가 C3 접선–현 기울기다. 오늘의 Ω_Λ = 1 − Ω_m은 바뀌지 않는다.
H_Λ t(a)는 물질+Λ 배경으로 계산한다(1차 근사).

python -B -m examples.physics.rendering.ce_rendering_vacuum_tilt
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.integrate import quad, solve_ivp

from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R

C_KM_S = 299792.458
NUS = (0.5, 1.0, 3.0)
ADOPTED_NU = 0.5
BITS = math.log2(len(NUS))


def cycle_phase(a: float, om: float) -> float:
    """H_Λ t(a), 평탄 물질+진공 배경."""
    return (2 / 3) * math.asinh(math.sqrt((1 - om) / om) * a ** 1.5)


def density(c: dict, nu: float | None) -> Callable[[float], float]:
    """ρ_DE(a)/ρ_DE(1). nu=None이면 Λ."""
    if nu is None:
        return lambda a: 1.0
    om, xi2 = c["Om"], c["a"] ** (2 / 3)
    norm = 1 + xi2 * math.cos(nu * cycle_phase(1.0, om))
    return lambda a: (1 + xi2 * math.cos(nu * cycle_phase(a, om))) / norm


def w_of(c: dict, nu: float | None, z: float, eps: float = 1e-5) -> float:
    f, a = density(c, nu), 1 / (1 + z)
    return -1 - (math.log(f(a * (1 + eps))) - math.log(f(a * (1 - eps)))) / (3 * (math.log1p(eps) - math.log1p(-eps)))


def theta_star_100(wb: float, wc: float, h: float, f: Callable[[float], float]) -> float:
    wm = wb + wc
    ol = 1 - (wm + DV.OMEGA_R_H2) / h ** 2
    hub = lambda z: 100 * h * math.sqrt((wm * (1 + z) ** 3 + DV.OMEGA_R_H2 * (1 + z) ** 4) / h ** 2 + ol * f(1 / (1 + z)))
    zs = DV.z_star(wb, wm)
    cs = lambda z: C_KM_S / math.sqrt(3 * (1 + 3 * wb / (4 * DV.OMEGA_GAMMA_H2) / (1 + z)))
    rs = quad(lambda z: cs(z) / hub(z), zs, np.inf, limit=400)[0]
    dm = quad(lambda z: C_KM_S / hub(z), 0, 10, limit=400)[0] + quad(lambda z: C_KM_S / hub(z), 10, zs, limit=400)[0]
    return 100 * rs / dm


def bao_vectors(om: float, f: Callable[[float], float]) -> np.ndarray:
    z = np.linspace(0.0, 2.5, 25001)
    E = np.sqrt(om * (1 + z) ** 3 + (1 - om) * np.array([f(1 / (1 + zi)) for zi in z]))
    inv = 1 / E
    dM = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2 * np.diff(z))])
    m = np.interp(R.BAO_Z, z, dM)
    hh = 1 / np.interp(R.BAO_Z, z, E)
    return np.array([{"dm": mi, "dh": hi, "dv": (zi * mi * mi * hi) ** (1 / 3)}[k]
                     for mi, hi, zi, k in zip(m, hh, R.BAO_Z, R.BAO_KIND)])


def growth_today(om: float, f: Callable[[float], float]) -> float:
    e2 = lambda a: om / a ** 3 + (1 - om) * f(a)

    def rhs(lna, y):
        a = math.exp(lna)
        eps = 1e-6
        dln = (math.log(e2(a * (1 + eps))) - math.log(e2(a * (1 - eps)))) / (math.log1p(eps) - math.log1p(-eps))
        return [y[1], -(2 + 0.5 * dln) * y[1] + 1.5 * om / (a ** 3 * e2(a)) * y[0]]
    return float(solve_ivp(rhs, (math.log(1e-3), 0.0), [1e-3, 1e-3], rtol=1e-10, atol=1e-14).y[0, -1])


def s8(c: dict, nu: float | None) -> float:
    return NL.s8(c) * growth_today(c["Om"], density(c, nu)) / growth_today(c["Om"], density(c, None))


def branch_rows(c: dict, nu: float | None) -> dict:
    f = density(c, nu)
    wb, wc, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    b = bao_vectors(c["Om"], f)
    r = C_KM_S / (100 * h * rd) * b - R.BAO_Y
    fisher = float(b @ R.BAO_CINV @ b)
    af = float(b @ R.BAO_CINV @ R.BAO_Y) / fisher
    rf = af * b - R.BAO_Y
    h0b = C_KM_S / (af * rd)
    return {"theta_100": theta_star_100(wb, wc, h, f), "bao_chi2_fixed": float(r @ R.BAO_CINV @ r),
            "bao_chi2_free": float(rf @ R.BAO_CINV @ rf),
            "cmb_bao_tension": (h0b - 100 * h) * math.sqrt(fisher) / h0b * af, "S8": s8(c, nu)}


def score(nu: float | None, rows_from: str = "IV", ruler: str = "fixed", pmns: str = "SK") -> dict:
    """ν 장부 행에서 θ*·S8·BAO를 W2로 바꿔 다시 채점. α_s 흔들림 폭은 ν 장부 행의 σ를 그대로 쓴다."""
    c = R.core(R.calibrated_alpha_s()[0])
    br = branch_rows(c, nu)
    base = NL.score(rows_from, ruler, pmns)
    rows = [dict(o) for o in base["rows"]]
    for o in rows:
        if o["key"] == "100 theta*":
            o["pred"] = br["theta_100"]
        elif o["key"] in {n for n, *_ in GR.LENSING_S8}:
            o["pred"] = br["S8"]
        else:
            continue
        o["pull"] = (o["pred"] - o["obs"]) / o["sigma"]
    chib = br["bao_chi2_fixed"] if ruler == "fixed" else br["bao_chi2_free"]
    q = [o for o in rows if o["block"] == "Q"]
    m = [o for o in rows if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + chib
    nq, nm = len(q), len(m) + 13
    return {"rows": rows, "branch": br, "rmse_Q": math.sqrt(cq / nq), "rmse_M": math.sqrt(cm / nm),
            "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "chi2": cq + cm, "N": nq + nm,
            "k_continuous": base["k_continuous"], "bits": BITS if nu is not None else 0.0}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    om = c["Om"]
    print(f"xi^2 = {c['a'] ** (2 / 3):.4f}; H_L t window z=2.5..0: {cycle_phase(1 / 3.5, om):.3f}..{cycle_phase(1, om):.3f}")
    for nu in (None,) + NUS:
        br = branch_rows(c, nu)
        ws = [round(w_of(c, nu, z), 4) for z in (0, 0.5, 1, 2)]
        th = (br["theta_100"] - DV.theta_star_100(*DV.PLANCK_BEST)) / DV.THETA_SIGMA
        print(f"nu={nu}: w(0,.5,1,2)={ws} theta={th:+.2f} BAO fixed={br['bao_chi2_fixed']:.2f} "
              f"free={br['bao_chi2_free']:.2f} CMB-BAO={br['cmb_bao_tension']:+.2f} S8={br['S8']:.4f}")
        for rows_from in ("IV", "full"):
            for ruler in ("fixed", "free"):
                s = score(nu, rows_from, ruler)
                print(f"   [{rows_from} {ruler}] N={s['N']} RMSE={s['rmse_all']:.3f} chi2={s['chi2']:.2f}")


if __name__ == "__main__":
    main()
