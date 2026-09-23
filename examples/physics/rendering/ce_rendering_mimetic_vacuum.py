"""W3: 진공 조화 항의 작용판 — 모방 시계 φ와 진공→먼지 확정. 원장: 43장 §43.28.

잠긴 모듈(v1–v9)은 import만 한다.

작용: S = ∫√−g [R/2κ + λ(g^{μν}∂φ∂φ + ω²) − V(φ)] + S_m, ω = H_Λ/2,
V = ρ_V0 [1 + ξ² cos φ]/[1 + ξ² cos φ0] (C6 최저 조화, ξ² = α_s^{2/3}).
- λ 제약이 FLRW에서 φ = ω t를 강제한다(§43.16의 회전). φ에 독립 운동 에너지는 없고 λ가 먼지로 행동한다.
- 일반 공변 작용이라 비앙키 항등식이 성립한다: ρ̇_d + 3Hρ_d = −V̇, 진공은 p = −V.
- 탄생(φ = 0)에서 ρ_d = 0. c_1 < 0이면 ρ_d < 0(음의 에너지)이므로 부호는 에너지 양수 조건으로 정해진다.
- c_1 → 0이면 C5·표준 FLRW.
해석 선택 1 bit: (a) CE Ω_m = 오늘의 전체 물질, (b) CE Ω_m = 처음부터 있던 물질(생긴 먼지는 더해짐).
θ*는 같은 코드에서 W3 − Λ의 이동을 ν 장부 경로의 값에 더한다(ν 처리 경로 차이를 상쇄).
성장(S8)은 이 모듈에서 다시 계산하지 않는다.

python -B -m examples.physics.rendering.ce_rendering_mimetic_vacuum
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

C_KM_S = 299792.458
GRID = np.logspace(-6, 0, 200001)


@lru_cache(maxsize=8)
def _profiles(alpha_s: float, xi2_scale: float, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """(V(a)/V0, a³ρ_d(a)/V0) on GRID."""
    c = R.core(alpha_s)
    xi2 = xi2_scale * c["a"] ** (2 / 3)
    phi = 0.5 * np.array([VT.cycle_phase(a, c["Om"]) for a in GRID])
    v = (1 + sign * xi2 * np.cos(phi)) / (1 + sign * xi2 * math.cos(phi[-1]))
    created = -cumulative_trapezoid(GRID ** 3 * np.gradient(v, GRID), GRID, initial=0.0)
    return v, created


def model(reading: str = "b", xi2_scale: float = 1.0, sign: float = 1.0) -> dict:
    a0 = R.calibrated_alpha_s()[0]
    c = R.core(a0)
    v, created = _profiles(a0, xi2_scale, sign)
    om, q = c["Om"], c["q"]
    if reading == "a":
        ov0 = 1 - om
        prim = om - ov0 * created[-1]
    else:
        ov0 = (1 - om) / (1 + created[-1])
        prim = om
    wb, _, h = NL.early_densities(c)
    wc = (prim - q) * h * h - NL.omega_nu_h2(c)
    dark = lambda a: ov0 * np.interp(a, GRID, v) + ov0 * np.interp(a, GRID, created) / np.asarray(a) ** 3
    vac = lambda a: ov0 * np.interp(a, GRID, v)
    return {"c": c, "h": h, "wb": wb, "wc": wc, "prim": prim, "ov0": ov0, "od0": ov0 * created[-1],
            "dark": dark, "vacuum": vac, "min_dust": float(created.min())}


def _theta(m: dict, dark) -> float:
    wb, wc, h, prim = m["wb"], m["wc"], m["h"], m["prim"]
    wm = wb + wc
    hub = lambda z: 100 * h * math.sqrt(wm * (1 + z) ** 3 / h ** 2 + DV.OMEGA_R_H2 * (1 + z) ** 4 / h ** 2
                                        + (prim - wm / h ** 2) * (1 + z) ** 3 + float(dark(1 / (1 + z))))
    zs = DV.z_star(wb, wm)
    cs = lambda z: C_KM_S / math.sqrt(3 * (1 + 3 * wb / (4 * DV.OMEGA_GAMMA_H2) / (1 + z)))
    rs = quad(lambda z: cs(z) / hub(z), zs, np.inf, limit=400)[0]
    dm = quad(lambda z: C_KM_S / hub(z), 0, 10, limit=400)[0] + quad(lambda z: C_KM_S / hub(z), 10, zs, limit=400)[0]
    return 100 * rs / dm


def theta_star_100(m: dict) -> float:
    lam = lambda a: 1 - m["prim"] + 0 * np.asarray(a)
    return DV.theta_star_100(m["wb"], m["wc"], m["h"]) + _theta(m, m["dark"]) - _theta(m, lam)


def bao_vectors(m: dict) -> np.ndarray:
    z = np.linspace(0.0, 2.5, 25001)
    a = 1 / (1 + z)
    E = np.sqrt(m["prim"] * (1 + z) ** 3 + m["dark"](a))
    inv = 1 / E
    dM = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2 * np.diff(z))])
    mm = np.interp(R.BAO_Z, z, dM)
    hh = 1 / np.interp(R.BAO_Z, z, E)
    return np.array([{"dm": mi, "dh": hi, "dv": (zi * mi * mi * hi) ** (1 / 3)}[k]
                     for mi, hi, zi, k in zip(mm, hh, R.BAO_Z, R.BAO_KIND)])


def continuity_residual(m: dict) -> float:
    """max |dρ/dlna + 3(ρ + p)|, ρ = V + ρ_d, p = −V (비앙키)."""
    a = np.linspace(0.2, 1.0, 2001)
    rho, vac = m["dark"](a), m["vacuum"](a)
    return float(np.abs(np.gradient(rho, np.log(a)) + 3 * (rho - vac)).max())


def w_eff(m: dict, z: float, eps: float = 1e-4) -> float:
    """처음부터 있던 물질을 빼고 남은 어두운 부문 전체를 하나의 유체로 볼 때의 w."""
    a = 1 / (1 + z)
    lo, hi = float(m["dark"](a * (1 - eps))), float(m["dark"](a * (1 + eps)))
    return -1 - (math.log(hi) - math.log(lo)) / (3 * (math.log1p(eps) - math.log1p(-eps)))


def growth_today(m: dict | None) -> float:
    """D(a=1), 처음부터 있던 물질과 생긴 먼지가 함께 뭉친다(에너지 전달의 δ 보정은 1차에서 생략). None이면 Λ."""
    if m is None:
        om = R.core(R.calibrated_alpha_s()[0])["Om"]
        e2 = lambda a: om / a ** 3 + 1 - om
        cl = lambda a: om / a ** 3
    else:
        prim, ov0 = m["prim"], m["ov0"]
        dust = lambda a: m["dark"](a) - m["vacuum"](a)
        e2 = lambda a: prim / a ** 3 + float(m["dark"](a))
        cl = lambda a: prim / a ** 3 + float(dust(a))

    def rhs(lna, y):
        a = math.exp(lna)
        eps = 1e-5
        dln = (math.log(e2(a * (1 + eps))) - math.log(e2(a * (1 - eps)))) / (math.log1p(eps) - math.log1p(-eps))
        return [y[1], -(2 + 0.5 * dln) * y[1] + 1.5 * cl(a) / e2(a) * y[0]]
    from scipy.integrate import solve_ivp
    return float(solve_ivp(rhs, (math.log(1e-3), 0.0), [1e-3, 1e-3], rtol=1e-10, atol=1e-14).y[0, -1])


def s8(m: dict) -> float:
    """ν 장부 S8에 성장비를 곱하고 S8 = σ8 √(Ω_m/0.3)의 Ω_m을 오늘의 전체 물질로 바꾼다."""
    om_ce = m["c"]["Om"]
    om_today = m["prim"] + m["od0"]
    return NL.s8(m["c"]) * growth_today(m) / growth_today(None) * math.sqrt(om_today / om_ce)


def score(reading: str = "b", ruler: str = "fixed", sign: float = 1.0, rows_from: str = "IV") -> dict:
    m = model(reading, sign=sign)
    rd = BR.r_drag(m["wb"], m["wc"], m["h"])
    b = bao_vectors(m)
    fisher = float(b @ R.BAO_CINV @ b)
    af = float(b @ R.BAO_CINV @ R.BAO_Y) / fisher
    r = (C_KM_S / (100 * m["h"] * rd) if ruler == "fixed" else af) * b - R.BAO_Y
    chib = float(r @ R.BAO_CINV @ r)
    h0b = C_KM_S / (af * rd)
    rows = [dict(o) for o in NL.score(rows_from, ruler)["rows"]]
    th = theta_star_100(m)
    lens = {n for n, *_ in __import__("examples.physics.rendering.ce_rendering_growth", fromlist=["x"]).LENSING_S8}
    s8v = s8(m) if rows_from == "full" else None
    for o in rows:
        if o["key"] == "100 theta*":
            o["pred"] = th
        elif o["key"] == "omega_c h^2":
            o["pred"] = m["wc"]
        elif o["key"] in lens:
            o["pred"] = s8v
        else:
            continue
        o["pull"] = (o["pred"] - o["obs"]) / o["sigma"]
    n = len(rows) + 13
    return {"reading": reading, "rows": rows, "bao_chi2": chib, "rmse_all": math.sqrt((sum(o["pull"] ** 2 for o in rows) + chib) / n),
            "N": n, "cmb_bao_tension": (h0b - 100 * m["h"]) * math.sqrt(fisher) / h0b * af, "model": m}


def main() -> None:
    for reading in ("a", "b"):
        s = score(reading)
        m = s["model"]
        pulls = {o["key"]: round(o["pull"], 2) for o in s["rows"] if o["key"] in ("100 theta*", "omega_c h^2")}
        print(f"reading {reading}: Omega_d0={m['od0']:.5f} continuity={continuity_residual(m):.1e} {pulls} "
              f"BAO fixed={s['bao_chi2']:.2f} CMB-BAO={s['cmb_bao_tension']:+.2f} V39={s['rmse_all']:.3f} "
              f"free-ruler V39={score(reading, 'free')['rmse_all']:.3f}")
    m = model("b")
    print("w_eff(z=0,0.5,1,2):", [round(w_eff(m, z), 4) for z in (0, 0.5, 1, 2)], f"S8 = {s8(m):.4f}")
    for ruler in ("fixed", "free"):
        print(f"43 rows {ruler}: {score('b', ruler, rows_from='full')['rmse_all']:.3f}")
    print("sign -1: min created dust a^3 rho_d/V0 =", f"{model('b', sign=-1.0)['min_dust']:.2e} (negative energy -> forbidden)")


if __name__ == "__main__":
    main()
