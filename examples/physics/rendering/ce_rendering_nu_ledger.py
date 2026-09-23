"""중성미자 장부 정정: CE 예측 중성미자(P17)를 이른 우주의 차가운 물질에서 뺀다. 원장: 43장 §43.24, 사전 등록 v8.

잠긴 모듈(v1–v7)은 import만 한다.

CE의 Ω_m = q + Ω_DM은 오늘의 전체 물질이다. 그런데 비교 대상인 Planck의 ω_c = 0.1200은 차가운 암흑물질만 센
값이고(0.06 eV 중성미자는 따로), θ*·r_d·σ8의 같은 코드 교정 기준도 그 값이다. CE의 중성미자 합 59.16 meV는
재결합·끌림 시기(T_ν ≈ 0.2 eV)에 상대론적이라 차가운 물질이 아니다. 따라서 모든 이른 우주 행에서
ω_c = Ω_DM h² − ω_ν, ω_ν = Σm_ν / 93.14 eV로 센다. 늦은 우주(BAO 거리 모양, Ω_m)는 그대로다.
연속 매개변수는 늘지 않는다. 긴장을 보고 나서 찾은 정정이므로 그 사실을 기록한다.

python -B -m examples.physics.rendering.ce_rendering_nu_ledger
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_neutrino as NU
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R

NU_EV_PER_OMEGA_H2 = 93.14


def omega_nu_h2(c: dict) -> float:
    return sum(NU.neutrino_masses_mev(c)) / 1000.0 / NU_EV_PER_OMEGA_H2


def early_densities(c: dict) -> tuple[float, float, float]:
    """(ω_b, ω_c 차가운 물질만, h)."""
    h = PL.h_rings(c)
    return c["q"] * h * h, (c["Om"] - c["q"]) * h * h - omega_nu_h2(c), h


def rd_and_h(c: dict) -> tuple[float, float]:
    wb, wc, h = early_densities(c)
    return BR.r_drag(wb, wc, h), h


def s8(c: dict) -> float:
    wb, wc, h = early_densities(c)
    raw = GR.sigma8_raw(R.scalar_amplitude(c) * 1e-9, 1 - 2 / c["Ne"], wb, wc, h)
    return raw * GR.PLANCK_SIGMA8 / GR.sigma8_raw(*GR.PLANCK_PRIMARY) * math.sqrt(c["Om"] / 0.3)


def bao_chi2_fixed_ruler(c: dict) -> float:
    rd, h = rd_and_h(c)
    r = BR.C_KM_S / (100 * h * rd) * BR.bao_vectors(c["Om"]) - R.BAO_Y
    return float(r @ R.BAO_CINV @ r)


def h0_from_bao(c: dict) -> tuple[float, float]:
    A, sA = BR.bao_ruler_fit(c["Om"])
    rd, _ = rd_and_h(c)
    h0 = BR.C_KM_S / (A * rd)
    return h0, h0 * sA / A


def _rescore(rows: list[dict]) -> None:
    a0, sa = R.calibrated_alpha_s()
    f = {"100 theta*": (lambda a: DV.theta_star_100(*early_densities(R.core(a))), DV.THETA_SIGMA, DV.THETA_SIGMA),
         "omega_c h^2": (lambda a: early_densities(R.core(a))[1], DV.OMEGA_C_H2_ERR, DV.OMEGA_C_H2_ERR)}
    for name, _, up, dn in GR.LENSING_S8:
        f[name] = (lambda a: s8(R.core(a)), up, dn)
    for o in rows:
        if o["key"] in f:
            g, up, dn = f[o["key"]]
            pred = g(a0)
            st = abs(g(a0 + sa) - g(a0 - sa)) / 2
            sig = math.hypot(up if pred >= o["obs"] else dn, st)
            o.update(pred=pred, sigma=sig, pull=(pred - o["obs"]) / sig, note=o["note"] + "; ν 장부")


def score(rows_from: str = "IV", ruler: str = "free", pmns: str = "SK") -> dict:
    """rows_from: 'IV'(39행) 또는 'full'(43행). ruler: 'free'(BAO 눈금 적합) 또는 'fixed'(CE r_d)."""
    base = PL.score_variant_iv(pmns) if rows_from == "IV" else NU.score_full(pmns)
    rows = [dict(o) for o in base["rows"]]
    _rescore(rows)
    c = R.core(R.calibrated_alpha_s()[0])
    chib = bao_chi2_fixed_ruler(c) if ruler == "fixed" else R.bao_chi2(c["Om"])
    q = [o for o in rows if o["block"] == "Q"]
    m = [o for o in rows if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + chib
    nq, nm = len(q), len(m) + 13
    return {"rows": rows, "bao_chi2": chib, "rmse_Q": math.sqrt(cq / nq), "rmse_M": math.sqrt(cm / nm),
            "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "N": nq + nm,
            "k_continuous": base["k_continuous"] - (1 if ruler == "fixed" else 0)}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    wb, wc, h = early_densities(c)
    rd, _ = rd_and_h(c)
    h0, s = h0_from_bao(c)
    print(f"omega_nu h^2 = {omega_nu_h2(c):.6f}; omega_c = {wc:.5f}; r_d = {rd:.3f} Mpc; h r_d = {h * rd:.3f} Mpc")
    print(f"H0 from BAO = {h0:.3f} +/- {s:.3f} vs {100 * h:.3f} ({(h0 - 100 * h) / s:+.2f} sigma); S8 = {s8(c):.4f}")
    for rows_from in ("IV", "full"):
        for ruler in ("free", "fixed"):
            r = score(rows_from, ruler)
            print(f"[{rows_from} {ruler}] N={r['N']} k={r['k_continuous']} BAO chi2={r['bao_chi2']:.2f} "
                  f"RMSE Q={r['rmse_Q']:.3f} M={r['rmse_M']:.3f} ALL={r['rmse_all']:.3f}")
    for o in score("full", "fixed")["rows"]:
        if "ν 장부" in o["note"]:
            print(f"  {o['key']}: {o['pred']:.5f} vs {o['obs']} ({o['pull']:+.2f})")


if __name__ == "__main__":
    main()
