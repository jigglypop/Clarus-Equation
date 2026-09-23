"""판본 IV: 플랑크 단위 판독 규칙과 유클리드 회전. 원장: 43장 §43.12, 사전 등록 v3.

v1(레지스트리)과 v2(유도 검사) 모듈은 해시로 잠겨 있어 import만 한다.

R-Pl  플랑크 척도 대비로 읽는 무차원 양은 모두 Λ^odd 4통로 고리 (1 + α_s/4π)를 받는다.
      v/M_Pl에서 정한 규칙을 지평선 판독 H0·t_Pl로 옮긴 전이 시험이며 새 상수는 없다.
C2    O1의 기울기는 유클리드 드 시터(4-구면) 안의 회전이다. 두 회전 생성자(기록 틀과 현재 관측자 틀)
      사이의 각이 π/8이면 정규화한 킬링 내적은 cos(π/8)이고, 로렌츠 쪽에서는 허수 빠르기 iπ/8의
      cosh(iπ/8) = cos(π/8)로 나타난다. 실수 속도는 생기지 않는다.

python -B -m examples.physics.rendering.ce_rendering_planck_readout
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_registry as R

DESI_BAO_BBN_H0, DESI_BAO_BBN_ERR = 68.51, 0.58


def planck_unit_factor(c: dict) -> float:
    return 1.0 + c["a"] / (4.0 * math.pi)


def h_rings(c: dict) -> float:
    """R-Pl을 적용한 지평선 판독의 h (적합 없음)."""
    return R.hubble_readout(c, False) / 100.0 * planck_unit_factor(c)


def score_variant_iv(pmns: str = "SK") -> dict:
    """판본 III에서 θ* 적합 h를 R-Pl의 h로 바꾸고 θ*를 채점 행으로 되돌린다."""
    base = R.score("I", pmns)
    a0, sa = R.calibrated_alpha_s()
    drop = {"Omega_b", "Omega_m", "H0 Planck", "H0 TDCOSMO", "H0 TRGB", "H0 SH0ES"}
    rows = [o for o in base["rows"] if o["key"] not in drop]

    def add(key, f, obs, up, dn, note, bits=0.0):
        pred = f(a0)
        st = abs(f(a0 + sa) - f(a0 - sa)) / 2
        sig = math.hypot(up if pred >= obs else dn, st)
        rows.append({"key": key, "block": "M", "pred": pred, "obs": obs, "sigma": sig,
                     "pull": (pred - obs) / sig, "status": "산출", "bits": bits, "note": note})

    hr = lambda a: h_rings(R.core(a))
    theta_ref = DV.theta_star_100(*DV.PLANCK_BEST)
    add("100 theta*", lambda a: DV.theta_star_100(R.core(a)["q"] * hr(a) ** 2,
                                                  (R.core(a)["Om"] - R.core(a)["q"]) * hr(a) ** 2, hr(a)),
        theta_ref, DV.THETA_SIGMA, DV.THETA_SIGMA, "R-Pl h, 같은 코드의 Planck 기준", 1.6)
    add("omega_b h^2", lambda a: R.core(a)["q"] * hr(a) ** 2, DV.OMEGA_B_H2_OBS, DV.OMEGA_B_H2_ERR,
        DV.OMEGA_B_H2_ERR, "q h^2")
    add("omega_c h^2", lambda a: (R.core(a)["Om"] - R.core(a)["q"]) * hr(a) ** 2, DV.OMEGA_C_H2_OBS,
        DV.OMEGA_C_H2_ERR, DV.OMEGA_C_H2_ERR, "Ω_DM h^2")
    for name, direct, v, up, dn in R.H0_READOUTS:
        if direct:
            add(name, lambda a: 100 * hr(a) / math.cos(R.TIME_AXIS_TILT), v, up, dn, "h/cos(π/8)")
    q = [o for o in rows if o["block"] == "Q"]
    m = [o for o in rows if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + base["bao_chi2"]
    nq, nm = len(q), len(m) + 13
    return {"variant": "IV", "pmns": pmns, "rows": rows, "h": h_rings(R.core(a0)),
            "rmse_Q": math.sqrt(cq / nq), "rmse_M": math.sqrt(cm / nm),
            "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "N": nq + nm,
            "k_continuous": base["k_continuous"], "bits": sum(o["bits"] for o in rows)}


def killing_alignment(phi: float) -> float:
    """C2: R^5에 묻은 4-구면에서 (X0,X1) 회전과 X0를 X2 쪽으로 phi만큼 돌린 (X0',X1) 회전의
    정규화한 생성자 내적. 값은 cos(phi)."""
    def gen(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.outer(u, v) - np.outer(v, u)
    e = np.eye(5)
    a = gen(e[0], e[1])
    b = gen(math.cos(phi) * e[0] + math.sin(phi) * e[2], e[1])
    return float(np.sum(a * b) / math.sqrt(np.sum(a * a) * np.sum(b * b)))


def imaginary_rapidity_factor(phi: float) -> complex:
    """cosh(iφ) = cos φ: 유클리드 회전각 φ는 로렌츠 쪽의 허수 빠르기다."""
    return cmath.cosh(1j * phi)


def main() -> None:
    a0 = R.calibrated_alpha_s()[0]
    c = R.core(a0)
    h = h_rings(c)
    print(f"R-Pl: h = {h:.5f}, θ* pull {DV.ce_theta_pull(c, h):+.2f}, rings H0 {100 * h:.2f}, "
          f"direct H0 {100 * h / math.cos(R.TIME_AXIS_TILT):.2f}, DESI BAO+BBN pull "
          f"{(100 * h - DESI_BAO_BBN_H0) / DESI_BAO_BBN_ERR:+.2f}")
    print(f"C2: Killing alignment at π/8 = {killing_alignment(math.pi / 8):.12f}, cos(π/8) = {math.cos(math.pi / 8):.12f}, "
          f"cosh(iπ/8) = {imaginary_rapidity_factor(math.pi / 8)}")
    for pm in ("SK", "noSK"):
        res = score_variant_iv(pm)
        big = [(o["key"], round(o["pull"], 2)) for o in res["rows"] if abs(o["pull"]) > 1.5]
        print(f"[IV {pm}] RMSE Q={res['rmse_Q']:.3f} M={res['rmse_M']:.3f} ALL={res['rmse_all']:.3f} "
              f"N={res['N']} k={res['k_continuous']} bits={res['bits']:.1f} {big}")


if __name__ == "__main__":
    main()
