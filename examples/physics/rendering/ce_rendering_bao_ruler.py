"""판본 V: BAO 눈금을 적합하지 않고 CE의 음향 눈금으로 예측한다. 원장: 43장 §43.13, 사전 등록 v4.

잠긴 모듈(레지스트리 v1, 유도 v2, 플랑크 판독 v3)은 import만 한다.

BAO 행의 눈금 A = c/(H0 r_d)를 판본 I–IV는 자유롭게 맞췄다. 여기서는 R-Pl의 h와 CE 초기 우주의
(ω_b, ω_c)로 r_d를 계산해 A를 고정한다. r_d는 Eisenstein–Hu 끌림 적색편이와 음속 적분이며,
Planck 최적값의 같은 코드 결과를 147.09 Mpc로 맞추는 상대 교정만 쓴다.

python -B -m examples.physics.rendering.ce_rendering_bao_ruler
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad

from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R

C_KM_S = 299792.458
PLANCK_RD_MPC = 147.09


def z_drag(wb: float, wm: float) -> float:
    b1 = 0.313 * wm ** -0.419 * (1 + 0.607 * wm ** 0.674)
    b2 = 0.238 * wm ** 0.223
    return 1291 * wm ** 0.251 / (1 + 0.659 * wm ** 0.828) * (1 + b1 * wb ** b2)


@lru_cache(maxsize=4096)
def r_drag_raw(wb: float, wc: float, h: float) -> float:
    wm = wb + wc
    ol = 1 - (wm + DV.OMEGA_R_H2) / h ** 2
    hub = lambda z: 100 * h * math.sqrt((wm * (1 + z) ** 3 + DV.OMEGA_R_H2 * (1 + z) ** 4) / h ** 2 + ol)
    cs = lambda z: C_KM_S / math.sqrt(3 * (1 + 3 * wb / (4 * DV.OMEGA_GAMMA_H2) / (1 + z)))
    return quad(lambda z: cs(z) / hub(z), z_drag(wb, wm), np.inf, limit=400)[0]


def r_drag(wb: float, wc: float, h: float) -> float:
    return r_drag_raw(wb, wc, h) * PLANCK_RD_MPC / r_drag_raw(*DV.PLANCK_BEST)


def ce_rd_and_h(c: dict) -> tuple[float, float]:
    h = PL.h_rings(c)
    return r_drag(c["q"] * h * h, (c["Om"] - c["q"]) * h * h, h), h


def bao_vectors(om: float) -> np.ndarray:
    z = np.linspace(0.0, 2.5, 25001)
    E = np.sqrt(om * (1 + z) ** 3 + (1 - om))
    inv = 1.0 / E
    dM = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2.0 * np.diff(z))])
    m = np.interp(R.BAO_Z, z, dM)
    hh = 1.0 / np.interp(R.BAO_Z, z, E)
    return np.array([{"dm": mi, "dh": hi, "dv": (zi * mi * mi * hi) ** (1 / 3)}[k]
                     for mi, hi, zi, k in zip(m, hh, R.BAO_Z, R.BAO_KIND)])


def bao_ruler_fit(om: float) -> tuple[float, float]:
    b = bao_vectors(om)
    fisher = float(b @ R.BAO_CINV @ b)
    return float(b @ R.BAO_CINV @ R.BAO_Y) / fisher, 1.0 / math.sqrt(fisher)


def bao_chi2_fixed_ruler(c: dict) -> float:
    rd, h = ce_rd_and_h(c)
    A = C_KM_S / (100 * h * rd)
    r = A * bao_vectors(c["Om"]) - R.BAO_Y
    return float(r @ R.BAO_CINV @ r)


def h0_from_bao(c: dict) -> tuple[float, float]:
    """BAO가 맞춘 눈금과 CE r_d로 읽은 H0와 그 통계 오차."""
    A, sA = bao_ruler_fit(c["Om"])
    rd, _ = ce_rd_and_h(c)
    h0 = C_KM_S / (A * rd)
    return h0, h0 * sA / A


def score_variant_v(pmns: str = "SK") -> dict:
    iv = PL.score_variant_iv(pmns)
    c = R.core(R.calibrated_alpha_s()[0])
    chib = bao_chi2_fixed_ruler(c)
    q = [o for o in iv["rows"] if o["block"] == "Q"]
    m = [o for o in iv["rows"] if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + chib
    nq, nm = len(q), len(m) + 13
    return {"variant": "V", "pmns": pmns, "bao_chi2": chib, "rmse_Q": math.sqrt(cq / nq),
            "rmse_M": math.sqrt(cm / nm), "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "N": nq + nm,
            "k_continuous": iv["k_continuous"] - 1, "bits": iv["bits"]}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    rd, h = ce_rd_and_h(c)
    h0, s = h0_from_bao(c)
    print(f"CE r_d = {rd:.2f} Mpc, h r_d = {h * rd:.2f} Mpc; H0 from BAO ruler = {h0:.2f} +/- {s:.2f} "
          f"vs rings {100 * h:.2f} ({(h0 - 100 * h) / s:+.2f} σ)")
    for pm in ("SK", "noSK"):
        res = score_variant_v(pm)
        print(f"[V {pm}] BAO chi2 fixed ruler {res['bao_chi2']:.2f}; RMSE Q={res['rmse_Q']:.3f} M={res['rmse_M']:.3f} "
              f"ALL={res['rmse_all']:.3f} N={res['N']} k={res['k_continuous']}")


if __name__ == "__main__":
    main()
