"""물질 뭉침 S8의 무적합 예측과 두 원의 구분. 원장: 43장 §43.17, 사전 등록 v6.

잠긴 모듈(v1–v5)은 import만 한다.

G1  복소 척도인자(§43.16)에서 중력은 |A|만 보므로 밀도 요동의 성장은 표준 그대로다. 약한 렌즈는
    각도의 비라 나이테처럼 회전과 무관하다. CE의 A_s, n_s, ω_b, ω_c, h(R-Pl)로 σ8을 계산한다.
    전달함수는 Eisenstein–Hu(1998) 무진동 형태이며, Planck 최적값의 같은 코드 결과를 σ8 = 0.8111로
    맞추는 상대 교정만 쓴다.
X1  시간축의 돎 φ와 질량 행렬의 순환 위상 θ를 같은 원(θ = 3φ)으로 두면 진공 에너지가 약 31% 줄어
    Ω_Λ가 무너진다. 두 원은 서로 다른 성분이다.

python -B -m examples.physics.rendering.ce_rendering_growth
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from examples.physics.rendering import ce_rendering_cycle as CY
from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R

PLANCK_SIGMA8 = 0.8111
PLANCK_PRIMARY = (2.0989e-9, 0.9649, 0.02237, 0.1200, 0.6736)
LENSING_S8 = (  # (이름, 값, +σ, −σ) — Planck S8은 Planck 행과 겹쳐 제외
    ("S8 KiDS-Legacy", 0.815, 0.016, 0.021),
    ("S8 DES Y3 3x2pt", 0.776, 0.017, 0.017),
)


def transfer_eh(k: np.ndarray, wm: float, wb: float, h: float, tcmb: float = 2.7255) -> np.ndarray:
    om, fb = wm / h ** 2, wb / wm
    theta = tcmb / 2.7
    s = 44.5 * math.log(9.83 / wm) / math.sqrt(1 + 10 * wb ** 0.75)
    alpha = 1 - 0.328 * math.log(431 * wm) * fb + 0.38 * math.log(22.3 * wm) * fb ** 2
    gamma = om * h * (alpha + (1 - alpha) / (1 + (0.43 * k * h * s) ** 4))
    q = k * theta ** 2 / gamma
    l0 = np.log(2 * math.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1 + 62.5 * q)
    return l0 / (l0 + c0 * q ** 2)


def growth_today(om: float) -> float:
    e = lambda a: math.sqrt(om / a ** 3 + 1 - om)
    return 2.5 * om * e(1.0) * quad(lambda a: 1.0 / (a * e(a)) ** 3, 0, 1)[0]


def sigma8_raw(a_s: float, n_s: float, wb: float, wc: float, h: float) -> float:
    wm = wb + wc
    om = wm / h ** 2
    k = np.logspace(-4, 2, 6000)
    kp = 0.05 / h
    dm2 = ((4.0 / 25.0) * (k * 2997.92458) ** 4 / om ** 2 * transfer_eh(k, wm, wb, h) ** 2
           * growth_today(om) ** 2 * a_s * (k / kp) ** (n_s - 1))
    x = k * 8.0
    w = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return math.sqrt(np.trapezoid(dm2 * w ** 2, np.log(k)))


def ce_s8(c: dict) -> tuple[float, float]:
    h = PL.h_rings(c)
    wb, wc = c["q"] * h * h, (c["Om"] - c["q"]) * h * h
    s8 = sigma8_raw(R.scalar_amplitude(c) * 1e-9, 1 - 2 / c["Ne"], wb, wc, h) * PLANCK_SIGMA8 / sigma8_raw(*PLANCK_PRIMARY)
    return s8, s8 * math.sqrt(c["Om"] / 0.3)


def theta_equals_three_phi_vacuum_ratio(c: dict, r: float = 0.35) -> float:
    theta = 3 * CY.tangent_chord_tilt(c["Om"])
    xs = lambda th: [1 + 2 * r * math.cos((th + 2 * math.pi * j) / 3) for j in range(3)]
    f = lambda th: sum(x * x * math.log(x) for x in xs(th))
    return (f(theta) - f(math.pi)) / (f(0.0) - f(math.pi))


def score_variant_iv_with_lensing(pmns: str = "SK") -> dict:
    iv = PL.score_variant_iv(pmns)
    a0, sa = R.calibrated_alpha_s()
    rows = list(iv["rows"])
    for name, v, up, dn in LENSING_S8:
        pred = ce_s8(R.core(a0))[1]
        st = abs(ce_s8(R.core(a0 + sa))[1] - ce_s8(R.core(a0 - sa))[1]) / 2
        sig = math.hypot(up if pred >= v else dn, st)
        rows.append({"key": name, "block": "M", "pred": pred, "obs": v, "sigma": sig, "pull": (pred - v) / sig,
                     "status": "산출", "bits": 0.0, "note": "G1 무적합"})
    q = [o for o in rows if o["block"] == "Q"]
    m = [o for o in rows if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + R.bao_chi2(R.core(a0)["Om"])
    nq, nm = len(q), len(m) + 13
    return {"rows": rows, "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "N": nq + nm,
            "rmse_Q": math.sqrt(cq / nq), "rmse_M": math.sqrt(cm / nm), "k_continuous": iv["k_continuous"]}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    s8, S8 = ce_s8(c)
    print(f"CE sigma8 = {s8:.4f}, S8 = {S8:.4f}")
    res = score_variant_iv_with_lensing("SK")
    for o in res["rows"]:
        if o["key"].startswith("S8"):
            print(f"   {o['key']:18s} obs {o['obs']} pull {o['pull']:+.2f}")
    print(f"[IV + lensing SK] RMSE Q={res['rmse_Q']:.3f} M={res['rmse_M']:.3f} ALL={res['rmse_all']:.3f} N={res['N']}")
    print(f"theta = 3 phi: U(theta)/U(0) = {theta_equals_three_phi_vacuum_ratio(c):.3f} (rejected)")


if __name__ == "__main__":
    main()
