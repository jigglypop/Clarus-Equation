"""W2의 작용 수준 읽기: 복소 척도인자 위상의 가장 낮은 실수 조화. 원장: 43장 §43.27. 예측값을 바꾸지 않는다.

C6(후보 공리): 진공 작용 항은 a³ V(φ)이고 V는 A/|A| = e^{iφ}의 해석 함수다. 작용이 실수이고 A ↔ Ā(허수 회전의
뒤집기, φ → −φ)에 불변이면 V = ρ_Λ[1 + Σ_k c_k cos(kφ)]로 sin 항이 없다. φ(t) = ω t, ω = H_Λ/2(§43.16),
탄생에서 φ = 0이므로 k번째 조화는 W2의 ν = k/2다. 가장 낮은 조화 k = 1이 ν = 1/2이다.
남는 선택: c_1의 부호(1 bit, 자료로 정함), 진폭 |c_1| = ξ² = α_s^{2/3} = A_2/2 = sinθ_W/2(E4에 의한 항등식, 유도 아님).

python -B -m examples.physics.rendering.ce_rendering_vacuum_harmonic
"""

from __future__ import annotations

import math
from typing import Callable

from examples.physics.rendering import ce_rendering_complex_scale as CS
from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

C_KM_S = 299792.458


def phase_rate_over_h_lambda(c: dict) -> float:
    """§43.16의 ω를 H_Λ = H0 √Ω_Λ로 나눈 값(1/2이어야 한다)."""
    *_, omega, h0 = CS.background(c)
    return omega / (h0 * math.sqrt(1 - c["Om"]))


def harmonic_density(c: dict, k: int, sign: float = 1.0) -> Callable[[float], float]:
    om, xi2 = c["Om"], c["a"] ** (2 / 3)
    phi = lambda a: 0.5 * VT.cycle_phase(a, om)          # φ = H_Λ t / 2
    norm = 1 + sign * xi2 * math.cos(k * phi(1.0))
    return lambda a: (1 + sign * xi2 * math.cos(k * phi(a))) / norm


def score(k: int, sign: float = 1.0, rows_from: str = "IV", ruler: str = "fixed") -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    f = harmonic_density(c, k, sign)
    wb, wc, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    b = VT.bao_vectors(c["Om"], f)
    fisher = float(b @ R.BAO_CINV @ b)
    af = float(b @ R.BAO_CINV @ R.BAO_Y) / fisher
    r = (C_KM_S / (100 * h * rd) if ruler == "fixed" else af) * b - R.BAO_Y
    chib = float(r @ R.BAO_CINV @ r)
    h0b = C_KM_S / (af * rd)
    s8 = NL.s8(c) * VT.growth_today(c["Om"], f) / VT.growth_today(c["Om"], lambda a: 1.0)
    rows = [dict(o) for o in NL.score(rows_from, ruler)["rows"]]
    for o in rows:
        if o["key"] == "100 theta*":
            o["pred"] = VT.theta_star_100(wb, wc, h, f)
        elif o["key"] in {n for n, *_ in GR.LENSING_S8}:
            o["pred"] = s8
        else:
            continue
        o["pull"] = (o["pred"] - o["obs"]) / o["sigma"]
    n = len(rows) + 13
    theta = next(o["pull"] for o in rows if o["key"] == "100 theta*")
    return {"k": k, "sign": sign, "rmse_all": math.sqrt((sum(o["pull"] ** 2 for o in rows) + chib) / n), "N": n,
            "theta_pull": theta, "cmb_bao_tension": (h0b - 100 * h) * math.sqrt(fisher) / h0b * af}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    print(f"omega / H_Lambda = {phase_rate_over_h_lambda(c):.12f}  ->  harmonic k gives nu = k/2")
    print(f"xi^2 = {c['a'] ** (2 / 3):.12f}; sin(theta_W)/2 = {math.sqrt(4 * c['a'] ** (4 / 3)) / 2:.12f} (E4 identity)")
    for k in (1, 2, 6):
        for sign in (1.0, -1.0):
            s = score(k, sign)
            print(f"k={k} sign={sign:+.0f}: V39={s['rmse_all']:.3f} theta={s['theta_pull']:+.2f} CMB-BAO={s['cmb_bao_tension']:+.2f}")


if __name__ == "__main__":
    main()
