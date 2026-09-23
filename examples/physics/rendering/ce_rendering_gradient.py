"""CMB–BAO 그라데이션: 기록 시기에 따라 연속으로 변하는 판독 기울기. 원장: 43장 §43.32. 예측값을 바꾸지 않는다.

사용자 가설: CMB(가장 이른 기록, 기울기 0)와 오늘의 직접 판독(π/8) 사이에서 BAO(z = 0.3–2.3)는 중간 기울기를 받는다.
사전 규칙(계산 전 고정): BAO 관측량 D/r_d에 인자 s(z)를 곱한다. 연속 매개변수 0. 모양은 C3 기하에서만.
 G1 나이테 현: s = cos(ψ(z)/2), ψ = H_Λ t(z)(끌림 시기 ψ ≈ 0에서 기록 시기까지의 현).
 G2 현재 현: s = cos(Ψ0/2) / cos((Ψ0 − ψ(z))/2).
 G3 진공 점유율: s = cos[(π/8) Ω_Λ(z)/Ω_Λ(0)] (§43.13에서 기각된 모양의 재확인).
기준은 사전 등록 v9(ν 장부 + W2, 판본 V 0.909). θ*·S8 행은 바뀌지 않는다.
진단: 눈금 자유 적합으로 z 구간별 척도 요구량을 본다.

python -B -m examples.physics.rendering.ce_rendering_gradient
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

C_KM_S = 299792.458


def _psi(z: np.ndarray, om: float) -> np.ndarray:
    return np.array([VT.cycle_phase(1 / (1 + zi), om) for zi in np.atleast_1d(z)])


def factor(profile: str, z: np.ndarray, om: float) -> np.ndarray:
    z = np.atleast_1d(z)
    if profile == "none":
        return np.ones_like(z, dtype=float)
    psi, psi0 = _psi(z, om), VT.cycle_phase(1.0, om)
    if profile == "G1":
        return np.cos(psi / 2)
    if profile == "G2":
        return math.cos(psi0 / 2) / np.cos((psi0 - psi) / 2)
    if profile == "G1m":   # [영감] 물질 나이테는 기울기를 물질 몫 Ω_m만큼 읽는다(사후 선택, 2 bit)
        return 1 - om * (1 - np.cos(psi / 2))
    if profile == "G3":
        share = (1 - om) / (om * (1 + z) ** 3 + 1 - om) / (1 - om)
        return np.cos(R.TIME_AXIS_TILT * share)
    raise ValueError(profile)


def bao_rows(profile: str) -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    _, _, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    b = VT.bao_vectors(c["Om"], VT.density(c, VT.ADOPTED_NU)) * factor(profile, R.BAO_Z, c["Om"])
    fisher = float(b @ R.BAO_CINV @ b)
    af = float(b @ R.BAO_CINV @ R.BAO_Y) / fisher
    a_ce = C_KM_S / (100 * h * rd)
    r = a_ce * b - R.BAO_Y
    rf = af * b - R.BAO_Y
    return {"bao_chi2_fixed": float(r @ R.BAO_CINV @ r), "bao_chi2_free": float(rf @ R.BAO_CINV @ rf),
            "scale_offset": af / a_ce - 1, "tension_sigma": (af / a_ce - 1) / (1 / math.sqrt(fisher) / a_ce),
            "factor_range": (float(factor(profile, R.BAO_Z, c["Om"]).min()), float(factor(profile, R.BAO_Z, c["Om"]).max()))}


def joint_v(profile: str) -> float:
    base = VT.score(VT.ADOPTED_NU)
    chi_rows = base["chi2"] - base["branch"]["bao_chi2_fixed"]
    return math.sqrt((chi_rows + bao_rows(profile)["bao_chi2_fixed"]) / base["N"])


def required_scale_by_redshift() -> list[tuple[float, float, float]]:
    """각 BAO 점을 따로 볼 때 CE 눈금 대비 필요한 척도(관측/예측 − 1)와 오차."""
    c = R.core(R.calibrated_alpha_s()[0])
    _, _, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    pred = C_KM_S / (100 * h * rd) * VT.bao_vectors(c["Om"], VT.density(c, VT.ADOPTED_NU))
    err = np.sqrt(np.diag(np.linalg.inv(R.BAO_CINV)))
    return [(float(z), float(y / p - 1), float(e / p)) for z, y, p, e in zip(R.BAO_Z, R.BAO_Y, pred, err)]


def amplitude_fit(profile: str = "G1") -> dict:
    """진단: s = 1 − k(1 − s_profile)의 k를 고정 눈금 BAO χ²로 맞춘 값과 Δχ² = 1 폭."""
    from scipy.optimize import brentq, minimize_scalar
    c = R.core(R.calibrated_alpha_s()[0])
    _, _, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    b0 = VT.bao_vectors(c["Om"], VT.density(c, VT.ADOPTED_NU))
    a_ce = C_KM_S / (100 * h * rd)
    g = 1 - factor(profile, R.BAO_Z, c["Om"])

    def chi(k):
        r = a_ce * b0 * (1 - k * g) - R.BAO_Y
        return float(r @ R.BAO_CINV @ r)
    best = minimize_scalar(chi, bounds=(-2, 5), method="bounded")
    lo = brentq(lambda k: chi(k) - best.fun - 1, best.x - 3, best.x)
    hi = brentq(lambda k: chi(k) - best.fun - 1, best.x, best.x + 3)
    return {"k": best.x, "k_lo": lo, "k_hi": hi, "chi2_min": best.fun, "chi2_k0": chi(0.0), "chi2_k1": chi(1.0),
            "chi2_k_Om": chi(c["Om"]), "Om": c["Om"]}


def joint_rows(profile: str, rows_from: str = "IV") -> float:
    base = VT.score(VT.ADOPTED_NU, rows_from, "fixed")
    chi_rows = base["chi2"] - base["branch"]["bao_chi2_fixed"]
    return math.sqrt((chi_rows + bao_rows(profile)["bao_chi2_fixed"]) / base["N"])


def main() -> None:
    fit = amplitude_fit("G1")
    print("amplitude diagnostic (G1 shape):", {k: round(v, 4) for k, v in fit.items()})
    print(f"43 rows fixed ruler: none {joint_rows('none', 'full'):.3f}, G1m {joint_rows('G1m', 'full'):.3f}")
    for prof in ("none", "G1", "G2", "G3", "G1m"):
        r = bao_rows(prof)
        print(f"{prof}: factor {r['factor_range'][0]:.4f}..{r['factor_range'][1]:.4f} BAO fixed={r['bao_chi2_fixed']:.2f} "
              f"free={r['bao_chi2_free']:.2f} offset={100 * r['scale_offset']:+.3f}% ({r['tension_sigma']:+.2f} s) V39={joint_v(prof):.3f}")
    print("required (obs/pred - 1) by point:")
    for z, d, e in required_scale_by_redshift():
        print(f"  z={z:.3f}: {100 * d:+.2f}% +/- {100 * e:.2f}%")


if __name__ == "__main__":
    main()
