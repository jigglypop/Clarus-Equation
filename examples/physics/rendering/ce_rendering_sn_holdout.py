"""초신성 보류 시험: 렌더링 식 체계를 한 번도 쓰지 않은 Pantheon 자료로 채점한다. 원장: 43장 §43.37.

계산 전에 고정한 규칙(2026-09-23):
- 자료: 공식 Pantheon DS17 40구간(`benchmarks/cosmology/pantheon_binned_v1`), 통계+계통 공분산 전체.
  절대등급과 H0는 절편 하나로 해석적으로 profile한다. 렌더링 작업의 어떤 선택에도 쓰이지 않았다.
- CE 판본(연속 적합 0, α_s는 s_Z²로 교정):
  L0  코어 Ω_m, 상수 진공.
  W2  v9 채택 진공 기울기(ν = 1/2).
  W3  모방 진공 (b) 읽기(경쟁 판본).
  G1m의 BAO 척도 인자는 표준 자 D/r_d의 판독이므로 주 판정에서는 초신성에 곱하지 않는다(읽기 a).
  인자를 광도거리에도 곱하는 읽기 b는 보조로만 보고한다.
- 비교: 같은 자료로 Ω_m 하나를 맞춘 평탄 ΛCDM(χ²_min), Planck 2018 Ω_m = 0.3153.
- 판정: CE 판본의 χ² − χ²_min ≤ 4(1개 매개변수 2σ)면 통과, > 9면 기각.
  W2가 L0보다 Δχ² > +1 나쁘면 W2의 추가는 보류 시험을 통과하지 못한 것으로 적는다.

python -B -m examples.physics.rendering.ce_rendering_sn_holdout
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from examples.physics.darksector.kinetic_dark_sector_gate import load_pantheon_binned
from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_mimetic_vacuum as MV
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

PLANCK_OM = 0.3153


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = load_pantheon_binned()
    return np.array(d.redshift), np.array(d.apparent_magnitude), np.linalg.inv(np.array(d.covariance))


def comoving_distance(z_obs: np.ndarray, e_of_z: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    z = np.linspace(0.0, float(z_obs.max()) * 1.001, 40001)
    inv = 1 / e_of_z(z)
    dm = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2 * np.diff(z))])
    return np.interp(z_obs, z, dm)


def profiled_chi2(dm: np.ndarray) -> float:
    z, mb, cinv = _data()
    shape = 5 * np.log10((1 + z) * dm)
    r = mb - shape
    ones = np.ones_like(r)
    m = float(ones @ cinv @ r) / float(ones @ cinv @ ones)
    res = r - m
    return float(res @ cinv @ res)


def lcdm_e(om: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda z: np.sqrt(om * (1 + z) ** 3 + 1 - om)


def chi2_lcdm(om: float) -> float:
    z, _, _ = _data()
    return profiled_chi2(comoving_distance(z, lcdm_e(om)))


def best_lcdm() -> dict:
    res = minimize_scalar(chi2_lcdm, bounds=(0.1, 0.6), method="bounded", options={"xatol": 1e-6})
    om = float(res.x)
    lo = float(minimize_scalar(lambda x: (chi2_lcdm(x) - res.fun - 1) ** 2, bounds=(0.1, om), method="bounded").x)
    hi = float(minimize_scalar(lambda x: (chi2_lcdm(x) - res.fun - 1) ** 2, bounds=(om, 0.6), method="bounded").x)
    return {"Om": om, "chi2": float(res.fun), "lo": lo, "hi": hi}


def ce_models() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    c = R.core(R.calibrated_alpha_s()[0])
    om = c["Om"]
    f2 = VT.density(c, VT.ADOPTED_NU)
    m3 = MV.model("b")
    return {
        "L0": lcdm_e(om),
        "W2": lambda z: np.sqrt(om * (1 + z) ** 3 + (1 - om) * np.array([f2(1 / (1 + zi)) for zi in np.atleast_1d(z)])),
        "W3": lambda z: np.sqrt(m3["prim"] * (1 + z) ** 3 + m3["dark"](1 / (1 + np.asarray(z)))),
    }


def score() -> dict:
    z, _, _ = _data()
    c = R.core(R.calibrated_alpha_s()[0])
    best = best_lcdm()
    out = {"best": best, "planck": chi2_lcdm(PLANCK_OM), "N": len(z)}
    for name, e in ce_models().items():
        out[name] = profiled_chi2(comoving_distance(z, e))
    dm_w2 = comoving_distance(z, ce_models()["W2"])
    out["W2+G1m(b)"] = profiled_chi2(dm_w2 * GD.factor("G1m", z, c["Om"]))
    return out


def verdict(out: dict) -> dict:
    ref = out["best"]["chi2"]
    v = {}
    for k in ("L0", "W2", "W3"):
        d = out[k] - ref
        v[k] = "pass" if d <= 4 else ("fail" if d > 9 else "tension")
    v["W2 vs L0"] = "kept" if out["W2"] - out["L0"] <= 1 else "not supported"
    return v


def main() -> None:
    out = score()
    b = out["best"]
    print(f"Pantheon 40 bins: LCDM best Om={b['Om']:.4f} (+{b['hi'] - b['Om']:.4f}/-{b['Om'] - b['lo']:.4f}) chi2={b['chi2']:.2f}; "
          f"Planck Om chi2={out['planck']:.2f}")
    for k in ("L0", "W2", "W3", "W2+G1m(b)"):
        print(f"  {k:10s} chi2={out[k]:.3f}  delta vs best={out[k] - b['chi2']:+.3f}")
    print("verdict:", verdict(out))


if __name__ == "__main__":
    main()
