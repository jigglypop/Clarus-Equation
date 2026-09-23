"""간헐 렌더링(IR) — 순환은 원일 필요가 없고, 흐려졌다가 다시 생겨난다. 원장: 43장 §43.42. 예측값을 바꾸지 않는다.

계산 전에 적은 전제·주장·kill:

전제. C′(§43.35): 기록은 cos 2ψ > 0일 때만 생긴다(ψ = 출생 기록 틀에서 현재 틀까지의 기울기).
   C3 일반화: ψ는 순환 곡선 위에서 출생점 P의 접선과 현 PX의 각이다. 곡선은 원일 필요 없이 매끄럽고 강하게 볼록한
   닫힌 곡선이면 된다. C3의 “실제 시간 = 유클리드 호” 동일시는 원일 때의 시각표에만 쓴다.
주장 IR. (i) 한 바퀴 동안 ψ는 0에서 π까지 단조 증가한다(모양 무관). (ii) 따라서 한 바퀴에 흐려짐 구간
   ψ ∈ (π/4, 3π/4)이 정확히 한 번 있고, 3π/4에서 렌더링이 다시 생겨나며, ψ = π에서 e^{2iψ}가 출생 틀로 돌아간다.
   (iii) 렌더링 구간 안에서 위상에 대해 고른 관측자의 평균 기울기는 π/8(모양 무관) — D′의 공정성 (F)를 전형성 (T)로 대체.
kill. K1 볼록 곡선에서 ψ가 단조가 아니거나 총 회전이 π가 아님. K2 오늘 기울기가 흐려짐 구간 안.
   K3(변형 IR-flux) 관측 플럭스가 기록 에너지 실수부 cos 2Δψ를 따른다면 Pantheon 모양 χ²가 §43.37 기각선(Δχ² > 9)을 넘음.

python -B -m examples.physics.rendering.ce_rendering_intermittent
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_sn_holdout as SN

GYR_PER_INV_KMS_MPC = 977.79


def rendered(psi: float | np.ndarray) -> bool | np.ndarray:
    return np.cos(2 * np.asarray(psi)) > 0


def chord_tilt_on_curve(xy: np.ndarray) -> np.ndarray:
    """닫힌 곡선 점열 xy[0..n) 위에서 출생점 xy[0]의 접선과 현 xy[0]→xy[k]의 각(연속으로 펼침)."""
    t0 = xy[1] - xy[-1]
    base = math.atan2(t0[1], t0[0])
    d = xy[1:] - xy[0]
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]) - base)
    return np.concatenate([[0.0], ang - (2 * np.pi * np.round(ang[0] / (2 * np.pi)))])


def convex_curve(kind: str, n: int = 20000, seed: int = 0) -> np.ndarray:
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if kind == "circle":
        return np.c_[np.cos(s), np.sin(s)]
    if kind.startswith("ellipse"):
        e = float(kind.split(":")[1])
        return np.c_[np.cos(s), math.sqrt(1 - e * e) * np.sin(s)]
    if kind == "random":
        rng = np.random.default_rng(seed)       # 강하게 볼록: 반지름 1 + 작은 조화
        r = 1 + sum(rng.uniform(-0.03, 0.03) * np.cos(k * s + rng.uniform(0, 6.3)) for k in (2, 3))
        return np.c_[r * np.cos(s), r * np.sin(s)]
    raise ValueError(kind)


def loop_structure(kind: str) -> dict:
    """K1: ψ의 단조성, 총 회전, π/4·3π/4 교차 횟수, 흐려짐이 차지하는 호 비율."""
    psi = chord_tilt_on_curve(convex_curve(kind))
    blur = ~rendered(psi)
    return {"monotone": bool(np.all(np.diff(psi) >= -1e-12)), "total": float(psi[-1]),
            "cross_pi4": int(np.sum(np.diff((psi > math.pi / 4).astype(int)) != 0)),
            "cross_3pi4": int(np.sum(np.diff((psi > 3 * math.pi / 4).astype(int)) != 0)),
            "blur_fraction_of_loop": float(blur.mean())}


def circle_timetable() -> dict:
    """원 + C3 동일시(ψ = H_Λ t/2)에서의 시각표. K2: 오늘 ψ₀ < π/4."""
    c = R.core(R.calibrated_alpha_s()[0])
    om, ol = c["Om"], 1 - c["Om"]
    _, _, h = NL.early_densities(c)
    t_lambda = GYR_PER_INV_KMS_MPC / (100 * h * math.sqrt(ol))
    hl_t0 = 2 / 3 * math.asinh(math.sqrt(ol / om))
    return {"t0_Gyr": hl_t0 * t_lambda, "psi0": hl_t0 / 2, "psi0_over_pi4": hl_t0 / 2 / (math.pi / 4),
            "blur_starts_Gyr": math.pi / 2 * t_lambda, "reemerges_Gyr": 3 * math.pi / 2 * t_lambda,
            "loop_closes_Gyr": 2 * math.pi * t_lambda}


def typical_tilt(n: int = 1_000_001) -> dict:
    """(iii) 출생 뒤 렌더링 구간 [0, π/4)에서 위상 균일 평균, 원일 때 시간 균일 평균, 타원(e=0.8) 시간 균일 평균."""
    ph = np.linspace(0, math.pi / 4, n)
    out = {"phase_uniform": float(ph.mean())}
    for kind in ("circle", "ellipse:0.8"):
        psi = chord_tilt_on_curve(convex_curve(kind))
        out[f"arc_uniform_{kind}"] = float(psi[psi < math.pi / 4].mean())
    return out


def psi_of_z(z: np.ndarray, om: float) -> np.ndarray:
    ol = 1 - om
    return (2 / 3) * np.arcsinh(np.sqrt(ol / om) * (1 + np.asarray(z)) ** -1.5) / 2


def flux_variant_chi2() -> dict:
    """K3: 플럭스 ∝ cos 2(ψ₀ − ψ_e) (상대) 또는 cos 2ψ₀ / cos 2ψ_e (절대비)일 때 Pantheon 모양 χ²."""
    c = R.core(R.calibrated_alpha_s()[0])
    om = c["Om"]
    z, _, _ = SN._data()
    dm = SN.comoving_distance(z, SN.lcdm_e(om))
    psi0, psie = psi_of_z(0.0, om), psi_of_z(z, om)
    base = SN.profiled_chi2(dm)
    rel = np.cos(2 * (psi0 - psie))
    ab = np.cos(2 * psi0) / np.cos(2 * psie)
    return {"L0": base, "flux_rel": SN.profiled_chi2(dm / np.sqrt(rel)), "flux_abs": SN.profiled_chi2(dm / np.sqrt(ab)),
            "best_lcdm": SN.best_lcdm()["chi2"], "dim_rel_z1": float(np.interp(1.0, z, rel))}


def main() -> None:
    for kind in ("circle", "ellipse:0.6", "ellipse:0.9", "random"):
        print(f"K1 {kind:12s}", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in loop_structure(kind).items()})
    print("K2 circle timetable:", {k: round(v, 4) for k, v in circle_timetable().items()})
    print("(iii) typical tilt:", {k: round(v, 6) for k, v in typical_tilt().items()}, f"pi/8={math.pi / 8:.6f}")
    f = flux_variant_chi2()
    print("K3 flux variant:", {k: round(v, 3) for k, v in f.items()},
          f"Δχ² rel {f['flux_rel'] - f['best_lcdm']:+.2f}, abs {f['flux_abs'] - f['best_lcdm']:+.2f} (reject > 9)")


if __name__ == "__main__":
    main()
