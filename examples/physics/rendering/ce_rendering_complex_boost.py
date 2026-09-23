"""정리 C·D의 재증명 검산 — 복소 빠르기와 기록 에너지 양수. 원장: 43장 §43.35. 예측값을 바꾸지 않는다.

기록 틀 u에서 판독 틀로 가는 변환을 복소화한 2차원 부스트 Λ(ζ), ζ = η + iφ로 둔다. η는 실제 부스트,
φ는 C2(§43.12)의 유클리드 기울기다. 계산 전에 적은 두 분기와 kill 조건:

분기 R (실수 읽기, §43.29의 v = c tan θ). u가 주는 보조 계량 δ_u = g + 2u⊗u로 각을 잰다.
   g-널 방향은 δ_u-각 π/4이고 이등분은 실제 속도 tan(π/8)c를 준다. kill: CMB 쌍극자 β ≈ 1.2e-3.
분기 I (허수 기울기). 반례 I-0: w = Λ(iφ)u는 모든 φ에서 g(w,w) = −1이므로 "시간꼴 ⇔ |φ| < π/4"는 거짓이다.
   전제 E_rec: 판독 틀이 받는 빛 기록(양방향 널 먼지 T = E k⊗k)의 에너지 밀도 T(w,w)는 실수부가 양수다.
   C′: E_rec ⇔ |Im ζ| < π/4 ⇔ |tanh ζ| < 1, 실수 부스트에 불변.  D′: Im ζ의 Haar 측도 + 공정성(F) ⇒ π/8.
kill: K1 E_rec 영역 ≠ 띠, K2 Im ζ가 실수 부스트에 불변 아님, K3 γ(iπ/8) ≠ cos(π/8), K4 오늘 기울기 판독 ≥ π/4.

python -B -m examples.physics.rendering.ce_rendering_complex_boost
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from examples.physics.rendering import ce_rendering_registry as R

G = np.diag([-1.0, 1.0])            # (t, x), c = 1
U = np.array([1.0, 0.0])            # 기록 틀의 4-속도
NULLS = (np.array([1.0, 1.0]), np.array([1.0, -1.0]))
CMB_DIPOLE_BETA = 1.23e-3           # 태양계의 CMB 대비 속도 369.8 km/s / c


def boost(zeta: complex) -> np.ndarray:
    ch, sh = cmath.cosh(zeta), cmath.sinh(zeta)
    return np.array([[ch, sh], [sh, ch]], complex)


def bilinear(a: np.ndarray, b: np.ndarray) -> complex:
    """g의 복소 쌍선형 확장(켤레 없음)."""
    return complex(a @ G @ b)


def tilted_axis_norm(phi: float) -> complex:
    w = boost(1j * phi) @ U
    return bilinear(w, w)


def record_energy(zeta: complex, k: np.ndarray) -> complex:
    """널 먼지 T = k⊗k의 판독 틀 에너지 밀도 T(w,w) = (k·w)²."""
    w = boost(zeta) @ U
    return bilinear(k, w) ** 2


def strip_equivalence_scan(trials: int = 20000, seed: int = 7) -> dict:
    """K1: E_rec(양방향 Re > 0), |tanh ζ| < 1, |Im ζ| < π/4가 같은 집합인지. 경계 1e-6 안은 제외."""
    rng = np.random.default_rng(seed)
    bad, used = 0, 0
    for _ in range(trials):
        eta, phi = rng.uniform(-3, 3), rng.uniform(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3)
        if abs(abs(phi) - math.pi / 4) < 1e-6:
            continue
        z = complex(eta, phi)
        e_rec = all(record_energy(z, k).real > 0 for k in NULLS)
        disk = abs(cmath.tanh(z)) < 1
        strip = abs(phi) < math.pi / 4
        bad += not (e_rec == disk == strip)
        used += 1
    return {"trials": used, "mismatches": bad}


def real_boost_invariance(trials: int = 200, seed: int = 9) -> float:
    """K2: Λ(η_r)Λ(ζ) = Λ(ζ + η_r)이므로 Im ζ는 실수 부스트에 불변. 행렬 차의 최댓값."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        z = complex(rng.uniform(-2, 2), rng.uniform(-0.7, 0.7))
        er = rng.uniform(-3, 3)
        worst = max(worst, float(np.abs(boost(er) @ boost(z) - boost(z + er)).max()))
    return worst


def delta_u_angle(v: np.ndarray) -> float:
    """분기 R: δ_u = g + 2u⊗u(여기서는 유클리드 단위행렬)에서 u와 v의 각."""
    d = np.eye(2)
    return math.acos(abs(float(U @ d @ v)) / math.sqrt(float(v @ d @ v)))


def real_branch() -> dict:
    v = math.tan(math.pi / 8)
    return {"null_angle": [delta_u_angle(k) for k in NULLS], "bisector_beta": v,
            "beta_over_dipole": v / CMB_DIPOLE_BETA, "gamma": 1 / math.sqrt(1 - v * v),
            "delta_norm_ratio": math.sqrt(1 + v * v)}


def hyperbolic_distance_to_tilt(phi: float) -> float:
    """띠 |Im ζ| < π/4 의 쌍곡 거리: tanh가 띠를 단위원판으로 보내므로 d(0, iφ) = 2 artanh(tan φ)."""
    return 2 * math.atanh(math.tan(phi))


def haar_bisector(limit: float = math.pi / 4, n: int = 200001) -> float:
    """D′: 평탄 측도에서 두 끝(0, limit)까지 거리의 최댓값을 최소화."""
    th = np.linspace(0.0, limit, n)
    return float(th[np.argmin(np.maximum(th, limit - th))])


def lorentz_factor(zeta: complex) -> complex:
    return cmath.cosh(zeta)


def tilt_readings_today() -> dict:
    """K4: 오늘의 기울기 판독(고정 π/8, 접선–현 H_Λt/2, 나선 arctan(√Ω_Λ/2))이 한계 π/4 안인지."""
    om = R.core(R.calibrated_alpha_s()[0])["Om"]
    ol = 1 - om
    hl_t0 = 2 / 3 * math.asinh(math.sqrt(ol / om))
    return {"fixed": math.pi / 8, "tangent_chord": hl_t0 / 2, "spiral": math.atan(math.sqrt(ol) / 2),
            "spiral_future_max": math.atan(0.5), "tangent_chord_hits_limit_at_t_over_t0": (math.pi / 2) / hl_t0}


def main() -> None:
    print("I-0. g(w,w) for w = Λ(iφ)u:", {round(p, 3): round(tilted_axis_norm(p).real, 12) for p in (0.2, math.pi / 4, 1.2)})
    print("K1. E_rec / |tanh ζ|<1 / |Im ζ|<π/4:", strip_equivalence_scan())
    print(f"K2. real-boost invariance of Im ζ: max matrix error {real_boost_invariance():.1e}")
    g = lorentz_factor(1j * math.pi / 8)
    print(f"K3. γ(iπ/8) = {g.real:.12f} {g.imag:+.1e}i, cos(π/8) = {math.cos(math.pi / 8):.12f}, 1/γ = {1 / g.real:.6f}")
    print("K4. tilt readings today:", {k: round(v, 5) for k, v in tilt_readings_today().items()}, f"limit {math.pi / 4:.5f}")
    print("R. real branch:", {k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v]) for k, v in real_branch().items()})
    print(f"D′. Haar bisector {haar_bisector():.8f} (π/8 = {math.pi / 8:.8f}); hyperbolic d(0, iφ) at φ = π/8, π/4−1e-9:",
          round(hyperbolic_distance_to_tilt(math.pi / 8), 5), round(hyperbolic_distance_to_tilt(math.pi / 4 - 1e-9), 2))


if __name__ == "__main__":
    main()
