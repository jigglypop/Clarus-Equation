"""위상 잠금(자기재귀) — 기울기는 원의 각이 아니라 서로 물린 두 위상의 잠긴 차이다. 원장: 43장 §43.50.

사용자 가설: “원처럼 각도가 있는 게 아니고 위상이 물리는 것, 자기재귀처럼.” 예측값을 바꾸지 않는다.
원 사상 θ_{n+1} = θ_n + Ω − (K/2) sin 2θ_n (결합 sin 2θ는 C′의 기록 에너지 e^{2iθ}에서). 적재율 L = 2Ω/K.
계산 전에 적은 판본과 kill:

P-1 (구조). 잠긴 고정점 θ* = ½ arcsin L은 L ≤ 1에서만 있고 범위가 [0, π/4)다(C′). L = 1/√2에서 θ* = π/8(D′).
   L > 1이면 미끄럼(흐려짐), 분기 바로 위에서 잠금 구간의 평균 길이는 (L − 1)^{−1/2}로 줄어든다(1형 간헐성, IR).
P-2 (자료, 연속 적합 없음). 적재율 = 진공 점유율 L(t) = Ω_Λ(t). 선택 전에 √Ω_Λ, 2Ω_Λ(1 − Ω_Λ)도 따져 봤으므로 1.6 bit.
   오늘의 직접 판독 = 나이테 / cos(½ arcsin Ω_Λ0), BAO 인자 = 1 − Ω_m(1 − cos(½ arcsin Ω_Λ(z)))(정리 E).
kill. BAO χ²가 그라데이션 없음(16.10)보다 나쁘거나 판본 V가 0.909보다 나쁘면 대체안으로 기각.

python -B -m examples.physics.rendering.ce_rendering_phase_lock
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT


def circle_map(theta: float, load: float, k: float = 0.5) -> float:
    omega = load * k / 2
    return theta + omega - (k / 2) * math.sin(2 * theta)


def locked_point(load: float, k: float = 0.5) -> dict:
    """고정점과 안정성 승수 μ = 1 − K cos 2θ*. 반복으로 확인."""
    if load > 1:
        return {"locked": False}
    th = 0.5 * math.asin(load)
    x = 0.0
    for _ in range(20000):
        x = circle_map(x, load, k)
    x_mod = (x + math.pi / 2) % math.pi - math.pi / 2
    return {"locked": True, "theta_star": th, "iterated": x_mod, "mu": 1 - k * math.cos(2 * th)}


def laminar_lengths(loads=(1.0005, 1.002, 1.008, 1.032), k: float = 0.5, steps: int = 400000) -> dict:
    """분기 위: 렌더링 창(cos 2θ > 0)에 머무는 구간의 평균 길이와 흐려짐 비율. 이론: 길이 ∝ (L − 1)^{−1/2}."""
    out = {}
    for load in loads:
        x, run, runs, blur = 0.0, 0, [], 0
        for _ in range(steps):
            x = circle_map(x, load, k)
            if math.cos(2 * x) > 0:
                run += 1
            else:
                blur += 1
                if run:
                    runs.append(run)
                    run = 0
        out[load] = {"mean_laminar": float(np.mean(runs)), "blur_fraction": blur / steps,
                     "scaled": float(np.mean(runs)) * math.sqrt(load - 1)}
    return out


def slip_depth(load: float = 1.0005, k: float = 0.5, steps: int = 400000) -> dict:
    """분기 위 미끄럼의 시간 비율: 선명(cos 2θ > ½), 문턱(|cos 2θ| ≤ ½), 깊은 흐려짐(cos 2θ < −½)."""
    x, counts = 0.0, {"clear": 0, "edge": 0, "deep_blur": 0}
    for _ in range(steps):
        x = circle_map(x, load, k)
        c2 = math.cos(2 * x)
        counts["deep_blur" if c2 < -0.5 else ("edge" if abs(c2) <= 0.5 else "clear")] += 1
    return {k2: v / steps for k2, v in counts.items()}


def data_test() -> dict:
    """P-2: 적재율 = Ω_Λ(t). 직접 판독 H₀와 BAO 인자로 판본 V를 다시 채점."""
    c = R.core(R.calibrated_alpha_s()[0])
    om, ol0 = c["Om"], 1 - c["Om"]
    ring = 100 * NL.early_densities(c)[2]
    th0 = 0.5 * math.asin(ol0)
    direct = ring / math.cos(th0)
    z = R.BAO_Z
    ol_z = ol0 / (om * (1 + z) ** 3 + ol0)
    factor = 1 - om * (1 - np.cos(0.5 * np.arcsin(ol_z)))
    rd, h = NL.rd_and_h(c)
    b = VT.bao_vectors(om, VT.density(c, VT.ADOPTED_NU))
    r = GD.C_KM_S / (100 * h * rd) * b * factor - R.BAO_Y
    bao = float(r @ R.BAO_CINV @ r)
    base = VT.score(VT.ADOPTED_NU)
    chi = 0.0
    h0_rows = {}
    for o in base["rows"]:
        p = o["pull"]
        if o["key"].startswith("H0 "):
            p = (direct - o["obs"]) / o["sigma"]
            h0_rows[o["key"]] = round(p, 2)
        chi += p ** 2
    v39 = math.sqrt((chi + bao) / base["N"])
    return {"theta0": th0, "pi_8": math.pi / 8, "direct_H0": direct, "h0_pulls": h0_rows, "bao_chi2": bao,
            "bao_chi2_G1m": GD.bao_rows("G1m")["bao_chi2_fixed"], "bao_chi2_none": GD.bao_rows("none")["bao_chi2_fixed"],
            "V39": v39, "V39_current": GD.joint_v("G1m"), "load_for_pi_8": math.sin(math.pi / 4), "Omega_L0": ol0}


def main() -> None:
    for load in (0.0, 0.5, 1 / math.sqrt(2), 0.95, 1.0):
        print("P-1 lock", round(load, 4), {k: (round(v, 6) if isinstance(v, float) else v) for k, v in locked_point(load).items()})
    print("P-1 lock", 1.01, locked_point(1.01))
    for load, v in laminar_lengths().items():
        print(f"P-1 intermittency L={load}: mean laminar {v['mean_laminar']:.1f}, blur fraction {v['blur_fraction']:.4f}, "
              f"laminar*sqrt(L-1) {v['scaled']:.3f}")
    print("P-2 data:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data_test().items()})


if __name__ == "__main__":
    main()
