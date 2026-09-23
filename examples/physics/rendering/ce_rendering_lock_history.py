"""잠금의 역사 — 과거가 현재를 끌기 시작한 뒤 기울기가 어떻게 자랐나. 원장: 43장 §43.52. 예측값을 바꾸지 않는다.

지금의 채택 규칙은 오늘의 직접 판독에 고정 π/8(O1)을, BAO 그라데이션에 접선–현 ψ/2 = H_Λt/2(G1m)를 쓴다.
같은 기울기의 과거와 현재를 두 규칙이 따로 다룬다(오늘 0.398 대 0.3927). 인과 잠금(§43.51)의 하나의 θ(t)로 둘을 함께
채점한다. 계산 전에 적은 판본(연속 적합 없음, 시간 척도는 H_Λ 하나, 판본 선택 1 bit):

V-a  dθ/du = ½(1 − √2 sin 2θ), u = H_Λ t: 시작 속도가 접선–현(½)과 같고 잠금점은 무차별의 π/8.
V-b  dθ/du = 1/(2√2) − ½ sin 2θ: 결합 K = H_Λ.
V-c  탄생부터 잠김: 모든 시기 θ = π/8.
기준  O1 π/8 + G1m ψ/2(현재 채택).
kill. 판본 V > 0.909 또는 H₀ 행 |pull| > 3이면 대체안으로 기각.

python -B -m examples.physics.rendering.ce_rendering_lock_history
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

RATE = {"V-a": lambda th: 0.5 * (1 - math.sqrt(2) * math.sin(2 * th)),
        "V-b": lambda th: 1 / (2 * math.sqrt(2)) - 0.5 * math.sin(2 * th)}


def theta_of_u(variant: str, u: np.ndarray) -> np.ndarray:
    if variant == "V-c":
        return np.full_like(np.asarray(u, float), math.pi / 8)
    f = RATE[variant]
    umax = float(np.max(u))
    sol = solve_ivp(lambda _, y: [f(y[0])], (0.0, umax), [0.0], dense_output=True, rtol=1e-10, atol=1e-12)
    return sol.sol(np.asarray(u, float))[0]


def score(variant: str) -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    om = c["Om"]
    ring = 100 * NL.early_densities(c)[2]
    u0 = VT.cycle_phase(1.0, om)
    u_z = np.array([VT.cycle_phase(1 / (1 + z), om) for z in R.BAO_Z])
    if variant == "ref":
        th0 = math.pi / 8
        th_z = u_z / 2
    else:
        th0 = float(theta_of_u(variant, np.array([u0]))[0])
        th_z = theta_of_u(variant, u_z)
    direct = ring / math.cos(th0)
    factor = 1 - om * (1 - np.cos(th_z))
    rd, h = NL.rd_and_h(c)
    b = VT.bao_vectors(om, VT.density(c, VT.ADOPTED_NU))
    r = GD.C_KM_S / (100 * h * rd) * b * factor - R.BAO_Y
    bao = float(r @ R.BAO_CINV @ r)
    base = VT.score(VT.ADOPTED_NU)
    chi, pulls = 0.0, {}
    for o in base["rows"]:
        p = o["pull"]
        if o["key"].startswith("H0 "):
            p = (direct - o["obs"]) / o["sigma"]
            pulls[o["key"]] = round(p, 2)
        chi += p ** 2
    v39 = math.sqrt((chi + bao) / base["N"])
    return {"theta_today": th0, "theta_z_range": (float(th_z.min()), float(th_z.max())), "direct_H0": direct,
            "h0_pulls": pulls, "bao_chi2": bao, "V39": v39,
            "kill": v39 > 0.909 or any(abs(v) > 3 for v in pulls.values())}


def main() -> None:
    for v in ("ref", "V-a", "V-b", "V-c"):
        s = score(v)
        print(f"{v:4s}", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items()})


if __name__ == "__main__":
    main()
