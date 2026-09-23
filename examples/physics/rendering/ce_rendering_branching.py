"""우주 성분의 가지치기 유도 — 인과 계보가 끊기면 보이는 물질, 이어지면 암흑. 원장: 43장 §43.54. 예측값을 바꾸지 않는다.

코어의 q = e^{−D(1−q)}의 작은 근(= Ω_b)은 포아송 Galton–Watson 과정의 소멸 확률이다. 계산 전에 적은 판본과 kill:

정리 GW(표준). 자식 수 생성함수 G에서 소멸 확률은 s = G(s)의 [0,1] 안 가장 작은 근이고, s_{n+1} = G(s_n), s_0 = 0은
   “n세대 안에 끊길 확률”로서 그 근에 수렴한다. 포아송(D): G(s) = e^{D(s−1)}.
공리 BR1 각 기록의 인과적 후손 수는 독립 포아송, 평균 D = 3 + δ. BR2 끊긴 계보 = 보이는 바리온(Ω_b = q).
   BR3 살아남은 계보는 묶임(비율 α_s D) 대 자유(비율 1)의 경쟁: Ω_DM = (1 − q)α_sD/(1 + α_sD), Ω_Λ = (1 − q)/(1 + α_sD).
kill. K1 몬테카를로 끊김 비율이 q와 3σ 밖. K2 반복이 작은 근으로 수렴하지 않음.
   K3(특이성) 같은 평균의 다른 법칙(이항 4·8·64, 기하)도 ω_b를 2σ 안에서 맞추면 “특이하지 않음”으로 보고.
   K4 끊긴 계보 크기 = Borel(Dq): P(1) = e^{−Dq}, 평균 1/(1 − Dq)를 몬테카를로로 확인.

python -B -m examples.physics.rendering.ce_rendering_branching
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R

OMEGA_B_OBS = (0.02237, 0.00015)
LAWS = {
    "Poisson": lambda s, d: math.exp(d * (s - 1)),
    "Binomial(4)": lambda s, d: (1 - d / 4 + d / 4 * s) ** 4,
    "Binomial(8)": lambda s, d: (1 - d / 8 + d / 8 * s) ** 8,
    "Binomial(64)": lambda s, d: (1 - d / 64 + d / 64 * s) ** 64,
    "Geometric": lambda s, d: 1 / (1 + d * (1 - s)),
}


def small_root(law: str, d: float) -> float:
    g = LAWS[law]
    return brentq(lambda s: s - g(s, d), 0.0, 1 - 1e-9) if g(0.0, d) > 0 else 0.0


def iterate_to_extinction(d: float, n: int = 200) -> dict:
    """K2: s_{n+1} = G(s_n), s_0 = 0 (n세대 안의 끊김 확률)."""
    s, hist = 0.0, []
    for _ in range(n):
        s = math.exp(d * (s - 1))
        hist.append(s)
    return {"limit": s, "gen1": hist[0], "gen5": hist[4], "gen20": hist[19]}


def monte_carlo(d: float, trials: int = 40000, cap: int = 400, gens: int = 200, seed: int = 17) -> dict:
    """K1·K4: 포아송(d) 계보를 흉내 내어 끊김 비율과 끊긴 계보의 크기 분포를 센다."""
    rng = np.random.default_rng(seed)
    extinct, sizes = 0, []
    for _ in range(trials):
        pop, total = 1, 1
        for _ in range(gens):
            if pop == 0 or pop > cap:
                break
            pop = int(rng.poisson(d, size=pop).sum())
            total += pop
        if pop == 0:
            extinct += 1
            sizes.append(total)
    frac = extinct / trials
    sizes = np.array(sizes)
    return {"extinct_fraction": frac, "mc_sigma": math.sqrt(frac * (1 - frac) / trials),
            "size1_fraction": float(np.mean(sizes == 1)), "mean_size": float(sizes.mean())}


def law_specificity() -> dict:
    """K3: 같은 평균 D에서 법칙별 q와 ω_b = q h² 의 pull(h는 CE의 R-Pl 값)."""
    c = R.core(R.calibrated_alpha_s()[0])
    h = NL.early_densities(c)[2]
    out = {}
    for law in LAWS:
        q = small_root(law, c["D"])
        out[law] = {"q": q, "omega_b": q * h * h, "pull": (q * h * h - OMEGA_B_OBS[0]) / OMEGA_B_OBS[1]}
    return out


def composition() -> dict:
    """BR3: 코어와의 일치와 분할의 비, Borel(Dq) 이론값."""
    c = R.core(R.calibrated_alpha_s()[0])
    d, q, x = c["D"], c["q"], c["a"] * c["D"]
    om_dm = (1 - q) * x / (1 + x)
    om_l = (1 - q) / (1 + x)
    mu = d * q
    return {"D": d, "q": q, "Omega_DM": om_dm, "Omega_L": om_l, "Om_core": c["Om"], "Om_branch": q + om_dm,
            "DM_over_L": om_dm / om_l, "alpha_s_D": x, "borel_mu": mu, "borel_P1": math.exp(-mu),
            "borel_mean": 1 / (1 - mu)}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    print("K2 iteration:", {k: round(v, 6) for k, v in iterate_to_extinction(c["D"]).items()}, "small root q =", round(c["q"], 6))
    mc = monte_carlo(c["D"])
    print("K1/K4 Monte Carlo:", {k: round(v, 5) for k, v in mc.items()},
          f"| (frac − q)/σ = {(mc['extinct_fraction'] - c['q']) / mc['mc_sigma']:+.2f}")
    for law, r in law_specificity().items():
        print(f"K3 {law:13s} q={r['q']:.5f} omega_b={r['omega_b']:.5f} pull={r['pull']:+.1f}")
    print("composition:", {k: round(v, 5) for k, v in composition().items()})


if __name__ == "__main__":
    main()
