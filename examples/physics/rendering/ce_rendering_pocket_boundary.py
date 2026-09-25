"""주머니의 경계 — CE 공리(QG1(나)·R1)로 첫 사건의 네 양(분포·인과 압력·속도·경계면 상태)을 세운다.
원장: 43장 §43.108.

사용자(2026-09-26): "다음"(직전 제안: 면 갈래로, 주머니를 둘러싼 '영원히 떨리는 경계'의 상태를 식으로 세운다).

계산 전 고정(2026-09-26):
공리 점검(외부 모형 전에). §43.107 F1의 기하와 대가(첫 사건은 면, 주머니 무수)는 표준 확률 급팽창을 썼다. 표준은 확정 전
  떨림이 조각마다 제 H로 팽창을 몬다고 둔다(부피 가중). CE는 QG1(나)(§43.30 정리 A: 원천 = Tr ρT, §43.85: 요동은 계량을
  흔들지 않는다)과 R1(§43.31: 기록은 제 빛원뿔 안만 원천을 갱신한다)을 둔다. 그래서 두 판본을 따로 세운다.
  S(표준): 확정 전 떨림도 국소 원천 → 스스로 불어나는 영역(영원), 출구는 면, 경계는 층
     [부피 전이 P_ζ = 1/6(Creminelli+2008, φ̇²/H⁴ = 3/(2π²)) ~ 시계 경계 P_ζ = 1].
  Q(CE): 확정 전 떨림은 평균으로만 원천 → 경계 밖은 균질한 바다(스스로 불어나지 않는다; 선행 결론: Lechuga–Sudarsky 2023,
     arXiv:2308.01383). 첫 기록 하나가 R1로 제 빛원뿔만 확정 → 첫 사건은 점, 경계는 빛꼴. 바다가 거의 드 시터 불변이면
     (ε ≪ 1) 안쪽은 O(3,1) 열린 거품(§43.106 G1). 기록이 φ_q 근처에서 시작하면 안쪽 급팽창은 N_q(§43.107).
     기록 여럿이 합쳐 평탄 조각이 되는지, 하나의 거품인지는 기록률이 정한다(CE는 아직 정하지 않음, RD3).
정의(공통): r = √P_ζ = (H/2π)/(√(2ε)M_P). 인과 압력 비 Π = 1/r, φ 등위면의 확정 앞면 속도 v_f = c/r(떨림 하나/허블 길이),
  허블 조각당 첫 사건 시각의 흔들림 δN = r, 지평 엔트로피 증가 dS/dN = 2/r²(느린 굴림 항등식, Arkani-Hamed+2007 계열).
  G1 시계 τ의 앞면은 v_f = t/ρ(ρ = 반지름).
계산: C1 r·v_f·dS/dN·ε·H/H_*를 피벗·급팽창 끝·φ_q에서. C2 S 층: N(P_ζ = 1/6), N(1), 두께, 부피 앞면 속도 σ(√6 − 1/r).
  C3 Q: Ω_k = e^{−2(N_q − N(a0H0))}(+), 기록 시작이 10⁵ e-fold 늦어도 ≈ 0인지, 곡률이 보이려면 기록이 끝나기 몇 e-fold
  전에 시작해야 하는지. C4 항등식 검산: ∫ 2/P_ζ dN(φ_q → 끝, 수치) = 24π²(1/V_end − 1/V_q)(닫힌 꼴). C5 G1 앞면: 중심 ∞, 경계 c.
kill: K1 C4 상대 오차 > 10⁻⁶ → 코드·모형 오류. K2 ε(φ_q) ≥ 0.01이면 Q의 거품 기하 기각(바다가 드 시터 불변이 아님).
  K3 N_q ≥ 지평 엔트로피 증가분 → F1 기각(Arkani-Hamed 상한). K4(관측, 미래) P23 기각(중력 매개 얽힘 5σ) → Q 기각, S로.
  Ω_k < 0 3σ → Q의 열린 거품 기각. Ω_k ≠ 0 3σ → F1 기각.
예상(계산 전, 해석 유도): K1–K3 통과. S 층 두께 ≈ (1 − 1/√6)N_q ≈ 0.59 N_q. S의 부피 앞면은 P_ζ = 1에서 오르막 √6 − 1.
감사(2026-09-26, 사용자 "정확도는?", 독립 검산 opus, 원장 §43.111): 숫자는 모두 재현되었다. 철회: "바다가 드 시터 불변이라 한 점
  기록의 안쪽은 O(3,1) 열린 거품"(ε는 계량만 잰다. 상태는 굴러가는 ⟨φ⟩가 평탄 조각을 고르고, 굴림/떨림 = 1/r = 1이다.
  기록은 장을 다시 놓지 않으므로 안쪽은 평탄하다). 그래서 K2는 잘못 세운 검사다. 미정으로 내림: "층·주머니 무수는 표준 모형 탓"
  (기록된 떨림은 R1로 국소 원천이다. 층의 97%에서 기록 예산이 팽창을 넘을 수 있어 자기 증식이 되살아날 수 있다).
  조건 명시: Q는 QG1(나)를 비관측 성분의 정확한 법칙으로 둘 때만 선다(56장은 평균장을 근사로만 인정한다). audit()이 재현한다.

python -B -m examples.physics.rendering.ce_rendering_pocket_boundary
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_bubble_capacity as BCP

P_CLOCK = 1.0                    # 떨림 = 굴림(φ 등위면이 빛꼴)
P_VOLUME = 1 / 6                 # Creminelli+2008 부피 전이


def _model() -> dict:
    """정확한 스타로빈스키(M_P = 1): x = e^{κφ}, V = V0(1 − 1/x)², ε = (4/3)/(x − 1)², V0는 N_e에서 P_ζ = A_s로 맞춘다."""
    from scipy.optimize import brentq
    c = BCP._ctx()
    ne, a_s = c["ne"], c["A_s"]
    x_end = 1 + 2 / math.sqrt(3)
    n_of = lambda y: 0.75 * ((y - x_end) - math.log(y / x_end))
    eps = lambda y: (4 / 3) / (y - 1) ** 2
    shape = lambda y: (1 - 1 / y) ** 2                                           # V/V0
    x_star = brentq(lambda y: n_of(y) - ne, x_end * 1.0001, 1e4)
    v0 = 24 * math.pi ** 2 * a_s * eps(x_star) / shape(x_star)                   # P_ζ = V/(24π²ε)
    p_of = lambda y: v0 * shape(y) / (24 * math.pi ** 2 * eps(y))
    x_at = lambda p: brentq(lambda y: p_of(y) - p, x_star, 1e12)
    return {"ne": ne, "a_s": a_s, "n_hub": c["n_hub"], "x_end": x_end, "x_star": x_star, "n_of": n_of, "eps": eps,
            "shape": shape, "v0": v0, "p_of": p_of, "x_at": x_at}


def ratios() -> dict:
    """C1: 한 비율 r = √P_ζ가 네 양을 정한다."""
    m = _model()
    h = lambda y: math.sqrt(m["v0"] * m["shape"](y) / 3)
    pts = {"pivot": m["x_star"], "end": m["x_end"], "boundary": m["x_at"](P_CLOCK)}
    out = {}
    for k, y in pts.items():
        p = m["p_of"](y)
        r = math.sqrt(p)
        out[k] = {"N": m["n_of"](y), "phi_MP": math.sqrt(1.5) * math.log(y), "P_zeta": p, "r": r,
                  "causal_push": 1 / r, "front_speed_c": 1 / r, "dS_dN": 2 / p, "eps": m["eps"](y),
                  "H_over_H_star": h(y) / h(m["x_star"])}
    return out


def standard_layer() -> dict:
    """C2: 표준(S)에서 경계는 시계 경계(P_ζ = 1)와 부피 전이(P_ζ = 1/6) 사이의 층이다."""
    m = _model()
    n1, n6 = m["n_of"](m["x_at"](P_CLOCK)), m["n_of"](m["x_at"](P_VOLUME))
    front = {k: math.sqrt(6) - 1 / math.sqrt(p) for k, p in (("P=1", P_CLOCK), ("P=1/6", P_VOLUME), ("pivot", m["a_s"]))}
    return {"N_clock": n1, "N_volume": n6, "thickness": n1 - n6, "thickness_over_Nq": (n1 - n6) / n1,
            "volume_front_sigma_per_efold": front}


def qg1_bubble(delay: float = 1e5) -> dict:
    """C3: CE(Q)의 거품 — 곡률, 늦은 시작, 드 시터 불변성."""
    m = _model()
    x_q = m["x_at"](P_CLOCK)
    n_q = m["n_of"](x_q)
    log10 = lambda n: -2 * (n - m["n_hub"]) / math.log(10)
    return {"N_q": n_q, "log10_Omega_k": log10(n_q), "log10_Omega_k_delayed": log10(n_q - delay), "sign": "+",
            "eps_at_boundary": m["eps"](x_q), "dS_invariant": m["eps"](x_q) < 0.01,
            "start_needed_for_killed_curvature": m["n_hub"] + 0.5 * math.log(1 / BCP.UPPER),
            "start_needed_for_window": m["n_hub"] + 0.5 * math.log(1 / BCP.WINDOW_OBS[0])}


def entropy_identity() -> dict:
    """C4: dS/dN = 2/P_ζ를 φ_q에서 끝까지 적분한 값과 지평 엔트로피 24π²/V의 차를 대조한다."""
    from scipy.integrate import quad
    m = _model()
    x_q = m["x_at"](P_CLOCK)
    integrand = lambda u: (2 / m["p_of"](math.exp(u))) * 0.75 * (1 - math.exp(-u)) * math.exp(u)   # u = ln x
    num, _ = quad(integrand, math.log(m["x_end"]), math.log(x_q), limit=400, epsrel=1e-12)
    closed = 24 * math.pi ** 2 / m["v0"] * (1 / m["shape"](m["x_end"]) - 1 / m["shape"](x_q))
    n_q = m["n_of"](x_q)
    return {"numeric": num, "closed": closed, "rel_err": abs(num / closed - 1), "N_q": n_q,
            "S_star": 24 * math.pi ** 2 / (m["v0"] * m["shape"](m["x_star"])), "gain_over_Nq": closed / n_q}


def g1_front(tau: float = 1.0) -> dict:
    """C5: 한 점 사건의 시계 τ = √(t² − ρ²)의 등위면 앞면 속도 t/ρ(단위 c)."""
    out = {"rho=0": math.inf}
    for rho in (0.1, 1.0, 10.0, 100.0):
        out[f"rho={rho:g}"] = math.sqrt(tau ** 2 + rho ** 2) / rho
    out["boundary(tau=1e-9)"] = math.sqrt(1e-18 + 1.0)                             # 빛원뿔 위(ρ = t)
    return out


def audit() -> dict:
    """§43.111 감사: 상태의 굴림/떨림(열린 거품 철회의 근거), 층 안에서 기록 예산이 팽창을 넘을 수 있는 몫."""
    m = _model()
    x_q = m["x_at"](P_CLOCK)
    n1, n6 = m["n_of"](x_q), m["n_of"](m["x_at"](P_VOLUME))
    n_rl = m["n_of"](m["x_at"](2 / (3 * math.log(2))))                            # f_max = 3(1비트 비용)인 곳
    return {"roll_over_kick_at_boundary": 1 / math.sqrt(m["p_of"](x_q)), "eps_at_boundary_metric_only": m["eps"](x_q),
            "layer_fraction_fmax_over_3": (n_rl - n6) / (n1 - n6), "fmax_bit_at_P_1_6": 2 / (P_VOLUME * math.log(2))}


def verdict() -> dict:
    ei, qb = entropy_identity(), qg1_bubble()
    return {"K1_identity_pass": ei["rel_err"] < 1e-6, "K2_dS_invariant_pass": qb["dS_invariant"],
            "K3_entropy_bound_pass": ei["N_q"] < ei["closed"]}


def main() -> None:
    for k, v in ratios().items():
        print(k, {a: f"{b:.4g}" for a, b in v.items()})
    print("standard_layer", standard_layer())
    print("qg1_bubble", qg1_bubble())
    print("entropy_identity", entropy_identity())
    print("g1_front", g1_front())
    print("verdict", verdict())


if __name__ == "__main__":
    main()
