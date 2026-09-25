"""분별의 끝 — 시작과 같은 기록 예산 문턱을 늦은 우주에 대면, 기록은 언제 팽창을 따라잡지 못하게 되는가.
원장: 43장 §43.110.

사용자(2026-09-26): "다음"(직전 제안: 기록 예산 기준을 우주의 끝에 대어 '분별이 끝나는 조건'을 세운다. 늦은 우주는 엔트로피가
  가득 차 있지 않으므로 가득 찰 때까지의 열역학을 먼저 세운다).

계산 전 고정(2026-09-26):
전제: §43.109의 기준(허블 부피당 e-fold당 기록 예산 B가 3c 이상이어야 새 부피의 확정이 팽창을 따라잡는다, c = 기록 하나의
  비용, 최소 ln 2). RL. CE 우주론(Ω_m = 코어 값, Ω_Λ = 1 − Ω_m, ln S_Λ = §43.106 감사 값 282.109, H_Λ = √(π/S_Λ)/t_P).
열역학(해석): 늦은 우주의 허블 부피(반지름 1/H)는 가득 차 있지 않다. 가득 참(S = S_Λ)까지 남은 자리는 안의 물질 에너지 E를
  T_GH로 나눈 만큼이다(첫째 법칙). 한 e-fold에 부피 밖으로 빠져나가는 물질 에너지가 새로 생기는 자리다:
  B = 2εS = 3(Ω_m a⁻³)/(Ω_Λ + Ω_m a⁻³) · S_Λ Ω_Λ/(Ω_Λ + Ω_m a⁻³) → 3(Ω_m/Ω_Λ)a⁻³S_Λ.
  같은 식이 급팽창에서는 2/P_ζ였다(§43.108). 곧 늦은 우주의 유효 P_ζ = 1/(εS)다.
주장:
  E1 전체의 끝: B = 3c ⟺ 허블 부피당 평균 물질 에너지 = c·k_B T_GH(1비트). ln a_end ≈ (1/3)ln[(Ω_m/Ω_Λ)S_Λ/c](오늘 a = 1).
     그 뒤 새 허블 부피는 확정되지 못한다(평균 부피 기준 상한). 시작(§43.109)과 같은 문턱을 반대 방향으로 지난다.
  E2 비교: 허블 부피당 평균 바리온이 1개가 되는 때(ln a_b = (1/3)ln N_b)가 먼저 온다(입자 한계). E1은 열역학 상한일 뿐이다.
  E3 우리 조각의 끝: 결국 묶인 국부 은하군만 남는다. 남은 기록 예산은 M_LG c²/(k_B T_GH c) 비트다(M_LG는 외부 값 2–5×10¹² M_sun).
     쓰이는 시간은 열적 죽음의 시간표(외부: Adams–Laughlin 1997)이고 여기서 계산하지 않는다.
  E4 [읽기] 끝난 뒤의 바다는 첫 사건 전의 바다와 같은 종류다(기록 없음, 유효 P_ζ → ∞). §43.104의 KMS 상태라 열적 시계는 돈다.
kill: K1 오늘 B ≤ 3c이면 기각(우리는 기록을 하고 있다). K2 c ∈ {ln 2, 1, 3 ln 2}에서 ln a_end의 폭이 0.5 e-fold를 넘으면 "끝은
  비용에 둔감하다"를 기각한다. K3 E1이 E2보다 먼저 오면(ln a_end < ln a_b) "E1은 입자 한계보다 느슨한 상한"이라는 해석을 기각한다.
예상(계산 전, 해석): K1 통과(오늘 B ~ e^{281}), K2 폭 (1/3)ln 3 = 0.37, K3 차이 ≈ (1/3)ln(m_p c²/k_B T_GH) ≈ 33 e-fold.
관측으로 가를 수 없는 먼 미래의 구조 계산이다. 위험한 예측이 아니다.

python -B -m examples.physics.rendering.ce_rendering_distinction_end
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_first_event as FEV

T_PLANCK_S = 5.391247e-44
HBAR_J_S = 1.054571817e-34
K_B = 1.380649e-23
C_M_S = 299792458.0
G_SI = 6.67430e-11
M_SUN_KG = 1.98892e30
M_PROTON_KG = 1.67262192e-27
GYR_S = 3.15576e16
COSTS = {"bit": math.log(2), "nat": 1.0, "3bit": 3 * math.log(2)}


def _cosmo() -> dict:
    k = FEV._core()
    om = k["c"]["Om"]
    ln_s_lam = FEV.audit()["ln_S_Lambda"]
    h_lam = math.sqrt(math.pi) * math.exp(-ln_s_lam / 2) / T_PLANCK_S          # S = π/(H t_P)²
    return {"om": om, "ol": 1 - om, "q": k["c"]["q"], "ln_S_L": ln_s_lam, "H_L": h_lam,
            "kT_GH_J": HBAR_J_S * h_lam / (2 * math.pi), "T_GH_K": HBAR_J_S * h_lam / (2 * math.pi * K_B)}


def budget_ln(ln_a: float, x: dict) -> float:
    """ln B(a): B = 2εS, ε = (3/2)Ω_m a⁻³/(Ω_Λ + Ω_m a⁻³), S = S_Λ Ω_Λ/(Ω_Λ + Ω_m a⁻³)."""
    m = x["om"] * math.exp(-3 * ln_a)
    return math.log(3 * m) + x["ln_S_L"] + math.log(x["ol"]) - 2 * math.log(x["ol"] + m)


def age_gyr(ln_a: float, x: dict) -> float:
    """평탄 물질 + Λ: t = (2/3H_Λ) arsinh(√(Ω_Λ/Ω_m) a^{3/2})."""
    y = 0.5 * math.log(x["ol"] / x["om"]) + 1.5 * ln_a
    arsinh = y + math.log(2) if y > 20 else math.asinh(math.exp(y))
    return 2 / (3 * x["H_L"]) * arsinh / GYR_S


def global_end() -> dict:
    """E1·E2: 새 허블 부피의 확정이 끝나는 때(비용별), 평균 바리온 1개의 때."""
    from scipy.optimize import brentq
    x = _cosmo()
    out = {"today_ln_budget": budget_ln(0.0, x), "T_GH_K": x["T_GH_K"], "age_today_Gyr": age_gyr(0.0, x)}
    for k, c in COSTS.items():
        ln_a = brentq(lambda la: budget_ln(la, x) - math.log(3 * c), 1.0, 200.0)
        out[k] = {"ln_a_end": ln_a, "t_end_Gyr": age_gyr(ln_a, x),
                  "approx": (math.log(x["om"] / x["ol"]) + x["ln_S_L"] - math.log(c)) / 3,
                  "from_equality": ln_a - math.log(x["om"] / x["ol"]) / 3}
    # 허블 부피(반지름 c/H_Λ)당 평균 바리온 수, 오늘
    h0_si = x["H_L"] / math.sqrt(x["ol"])
    n_b = x["q"] * 3 * h0_si ** 2 / (8 * math.pi * G_SI) / M_PROTON_KG
    ln_nb = math.log(n_b * 4 * math.pi / 3 * (C_M_S / x["H_L"]) ** 3)
    out["ln_N_baryon_per_hubble_volume_today"] = ln_nb
    out["ln_a_last_baryon"] = ln_nb / 3
    out["t_last_baryon_Gyr"] = age_gyr(ln_nb / 3, x)
    out["mirror"] = {"A1_capacity_ln_S_over_3": x["ln_S_L"] / 3,
                     "end_from_equality_bit": out["bit"]["from_equality"]}
    return out


def local_budget() -> dict:
    """E3: 묶인 국부 은하군이 앞으로 만들 수 있는 기록 비트 수의 상한."""
    x = _cosmo()
    return {f"M_LG={m:.0e}": m * M_SUN_KG * C_M_S ** 2 / (x["kT_GH_J"] * math.log(2)) for m in (2e12, 3e12, 5e12)}


def verdict() -> dict:
    g = global_end()
    ends = [g[k]["ln_a_end"] for k in COSTS]
    return {"K1_today_above_threshold": g["today_ln_budget"] > math.log(3 * COSTS["3bit"]),
            "K2_cost_insensitive": max(ends) - min(ends) <= 0.5,
            "K3_thermo_after_particle": g["bit"]["ln_a_end"] > g["ln_a_last_baryon"]}


def main() -> None:
    g = global_end()
    for k, v in g.items():
        print(k, v)
    print("local_budget_bits", {k: f"{v:.3g}" for k, v in local_budget().items()})
    print("verdict", verdict())


if __name__ == "__main__":
    main()
