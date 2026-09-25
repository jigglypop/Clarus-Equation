"""태초의 확정 거품 — 사건 하나가 퍼지는 분포·인과 압력·속도·경계, 그리고 남은 두 척도. 원장: 43장 §43.106.

사용자(2026-09-25–26):
- "태초에 아무것도 정해지지 않은 상태에 특정 하나로 정해진 사건이 있다고 할 때, 퍼져나가는 분포, 내부에서 받는 인과
  압력, 퍼져나가는 속도, 경계면의 상태를 표현해야 함."
- "순식간에 기하급수로 퍼져 인과사슬이 무량대수가 되지만, 전체로 보면 경계와 중앙의 차이가 크다. 식을 유효하게."
- "진행해봐. 식 남은 곳 풀 수 있을까."
전제(CE): 시작 전에는 분별이 없다(무게가 고름, 모듈러 시간 없음, §43.104 MK2). 첫 확정 사건 E₀. R1(빛원뿔 안에서만
퍼짐). BR1(기록마다 Poisson(D) 인과 후손, D = 3 + δ).

계산 전 고정(2026-09-26). 정의·정리(수학이라 통과가 예상되는 검산):
G1 로런츠: 사건 하나 말고 구조가 없으면 모든 양은 τ = √(t² − r²/c²)의 함수다. 같은 나이 면은 곡률 −1/(cτ)²의 쌍곡면이고
   a = cτ, H = 1/τ다. 곡률 항은 w = −1/3 유체라 ä = 0이다. 허블 법칙은 v = r/t이고 경계는 τ = 0이다.
G2 가지치기: q = e^{−D(1−q)}(= Ω_b), N_g/D^g → W(평균 1, 분산 1/(D−1), 생존 조건 평균 1/(1−q)). 모의실험으로 검산한다.
G3 유효성(포화): dA/dτ = λA(1 − A), λ = ln D/τ₀, A = N/N_max(§43.102 끼임 평균장에서 γ = 0).
   τ_s = ln(N_max − 1)/λ, 경계 껍질 두께 Δr ≈ cτ_s²/(2t), 실험실 능선 γ = t/τ_s.
G4 안에서 본 모습: 밀른 빛 경로에서 거리 χ의 모습은 τ_e = τe^{−χ} 때의 것이다. 확정 지평은 χ_s = ln(τ/τ_s)이고,
   경계(τ = 0)는 모든 방향의 χ → ∞(입자 지평)에 있다.
남은 두 척도의 닫힘 시도(후보는 계산 전 목록):
용량 N_max  C1 CE 지평 기록 수 S_CE: ln S = (π²/2)N_e − πδ(1−q). CE H0 식의 지수이고 §43.94–43.97의 로그 화소 수다.
            렌더링 영역(§43.104)의 용량으로서 주 후보다.
            C2 급팽창 지평 엔트로피 S_inf = π/(H_inf ℓ_P)². H_inf = π√(6A_s) M_P/N_e(CE A_s, r = 12/N_e²)이다.
            C3 급팽창 조각 수 e^{3N_e}.
다리 가설 B: 가지치기 동안 공간은 기록으로 이루어진다(a³ ∝ N). 포화에서 급팽창이 끝난다. 따라서 N_총 = ln N_max/3,
            Ω_k ≈ e^{−2(N_총 − N(a0H0))}, N(a0H0) = N_e + ln(k*/a0H0), k* = 0.05 Mpc⁻¹.
            kill: Ω_k > 0.0054(DESI DR2 + CMB 0.0021 ± 0.0011의 3σ 위).
            한 세대 τ₀ = ln D/(3H_inf)를 자연 척도(t_P, 1/H_inf, 2π/H_inf)와 비교해 보고한다(kill 아님).
부호 시험: 사건 하나의 기하는 Ω_k ≥ 0을 요구한다. Ω_k < 0이 3σ로 확정되면 기각한다. P24(Ω_k = 0)와 나란히 적는다.
한계: 다리 B 없이는 τ₀가 관측되지 않는다(포화가 수백 세대 안에 끝난다). C1은 늦은 우주 지평의 용량이다. 이것을 급팽창
끝의 포화 용량으로 쓰는 것은 범주가 어긋날 수 있으므로 보고한다.

추가(2026-09-26, 사용자 "좀더 고도화해보자. 맞게 해주는 값이 있을거 수학적으로"). 계산 전 고정:
H1 한 세대: 다리 B 안에서 정상성(기록 밀도 일정, 자기닮음)이 τ₀를 강제한다. 한 세대 동안 부피가 D배, 길이가 D^{1/n}배,
   τ₀H = ln D/n이다. 인과(자손은 빛원뿔 안)로 허블 부피당 기록 수는 (4π/3)(n/ln D)³ 이상이다. 이 수는 관측량에 들어가지 않는다.
H2 다리의 지수 n: 부피(n = 3)와 면적(n = 2, HP 화소가 면적)을 함께 본다. N_총 = ln S_CE/n.
H3 C1의 범주 단서: 렌더링 영역은 역사 전체에서 하나(§43.104 MK7)라 용량 S_CE는 시간 조각이 아니라 영역의 것이다(논증).
H4 [목격 후] DESI 기미를 맞추는 N_총 ≈ 65.7이 §43.97의 "급팽창 뒤 65.70 e-fold"와 같아 보였다(급팽창 길이 = 급팽창 뒤 길이).
   65.70은 순간 재가열 값이고 N(a0H0) = 62.6은 CE N_e(단단한 재가열) 값이다. N_* + N_after = ln(H_inf/H0) − ln(k*/a0H0)로
   같은 가정 안에서 다시 계산해 판정한다. 가정을 섞어야만 맞으면 착시로 적고 버린다.
감사(2026-09-26, 사용자 "정확도 다시 체크해봐", 독립 검산 에이전트): 숫자는 모두 재현되었다. H3는 다리 B와 모순이라 철회한다
(관측자 영역에는 e^{187.8}개만 들어가고, 용량이 영역의 것이면 Ω_k = 0.76). "곡률 유체가 중력과 비긴다"도 철회한다(ρ + 3p = 0).
audit()이 감사 수치를 재현한다.

python -B -m examples.physics.rendering.ce_rendering_first_event
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_planck_readout as PL
from examples.physics.rendering import ce_rendering_registry as R

C_KM_S = 299792.458
M_P_REDUCED_GEV = 2.435e18
M_PL_GEV = 1.220890e19
HBAR_GEV_S = 6.582119569e-25
T_PLANCK_S = 5.391247e-44
K_PIVOT = 0.05                       # Mpc⁻¹
OMEGA_K_DATA = (0.0021, 0.0011)      # DESI DR2 + CMB
MURYANG = 1e68


def _core() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    return {"c": c, "D": c["D"], "q": c["q"], "d": c["d"], "Ne": c["Ne"], "A_s": R.scalar_amplitude(c) * 1e-9,
            "H0": 100 * PL.h_rings(c)}


# ------------------------------------------------------------------ G1–G4
def lorentz_geometry(samples: int = 200, seed: int = 0) -> dict:
    """G1: 부스트해도 τ는 그대로고, 같은 τ의 점들은 서로 v = r/t로 멀어진다(로런츠 합성 속도로 검산)."""
    rng = np.random.default_rng(seed)
    worst_tau, worst_hubble = 0.0, 0.0
    for _ in range(samples):
        t = rng.uniform(1, 5)
        r = rng.uniform(0, 0.99) * t
        tau = math.sqrt(t * t - r * r)
        beta = rng.uniform(-0.95, 0.95)
        g = 1 / math.sqrt(1 - beta * beta)
        tp, rp = g * (t - beta * r), g * (r - beta * t)                     # 1+1 부스트
        worst_tau = max(worst_tau, abs(math.sqrt(tp * tp - rp * rp) - tau))
        v1, v2 = r / t, rng.uniform(-0.99, 0.99)                             # 두 기록의 E₀ 틀 속도
        rel = (v1 - v2) / (1 - v1 * v2)                                      # 한 기록에서 본 다른 기록의 속도
        # 같은 나이 τ에서 두 기록의 쌍곡 거리 ρ = cτ·|artanh v1 − artanh v2|, 멀어지는 속도 = tanh(ρ/cτ) ≈ ρ/τ(허블)
        worst_hubble = max(worst_hubble, abs(abs(rel) - math.tanh(abs(math.atanh(v1) - math.atanh(v2)))))
    return {"tau_invariance_err": worst_tau, "hubble_rapidity_err": worst_hubble, "curvature_w": -1 / 3,
            "passed": worst_tau < 1e-12 and worst_hubble < 1e-12}


def branching(runs: int = 4000, gens: int = 30, seed: int = 0) -> dict:
    """G2: 갈턴–왓슨(Poisson(D)) 통계와 모의실험."""
    k = _core()
    D, q = k["D"], k["q"]
    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(runs):
        n = 1
        for _g in range(gens):
            n = int(rng.poisson(D * n)) if n < 10 ** 6 else int(round(n * D))
            if n == 0:
                break
        ws.append(n / D ** gens if n else 0.0)
    ws = np.array(ws)
    alive = ws > 0
    ln_s = (math.pi ** 2 / 2) * k["Ne"] - math.pi * k["d"] * (1 - q)
    return {"D": D, "q": q, "var_W": 1 / (D - 1), "mean_W_alive": 1 / (1 - q),
            "mc_extinct": float(1 - alive.mean()), "mc_mean_W_alive": float(ws[alive].mean()),
            "mc_W_alive_5_95": (float(np.percentile(ws[alive], 5)), float(np.percentile(ws[alive], 95))),
            "gens_to_muryang": math.log(MURYANG) / math.log(D), "ln_S_CE": ln_s,
            "gens_to_S_CE": ln_s / math.log(D)}


def logistic_profile(n_max_ln: float, t_over_ts: float, points: int = 7) -> dict:
    """G3: E₀ 틀의 확정도 A(t, r)와 껍질. 단위 c = 1, τ_s = 1."""
    lam = n_max_ln                                                           # λτ_s = ln N_max(근사), τ_s = 1
    t = t_over_ts
    rs = np.linspace(0, t, points)
    tau = np.sqrt(np.maximum(t * t - rs * rs, 0))
    a = 1 / (1 + np.exp(n_max_ln - lam * tau))
    skin = t - math.sqrt(t * t - 1)
    return {"r_over_ct": (rs / t).round(3).tolist(), "A": a.round(6).tolist(), "skin_over_ct": skin / t,
            "skin_approx": 1 / (2 * t * t), "lab_ridge_gamma": t}


def inside_view(tau_over_ts: float) -> dict:
    """G4: 확정 지평 χ_s = ln(τ/τ_s)."""
    return {"chi_s": math.log(tau_over_ts), "tau_e_at": {x: round(tau_over_ts * math.exp(-x), 4) for x in (0.5, 1, 2, 4)}}


# ------------------------------------------------------------------ 남은 두 척도
def inflation_scale() -> dict:
    k = _core()
    h_red = math.pi * math.sqrt(6 * k["A_s"]) / k["Ne"]                      # H_inf / M_P(환산)
    h_gev = h_red * M_P_REDUCED_GEV
    s_inf = math.pi / (h_gev / M_PL_GEV) ** 2
    return {"H_inf_over_MP": h_red, "H_inf_GeV": h_gev, "S_inf": s_inf, "ln_S_inf": math.log(s_inf),
            "hubble_time_s": HBAR_GEV_S / h_gev}


def closure() -> dict:
    """용량 후보 C1–C3 × 다리 B: 곡률과 한 세대 척도."""
    k = _core()
    D, ne = k["D"], k["Ne"]
    inf = inflation_scale()
    n_hub = ne + math.log(K_PIVOT * C_KM_S / k["H0"])                       # 오늘 지평 규모가 빠져나간 e-fold 수
    caps = {"C1_S_CE": branching()["ln_S_CE"], "C2_S_inf": inf["ln_S_inf"], "C3_patches": 3 * ne}
    upper = OMEGA_K_DATA[0] + 3 * OMEGA_K_DATA[1]
    out = {"N_at_a0H0": n_hub, "Omega_k_upper_3sigma": upper, "caps": {}}
    for name, ln_n in caps.items():
        n_tot = ln_n / 3
        ok = math.exp(min(-2 * (n_tot - n_hub), 700.0))
        out["caps"][name] = {"ln_N_max": ln_n, "N_total": n_tot, "Omega_k": ok, "killed": ok > upper}
    tau0 = math.log(D) / (3 * inf["H_inf_GeV"]) * HBAR_GEV_S
    out["tick"] = {"tau0_s": tau0, "over_t_P": tau0 / T_PLANCK_S, "over_hubble_time": tau0 / inf["hubble_time_s"],
                   "over_thermal_2pi": tau0 / (2 * math.pi * inf["hubble_time_s"])}
    return out


def stationary_tick() -> dict:
    """H1·H2: 다리 B 안의 정상성. 한 세대 동안 부피 D배 → τ₀H = ln D/n, 길이 D^{1/n}배."""
    k = _core()
    D = k["D"]
    ln_s = branching()["ln_S_CE"]
    n_hub = k["Ne"] + math.log(K_PIVOT * C_KM_S / k["H0"])
    out = {}
    for n in (3, 2):
        n_tot = ln_s / n
        out[f"n={n}"] = {"tau0_H": math.log(D) / n, "magnification_per_generation": D ** (1 / n),
                         "generations_per_efold": n / math.log(D), "N_total": n_tot,
                         "Omega_k": math.exp(-2 * (n_tot - n_hub)),
                         "min_records_per_hubble_volume": (4 * math.pi / 3) * (n / math.log(D)) ** 3}
    return out


def balance_check() -> dict:
    """H4 [목격 후]: 급팽창 길이 = 급팽창 뒤 길이. 같은 가정 안에서 다시 계산한다."""
    from examples.physics.rendering import ce_rendering_parabolic_lock as PLK
    k = _core()
    inf = inflation_scale()
    h0_gev = k["H0"] / 3.0856775814913673e19 * HBAR_GEV_S
    ln_k = math.log(K_PIVOT * C_KM_S / k["H0"])
    total = math.log(inf["H_inf_GeV"] / h0_gev) - ln_k                         # N_* + N_after
    n_after_instant = PLK.constants()["n_after_efolds"]                          # §43.97: 65.70
    n_star_instant = total - n_after_instant
    cases = {"CE_stiff(N_*=N_e)": k["Ne"], "instant(N_* from 65.70)": n_star_instant}
    out = {"N_star_plus_N_after": total, "n_after_instant": n_after_instant, "n_star_instant": n_star_instant}
    for name, n_star in cases.items():
        n_after = total - n_star
        n_hub = n_star + ln_k
        out[name] = {"N_after": n_after, "N_at_a0H0": n_hub, "Omega_k_if_Ntotal_eq_Nafter": math.exp(-2 * (n_after - n_hub))}
    out["mixed_artifact"] = math.exp(-2 * (n_after_instant - (k["Ne"] + ln_k)))   # 가정을 섞은 값
    upper = OMEGA_K_DATA[0] + 3 * OMEGA_K_DATA[1]
    out["consistent_cases_fit"] = [n for n in cases if 1e-4 < out[n]["Omega_k_if_Ntotal_eq_Nafter"] < upper]
    return out


def audit() -> dict:
    """정확도 재점검(2026-09-26, 독립 검산 뒤). 곡률 바닥, 관측자 영역의 기록 수, '용량 = 관측자 영역'이면 곡률,
    최종 지평 용량 S_Λ, DESI 기미에 맞춘 재가열."""
    from scipy.integrate import quad
    k = _core()
    om, ne = k["c"]["Om"], k["Ne"]
    ln_k = math.log(K_PIVOT * C_KM_S / k["H0"])
    n_hub = ne + ln_k
    upper = OMEGA_K_DATA[0] + 3 * OMEGA_K_DATA[1]
    gap = 0.5 * math.log(1 / upper)
    ln_s_ce = branching()["ln_S_CE"]
    ln_s_lam = ln_s_ce - math.log(1 - om)                                    # S_Λ = S_H0/Ω_Λ(같은 H0)
    chi_e = quad(lambda a: 1 / (a * a * math.sqrt(om / a ** 3 + 1 - om)), 1, math.inf)[0]   # 오늘 사건 지평 / (c/H0)
    inf = inflation_scale()
    h0_gev = k["H0"] / 3.0856775814913673e19 * HBAR_GEV_S
    total = math.log(inf["H_inf_GeV"] / h0_gev) - ln_k
    tune = lambda om_k: (total - ln_k - 0.5 * math.log(1 / om_k)) / 2        # 균형 + 이 Ω_k를 주는 N_*
    return {"floor_ln_Nmax_volume": 3 * (n_hub + gap), "floor_ln_Nmax_area": 2 * (n_hub + gap),
            "ln_records_in_observer_domain": 3 * n_hub,
            "ln_records_in_domain_at_min_density": 3 * n_hub + math.log(stationary_tick()["n=3"]["min_records_per_hubble_volume"]),
            "event_horizon_over_hubble": chi_e, "Omega_k_if_bubble_is_domain": 1 / chi_e ** 2,
            "ln_S_Lambda": ln_s_lam, "N_total_S_Lambda_volume": ln_s_lam / 3, "N_total_S_Lambda_area": ln_s_lam / 2,
            "Omega_k_S_Lambda_volume": math.exp(-2 * (ln_s_lam / 3 - n_hub)),
            "tuned_N_star_for_DESI": tune(OMEGA_K_DATA[0]),
            "tuned_N_star_1sigma": (tune(OMEGA_K_DATA[0] + OMEGA_K_DATA[1]), tune(OMEGA_K_DATA[0] - OMEGA_K_DATA[1]))}


def sign_test() -> dict:
    """사건 하나의 기하: Ω_k ≥ 0. P24: Ω_k = 0."""
    m, s = OMEGA_K_DATA
    return {"data": OMEGA_K_DATA, "pull_of_zero": (0 - m) / s, "open_killed": m + 3 * s < 0, "P24_killed": abs(m / s) > 3}


def main() -> None:
    print("G1 Lorentz:", lorentz_geometry())
    b = branching()
    print("G2 branching:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})
    print("G3 profile (ln N_max = ln S_CE, t = 10 tau_s):", logistic_profile(b["ln_S_CE"], 10.0))
    print("G4 inside view (tau = 100 tau_s):", inside_view(100.0))
    print("inflation scale:", inflation_scale())
    print("closure:", closure())
    print("sign test:", sign_test())
    print("H1/H2 stationary tick:", stationary_tick())
    print("H4 balance check (post hoc):", balance_check())


if __name__ == "__main__":
    main()
