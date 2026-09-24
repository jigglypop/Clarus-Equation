"""PH5(T/ω = N_e)의 독립 근거 시험과 N_e의 표준 재가열 정합. 원장: 43장 §43.95. 예측값을 바꾸지 않는다.

§43.94의 원리 PH(세 축 NS 마요라나 기록 탑)는 선도 계수 π²/2를 주지만 전제 PH5(T/ω = N_e)가 새 선택이었다.
x = βω는 좌표에 무관하고 공간 원과 열적 원의 길이 비 L/β로만 정해진다(주기 모드 ω = 2π/L이면 x = 2πβ/L).
계산 전에 적은 규칙. 사전 목격: 자연 원환의 x = 4π²/N_e와 순간 재가열의 N_* ≈ 55.6을 암산으로 먼저 보았다.

(가) 자연 원환 후보(기존 공리만): 공간 원 = 급팽창 기간 N_e e-fold(TT: e-fold마다 기록 위상 1 rad),
    열적 원 = 유클리드 주기 2π e-fold(Gibbons–Hawking, C2). N1 주기 모드, N2 구간 모드(ω = π/L),
    N3 모듈러 흐름 정규화(시간을 s = N/2π로 잰 같은 원환 — 불변성 확인). 하나라도 x = 1/N_e이면 PH5 뒷받침.
(나) π²를 주는 다른 원리: 폭 L의 흡수 경계 구간에서 확산 계수 D의 생존 확률 P ≈ A e^{−π² D t/L²}.
    D t/L² = N_e/4로 두면 선도항이 (π²/4) N_e = ln(M_Pl/H)의 선도항과 같다. 원리 고유의 상수 −ln A
    (D1 가운데 출발 A = 4/π, D2 고른 출발 A = 8/π²)를 필요한 상수(오늘·최종 지평선, ±σ)와 비교. 2σ 안이면 적중.
(다) N_e의 표준 정합: N_* = ln(a_end/a_reh) + ln(a_reh/a₀) + ln(H_*/H₀) − ln(k_*/a₀H₀),
    엔트로피 보존 a_reh/a₀ = (g_s0/g_reh)^{1/3} T₀/T_reh, ρ_reh = (π²/30) g_reh T_reh⁴, 순간 재가열 ρ_reh = ρ_end,
    H_*² = V_*/(3M²), V_* = (3π²/2) A_s r M⁴(CE의 A_s와 r = 12/N_e²), ρ_end = (3/2) V_end(Starobinsky 퍼텐셜, ε_V = 1),
    k_* = 0.05 Mpc⁻¹, H₀ = CE 이른 판독, T₀ = 2.7255 K, g_s0 = 3.931, g_reh = 106.75.
    판정: N_e − N_*,순간 ≤ 0.5이면 정합(w ≤ 1/3 재가열 허용). 크면 CE는 w > 1/3 재가열을 요구한다(기각 아님).
    요구량: 일정한 w의 재가열은 N_*를 Δ(3w − 1)/(12(1 + w)) 바꾼다(Δ = ln(ρ_end/ρ_reh)). w = 1에서 필요한 Δ와 T_reh를 보고.

python -B -m examples.physics.rendering.ce_rendering_tower_temperature
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_horizon_towers as HTW
from examples.physics.rendering import ce_rendering_registry as R

PI = math.pi
M_RED_GEV = 2.435e18
GEV_PER_KMSMPC = 2.1332e-44           # H₀ = 1 km/s/Mpc in GeV (ħ = 1)
K_TO_GEV = 8.617333e-14
T0_K = 2.7255
G_S0, G_REH = 3.931, 106.75
K_PIVOT = 0.05                         # Mpc⁻¹
C_KMS = 299792.458


def _core() -> dict:
    return R.core(R.calibrated_alpha_s()[0])


# ------------------------------------------------------------------ (가) 자연 원환
def ns_three_towers_entropy(x: float) -> float:
    lnz, be = HTW.tower("NS", x)
    return 3.0 * (lnz + be)


def natural_tori() -> dict:
    ne = _core()["Ne"]
    beta = 2.0 * PI                    # 유클리드 주기(e-fold)
    cands = {
        "N1 TT 주기 모드": 2.0 * PI * beta / ne,
        "N2 TT 구간 모드": PI * beta / ne,
        "N3 모듈러 정규화": 2.0 * PI * (beta / (2 * PI)) / (ne / (2 * PI)),   # 같은 원환을 s = N/2π로 잰 것
    }
    out = {}
    for name, x in cands.items():
        s = ns_three_towers_entropy(x)
        out[name] = {"x": x, "x_times_Ne": x * ne, "S": s, "S_over_Ne": s / ne}
    need = HTW.LEAD * ne
    x1 = cands["N1 TT 주기 모드"]
    return {"candidates": out, "PH5_x": 1.0 / ne, "supported": any(abs(c * ne - 1.0) < 1e-9 for c in cands.values()),
            "gap_factor": x1 * ne, "lead_over_natural": need / out["N1 TT 주기 모드"]["S"],
            "natural_S_over_Ne_is_one_eighth": abs(out["N1 TT 주기 모드"]["S_over_Ne"] - 0.125) < 1e-6}


# ------------------------------------------------------------------ (나) 확산 생존
def diffusion_principle() -> dict:
    t = HTW.target()
    c = _core()
    loop = 1.0 + c["a"] / (4.0 * PI)
    ln_mh_today = t["S_needed"] / 2.0 - 0.5 * math.log(PI) - math.log(loop)
    ln_mh_final = ln_mh_today - 0.5 * math.log(1.0 - c["Om"])
    lead = (PI ** 2 / 4.0) * c["Ne"]
    sig = t["sigma_S"] / 2.0
    need = {"today": ln_mh_today - lead, "final": ln_mh_final - lead}
    cands = {"D1 가운데 출발 (A = 4/π)": -math.log(4.0 / PI), "D2 고른 출발 (A = 8/π²)": -math.log(8.0 / PI ** 2)}
    out = {name: {tg: (v - need[tg]) / sig for tg in need} for name, v in cands.items()}
    hits = [n for n, d in out.items() if min(abs(p) for p in d.values()) <= 2.0]
    return {"need": need, "sigma": sig, "pulls": out, "hits": hits}


# ------------------------------------------------------------------ (다) N_e의 표준 정합
def starobinsky_end_ratio(ne: float) -> float:
    """V_end/V_* for V = V₀(1 − e^{−y})², y = √(2/3)φ. 선도 차수 e^{−y_*} = 3/(4N), 끝은 ε_V = 1."""
    u_star = 3.0 / (4.0 * ne)
    u_end = 1.0 / (1.0 + 2.0 / math.sqrt(3.0))
    return (1.0 - u_end) ** 2 / (1.0 - u_star) ** 2


def matching() -> dict:
    c = _core()
    ne = c["Ne"]
    a_s = R.scalar_amplitude(c) * 1e-9
    r = 12.0 / ne ** 2
    v_star = 1.5 * PI ** 2 * a_s * r                                   # M_red⁴ 단위
    h_star = M_RED_GEV * math.sqrt(v_star / 3.0)
    rho_end = 1.5 * starobinsky_end_ratio(ne) * v_star * M_RED_GEV ** 4
    t_reh = (30.0 * rho_end / (PI ** 2 * G_REH)) ** 0.25
    h0_kms = R.hubble_readout(c, False) * (1.0 + c["a"] / (4.0 * PI))
    h0_gev = h0_kms * GEV_PER_KMSMPC
    ln_a_reh_a0 = math.log(T0_K * K_TO_GEV / t_reh) + math.log(G_S0 / G_REH) / 3.0
    n_inst = ln_a_reh_a0 + math.log(h_star / h0_gev) - math.log(K_PIVOT / (h0_kms / C_KMS))
    excess = ne - n_inst
    delta_w1 = 12.0 * excess * (1.0 + 1.0) / (3.0 * 1.0 - 1.0) if excess > 0 else 0.0
    return {"N_e": ne, "N_star_instant": n_inst, "excess": excess, "consistent": excess <= 0.5,
            "H_star_GeV": h_star, "T_reh_instant_GeV": t_reh,
            "kination_ln_rho_end_over_rho_reh": delta_w1, "kination_T_reh_GeV": t_reh * math.exp(-delta_w1 / 4.0),
            "w0_reheating_change_per_Delta": (3 * 0.0 - 1) / (12 * (1 + 0.0))}


def main() -> None:
    nt = natural_tori()
    for name, d in nt["candidates"].items():
        print(f"(가) {name}: x·N_e = {d['x_times_Ne']:.4f}  S/N_e = {d['S_over_Ne']:.5f}")
    print("    PH5 supported:", nt["supported"], "| gap factor x·N_e =", round(nt["gap_factor"], 4),
          "(4π² =", round(4 * PI ** 2, 4), ") | lead / natural S =", round(nt["lead_over_natural"], 4),
          "| natural S/N_e = 1/8:", nt["natural_S_over_Ne_is_one_eighth"])
    d = diffusion_principle()
    print("(나) need:", {k: round(v, 4) for k, v in d["need"].items()}, "σ", round(d["sigma"], 4))
    for name, p in d["pulls"].items():
        print(f"     {name}: pulls {({k: round(v, 1) for k, v in p.items()})}")
    print("     hits:", d["hits"])
    m = matching()
    print("(다) matching:", {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == "__main__":
    main()
