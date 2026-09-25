"""거품 용량의 후보 — 첫 칸에서 자란 거품 전체의 기록 용량(곧 급팽창 총길이)을 정할 원리를 최대한 모아 한 번에 대조한다.
원장: 43장 §43.107.

사용자(2026-09-26): "후보좀 최대한 제시해봐." 배경: §43.106 감사에서 "용량 = 관측자 영역"(H3)이 다리 B와 모순이라 철회되었고,
용량은 미결이다. 다리 B(a^n ∝ 기록 수)에서 N_총 = (ln N_max − ln N_H)/n이다. N_H는 첫 칸(허블 부피)의 기록 수다.

계산 전 고정(2026-09-26):
kill K-curv: Ω_k ≈ e^{−2(N_총 − N(a0H0))} > 0.0054(DESI DR2 + CMB 0.0021 ± 0.0011의 3σ 위), 또는 N_총 ≤ N(a0H0)(Ω_k = O(1)).
창: 관측 창 10⁻⁵ < Ω_k < 0.0054, DESI 1σ 창 0.001–0.0032. DESI는 동결 전 자료라 창에 들어도 적중이 아니라 보류 자료 확인이다.
기록 밀도 민감도: N_H ∈ {1, 73(인과 하한, §43.106 H1), S_inf(홀로그래피 상한)}.
범주 단서: 갈래 A는 관측자 영역의 양을 거품 용량으로 쓰는 것이라 §43.106 감사의 문제를 그대로 안는다(판정은 하되 표시).
후보(모두 계산 전 목록):
A 관측자 영역의 개수(다리 n = 3, 2): A1 최종 지평 엔트로피 S_Λ, A2 허블 구 엔트로피 S_H0(CE H0 식 지수),
  A3 허블 부피의 복사 엔트로피(s₀ = 2891.2 cm⁻³), A4 허블 부피의 광자 수(n_γ = 410.73 cm⁻³),
  A5 허블 부피의 바리온 수(CE q, h), A6 관측 가능한 우주의 초대질량 블랙홀 엔트로피(외부 값 3.1×10¹⁰⁴,
  Egan & Lineweaver 2010), A7 허블 부피의 플랑크 부피 수.
B 급팽창 총길이 원리: B1 딱 필요한 만큼 N(a0H0), B2 N_e, B3 ζ(2)N_e(바젤 역사, §43.97), B4 2N_e(36D, 생성자 24),
  B5 12D(Bool 8통로), B6 TT 한 바퀴 2π(급팽창 H에서 e-fold당 1 rad), B7 TT 렌더링 창 π/2,
  B8 거울(첫 사건–피벗 = 피벗–오늘, N_총 = N_e + N_* + N_after).
C 물리적 상한(다리): C1 급팽창 패치 드 시터 엔트로피 S_inf, C2 첫 칸의 란다우어 에너지 기록 수 S_inf/ln 2,
  C3 전체 무게/q(1/q), C4 무량대수 10⁶⁸, C5 e^{3N_e}.
D CE 식 안의 수(숫자 맞추기 위험, 세어서 보고): D1 (π²/2)N_e, D2 12N_e, D3 2πN_e, D4 ln S_Λ + ln S_inf.
E 거품 자체의 홀로그래피: E1 부피 다리 + 거품 면적 용량(N_H e^{3N} = S_inf e^{2N} → N_총 = ln S_inf − ln N_H),
  E2 면적 다리 + 면적 용량(포화 없음 → 미결).
F 다리 없음: F1 분별의 시작 = 양자 확산의 끝(P_ζ = 1): 스타로빈스키형 P_ζ(N) = A_s(N/N_e)²에서 N_q = N_e/√A_s,
  F2 영원한 급팽창(N_총 → ∞), F3 경관 전형성(분포 예측, 값 없음, 목록만).

추가(2026-09-26, 사용자 "진행해봐 니가 봤을때 가능성있는곳부터"). 위 대조를 본 뒤, 파기 전에 고정했다:
순위(판단): F1(원리·무매개변수·다리 불필요) > A3·A4(관측 창, 원리 약함) > B3(구조, 시험 불가).
F1a 정밀: 정확한 스타로빈스키 V = V₀(1 − e^{−κφ})², κ = √(2/3)/M_P. x = e^{κφ}로 N(x) = (3/4)[(x − x_end) − ln(x/x_end)],
    x_end = 1 + 2/√3, P_ζ ∝ (x − 1)⁴/x². V₀는 N(x_*) = N_e에서 P_ζ = A_s로 맞춘다. P_ζ(x_q) = 1의 N_q를 근사 N_e/√A_s와 비교한다.
F1b 문턱: 양자 확산 경계 P_ζ ∈ {0.1, 1, 10}(인자 몇 배의 기준 차이). N_q ∝ P^{−1/2}. 곡률 판정이 바뀌면 보고한다.
F1c 기하: 확산에서 빠져나오는 곳은 점이 아니라 영역이다. 그래서 §43.106 G1의 로런츠 불변 열린 거품(점 사건)은 F1에 적용되지 않고,
    주머니는 평탄 조각이다(Ω_k = e^{−2(N_q − N(a0H0))}). 이것을 계산이 아니라 논증으로 적는다.
F1 kill: Ω_k ≠ 0이 3σ로 확정되면 F1(과 P24)은 기각된다.
A-RL 시험: A3·A4를 "기록 = 엔트로피 버림(RL, 비트당 k_B ln 2)"으로 정당화할 수 있는가. 기록이 엔트로피처럼 부피에 비례하면(크기 성질)
    거품의 기록 수도 부피에 비례해 포화가 없다. 그러면 용량이 될 수 없다(구조 판정). 창 적중의 우연 기대는 칸 수로 센다.

python -B -m examples.physics.rendering.ce_rendering_bubble_capacity
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_first_event as FEV

C_KM_S = 299792.458
MPC_CM = 3.0856775814913673e24
L_PLANCK_CM = 1.616255e-33
S0_CM3 = 2891.2                 # 오늘 엔트로피 밀도(광자 + 중성미자, k_B 단위)
N_GAMMA_CM3 = 410.73            # CMB 광자 수 밀도
RHO_C_H2_G_CM3 = 1.87834e-29
M_PROTON_G = 1.67262192e-24
S_SMBH_OBS = 3.1e104            # Egan & Lineweaver 2010(입자 지평 부피), 외부 값
UPPER = 0.0021 + 3 * 0.0011
WINDOW_OBS = (1e-5, UPPER)
WINDOW_DESI = (0.001, 0.0032)


def _ctx() -> dict:
    k = FEV._core()
    c = k["c"]
    ne, h = k["Ne"], k["H0"] / 100
    r_h = C_KM_S / k["H0"] * MPC_CM
    v_h = 4 * math.pi / 3 * r_h ** 3
    fb = FEV.branching()
    au = FEV.audit()
    inf = FEV.inflation_scale()
    bc = FEV.balance_check()
    n_b = c["q"] * h * h * RHO_C_H2_G_CM3 / M_PROTON_G
    return {"ne": ne, "D": k["D"], "q": c["q"], "A_s": k["A_s"],
            "n_hub": ne + math.log(0.05 * C_KM_S / k["H0"]), "ln_S_inf": inf["ln_S_inf"],
            "ln_S_H0": fb["ln_S_CE"], "ln_S_L": au["ln_S_Lambda"], "v_h": v_h,
            "n_after": bc["CE_stiff(N_*=N_e)"]["N_after"], "n_b": n_b}


def candidates() -> list[dict]:
    x = _ctx()
    ne, D = x["ne"], x["D"]
    ln = lambda v: math.log(v)
    out = [
        {"id": "A1", "label": "최종 지평 엔트로피 S_Λ", "ln_N": x["ln_S_L"]},
        {"id": "A2", "label": "허블 구 엔트로피 S_H0", "ln_N": x["ln_S_H0"]},
        {"id": "A3", "label": "허블 부피 복사 엔트로피", "ln_N": ln(S0_CM3 * x["v_h"])},
        {"id": "A4", "label": "허블 부피 광자 수", "ln_N": ln(N_GAMMA_CM3 * x["v_h"])},
        {"id": "A5", "label": "허블 부피 바리온 수", "ln_N": ln(x["n_b"] * x["v_h"])},
        {"id": "A6", "label": "초대질량 블랙홀 엔트로피(외부)", "ln_N": ln(S_SMBH_OBS)},
        {"id": "A7", "label": "허블 부피 플랑크 부피 수", "ln_N": ln(x["v_h"] / L_PLANCK_CM ** 3)},
        {"id": "B1", "label": "딱 필요한 만큼", "N_total": x["n_hub"]},
        {"id": "B2", "label": "N_e", "N_total": ne},
        {"id": "B3", "label": "ζ(2)N_e 바젤 역사", "N_total": math.pi ** 2 / 6 * ne},
        {"id": "B4", "label": "2N_e(생성자 24)", "N_total": 2 * ne},
        {"id": "B5", "label": "12D(Bool 8통로)", "N_total": 12 * D},
        {"id": "B6", "label": "TT 한 바퀴 2π", "N_total": 2 * math.pi},
        {"id": "B7", "label": "TT 렌더링 창 π/2", "N_total": math.pi / 2},
        {"id": "B8", "label": "거울(첫 사건–피벗 = 피벗–오늘)", "N_total": ne + ne + x["n_after"]},
        {"id": "C1", "label": "급팽창 패치 드 시터 엔트로피", "ln_N": x["ln_S_inf"]},
        {"id": "C2", "label": "첫 칸 란다우어 기록 수", "ln_N": x["ln_S_inf"] - ln(math.log(2))},
        {"id": "C3", "label": "전체 무게/q", "ln_N": ln(1 / x["q"])},
        {"id": "C4", "label": "무량대수 10^68", "ln_N": 68 * math.log(10)},
        {"id": "C5", "label": "e^{3N_e}", "ln_N": 3 * ne},
        {"id": "D1", "label": "(π²/2)N_e", "ln_N": math.pi ** 2 / 2 * ne},
        {"id": "D2", "label": "12N_e", "ln_N": 12 * ne},
        {"id": "D3", "label": "2πN_e", "ln_N": 2 * math.pi * ne},
        {"id": "D4", "label": "ln S_Λ + ln S_inf", "ln_N": x["ln_S_L"] + x["ln_S_inf"]},
        {"id": "E1", "label": "부피 다리 + 거품 면적 용량", "N_total_fn": lambda nh: x["ln_S_inf"] - math.log(nh)},
        {"id": "F1", "label": "분별의 시작 = 양자 확산의 끝(P_ζ = 1)", "N_total": ne / math.sqrt(x["A_s"])},
        {"id": "F2", "label": "영원한 급팽창", "N_total": math.inf},
    ]
    return out


def _omega(n_tot: float, n_hub: float) -> float:
    if n_tot <= n_hub:
        return math.inf                                                       # Ω_k = O(1)
    return math.exp(-2 * (n_tot - n_hub)) if math.isfinite(n_tot) else 0.0


def evaluate() -> dict:
    x = _ctx()
    n_hub = x["n_hub"]
    nh_grid = {"1": 1.0, "73": FEV.stationary_tick()["n=3"]["min_records_per_hubble_volume"], "S_inf": math.exp(x["ln_S_inf"])}
    rows = []
    for c in candidates():
        cases = {}
        if "ln_N" in c:
            for n in (3, 2):
                for key, nh in nh_grid.items():
                    cases[f"n={n},N_H={key}"] = (c["ln_N"] - math.log(nh)) / n
        elif "N_total_fn" in c:
            for key, nh in nh_grid.items():
                cases[f"N_H={key}"] = c["N_total_fn"](nh)
        else:
            cases["direct"] = c["N_total"]
        res = {}
        for k, n_tot in cases.items():
            om = _omega(n_tot, n_hub)
            res[k] = {"N_total": n_tot, "Omega_k": om, "killed": om > UPPER,
                      "obs_window": WINDOW_OBS[0] < om < WINDOW_OBS[1], "desi_1sigma": WINDOW_DESI[0] <= om <= WINDOW_DESI[1]}
        rows.append({**{k: v for k, v in c.items() if k != "N_total_fn"}, "cases": res})
    base_key = lambda r: "n=3,N_H=1" if "ln_N" in r else ("N_H=1" if "N_H=1" in r["cases"] else "direct")
    summary = {"n_candidates": len(rows), "n_hub": n_hub,
               "killed_base": [r["id"] for r in rows if r["cases"][base_key(r)]["killed"]],
               "obs_window_any": sorted({r["id"] for r in rows for v in r["cases"].values() if v["obs_window"]}),
               "desi_window_any": sorted({r["id"] for r in rows for v in r["cases"].values() if v["desi_1sigma"]}),
               "survive_base": [r["id"] for r in rows if not r["cases"][base_key(r)]["killed"]]}
    return {"rows": rows, "summary": summary}


def starobinsky_exit() -> dict:
    """F1a·F1b: 정확한 스타로빈스키에서 P_ζ = 문턱이 되는 e-fold 수(양자 확산의 끝 = 분별의 시작)."""
    from scipy.optimize import brentq
    x = _ctx()
    ne, a_s = x["ne"], x["A_s"]
    x_end = 1 + 2 / math.sqrt(3)
    n_of = lambda y: 0.75 * ((y - x_end) - math.log(y / x_end))
    f = lambda y: (y - 1) ** 4 / y ** 2
    x_star = brentq(lambda y: n_of(y) - ne, x_end * 1.0001, 1e4)
    out = {"x_star": x_star, "phi_star_MP": math.sqrt(1.5) * math.log(x_star), "approx_N_q": ne / math.sqrt(a_s)}
    for p in (0.1, 1.0, 10.0):
        target = p * f(x_star) / a_s
        x_q = brentq(lambda y: f(y) - target, x_star, 1e12)
        n_q = n_of(x_q)
        out[f"P={p:g}"] = {"N_q": n_q, "phi_q_MP": math.sqrt(1.5) * math.log(x_q), "ratio_to_approx": n_q / (ne / math.sqrt(a_s / p)),
                           "Omega_k": _omega(n_q, x["n_hub"])}
    return out


def chance_in_windows() -> dict:
    """A-RL 보조: 모든 칸 가운데 창에 든 칸의 수와, N_총 분포에서 우연히 DESI 1σ 창에 들 기대 수."""
    ev = evaluate()
    n_hub = ev["summary"]["n_hub"]
    cells = [v for r in ev["rows"] for v in r["cases"].values()]
    lo, hi = n_hub + 0.5 * math.log(1 / WINDOW_DESI[1]), n_hub + 0.5 * math.log(1 / WINDOW_DESI[0])
    near = [v for v in cells if n_hub < v["N_total"] < n_hub + 100]
    density = len(near) / 100.0                                              # N_총 60–160 구간의 칸 밀도(e-fold당)
    expected = density * (hi - lo)
    return {"n_cells": len(cells), "desi_cells": sum(v["desi_1sigma"] for v in cells),
            "obs_cells": sum(v["obs_window"] for v in cells), "desi_window_width_efold": hi - lo,
            "expected_desi_by_chance": expected, "p_at_least_one": 1 - math.exp(-expected)}


def _fmt(v: dict) -> str:
    om = "O(1)" if v["Omega_k"] == math.inf else f"{v['Omega_k']:.2g}"
    tags = (" [창]" if v["obs_window"] else "") + (" [DESI]" if v["desi_1sigma"] else "") + (" ✗" if v["killed"] else "")
    return f"N={v['N_total']:.4g}, Ωk={om}{tags}"


def main() -> None:
    ev = evaluate()
    for r in ev["rows"]:
        print(f"{r['id']} {r['label']}: " + "; ".join(f"{k}: {_fmt(v)}" for k, v in r["cases"].items()))
    print("summary:", ev["summary"])


if __name__ == "__main__":
    main()
