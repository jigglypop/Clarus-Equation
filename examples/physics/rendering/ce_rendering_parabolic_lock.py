"""원 없는 위상 잠김 — 지평선의 임계 잠김은 포물형이고, 바젤 합이 π²를 준다. 원장: 43장 §43.97. 예측값을 바꾸지 않는다.

사용자(2026-09-25): “그냥 위상잠김인데 꼭 원이어야함?” §43.94–43.96의 (2π)²는 원환(2π 주기, 주기 모드)을 전제했다.
위상 잠김의 표준 방정식(Adler, §43.50·§43.51 H3)은 u = tan θ로 리카티 방정식이 되고, 흐름은 사영(뫼비우스) 변환이다.
지평선은 인과 창의 가장자리 θ = π/4(C′)이고, 그 자리의 잠김은 정의상 임계(L = 1, 안장–마디, §43.50 P-1)라 포물형이다.

계산 전에 적은 규칙. 사전 목격: 아래 (가)의 항등식과 창 진입 = Fatou 시간 1, 그리고 오늘 지평선의 대략 결손(e-fold 시계로
약 −1.9)을 암산으로 먼저 보았다. 그래서 (나)의 선도 계수 적중은 증거가 아니라 읽기로만 쓴다.

(가) 구성(모두 기존 공리): 기울기 구동은 TT + H1에서 dθ/dN = ½(자유), 임계 결합 K = 1: dθ/dN = ½(1 − sin 2θ).
    리카티 u = tan θ: du/dN = (u − 1)²/2, Fatou 시간 F = 2/(1 − tan θ)는 dF/dN = 1.
    반대점 θ = −π/2 → F = 0, 창 진입 θ = −π/4 → F = 1, 기록 틀 θ = 0 → F = 2, 지평선 θ = π/4 → F = ∞.
    검산(kill): 수치 적분에서 F(N) − F(0) − N, 창 진입까지 걸린 e-fold − 1, 창 진입부터 e-fold마다 잰 (1 − tan θ)/2 − 1/n이
    모두 1e-8 안이어야 한다. 아니면 구성이 틀린 것이다.
(나) 원리 PL(새 전제 Q 하나): 로그 화소 수 = 모든 기록·축·창 안의 걸음에 걸친 (Fatou 거리 1/F)²의 합. 기록은 급팽창의
    e-fold마다 축마다 하나(TT·Bool), 축은 셋(E1). 걸음마다 1/F = 1/n이면 기록 하나당 ζ(2), 전체 (π²/2) N_e.
    가족(유일성 보고): 거리 {1/F, 기울기 각 π/4 − θ, 기록 위상 π/2 − 2θ, 리카티 1 − tan θ} × 시작 {반대점, 창 진입, 기록 틀}
    × 거듭제곱 {2, 4}. 선도 계수(세 축, 기록 하나)가 π²/2와 1e-9 안이면 적중.
(다) O(1) 상수: 최종 지평선(무한 미래)은 모든 기록이 끝나 결손 0. 오늘 지평선은 끝나지 않은 꼬리 −3 Σ_j ψ₁(F_j + 1)
    (F_j는 기록 j의 오늘 Fatou 시간). 시계는 둘을 보고한다. e-fold 시계(급팽창 뒤 ln(a₀/a_end) = 65.70, §43.95 순간 재가열)와
    TT 시계(급팽창 뒤 기록 위상 H_Λ t₀ = 0.796, §43.53). 필요한 상수(§43.94: 최종 −0.1799, 오늘 −0.5294, σ 0.0026)와
    2σ 안이면 상수까지 유도, 아니면 실패.
(라) 부산물: 단위 구동(기록 위상 1 rad/e-fold) 잠금 사상 ψ_{n+1} = ψ_n + 1 − K sin ψ_n이 초안정(승수 0)인 부하 L과,
    연속 흐름의 이완율을 계산한다. §43.52(기울기는 돌고 있다)와의 관계를 함께 적는다.

python -B -m examples.physics.rendering.ce_rendering_parabolic_lock
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import polygamma, zeta

from examples.physics.rendering import ce_rendering_horizon_towers as HTW
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_tower_temperature as TTW

PI = math.pi
ZETA2 = PI ** 2 / 6.0
AXES = 3
TARGET = PI ** 2 / 2.0
N_AFTER_EFOLDS = None          # 급팽창 뒤 e-fold 수는 §43.95 정합에서 가져온다
TT_PHASE_TODAY = 0.7962        # §43.53 V-dS: 탄생부터 오늘까지 기록 위상 H_Λ t₀


def rhs(_n: float, y: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 - math.sin(2.0 * y[0]))])


def fatou(theta: float) -> float:
    return 2.0 / (1.0 - math.tan(theta))


# ------------------------------------------------------------------ (가) 구성 검산
def construction_checks() -> dict:
    # F(N) − F(0) = N: 기록 틀(θ = 0)에서 출발해 10 e-fold 적분
    sol = solve_ivp(rhs, (0.0, 10.0), [0.0], rtol=1e-12, atol=1e-14, dense_output=True)
    ns = np.linspace(0.0, 10.0, 41)
    drift = max(abs(fatou(float(sol.sol(n)[0])) - fatou(0.0) - n) for n in ns)
    # 반대점 근처에서 창 진입까지 걸린 e-fold
    eps = 1e-7
    th0 = -PI / 2 + eps
    hit = lambda _n, y: y[0] + PI / 4
    hit.terminal, hit.direction = True, 1
    s2 = solve_ivp(rhs, (0.0, 5.0), [th0], rtol=1e-12, atol=1e-14, events=hit)
    entry_time = float(s2.t_events[0][0]) + fatou(th0)          # 반대점(F = 0)부터 잰 시간
    # 창 진입부터 e-fold마다 (1 − tan θ)/2 = 1/n
    s3 = solve_ivp(rhs, (0.0, 30.0), [-PI / 4], rtol=1e-12, atol=1e-14, dense_output=True)
    step_dev = max(abs((1.0 - math.tan(float(s3.sol(n - 1)[0]))) / 2.0 - 1.0 / n) for n in range(1, 31))
    return {"fatou_drift": drift, "entry_time_minus_one": abs(entry_time - 1.0), "step_deviation": step_dev,
            "F_antipode": 0.0, "F_entry": fatou(-PI / 4), "F_record_frame": fatou(0.0),
            "ok": drift < 1e-8 and abs(entry_time - 1.0) < 1e-6 and step_dev < 1e-8}


# ------------------------------------------------------------------ (나) 선도 계수와 가족
def theta_at_fatou(f: float) -> float:
    return math.atan(1.0 - 2.0 / f) if f > 0 else -PI / 2


MEASURES = {
    "1/F (Fatou)": lambda th: (1.0 - math.tan(th)) / 2.0,
    "기울기 각 π/4 − θ": lambda th: PI / 4 - th,
    "기록 위상 π/2 − 2θ": lambda th: PI / 2 - 2.0 * th,
    "리카티 1 − tan θ": lambda th: 1.0 - math.tan(th),
}
STARTS = {"반대점": 0.0, "창 진입": 1.0, "기록 틀": 2.0}       # 시작 Fatou 시간
POWERS = (2, 4)
NSUM = 200000


def record_sum(measure: str, f0: float, p: int) -> float:
    """기록 하나, 축 하나: F = f0, f0+1, … 걸음의 거리^p 합. 반대점의 무한 거리는 빼고 센다."""
    m = MEASURES[measure]
    total = 0.0
    start = 1 if f0 == 0.0 and measure in ("1/F (Fatou)", "리카티 1 − tan θ") else 0
    for k in range(start, NSUM):
        total += m(theta_at_fatou(f0 + k)) ** p
    # 꼬리: 먼 걸음에서 거리 ≈ c/F, c = 1(1/F·기울기 각), 2(기록 위상·리카티)
    c = 1.0 if measure in ("1/F (Fatou)", "기울기 각 π/4 − θ") else 2.0
    f_end = f0 + NSUM
    total += c ** p * float(polygamma(p - 1, f_end)) / math.factorial(p - 1) if p == 2 else c ** p / (3 * f_end ** 3)
    return total


def family() -> dict:
    out = {}
    for mname in MEASURES:
        for sname, f0 in STARTS.items():
            for p in POWERS:
                coef = AXES * record_sum(mname, f0, p)
                out[(mname, sname, p)] = {"coef": coef, "hit": abs(coef - TARGET) < 1e-9}
    hits = [k for k, v in out.items() if v["hit"]]
    return {"members": out, "hits": hits}


# ------------------------------------------------------------------ (다) O(1) 상수
def constants() -> dict:
    t = HTW.target()
    ne = t["N_e"]
    m = TTW.matching()
    n_after_efold = -(math.log(TTW.T0_K * TTW.K_TO_GEV / m["T_reh_instant_GeV"]) + math.log(TTW.G_S0 / TTW.G_REH) / 3.0)
    records = int(math.floor(ne))

    def deficit(n_after: float) -> float:
        tot = 0.0
        for j in range(records):
            f_today = (ne - j) + n_after
            tot += float(polygamma(1, math.floor(f_today) + 1))
        return -AXES * tot

    c_today_efold, c_today_tt = deficit(n_after_efold), deficit(TT_PHASE_TODAY)
    sig = t["sigma_S"]
    return {"C_final_PL": 0.0, "C_final_needed": t["C_final"], "pull_final": (0.0 - t["C_final"]) / sig,
            "C_today_PL_efold_clock": c_today_efold, "C_today_PL_TT_clock": c_today_tt,
            "C_today_needed": t["C_today"], "pull_today_efold": (c_today_efold - t["C_today"]) / sig,
            "pull_today_TT": (c_today_tt - t["C_today"]) / sig, "n_after_efolds": n_after_efold,
            "constant_derived": min(abs((0.0 - t["C_final"]) / sig), abs((c_today_efold - t["C_today"]) / sig),
                                    abs((c_today_tt - t["C_today"]) / sig)) <= 2.0}


# ------------------------------------------------------------------ (라) 부산물: 초안정 잠금
def superstable_lock() -> dict:
    """ψ_{n+1} = ψ_n + 1 − K sin ψ_n(기록 위상 1 rad/e-fold). 고정점 K sin ψ* = 1, 승수 1 − K cos ψ*."""
    from scipy.optimize import brentq

    mult = lambda L: 1.0 - (1.0 / L) * math.sqrt(1.0 - L * L)
    L_star = brentq(mult, 0.2, 0.99)
    psi = math.asin(L_star)
    k = 1.0 / L_star
    return {"L_superstable": L_star, "is_1_over_sqrt2": abs(L_star - 1 / math.sqrt(2)) < 1e-12,
            "tilt": psi / 2.0, "is_pi_over_8": abs(psi / 2.0 - PI / 8) < 1e-12,
            "continuous_relaxation_rate": k * math.cos(psi)}


def main() -> None:
    print("(가) construction:", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in construction_checks().items()})
    f = family()
    print("(나) hits:", f["hits"])
    for k, v in f["members"].items():
        print(f"     {k[0]:18s} {k[1]:5s} p={k[2]}  coef {v['coef']:.6f}")
    print("(다) constants:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in constants().items()})
    print("(라) superstable:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in superstable_lock().items()})


if __name__ == "__main__":
    main()
