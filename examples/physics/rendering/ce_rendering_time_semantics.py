"""관측 사상 M의 시간 의미론 — 고유시간은 틀 사이 비가 아니다. 원장: 43장 §43.90. 예측값을 바꾸지 않는다.

문제. 사상 M(§43.48)은 “틀 사이 비는 cos θ를 받는다”고만 적었다. 현재 틀의 막대로 잰 H₀가 73.36이라는 사실을 두 가지로
읽을 수 있다. 계산 전에 적은 판본·자료·kill(별 나이 수치는 규칙을 적기 전에 초록에서 보았다. 그래서 주 판정 자료와 kill을
관대한 쪽으로 고정했다):

U(단위) 현재 틀의 길이·시간 단위가 탄생 틀과 cos θ₀만큼 다르다. 현재 단위의 우주 나이는 t₀ cos θ₀다.
P(고유시간) 한 세계선의 고유시간(별 나이, 우주 나이, 은하 항성 종족의 나이)은 틀과 무관한 스칼라다. 투영은 먼 천체까지의
     공간꼴·널 간격을 현재 단위(막대, 또는 c × 국소 시계)로 읽을 때만 받는다. §43.16의 판독 공리(직접 판독 = |d ln A/dt|,
     시간 t는 공통)와 같은 읽기다.
자료 주 판정: 구상성단 III(arXiv:2503.19481) t_GC = 13.39 ± 0.10 ± 0.23 Gyr. 보고: 구상성단 IV(arXiv:2603.04872),
     은하수 별(arXiv:2509.02692, arXiv:2607.00764).
kill (K1) 가장 오래된 천체 나이가 판본의 우주 나이를 2σ 넘게 넘으면 그 판본을 기각(형성 지연 0, 가장 관대).
     (K2) P의 관측량 분류가 P32의 교정 종류 목록(§43.48, §43.53)과 하나라도 어긋나면 P는 M의 정식화가 아니다.
예측 후보 P에서 우주 시계(은하 나이 차 dt와 적색편이 dz)는 참 H(z)를 읽는다. H₀ = 67.77(현재 막대 73.36이 아님).

θ₀는 π/8(등록값)과 열적 시간 판본 H_Λt₀/2(§43.53, π/8 입력 없음)를 함께 쓴다.

python -B -m examples.physics.rendering.ce_rendering_time_semantics
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_registry as R

GYR_PER_INV_KMS_MPC = 977.79
H_EARLY = 67.772                               # R-Pl 판독(§43.12), 이른 기록 교정의 H₀
FORMATION_DELAY = (0.0, 0.2)                   # Gyr, 주 판정은 0(가장 관대)

# (이름, 공개 연월, 나이, 통계 σ, 계통 σ)
STAR_AGES = (
    ("구상성단 III arXiv:2503.19481", "2025-03", 13.39, 0.10, 0.23),
    ("구상성단 IV arXiv:2603.04872", "2026-03", 13.61, 0.25, 0.23),
    ("은하수 별 160 arXiv:2509.02692", "2025-09", 13.6, 1.0, 1.3),
    ("은하수 별 15만 arXiv:2607.00764(통계만)", "2026-07", 13.73, 0.15, 0.0),
)
PRIMARY = 0
CC_H0 = (66.7, 5.3)                            # 우주 시계만, 평탄 ΛCDM(Moresco 2023)


def core_omegas() -> tuple[float, float]:
    c = R.core(R.calibrated_alpha_s()[0])
    return c["Om"], 1.0 - c["Om"]


def tilt_today() -> dict:
    om, ol = core_omegas()
    hl_t0 = (2.0 / 3.0) * math.asinh(math.sqrt(ol / om))
    return {"pi/8": math.pi / 8, "thermal": hl_t0 / 2.0}


def universe_age(h0: float = H_EARLY) -> float:
    """평탄 물질 + Λ의 고유시간 나이(복사 무시, 0.01 Gyr 수준)."""
    om, ol = core_omegas()
    return GYR_PER_INV_KMS_MPC / h0 * (2.0 / (3.0 * math.sqrt(ol))) * math.asinh(math.sqrt(ol / om))


def version_ages() -> dict:
    t0 = universe_age()
    th = tilt_today()
    return {"P": t0, "U(pi/8)": t0 * math.cos(th["pi/8"]), "U(thermal)": t0 * math.cos(th["thermal"]),
            "true H0=73.36 (early-universe fix)": universe_age(H_EARLY / math.cos(th["pi/8"]))}


def age_test() -> dict:
    ages = version_ages()
    out = {}
    for ver, tu in ages.items():
        rows = []
        for name, date, age, st, sy in STAR_AGES:
            s = math.hypot(st, sy)
            rows.append({"data": name, "published": date,
                         "excess_sigma": {d: (age - (tu - d)) / s for d in FORMATION_DELAY}})
        prim = rows[PRIMARY]["excess_sigma"][0.0]
        out[ver] = {"t_universe": tu, "rows": rows, "primary_excess": prim, "killed": prim > 2.0}
    return out


# 관측량의 종류: (이름, P32 분류, P의 분류 근거)
#   P32 분류: "early"(이른 기록 교정 67.77), "present"(현재 막대·시계 교정 73.36), None(P32에 없음)
#   P 분류: 먼 천체까지의 거리를 현재 단위로 읽는가(→ present), 아니면 비·고유시간만 쓰는가(→ early)
OBSERVABLES = (
    ("CMB 음향 각도", "early", "비 r_s/D_M"),
    ("BAO + BBN", "early", "비 D/r_d, 이른 자"),
    ("역거리 사다리(BAO로 교정한 SN)", "early", "SN 상대 거리를 이른 자로 교정"),
    ("세페이드·TRGB·JAGB 사다리", "present", "먼 SN 거리를 현재 막대(시차)로 읽음"),
    ("메이저", "present", "먼 은하 거리를 현재 막대(각지름 + 국소 속도)로 읽음"),
    ("시간 지연 렌즈", "present", "D_Δt = cΔt/Δφ, 거리를 c × 국소 시계로 읽음"),
    ("표준 사이렌", "present", "d_L을 파형(국소 시계)으로 읽음"),
    ("우주 시계 H(z)", None, "dz(비)와 dt(항성 종족의 고유시간 차), 거리 없음"),
    ("가장 오래된 별의 나이", None, "고유시간"),
)


def p_classify(reason: str) -> str:
    reads_distance_in_present_units = ("현재 막대" in reason) or ("국소 시계" in reason)
    return "present" if reads_distance_in_present_units else "early"


def classification() -> dict:
    rows = [{"observable": n, "P32": p32, "P": p_classify(why), "why": why} for n, p32, why in OBSERVABLES]
    conflicts = [r["observable"] for r in rows if r["P32"] is not None and r["P32"] != r["P"]]
    new = [r["observable"] for r in rows if r["P32"] is None]
    return {"rows": rows, "conflicts": conflicts, "newly_classified": new}


def cc_prediction() -> dict:
    th = tilt_today()
    present = H_EARLY / math.cos(th["pi/8"])
    return {"P": H_EARLY, "if_present_reading": present, "obs": CC_H0,
            "pull_P": (CC_H0[0] - H_EARLY) / CC_H0[1], "pull_present": (CC_H0[0] - present) / CC_H0[1],
            "sigma_needed_3sigma_split": (present - H_EARLY) / 3.0}


def main() -> None:
    print("tilt today:", {k: round(v, 5) for k, v in tilt_today().items()})
    for ver, r in age_test().items():
        print(f"[{ver}] t_U = {r['t_universe']:.3f} Gyr  primary excess {r['primary_excess']:+.2f}σ  killed={r['killed']}")
        for row in r["rows"]:
            print(f"     {row['data']:40s} {row['published']}  excess(delay 0 / 0.2) "
                  f"{row['excess_sigma'][0.0]:+.2f} / {row['excess_sigma'][0.2]:+.2f}")
    c = classification()
    for row in c["rows"]:
        print(f"   {row['observable']:28s} P32={row['P32']!s:8s} P={row['P']:8s} ({row['why']})")
    print("K2 conflicts:", c["conflicts"], "| newly classified:", c["newly_classified"])
    print("cosmic chronometer prediction:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cc_prediction().items()})


if __name__ == "__main__":
    main()
