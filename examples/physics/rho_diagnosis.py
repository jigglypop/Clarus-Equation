"""rho 불일치 진단: 수식이 틀린가, 매핑이 틀린가?

CE 수식: rho = D_eff * eps^2, eps^2 = exp(-(1-eps^2)*D_eff)
CE 예측: rho = 0.155 (D_eff=3.178, eps^2=0.0487)
측정값:  rho_night = 0.31 (급성 회복 4건 평균)

세 가지 가능성을 수치적으로 분석한다:
  (A) 수식은 맞고, D_eff가 다르다
  (B) 수식은 맞고, "1적용 != 1밤"이다
  (C) 수식 자체를 바꿔야 한다

Usage: py rho_diagnosis.py
"""

import numpy as np
from scipy.optimize import brentq

ALPHA_S = 0.11789
RHO_MEASURED = 0.311
RHO_CE = 0.155
D_CE = 3.178
EPS2_CE = 0.0487
D_SWEEP_VALUES = [1.5, 2.0, 2.35, 2.5, 3.0, 3.178, 3.5, 4.0, 5.0]
KAPPA_SWEEP_VALUES = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
DELTA_MIX_SHIFT_VALUES = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]


def bootstrap_fixed_point(D):
    """eps^2 = exp(-(1-eps^2)*D) 의 비자명 해."""
    if D <= 1:
        return None

    def eq(x):
        return np.exp(-(1 - x) * D) - x

    lo = max(np.exp(-D) * 0.5, 1e-300)
    return brentq(eq, lo, 1.0 - 1e-12)


def rho_from_D(D):
    eps2 = bootstrap_fixed_point(D)
    if eps2 is None:
        return None, None
    return D * eps2, eps2


def bootstrap_fixed_point_interacting(D, kappa):
    """A1 비가산 상호작용 항 kappa를 포함한 고정점."""
    if D <= 1:
        return None

    def eq(x):
        return np.exp(-(1 - x) * D - kappa * (1 - x) ** 2) - x

    return brentq(eq, 1e-12, 1.0 - 1e-12)


def rho_from_D_interacting(D, kappa):
    eps2 = bootstrap_fixed_point_interacting(D, kappa)
    if eps2 is None:
        return None, None
    rho = eps2 * (D + 2 * kappa * (1 - eps2))
    return rho, eps2


def D_from_rho(rho_target):
    """주어진 rho에 대응하는 D_eff를 역산."""

    def eq(D):
        rho, _ = rho_from_D(D)
        if rho is None:
            return rho_target
        return rho - rho_target

    return brentq(eq, 1.001, 50.0)


def energy_distribution(D, eps2=None):
    """CE 에너지 분배 (xa, xs, xb) 계산."""
    if eps2 is None:
        eps2 = bootstrap_fixed_point(D)
    if eps2 is None:
        return None

    sigma = 1 - eps2
    delta = D - 3 if D > 3 else 0
    ratio = ALPHA_S * D * (1 + eps2 * delta)
    xs = sigma * ratio / (1 + ratio)
    xb = sigma / (1 + ratio)
    return eps2, xs, xb


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def analyze_hypothesis_a():
    D_brain = D_from_rho(RHO_MEASURED)
    eps2_brain = bootstrap_fixed_point(D_brain)
    distribution = energy_distribution(D_brain)
    return {
        "D_brain": D_brain,
        "eps2_brain": eps2_brain,
        "distribution": distribution,
        "outside_sparse_range": eps2_brain > 0.10,
    }


def analyze_hypothesis_b():
    mapping_nights = np.log(RHO_CE) / np.log(RHO_MEASURED)
    return {
        "mapping_nights": mapping_nights,
        "applications_per_night": 1.0 / mapping_nights,
        "hours_per_application": mapping_nights * 8.0,
    }


def analyze_a1_sensitivity():
    base_rho, base_eps2 = rho_from_D(D_CE)
    d_eps2_d_kappa = -(base_eps2 * (1 - base_eps2) ** 2) / (1 - base_rho)
    d_rho_d_kappa = D_CE * d_eps2_d_kappa + 2 * base_eps2 * (1 - base_eps2)

    sweep = []
    for kappa in KAPPA_SWEEP_VALUES:
        rho, eps2 = rho_from_D_interacting(D_CE, kappa)
        distribution = energy_distribution(D_CE, eps2)
        sweep.append(
            {
                "kappa": kappa,
                "rho": rho,
                "eps2": eps2,
                "distribution": distribution,
            }
        )

    return {
        "base_eps2": base_eps2,
        "base_rho": base_rho,
        "d_eps2_d_kappa": d_eps2_d_kappa,
        "d_rho_d_kappa": d_rho_d_kappa,
        "sweep": sweep,
    }


def analyze_i1_sensitivity():
    base_rho, base_eps2 = rho_from_D(D_CE)
    d_eps2_d_D = -(base_eps2 * (1 - base_eps2)) / (1 - base_rho)
    d_rho_d_D = base_eps2 * (1 - D_CE) / (1 - base_rho)

    delta = D_CE - 3.0
    ratio = ALPHA_S * D_CE * (1 + base_eps2 * delta)
    d_ratio_d_D = ALPHA_S * ((1 + base_eps2 * delta) + D_CE * (base_eps2 + delta * d_eps2_d_D))
    sigma = 1 - base_eps2
    d_xs_d_D = -(d_eps2_d_D * ratio) / (1 + ratio) + sigma * d_ratio_d_D / (1 + ratio) ** 2
    d_xb_d_D = -(d_eps2_d_D) / (1 + ratio) - sigma * d_ratio_d_D / (1 + ratio) ** 2

    sweep = []
    for delta_shift in DELTA_MIX_SHIFT_VALUES:
        D_shifted = D_CE + delta_shift
        rho, eps2 = rho_from_D(D_shifted)
        distribution = energy_distribution(D_shifted, eps2)
        sweep.append(
            {
                "delta_shift": delta_shift,
                "D": D_shifted,
                "rho": rho,
                "eps2": eps2,
                "distribution": distribution,
            }
        )

    return {
        "base_delta_mix": D_CE - 3.0,
        "d_eps2_d_D": d_eps2_d_D,
        "d_rho_d_D": d_rho_d_D,
        "d_xs_d_D": d_xs_d_D,
        "d_xb_d_D": d_xb_d_D,
        "sweep": sweep,
    }


def print_current_ce():
    print("[현재 CE]")
    print(f"  D_eff = {D_CE}, eps^2 = {EPS2_CE}")
    print(f"  rho = {RHO_CE}")
    print(f"  에너지: xa={EPS2_CE*100:.2f}%, xs=26.2%, xb=68.9%")


def print_hypothesis_a(result):
    print_section("가설 (A): 수식은 맞고, 뇌의 D_eff가 다르다")
    print(f"\n  rho_measured = {RHO_MEASURED} → D_eff = {result['D_brain']:.3f}")
    print(
        f"  eps^2 = {result['eps2_brain']:.4f} "
        f"({result['eps2_brain']*100:.2f}%)"
    )
    if result["distribution"] is not None:
        xa, xs, xb = result["distribution"]
        print(f"  에너지 분배: xa={xa*100:.2f}%, xs={xs*100:.2f}%, xb={xb*100:.2f}%")
    print(f"\n  문제: eps^2 = {result['eps2_brain']*100:.2f}%")
    print("  실제 뇌의 sparse coding은 1-5% 범위.")
    if result["outside_sparse_range"]:
        print(f"  → {result['eps2_brain']*100:.1f}%는 관측과 맞지 않는다.")
        print("  → 가설 (A)는 기각 가능성 높음.")
    else:
        print("  → 관측 범위 안에 있다.")


def print_hypothesis_b(result):
    print_section("가설 (B): 수식은 맞고, '1적용 ≠ 1밤'이다")
    print(f"\n  rho_CE = rho_night^k → k = ln({RHO_CE})/ln({RHO_MEASURED})")
    print(f"  k = {result['mapping_nights']:.3f}")
    print(
        f"\n  해석: CE의 '부트스트랩 사상 1회 적용'은 "
        f"약 {result['mapping_nights']:.1f}밤의 회복 수면에 대응."
    )
    print(f"  역으로: 1밤에 약 {result['applications_per_night']:.2f}회의 B 적용이 일어난다.")
    print(f"  (완전한 NREM+REM+재수렴에 약 {result['hours_per_application']:.0f}시간 필요)")
    print("\n  검증 가능성:")
    print("  - 수면 다원검사에서 첫날 밤의 SWS rebound가 불완전하고,")
    print("    2번째 밤까지 REM rebound가 이어진다는 관측과 정합하는가?")
    print("  - 실제: Borbely(1982)의 2-process 모델에서도 1밤으로는")
    print("    SWS debt가 완전 해소되지 않는다고 알려져 있다.")
    print("  → 가설 (B)는 기존 수면학과 정합한다.")


def print_hypothesis_c():
    print_section("가설 (C): 수식 자체를 바꿔야 한다")
    print("\n  수식을 바꿔야 하는 경우:")
    print("  1. 부트스트랩 방정식 eps^2 = exp(-(1-eps^2)*D) 자체가 틀린 경우")
    print("     → 이 방정식은 물리(우주론) 쪽에서 독립 검증됨 (Planck 데이터)")
    print("     → 바꿀 이유 없음")
    print()
    print("  2. rho = D * eps^2 (고정점 도함수) 가 틀린 경우")
    print("     → 이것은 g(x) = exp(-(1-x)D)의 x=eps^2에서의 미분값")
    print("     → g'(eps^2) = D * exp(-(1-eps^2)D) = D * eps^2")
    print("     → 수학적으로 틀릴 수 없음")
    print()
    print("  3. 수축 사상 B의 구조가 linear affine이 아닌 경우")
    print("     → B(p) = (1-rho)*p* + rho*p 는 최소 모델")
    print("     → 실제 뇌의 수면 역학은 비선형항이 있을 수 있음")
    print("     → 이것은 '수식을 바꾸는 것'이 아니라 '높은 차수 항을 추가하는 것'")
    print()
    print("  4. 일일 섭동 u_n의 상태 의존성")
    print("     → 현재: e_{n+1} = rho * e_n + u (u 상수)")
    print("     → 실제: u = u(e_n) 일 수 있음 (이탈이 크면 각성 중 섭동도 커짐)")
    print("     → 이것도 '수식 변경'이 아니라 '섭동 모델 정밀화'")


def print_a1_sensitivity(result):
    print_section("A1 위배: 비가산 채널 상호작용 민감도")
    print("\n  핵심:")
    print("  - 채널별 계수 lambda_i가 비균일해도, 로그 억압이 가산적이면 식은 안 바뀐다.")
    print("  - 진짜 A1 위배는 비가산 상호작용 항이 들어올 때 시작한다.")
    print()
    print("  가장 단순한 변형:")
    print("    eps^2 = exp(-(1-eps^2)D - kappa(1-eps^2)^2)")
    print()
    print("  kappa = 0 에서 CE로 복귀.")
    print(
        f"  d(eps^2)/dkappa|0 = {result['d_eps2_d_kappa']:.4f}, "
        f"d(rho)/dkappa|0 = {result['d_rho_d_kappa']:.4f}"
    )
    print("  → kappa > 0 이면 xa와 rho가 함께 감소, kappa < 0 이면 둘 다 증가.")
    print()
    print(f"  {'kappa':>8s} {'xa%':>8s} {'rho':>8s} {'xs%':>8s} {'xb%':>8s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for row in result["sweep"]:
        xa, xs, xb = row["distribution"]
        print(
            f"  {row['kappa']:>8.2f} {xa*100:>7.2f}% {row['rho']:>8.4f} "
            f"{xs*100:>7.2f}% {xb*100:>7.2f}%"
        )
    print()
    print("  해석:")
    print("  - kappa = +0.10 이면 xa ≈ 4.37%, rho ≈ 0.1472")
    print("  - kappa = -0.10 이면 xa ≈ 5.41%, rho ≈ 0.1618")
    print("  - 즉 현재의 4.87%, 0.155는 A1 위배가 크지 않다는 제약을 준다.")


def print_i1_sensitivity(result):
    print_section("I1 민감도: delta_mix가 흔들리면 얼마나 변하나")
    print("\n  핵심:")
    print("  - A1은 가산성이 유지되면 식의 형태가 안 바뀐다.")
    print("  - I1은 D_eff = 3 + delta_mix 자체를 건드리므로, 코어 고정점이 바로 이동한다.")
    print()
    print(f"  기준값: delta_mix = {result['base_delta_mix']:.5f}, D_eff = {D_CE:.3f}")
    print(
        f"  d(eps^2)/d(delta_mix) = d(eps^2)/dD = {result['d_eps2_d_D']:.4f}, "
        f"d(rho)/d(delta_mix) = {result['d_rho_d_D']:.4f}"
    )
    print(
        f"  d(xs)/d(delta_mix) = {result['d_xs_d_D']:.4f}, "
        f"d(xb)/d(delta_mix) = {result['d_xb_d_D']:.4f}"
    )
    print("  → delta_mix가 커지면 xa와 rho는 감소, xs는 증가, xb는 소폭 감소.")
    print()
    print(f"  {'Δdelta':>8s} {'D_eff':>8s} {'xa%':>8s} {'rho':>8s} {'xs%':>8s} {'xb%':>8s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for row in result["sweep"]:
        xa, xs, xb = row["distribution"]
        print(
            f"  {row['delta_shift']:>+8.2f} {row['D']:>8.3f} {xa*100:>7.2f}% "
            f"{row['rho']:>8.4f} {xs*100:>7.2f}% {xb*100:>7.2f}%"
        )
    print()
    print("  해석:")
    print("  - delta_mix를 -0.10 낮추면 xa ≈ 5.45%, rho ≈ 0.1676")
    print("  - delta_mix를 +0.10 높이면 xa ≈ 4.35%, rho ≈ 0.1425")
    print("  - 즉 I1은 A1보다 훨씬 직접적으로 코어 수치를 움직이는 가정이다.")


def print_conclusion(result):
    print_section("결론")
    print(
        f"""
  코어 수식 (부트스트랩 방정식, 고정점, 수축률)은 바꿀 필요 없다.
  
  바꿔야 하는 것: sleep.md의 매핑 문장.

  현재 문서:
    "한 번의 수면-각성 순환이 B의 한 번 적용에 대응" [Bridge]

  수정안:
    "B의 한 번 완전 적용은 약 {result['mapping_nights']:.1f}밤(~{result['hours_per_application']:.0f}시간)의
     회복 수면에 대응한다. 단일 야간의 수축률은 rho_night ~ rho^(1/{result['mapping_nights']:.1f})
     ≈ {RHO_MEASURED:.2f}이다."

  이것은 CE 코어 수식의 변경이 아니라,
  코어와 뇌 사이의 Bridge 층 매핑의 정밀화다.

  구체적으로:
    - 부트스트랩 방정식: 그대로
    - 고정점 유일성: 그대로
    - 수축 사상 구조: 그대로
    - (I1) D_eff = 3.178: 그대로 (eps^2 = 4.87%가 sparse coding 관측과 정합하므로)
    - "1적용 = 1밤" -> "1적용 ~ {result['mapping_nights']:.1f}밤" 으로 보정
"""
    )


def print_d_sweep():
    print("참고: D_eff에 따른 rho와 eps^2")
    print(f"  {'D_eff':>8s} {'eps^2':>8s} {'rho':>8s} {'xa%':>6s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for D in D_SWEEP_VALUES:
        rho, eps2 = rho_from_D(D)
        if rho is not None:
            print(f"  {D:>8.3f} {eps2:>8.4f} {rho:>8.4f} {eps2*100:>5.1f}%")


def main():
    print("=" * 70)
    print("rho 불일치 진단: 수식 vs 매핑")
    print("=" * 70)

    print_current_ce()
    hypothesis_a = analyze_hypothesis_a()
    hypothesis_b = analyze_hypothesis_b()
    a1_sensitivity = analyze_a1_sensitivity()
    i1_sensitivity = analyze_i1_sensitivity()

    print_hypothesis_a(hypothesis_a)
    print_hypothesis_b(hypothesis_b)
    print_hypothesis_c()
    print_a1_sensitivity(a1_sensitivity)
    print_i1_sensitivity(i1_sensitivity)
    print_conclusion(hypothesis_b)
    print_d_sweep()


if __name__ == "__main__":
    main()
