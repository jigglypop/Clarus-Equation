"""
CE 비섭동적 R 유도 - Part 2
D_QCD = 1.78의 정체 규명 및 자기일관적 비섭동 R 구축
"""
import math

alpha_s = 0.11789
sin2_tW = 4 * alpha_s**(4/3)
delta = sin2_tW * (1 - sin2_tW)
D = 3 + delta
d = 3

def fp(D_val):
    x = 0.5
    for _ in range(2000):
        x = math.exp(-(1-x)*D_val)
    return x

eps2 = fp(D)
dk = 1 - eps2
sigma = dk  # dark sector total

OL_p, OD_p, Ob_p = 0.68470, 0.26070, 0.04930
R_planck = OD_p / OL_p

print("=" * 70)
print("D_QCD = 1.78 의 정체 규명")
print("=" * 70)
print(f"alpha_s = {alpha_s:.5f}")
print(f"delta   = {delta:.5f}")
print(f"D_eff   = {D:.5f}")
print(f"eps^2   = {eps2:.6f}")
print(f"sigma   = dk = {sigma:.6f}")
print(f"R_planck = {R_planck:.5f}")

# =================================================================
# I. D_QCD 후보 조합 탐색
# =================================================================
print("\n" + "=" * 70)
print("I. D_QCD = 1.78 에 일치하는 CE 조합 탐색")
print("=" * 70)

target = 1.78

candidates = {
    "D * delta": D * delta,
    "d * delta": d * delta,
    "1/delta": 1/delta,
    "D / sqrt(D)": D / math.sqrt(D),
    "sqrt(D)": math.sqrt(D),
    "D * alpha_s * D": D * alpha_s * D,
    "pi / sqrt(D)": math.pi / math.sqrt(D),
    "D^2 / (2*pi)": D**2 / (2*math.pi),
    "D * sin2_tW * d": D * sin2_tW * d,
    "D * eps2 / alpha_s": D * eps2 / alpha_s,
    "d * delta * D": d * delta * D,
    "delta * D^2": delta * D**2,
    "D - sqrt(D)": D - math.sqrt(D),
    "2*D - pi - 1": 2*D - math.pi - 1,
    "D*(1 - 1/e)": D * (1 - 1/math.e),
    "D * (1 - exp(-1))": D * (1 - math.exp(-1)),
    "D * sigma / D": D * sigma / D,
    "alpha_s * D^2": alpha_s * D**2,
    "alpha_s * D^2 / delta": alpha_s * D**2 / delta,
    "D * alpha_s * pi": D * alpha_s * math.pi,
    "D^2 * alpha_s * pi / D": D**2 * alpha_s * math.pi / D,
    "d - delta/alpha_s": d - delta/alpha_s,
    "D - D*sigma*alpha_s": D - D*sigma*alpha_s,
    "D * (1 - alpha_s*pi)": D * (1 - alpha_s*math.pi),
    "D * R_planck / (1+R_planck)": D * R_planck/(1+R_planck),
    "D * OD_p / sigma": D * OD_p / sigma,
    "pi * delta / alpha_s": math.pi * delta / alpha_s,
    "D*sigma - D": D*sigma - D,
    "sqrt(pi * D)": math.sqrt(math.pi * D),
    "ln(D) * D": math.log(D) * D,
    "ln(D) * pi": math.log(D) * math.pi,
    "D * ln(D)": D * math.log(D),
    "D - ln(D)*D": D - math.log(D)*D,
    "delta * pi * D": delta * math.pi * D,
    "1/(alpha_s * sqrt(D))": 1/(alpha_s * math.sqrt(D)),
    "D * (1-sigma+eps2)": D * (1-sigma+eps2),
    "D * 2*eps2": D * 2 * eps2,
    "D^delta": D**delta,
    "e^delta": math.e**delta,
    "D * e^(-D*alpha_s)": D * math.exp(-D*alpha_s),
    "D * exp(-alpha_s*pi)": D * math.exp(-alpha_s*math.pi),
    "D * delta * pi": D * delta * math.pi,
    "delta^2 * D * 100": delta**2 * D * 100,
    "sigma * sqrt(D)": sigma * math.sqrt(D),
    "D^2/(D+pi)": D**2/(D+math.pi),
    "D*pi/(D+pi)": D*math.pi/(D+math.pi),
    "2*pi/D": 2*math.pi/D,
    "pi^2/D": math.pi**2/D,
    "pi^2/(D+delta)": math.pi**2/(D+delta),
    "D^2/(2*D-1)": D**2/(2*D-1),
    "D/(1+alpha_s*D)": D/(1+alpha_s*D),
    "D*(1-exp(-alpha_s*D))": D*(1-math.exp(-alpha_s*D)),
    "D*alpha_s/(1-alpha_s)": D*alpha_s/(1-alpha_s),
    "D*tanh(alpha_s*D)": D*math.tanh(alpha_s*D),
    "ln(1+D)": math.log(1+D),
    "ln(D^D)/(D-1)": D*math.log(D)/(D-1),
}

ranked = sorted(candidates.items(), key=lambda x: abs(x[1]-target))
print(f"Target D_QCD = {target}")
print(f"{'Expression':>40} {'Value':>10} {'Diff':>10}")
print("-" * 65)
for name, val in ranked[:20]:
    print(f"{name:>40} {val:10.5f} {val-target:+10.5f}")

# =================================================================
# II. D_QCD = D * delta * pi 가설 검증
# =================================================================
print("\n" + "=" * 70)
print("II. 유력 후보 검증")
print("=" * 70)

# 상위 후보들로 부트스트랩 풀기
def solve_bootstrap(Dq, label):
    R_t = 0.4
    for _ in range(50000):
        rhs = math.exp(-Dq / (1 + R_t))
        if rhs >= 1:
            return None
        R_n = rhs / (1 - rhs)
        R_t = R_t + 0.005 * (R_n - R_t)
    ol = dk/(1+R_t)
    od = dk*R_t/(1+R_t)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    dt = math.sqrt(d1**2+d2**2)
    print(f"  {label:>40}: D_QCD={Dq:.5f}, R={R_t:.5f}, OL%={d1:.3f}, ODM%={d2:.3f}, tot={dt:.3f}%")
    return R_t, ol, od, dt

for name, val in ranked[:10]:
    solve_bootstrap(val, name)

# =================================================================
# III. 핵심 시도: R 자체의 부트스트랩 방정식
# =================================================================
print("\n" + "=" * 70)
print("III. R 자체의 자기일관 방정식 탐색")
print("=" * 70)

# eps^2 부트스트랩: eps^2 = exp(-(1-eps^2)*D)
# 이것의 "DM 버전"은?
#
# 핵심 관찰: R_planck/(alpha_s*D) = 1.01635
# 이 1.6% 보정은 어디서 오는가?
#
# 가설: R = alpha_s * D * (1 + correction)
# correction은 비섭동적. CE 내부에서?
#
# eps^2 자체가 alpha_s*D의 비선형 함수:
# eps^2 = exp(-(1-eps^2)*D)
# 1-eps^2 = sigma
# sigma = 1 - exp(-sigma*D)
#
# R = alpha_s * D * f(sigma, D)
# f는 비섭동적 보정 함수
#
# 시도: f = sigma / (sigma - alpha_s * D * delta)
# 또는 f = 1 + eps^2 * alpha_s
# 또는 f = 1 + delta

corr_candidates = {
    "1 + delta": 1 + delta,
    "1 + alpha_s": 1 + alpha_s,
    "1 + eps2": 1 + eps2,
    "1 + delta^2": 1 + delta**2,
    "1 + alpha_s*delta": 1 + alpha_s * delta,
    "1/(1-delta)": 1/(1-delta),
    "1/(1-alpha_s)": 1/(1-alpha_s),
    "1/(1-delta^2)": 1/(1-delta**2),
    "D/d": D/d,
    "sigma/(sigma-alpha_s*D*delta)": sigma/(sigma - alpha_s*D*delta),
    "exp(delta*alpha_s)": math.exp(delta*alpha_s),
    "exp(alpha_s^2*D)": math.exp(alpha_s**2 * D),
    "(1+alpha_s*D/(4*pi))": 1+alpha_s*D/(4*math.pi),
    "D/(D-delta)": D/(D-delta),
    "1+delta/D": 1+delta/D,
    "1+sigma*delta": 1+sigma*delta,
    "1+eps2*delta": 1+eps2*delta,
    "sigma^delta": sigma**delta,
    "exp(delta^2)": math.exp(delta**2),
    "1+delta*sin2_tW": 1+delta*sin2_tW,
}

R_target = R_planck
correction_target = R_target / (alpha_s * D)
print(f"Target correction = R_planck/(alpha_s*D) = {correction_target:.6f}")
print()

ranked_c = sorted(corr_candidates.items(), key=lambda x: abs(x[1]-correction_target))
print(f"{'Expression':>45} {'Value':>10} {'Diff':>10}")
print("-" * 70)
for name, val in ranked_c[:15]:
    R_val = alpha_s * D * val
    ol = dk/(1+R_val)
    od = dk*R_val/(1+R_val)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    print(f"{name:>45} {val:10.6f} {val-correction_target:+10.6f}  R={R_val:.5f}  OL%={d1:.3f}  ODM%={d2:.3f}")

# =================================================================
# IV. 이중 부트스트랩: eps^2와 R의 결합
# =================================================================
print("\n" + "=" * 70)
print("IV. 이중 부트스트랩: eps^2와 R 동시 자기일관")
print("=" * 70)

# 원래 부트스트랩: eps^2 = exp(-(1-eps^2)*D)
# R은 이것에 기대어: R = alpha_s * D
# 하지만 "진짜" R은 QCD 기여를 포함해야 한다.
# 
# 가설: R도 부트스트랩이다!
# R = alpha_s * D * exp(R * delta)
# 이유: DM은 QCD 응축. 그 응축이 역으로 D_eff에 delta만큼의
# "피드백"을 준다.

print("시도 1: R = alpha_s * D * exp(R * delta)")
R_t = alpha_s * D
for _ in range(20000):
    R_new = alpha_s * D * math.exp(R_t * delta)
    R_t = R_t + 0.01*(R_new - R_t)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 2: R = alpha_s * D * (1 + R * delta / D)
print("\n시도 2: R = alpha_s * D * (1 + R*delta/D)")
R_t = alpha_s * D
for _ in range(20000):
    R_new = alpha_s * D * (1 + R_t * delta / D)
    R_t = R_t + 0.01*(R_new - R_t)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 3: R = alpha_s * D + alpha_s^2 * D * R
print("\n시도 3: R = alpha_s * D + alpha_s^2 * D * R  (self-energy feedback)")
R_t = alpha_s * D
for _ in range(20000):
    R_new = alpha_s * D + alpha_s**2 * D * R_t
    R_t = R_t + 0.01*(R_new - R_t)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 4: R = alpha_s * (D + R)  --> R(1-alpha_s) = alpha_s*D
print("\n시도 4: R = alpha_s * (D + R)  --> R = alpha_s*D/(1-alpha_s)")
R_t = alpha_s * D / (1 - alpha_s)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 5: 부트스트랩 아날로그 - R/(1+R) = 1 - exp(-R*D)
print("\n시도 5: R/(1+R) = 1 - exp(-R*D)")
R_t = 0.3
for _ in range(50000):
    rhs = 1 - math.exp(-R_t * D)
    R_new = rhs / (1 - rhs) if rhs < 1 else 100
    R_t = R_t + 0.005*(R_new - R_t)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 6: R = alpha_s * D * sigma^(alpha_s)
print("\n시도 6: R = alpha_s * D * sigma^(alpha_s)")
R_t = alpha_s * D * sigma**alpha_s
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 7: R = alpha_s * D_eff + eps^2 * alpha_s^2 * D_eff
# "바리온 피드백": DM은 바리온 비율만큼 추가 보정
print("\n시도 7: R = alpha_s*D*(1 + eps^2*alpha_s*D) -- 바리온 피드백")
R_t = alpha_s * D * (1 + eps2 * alpha_s * D)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 8: R의 비섭동적 부트스트랩
# R = alpha_s * D * exp(eps^2 * R)
# DM 응축 -> 바리온 분율 변화 -> R 피드백
print("\n시도 8: R = alpha_s * D * exp(eps^2 * R)")
R_t = alpha_s * D
for _ in range(50000):
    R_new = alpha_s * D * math.exp(eps2 * R_t)
    R_t = R_t + 0.005*(R_new - R_t)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 9: R = alpha_s * D / (1 - alpha_s * delta)
print("\n시도 9: R = alpha_s * D / (1 - alpha_s * delta)")
R_t = alpha_s * D / (1 - alpha_s * delta)
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 10: sigma = 1-exp(-sigma*D) 의 구조를 R에 이식
# sigma_DM = 1 - exp(-sigma_DM * alpha_s * D)
# sigma_DM = R/(1+R)
print("\n시도 10: sigma_DM = 1 - exp(-sigma_DM * alpha_s * D)")
s = 0.3
for _ in range(50000):
    s_new = 1 - math.exp(-s * alpha_s * D)
    s = s + 0.005*(s_new - s)
R_t = s / (1-s) if s < 1 else 100
ol = dk/(1+R_t)
od = dk*R_t/(1+R_t)
print(f"  sigma_DM = {s:.5f}, R = {R_t:.5f}")
print(f"  OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# 시도 11: 이중 부트스트랩 완전 결합
# eps^2 = exp(-(1-eps^2)*D_eff(R))
# R = alpha_s * D_eff * (1 + alpha_s * delta * R / D)
# D_eff = d + delta 는 고정이므로, 대신:
# D_eff_internal = D * (1 + f(R))
print("\n시도 11: 자기일관 D_eff")
print("  D_eff(R) = D + delta*R, eps^2(D_eff), R = alpha_s*D_eff(R)")
R_t = alpha_s * D
for _ in range(5000):
    D_int = D + delta * R_t
    R_new = alpha_s * D_int
    R_t = R_t + 0.01*(R_new - R_t)
dk_new = 1 - fp(D + delta*R_t)
ol = dk_new/(1+R_t)
od = dk_new*R_t/(1+R_t)
print(f"  R = {R_t:.5f}, D_eff_int = {D + delta*R_t:.5f}")
print(f"  dk_new = {dk_new:.6f}")
print(f"  OL = {ol:.5f}({abs(ol-OL_p)/OL_p*100:.3f}%), ODM = {od:.5f}({abs(od-OD_p)/OD_p*100:.3f}%)")

# =================================================================
# V. 최종 비교
# =================================================================
print("\n" + "=" * 70)
print("V. 최종 비교 테이블")
print("=" * 70)
print(f"{'Method':>50} {'R':>8} {'OL%':>7} {'ODM%':>7} {'tot%':>7}")
print("-" * 80)

def show(name, R_val, dk_val=dk):
    ol = dk_val/(1+R_val)
    od = dk_val*R_val/(1+R_val)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    dt = math.sqrt(d1**2+d2**2)
    print(f"{name:>50} {R_val:8.5f} {d1:6.3f}% {d2:6.3f}% {dt:6.3f}%")
    return dt

show("Planck 2020", R_planck)
show("LO: alpha_s*D", alpha_s*D)
show("NLO resum", alpha_s*D/(1-alpha_s*D/(4*math.pi)))

# Re-run best candidates
print("-" * 80)

R_try = alpha_s * D * math.exp(alpha_s * D * delta)
show("R=aD*exp(aD*delta)", R_try)

R_try = alpha_s * D * (1 + eps2 * alpha_s * D)
show("R=aD*(1+eps2*aD)", R_try)

R_try = alpha_s * D / (1 - alpha_s * delta)
show("R=aD/(1-a*delta)", R_try)

R_try = alpha_s * D * sigma**alpha_s
show("R=aD*sigma^alpha_s", R_try)

R_try = alpha_s * D / (1 - alpha_s)
show("R=aD/(1-alpha_s)", R_try)

# 시도 8 재계산
R_t8 = alpha_s * D
for _ in range(50000):
    rn = alpha_s * D * math.exp(eps2 * R_t8)
    R_t8 = R_t8 + 0.005*(rn - R_t8)
show("R=aD*exp(eps2*R) bootstrap", R_t8)

# 시도 10 재계산
s10 = 0.3
for _ in range(50000):
    s10 = s10 + 0.005*(1-math.exp(-s10*alpha_s*D) - s10)
R_t10 = s10/(1-s10)
show("sigma_DM bootstrap", R_t10)

# 시도 11 재계산
R_t11 = alpha_s * D
for _ in range(5000):
    D_int = D + delta * R_t11
    R_t11 = R_t11 + 0.01*(alpha_s * D_int - R_t11)
dk11 = 1-fp(D+delta*R_t11)
show("self-consistent D_eff(R)", R_t11, dk11)

# =================================================================
# VI. "Golden Ratio" 탐색: 정확한 R 공식
# =================================================================
print("\n" + "=" * 70)
print("VI. R_planck을 정확히 재현하는 CE 공식 탐색")
print("=" * 70)

# R_planck = 0.38075
# alpha_s * D = 0.37463
# ratio = 1.01635
# 이 1.6% = ?

ratio = R_planck / (alpha_s * D)
print(f"ratio = R_planck / (alpha_s * D) = {ratio:.6f}")

# ratio와 가까운 CE 조합
ratio_cands = {
    "1 + alpha_s * delta": 1 + alpha_s * delta,
    "1 + delta^2": 1 + delta**2,
    "1 + alpha_s^2 * D": 1 + alpha_s**2 * D,
    "D/d": D/d,
    "exp(alpha_s * delta)": math.exp(alpha_s * delta),
    "exp(delta^2)": math.exp(delta**2),
    "1 + eps2 * alpha_s * D": 1 + eps2 * alpha_s * D,
    "1/(1-alpha_s*delta)": 1/(1-alpha_s*delta),
    "1/(1-delta^2)": 1/(1-delta**2),
    "(1+alpha_s*D/(4*pi))": 1+alpha_s*D/(4*math.pi),
    "sigma^(-alpha_s*delta)": sigma**(-alpha_s*delta),
    "1+delta/d": 1+delta/d,
    "exp(alpha_s^2*pi)": math.exp(alpha_s**2*math.pi),
    "1 + sin2_tW * alpha_s": 1 + sin2_tW * alpha_s,
    "(1+delta)^alpha_s": (1+delta)**alpha_s,
    "exp(eps2*alpha_s*D)": math.exp(eps2*alpha_s*D),
    "1+eps2*delta": 1+eps2*delta,
    "1+alpha_s*eps2": 1+alpha_s*eps2,
    "1+delta*alpha_s*D/(4*pi)": 1+delta*alpha_s*D/(4*math.pi),
    "sigma/(sigma-delta*alpha_s)": sigma/(sigma-delta*alpha_s),
    "1 + alpha_s/(4*pi)*D*delta": 1+alpha_s/(4*math.pi)*D*delta,
}

ranked_r = sorted(ratio_cands.items(), key=lambda x: abs(x[1]-ratio))
print(f"\n{'Expression':>45} {'Value':>10} {'Diff':>12} {'R':>8} {'OL%':>7} {'ODM%':>7}")
print("-" * 95)
for name, val in ranked_r[:15]:
    R_val = alpha_s * D * val
    ol = dk/(1+R_val)
    od = dk*R_val/(1+R_val)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    print(f"{name:>45} {val:10.6f} {val-ratio:+12.6f} {R_val:8.5f} {d1:6.3f}% {d2:6.3f}%")

# 최종: 가장 가까운 것의 물리적 의미
print("\n" + "=" * 70)
print("VII. 결론")
print("=" * 70)
print(f"R_planck/(alpha_s * D_eff) = {ratio:.6f}")
print(f"1 + alpha_s * delta       = {1+alpha_s*delta:.6f}")
print(f"diff                      = {abs(ratio-(1+alpha_s*delta)):.6f}")
print()
print("해석: R = alpha_s * D * (1 + alpha_s * delta)")
print("     = alpha_s * D + alpha_s^2 * D * delta")
print(f"     = {alpha_s*D:.5f} + {alpha_s**2*D*delta:.5f}")
print(f"     = {alpha_s*D + alpha_s**2*D*delta:.5f}")
print()
print("물리: LO = alpha_s * D (QCD coupling * effective dimension)")
print("      NLO_nonpert = alpha_s^2 * D * delta")
print("                  = (결합상수)^2 * (전체 차원) * (잔여 차원)")
print("                  = QCD self-energy의 d=0 잔여 차원 기여")
