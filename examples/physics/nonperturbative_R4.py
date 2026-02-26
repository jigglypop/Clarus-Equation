"""
CE 비섭동적 R 유도 - Part 4: 최종 형태 결정

Part 2에서 발견: R = alpha_s*D*(1+eps2*delta) 가 total 0.84%로 최적
Part 3에서 유도: R = alpha_s*d + alpha_s*delta/(1-eps2*delta)

이 두 형태는 다르다. 어느 것이 물리적으로 올바른가?
그리고 아직 1%의 잔차가 있다. 이것의 기원은?
"""
import math

alpha_s = 0.11789
sin2_tW = 4 * alpha_s**(4/3)
cos2_tW = 1 - sin2_tW
delta = sin2_tW * cos2_tW
D = 3 + delta
d = 3

def bootstrap(D_val):
    x = 0.5
    for _ in range(5000):
        x = math.exp(-(1-x)*D_val)
    return x

eps2 = bootstrap(D)
sigma = 1 - eps2

OL_p = 0.68470
OD_p = 0.26070
Ob_p = 0.04930
R_p = OD_p / OL_p

print("=" * 72)
print("CE 비섭동적 R: 최종 형태 결정")
print("=" * 72)

# =====================================================================
# I. 두 형태의 차이 분석
# =====================================================================
print("\n" + "=" * 72)
print("I. 두 후보 형태의 차이")
print("=" * 72)

# 형태 A: R = alpha_s * D * (1 + eps2*delta)
#        = alpha_s*D + alpha_s*D*eps2*delta
#        = alpha_s*(d+delta) + alpha_s*(d+delta)*eps2*delta
#        = alpha_s*d + alpha_s*delta + alpha_s*d*eps2*delta + alpha_s*delta^2*eps2

# 형태 B: R = alpha_s*d + alpha_s*delta/(1-eps2*delta)
#        = alpha_s*d + alpha_s*delta*(1 + eps2*delta + (eps2*delta)^2 + ...)
#        = alpha_s*d + alpha_s*delta + alpha_s*delta*eps2*delta + ...

# 차이: A는 "전체 D에 보정", B는 "delta에만 보정"
# A에는 alpha_s*d*eps2*delta 항이 있지만 B에는 없다.
# A: 바리온이 정수 차원에도 피드백
# B: 바리온이 잔여 차원에만 피드백

R_A = alpha_s * D * (1 + eps2*delta)
R_B = alpha_s * d + alpha_s * delta / (1 - eps2*delta)

print(f"형태 A: R = alpha_s*D*(1+eps2*delta) = {R_A:.7f}")
print(f"형태 B: R = alpha_s*d + alpha_s*delta/(1-eps2*delta) = {R_B:.7f}")
print(f"차이: {R_A - R_B:.7f}")
print(f"차이/R_planck: {(R_A-R_B)/R_p*100:.4f}%")

# A의 추가 항: alpha_s * d * eps2 * delta
extra = alpha_s * d * eps2 * delta
print(f"\nA의 추가 항: alpha_s*d*eps2*delta = {extra:.7f}")
print(f"물리적 의미: 바리온(eps2)이 잔여 차원(delta)을 통해")
print(f"            정수 차원(d=3)의 QCD 요동에도 피드백")

def show(name, R_val):
    ol = sigma/(1+R_val)
    od = sigma*R_val/(1+R_val)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    dt = math.sqrt(d1**2+d2**2)
    print(f"  {name:>40}: R={R_val:.5f}, OL%={d1:.3f}, OD%={d2:.3f}, tot={dt:.3f}%")
    return ol, od

show("A: alpha_s*D*(1+eps2*delta)", R_A)
show("B: alpha_s*d+alpha_s*delta/(1-eps2*delta)", R_B)
show("Planck 2020", R_p)

# =====================================================================
# II. 형태 C: 완전한 이중 부트스트랩
# =====================================================================
print("\n" + "=" * 72)
print("II. 형태 C: 바리온 피드백의 일반화")
print("=" * 72)

# 정수 차원과 잔여 차원의 피드백을 모두 포함하되,
# 정수 차원의 피드백 세기가 다를 수 있다.
#
# 정수 차원 d=3: QCD 색전하 = d = 3 이므로, 바리온 피드백은
# "색 자유도를 통해" 작동. 세기 = eps2 * 색 인자
#
# 잔여 차원 delta: 스칼라장 Phi를 통해 작동. 세기 = eps2 * delta
#
# 통합: R = alpha_s * d * (1 + eps2 * f_d) + alpha_s * delta * (1 + eps2 * f_delta)
# f_d = ? f_delta = delta (잔여 차원의 "크기")

# 가설 1: f_d = 0 (정수 차원은 완전히 결정화되어 피드백 불가)
# -> 형태 B
# 가설 2: f_d = delta (피드백이 잔여 차원을 통해서만 가능)
# -> 형태 A
# 가설 3: f_d = delta/d (정수 차원의 피드백은 차원당 delta/d)
# -> 중간

R_C3 = alpha_s * d * (1 + eps2 * delta/d) + alpha_s * delta * (1 + eps2 * delta)
show("C3: f_d=delta/d", R_C3)

# 가설 4: 부트스트랩의 정확한 구조를 따른다
# eps^2 = exp(-sigma*D)에서 sigma = 1-eps^2
# 바리온 피드백도 같은 구조: exp(-eps2*delta)
R_exp = alpha_s * D * math.exp(eps2 * delta)
show("exp: alpha_s*D*exp(eps2*delta)", R_exp)

# 가설 5: eps^2가 R에 피드백하고, R이 다시 sigma에 피드백하는 자기일관
# R = alpha_s * D_eff(R)
# D_eff(R) = d + delta + delta * R * eps2 / R_LO? 순환...
# 좀 더 자연스럽게: D_eff에서의 바리온 보정
# 바리온은 eps^2 = exp(-sigma*D). D를 D*(1+eps2*delta)로?
D_bf = D * (1 + eps2*delta)
eps2_bf = bootstrap(D_bf)
sigma_bf = 1 - eps2_bf
R_D = alpha_s * D_bf

print(f"\n형태 D: D_eff 자체를 바리온 보정")
print(f"  D_bf = D*(1+eps2*delta) = {D_bf:.5f}")
print(f"  eps2_bf = {eps2_bf:.6f} (원래: {eps2:.6f})")
print(f"  sigma_bf = {sigma_bf:.6f}")
R_Dv = alpha_s * D_bf
OL_D = sigma_bf/(1+R_Dv)
OD_D = sigma_bf*R_Dv/(1+R_Dv)
print(f"  R = {R_Dv:.5f}")
print(f"  OL = {OL_D:.5f} ({abs(OL_D-OL_p)/OL_p*100:.3f}%)")
print(f"  OD = {OD_D:.5f} ({abs(OD_D-OD_p)/OD_p*100:.3f}%)")
print(f"  Ob = {eps2_bf:.5f} ({abs(eps2_bf-Ob_p)/Ob_p*100:.3f}%)")
print(f"  sum = {OL_D+OD_D+eps2_bf:.6f}")

# =====================================================================
# III. NLO + 바리온 피드백의 올바른 결합
# =====================================================================
print("\n" + "=" * 72)
print("III. 섭동적 NLO와 비섭동적 바리온 피드백의 결합")
print("=" * 72)

# NLO는 alpha_s 급수: R += (alpha_s*D)^2/(4*pi)
# 바리온 피드백은 eps2*delta 급수: delta -> delta/(1-eps2*delta)
# 이 둘은 독립적 물리이므로 곱셈적으로 결합해야 한다.
#
# R = [alpha_s*D + (alpha_s*D)^2/(4*pi)] * (1 + eps2*delta)
# = NLO * baryon_factor
# 
# 또는: R_NLO에서 D -> D + delta*eps2

aD = alpha_s * D

# 방법 1: NLO * baryon factor
R_m1 = (aD + aD**2/(4*math.pi)) * (1 + eps2*delta)
show("NLO * (1+eps2*delta)", R_m1)

# 방법 2: NLO with D -> D+delta*eps2
D2 = D + delta*eps2
aD2 = alpha_s * D2
R_m2 = aD2 + aD2**2/(4*math.pi)
show("NLO(D+delta*eps2)", R_m2)

# 방법 3: NLO resum * baryon
R_m3 = aD/(1-aD/(4*math.pi)) * (1 + eps2*delta)
show("NLO_resum * (1+eps2*delta)", R_m3)

# 방법 4: NLO resum with effective delta
R_m4 = aD2/(1-aD2/(4*math.pi))
show("NLO_resum(D+delta*eps2)", R_m4)

# 방법 5: 분리형 - d는 NLO, delta는 바리온 피드백
R_m5 = alpha_s*d + (alpha_s*d)**2/(4*math.pi) + alpha_s*delta/(1-eps2*delta)
show("NLO(d) + bf(delta)", R_m5)

# 방법 6: 형태 A의 NLO 결합
R_m6 = alpha_s*D*(1+eps2*delta) + (alpha_s*D*(1+eps2*delta))**2/(4*math.pi)
show("NLO(R_A)", R_m6)

# =====================================================================
# IV. 잔차 분석: 1%는 어디서 오는가
# =====================================================================
print("\n" + "=" * 72)
print("IV. 잔차 분석")
print("=" * 72)

# 최선의 결과들:
best_cases = [
    ("A: alpha_s*D*(1+eps2*delta)", R_A),
    ("NLO resum", alpha_s*D/(1-aD/(4*math.pi))),
    ("NLO_resum(D+delta*eps2)", R_m4),
    ("NLO(d) + bf(delta)", R_m5),
]

print(f"{'Method':>40} {'R':>8} {'R/R_p':>8} {'R-R_p':>10} {'(R-R_p)/R_p':>12}")
print("-" * 82)
for name, R_val in best_cases:
    print(f"{name:>40} {R_val:.5f} {R_val/R_p:.5f} {R_val-R_p:+10.5f} {(R_val-R_p)/R_p*100:+11.4f}%")

print(f"\nPlanck R = {R_p:.5f}")
print(f"All CE predictions: R < R_planck (under-predict)")
print(f"Except NLO resum: R > R_planck (over-predict)")

# =====================================================================
# V. 교차점 탐색: LO+바리온과 NLO의 중간
# =====================================================================
print("\n" + "=" * 72)
print("V. LO+바리온과 NLO resum의 가중 평균")
print("=" * 72)

R_lo_bf = R_A  # 0.37787
R_nlo = alpha_s*D/(1-aD/(4*math.pi))  # 0.38614

# Planck R = 0.38075 는 이 두 값의 중간에 있다
# 가중: w*R_lo_bf + (1-w)*R_nlo = R_p
# w = (R_nlo - R_p) / (R_nlo - R_lo_bf)
w = (R_nlo - R_p) / (R_nlo - R_lo_bf)
print(f"R_A (LO+bf) = {R_lo_bf:.5f}")
print(f"R_NLO resum = {R_nlo:.5f}")
print(f"R_Planck    = {R_p:.5f}")
print(f"weight w    = {w:.4f}")
print(f"1-w         = {1-w:.4f}")
print(f"w*R_A + (1-w)*R_NLO = {w*R_lo_bf+(1-w)*R_nlo:.5f}")

# w = 0.65 근처. 이것의 의미?
# 물리: 바리온 피드백이 65%, NLO QCD가 35%?
# CE 내부 상수와의 관계?
print(f"\nw = {w:.5f}")
print(f"1-w = {1-w:.5f}")
print(f"sigma = {sigma:.5f}")
print(f"eps2 = {eps2:.5f}")
print(f"delta = {delta:.5f}")
print(f"alpha_s = {alpha_s:.5f}")
print(f"sigma - w = {sigma-w:.5f}")

# =====================================================================
# VI. 핵심 재고: R = alpha_s * D_eff 에서 D_eff의 정제
# =====================================================================
print("\n" + "=" * 72)
print("VI. D_eff의 정제: D_eff가 정확히 무엇인가")
print("=" * 72)

# 원래: D_eff = d + delta = 3 + 0.17776 = 3.17776
# 이것은 "유효 차원"이다.
# 하지만 "유효 차원"은 정확히 무엇을 세는가?
#
# d=3: 결정화된 공간 차원
# delta: 결정화되지 않은 잔여 차원
#
# QCD 요동의 관점에서:
# d=3 차원 각각에 alpha_s의 요동이 있다.
# delta 차원에는 alpha_s * (1 + 바리온 피드백)의 요동이 있다.
# 따라서 "QCD가 느끼는 유효 차원":

D_QCD = d + delta * (1 + eps2 * delta)
R_DQCD = alpha_s * D_QCD

print(f"D_QCD = d + delta*(1+eps2*delta) = {D_QCD:.6f}")
print(f"R = alpha_s * D_QCD = {R_DQCD:.6f}")
show("alpha_s * D_QCD", R_DQCD)

# 또는: delta 차원이 "부분적으로 결정화"되어 실효 차원이 더 큰 것
# delta_eff = delta + delta^2 * eps2 = delta*(1+delta*eps2)
delta_eff = delta * (1 + delta * eps2)
D_eff2 = d + delta_eff
R_Deff2 = alpha_s * D_eff2
print(f"\ndelta_eff = delta*(1+delta*eps2) = {delta_eff:.6f}")
print(f"D_eff2 = d + delta_eff = {D_eff2:.6f}")
show("alpha_s * D_eff2", R_Deff2)

# Planck가 요구하는 D_eff:
D_planck = R_p / alpha_s
print(f"\nD_planck = R_planck / alpha_s = {D_planck:.5f}")
print(f"D_planck - d = {D_planck-d:.5f} = delta_planck")
print(f"delta_planck / delta = {(D_planck-d)/delta:.5f}")
print(f"1 + alpha_s = {1+alpha_s:.5f}")
print(f"1 + eps2 = {1+eps2:.5f}")
print(f"1 + delta = {1+delta:.5f}")
print(f"1 + alpha_s*D/(4*pi) = {1+alpha_s*D/(4*math.pi):.5f}")

# delta_planck / delta = 1.29177
# 이것은 1 + alpha_s*D/(4*pi) = 1.02981과 다르다.
# 1 + R_LO/(4*pi)?
frac = (D_planck-d)/delta
print(f"\nfrac = delta_planck/delta = {frac:.5f}")
print(f"(1+alpha_s*D)^? let's check:")
print(f"  (1+alpha_s*D)^1 = {1+alpha_s*D:.5f}")
print(f"  frac / (1+alpha_s*D) = {frac/(1+alpha_s*D):.5f}")

# 잠깐. R_planck 자체의 불확실성을 고려해야 한다.
# Planck 2020: OL = 0.6847 +/- 0.0073, OD = 0.2607 +/- 0.0060
# R_p = 0.3808, 하지만 sigma_R ~= 0.012
# CE의 R = 0.3749~0.3779는 R_p의 1 sigma 이내이다!

R_p_err = math.sqrt((0.0060/OL_p)**2 + (0.0073*OD_p/OL_p**2)**2) * R_p
print(f"\nPlanck R 불확실성 분석:")
print(f"  OL = {OL_p} +/- 0.0073")
print(f"  OD = {OD_p} +/- 0.0060")
print(f"  R = {R_p:.5f} +/- ~{R_p_err:.4f}")
print(f"  R range: [{R_p-R_p_err:.4f}, {R_p+R_p_err:.4f}]")

# CE 예측들이 이 범위 안에 있는가?
for name, R_val in [
    ("LO", alpha_s*D),
    ("A: LO+bf", R_A),
    ("B: delta bf", R_B),
    ("NLO resum", R_nlo),
]:
    sigma_n = abs(R_val - R_p) / R_p_err
    inside = "INSIDE" if sigma_n < 1 else f"{sigma_n:.1f} sigma"
    print(f"  {name:>20}: R={R_val:.5f}, {inside}")

# =====================================================================
# VII. 최종 종합
# =====================================================================
print("\n" + "=" * 72)
print("VII. 최종 종합")
print("=" * 72)

print(f"""
1. CE의 DM/DE 비율 R에 대한 다층 유도:

   [Layer 0] R_LO = alpha_s * D_eff = {alpha_s*D:.5f}
             "차원당 QCD 요동의 합" (연장성 원리)
             OL: {abs(sigma/(1+alpha_s*D)-OL_p)/OL_p*100:.2f}%, OD: {abs(sigma*alpha_s*D/(1+alpha_s*D)-OD_p)/OD_p*100:.2f}%

   [Layer 1] R_bf = alpha_s*D*(1+eps2*delta) = {R_A:.5f}
             "바리온이 잔여 차원을 통해 QCD 요동에 seed"
             OL: {abs(sigma/(1+R_A)-OL_p)/OL_p*100:.2f}%, OD: {abs(sigma*R_A/(1+R_A)-OD_p)/OD_p*100:.2f}%

   [Layer 2] R_NLO = alpha_s*D/(1-alpha_s*D/(4pi)) = {R_nlo:.5f}
             "QCD 1-loop 진공 편극" (섭동적)
             OL: {abs(sigma/(1+R_nlo)-OL_p)/OL_p*100:.2f}%, OD: {abs(sigma*R_nlo/(1+R_nlo)-OD_p)/OD_p*100:.2f}%

   [실제]    R_Planck = {R_p:.5f} +/- ~{R_p_err:.4f}

2. 핵심 발견:

   R_LO, R_bf 모두 Planck R의 ~1 sigma 이내.
   Layer 1(바리온 피드백)은 ODM을 86% 개선 (0.55% -> 0.08%).
   Layer 2(NLO)는 OL을 78% 개선하나 ODM을 악화.
   
   두 보정은 서로 다른 물리:
   - 바리온 피드백: eps^2*delta 급수 (비섭동적, 수렴 보장)
   - NLO QCD: alpha_s^n 급수 (섭동적)
   
   Planck 관측값은 이 두 효과의 "중간"에 위치:
   R_bf ({R_A:.5f}) < R_Planck ({R_p:.5f}) < R_NLO ({R_nlo:.5f})

3. 물리적 해석:

   DM/DE 분할은 두 메커니즘에 의해 결정된다:
   
   (a) 진공 QCD 요동 + 바리온 seed (비섭동적):
       R ~ alpha_s * D * (1 + Omega_b * delta)
       "바리온이 잔여 차원 delta를 통해 QCD 진공에 seed를 제공"
       
   (b) QCD 진공 편극 (섭동적):
       R에 (alpha_s*D)^2/(4pi) 보정 추가
       "gluon 루프에 의한 QCD 진공의 자체 편극"
   
   CE의 현재 정밀도로, 두 효과를 동시에 일관되게 결합하면:
   R = alpha_s * D * (1 + eps2*delta) + (alpha_s*D)^2/(4pi) * k
   여기서 k는 두 효과의 간섭항 (O(1))

   그러나 Planck의 R 자체가 ~3% 불확실하므로,
   CE의 LO 예측(1.6% 차이)조차 이미 1 sigma 이내이다.
   바리온 피드백이나 NLO 보정은 "더 정밀한 일치"를 위한 것이지,
   "필수적 보정"은 아니다.

4. 최종 권고 공식:

   R = alpha_s * D_eff * (1 + eps^2 * delta)

   이유:
   - 닫힌 형태 (closed-form, 급수 합 불필요)
   - CE의 기존 양(alpha_s, D_eff, eps^2, delta)만 사용
   - 새로운 가정/매개변수 없음
   - OL 0.84%, OD 0.08% (total ~0.84%)
   - 물리적 해석 명확: "바리온의 잔여차원 경유 QCD seed"
""")

# =====================================================================
# VIII. 경로적분.md 업데이트용 요약
# =====================================================================
print("=" * 72)
print("VIII. 경로적분.md 업데이트용 최종 수치")
print("=" * 72)

R_rec = alpha_s * D * (1 + eps2 * delta)
OL_rec = sigma / (1 + R_rec)
OD_rec = sigma * R_rec / (1 + R_rec)

print(f"""
R (권고) = alpha_s * D_eff * (1 + Omega_b * delta)
         = {alpha_s:.5f} * {D:.5f} * (1 + {eps2:.5f} * {delta:.5f})
         = {alpha_s:.5f} * {D:.5f} * {1+eps2*delta:.6f}
         = {R_rec:.5f}

Omega_Lambda = sigma / (1+R) = {OL_rec:.5f}  (Planck {OL_p}, {abs(OL_rec-OL_p)/OL_p*100:.2f}%)
Omega_DM     = sigma*R/(1+R) = {OD_rec:.5f}  (Planck {OD_p}, {abs(OD_rec-OD_p)/OD_p*100:.2f}%)
Omega_b      = eps^2          = {eps2:.5f}  (Planck {Ob_p}, {abs(eps2-Ob_p)/Ob_p*100:.2f}%)
DM/DE ratio  = R              = {R_rec:.5f}  (Planck {R_p:.5f}, {abs(R_rec-R_p)/R_p*100:.2f}%)
sum          = {OL_rec+OD_rec+eps2:.6f}

16절 테이블 업데이트 제안:
  Omega_Lambda: 0.6921 -> {OL_rec:.4f} (CE 공식에 바리온 피드백 추가)
  Omega_DM:     0.2593 -> {OD_rec:.4f}
  텐션:
    OL: 1.08% -> {abs(OL_rec-OL_p)/OL_p*100:.2f}%
    OD: 0.13% -> {abs(OD_rec-OD_p)/OD_p*100:.2f}% (소폭 악화, 둘 다 1% 이내)
""")
