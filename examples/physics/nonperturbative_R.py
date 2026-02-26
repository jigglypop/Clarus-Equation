"""
CE 비섭동적 DM/DE 분할: R의 자기일관 유도
섭동 급수 R = alpha_s*D + ... 대신, CE 내부 관계식으로 R을 비섭동적으로 결정
"""
import math

alpha_s = 0.11789
sin2_tW = 4 * alpha_s**(4/3)
delta = sin2_tW * (1 - sin2_tW)
D = 3 + delta

def fp(D_val):
    x = 0.5
    for _ in range(100000):
        x = math.exp(-(1-x)*D_val)
    return x

eps2 = fp(D)
dk = 1 - eps2
OL_p, OD_p, Ob_p = 0.68470, 0.26070, 0.04930

print("=" * 65)
print("CE 비섭동적 R 유도")
print("=" * 65)
print(f"alpha_s = {alpha_s}")
print(f"D_eff   = {D:.5f}")
print(f"eps^2   = {eps2:.6f}")
print(f"dk      = 1-eps^2 = {dk:.6f}")

# =================================================================
# I. 기존 관계식 정리
# =================================================================
print("\n" + "=" * 65)
print("I. CE 내부의 R 관련 관계식")
print("=" * 65)

# (1) 섭동적: R = alpha_s * D_eff (LO)
R_LO = alpha_s * D
print(f"(1) LO:  R = alpha_s * D = {R_LO:.5f}")

# (2) 섭동적: R = alpha_s * pi (A4 공리)
R_A4 = alpha_s * math.pi
print(f"(2) A4:  R = alpha_s * pi = {R_A4:.5f}")

# (3) 비섭동적: F = 1 + R (양성자 반경 폼팩터)
#     F = 1 + alpha_s * D_eff = 1.3746 (LO)
#     F = M_H / M_Z = 125.25/91.1876 = 1.3738
F_obs = 125.25 / 91.1876
R_F = F_obs - 1
print(f"(3) F:   R = M_H/M_Z - 1 = {F_obs:.4f} - 1 = {R_F:.5f}")

# (4) DM/baryon 비율에서
#     Omega_DM/Omega_b = R * (1+R) / (R * eps2... no)
#     Omega_DM = dk * R / (1+R)
#     Omega_b = eps2
#     Omega_DM/Omega_b = dk * R / (eps2 * (1+R))
#     Planck: 0.2607/0.0493 = 5.288
DM_b_obs = OD_p / Ob_p
print(f"(4) DM/b ratio (Planck) = {DM_b_obs:.3f}")

# =================================================================
# II. 비섭동적 R: 부트스트랩 확장
# =================================================================
print("\n" + "=" * 65)
print("II. R의 부트스트랩화")
print("=" * 65)

# 핵심 아이디어:
# eps^2 = exp(-(1-eps^2)*D) 는 "생존 확률" 부트스트랩
# R은 "암흑 분할"에 대한 자기일관 방정식이어야 한다
#
# 물리적 근거: DM은 QCD 요동의 응축.
# QCD 요동의 세기는 alpha_s이고, alpha_s는 D에서 유도됨.
# 그런데 D 자체가 R에 의존한다면? -> 자기일관성
#
# 시도 1: R = alpha_s(R) * D_eff
# alpha_s가 R에 의존? CE에서 alpha_s는 기하학적이라 R에 무관.
# -> 실패.

# 시도 2: R도 부트스트랩
# 부트스트랩에서: eps^2 = exp(-(1-eps^2)*D)
# 양변에 "dark sector" 버전:
# sigma = 1 - eps^2 = dark total
# sigma = 1 - exp(-sigma * D)
#
# DM 분율: R/(1+R) = Omega_DM / (Omega_DM + Omega_Lambda)
#                   = Omega_DM / dark_total
# 즉 R/(1+R) = DM 내부 분율
#
# QCD 1-loop에서: DM/dark = alpha_s*D / (1 + alpha_s*D)
# 이것도 부트스트랩 형태로?
# R/(1+R) = 1 - exp(-R * something)???

# 시도 3: 연장성 원리
# 부트스트랩: eps^2 = exp(-sigma * D)
# "내부 부트스트랩": r = exp(-r_bar * D_internal)
# r = R/(1+R), r_bar = 1-r = 1/(1+R)
# r = exp(-r_bar * D_int)
# R/(1+R) = exp(-(1/(1+R)) * D_int)
#
# D_int는 뭔가? QCD의 유효 차원.
# CE에서 alpha_s = sum of gauge couplings의 QCD part
# QCD 섹터의 유효 차원 = d * alpha_s / alpha_total?
# = 3 * 0.11789 / 0.15924 = 2.221?

alpha_total = 1/(2*math.pi)
D_QCD = 3 * alpha_s / alpha_total
print(f"D_QCD = d * alpha_s/alpha_total = {D_QCD:.5f}")

# R/(1+R) = exp(-(1/(1+R)) * D_QCD)
# R = 자기일관해
r = 0.3
for _ in range(100000):
    r_new = math.exp(-(1/(1+r)) * D_QCD)
    r = r + 0.001*(r_new - r)  # damped iteration

R_boot = r / (1-r) if r < 1 else r
print(f"부트스트랩 R: r=R/(1+R) = {r:.5f}, R = {r/(1-r):.5f}")

# 직접 반복: R/(1+R) = exp(-D_QCD/(1+R))
R_iter = 0.4
for _ in range(100000):
    rhs = math.exp(-D_QCD / (1 + R_iter))
    R_new = rhs / (1 - rhs) if rhs < 1 else 100
    R_iter = R_iter + 0.01 * (R_new - R_iter)

OL_boot = dk / (1 + R_iter)
OD_boot = dk * R_iter / (1 + R_iter)
print(f"\nR_bootstrap = {R_iter:.5f}")
print(f"OL = {OL_boot:.5f} ({abs(OL_boot-OL_p)/OL_p*100:.3f}%)")
print(f"ODM = {OD_boot:.5f} ({abs(OD_boot-OD_p)/OD_p*100:.3f}%)")

# =================================================================
# III. F = M_H/M_Z 로부터의 비섭동적 R
# =================================================================
print("\n" + "=" * 65)
print("III. F = M_H/M_Z 에서 R 유도")
print("=" * 65)

# F = 1 + R (CE 정의)
# M_H = M_Z * F -> R = M_H/M_Z - 1
# 이건 이미 관측값이지만, CE 내부적으로 F를 유도할 수 있다.
#
# CE: F = 1 + alpha_s * D_eff (LO)
# 하지만 M_H/M_Z = 1.3738은 비섭동적 값이다.
# F를 비섭동적으로?
#
# 시도: F = exp(alpha_s * D_eff * ...) 의 비섭동적 버전
# F = exp(R)의 truncation이 F = 1+R.
# 비섭동적: F = exp(R_nonpert)?

# R from F = exp(R):
# 1.3738 = exp(R) -> R = ln(1.3738) = 0.3175
R_exp = math.log(F_obs)
OL_exp = dk / (1 + R_exp)
OD_exp = dk * R_exp / (1 + R_exp)
print(f"F = exp(R): R = ln(M_H/M_Z) = ln({F_obs:.4f}) = {R_exp:.5f}")
print(f"OL = {OL_exp:.5f} ({abs(OL_exp-OL_p)/OL_p*100:.3f}%)")
print(f"ODM = {OD_exp:.5f} ({abs(OD_exp-OD_p)/OD_p*100:.3f}%)")

# R from F = 1/(1-R):
# 1.3738 = 1/(1-R) -> R = 1 - 1/1.3738 = 0.2720
R_geo = 1 - 1/F_obs
OL_geo = dk / (1 + R_geo)
OD_geo = dk * R_geo / (1 + R_geo)
print(f"\nF = 1/(1-R): R = 1-M_Z/M_H = {R_geo:.5f}")
print(f"OL = {OL_geo:.5f} ({abs(OL_geo-OL_p)/OL_p*100:.3f}%)")
print(f"ODM = {OD_geo:.5f} ({abs(OD_geo-OD_p)/OD_p*100:.3f}%)")

# =================================================================
# IV. 자기일관 부트스트랩 with QCD 차원
# =================================================================
print("\n" + "=" * 65)
print("IV. D_QCD 스캔")
print("=" * 65)

# R/(1+R) = exp(-D_qcd/(1+R)) 를 다양한 D_qcd에 대해 풀고
# Planck에 가장 가까운 것을 찾는다

print(f"{'D_QCD':>10} {'R':>10} {'OL':>10} {'OL%':>8} {'ODM':>10} {'ODM%':>8}")
print("-" * 60)

best_Dq = 0
best_metric = 1e10
for i in range(1, 500):
    Dq = i * 0.01
    R_t = 0.4
    for _ in range(100000):
        rhs = math.exp(-Dq / (1 + R_t))
        if rhs >= 1:
            R_t = 100
            break
        R_n = rhs / (1 - rhs)
        R_t = R_t + 0.01 * (R_n - R_t)
    ol = dk/(1+R_t)
    od = dk*R_t/(1+R_t)
    d_ol = abs(ol-OL_p)/OL_p*100
    d_od = abs(od-OD_p)/OD_p*100
    metric = math.sqrt(d_ol**2 + d_od**2)
    if metric < best_metric:
        best_metric = metric
        best_Dq = Dq
        best_R = R_t
        best_ol = ol
        best_od = od
    if i % 50 == 0:
        print(f"{Dq:10.2f} {R_t:10.5f} {ol:10.5f} {d_ol:7.3f}% {od:10.5f} {d_od:7.3f}%")

print("-" * 60)
print(f"Best D_QCD = {best_Dq:.2f}")
print(f"R = {best_R:.5f}")
print(f"OL = {best_ol:.5f} ({abs(best_ol-OL_p)/OL_p*100:.3f}%)")
print(f"ODM = {best_od:.5f} ({abs(best_od-OD_p)/OD_p*100:.3f}%)")

# D_QCD 해석
print(f"\nD_QCD = {best_Dq:.2f} 의 해석:")
print(f"  D_QCD / D_eff = {best_Dq/D:.4f}")
print(f"  D_QCD / d = {best_Dq/3:.4f}")
print(f"  D_QCD / pi = {best_Dq/math.pi:.4f}")
print(f"  D_QCD / (alpha_s * D^2) = {best_Dq/(alpha_s*D**2):.4f}")
print(f"  D_QCD / (d * alpha_s/alpha_total) = {best_Dq/D_QCD:.4f}")

# =================================================================
# V. 핵심 발견: R = alpha_s * pi vs R = alpha_s * D_eff
# =================================================================
print("\n" + "=" * 65)
print("V. A4 공리 재검토: R = alpha_s * pi vs alpha_s * D_eff")
print("=" * 65)

# 원래 A4: Omega_DM/Omega_Lambda = alpha_s * pi
# 정제: Omega_DM/Omega_Lambda = alpha_s * D_eff (연장성 원리)
# 차이?

R_pi = alpha_s * math.pi
R_D = alpha_s * D

OL_pi = dk/(1+R_pi)
OD_pi = dk*R_pi/(1+R_pi)

OL_D = dk/(1+R_D)
OD_D = dk*R_D/(1+R_D)

print(f"R = alpha_s * pi   = {R_pi:.5f}  ->  OL={OL_pi:.5f}({abs(OL_pi-OL_p)/OL_p*100:.3f}%)  ODM={OD_pi:.5f}({abs(OD_pi-OL_p)/OL_p*100:.3f}%)")
print(f"R = alpha_s * D    = {R_D:.5f}  ->  OL={OL_D:.5f}({abs(OL_D-OL_p)/OL_p*100:.3f}%)  ODM={OD_D:.5f}({abs(OD_D-OD_p)/OD_p*100:.3f}%)")

# 중간: R = alpha_s * (pi + delta) / (1 + delta/pi) ?
# 또는 R = alpha_s * sqrt(pi * D_eff)?
R_sqrt = alpha_s * math.sqrt(math.pi * D)
OL_sq = dk/(1+R_sqrt)
OD_sq = dk*R_sqrt/(1+R_sqrt)
print(f"R = alpha_s*sqrt(pi*D) = {R_sqrt:.5f}  ->  OL={OL_sq:.5f}({abs(OL_sq-OL_p)/OL_p*100:.3f}%)  ODM={OD_sq:.5f}({abs(OD_sq-OD_p)/OD_p*100:.3f}%)")

# NLO 재합산
R_resum = alpha_s * D / (1 - alpha_s*D/(4*math.pi))
OL_re = dk/(1+R_resum)
OD_re = dk*R_resum/(1+R_resum)
print(f"R = aD/(1-aD/4pi)  = {R_resum:.5f}  ->  OL={OL_re:.5f}({abs(OL_re-OL_p)/OL_p*100:.3f}%)  ODM={OD_re:.5f}({abs(OD_re-OD_p)/OD_p*100:.3f}%)")

# R = alpha_s * D * exp(alpha_s*D/(4pi)) (exponentiated NLO)
R_expNLO = alpha_s * D * math.exp(alpha_s*D/(4*math.pi))
OL_en = dk/(1+R_expNLO)
OD_en = dk*R_expNLO/(1+R_expNLO)
print(f"R = aD*exp(aD/4pi) = {R_expNLO:.5f}  ->  OL={OL_en:.5f}({abs(OL_en-OL_p)/OL_p*100:.3f}%)  ODM={OD_en:.5f}({abs(OD_en-OD_p)/OD_p*100:.3f}%)")

# Planck exact
R_planck = OD_p / OL_p
print(f"\nPlanck exact: R = {R_planck:.5f}")
print(f"  R / (alpha_s * D) = {R_planck / (alpha_s*D):.5f}")
print(f"  R / (alpha_s * pi) = {R_planck / (alpha_s*math.pi):.5f}")
print(f"  R / alpha_s = {R_planck/alpha_s:.5f}")
print(f"  R / alpha_s / D = {R_planck/alpha_s/D:.5f}")
print(f"  sqrt(R/alpha_s) = {math.sqrt(R_planck/alpha_s):.5f}")
print(f"  pi = {math.pi:.5f}")
print(f"  D_eff = {D:.5f}")
print(f"  (R/alpha_s - D) = {R_planck/alpha_s - D:.5f}")
print(f"  (R/alpha_s - pi) = {R_planck/alpha_s - math.pi:.5f}")

# =================================================================
# VI. 최종 비교 테이블
# =================================================================
print("\n" + "=" * 65)
print("VI. 최종 비교")
print("=" * 65)
print(f"{'Method':>30} {'R':>8} {'OL':>9} {'OL%':>7} {'ODM':>9} {'ODM%':>7} {'total%':>7}")
print("-" * 78)

def row(name, R_val):
    ol = dk/(1+R_val)
    od = dk*R_val/(1+R_val)
    d1 = abs(ol-OL_p)/OL_p*100
    d2 = abs(od-OD_p)/OD_p*100
    dt = math.sqrt(d1**2+d2**2)
    print(f"{name:>30} {R_val:8.5f} {ol:9.5f} {d1:6.3f}% {od:9.5f} {d2:6.3f}% {dt:6.3f}%")

row("Planck 2020", R_planck)
row("LO: alpha_s*D", R_D)
row("A4: alpha_s*pi", R_pi)
row("NLO resum: aD/(1-aD/4pi)", R_resum)
row("exp NLO: aD*exp(aD/4pi)", R_expNLO)
row("F=exp(R): ln(MH/MZ)", R_exp)
row("F=1/(1-R): 1-MZ/MH", R_geo)
row("sqrt(pi*D)*alpha_s", R_sqrt)
row("bootstrap D_QCD", best_R)
