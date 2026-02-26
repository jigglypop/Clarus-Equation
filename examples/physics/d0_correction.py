"""
CE d=0 경계조건 보정: 전이 경로 적분
d=0 -> d=3 차원 결정화 과정에서의 보정항 계산
"""
import math

alpha_s = 0.11789
sin2_tW = 4 * alpha_s**(4/3)
delta = sin2_tW * (1 - sin2_tW)
D_eff = 3 + delta


def bootstrap_fp(D, tol=1e-15):
    if D < 0.01:
        return 1.0 - D + D**2/2
    x = 0.5
    for _ in range(100000):
        xn = math.exp(-(1-x)*D)
        if abs(xn - x) < tol:
            return xn
        x = xn
    return x


def digamma(x):
    h = 1e-8
    return (math.lgamma(x+h) - math.lgamma(x-h)) / (2*h)


eps2 = bootstrap_fp(D_eff)
print("=" * 65)
print("CE d=0 BOUNDARY CORRECTION")
print("=" * 65)

print(f"\nalpha_s = {alpha_s}")
print(f"delta   = {delta:.5f}")
print(f"D_eff   = {D_eff:.5f}")
print(f"eps^2   = {eps2:.6f}")

# I. 전이 궤적
print("\n--- I. 전이 궤적 eps^2(D) ---")
N = 10000
dD = D_eff / N
traj = [(i*dD, bootstrap_fp(i*dD)) for i in range(N+1)]

for D_val, e2_val in traj[::2000]:
    print(f"  D={D_val:7.4f}  eps^2={e2_val:.6f}  sigma={1-e2_val:.6f}")
print(f"  D={traj[-1][0]:7.4f}  eps^2={traj[-1][1]:.6f}  sigma={1-traj[-1][1]:.6f}")

# II. NLO 기준값
dk = 1 - eps2
R_NLO = alpha_s * D_eff + (alpha_s * D_eff)**2 / (4*math.pi)
OL_NLO = dk / (1 + R_NLO)
OD_NLO = dk * R_NLO / (1 + R_NLO)

print("\n--- II. NLO 기준값 ---")
print(f"R_NLO = {R_NLO:.5f}")
print(f"OL    = {OL_NLO:.5f}")
print(f"ODM   = {OD_NLO:.5f}")

# III. d=0 1-loop 자기에너지 보정
# Phi 자기에너지: d=D_eff에서 계산 vs d=0 잔해(delta)에서 계산
# dim reg 유한 부분 차이가 D_eff에 보정을 준다
print("\n--- III. d=0 1-loop 자기에너지 보정 ---")

psi_delta = digamma(1 - delta/2)
psi_3 = digamma(-0.5)
pole_delta = 1/(delta - 4)
pole_3 = 1/(3 - 4)
finite_delta = pole_delta + psi_delta
finite_3 = pole_3 + psi_3
Delta_fin = finite_delta - finite_3

lHP = delta**2
corr_1loop = lHP / (16 * math.pi**2) * Delta_fin

print(f"psi(1-delta/2) = {psi_delta:.5f}")
print(f"psi(-1/2)      = {psi_3:.5f}")
print(f"1/(delta-4)    = {pole_delta:.5f}")
print(f"1/(3-4)        = {pole_3:.5f}")
print(f"Delta_finite   = {Delta_fin:.5f}")
print(f"lambda_HP      = {lHP:.5f}")
print(f"correction     = {corr_1loop:.8f} ({corr_1loop*100:.5f}%)")

# 이 보정은 너무 작다(0.005%). dim reg 자체가 아니라 다른 접근이 필요.

# IV. 물리적 보정: 전이 구간 적분
print("\n--- IV. 전이 구간 적분 보정 ---")

# 부트스트랩은 고정점 방정식. 그런데 물리적 전이는 D=0 -> D=D_eff.
# 전이 동안 eps^2(D)가 연속적으로 변한다.
# 관측되는 Omega_b는 고정점 값이 아니라 동결 시점의 값.
#
# 동결 조건: Hubble rate = 전이 rate
# H(tau) = Gamma_transition(tau) 에서 동결
#
# 전이 rate: d(eps^2)/dD ~ eps^2 * D (부트스트랩 미분)
# 전이가 "빨리" 일어나면 tau* ~ 1, 느리면 tau* < 1

# eps^2(D)의 기울기 (전이 속도)
derivs = []
for i in range(1, N):
    d_eps = (traj[i+1][1] - traj[i-1][1]) / (2*dD)
    derivs.append((traj[i][0], d_eps))

max_deriv_D = max(derivs, key=lambda x: abs(x[1]))
print(f"최대 |deps^2/dD| at D = {max_deriv_D[0]:.4f}, rate = {max_deriv_D[1]:.6f}")

# 전이 시점: 전체 전이의 90%가 완료된 D
eps2_range = traj[0][1] - traj[-1][1]  # ~0.951
for D_val, e2_val in traj:
    frac = (traj[0][1] - e2_val) / eps2_range
    if frac >= 0.90:
        D_90 = D_val
        eps2_90 = e2_val
        break

print(f"90% 전이 완료: D = {D_90:.4f}, eps^2 = {eps2_90:.6f}")
print(f"tau_90 = D_90/D_eff = {D_90/D_eff:.4f}")

# V. 유효 D_eff 보정: 전이 경로 가중
print("\n--- V. 유효 D_eff 보정 ---")

# 물리적 아이디어:
# 부트스트랩은 D=D_eff에서의 고정점을 준다.
# 하지만 전이가 D=0에서 시작하므로,
# 유효 D는 경로 위에서 "체류 시간" 가중 평균이다.
#
# 가장 자연스러운 가중: eps^2(D) 자체
# (생존 경로가 많을수록 그 D에서 오래 머문다)
#
# D_eff_corrected = int_0^{D_eff} D * eps^2(D) dD / int_0^{D_eff} eps^2(D) dD

num = 0
den = 0
for i in range(N):
    Di = 0.5*(traj[i][0] + traj[i+1][0])
    ei = 0.5*(traj[i][1] + traj[i+1][1])
    num += Di * ei * dD
    den += ei * dD

D_weighted = num / den
print(f"<D>_eps2 = {D_weighted:.5f}")
print(f"D_eff    = {D_eff:.5f}")
print(f"비율     = {D_weighted/D_eff:.6f}")
print(f"보정     = {(D_weighted/D_eff - 1)*100:.3f}%")

# 이건 d=0 근방에서 eps^2 ~ 1이니 가중이 큰 쪽으로 당겨진다.
# D_weighted < D_eff (작아진다) -> eps^2 커진다 -> Omega_b 커진다

eps2_w = bootstrap_fp(D_weighted)
dk_w = 1 - eps2_w
R_w = alpha_s * D_weighted + (alpha_s * D_weighted)**2 / (4*math.pi)
OL_w = dk_w / (1 + R_w)
OD_w = dk_w * R_w / (1 + R_w)

print(f"\neps^2(D_weighted) = {eps2_w:.6f}")
print(f"Omega_L  = {OL_w:.5f}")
print(f"Omega_DM = {OD_w:.5f}")

# VI. 반대: 작용 가중
print("\n--- VI. 작용 가중 (sigma = 1-eps^2) ---")

# 억압(접힘)이 일어난 양 sigma에 비례하는 가중
# 이건 전이가 많이 진행된 곳에 가중을 준다
num2 = 0
den2 = 0
for i in range(N):
    Di = 0.5*(traj[i][0] + traj[i+1][0])
    si = 1 - 0.5*(traj[i][1] + traj[i+1][1])
    num2 += Di * si * dD
    den2 += si * dD

D_sigma = num2 / den2
print(f"<D>_sigma = {D_sigma:.5f}")
print(f"비율      = {D_sigma/D_eff:.6f}")

eps2_s = bootstrap_fp(D_sigma)
dk_s = 1 - eps2_s
R_s = alpha_s * D_sigma + (alpha_s * D_sigma)**2 / (4*math.pi)
OL_s = dk_s / (1 + R_s)
OD_s = dk_s * R_s / (1 + R_s)
print(f"eps^2 = {eps2_s:.6f}, OL = {OL_s:.5f}, ODM = {OD_s:.5f}")

# VII. 가장 정교한 접근: NNLO 보정
print("\n--- VII. NNLO (2-loop) QCD 보정 ---")

# R = alpha_s D + (alpha_s D)^2/(4pi) + c2 * (alpha_s D)^3/(4pi)^2
# c2는 QCD beta function에서 결정
# beta_0 = 11 - 2/3 * n_f, n_f=6 -> beta_0 = 7
# c2 = beta_0/(2pi) = 7/(2pi) = 1.114

beta_0 = 11 - 2/3 * 6
c2 = beta_0 / (2*math.pi)
aD = alpha_s * D_eff
R_NNLO = aD + aD**2/(4*math.pi) + c2 * aD**3/(4*math.pi)**2

OL_NNLO = dk / (1 + R_NNLO)
OD_NNLO = dk * R_NNLO / (1 + R_NNLO)

print(f"beta_0 = {beta_0:.1f}")
print(f"c2     = {c2:.4f}")
print(f"R_NNLO = {R_NNLO:.5f}")
print(f"OL     = {OL_NNLO:.5f}")
print(f"ODM    = {OD_NNLO:.5f}")
print(f"DM/DE  = {OD_NNLO/OL_NNLO:.5f}")

# VIII. 최종 비교
print("\n" + "=" * 65)
print("VIII. 최종 비교")
print("=" * 65)
fmt = "{:>25s} {:>10s} {:>8s} {:>10s} {:>8s} {:>10s} {:>8s}"
print(fmt.format("", "Omega_b", "diff", "Omega_L", "diff", "Omega_DM", "diff"))
print("-" * 85)

Ob_p, OL_p, OD_p = 0.04930, 0.68470, 0.26070

def row(name, ob, ol, od):
    print(f"{name:>25s} {ob:10.5f} {abs(ob-Ob_p)/Ob_p*100:7.3f}% {ol:10.5f} {abs(ol-OL_p)/OL_p*100:7.3f}% {od:10.5f} {abs(od-OD_p)/OD_p*100:7.3f}%")

row("Planck 2020", Ob_p, OL_p, OD_p)
row("LO", eps2, dk/(1+alpha_s*D_eff), dk*alpha_s*D_eff/(1+alpha_s*D_eff))
row("NLO", eps2, OL_NLO, OD_NLO)
row("NNLO", eps2, OL_NNLO, OD_NNLO)
row("NLO + D_weighted", eps2_w, OL_w, OD_w)
row("NLO + D_sigma", eps2_s, OL_s, OD_s)

# IX. c2 피팅
print("\n--- IX. c2 최적화 (NNLO 계수) ---")
# R = aD + aD^2/(4pi) + c2 * aD^3/(4pi)^2
# OL = dk/(1+R), ODM = dk*R/(1+R)
# Planck: OL=0.6847, ODM=0.2607
# 최적 c2?

best_c2 = None
best_chi2 = 1e10
for ic in range(-500, 500):
    cc = ic * 0.01
    Rc = aD + aD**2/(4*math.pi) + cc * aD**3/(4*math.pi)**2
    OLc = dk/(1+Rc)
    ODc = dk*Rc/(1+Rc)
    chi2 = ((OLc - OL_p)/0.007)**2 + ((ODc - OD_p)/0.005)**2
    if chi2 < best_chi2:
        best_chi2 = chi2
        best_c2 = cc

Rb = aD + aD**2/(4*math.pi) + best_c2 * aD**3/(4*math.pi)**2
OLb = dk/(1+Rb)
ODb = dk*Rb/(1+Rb)
print(f"최적 c2 = {best_c2:.2f}")
print(f"R  = {Rb:.5f}")
print(f"OL = {OLb:.5f} ({abs(OLb-OL_p)/OL_p*100:.3f}%)")
print(f"ODM= {ODb:.5f} ({abs(ODb-OD_p)/OD_p*100:.3f}%)")
print(f"chi2 = {best_chi2:.4f}")
print(f"QCD 비교: beta_0/(2pi) = {beta_0/(2*math.pi):.3f}")
print(f"  c2={best_c2:.2f} vs beta_0/(2pi)={beta_0/(2*math.pi):.3f}")
