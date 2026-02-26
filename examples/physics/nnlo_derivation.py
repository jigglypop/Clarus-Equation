"""
CE NNLO: QCD beta function에서 c2 계수 유도
R = alpha_s*D + (alpha_s*D)^2/(4pi) + c2*(alpha_s*D)^3/(4pi)^2
"""
import math

alpha_s = 0.11789
sin2_tW = 4 * alpha_s**(4/3)
delta = sin2_tW * (1 - sin2_tW)
D = 3 + delta
aD = alpha_s * D

def fp(D_val):
    x = 0.5
    for _ in range(100000):
        x = math.exp(-(1-x)*D_val)
    return x

eps2 = fp(D)
dk = 1 - eps2
OL_p, OD_p = 0.68470, 0.26070

print("=" * 65)
print("CE NNLO: QCD beta function -> c2")
print("=" * 65)
print(f"aD = alpha_s * D_eff = {aD:.5f}")
print(f"(aD)^2/(4pi) = {aD**2/(4*math.pi):.6f}")
print(f"(aD)^3/(4pi)^2 = {aD**3/(4*math.pi)**2:.7f}")
print()

# QCD beta function 계수
nf_vals = {3: "QCD confinement", 5: "bottom threshold", 6: "top threshold"}
beta_data = {}
for nf in [3, 5, 6]:
    b0 = (33 - 2*nf) / 12
    b1 = (153 - 19*nf) / 24
    b2 = (2857 - 5033/9*nf + 325/27*nf**2) / 128
    beta_data[nf] = (b0, b1, b2)
    print(f"nf={nf} ({nf_vals[nf]}): b0={b0:.4f}, b1={b1:.4f}, b2={b2:.4f}")

print()
print("=" * 65)
print("NNLO 계수 후보")
print("=" * 65)

def calc_R(c2):
    return aD + aD**2/(4*math.pi) + c2 * aD**3/(4*math.pi)**2

def calc_obs(R):
    ol = dk/(1+R)
    od = dk*R/(1+R)
    return ol, od

candidates = []

# 1. c2 = C_A = N_c = d = 3 (adjoint Casimir)
candidates.append(("C_A = N_c = d = 3", 3.0))

# 2. c2 = C_F = (N_c^2-1)/(2*N_c) = 4/3
candidates.append(("C_F = 4/3", 4/3))

# 3. c2 = b1/b0 for each nf
for nf in [3, 5, 6]:
    b0, b1, _ = beta_data[nf]
    candidates.append((f"b1/b0 (nf={nf})", b1/b0))

# 4. c2 = 11*C_A/(12*pi) (pure glue beta_0)
candidates.append(("11*C_A/(12pi)", 11*3/(12*math.pi)))

# 5. c2 = pi (geometric)
candidates.append(("pi", math.pi))

# 6. c2 = D_eff (self-referential)
candidates.append(("D_eff", D))

# 7. c2 = 1/delta
candidates.append(("1/delta", 1/delta))

# 8. Adler function: c2 = 1.986 - 0.1153*nf
for nf in [3, 5]:
    candidates.append((f"Adler (nf={nf})", 1.986 - 0.1153*nf))

# 9. c2 = C_A * C_F = 3 * 4/3 = 4
candidates.append(("C_A*C_F = 4", 4.0))

# 10. c2 = (beta_0)^2 / (4*pi) for nf=3
b0_3 = beta_data[3][0]
candidates.append((f"b0^2/(4pi) nf=3", b0_3**2/(4*math.pi)))

print()
hdr = f"{'Method':>30s} {'c2':>8s} {'R':>10s} {'OL':>10s} {'OL%':>8s} {'ODM':>10s} {'ODM%':>8s}"
print(hdr)
print("-" * 85)

# Planck reference
print(f"{'Planck 2020':>30s} {'---':>8s} {'0.38075':>10s} {OL_p:10.5f} {'---':>8s} {OD_p:10.5f} {'---':>8s}")

# LO
R_LO = aD
ol_lo, od_lo = calc_obs(R_LO)
print(f"{'LO (c2=0)':>30s} {'0':>8s} {R_LO:10.5f} {ol_lo:10.5f} {abs(ol_lo-OL_p)/OL_p*100:7.3f}% {od_lo:10.5f} {abs(od_lo-OD_p)/OD_p*100:7.3f}%")

# NLO
R_NLO = aD + aD**2/(4*math.pi)
ol_nlo, od_nlo = calc_obs(R_NLO)
print(f"{'NLO':>30s} {'1':>8s} {R_NLO:10.5f} {ol_nlo:10.5f} {abs(ol_nlo-OL_p)/OL_p*100:7.3f}% {od_nlo:10.5f} {abs(od_nlo-OD_p)/OD_p*100:7.3f}%")

print("-" * 85)

best = None
best_metric = 1e10

for name, c2 in candidates:
    R = calc_R(c2)
    ol, od = calc_obs(R)
    d_ol = abs(ol - OL_p) / OL_p * 100
    d_od = abs(od - OD_p) / OD_p * 100
    metric = math.sqrt(d_ol**2 + d_od**2)
    mark = ""
    if metric < best_metric:
        best_metric = metric
        best = (name, c2, R, ol, od, d_ol, d_od)
        mark = " <--"
    print(f"{name:>30s} {c2:8.4f} {R:10.5f} {ol:10.5f} {d_ol:7.3f}% {od:10.5f} {d_od:7.3f}%{mark}")

print("-" * 85)
print(f"\n{'BEST':>30s}: {best[0]}, c2={best[1]:.4f}")
print(f"  OL  = {best[3]:.5f} ({best[5]:.3f}%)")
print(f"  ODM = {best[4]:.5f} ({best[6]:.3f}%)")

# 최적 c2 정밀 탐색
print("\n" + "=" * 65)
print("최적 c2 정밀 탐색 (chi^2 최소화)")
print("=" * 65)

sig_OL = 0.0073  # Planck error
sig_OD = 0.0055

best_c2_fine = 0
best_chi2 = 1e10
for i in range(-10000, 10000):
    c2t = i * 0.01
    Rt = calc_R(c2t)
    olt, odt = calc_obs(Rt)
    chi2 = ((olt - OL_p)/sig_OL)**2 + ((odt - OD_p)/sig_OD)**2
    if chi2 < best_chi2:
        best_chi2 = chi2
        best_c2_fine = c2t

Rf = calc_R(best_c2_fine)
olf, odf = calc_obs(Rf)
print(f"최적 c2 = {best_c2_fine:.2f}")
print(f"chi^2   = {best_chi2:.4f}")
print(f"R       = {Rf:.6f}")
print(f"OL      = {olf:.5f} ({abs(olf-OL_p)/OL_p*100:.3f}%)")
print(f"ODM     = {odf:.5f} ({abs(odf-OD_p)/OD_p*100:.3f}%)")

# 이 최적 c2가 QCD 계수와 어떤 관계인지
print(f"\n최적 c2 = {best_c2_fine:.2f} 의 분석:")
print(f"  c2 / C_A = {best_c2_fine/3:.4f}")
print(f"  c2 / C_F = {best_c2_fine/(4/3):.4f}")
print(f"  c2 / pi = {best_c2_fine/math.pi:.4f}")
print(f"  c2 / D_eff = {best_c2_fine/D:.4f}")
for nf in [3, 5]:
    b0, b1, _ = beta_data[nf]
    print(f"  c2 / (b1/b0, nf={nf}) = {best_c2_fine/(b1/b0):.4f}")

# N3LO까지 포함 가능?
print("\n" + "=" * 65)
print("N3LO 효과 추정")
print("=" * 65)
# (aD)^4 / (4pi)^3
n3lo_term = aD**4 / (4*math.pi)**3
print(f"(aD)^4/(4pi)^3 = {n3lo_term:.8f}")
print(f"NLO  기여: {aD**2/(4*math.pi):.6f}")
print(f"NNLO 기여 (c2=1): {aD**3/(4*math.pi)**2:.7f}")
print(f"N3LO 기여 (c3=1): {n3lo_term:.8f}")
print(f"NLO/LO  = {aD/(4*math.pi)*100:.3f}%")
print(f"NNLO/NLO = {aD/(4*math.pi)*100:.3f}%")
print(f"-> 급수 수렴: 각 차수마다 {aD/(4*math.pi)*100:.1f}% 감소")
print(f"-> NNLO 이후 무한 재합산:")
geo_sum = 1 / (1 - aD/(4*math.pi))
print(f"   기하급수합 1/(1-aD/4pi) = {geo_sum:.5f}")
R_resum = aD * geo_sum
ol_re, od_re = calc_obs(R_resum)
print(f"   R_resum = {R_resum:.5f}")
print(f"   OL = {ol_re:.5f} ({abs(ol_re-OL_p)/OL_p*100:.3f}%)")
print(f"   ODM = {od_re:.5f} ({abs(od_re-OD_p)/OD_p*100:.3f}%)")

# Pade [1,1] resummation
# R = aD * (1 + a1*x) / (1 - a2*x) where x = aD/(4pi)
# a1 + a2 = 1 (NLO), a1*a2 = ? (from NNLO)
# 단순: [0,1] Pade = aD / (1 - aD/(4pi))
R_pade01 = aD / (1 - aD/(4*math.pi))
ol_p01, od_p01 = calc_obs(R_pade01)
print(f"\n   Pade [0,1]: R = aD/(1-aD/4pi) = {R_pade01:.5f}")
print(f"   OL = {ol_p01:.5f} ({abs(ol_p01-OL_p)/OL_p*100:.3f}%)")
print(f"   ODM = {od_p01:.5f} ({abs(od_p01-OD_p)/OD_p*100:.3f}%)")
