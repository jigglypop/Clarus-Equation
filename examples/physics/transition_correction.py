"""
CE: d=0 -> d=3 전이 보정 계산
================================
부트스트랩 궤적 eps2(D)를 연속 함수로 다루고,
C급 관측량(A_s, eta)의 갭을 전이 위치 tau_*로 정량적으로 좁힌다.

Lambert W를 뉴턴법으로 구현 (외부 의존성 없음).
"""
import math

def lambert_w0(z, tol=1e-15, maxiter=100):
    """Lambert W 함수 주가지 (W0). we^w = z의 해."""
    if z == 0:
        return 0.0
    if z > -1/math.e + 1e-10:
        w = math.log(1 + z) if z > 0 else z
    else:
        w = -1.0 + 1e-6
    for _ in range(maxiter):
        ew = math.exp(w)
        f = w * ew - z
        df = ew * (1 + w)
        if abs(df) < 1e-30:
            break
        dw = f / df
        w -= dw
        if abs(dw) < tol:
            break
    return w

# ================================================================
# 1. CE 기본 상수
# ================================================================
d = 3
alpha_total = 1.0 / (2.0 * math.pi)
alpha_s = 0.11789
sin2_tW = 4.0 * alpha_s**(4.0/3.0)
cos2_tW = 1.0 - sin2_tW
delta = sin2_tW * cos2_tW
D_eff = d + delta
N_gauge = 12

alpha_w = (alpha_total - alpha_s) / (1.0 + sin2_tW)
alpha_em = alpha_w * sin2_tW
J = 4.0 * alpha_s**(11.0/2.0)
g_star = 106.75

print("=" * 65)
print("CE 기본 상수")
print("=" * 65)
print(f"  alpha_s     = {alpha_s}")
print(f"  sin2_tW     = {sin2_tW:.6f}")
print(f"  delta       = {delta:.6f}")
print(f"  D_eff       = {D_eff:.6f}")
print(f"  alpha_w     = {alpha_w:.6f}")
print(f"  alpha_em    = {alpha_em:.6f}")
print(f"  J           = {J:.4e}")
print()

# ================================================================
# 2. 부트스트랩 궤적
# ================================================================
def eps2(D):
    if D <= 1e-12:
        return 1.0
    z = -D * math.exp(-D)
    w = lambert_w0(z)
    return -w / D

def deps2_dD(D):
    e2 = eps2(D)
    sigma = 1.0 - e2
    denom = 1.0 - e2 * D
    if abs(denom) < 1e-15:
        return 0.0
    return -e2 * sigma / denom

print("=" * 65)
print("부트스트랩 궤적 eps2(D)")
print("=" * 65)
print(f"  {'D':>8s}  {'eps2':>10s}  {'deps2/dD':>12s}  {'sigma':>8s}")
print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*8}")
for D_val in [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, D_eff]:
    e2 = eps2(D_val)
    de2 = deps2_dD(D_val)
    print(f"  {D_val:8.4f}  {e2:10.6f}  {de2:12.6f}  {1-e2:8.5f}")
print()

# ================================================================
# 3. A_s 계산: 고정점 vs 전이 보정
# ================================================================
print("=" * 65)
print("A_s 계산")
print("=" * 65)

A_s_obs = 2.1e-9

def compute_As(D_read, N_e):
    e2 = eps2(D_read)
    de2 = deps2_dD(D_read)
    sigma = 1.0 - e2
    As = (de2**2) / (sigma**2) * e2 / (2.0 * math.pi * N_e**2)
    return As

N_e_total = (d / 2.0) * D_eff * N_gauge

As_fixed = compute_As(D_eff, N_e_total)
print(f"[고정점] D = {D_eff:.5f}, N_e = {N_e_total:.1f}")
print(f"  A_s     = {As_fixed:.4e}")
print(f"  관측    = {A_s_obs:.1e}")
print(f"  비율    = {As_fixed/A_s_obs:.4f}  ({(1-As_fixed/A_s_obs)*100:+.1f}%)")
print()

print("--- Delta_N 스캔 (horizon exit 보정) ---")
print(f"  {'dN':>5s}  {'tau*':>6s}  {'D_read':>7s}  {'eps2':>8s}  {'de2/dD':>9s}  {'N_e':>5s}  {'A_s':>10s}  {'ratio':>6s}")

best_As_ratio = 999
for i in range(41):
    dN = i * 0.5
    tau_star = 1.0 - dN / N_e_total
    if tau_star < 0.7:
        break
    D_read = tau_star * D_eff
    N_e_eff = N_e_total - dN
    if N_e_eff < 20:
        break
    As_corr = compute_As(D_read, N_e_eff)
    ratio = As_corr / A_s_obs
    e2_r = eps2(D_read)
    de2_r = deps2_dD(D_read)
    if abs(ratio - 1.0) < abs(best_As_ratio - 1.0):
        best_As_ratio = ratio
        best_As_dN = dN
        best_As_val = As_corr
        best_As_tau = tau_star
    print(f"  {dN:5.1f}  {tau_star:6.4f}  {D_read:7.4f}  {e2_r:8.5f}  {de2_r:9.5f}  {N_e_eff:5.1f}  {As_corr:10.3e}  {ratio:6.3f}")

print()
print(f"*** A_s 최적 매칭 ***")
print(f"  Delta_N  = {best_As_dN:.1f} e-folds")
print(f"  tau*     = {best_As_tau:.4f}")
print(f"  A_s      = {best_As_val:.4e}")
print(f"  A_s/obs  = {best_As_ratio:.4f}  ({(best_As_ratio-1)*100:+.1f}%)")
print()

# ================================================================
# 4. eta 계산: 고정점 vs 전이 보정
# ================================================================
print("=" * 65)
print("eta (바리온-광자 비) 계산")
print("=" * 65)

eta_obs = 6.1e-10

def compute_eta(D_read):
    """D_read에서의 결합상수를 재계산하여 eta를 구한다."""
    delta_at_D = D_read - d
    if delta_at_D <= 0:
        return None, None, None
    disc = 1.0 - 4.0 * delta_at_D
    if disc < 0:
        return None, None, None
    sin2_at_D = (1.0 - math.sqrt(disc)) / 2.0
    alpha_s_at_D = (sin2_at_D / 4.0)**(3.0/4.0)
    alpha_w_at_D = (alpha_total - alpha_s_at_D) / (1.0 + sin2_at_D)
    J_at_D = 4.0 * alpha_s_at_D**(11.0/2.0)
    v_w = alpha_s_at_D
    eta = (405.0 * 25.0 * alpha_w_at_D**5) / (4.0 * math.pi**2 * g_star * v_w) * J_at_D / v_w
    return eta, alpha_w_at_D, alpha_s_at_D

eta_fixed, _, _ = compute_eta(D_eff)
print(f"[고정점] D = {D_eff:.5f}")
print(f"  eta     = {eta_fixed:.4e}")
print(f"  관측    = {eta_obs:.1e}")
print(f"  비율    = {eta_fixed/eta_obs:.4f}  ({(1-eta_fixed/eta_obs)*100:+.1f}%)")
print()

print("--- h 스캔 (D = D_eff - h) ---")
print(f"  {'h':>6s}  {'D_read':>7s}  {'alpha_s':>8s}  {'alpha_w':>8s}  {'eta':>10s}  {'ratio':>6s}")

best_eta_ratio = 999
for i in range(36):
    h = i * 0.005
    D_read = D_eff - h
    if D_read <= d:
        break
    eta_corr, aw_corr, as_corr = compute_eta(D_read)
    if eta_corr is None:
        continue
    ratio = eta_corr / eta_obs
    if abs(ratio - 1.0) < abs(best_eta_ratio - 1.0):
        best_eta_ratio = ratio
        best_eta_h = h
        best_eta_val = eta_corr
        best_eta_aw = aw_corr
        best_eta_as = as_corr
    print(f"  {h:6.3f}  {D_read:7.4f}  {as_corr:8.5f}  {aw_corr:8.5f}  {eta_corr:10.3e}  {ratio:6.3f}")

print()
print(f"*** eta 최적 매칭 ***")
print(f"  h        = {best_eta_h:.3f}")
print(f"  D_read   = {D_eff - best_eta_h:.4f}")
print(f"  alpha_s  = {best_eta_as:.5f}")
print(f"  alpha_w  = {best_eta_aw:.5f}")
print(f"  eta      = {best_eta_val:.4e}")
print(f"  eta/obs  = {best_eta_ratio:.4f}  ({(best_eta_ratio-1)*100:+.1f}%)")
print()

# ================================================================
# 5. 자기일관성 검사: 두 tau_*가 물리적으로 정합한가
# ================================================================
print("=" * 65)
print("자기일관성 검사")
print("=" * 65)

tau_As = best_As_tau
h_eta = best_eta_h
tau_eta = 1.0 - h_eta / D_eff

# A_s의 tau_*는 horizon exit (인플레이션 중)
# eta의 tau_*는 EWSB 상전이 (인플레이션 후)
# 시간 순서: tau_As < tau_eta < 1 이어야 한다

print(f"  tau*(A_s)  = {tau_As:.4f}  (인플레이션 horizon exit)")
print(f"  tau*(eta)  = {tau_eta:.4f}  (전자약 상전이)")
print(f"  tau 순서   : tau*(A_s) {'<' if tau_As < tau_eta else '>'} tau*(eta) {'<' if tau_eta < 1 else '>'} 1")
print()
if tau_As < tau_eta < 1.0:
    print("  순서 정합: 인플레이션(먼저) -> EWSB(나중) -> 고정점. OK")
else:
    print("  순서 부정합: 물리적 시간 순서와 맞지 않음.")
print()

# ================================================================
# 6. T_CMB 보정
# ================================================================
print("=" * 65)
print("T_CMB 전파 보정")
print("=" * 65)

alpha_val = 1.0 / 137.036
m_e = 0.511e-3  # GeV
E_I = alpha_val**2 * m_e / 2  # 13.6 eV in GeV
k_B = 8.617e-14  # GeV/K

def compute_T_CMB(eta_val):
    L = math.log(1.0 / eta_val) + 1.5 * math.log(m_e / (2 * math.pi * 0.3 * 1e-3))
    T_rec = (E_I / (L * k_B))
    z_rec = 1089
    return T_rec / (1 + z_rec)

T_fixed = compute_T_CMB(eta_fixed)
T_corr = compute_T_CMB(best_eta_val)
T_obs = 2.7255

print(f"  [고정점] eta = {eta_fixed:.3e}  =>  T_CMB = {T_fixed:.3f} K  (관측 {T_obs} K, 차이 {(T_fixed/T_obs-1)*100:+.1f}%)")
print(f"  [보정후] eta = {best_eta_val:.3e}  =>  T_CMB = {T_corr:.3f} K  (관측 {T_obs} K, 차이 {(T_corr/T_obs-1)*100:+.1f}%)")
print()

# ================================================================
# 7. 종합
# ================================================================
print("=" * 65)
print("종합: 전이 보정 전후 비교")
print("=" * 65)
print(f"  {'관측량':>10s}  {'고정점':>12s}  {'보정후':>12s}  {'관측':>12s}  {'고정점%':>8s}  {'보정후%':>8s}")
print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}")
print(f"  {'A_s':>10s}  {As_fixed:12.3e}  {best_As_val:12.3e}  {A_s_obs:12.1e}  {(As_fixed/A_s_obs-1)*100:+8.1f}%  {(best_As_val/A_s_obs-1)*100:+8.1f}%")
print(f"  {'eta':>10s}  {eta_fixed:12.3e}  {best_eta_val:12.3e}  {eta_obs:12.1e}  {(eta_fixed/eta_obs-1)*100:+8.1f}%  {(best_eta_val/eta_obs-1)*100:+8.1f}%")
print(f"  {'T_CMB':>10s}  {T_fixed:12.3f}  {T_corr:12.3f}  {T_obs:12.4f}  {(T_fixed/T_obs-1)*100:+8.1f}%  {(T_corr/T_obs-1)*100:+8.1f}%")

# ================================================================
# 8. NLO: DM/DE 분할
# ================================================================
print()
print("=" * 65)
print("NLO: DM/DE 분할")
print("=" * 65)

h_p = 0.6736
Omega_c_obs = 0.1200 / h_p**2
Omega_b_obs = 0.02237 / h_p**2
Omega_L_obs = 1 - Omega_b_obs - Omega_c_obs

R_LO = alpha_s * D_eff
R_NLO = alpha_s * D_eff + (alpha_s * D_eff)**2 / (4 * math.pi)

sigma = 1 - eps2(D_eff)
OL_LO = sigma / (1 + R_LO)
ODM_LO = sigma * R_LO / (1 + R_LO)
OL_NLO = sigma / (1 + R_NLO)
ODM_NLO = sigma * R_NLO / (1 + R_NLO)

print(f"  {'':>12s}  {'LO':>9s}  {'NLO':>9s}  {'관측':>9s}  {'LO%':>7s}  {'NLO%':>7s}")
print(f"  {'Omega_L':>12s}  {OL_LO:9.5f}  {OL_NLO:9.5f}  {Omega_L_obs:9.5f}  {(OL_LO/Omega_L_obs-1)*100:+7.2f}  {(OL_NLO/Omega_L_obs-1)*100:+7.2f}")
print(f"  {'Omega_DM':>12s}  {ODM_LO:9.5f}  {ODM_NLO:9.5f}  {Omega_c_obs:9.5f}  {(ODM_LO/Omega_c_obs-1)*100:+7.2f}  {(ODM_NLO/Omega_c_obs-1)*100:+7.2f}")

# ================================================================
# 9. CKM NLO 보정
# ================================================================
print()
print("=" * 65)
print("CKM NLO 보정")
print("=" * 65)

V_us_tree = sin2_tW
V_us_NLO = sin2_tW / (1 + alpha_s / (2 * math.pi))
V_us_obs = 0.22650

V_ub_tree = alpha_s**(8.0/3.0)
F = 1 + alpha_s * D_eff
V_ub_NLO = V_ub_tree * F**(1.0/3.0)
V_ub_obs = 0.00382

print(f"  {'':>10s}  {'Tree':>9s}  {'NLO':>9s}  {'관측':>9s}  {'Tree%':>8s}  {'NLO%':>8s}")
print(f"  {'|V_us|':>10s}  {V_us_tree:9.5f}  {V_us_NLO:9.5f}  {V_us_obs:9.5f}  {(V_us_tree/V_us_obs-1)*100:+8.2f}  {(V_us_NLO/V_us_obs-1)*100:+8.2f}")
print(f"  {'|V_ub|':>10s}  {V_ub_tree:9.5f}  {V_ub_NLO:9.5f}  {V_ub_obs:9.5f}  {(V_ub_tree/V_ub_obs-1)*100:+8.2f}  {(V_ub_NLO/V_ub_obs-1)*100:+8.2f}")
