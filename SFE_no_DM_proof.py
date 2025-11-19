
import numpy as np

# 1. 목표: "암흑물질 입자는 없다" (No CDM Hypothesis)
# 오직 SFE의 억압장(Phi)과 가변광속(VSL) 효과로만 관측 데이터 설명하기.

# 2. "은하 vs 은하단" 비율 역전 문제 해결 (Ratio Reversal Problem)
# 은하 (Small Scale): M_DM / M_b ~ 10
# 은하단 (Large Scale): M_DM / M_b ~ 50
# SFE의 무기: 
# (1) rho_Phi ~ rho_m^2 (비선형 응집) -> 이것만으로는 부족했음 (오차 98%)
# (2) VSL 중력 효과: 중력 결합 상수 G_eff 자체가 Scale-dependent 할 수 있음.
#     C(X) 비국소성 (22장): X = r / lambda. 
#     r이 커질수록(Large Scale), C(X)가 작아지나? 커지나?
#     23장 유도: lambda^4 ~ 1 / C(X). 
#     rho_Phi ~ lambda^2 * C(X) ~ sqrt(1/C(X)) * C(X) ~ sqrt(C(X)).
#     즉, C(X)가 크면 억압장 밀도(rho_Phi)도 커짐.
    
# C(X)의 특성 (22장):
# Small Scale (r << lambda): C(X) ~ Constant (국소 근사)
# Large Scale (r >> lambda): C(X)가 "경계 효과"로 인해 변함?
# 혹은 VSL 효과로 인해 lambda 자체가 변함?

# 신규 가설: VSL에 의한 유효 척도 변화 (Effective Scale Shift)
# c(Phi) = c0 / n(Phi).
# 은하단(저밀도) -> Phi 작음 -> n 작음 -> c 빠름.
# 은하(고밀도) -> Phi 큼 -> n 큼 -> c 느림.
# 중력파 속도 c_g = c 라고 가정하면,
# 중력 포텐셜 Phi_grav ~ G M / r * (Something related to c?)
# GR에서 G_eff는 c^4에 반비례하지 않음? (아인슈타인 상수 kappa = 8pi G / c^4)
# G_eff ~ c^4 이면... c가 빠르면 G_eff가 엄청 작아짐?
# -> 은하단(c 빠름)에서 G_eff 작아짐 -> 중력 약화? -> 암흑물질 설명에 불리.
# 반대로 kappa를 상수로 본다면? -> G ~ c^4. c 빠르면 G 엄청 큼.
# -> 은하단(c 빠름)에서 G 엄청 큼 -> 중력 50배 강화 가능!

def calculate_VSL_gravity_boost():
    # 가설: 아인슈타인 방정식의 결합 상수 kappa = 8pi G / c^4 가 불변 상수(Fundamental Constant)다.
    # 즉, G(Phi) = kappa * c(Phi)^4 / (8pi)
    # c(Phi)가 변하면 G도 변한다.
    
    # 1. 은하 (Galaxy) - 고밀도, Phi 큼
    # n_gal ~ 1 (현재 우주 평균과 유사하거나 약간 큼)
    # c_gal ~ c0
    # G_gal ~ G0
    
    # 2. 은하단 (Cluster) - 저밀도, Phi 작음 (Void에 가까운 영역 포함)
    # n_clus < 1 (억압장 밀도가 은하보다 낮으므로)
    # n_clus ~ 0.5 정도라고 가정해보자. (2배 빠름)
    # c_clus = 2 * c0
    # G_clus = G0 * (2)^4 = 16 * G0
    # -> 중력이 16배 강해짐.
    # -> 바리온 질량만으로도 16배 무거워 보이는 효과!
    # -> 50배 설명하려면? c가 약 2.66배 빠르면 됨 (2.66^4 ~ 50).
    
    # 3. 체크: 은하단 내부의 억압장 밀도 rho_Phi
    # 은하단은 은하들이 듬성듬성 있는 곳. 평균 밀도는 낮음.
    # 따라서 억압장 밀도도 낮음.
    # -> n(Phi)가 1보다 작아지는 게 물리적으로 타당함.
    # -> "저밀도 지역에서 중력이 더 강하게 작용한다"는 역설적 결론 도출.
    # -> 이것이 암흑물질(특히 저밀도 영역의 질량 결손)을 완벽히 설명함!
    
    # 4. S8 Tension 과의 충돌?
    # S8은 "Large Scale(저밀도)에서 덜 뭉쳐야 함".
    # 그런데 위 논리라면 G가 커져서 "더 잘 뭉쳐야 함".
    # -> 충돌 발생?
    # -> 아니, S8은 "현재(z=0)의 성장률". 현재 우주는 전체적으로 Phi가 큼(n~1).
    # -> 은하단 내부(Virialized)와 거대 구조 선형 영역(Linear)은 다름.
    # -> Virialized Cluster: 고온, 입자 속도 빠름. c 효과?
    # -> Linear Growth: 배경 우주 평균값.
    # -> 이 부분 정밀 분리가 필요.
    
    c_boost_factor = 2.7 # 은하단에서 빛이 2.7배 빠르다고 가정
    G_boost_factor = c_boost_factor**4
    
    needed_cluster_boost = 50.0
    error = abs(G_boost_factor - needed_cluster_boost) / needed_cluster_boost * 100
    
    return G_boost_factor, error

G_boost, err = calculate_VSL_gravity_boost()

print("=== 블록 1: 하드 VSL G-증폭 토이 ===")
print("가설: kappa = 8pi G / c^4 = const,  c_clus / c0 = 2.7")
print("-" * 50)
print(f"G_boost = (c_clus/c0)^4 = 2.7^4 = {G_boost:.1f}")
print("필요 G_boost(관측 DM 비율) ≈ 50.0")
print(f"상대 오차(%) = {err:.1f}")

def two_c_skeleton():
    """
    GR/양자 식의 c_* 는 불변으로 두고,
    관측 광속 c_gamma(Φ) = c_* / n(Φ) 만 억압장 매질에 따라 달라진다고 보는
    two-c 구조의 1차 스켈레톤.
    """
    c_star = 1.0  # 단위계에서 정규화
    n_mean = 1.0
    c_gamma_mean = c_star / n_mean

    n_gal = 1.05
    c_gamma_gal = c_star / n_gal

    n_clus = 0.90
    c_gamma_clus = c_star / n_clus

    return {
        "c_star": c_star,
        "n_mean": n_mean,
        "n_gal": n_gal,
        "n_clus": n_clus,
        "c_gamma_mean": c_gamma_mean,
        "c_gamma_gal": c_gamma_gal,
        "c_gamma_clus": c_gamma_clus,
    }

tc = two_c_skeleton()
print("\n=== 블록 2: two-c 스켈레톤 수치 ===")
print(f"c_* = {tc['c_star']:.2f}")
print(f"n_mean = {tc['n_mean']:.2f},   c_gamma_mean / c_* = {tc['c_gamma_mean']:.2f}")
print(f"n_gal  = {tc['n_gal']:.2f}, c_gamma_gal  / c_* = {tc['c_gamma_gal']:.3f}")
print(f"n_clus = {tc['n_clus']:.2f}, c_gamma_clus / c_* = {tc['c_gamma_clus']:.3f}")

H0_km_s_Mpc = 67.4
Omega_m = 0.315
Omega_Phi = 0.675
c_si = 2.99792458e8
G_si = 6.67430e-11
H0_si = H0_km_s_Mpc * 1000.0 / 3.085677581e22
rho_c = 3.0 * H0_si**2 / (8.0 * np.pi * G_si)
rho_m_bar = Omega_m * rho_c
rho_Phi_bar = Omega_Phi * rho_c
ratio_Phi_to_m = rho_Phi_bar / rho_m_bar

delta_gal = 1e8
delta_clus = 1e4
rho_b_gal = delta_gal * rho_m_bar
rho_b_clus = delta_clus * rho_m_bar

R_gal_theory = 1.0 + ratio_Phi_to_m * delta_gal
R_clus_theory = 1.0 + ratio_Phi_to_m * delta_clus

print("\n=== 블록 3: SFE 코어 식에서 직접 유도한 R(δ) ===")
print(f"rho_c = {rho_c:.3e} kg/m^3")
print(f"rho_m_bar = {rho_m_bar:.3e} kg/m^3")
print(f"rho_Phi_bar = {rho_Phi_bar:.3e} kg/m^3")
print(f"rho_Phi_bar / rho_m_bar = {ratio_Phi_to_m:.3f}")
print(f"delta_gal = {delta_gal:.1e}, delta_clus = {delta_clus:.1e}")
print(f"R_gal_theory  = {R_gal_theory:.3e}")
print(f"R_clus_theory = {R_clus_theory:.3e}")


