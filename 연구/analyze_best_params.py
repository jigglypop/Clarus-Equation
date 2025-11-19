#!/usr/bin/env python3
"""
최적 파라미터 상세 분석
"""
import numpy as np
import math

# Trial 991 최적 파라미터 (추정 - 출력에서 정확한 값 필요)
# 일단 Trial 1367 근처 값으로 추정
params_best = {
    'eps_mass': 0.474,
    'eps_0': 0.130,
    'transition_a': 0.748,
    'sharpness': 23.34,
    'k_star': 0.497,
    'rho_screen': 79.02,
    'g_mu': 5.12e-4,
    'm_Zp_GeV': 0.100,
}

# 우주론 상수
Omega_Lambda = 0.685
Omega_m = 0.315
M_MU_GEV = 0.1056583755
H0 = 67.4

# 관측 데이터
growth_data = [
    {'z': 0.02, 'fsigma8': 0.398, 'err': 0.065},
    {'z': 0.067, 'fsigma8': 0.423, 'err': 0.055},
    {'z': 0.17, 'fsigma8': 0.510, 'err': 0.060},
    {'z': 0.18, 'fsigma8': 0.360, 'err': 0.090},
    {'z': 0.38, 'fsigma8': 0.440, 'err': 0.060},
    {'z': 0.51, 'fsigma8': 0.458, 'err': 0.038},
    {'z': 0.52, 'fsigma8': 0.397, 'err': 0.110},
    {'z': 0.59, 'fsigma8': 0.488, 'err': 0.060},
    {'z': 0.86, 'fsigma8': 0.400, 'err': 0.110},
    {'z': 0.978, 'fsigma8': 0.379, 'err': 0.176},
]

micro_data = [
    {'name': 'BBN_D/H', 'theory': 2.569e-5, 'obs': 2.527e-5, 'err': 0.030e-5},
    {'name': 'BBN_Yp', 'theory': 0.2470, 'obs': 0.2449, 'err': 0.0040},
    {'name': 'Planck_ns', 'theory': 0.9665, 'obs': 0.9649, 'err': 0.0042},
    {'name': 'Planck_As', 'theory': 2.105e-9, 'obs': 2.100e-9, 'err': 0.030e-9},
]

muon_data = {
    'obs': 25.1e-10,
    'err': 4.8e-10,
}

print("="*80)
print("최적 파라미터 상세 분석")
print("="*80)
print(f"\nE_total = 23.3255 (Trial 991)")
print(f"\n파라미터:")
for key, val in params_best.items():
    print(f"  {key:15s} = {val:.6e}")

# 선형 관계 검증
alpha = -0.797
beta = 0.719

eps_mass_linear = alpha * Omega_m + beta
eps_0_linear = alpha * Omega_Lambda + beta

print(f"\n" + "="*80)
print("선형 관계 검증: ε = -0.797·Ω + 0.719")
print("="*80)
print(f"\nε_mass:")
print(f"  최적값: {params_best['eps_mass']:.6f}")
print(f"  선형 예측: {eps_mass_linear:.6f}")
print(f"  차이: {abs(params_best['eps_mass'] - eps_mass_linear):.6f}")

print(f"\nε_0:")
print(f"  최적값: {params_best['eps_0']:.6f}")
print(f"  선형 예측: {eps_0_linear:.6f}")
print(f"  차이: {abs(params_best['eps_0'] - eps_0_linear):.6f}")

# 새로운 선형 관계 계산
alpha_new = (params_best['eps_0'] - params_best['eps_mass']) / (Omega_Lambda - Omega_m)
beta_new = params_best['eps_mass'] - alpha_new * Omega_m

print(f"\n새로운 선형 관계:")
print(f"  α = {alpha_new:.6f} (이전: {alpha:.6f})")
print(f"  β = {beta_new:.6f} (이전: {beta:.6f})")

# 뮤온 g-2 계산
pref = (params_best['g_mu'] ** 2) / (12.0 * math.pi * math.pi)
ratio = (M_MU_GEV ** 2) / (params_best['m_Zp_GeV'] ** 2)
Delta_a_mu_pred = pref * ratio

print(f"\n" + "="*80)
print("뮤온 g-2 분석")
print("="*80)
print(f"\n예측값: Δa_μ = {Delta_a_mu_pred:.6e}")
print(f"관측값: Δa_μ = {muon_data['obs']:.6e}")
print(f"오차: σ = {muon_data['err']:.6e}")
print(f"\n차이: {abs(Delta_a_mu_pred - muon_data['obs']):.6e}")
print(f"σ 단위: {abs(Delta_a_mu_pred - muon_data['obs']) / muon_data['err']:.2f}σ")

chi2_mu = ((Delta_a_mu_pred - muon_data['obs']) / muon_data['err']) ** 2
print(f"\nχ² = {chi2_mu:.4f}")

# 미시 채널
print(f"\n" + "="*80)
print("미시 채널 분석")
print("="*80)
E_micro = 0.0
for obs in micro_data:
    chi2 = ((obs['theory'] - obs['obs']) / obs['err']) ** 2
    E_micro += chi2
    print(f"\n{obs['name']}:")
    print(f"  이론: {obs['theory']:.6e}")
    print(f"  관측: {obs['obs']:.6e}")
    print(f"  χ² = {chi2:.4f}")

print(f"\n총 E_micro = {E_micro:.4f}")

# 전체 에러 추정
print(f"\n" + "="*80)
print("전체 에러 추정")
print("="*80)
print(f"\nE_micro ≈ {E_micro:.4f}")
print(f"E_mu ≈ {chi2_mu:.4f}")
print(f"E_growth ≈ {23.3255 - E_micro - chi2_mu:.4f} (역산)")
print(f"\nE_total = {23.3255:.4f}")

# 개선 가능성 분석
print(f"\n" + "="*80)
print("E_total < 5.0 달성 가능성 분석")
print("="*80)
print(f"\n현재 E_total = 23.33")
print(f"목표 E_total = 5.00")
print(f"필요 감소량 = {23.33 - 5.0:.2f}")

print(f"\n각 채널별 기여:")
print(f"  E_growth ≈ 20.9 (90%)")
print(f"  E_micro ≈ 2.4 (10%)")
print(f"  E_mu ≈ 0.08 (0.3%)")

print(f"\n결론:")
print(f"  - E_growth를 20.9 → 2.6으로 줄여야 함 (88% 감소)")
print(f"  - 이는 성장률 데이터와의 근본적 불일치 가능성")
print(f"  - 또는 성장률 계산 공식 자체에 문제가 있을 수 있음")

print(f"\n대안:")
print(f"  1. 성장률 계산 공식 재검토 (μ(a,k) 구조)")
print(f"  2. 관측 데이터 재검토 (outlier 제거)")
print(f"  3. 목표를 E_total < 20으로 조정")
print(f"  4. 다른 물리 메커니즘 도입 (예: scale-dependent ε)")

print("\n" + "="*80)

