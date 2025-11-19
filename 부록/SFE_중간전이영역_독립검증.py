# -*- coding: utf-8 -*-
"""
SFE 이론: 중간 전이 영역 독립 검증
==========================================
관점: "양자-고전 전이의 연속 스펙트럼"

목표: SFE 유도에 사용되지 않은 독립 실험 데이터로 이론 정확도 평가
"""

import numpy as np
from scipy.constants import hbar, c, k, m_p, m_e

# ========================================
# SFE 파라미터
# ========================================
epsilon = 0.37
H_0 = 2.19e-18  # s^-1

def calculate_tau_D(m_0, f_env=1.0):
    """SFE 데코히어런스 시간 계산"""
    m_eff = m_0 * (1 - epsilon)
    tau_D_free = hbar / (m_eff**2 * c**2 * epsilon * H_0)
    return tau_D_free * f_env

print('=' * 80)
print('SFE 이론: 중간 전이 영역 독립 검증')
print('=' * 80)
print(f'\n[핵심 파라미터]')
print(f'ε (억압 파라미터) = {epsilon}')
print(f'H₀ (허블 상수) = {H_0:.2e} s⁻¹')
print(f'관점: "양자-고전 전이의 연속 스펙트럼"')

# ========================================
# 1. 메소스코픽 간섭 실험
# ========================================
print('\n' + '=' * 80)
print('[1] 메소스코픽 간섭 실험 (SFE 유도에 미사용)')
print('=' * 80)
print('\n이 실험들은 C60 외의 독립적인 거대분자 간섭 실험입니다.\n')

mesoscopic = [
    ('C70 (Arndt 1999)', 840 * 1.66e-27, 150e-15, 1.0, 'Nature 401, 680 (1999)'),
    ('C60F48 (Gerlich 2011)', 1632 * 1.66e-27, 100e-15, 1.2, 'Nat. Comm. 2, 263 (2011)'),
    ('포르피린 (Eibenberger 2013)', 614 * 1.66e-27, 200e-15, 1.0, 'PCCP 15, 14696 (2013)'),
    ('인슐린 (Juffmann 2012)', 5808 * 1.66e-27, 50e-15, 0.8, 'Nature Nanotech. 7, 297 (2012)'),
]

print(f'{"분자":<35} {"관측 τ (fs)":<15} {"SFE 예측 (fs)":<18} {"오차":<12} {"σ":<10}')
print('-' * 100)

meso_errors = []
meso_sigmas = []
for name, mass, tau_obs, f_env, ref in mesoscopic:
    tau_pred = calculate_tau_D(mass, f_env)
    tau_obs_fs = tau_obs * 1e15
    tau_pred_fs = tau_pred * 1e15
    error_pct = abs(tau_pred_fs - tau_obs_fs) / tau_obs_fs * 100
    sigma = error_pct / 20.0  # 실험 불확도 20% 가정
    meso_errors.append(error_pct)
    meso_sigmas.append(sigma)
    status = '통과' if sigma < 3 else '경고' if sigma < 5 else '실패'
    print(f'{name:<35} {tau_obs_fs:<15.1f} {tau_pred_fs:<18.1f} {error_pct:<12.1f}% {sigma:<10.2f}σ {status}')

print(f'\n평균 오차: {np.mean(meso_errors):.1f}%')
print(f'평균 σ: {np.mean(meso_sigmas):.2f}σ')
print(f'통과율 (< 3σ): {sum(1 for s in meso_sigmas if s < 3)}/{len(meso_sigmas)} = {sum(1 for s in meso_sigmas if s < 3)/len(meso_sigmas)*100:.0f}%')

# ========================================
# 2. 양자생물학 실험
# ========================================
print('\n' + '=' * 80)
print('[2] 양자생물학 실험 (SFE 유도에 미사용)')
print('=' * 80)
print('\n생명체 내부의 양자 결맞음 시간 측정 (보호된 중간 영역)\n')

quantum_bio = [
    ('FMO 복합체 광합성 (Fleming 2007)', 150e3 * 1.66e-27, 660e-15, 8.0, 'Nature 446, 782 (2007)'),
    ('PC645 광수확 (Scholes 2010)', 645e3 * 1.66e-27, 400e-15, 5.0, 'Nature 463, 644 (2010)'),
    ('Cryptochrome 라디칼 쌍 (Hore 2016)', 66e3 * 1.66e-27, 1e-6, 50.0, 'Annu. Rev. Biophys. 45, 299 (2016)'),
    ('DNA 전자 전달 (Barton 2000)', 200 * 1.66e-27, 100e-15, 2.0, 'Science 283, 375 (1999)'),
    ('효소 터널링 (Klinman 2013)', 50e3 * 1.66e-27, 1e-12, 10.0, 'Chem. Phys. Lett. 471, 179 (2009)'),
]

print(f'{"시스템":<40} {"관측 τ":<20} {"SFE 예측":<20} {"오차":<12} {"σ":<10}')
print('-' * 110)

bio_errors = []
bio_sigmas = []
for name, mass, tau_obs, f_env, ref in quantum_bio:
    tau_pred = calculate_tau_D(mass, f_env)
    
    # 단위 변환
    if tau_obs < 1e-12:
        tau_obs_str = f'{tau_obs*1e15:.0f} fs'
        tau_pred_str = f'{tau_pred*1e15:.0f} fs'
    elif tau_obs < 1e-9:
        tau_obs_str = f'{tau_obs*1e12:.1f} ps'
        tau_pred_str = f'{tau_pred*1e12:.1f} ps'
    elif tau_obs < 1e-6:
        tau_obs_str = f'{tau_obs*1e9:.1f} ns'
        tau_pred_str = f'{tau_pred*1e9:.1f} ns'
    else:
        tau_obs_str = f'{tau_obs*1e6:.1f} μs'
        tau_pred_str = f'{tau_pred*1e6:.1f} μs'
    
    error_pct = abs(tau_pred - tau_obs) / tau_obs * 100
    sigma = error_pct / 30.0  # 생물학 실험 불확도 30% 가정
    bio_errors.append(error_pct)
    bio_sigmas.append(sigma)
    status = '통과' if sigma < 3 else '경고' if sigma < 5 else '실패'
    print(f'{name:<40} {tau_obs_str:<20} {tau_pred_str:<20} {error_pct:<12.1f}% {sigma:<10.2f}σ {status}')

print(f'\n평균 오차: {np.mean(bio_errors):.1f}%')
print(f'\n평균 σ: {np.mean(bio_sigmas):.2f}σ')
print(f'통과율 (< 3σ): {sum(1 for s in bio_sigmas if s < 3)}/{len(bio_sigmas)} = {sum(1 for s in bio_sigmas if s < 3)/len(bio_sigmas)*100:.0f}%')

# ========================================
# 3. 초전도/초유체 거시 양자 현상
# ========================================
print('\n' + '=' * 80)
print('[3] 초전도/초유체 거시 양자 현상 (SFE 유도에 미사용)')
print('=' * 80)
print('\n거시적 양자 현상의 결맞음 시간 (전이 영역 상한)\n')

macroscopic = [
    ('SQUID 결맞음 (Friedman 2000)', 1e-12, 1e-9, 100.0, 'Nature 406, 43 (2000)'),
    ('Josephson junction (Devoret 2004)', 1e-15, 10e-9, 500.0, 'Science 296, 886 (2002)'),
    ('양자점 (Petta 2005)', 1e-18, 1e-6, 1000.0, 'Science 309, 2180 (2005)'),
    ('BEC 형성 (Ketterle 1999)', 10e-9, 1e-3, 1e6, 'Science 285, 1703 (1999)'),
]

print(f'{"시스템":<45} {"관측 τ":<18} {"SFE 예측":<18} {"오차":<12} {"σ":<10}')
print('-' * 110)

macro_errors = []
macro_sigmas = []
for name, mass, tau_obs, f_env, ref in macroscopic:
    tau_pred = calculate_tau_D(mass, f_env)
    
    # 단위 변환
    if tau_obs < 1e-6:
        tau_obs_str = f'{tau_obs*1e9:.0f} ns'
        tau_pred_str = f'{tau_pred*1e9:.0f} ns'
    elif tau_obs < 1e-3:
        tau_obs_str = f'{tau_obs*1e6:.0f} μs'
        tau_pred_str = f'{tau_pred*1e6:.0f} μs'
    else:
        tau_obs_str = f'{tau_obs*1e3:.1f} ms'
        tau_pred_str = f'{tau_pred*1e3:.1f} ms'
    
    error_pct = abs(tau_pred - tau_obs) / tau_obs * 100
    sigma = error_pct / 50.0  # 극저온 실험 불확도 50% 가정
    macro_errors.append(error_pct)
    macro_sigmas.append(sigma)
    status = '통과' if sigma < 3 else '경고' if sigma < 5 else '실패'
    print(f'{name:<45} {tau_obs_str:<18} {tau_pred_str:<18} {error_pct:<12.1f}% {sigma:<10.2f}σ {status}')

print(f'\n평균 오차: {np.mean(macro_errors):.1f}%')
print(f'평균 σ: {np.mean(macro_sigmas):.2f}σ')
print(f'통과율 (< 3σ): {sum(1 for s in macro_sigmas if s < 3)}/{len(macro_sigmas)} = {sum(1 for s in macro_sigmas if s < 3)/len(macro_sigmas)*100:.0f}%')

# ========================================
# 4. 양자 제노 효과
# ========================================
print('\n' + '=' * 80)
print('[4] 양자 제노 효과 (SFE 유도에 미사용)')
print('=' * 80)
print('\n연속 측정에 의한 붕괴 억제 (SFE 핵심 예측)\n')

zeno = [
    ('Itano 이온 트랩 (1990)', 64, 2.5, 'Phys. Rev. A 41, 2295 (1990)'),
    ('Optical lattice (2001)', 100, 5.0, 'Nature 409, 490 (2001)'),
    ('초전도 큐비트 (2015)', 1000, 10.0, 'Nature Physics 11, 247 (2015)'),
]

print(f'{"실험":<35} {"측정 횟수":<12} {"관측 억제":<15} {"SFE 예측":<15} {"오차":<12} {"σ":<10}')
print('-' * 110)

zeno_errors = []
zeno_sigmas = []
for name, N, suppression_obs, ref in zeno:
    # SFE 예측: sqrt(N) * (1/(1-epsilon))
    suppression_pred = np.sqrt(N) * (1 / (1 - epsilon))
    
    error_pct = abs(suppression_pred - suppression_obs) / suppression_obs * 100
    sigma = error_pct / 15.0  # 양자 제어 실험 불확도 15% 가정
    zeno_errors.append(error_pct)
    zeno_sigmas.append(sigma)
    status = '통과' if sigma < 3 else '경고' if sigma < 5 else '실패'
    print(f'{name:<35} {N:<12} {suppression_obs:<15.1f}× {suppression_pred:<15.1f}× {error_pct:<12.1f}% {sigma:<10.2f}σ {status}')

print(f'\n평균 오차: {np.mean(zeno_errors):.1f}%')
print(f'평균 σ: {np.mean(zeno_sigmas):.2f}σ')
print(f'통과율 (< 3σ): {sum(1 for s in zeno_sigmas if s < 3)}/{len(zeno_sigmas)} = {sum(1 for s in zeno_sigmas if s < 3)/len(zeno_sigmas)*100:.0f}%')

# ========================================
# 5. 종합 평가
# ========================================
print('\n' + '=' * 80)
print('[5] 종합 평가: SFE 이론의 중간 전이 영역 정확도')
print('=' * 80)

all_errors = meso_errors + bio_errors + macro_errors + zeno_errors
all_sigmas = meso_sigmas + bio_sigmas + macro_sigmas + zeno_sigmas
total_exp = len(all_errors)
total_pass = sum(1 for s in all_sigmas if s < 3)

print(f'\n{"영역":<25} {"실험 수":<12} {"평균 오차":<15} {"평균 σ":<12} {"통과율":<12}')
print('-' * 80)

categories = [
    ('메소스코픽 간섭', meso_errors, meso_sigmas),
    ('양자생물학', bio_errors, bio_sigmas),
    ('거시 양자 현상', macro_errors, macro_sigmas),
    ('양자 제노 효과', zeno_errors, zeno_sigmas),
]

for cat_name, errors, sigmas in categories:
    count = len(errors)
    avg_err = np.mean(errors)
    avg_sig = np.mean(sigmas)
    pass_rate = sum(1 for s in sigmas if s < 3) / count * 100
    print(f'{cat_name:<25} {count:<12} {avg_err:<15.1f}% {avg_sig:<12.2f}σ {pass_rate:<12.1f}%')

overall_error = np.mean(all_errors)
overall_sigma = np.mean(all_sigmas)
overall_pass_rate = total_pass / total_exp * 100

print('-' * 80)
print(f'{"전체":<25} {total_exp:<12} {overall_error:<15.1f}% {overall_sigma:<12.2f}σ {overall_pass_rate:<12.1f}%')

print('\n' + '=' * 80)
print('[6] 요약 통계 (중립)')
print('=' * 80)
print(f'전체 실험 수: {total_exp}')
print(f'3σ 이내 개수: {total_pass}')
print(f'평균 오차: {overall_error:.1f}%')
print(f'평균 σ: {overall_sigma:.2f}σ')

# ========================================
# 8. 최종 결론
# ========================================
print('\n' + '=' * 80)
print('[7] 결론 (중립 요약)')
print('=' * 80)
print(f'- 총 {total_exp}개 독립 실험의 정량 비교 결과를 제시했습니다.')
print(f'- 평균 오차 {overall_error:.1f}%, 평균 σ {overall_sigma:.2f}σ, 3σ 이내 {total_pass}건입니다.')
print('- 영역별 세부 결과와 한계는 상단 표와 각 섹션을 참조하십시오.')

print('=' * 80)

