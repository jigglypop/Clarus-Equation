#!/usr/bin/env python3
"""
SFE XGBoost 최적화 - Bayesian Optimization + Tree-based
"""
import numpy as np
import math
from tqdm import tqdm

# 기존 클래스 import
import sys
sys.path.append('.')
from SFE_global_error import (
    CosmologyParams, SFECoreParams, DMParams, ZPrimeParams, GlobalParams
)

# 데이터 정의
cosmo = CosmologyParams(H0=70.0, Omega_m0=0.3, Omega_r0=1e-5, Omega_L0=0.7)
dm = DMParams(Omega_dm_resid=0.1, m_X_eV=1e-22, sigma_over_m=0.1)
k_eff = 0.05

growth_data = [
    {'z': 0.57, 'fsigma8': 0.453, 'err': 0.05},
    {'z': 0.44, 'fsigma8': 0.413, 'err': 0.08},
    {'z': 0.60, 'fsigma8': 0.433, 'err': 0.067},
    {'z': 0.73, 'fsigma8': 0.437, 'err': 0.072},
]

micro_data = [
    {'name': 'C60_decoherence', 'obs': 1.0, 'err': 0.1},
    {'name': 'LIGO_thermal', 'obs': 1.0, 'err': 0.15},
    {'name': 'cosmic_muon_lifetime', 'obs': 1.0, 'err': 0.05},
]

muon_data = {'Delta_a_mu': 2.51e-9, 'err': 0.59e-9}

def micro_error(micro_data, eps_mass):
    chi2 = 0.0
    for obs in micro_data:
        pred = 1.0 + 0.1 * (eps_mass - 0.37)
        chi2 += ((pred - obs['obs']) / obs['err'])**2
    return chi2

def muon_g2_error(muon_data, g_mu, m_Zp_GeV):
    Delta_a_mu_pred = (g_mu**2) / (8 * math.pi**2 * m_Zp_GeV**2) * 1e9
    Delta_a_mu_obs = muon_data['Delta_a_mu']
    err = muon_data['err']
    return ((Delta_a_mu_pred - Delta_a_mu_obs) / err)**2

# 간소화된 성장률 계산
def solve_growth_D(a_arr, Om_a_arr, eps_grav_arr):
    n = len(a_arr)
    D = np.ones(n)
    f = np.ones(n)
    
    for i in range(1, n):
        da = a_arr[i] - a_arr[i-1]
        a = a_arr[i-1]
        Om = Om_a_arr[i-1]
        eps_g = eps_grav_arr[i-1]
        
        dD = f[i-1]
        df = -(3/a + (1/a) * (1.5*Om/(1-eps_g) - 1)) * f[i-1] + 1.5*Om/(1-eps_g) * D[i-1] / a**2
        
        D[i] = D[i-1] + da * dD
        f[i] = f[i-1] + da * df
    
    return D, f

def f_sigma8_SFE(z_arr, params, eps_grav_val):
    a_arr = np.array(1.0 / (1.0 + z_arr))
    Om_a_arr = np.array([
        params.cosmo.Omega_m0 * (1+z)**3 / 
        (params.cosmo.Omega_m0 * (1+z)**3 + (1 - params.cosmo.Omega_m0))
        for z in z_arr
    ])
    eps_grav_arr = np.full_like(a_arr, eps_grav_val)
    
    D, f = solve_growth_D(a_arr, Om_a_arr, eps_grav_arr)
    
    sigma8_0 = getattr(params.cosmo, 'sigma8_0', 0.8)
    sigma8_z = sigma8_0 * D / D[-1]
    f_growth = f * a_arr / D
    
    return float(f_growth[0]) * float(sigma8_z[0])

def fit_sigma8_for_growth(growth_data, params, eps_grav_val):
    z_arr = np.linspace(0.0, 2.0, 80)
    
    def chi2(sigma8_0):
        params.cosmo.sigma8_0 = sigma8_0
        chi2_sum = 0.0
        for obs in growth_data:
            z_obs = obs['z']
            fsig8_obs = obs['fsigma8']
            err = obs['err']
            
            fsig8_pred = f_sigma8_SFE(
                np.linspace(z_obs, 2.0, 80), params, eps_grav_val
            )
            chi2_sum += ((fsig8_pred - fsig8_obs) / err)**2
        return chi2_sum
    
    sigma8_range = np.linspace(0.7, 0.9, 8)
    chi2_vals = [chi2(s8) for s8 in sigma8_range]
    best_idx = np.argmin(chi2_vals)
    
    return sigma8_range[best_idx], chi2_vals[best_idx]

def eval_error(params_dict):
    """파라미터 -> E_total 계산"""
    eps_mass = params_dict['eps_mass']
    eps_0 = params_dict['eps_0']
    transition_a = params_dict['transition_a']
    sharpness = params_dict['sharpness']
    k_star = params_dict['k_star']
    rho_screen = params_dict['rho_screen']
    g_mu = params_dict['g_mu']
    m_Zp_GeV = params_dict['m_Zp_GeV']
    
    a_ref = 0.5
    eps_grav_a = eps_0 / (1.0 + math.exp(-sharpness * (a_ref - transition_a)))
    
    sfe = SFECoreParams(epsilon_mass=eps_mass, epsilon_grav=eps_0)
    zprime = ZPrimeParams(g_mu=g_mu, m_Zp_GeV=m_Zp_GeV)
    params = GlobalParams(cosmo=cosmo, sfe=sfe, dm=dm, zprime=zprime)
    
    try:
        _, E_growth = fit_sigma8_for_growth(growth_data, params, eps_grav_a)
        E_micro = micro_error(micro_data, eps_mass)
        E_mu = muon_g2_error(muon_data, g_mu, m_Zp_GeV)
        return E_growth + E_micro + E_mu
    except:
        return 1e10

# Bayesian Optimization with Tree Parzen Estimator
def tpe_optimize(n_iter=2000, target_E=5.0):
    """
    Tree-structured Parzen Estimator (TPE) 최적화
    """
    print("\n" + "="*70)
    print(f"TPE 베이지안 최적화 - {n_iter}회")
    print("="*70)
    
    # 이전 최적값 (E=9.93)에서 시작
    best_params = {
        'eps_mass': 0.355,
        'eps_0': 0.580,
        'transition_a': 0.520,
        'sharpness': 12.22,
        'k_star': 0.436,
        'rho_screen': 130.2,
        'g_mu': 3.36e-4,
        'm_Zp_GeV': 0.067,
    }
    
    best_E = eval_error(best_params)
    
    print(f"\n시작값: E_total = {best_E:.4f}")
    print(f"목표: E_total < {target_E}")
    
    # 탐색 이력 저장
    history = []
    history.append((best_params.copy(), best_E))
    
    pbar = tqdm(range(n_iter), desc="TPE 최적화", unit="iter")
    pbar.set_postfix(E_best=f"{best_E:.4f}", E_current="N/A")
    
    for it in pbar:
        # TPE: 좋은 샘플과 나쁜 샘플을 분리
        sorted_history = sorted(history, key=lambda x: x[1])
        n_good = max(1, len(sorted_history) // 5)  # 상위 20%
        good_samples = sorted_history[:n_good]
        
        # 좋은 샘플 주변에서 탐색 (80%), 랜덤 탐색 (20%)
        if np.random.rand() < 0.8 and len(good_samples) > 0:
            # 좋은 샘플 중 하나 선택
            base_params, _ = good_samples[np.random.randint(len(good_samples))]
            
            # Gaussian noise 추가
            new_params = {}
            for key in base_params.keys():
                if key == 'eps_0':
                    noise = np.random.normal(0, 0.1)
                    new_params[key] = np.clip(base_params[key] + noise, 0, 1)
                elif key == 'transition_a':
                    noise = np.random.normal(0, 0.1)
                    new_params[key] = np.clip(base_params[key] + noise, 0.1, 0.9)
                elif key == 'eps_mass':
                    noise = np.random.normal(0, 0.05)
                    new_params[key] = np.clip(base_params[key] + noise, 0.1, 1.0)
                elif key == 'sharpness':
                    noise = np.random.normal(0, 0.5)
                    new_params[key] = max(0.1, base_params[key] + noise)
                elif key == 'k_star':
                    noise = np.random.normal(0, 0.005)
                    new_params[key] = max(0.001, base_params[key] + noise)
                elif key == 'rho_screen':
                    noise = np.random.normal(0, base_params[key] * 0.2)
                    new_params[key] = max(100, base_params[key] + noise)
                elif key == 'g_mu':
                    noise = np.random.normal(0, base_params[key] * 0.3)
                    new_params[key] = max(1e-6, base_params[key] + noise)
                elif key == 'm_Zp_GeV':
                    noise = np.random.normal(0, base_params[key] * 0.3)
                    new_params[key] = max(1e-4, base_params[key] + noise)
        else:
            # 랜덤 탐색
            new_params = {
                'eps_mass': np.random.uniform(0.2, 0.6),
                'eps_0': np.random.uniform(0.0, 1.0),
                'transition_a': np.random.uniform(0.1, 0.9),
                'sharpness': np.random.uniform(0.5, 10.0),
                'k_star': np.random.uniform(0.005, 0.05),
                'rho_screen': 10**np.random.uniform(2, 4),
                'g_mu': 10**np.random.uniform(-5, -3),
                'm_Zp_GeV': 10**np.random.uniform(-3, -1),
            }
        
        # 오차 계산
        E_total = eval_error(new_params)
        pbar.set_postfix(E_best=f"{best_E:.4f}", E_current=f"{E_total:.4f}")
        
        # 이력에 추가
        history.append((new_params.copy(), E_total))
        
        # 최적값 업데이트
        if E_total < best_E:
            improvement = (best_E - E_total) / best_E * 100
            best_E = E_total
            best_params = new_params.copy()
            tqdm.write(f">>> [개선!] iter {it:5d}: E_total={E_total:.4f} (↓{improvement:.2f}%)")
        
        if best_E <= target_E:
            tqdm.write(f"\n목표 달성: E_total <= {target_E}")
            break
    
    pbar.close()
    
    print(f"\n{'='*70}")
    print("최종 최적 파라미터:")
    print(f"{'='*70}")
    for key, val in best_params.items():
        print(f"  {key:15s} = {val:.6e}")
    print(f"\n최종 E_total = {best_E:.4f}")
    
    return best_params

if __name__ == "__main__":
    result = tpe_optimize(n_iter=2000, target_E=5.0)

