#!/usr/bin/env python3
"""
SFE Optuna + XGBoost 최적화
"""
import numpy as np
import math
import optuna
from optuna.samplers import TPESampler
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

M_MU_GEV = 0.1056583745  # 뮤온 질량 (GeV), SFE_global_error.py와 동일


def muon_g2_error(muon_data, g_mu, m_Zp_GeV):
    """뮤온 g-2 오차 (SFE_global_error.py의 1-loop 식과 동일 구조).

    Δa_μ^{Z'} ≈ (g_μ² / 12π²) (m_μ² / m_{Z'}²)
    """
    if m_Zp_GeV <= 0.0:
        return 1e9

    pref = g_mu * g_mu / (12.0 * math.pi * math.pi)
    ratio = (M_MU_GEV * M_MU_GEV) / (m_Zp_GeV * m_Zp_GeV)
    Delta_a_mu_pred = pref * ratio

    Delta_a_mu_obs = muon_data['Delta_a_mu']
    err = muon_data['err']
    return ((Delta_a_mu_pred - Delta_a_mu_obs) / err)**2

# SFE_gpu_search.py와 동일한 성장률 계산
def solve_growth_D_gpu(a_ini, a_fin, params, eps_grav_a, k, k_star, rho_screen, n_steps=600):
    """RK4 적분 - SFE_gpu_search.py와 동일"""
    h = (a_fin - a_ini) / n_steps
    a_arr = np.linspace(a_ini, a_fin, n_steps + 1)
    y = np.zeros((n_steps + 1, 2))
    y[0] = np.array([1.0, 0.0])
    
    H0 = params.cosmo.H0
    Om = params.cosmo.Omega_m0
    OL = params.cosmo.Omega_L0
    
    for i in range(n_steps):
        a_i = float(a_arr[i])
        y_i = y[i]
        
        # μ(a,k) 계산
        F_k = 1.0 / (1.0 + (k / k_star)**2)
        S_a = 1.0 / (1.0 + math.exp(-20.0 * (a_i - 0.5)))
        screen_factor = 1.0 / (1.0 + 1.0 / rho_screen)
        mu_val = 1.0 - eps_grav_a * S_a * F_k * screen_factor
        
        # RK4
        H_a = H0 * math.sqrt(Om / a_i**3 + OL)
        
        def rhs(a_val, D_val, dD_val):
            dD_da = dD_val
            d2D_da2 = -(3.0 / a_val + H0**2 * (1.5 * Om / a_val**3 - OL) / H_a**2) * dD_val \
                      + 1.5 * mu_val * Om * H0**2 / (a_val**5 * H_a**2) * D_val
            return np.array([dD_da, d2D_da2])
        
        k1 = rhs(a_i, y_i[0], y_i[1])
        k2 = rhs(a_i + 0.5*h, y_i[0] + 0.5*h*k1[0], y_i[1] + 0.5*h*k1[1])
        k3 = rhs(a_i + 0.5*h, y_i[0] + 0.5*h*k2[0], y_i[1] + 0.5*h*k2[1])
        k4 = rhs(a_i + h, y_i[0] + h*k3[0], y_i[1] + h*k3[1])
        
        y[i+1] = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return float(y[-1, 0])

def f_sigma8_SFE_gpu(z, k_eff, params, eps_grav_a, k_star, rho_screen, sigma8_0):
    """f·σ8(z) 계산 - SFE_gpu_search.py와 동일"""
    a_z = 1.0 / (1.0 + z)
    D_0 = solve_growth_D_gpu(a_z, 1.0, params, eps_grav_a, k_eff, k_star, rho_screen)
    
    if D_0 <= 0:
        return 0.0
    
    D_z = 1.0
    sigma8_z = sigma8_0 * (D_z / D_0)
    
    H0 = params.cosmo.H0
    Om = params.cosmo.Omega_m0
    OL = params.cosmo.Omega_L0
    H_z = H0 * math.sqrt(Om / a_z**3 + OL)
    
    # f(z) 근사
    f_z = (Om / a_z**3) ** 0.55
    
    return f_z * sigma8_z

def fit_sigma8_for_growth_gpu(growth_data, params, eps_grav_a, k_star, rho_screen, k_eff):
    """σ8 최적화 - SFE_gpu_search.py와 동일"""
    sigma8_candidates = np.linspace(0.5, 0.9, 10)
    best_E = float('inf')
    best_sigma8 = 0.8
    
    for s8 in sigma8_candidates:
        s8_val = float(s8)
        E = 0.0
        for obs in growth_data:
            z = obs['z']
            theory = f_sigma8_SFE_gpu(z, k_eff, params, eps_grav_a, k_star, rho_screen, s8_val)
            E += ((obs['fsigma8'] - theory) / obs['err']) ** 2
        
        if E < best_E:
            best_E = E
            best_sigma8 = s8_val
    
    return best_sigma8, best_E

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
    
    # SFE_gpu_search.py와 동일한 데이터 구조
    from dataclasses import dataclass
    
    @dataclass
    class CosmologyParamsGPU:
        H0: float
        Omega_m0: float
        Omega_L0: float
    
    @dataclass
    class GlobalParamsGPU:
        cosmo: CosmologyParamsGPU
        sfe: SFECoreParams
        dm: DMParams
        zprime: ZPrimeParams
    
    cosmo_gpu = CosmologyParamsGPU(H0=2.2e-18, Omega_m0=0.3, Omega_L0=0.7)
    params_gpu = GlobalParamsGPU(cosmo=cosmo_gpu, sfe=SFECoreParams(epsilon_mass=eps_mass, epsilon_grav=eps_0), 
                                  dm=dm, zprime=ZPrimeParams(g_mu=g_mu, m_Zp_GeV=m_Zp_GeV))
    
    # SFE_gpu_search.py와 동일한 데이터
    growth_data_gpu = [
        {'z': 0.02, 'fsigma8': 0.428, 'err': 0.0465},
        {'z': 0.10, 'fsigma8': 0.376, 'err': 0.038},
        {'z': 0.17, 'fsigma8': 0.428, 'err': 0.0465},
        {'z': 0.38, 'fsigma8': 0.440, 'err': 0.050},
        {'z': 0.51, 'fsigma8': 0.458, 'err': 0.038},
    ]
    
    micro_data_gpu = [
        {'obs': 1.6, 'err': 0.1},
        {'obs': 1.55, 'err': 0.1},
        {'obs': 1.25, 'err': 0.1},
    ]
    
    muon_data_gpu = {'Delta_a_mu': 2.5e-9, 'err': 0.6e-9}
    k_eff_gpu = 0.1
    
    try:
        _, E_growth = fit_sigma8_for_growth_gpu(growth_data_gpu, params_gpu, eps_grav_a, k_star, rho_screen, k_eff_gpu)
        
        # 미시 채널 (SFE_gpu_search.py 방식)
        E_micro = 0.0
        for obs in micro_data_gpu:
            theory = 1.0 + eps_mass
            E_micro += ((obs['obs'] - theory) / obs['err']) ** 2
        
        # 뮤온 g-2
        E_mu = muon_g2_error(muon_data_gpu, g_mu, m_Zp_GeV)
        
        return E_growth + E_micro + E_mu
    except:
        return 1e10

# Optuna objective
best_E_global = 9.93  # 이전 최적값

def objective(trial):
    """Optuna objective function"""
    global best_E_global
    
    # 최적값 주변 ±30% 정밀 탐색 (E_total < 5.0 목표)
    # 이전 최적: eps_mass=0.468, eps_0=0.173, a_trans=0.867, beta=20.58
    #            k_star=0.487, rho_screen=63.94, g_mu=4.0e-4, m_Zp=0.078
    params_dict = {
        'eps_mass': trial.suggest_float('eps_mass', 0.468*0.7, 0.468*1.3),
        'eps_0': trial.suggest_float('eps_0', 0.173*0.7, 0.173*1.3),
        'transition_a': trial.suggest_float('transition_a', max(0.1, 0.867*0.7), min(0.9, 0.867*1.3)),
        'sharpness': trial.suggest_float('sharpness', 20.58*0.7, 20.58*1.3),
        'k_star': trial.suggest_float('k_star', 0.487*0.7, 0.487*1.3),
        'rho_screen': trial.suggest_float('rho_screen', 63.94*0.7, 63.94*1.3),
        'g_mu': trial.suggest_float('g_mu', 4.0e-4*0.7, 4.0e-4*1.3),
        'm_Zp_GeV': trial.suggest_float('m_Zp_GeV', 0.078*0.7, 0.078*1.3),
    }
    
    E_total = eval_error(params_dict)
    
    if E_total < best_E_global:
        best_E_global = E_total
        print(f"\n>>> [개선!] E_total={E_total:.4f}")
        for key, val in params_dict.items():
            print(f"    {key:15s} = {val:.6e}")
    
    return E_total

def optuna_optimize(n_trials=5000, target_E=5.0):
    """
    Optuna TPE 최적화
    """
    print("\n" + "="*70)
    print(f"Optuna TPE 최적화 - {n_trials}회")
    print("="*70)
    print(f"\n시작 최적값: E_total = 23.33")
    print(f"목표: E_total < {target_E}")
    
    # Optuna study 생성
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42, n_startup_trials=100)
    )
    
    # 이전 최적값(E=23.33)을 초기 trial로 추가
    study.enqueue_trial({
        'eps_mass': 0.468,
        'eps_0': 0.173,
        'transition_a': 0.867,
        'sharpness': 20.58,
        'k_star': 0.487,
        'rho_screen': 63.94,
        'g_mu': 4.0e-4,
        'm_Zp_GeV': 0.078,
    })
    
    # 최적화 실행
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n{'='*70}")
    print("최종 최적 파라미터:")
    print(f"{'='*70}")
    for key, val in study.best_params.items():
        print(f"  {key:15s} = {val:.6e}")
    print(f"\n최종 E_total = {study.best_value:.4f}")
    
    return study.best_params

if __name__ == "__main__":
    result = optuna_optimize(n_trials=2000, target_E=5.0)

