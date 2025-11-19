#!/usr/bin/env python3
"""
ODE constraint를 적용한 Optuna 최적화
ε = α·Ω + β 관계를 강제하면서 최적화
"""
import numpy as np
import math
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

# 상수
M_MU_GEV = 0.1056583755
G_F = 1.1663787e-5

# 우주론 파라미터
H0 = 67.4
Omega_m0 = 0.315
Omega_L0 = 0.685

# 관측 데이터
growth_data_gpu = [
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

micro_data_gpu = [
    {'name': 'BBN_D/H', 'theory': 2.569e-5, 'obs': 2.527e-5, 'err': 0.030e-5},
    {'name': 'BBN_Yp', 'theory': 0.2470, 'obs': 0.2449, 'err': 0.0040},
    {'name': 'Planck_ns', 'theory': 0.9665, 'obs': 0.9649, 'err': 0.0042},
    {'name': 'Planck_As', 'theory': 2.105e-9, 'obs': 2.100e-9, 'err': 0.030e-9},
]

muon_data_gpu = {
    'obs': 25.1e-10,
    'err': 4.8e-10,
}

k_eff_gpu = 0.02

best_E_global = 23.33

def solve_growth_D_gpu(a_ini, a_fin, eps_grav_a, k, k_star, rho_screen, n_steps=600):
    """RK4 적분"""
    h = (a_fin - a_ini) / n_steps
    a_arr = np.linspace(a_ini, a_fin, n_steps + 1)
    y = np.zeros((n_steps + 1, 2))
    y[0] = np.array([1.0, 0.0])
    
    for i in range(n_steps):
        a_i = float(a_arr[i])
        y_i = y[i]
        
        F_k = 1.0 / (1.0 + (k / k_star)**2)
        S_a = 1.0 / (1.0 + math.exp(-20.0 * (a_i - 0.5)))
        screen_factor = 1.0 / (1.0 + 1.0 / rho_screen)
        mu_val = 1.0 - eps_grav_a * S_a * F_k * screen_factor
        
        H_a = H0 * math.sqrt(Omega_m0 / a_i**3 + Omega_L0)
        
        def rhs(a_val, D_val, dD_val):
            dD_da = dD_val
            d2D_da2 = -(3.0 / a_val + H0**2 * (1.5 * Omega_m0 / a_val**3 - Omega_L0) / H_a**2) * dD_val \
                      + 1.5 * mu_val * Omega_m0 * H0**2 / (a_val**5 * H_a**2) * D_val
            return np.array([dD_da, d2D_da2])
        
        k1 = rhs(a_i, y_i[0], y_i[1])
        k2 = rhs(a_i + 0.5*h, y_i[0] + 0.5*h*k1[0], y_i[1] + 0.5*h*k1[1])
        k3 = rhs(a_i + 0.5*h, y_i[0] + 0.5*h*k2[0], y_i[1] + 0.5*h*k2[1])
        k4 = rhs(a_i + h, y_i[0] + h*k3[0], y_i[1] + h*k3[1])
        
        y[i+1] = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return float(y[-1, 0])

def f_sigma8_SFE_gpu(z, k_eff, eps_grav_a, k_star, rho_screen, sigma8_0):
    """f·σ8(z) 계산"""
    a_z = 1.0 / (1.0 + z)
    D_0 = solve_growth_D_gpu(a_z, 1.0, eps_grav_a, k_eff, k_star, rho_screen)
    
    if D_0 <= 0:
        return 0.0
    
    D_z = 1.0
    sigma8_z = sigma8_0 * (D_z / D_0)
    
    H_z = H0 * math.sqrt(Omega_m0 / a_z**3 + Omega_L0)
    f_z = (Omega_m0 / a_z**3) ** 0.55
    
    return f_z * sigma8_z

def fit_sigma8_for_growth_gpu(growth_data, eps_grav_a, k_star, rho_screen, k_eff):
    """σ8 최적화"""
    sigma8_candidates = np.linspace(0.5, 0.9, 10)
    best_E = float('inf')
    best_sigma8 = 0.8
    
    for s8 in sigma8_candidates:
        s8_val = float(s8)
        E = 0.0
        for obs in growth_data:
            z = obs['z']
            theory = f_sigma8_SFE_gpu(z, k_eff, eps_grav_a, k_star, rho_screen, s8_val)
            E += ((obs['fsigma8'] - theory) / obs['err']) ** 2
        
        if E < best_E:
            best_E = E
            best_sigma8 = s8_val
    
    return best_sigma8, best_E

def micro_error(micro_data):
    """미시 채널 에러"""
    E = 0.0
    for obs in micro_data:
        E += ((obs['theory'] - obs['obs']) / obs['err']) ** 2
    return E

def muon_g2_error(g_mu, m_Zp_GeV, muon_data):
    """뮤온 g-2 에러"""
    pref = (g_mu * g_mu) / (12.0 * math.pi * math.pi)
    ratio = (M_MU_GEV * M_MU_GEV) / (m_Zp_GeV * m_Zp_GeV)
    Delta_a_mu_pred = pref * ratio
    
    chi2 = ((Delta_a_mu_pred - muon_data['obs']) / muon_data['err']) ** 2
    return chi2

def eps_grav_at_a(a, eps_mass, eps_0, transition_a, sharpness):
    """ε_grav(a) 계산"""
    return eps_mass + (eps_0 - eps_mass) / (1.0 + math.exp(-sharpness * (a - transition_a)))

def eval_error_ode(params_dict):
    """ODE constraint 적용한 에러 계산"""
    # ODE constraint: ε = α·Ω + β
    alpha = params_dict['alpha']
    beta = params_dict['beta']
    
    # ε_mass와 ε_0를 Ω로부터 계산
    eps_mass = alpha * Omega_m0 + beta
    eps_0 = alpha * Omega_L0 + beta
    
    # 물리적 범위 체크
    if not (0.0 < eps_mass < 1.0) or not (0.0 < eps_0 < 1.0):
        return 1e10
    
    # 나머지 파라미터
    transition_a = params_dict['transition_a']
    sharpness = params_dict['sharpness']
    k_star = params_dict['k_star']
    rho_screen = params_dict['rho_screen']
    g_mu = params_dict['g_mu']
    m_Zp_GeV = params_dict['m_Zp_GeV']
    
    # ε_grav(a=0.5) 계산
    eps_grav_a = eps_grav_at_a(0.5, eps_mass, eps_0, transition_a, sharpness)
    
    # 성장률 에러
    try:
        _, E_growth = fit_sigma8_for_growth_gpu(
            growth_data_gpu, eps_grav_a, k_star, rho_screen, k_eff_gpu
        )
    except:
        E_growth = 1e10
    
    # 미시 채널 에러
    E_micro = micro_error(micro_data_gpu)
    
    # 뮤온 g-2 에러
    E_mu = muon_g2_error(g_mu, m_Zp_GeV, muon_data_gpu)
    
    E_total = E_growth + E_micro + E_mu
    
    return E_total

def objective(trial):
    """Optuna objective with ODE constraint"""
    global best_E_global
    
    # ODE 파라미터: ε = α·Ω + β
    # 이전 최적: α = -0.930, β = 0.767
    params_dict = {
        'alpha': trial.suggest_float('alpha', -1.5, -0.3),
        'beta': trial.suggest_float('beta', 0.5, 1.0),
        'transition_a': trial.suggest_float('transition_a', 0.5, 0.9),
        'sharpness': trial.suggest_float('sharpness', 15.0, 30.0),
        'k_star': trial.suggest_float('k_star', 0.3, 0.7),
        'rho_screen': trial.suggest_float('rho_screen', 40.0, 120.0),
        'g_mu': trial.suggest_float('g_mu', 3e-4, 7e-4),
        'm_Zp_GeV': trial.suggest_float('m_Zp_GeV', 0.05, 0.12),
    }
    
    E_total = eval_error_ode(params_dict)
    
    if E_total < best_E_global:
        best_E_global = E_total
        # ε 값 계산
        eps_mass = params_dict['alpha'] * Omega_m0 + params_dict['beta']
        eps_0 = params_dict['alpha'] * Omega_L0 + params_dict['beta']
        print(f"\n새로운 최적값 발견! E_total = {E_total:.4f}")
        print(f"  α = {params_dict['alpha']:.6f}, β = {params_dict['beta']:.6f}")
        print(f"  ε_mass = {eps_mass:.6f}, ε_0 = {eps_0:.6f}")
    
    return E_total

def optuna_optimize_ode(n_trials=3000, target_E=20.0):
    """ODE constraint 최적화"""
    print("\n" + "="*70)
    print(f"ODE Constraint 최적화 - {n_trials}회")
    print("="*70)
    print(f"\nConstraint: ε = α·Ω + β")
    print(f"시작 최적값: E_total = 23.33")
    print(f"목표: E_total < {target_E}")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42, n_startup_trials=100)
    )
    
    # 이전 최적값 enqueue
    study.enqueue_trial({
        'alpha': -0.930,
        'beta': 0.767,
        'transition_a': 0.748,
        'sharpness': 23.34,
        'k_star': 0.497,
        'rho_screen': 79.02,
        'g_mu': 5.12e-4,
        'm_Zp_GeV': 0.100,
    })
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n{'='*70}")
    print("최종 최적 파라미터:")
    print("="*70)
    
    best_params = study.best_params
    alpha_best = best_params['alpha']
    beta_best = best_params['beta']
    eps_mass_best = alpha_best * Omega_m0 + beta_best
    eps_0_best = alpha_best * Omega_L0 + beta_best
    
    print(f"\nODE 파라미터:")
    print(f"  α = {alpha_best:.6e}")
    print(f"  β = {beta_best:.6e}")
    print(f"\nε 파라미터 (계산됨):")
    print(f"  ε_mass = {eps_mass_best:.6e}")
    print(f"  ε_0    = {eps_0_best:.6e}")
    print(f"\n기타 파라미터:")
    for key in ['transition_a', 'sharpness', 'k_star', 'rho_screen', 'g_mu', 'm_Zp_GeV']:
        print(f"  {key:15s} = {best_params[key]:.6e}")
    
    print(f"\n최종 E_total = {study.best_value:.4f}")
    print("="*70)
    
    return study

if __name__ == "__main__":
    result = optuna_optimize_ode(n_trials=3000, target_E=20.0)

