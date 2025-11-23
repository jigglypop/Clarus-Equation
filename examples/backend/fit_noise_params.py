import numpy as np
from scipy.optimize import curve_fit


def filter_function_val(omega, pulses, T):
    """
    Calculates filter function |Y(omega)|^2 for a specific sequence.
    pulses: relative positions [0, 1]
    y(t): toggle function (+1/-1)
    """
    n_steps = 2000
    dt = T / n_steps
    t = np.linspace(0, T, n_steps, endpoint=False)
    
    y = np.ones(n_steps)
    current_sign = 1.0
    
    valid_pulses = [p for p in pulses if 0 < p < 1]
    pulse_indices = [int(p * n_steps) for p in valid_pulses]
    pulse_indices.sort()
    
    pulse_ptr = 0
    for i in range(n_steps):
        while pulse_ptr < len(pulse_indices) and i >= pulse_indices[pulse_ptr]:
            current_sign *= -1.0
            pulse_ptr += 1
        y[i] = current_sign
        
    if np.isscalar(omega):
        phase = omega * t
        Y = np.sum(y * np.exp(1j * phase)) * dt
        return np.abs(Y)**2
    else:
        Y_vals = []
        for w in omega:
             phase = w * t
             Y = np.sum(y * np.exp(1j * phase)) * dt
             Y_vals.append(np.abs(Y)**2)
        return np.array(Y_vals)


def chi_1overf(T, alpha, A, sequence_type='cpmg'):
    """
    Pure 1/f noise contribution to chi(T).
    chi_1f(T) = A * int (1/w^alpha) * |Y(w, T)|^2 dw / (2pi)
    """
    if T <= 0:
        return 0.0

    if sequence_type == 'cpmg':
        pulses = [(k - 0.5) / 8.0 for k in range(1, 9)]
    elif sequence_type == 'sfe':
        pulses = [0.0300, 0.1170, 0.2500, 0.4030, 0.5870, 0.7500, 0.8830, 0.9700]
    else:
        raise ValueError("Unknown sequence type")
        
    # Fixed frequency range (dimensionless, will be scaled by T internally in filter function)
    # Use normalized frequency z = omega * T
    z_min = 0.1
    z_max = 100.0
    
    nw = 150
    z_vals = np.logspace(np.log10(z_min), np.log10(z_max), nw)
    omegas = z_vals / T
    
    F_vals = filter_function_val(omegas, pulses, T)
    
    S_vals = 1.0 / (omegas**alpha)
    
    integrand = S_vals * F_vals
    
    integral = np.trapezoid(integrand, omegas)
    
    return A * integral / (2 * np.pi)


def coherence_curve_full(T_list, alpha, A, gamma_white, sequence_type):
    """
    Full coherence S(T) = exp(-gamma_white * T - chi_1f(T))
    gamma_white: white noise rate (T1, gate errors, etc.)
    """
    vals = []
    for T in T_list:
        if T == 0:
            vals.append(1.0)
        else:
            chi_1f = chi_1overf(T, alpha, A, sequence_type)
            chi_total = gamma_white * T + chi_1f
            vals.append(np.exp(-chi_total))
    return np.array(vals)


def fit_combined_full(x_data, alpha, A, gamma_white):
    """
    Combined fit with white noise.
    x_data: [T_cpmg..., T_sfe...]
    """
    n = len(x_data) // 2
    T_cpmg = x_data[:n]
    T_sfe = x_data[n:]
    
    S_cpmg = coherence_curve_full(T_cpmg, alpha, A, gamma_white, 'cpmg')
    S_sfe = coherence_curve_full(T_sfe, alpha, A, gamma_white, 'sfe')
    
    return np.concatenate([S_cpmg, S_sfe])


def main():
    print("=== SFE Noise Parameter Fitter (with White Noise) ===")
    print("Estimating (alpha, A, gamma_white) from IBM Fez data...")
    
    dt_ns = 4.5
    durations_dt = np.array([0, 4444, 8888])
    T_us = durations_dt * dt_ns * 1e-3 
    
    S_exp_cpmg = np.array([1.0, 0.3936, 0.15]) 
    S_exp_sfe = np.array([1.0, 0.5518, 0.28])
    
    x_combined = np.concatenate([T_us, T_us])
    y_combined = np.concatenate([S_exp_cpmg, S_exp_sfe])
    
    # Fitting with 3 params: alpha, A, gamma_white
    # alpha: 0.7 ~ 1.3 (pink noise)
    # A: > 0
    # gamma_white: > 0 (includes T1, gate errors)
    
    p0 = [0.8, 0.5, 0.02]
    bounds = ([0.6, 0.0, 0.0], [1.5, np.inf, 0.1])
    
    try:
        popt, pcov = curve_fit(fit_combined_full, x_combined, y_combined, p0=p0, bounds=bounds, maxfev=5000)
        alpha_fit, A_fit, gamma_fit = popt
        
        perr = np.sqrt(np.diag(pcov))
        
        print(f"\n[Fit Result]")
        print(f"Alpha (1/f exponent)  : {alpha_fit:.4f} ± {perr[0]:.4f}")
        print(f"A (1/f Strength)      : {A_fit:.4e} ± {perr[1]:.2e}")
        print(f"Gamma_white (T1+gate) : {gamma_fit:.4f} ± {perr[2]:.4f} /us")
        print("-" * 50)
        
        S_fit_cpmg = coherence_curve_full(T_us, alpha_fit, A_fit, gamma_fit, 'cpmg')
        S_fit_sfe = coherence_curve_full(T_us, alpha_fit, A_fit, gamma_fit, 'sfe')
        
        print(f"{'Time(us)':<10} | {'CPMG Exp':<10} {'CPMG Fit':<10} | {'SFE Exp':<10} {'SFE Fit':<10}")
        for i in range(len(T_us)):
            print(f"{T_us[i]:<10.2f} | {S_exp_cpmg[i]:<10.4f} {S_fit_cpmg[i]:<10.4f} | {S_exp_sfe[i]:<10.4f} {S_fit_sfe[i]:<10.4f}")
            
        print("-" * 50)
        
        # Relative contribution check
        if len(T_us) > 1:
            T_test = T_us[1]
            chi_1f_cpmg = chi_1overf(T_test, alpha_fit, A_fit, 'cpmg')
            chi_1f_sfe = chi_1overf(T_test, alpha_fit, A_fit, 'sfe')
            chi_white = gamma_fit * T_test
            
            print(f"[Contribution at T={T_test:.1f}us]")
            print(f"  White noise (T1+gate): chi_white = {chi_white:.4f}")
            print(f"  1/f (CPMG): chi_1f = {chi_1f_cpmg:.4f}")
            print(f"  1/f (SFE):  chi_1f = {chi_1f_sfe:.4f}")
            print(f"  1/f reduction: CPMG/SFE = {chi_1f_cpmg/chi_1f_sfe:.2f}x")
        
        # Physics Check
        if 0.7 <= alpha_fit <= 1.3:
             print("\n[Physics Check] Alpha is in typical 1/f range (0.7–1.3). PASS")
        else:
             print(f"\n[Physics Check] Alpha={alpha_fit:.2f} outside 1/f range. Review model.")
             
        # Rust Engine Params
        print("\n[Rust Engine Params]")
        print(f"SFE_NOISE_ALPHA = {alpha_fit:.4f}")
        print(f"SFE_NOISE_SCALE = {A_fit:.4e}")
        print(f"(Note: gamma_white absorbed into effective p_phy in QEC sim)")
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

