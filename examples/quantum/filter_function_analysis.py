import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import sys
import os
sys.path.append('../..')

os.makedirs('examples/results', exist_ok=True)

def compute_filter_function_python(pulse_times, duration, n_steps=8192, n_omega=512):
    y_time = np.ones(n_steps)
    current_sign = 1.0
    pulse_idx = 0
    
    for t in range(n_steps):
        t_norm = t / n_steps
        
        if pulse_idx < len(pulse_times) and t_norm >= pulse_times[pulse_idx]:
            current_sign *= -1.0
            pulse_idx += 1
        
        y_time[t] = current_sign
    
    Y_fft = np.fft.fft(y_time)
    
    dt = duration / n_steps
    freqs = np.fft.fftfreq(n_steps, dt)
    omega = 2 * np.pi * freqs
    
    Y_squared = np.abs(Y_fft * dt)**2
    
    omega_grid = np.linspace(0, np.pi/dt, n_omega)
    y_squared_grid = np.interp(omega_grid, omega[:n_steps//2], Y_squared[:n_steps//2])
    
    return omega_grid, y_squared_grid

def generate_cpmg_sequence(n_pulses):
    return [(j - 0.5) / n_pulses for j in range(1, n_pulses + 1)]

def generate_udd_sequence(n_pulses):
    return [np.sin((j * np.pi) / (2 * n_pulses + 2))**2 for j in range(1, n_pulses + 1)]

def plot_filter_comparison(duration=100.0, n_pulses=8):
    cpmg_seq = generate_cpmg_sequence(n_pulses)
    udd_seq = generate_udd_sequence(n_pulses)
    
    omega_cpmg, y2_cpmg = compute_filter_function_python(cpmg_seq, duration)
    omega_udd, y2_udd = compute_filter_function_python(udd_seq, duration)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(omega_cpmg, y2_cpmg, 'b-', label='CPMG', linewidth=2)
    axes[0, 0].set_xlabel('주파수 ω (rad/μs)')
    axes[0, 0].set_ylabel('|Y(ω)|²')
    axes[0, 0].set_title(f'CPMG 필터 함수 ({n_pulses}펄스)')
    axes[0, 0].set_xlim(0, 1.0)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(omega_udd, y2_udd, 'r-', label='UDD', linewidth=2)
    axes[0, 1].set_xlabel('주파수 ω (rad/μs)')
    axes[0, 1].set_ylabel('|Y(ω)|²')
    axes[0, 1].set_title(f'UDD 필터 함수 ({n_pulses}펄스)')
    axes[0, 1].set_xlim(0, 1.0)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].loglog(omega_cpmg[1:], y2_cpmg[1:], 'b-', label='CPMG', linewidth=2)
    axes[1, 0].loglog(omega_udd[1:], y2_udd[1:], 'r--', label='UDD', linewidth=2)
    axes[1, 0].set_xlabel('주파수 ω (rad/μs, log)')
    axes[1, 0].set_ylabel('|Y(ω)|² (log)')
    axes[1, 0].set_title('저주파 영역 비교')
    axes[1, 0].grid(True, alpha=0.3, which='both')
    axes[1, 0].legend()
    
    gain = y2_cpmg / (y2_udd + 1e-12)
    axes[1, 1].semilogx(omega_cpmg[1:], gain[1:], 'g-', linewidth=2)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('주파수 ω (rad/μs, log)')
    axes[1, 1].set_ylabel('G(ω) = |Y_CPMG|² / |Y_UDD|²')
    axes[1, 1].set_title('이득 함수 (G>1: UDD 우세)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 3)
    
    plt.tight_layout()
    plt.savefig('examples/results/filter_function_analysis.png', dpi=150)
    print(f"필터 함수 분석 저장: examples/results/filter_function_analysis.png")
    
    return omega_cpmg, y2_cpmg, y2_udd, gain

def compute_decoherence_with_spectrum(omega, y_squared, alpha=0.8, scale=1.0):
    s_omega = scale / (omega + 1e-10)**alpha
    
    integrand = s_omega * y_squared
    chi = np.trapz(integrand, omega) / (2 * np.pi)
    
    return chi

def duration_sweep_analysis(durations, n_pulses=8):
    cpmg_chi = []
    udd_chi = []
    ratios = []
    
    for T in durations:
        cpmg_seq = generate_cpmg_sequence(n_pulses)
        udd_seq = generate_udd_sequence(n_pulses)
        
        omega_cpmg, y2_cpmg = compute_filter_function_python(cpmg_seq, T)
        omega_udd, y2_udd = compute_filter_function_python(udd_seq, T)
        
        chi_cpmg = compute_decoherence_with_spectrum(omega_cpmg, y2_cpmg, alpha=0.8)
        chi_udd = compute_decoherence_with_spectrum(omega_udd, y2_udd, alpha=0.8)
        
        cpmg_chi.append(chi_cpmg)
        udd_chi.append(chi_udd)
        ratios.append(chi_cpmg / chi_udd if chi_udd > 0 else 1.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(durations, cpmg_chi, 'b-o', label='CPMG', linewidth=2, markersize=8)
    axes[0].plot(durations, udd_chi, 'r-s', label='UDD/SFE', linewidth=2, markersize=8)
    axes[0].set_xlabel('Duration T (μs)')
    axes[0].set_ylabel('디코히어런스 χ(T)')
    axes[0].set_title('Duration별 누적 위상 분산')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(durations, ratios, 'g-^', linewidth=2, markersize=8)
    axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='경계선 (r=1)')
    axes[1].set_xlabel('Duration T (μs)')
    axes[1].set_ylabel('이득 비 r = χ_CPMG / χ_UDD')
    axes[1].set_title('Duration 의존 성능 비교 (r>1: UDD 우세)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/results/duration_sweep_filter.png', dpi=150)
    print(f"Duration 스윕 분석 저장: examples/results/duration_sweep_filter.png")
    
    return durations, ratios

def main():
    print("=" * 70)
    print("SFE 필터 함수 분석 도구")
    print("=" * 70)
    
    print("\n1. 단일 Duration 필터 함수 비교")
    omega, y2_cpmg, y2_udd, gain = plot_filter_comparison(duration=100.0, n_pulses=8)
    
    low_freq_gain = gain[omega < 0.1].mean()
    mid_freq_gain = gain[(omega >= 0.1) & (omega < 0.5)].mean()
    print(f"   저주파 평균 이득: {low_freq_gain:.3f}")
    print(f"   중주파 평균 이득: {mid_freq_gain:.3f}")
    
    print("\n2. Duration 스윕 분석")
    durations = [10, 20, 30, 40, 50, 60, 80, 100]
    durs, ratios = duration_sweep_analysis(durations, n_pulses=8)
    
    max_idx = np.argmax(ratios)
    print(f"   최대 이득 지점: T={durs[max_idx]} μs, r={ratios[max_idx]:.3f}")
    
    print("\n3. 모멘트 계산")
    cpmg_seq = generate_cpmg_sequence(8)
    udd_seq = generate_udd_sequence(8)
    
    omega_c, y2_c = compute_filter_function_python(cpmg_seq, 100.0)
    omega_u, y2_u = compute_filter_function_python(udd_seq, 100.0)
    
    m0_cpmg = np.trapz(y2_c, omega_c)
    m1_cpmg = np.trapz(omega_c * y2_c, omega_c)
    m2_cpmg = np.trapz(omega_c**2 * y2_c, omega_c)
    
    m0_udd = np.trapz(y2_u, omega_u)
    m1_udd = np.trapz(omega_u * y2_u, omega_u)
    m2_udd = np.trapz(omega_u**2 * y2_u, omega_u)
    
    print(f"   CPMG: M0={m0_cpmg:.3e}, M1={m1_cpmg:.3e}, M2={m2_cpmg:.3e}")
    print(f"   UDD:  M0={m0_udd:.3e}, M1={m1_udd:.3e}, M2={m2_udd:.3e}")
    print(f"   비율: M0={m0_cpmg/m0_udd:.3f}, M1={m1_cpmg/m1_udd:.3f}, M2={m2_cpmg/m2_udd:.3f}")
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main()

