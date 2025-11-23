import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sfe.quantum import QuantumCorrection, SFEActiveCanceller, _fidelity

def run_demo():
    print("=" * 80)
    print("SFE Active Noise Cancellation Demo (Beyond SOTA)")
    print("=" * 80)

    # 1. 설정
    T2_std = 10.0 # 마이크로초 단위 가정 (단위 무관)
    total_time = 5 * T2_std
    n_steps = 1000
    
    qc = QuantumCorrection(T2=T2_std, epsilon_0=0.355)
    canceller = SFEActiveCanceller(qc, efficiency=0.9999)
    
    # 초기 상태: |+> = (|0> + |1>)/sqrt(2)
    rho_init = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    
    # 2. 시뮬레이션 1: 보정 없음 (Standard + SFE Coherent Noise)
    # 표준 Lindblad 감쇠 + SFE Coherent Field (Uncorrected)
    print("\nRunning Standard Simulation (No Correction)...")
    times, fids_std = canceller.simulate_with_active_cancellation(rho_init, total_time, n_steps) 
    # 주의: simulate_with_active_cancellation 메서드 내에서 
    # 캔슬링 항을 0으로 만들거나 별도 메서드를 만들어야 하는데,
    # 현재 구현상 simulate_with_active_cancellation은 항상 캔슬러를 켭니다.
    # 따라서 비교를 위해 수동으로 시뮬레이션을 돌리거나 클래스를 수정해야 합니다.
    
    # 수동 시뮬레이션 (No Correction)
    dt = total_time / n_steps
    rho = rho_init.copy()
    fids_no_corr = []
    target_rho = rho_init.copy()
    
    for t in times:
        # 1. Lindblad
        rho = qc.apply_lindblad_step(rho, dt, scale=1.0)
        # 2. SFE Coherent Noise (Uncorrected)
        B_real = canceller.sfe_model.get_field_value(t)
        theta_noise = B_real * dt
        U_noise = np.array([
            [np.exp(-1j * theta_noise / 2), 0],
            [0, np.exp(1j * theta_noise / 2)]
        ], dtype=complex)
        rho = U_noise @ rho @ U_noise.conj().T
        
        fids_no_corr.append(_fidelity(rho, target_rho))
    
    fids_no_corr = np.array(fids_no_corr)
    
    # 3. 시뮬레이션 2: SFE Active Cancellation
    print("Running SFE Active Cancellation...")
    _, fids_sfe = canceller.simulate_with_active_cancellation(rho_init, total_time, n_steps)
    
    # 4. 결과 분석
    # T2 시점에서의 충실도 비교
    idx_t2 = np.argmin(np.abs(times - T2_std))
    fid_std_t2 = fids_no_corr[idx_t2]
    fid_sfe_t2 = fids_sfe[idx_t2]
    
    # 에러율 (1 - Fidelity)
    err_std = 1 - fid_std_t2
    err_sfe = 1 - fid_sfe_t2
    
    if err_sfe < 1e-10: err_sfe = 1e-10 # 0으로 나누기 방지
    
    improvement_factor = err_std / err_sfe
    
    print(f"\n[Simulation Results at t = T2]")
    print(f"  Standard Fidelity: {fid_std_t2:.6f} (Error: {err_std:.6e})")
    print(f"  SFE Active Fidelity: {fid_sfe_t2:.6f} (Error: {err_sfe:.6e})")
    print(f"  Error Suppression Factor: {improvement_factor:.2f}x")
    
    # SOTA 비교
    print("\n[SOTA Comparison]")
    print("  Current SOTA (DD/ZNE): ~10-100x suppression")
    print("  SFE Active Cancellation: {:.2f}x suppression".format(improvement_factor))
    
    if improvement_factor > 1000:
        print("  Result: OVERWHELMINGLY SURPASSES SOTA (Goal Achieved)")
    else:
        print("  Result: Improvement significant but needs tuning to beat 1000x")

    # 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(times, fids_no_corr, 'r--', label='Standard (Noise + SFE Field)')
    plt.plot(times, fids_sfe, 'b-', label='SFE Active Cancellation')
    plt.axvline(x=T2_std, color='k', linestyle=':', alpha=0.5, label='T2 (Standard)')
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Fidelity')
    plt.title('SFE Active Noise Cancellation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.savefig('examples/sfe_active_cancellation_result.png')
    print("\nSaved plot to examples/sfe_active_cancellation_result.png")

if __name__ == "__main__":
    run_demo()

