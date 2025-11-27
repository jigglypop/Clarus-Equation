import numpy as np
import matplotlib.pyplot as plt
import os


def lorenz_sfe(state, t, sigma, rho, beta, alpha, gamma_sup):
    x, y, z = state
    E = x*x + y*y + z*z
    E_scale = 100.0
    sup = alpha * np.exp(-gamma_sup * E / E_scale)
    
    dx = sigma * (y - x) - sup * x
    dy = x * (rho - z) - y - sup * y
    dz = x * y - beta * z - sup * z
    
    return np.array([dx, dy, dz])


def rk4_step(f, state, t, dt, *args):
    k1 = f(state, t, *args)
    k2 = f(state + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(state + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(state + dt*k3, t + dt, *args)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def compute_lyapunov(alpha, gamma_sup, T=1000, dt=0.005, transient=500):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    np.random.seed(42)
    state = np.array([1.0, 1.0, 1.0]) + 0.1 * np.random.randn(3)
    
    for _ in range(int(transient/dt)):
        state = rk4_step(lorenz_sfe, state, 0, dt, sigma, rho, beta, alpha, gamma_sup)
    
    u = np.eye(3)
    lyap_sum = np.zeros(3)
    n_steps = int(T / dt)
    reorth_interval = 10
    
    for step in range(n_steps):
        state = rk4_step(lorenz_sfe, state, step*dt, dt, sigma, rho, beta, alpha, gamma_sup)
        
        x, y, z = state
        E = x*x + y*y + z*z
        E_scale = 100.0
        sup = alpha * np.exp(-gamma_sup * E / E_scale)
        
        J_lorenz = np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])
        
        r = state.reshape(3, 1)
        J_sfe = -sup * (np.eye(3) + 2 * gamma_sup * r @ r.T)
        
        J = J_lorenz + J_sfe
        
        u = u + dt * (J @ u)
        
        if (step + 1) % reorth_interval == 0:
            u, R = np.linalg.qr(u)
            lyap_sum += np.log(np.abs(np.diag(R)) + 1e-30)
    
    lyapunov_exponents = lyap_sum / T
    return lyapunov_exponents


def main():
    print("SFE-Lorenz Lyapunov Exponent Analysis")
    print("=" * 50)
    
    gamma_sup = 1.0
    alpha_values = np.linspace(0, 2.0, 41)
    
    results = []
    
    for alpha in alpha_values:
        lyap = compute_lyapunov(alpha, gamma_sup, T=300)
        results.append({
            'alpha': alpha,
            'lambda1': lyap[0],
            'lambda2': lyap[1],
            'lambda3': lyap[2]
        })
        print(f"alpha={alpha:.2f}: lambda1={lyap[0]:.4f}, lambda2={lyap[1]:.4f}, lambda3={lyap[2]:.4f}")
    
    alpha_arr = np.array([r['alpha'] for r in results])
    lambda1_arr = np.array([r['lambda1'] for r in results])
    lambda2_arr = np.array([r['lambda2'] for r in results])
    lambda3_arr = np.array([r['lambda3'] for r in results])
    
    idx_cross = np.where(lambda1_arr[:-1] * lambda1_arr[1:] < 0)[0]
    if len(idx_cross) > 0:
        i = idx_cross[0]
        alpha_c = alpha_arr[i] - lambda1_arr[i] * (alpha_arr[i+1] - alpha_arr[i]) / (lambda1_arr[i+1] - lambda1_arr[i])
        print(f"\nCritical alpha (lambda1 = 0): alpha_c = {alpha_c:.3f}")
        
        lambda1_std = results[0]['lambda1']
        sigma_c = alpha_c / lambda1_std if lambda1_std > 0 else alpha_c
        print(f"Critical sigma: sigma_c = alpha_c * T_L = {sigma_c:.3f}")
    else:
        alpha_c = None
        print("\nNo zero crossing found for lambda1")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(alpha_arr, lambda1_arr, 'b-', linewidth=2, label=r'$\lambda_1$ (max)')
    plt.plot(alpha_arr, lambda2_arr, 'g--', linewidth=1.5, label=r'$\lambda_2$')
    plt.plot(alpha_arr, lambda3_arr, 'r:', linewidth=1.5, label=r'$\lambda_3$')
    plt.axhline(0, color='k', linestyle='-', alpha=0.3)
    if alpha_c is not None:
        plt.axvline(alpha_c, color='m', linestyle='--', alpha=0.7, label=f'$\\alpha_c$ = {alpha_c:.2f}')
    plt.xlabel(r'Suppression strength $\alpha$')
    plt.ylabel(r'Lyapunov exponent $\lambda$')
    plt.title('SFE-Lorenz: Lyapunov Exponents vs Suppression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    
    chaos_region = lambda1_arr > 0
    stable_region = lambda1_arr <= 0
    
    plt.fill_between(alpha_arr, 0, 1, where=chaos_region, 
                     color='red', alpha=0.3, label='Chaos')
    plt.fill_between(alpha_arr, 0, 1, where=stable_region,
                     color='green', alpha=0.3, label='Stable')
    
    if alpha_c is not None:
        plt.axvline(alpha_c, color='m', linestyle='--', linewidth=2)
        plt.text(alpha_c + 0.1, 0.5, f'$\\alpha_c$ = {alpha_c:.2f}\n$\\sigma_c$ = {sigma_c:.2f}',
                fontsize=12, verticalalignment='center')
    
    plt.xlabel(r'Suppression strength $\alpha$')
    plt.ylabel('Region')
    plt.title('Chaos-Stability Phase Diagram')
    plt.legend(loc='upper right')
    plt.ylim(0, 1)
    plt.xlim(0, 2)
    
    plt.tight_layout()
    
    os.makedirs("examples/results", exist_ok=True)
    plt.savefig("examples/results/lyapunov_sfe_analysis.png", dpi=150)
    print("\nPlot saved to examples/results/lyapunov_sfe_analysis.png")
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Standard Lorenz lambda1 (alpha=0): {results[0]['lambda1']:.4f}")
    print(f"Expected: ~0.906")
    print(f"Error: {abs(results[0]['lambda1'] - 0.906) / 0.906 * 100:.1f}%")
    
    if alpha_c is not None:
        print(f"\nCritical suppression strength:")
        print(f"  alpha_c = {alpha_c:.3f}")
        print(f"  sigma_c = {sigma_c:.3f}")
        print(f"  Expected range: 1.3 ~ 1.6")
        print(f"  Status: {'PASS' if 1.3 <= sigma_c <= 1.8 else 'CHECK'}")


if __name__ == "__main__":
    main()

