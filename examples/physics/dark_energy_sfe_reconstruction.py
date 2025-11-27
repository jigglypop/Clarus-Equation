import numpy as np


def hubble_flat(a: np.ndarray, h0: float, omega_m: float, omega_lambda: float) -> np.ndarray:
    return h0 * np.sqrt(omega_m * a ** (-3.0) + omega_lambda)


def add_noise(values: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + noise_level * rng.standard_normal(values.shape)


def compute_curvature(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    curv = np.zeros(n)
    for i in range(1, n - 1):
        curv[i] = values[i - 1] - 2 * values[i] + values[i + 1]
    curv[0] = curv[1]
    curv[-1] = curv[-2]
    return curv


def sfe_curvature_denoise(values: np.ndarray, alpha: float) -> np.ndarray:
    curv = compute_curvature(values)
    curv_abs = np.abs(curv)
    threshold = np.percentile(curv_abs, 75)
    high_curv_mask = curv_abs > threshold
    
    result = values.copy()
    for i in range(1, len(values) - 1):
        if high_curv_mask[i]:
            neighbor_avg = 0.5 * (values[i - 1] + values[i + 1])
            result[i] = (1 - alpha) * values[i] + alpha * neighbor_avg
    return result


def iterative_sfe_denoise(values: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    result = values.copy()
    for _ in range(iterations):
        result = sfe_curvature_denoise(result, alpha)
    return result


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def run_experiment():
    n = 256
    a = np.linspace(0.1, 1.0, n)
    h0 = 1.0
    omega_m = 0.3
    omega_lambda = 0.7

    h_true = hubble_flat(a, h0, omega_m, omega_lambda)

    noise_level = 0.02
    seed = 777
    alpha = 0.5
    iterations = 3

    noisy = add_noise(h_true, noise_level, seed)
    base_rms = rms_error(noisy, h_true)

    h_sfe = iterative_sfe_denoise(noisy, alpha, iterations)
    sfe_rms = rms_error(h_sfe, h_true)

    ratio = base_rms / sfe_rms if sfe_rms > 0.0 else np.inf

    print("=== Dark Energy Hubble Curve SFE Verification ===")
    print(f"n_points: {n}")
    print(f"noise_level: {noise_level}")
    print(f"alpha: {alpha}, iterations: {iterations}")
    print(f"base_rms_error: {base_rms:.6f}")
    print(f"sfe_rms_error: {sfe_rms:.6f}")
    print(f"improvement_factor: {ratio:.4f}")


def sweep_params():
    n = 256
    a = np.linspace(0.1, 1.0, n)
    h0 = 1.0
    omega_m = 0.3
    omega_lambda = 0.7

    h_true = hubble_flat(a, h0, omega_m, omega_lambda)

    noise_level = 0.02
    seed = 777

    noisy = add_noise(h_true, noise_level, seed)
    base_rms = rms_error(noisy, h_true)

    alphas = np.linspace(0.1, 0.9, 9)
    iters_list = [1, 2, 3, 5, 10]
    
    best_alpha = 0.1
    best_iters = 1
    best_rms = base_rms
    
    for alpha in alphas:
        for iters in iters_list:
            h_sfe = iterative_sfe_denoise(noisy, alpha, iters)
            curr_rms = rms_error(h_sfe, h_true)
            if curr_rms < best_rms:
                best_rms = curr_rms
                best_alpha = alpha
                best_iters = iters

    best_ratio = base_rms / best_rms if best_rms > 0.0 else np.inf
    
    print("\n=== Sweep Parameters (Dark Energy) ===")
    print(f"n_points: {n}")
    print(f"base_rms_error: {base_rms:.6f}")
    print(f"best_alpha: {best_alpha:.2f}")
    print(f"best_iterations: {best_iters}")
    print(f"best_sfe_rms_error: {best_rms:.6f}")
    print(f"best_improvement_factor: {best_ratio:.4f}")


if __name__ == "__main__":
    run_experiment()
    sweep_params()
