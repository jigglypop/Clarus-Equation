import numpy as np


def true_folding_path(n_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_steps)
    return 1.0 - np.exp(-3.0 * t)


def add_noise(path: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return path + noise_level * rng.standard_normal(path.shape)


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


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def run_experiment():
    n_steps = 200
    noise_level = 0.05
    seed = 2025
    alpha = 0.5
    iterations = 3

    true_path = true_folding_path(n_steps)
    noisy_path = add_noise(true_path, noise_level, seed)

    base_rmsd = rmsd(noisy_path, true_path)

    denoised_path = iterative_sfe_denoise(noisy_path, alpha, iterations)
    sfe_rmsd = rmsd(denoised_path, true_path)

    ratio = base_rmsd / sfe_rmsd if sfe_rmsd > 0.0 else np.inf

    print("=== Protein Folding Path SFE Verification ===")
    print(f"n_steps: {n_steps}")
    print(f"noise_level: {noise_level}")
    print(f"alpha: {alpha}, iterations: {iterations}")
    print(f"base_rmsd: {base_rmsd:.6f}")
    print(f"sfe_rmsd: {sfe_rmsd:.6f}")
    print(f"improvement_factor: {ratio:.4f}")


def sweep_params():
    n_steps = 200
    noise_level = 0.05
    seed = 2025

    true_path = true_folding_path(n_steps)
    noisy_path = add_noise(true_path, noise_level, seed)

    base_rmsd = rmsd(noisy_path, true_path)

    alphas = np.linspace(0.1, 0.9, 9)
    iters_list = [1, 2, 3, 5, 10]
    
    best_alpha = 0.1
    best_iters = 1
    best_rmsd = base_rmsd
    
    for alpha in alphas:
        for iters in iters_list:
            denoised_path = iterative_sfe_denoise(noisy_path, alpha, iters)
            curr_rmsd = rmsd(denoised_path, true_path)
            if curr_rmsd < best_rmsd:
                best_rmsd = curr_rmsd
                best_alpha = alpha
                best_iters = iters

    best_ratio = base_rmsd / best_rmsd if best_rmsd > 0.0 else np.inf
    
    print("\n=== Sweep Parameters (Protein Folding) ===")
    print(f"n_steps: {n_steps}")
    print(f"base_rmsd: {base_rmsd:.6f}")
    print(f"best_alpha: {best_alpha:.2f}")
    print(f"best_iterations: {best_iters}")
    print(f"best_sfe_rmsd: {best_rmsd:.6f}")
    print(f"best_improvement_factor: {best_ratio:.4f}")


if __name__ == "__main__":
    run_experiment()
    sweep_params()
