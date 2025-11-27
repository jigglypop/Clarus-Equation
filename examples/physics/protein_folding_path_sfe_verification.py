import numpy as np


def true_folding_path(n_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_steps)
    # 단조 증가하면서 포화되는 간단한 접힘 경로 (0 -> 1)
    return 1.0 - np.exp(-3.0 * t)


def add_noise(path: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return path + noise_level * rng.standard_normal(path.shape)


def build_sfe_kernel_1d(n: int, lam: float, tau: float) -> np.ndarray:
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    k2 = k * k
    k4 = k2 * k2
    factor = np.exp(-(k2 + lam * k4) * tau)
    return factor


def sfe_smooth_1d(values: np.ndarray, lam: float, tau: float) -> np.ndarray:
    n = values.shape[0]
    factor = build_sfe_kernel_1d(n, lam, tau)
    vk = np.fft.fft(values)
    vk *= factor
    smoothed = np.fft.ifft(vk).real
    return smoothed


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def run_experiment():
    n_steps = 200
    noise_level = 0.05
    seed = 2025
    lam = 0.01
    # 기본 모드(k=1)에 대한 감쇠를 5% 이내로 제한
    target_atten = 0.95
    tau = -np.log(target_atten) / (1.0 + lam)

    true_path = true_folding_path(n_steps)
    noisy_path = add_noise(true_path, noise_level, seed)

    # baseline: 단순 노이즈가 섞인 경로
    base_rmsd = rmsd(noisy_path, true_path)

    # SFE: "오차장"만 평탄화 (리만 실험과 같은 구조)
    error_path = noisy_path - true_path
    smooth_error = sfe_smooth_1d(error_path, lam, tau)
    corrected_path = noisy_path - smooth_error
    sfe_rmsd = rmsd(corrected_path, true_path)

    ratio = base_rmsd / sfe_rmsd if sfe_rmsd > 0.0 else np.inf

    print("n_steps", n_steps)
    print("noise_level", noise_level)
    print("base_rmsd", base_rmsd)
    print("sfe_rmsd", sfe_rmsd)
    print("improvement_factor", ratio)


def sweep_tau():
    n_steps = 200
    noise_level = 0.05
    seed = 2025
    lam = 0.01

    true_path = true_folding_path(n_steps)
    noisy_path = add_noise(true_path, noise_level, seed)

    base_rmsd = rmsd(noisy_path, true_path)

    taus = np.linspace(0.0, 2.0, 41)
    best_tau = 0.0
    best_rmsd = base_rmsd
    for tau in taus:
        error_path = noisy_path - true_path
        smooth_error = sfe_smooth_1d(error_path, lam, tau)
        corrected_path = noisy_path - smooth_error
        curr_rmsd = rmsd(corrected_path, true_path)
        if curr_rmsd < best_rmsd:
            best_rmsd = curr_rmsd
            best_tau = tau

    best_ratio = base_rmsd / best_rmsd if best_rmsd > 0.0 else np.inf
    print("sweep_tau_protein")
    print("n_steps", n_steps)
    print("base_rmsd", base_rmsd)
    print("best_tau", best_tau)
    print("best_sfe_rmsd", best_rmsd)
    print("best_improvement_factor", best_ratio)


if __name__ == "__main__":
    run_experiment()
    sweep_tau()


