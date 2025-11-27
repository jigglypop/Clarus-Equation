import numpy as np


def hubble_flat(a: np.ndarray, h0: float, omega_m: float, omega_lambda: float) -> np.ndarray:
    return h0 * np.sqrt(omega_m * a ** (-3.0) + omega_lambda)


def add_noise(values: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + noise_level * rng.standard_normal(values.shape)


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
    lam = 0.01
    target_atten = 0.98
    tau = -np.log(target_atten) / (1.0 + lam)

    noisy = add_noise(h_true, noise_level, seed)
    base_rms = rms_error(noisy, h_true)

    # SFE: 허블 곡선 자체가 아니라, "관측 오차장"만 평탄화
    error_h = noisy - h_true
    smooth_error = sfe_smooth_1d(error_h, lam, tau)
    h_sfe = noisy - smooth_error
    sfe_rms = rms_error(h_sfe, h_true)

    ratio = base_rms / sfe_rms if sfe_rms > 0.0 else np.inf

    print("n_points", n)
    print("noise_level", noise_level)
    print("base_rms_error", base_rms)
    print("sfe_rms_error", sfe_rms)
    print("improvement_factor", ratio)


def sweep_tau():
    n = 256
    a = np.linspace(0.1, 1.0, n)
    h0 = 1.0
    omega_m = 0.3
    omega_lambda = 0.7

    h_true = hubble_flat(a, h0, omega_m, omega_lambda)

    noise_level = 0.02
    seed = 777
    lam = 0.01

    noisy = add_noise(h_true, noise_level, seed)
    base_rms = rms_error(noisy, h_true)

    taus = np.linspace(0.0, 2.0, 41)
    best_tau = 0.0
    best_rms = base_rms
    for tau in taus:
        error_h = noisy - h_true
        smooth_error = sfe_smooth_1d(error_h, lam, tau)
        h_sfe = noisy - smooth_error
        curr_rms = rms_error(h_sfe, h_true)
        if curr_rms < best_rms:
            best_rms = curr_rms
            best_tau = tau

    best_ratio = base_rms / best_rms if best_rms > 0.0 else np.inf
    print("sweep_tau_darkenergy")
    print("n_points", n)
    print("base_rms_error", base_rms)
    print("best_tau", best_tau)
    print("best_sfe_rms_error", best_rms)
    print("best_improvement_factor", best_ratio)


if __name__ == "__main__":
    run_experiment()
    sweep_tau()


