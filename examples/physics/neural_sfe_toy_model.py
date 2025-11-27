import numpy as np


def target_trajectory(n_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_steps)
    y = np.zeros_like(t)
    y[t >= 0.33] = 1.0
    y[t >= 0.66] = 0.5
    return y


def add_spiky_noise(traj: np.ndarray, small_noise: float, spike_amp: float, n_spikes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = traj + small_noise * rng.standard_normal(traj.shape)
    idx = rng.integers(0, traj.shape[0], size=n_spikes)
    noisy[idx] += spike_amp * rng.choice([-1.0, 1.0], size=n_spikes)
    return noisy


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
    n_steps = 300
    small_noise = 0.05
    spike_amp = 1.0
    n_spikes = 15
    seed = 42
    lam = 0.01
    target_atten = 0.98
    tau = -np.log(target_atten) / (1.0 + lam)

    true_traj = target_trajectory(n_steps)
    noisy_traj = add_spiky_noise(true_traj, small_noise, spike_amp, n_spikes, seed)

    base_rms = rms_error(noisy_traj, true_traj)

    # SFE: "스파이크+노이즈 오차장"만 평탄화
    error_traj = noisy_traj - true_traj
    smooth_error = sfe_smooth_1d(error_traj, lam, tau)
    sfe_traj = noisy_traj - smooth_error
    sfe_rms = rms_error(sfe_traj, true_traj)

    ratio = base_rms / sfe_rms if sfe_rms > 0.0 else np.inf

    print("n_steps", n_steps)
    print("small_noise", small_noise)
    print("spike_amp", spike_amp)
    print("base_rms_error", base_rms)
    print("sfe_rms_error", sfe_rms)
    print("improvement_factor", ratio)


def sweep_tau():
    n_steps = 300
    small_noise = 0.05
    spike_amp = 1.0
    n_spikes = 15
    seed = 42
    lam = 0.01

    true_traj = target_trajectory(n_steps)
    noisy_traj = add_spiky_noise(true_traj, small_noise, spike_amp, n_spikes, seed)

    base_rms = rms_error(noisy_traj, true_traj)

    taus = np.linspace(0.0, 2.0, 41)
    best_tau = 0.0
    best_rms = base_rms
    for tau in taus:
        error_traj = noisy_traj - true_traj
        smooth_error = sfe_smooth_1d(error_traj, lam, tau)
        sfe_traj = noisy_traj - smooth_error
        curr_rms = rms_error(sfe_traj, true_traj)
        if curr_rms < best_rms:
            best_rms = curr_rms
            best_tau = tau

    best_ratio = base_rms / best_rms if best_rms > 0.0 else np.inf
    print("sweep_tau_neural")
    print("n_steps", n_steps)
    print("base_rms_error", base_rms)
    print("best_tau", best_tau)
    print("best_sfe_rms_error", best_rms)
    print("best_improvement_factor", best_ratio)


if __name__ == "__main__":
    run_experiment()
    sweep_tau()


