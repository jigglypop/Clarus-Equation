import numpy as np


def true_zeros():
    return np.array(
        [
            14.134725141734693,
            21.022039638771555,
            25.010857580145688,
            30.424876125859514,
            32.935061587739190,
            37.586178158825671,
            40.918719012147496,
            43.327073280914999,
            48.005150881167160,
            49.773832477672302,
        ]
    )


def add_noise(values, noise_level, seed):
    rng = np.random.default_rng(seed)
    return values + noise_level * rng.standard_normal(values.shape)


def build_sfe_kernel_1d(n, lam, tau):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    k2 = k * k
    k4 = k2 * k2
    factor = np.exp(-(k2 + lam * k4) * tau)
    return factor


def sfe_smooth_1d(errors, lam, tau):
    n = errors.shape[0]
    factor = build_sfe_kernel_1d(n, lam, tau)
    ek = np.fft.fft(errors)
    ek *= factor
    smoothed = np.fft.ifft(ek).real
    return smoothed


def rms_error(errors):
    return float(np.sqrt(np.mean(errors * errors)))


def run_experiment():
    base = true_zeros()
    noise_level = 0.02
    seed = 123
    lam = 0.01
    tau = 0.8
    approx = add_noise(base, noise_level, seed)
    base_err_vec = approx - base
    base_rms = rms_error(base_err_vec)
    sfe_err_vec = sfe_smooth_1d(base_err_vec, lam, tau)
    sfe_rms = rms_error(sfe_err_vec)
    ratio = base_rms / sfe_rms if sfe_rms > 0.0 else np.inf
    print("n_zeros", base.shape[0])
    print("noise_level", noise_level)
    print("base_rms_error", base_rms)
    print("sfe_rms_error", sfe_rms)
    print("improvement_factor", ratio)


def sweep_tau():
    base = true_zeros()
    noise_level = 0.02
    seed = 123
    lam = 0.01

    approx = add_noise(base, noise_level, seed)
    base_err_vec = approx - base
    base_rms = rms_error(base_err_vec)

    taus = np.linspace(0.0, 2.0, 41)
    best_tau = 0.0
    best_rms = base_rms
    for tau in taus:
        sfe_err_vec = sfe_smooth_1d(base_err_vec, lam, tau)
        rms = rms_error(sfe_err_vec)
        if rms < best_rms:
            best_rms = rms
            best_tau = tau

    best_ratio = base_rms / best_rms if best_rms > 0.0 else np.inf
    print("sweep_tau_riemann")
    print("n_zeros", base.shape[0])
    print("base_rms_error", base_rms)
    print("best_tau", best_tau)
    print("best_sfe_rms_error", best_rms)
    print("best_improvement_factor", best_ratio)


if __name__ == "__main__":
    run_experiment()
    sweep_tau()


