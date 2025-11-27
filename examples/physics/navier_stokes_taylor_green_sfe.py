import numpy as np


def taylor_green_velocity(x, y, t, u0, k, nu):
    decay = np.exp(-2.0 * nu * k * k * t)
    ux = u0 * np.cos(k * x) * np.sin(k * y) * decay
    uy = -u0 * np.sin(k * x) * np.cos(k * y) * decay
    return ux, uy


def generate_reference_field(n, t, u0, k, nu, length):
    xs = np.linspace(0.0, length, n, endpoint=False)
    ys = np.linspace(0.0, length, n, endpoint=False)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    ux, uy = taylor_green_velocity(xx, yy, t, u0, k, nu)
    field = np.stack([ux, uy], axis=-1)
    return field


def add_noise(field, noise_level, seed):
    rng = np.random.default_rng(seed)
    noisy = field + noise_level * rng.standard_normal(field.shape)
    return noisy


def build_sfe_kernel_2d(shape, length, lam, tau):
    nx, ny = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=length / ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid * kx_grid + ky_grid * ky_grid
    k4 = k2 * k2
    factor = np.exp(-(k2 + lam * k4) * tau)
    return factor


def sfe_smooth_field(field, length, lam, tau):
    factor = build_sfe_kernel_2d(field.shape[:2], length, lam, tau)
    field_k = np.fft.fftn(field, axes=(0, 1))
    field_k[..., 0] *= factor
    field_k[..., 1] *= factor
    smoothed = np.fft.ifftn(field_k, axes=(0, 1)).real
    return smoothed


def l2_error(a, b):
    diff = a - b
    return np.sqrt(np.mean(diff * diff))


def run_experiment():
    n = 128
    length = 2.0 * np.pi
    u0 = 1.0
    k = 1.0
    nu = 0.01
    t = 1.0
    noise_level = 0.05
    seed = 1234
    lam = 0.01
    # 기본 모드 k=1에서 진폭 감쇠를 5% 이내로 제한하도록 tau를 선택
    target_atten = 0.95
    tau = -np.log(target_atten) / (1.0 + lam)
    ref = generate_reference_field(n, t, u0, k, nu, length)
    noisy = add_noise(ref, noise_level, seed)
    base_err = l2_error(noisy, ref)
    smoothed = sfe_smooth_field(noisy, length, lam, tau)
    sfe_err = l2_error(smoothed, ref)
    ratio = base_err / sfe_err if sfe_err > 0.0 else np.inf
    print("grid", n)
    print("base_l2_error", base_err)
    print("sfe_l2_error", sfe_err)
    print("improvement_factor", ratio)


def sweep_tau():
    n = 128
    length = 2.0 * np.pi
    u0 = 1.0
    k = 1.0
    nu = 0.01
    t = 1.0
    noise_level = 0.05
    seed = 1234
    lam = 0.01

    ref = generate_reference_field(n, t, u0, k, nu, length)
    noisy = add_noise(ref, noise_level, seed)
    base_err = l2_error(noisy, ref)

    taus = np.linspace(0.0, 2.0, 41)
    best_tau = 0.0
    best_err = base_err
    for tau in taus:
        smoothed = sfe_smooth_field(noisy, length, lam, tau)
        err = l2_error(smoothed, ref)
        if err < best_err:
            best_err = err
            best_tau = tau

    best_ratio = base_err / best_err if best_err > 0.0 else np.inf
    print("sweep_tau_navierstokes")
    print("base_l2_error", base_err)
    print("best_tau", best_tau)
    print("best_sfe_l2_error", best_err)
    print("best_improvement_factor", best_ratio)


if __name__ == "__main__":
    run_experiment()
    sweep_tau()


