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


def compute_curvature_2d(field):
    curv = np.zeros_like(field)
    for c in range(field.shape[-1]):
        for i in range(1, field.shape[0] - 1):
            for j in range(1, field.shape[1] - 1):
                laplacian = (
                    field[i - 1, j, c] + field[i + 1, j, c] +
                    field[i, j - 1, c] + field[i, j + 1, c] -
                    4 * field[i, j, c]
                )
                curv[i, j, c] = laplacian
    return curv


def sfe_curvature_denoise_2d(field, alpha):
    curv = compute_curvature_2d(field)
    curv_mag = np.sqrt(np.sum(curv ** 2, axis=-1))
    threshold = np.percentile(curv_mag, 75)
    high_curv_mask = curv_mag > threshold
    
    result = field.copy()
    for i in range(1, field.shape[0] - 1):
        for j in range(1, field.shape[1] - 1):
            if high_curv_mask[i, j]:
                neighbor_avg = 0.25 * (
                    field[i - 1, j] + field[i + 1, j] +
                    field[i, j - 1] + field[i, j + 1]
                )
                result[i, j] = (1 - alpha) * field[i, j] + alpha * neighbor_avg
    return result


def iterative_sfe_denoise_2d(field, alpha, iterations):
    result = field.copy()
    for _ in range(iterations):
        result = sfe_curvature_denoise_2d(result, alpha)
    return result


def l2_error(a, b):
    diff = a - b
    return np.sqrt(np.mean(diff * diff))


def run_experiment():
    n = 64
    length = 2.0 * np.pi
    u0 = 1.0
    k = 1.0
    nu = 0.01
    t = 1.0
    noise_level = 0.05
    seed = 1234
    alpha = 0.5
    iterations = 3

    ref = generate_reference_field(n, t, u0, k, nu, length)
    noisy = add_noise(ref, noise_level, seed)
    base_err = l2_error(noisy, ref)
    
    denoised = iterative_sfe_denoise_2d(noisy, alpha, iterations)
    sfe_err = l2_error(denoised, ref)
    
    ratio = base_err / sfe_err if sfe_err > 0.0 else np.inf
    
    print("=== Navier-Stokes Taylor-Green SFE Verification ===")
    print(f"grid: {n}x{n}")
    print(f"noise_level: {noise_level}")
    print(f"alpha: {alpha}, iterations: {iterations}")
    print(f"base_l2_error: {base_err:.6f}")
    print(f"sfe_l2_error: {sfe_err:.6f}")
    print(f"improvement_factor: {ratio:.4f}")


def sweep_params():
    n = 64
    length = 2.0 * np.pi
    u0 = 1.0
    k = 1.0
    nu = 0.01
    t = 1.0
    noise_level = 0.05
    seed = 1234

    ref = generate_reference_field(n, t, u0, k, nu, length)
    noisy = add_noise(ref, noise_level, seed)
    base_err = l2_error(noisy, ref)

    alphas = np.linspace(0.1, 0.9, 9)
    iters_list = [1, 2, 3, 5]
    
    best_alpha = 0.1
    best_iters = 1
    best_err = base_err
    
    for alpha in alphas:
        for iters in iters_list:
            denoised = iterative_sfe_denoise_2d(noisy, alpha, iters)
            curr_err = l2_error(denoised, ref)
            if curr_err < best_err:
                best_err = curr_err
                best_alpha = alpha
                best_iters = iters

    best_ratio = base_err / best_err if best_err > 0.0 else np.inf
    
    print("\n=== Sweep Parameters (Navier-Stokes) ===")
    print(f"grid: {n}x{n}")
    print(f"base_l2_error: {base_err:.6f}")
    print(f"best_alpha: {best_alpha:.2f}")
    print(f"best_iterations: {best_iters}")
    print(f"best_sfe_l2_error: {best_err:.6f}")
    print(f"best_improvement_factor: {best_ratio:.4f}")


if __name__ == "__main__":
    run_experiment()
    sweep_params()
