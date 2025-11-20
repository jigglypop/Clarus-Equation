import numpy as np

EPSILON = 0.384
M_MU_GEV = 0.105658  # GeV
DELTA_A_TARGET = 2.5e-9  # 25 x 10^-10


def loop_integral(m_phi_gev, m_mu=M_MU_GEV, num_points=10000):
    """Numerically integrate the standard scalar loop function."""
    ratio = (m_phi_gev / m_mu) ** 2
    x = np.linspace(0.0, 1.0, num_points)
    denom = (1 - x) ** 2 + x * ratio
    integrand = (1 - x) ** 2 * (1 + x) / denom
    return np.trapezoid(integrand, x)


def delta_a_mu(m_phi_gev, lam, flavor_scale=1.0):
    """Compute Δa_mu^SFE for given m_phi (GeV), lambda, and flavor scaling."""
    if m_phi_gev <= 0:
        raise ValueError("m_phi must be positive")
    if lam <= 0:
        raise ValueError("lambda must be positive")

    g_sq = EPSILON**2 * 2.0 * lam / (m_phi_gev**2)
    integral = loop_integral(m_phi_gev)
    delta = flavor_scale * g_sq * integral / (8.0 * np.pi**2)
    return delta


def lambda_required(m_phi_gev, flavor_scale=1.0, target=DELTA_A_TARGET):
    """Solve for lambda that matches target Δa_mu."""
    base_integral = loop_integral(m_phi_gev)
    prefactor = (
        flavor_scale
        * EPSILON**2
        * 2.0
        * base_integral
        / (8.0 * np.pi**2 * m_phi_gev**2)
    )
    if prefactor <= 0:
        return np.inf
    return target / prefactor


def sample_table():
    masses = np.array([0.1, 0.2, 0.5, 1.0, 2.0])  # GeV
    lam_guess = 1e-3
    rows = []
    for mphi in masses:
        delta = delta_a_mu(mphi, lam_guess)
        lam_match = lambda_required(mphi)
        rows.append((mphi, lam_guess, delta, lam_match))
    return rows


def delta_a_mu_grid(m_phi_vals, lam_vals, delta_mu_vals):
    m_phi_vals = np.asarray(m_phi_vals, dtype=float)
    lam_vals = np.asarray(lam_vals, dtype=float)
    delta_mu_vals = np.asarray(delta_mu_vals, dtype=float)

    shape = (m_phi_vals.size, lam_vals.size, delta_mu_vals.size)
    deltas = np.empty(shape, dtype=float)

    for i, mphi in enumerate(m_phi_vals):
        if mphi <= 0.0:
            raise ValueError("m_phi values must be positive")
        base_int = loop_integral(mphi)
        base_pref = EPSILON**2 * 2.0 * base_int / (8.0 * np.pi**2 * mphi**2)
        for j, lam in enumerate(lam_vals):
            if lam <= 0.0:
                raise ValueError("lambda values must be positive")
            for k, dmu in enumerate(delta_mu_vals):
                flavor_scale = (1.0 + dmu) ** 2
                deltas[i, j, k] = flavor_scale * lam * base_pref

    return deltas


def main():
    rows = sample_table()
    print("m_phi[GeV]\tλ_guess\tΔa_mu(guess)\tλ_required(for 25×10^-10)")
    for mphi, lam_guess, delta, lam_match in rows:
        print(
            f"{mphi:7.3f}\t{lam_guess:.1e}\t"
            f"{delta*1e10:10.3e}\t{lam_match:.3e}"
        )

    lam_cap = 1e-3
    mphi = 0.3
    delta_now = delta_a_mu(mphi, lam_cap)
    flavor_scale_needed = DELTA_A_TARGET / delta_now
    print("\nExample (m_phi=0.3 GeV, λ=1e-3):")
    print(f"  Δa_mu = {delta_now*1e10:.3e} ×10^-10")
    print(f"  Flavor scale needed for target: {flavor_scale_needed:.3e}×")


if __name__ == "__main__":
    main()

