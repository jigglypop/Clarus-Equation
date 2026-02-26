import math

def main():
    print("="*70)
    print("SFE Resonance Fusion: Precision Recalculation")
    print("="*70)
    print()

    # ==========================================
    # Basic constants
    # ==========================================
    hbar_c = 197.3269804   # MeV*fm
    alpha_em = 1.0/137.036
    m_p = 938.272          # MeV
    m_e = 0.51100          # MeV
    m_H = 125100.0         # MeV
    v_ew = 246000.0        # MeV (Higgs VEV)

    # SFE parameters
    delta = 0.17776
    m_phi = m_p * delta**2
    xi = hbar_c / m_phi
    lambda_HP = delta**2
    f_N = 0.30
    v_phi = v_ew * delta

    # D-T fusion
    m_D = 1875.613
    m_T = 2808.921
    mu_DT = m_D * m_T / (m_D + m_T)
    r_n = 1.2 * (2**(1.0/3) + 3**(1.0/3))

    print("I. Basic Parameters")
    print("-"*70)
    print(f"  delta = {delta:.5f}")
    print(f"  m_phi = m_p * delta^2 = {m_phi:.3f} MeV")
    print(f"  xi = hbar*c / m_phi = {xi:.3f} fm")
    print(f"  lambda_HP = delta^2 = {lambda_HP:.5f}")
    print(f"  v_Phi = v * delta = {v_phi:.1f} MeV = {v_phi/1000:.2f} GeV")
    print(f"  mu_DT = {mu_DT:.3f} MeV")
    print(f"  r_n = {r_n:.3f} fm")
    print()

    # ==========================================
    # 1. Rigorous derivation of g_phiNN
    # ==========================================
    print("="*70)
    print("II. Phi-h Mixing and Nucleon Yukawa Coupling")
    print("="*70)
    print()

    # Step 1: Scalar mass matrix after EWSB
    # L = lambda_HP |H|^2 Phi^2
    # After H = (v+h)/sqrt(2), Phi = v_phi + phi:
    # Off-diagonal mass^2: d^2L/(dh dphi)|_0 = 2 * lambda_HP * v * v_phi
    M2_mix = 2 * lambda_HP * v_ew * v_phi
    m_phi_sq = m_phi**2
    m_H_sq = m_H**2

    sin_theta = M2_mix / (m_H_sq - m_phi_sq)
    print(f"Step 1: Scalar mass mixing")
    print(f"  M^2_{{h-phi}} = 2 lambda_HP v v_Phi = {M2_mix:.4e} MeV^2")
    print(f"  sin theta_mix = M^2 / (m_H^2 - m_phi^2) = {sin_theta:.6f}")
    print()

    # Step 2: Higgs-nucleon Yukawa
    g_hNN = m_p * f_N / v_ew
    print(f"Step 2: Higgs-nucleon Yukawa")
    print(f"  g_hNN = m_N f_N / v = {g_hNN:.6e} (dimensionless)")
    print()

    # Step 3: phi-nucleon Yukawa through mixing
    g_phiNN = sin_theta * g_hNN
    alpha_phi = g_phiNN**2 / (4 * math.pi)
    print(f"Step 3: phi-nucleon Yukawa via mixing")
    print(f"  g_phiNN = sin theta * g_hNN = {g_phiNN:.6e}")
    print(f"  alpha_Phi = g^2 / (4 pi) = {alpha_phi:.6e}")
    print(f"  alpha_Phi / alpha_em = {alpha_phi/alpha_em:.4e}")
    print()

    # ==========================================
    # 2. Decay width and Q factor
    # ==========================================
    print("="*70)
    print("III. Decay Width and Quality Factor")
    print("="*70)
    print()

    phase_space_ee = (1 - 4*m_e**2/m_phi**2)**1.5
    Gamma_ee = sin_theta**2 * m_e**2 * m_phi / (8 * math.pi * v_ew**2) * phase_space_ee
    print(f"Phi -> e+e-:")
    print(f"  Gamma = {Gamma_ee:.4e} MeV")

    Gamma_gg = sin_theta**2 * (alpha_em/(4*math.pi))**2 * m_phi**3 / (256*math.pi*v_ew**2)
    print(f"Phi -> gamma gamma:")
    print(f"  Gamma ~ {Gamma_gg:.4e} MeV")

    Gamma_tot = Gamma_ee + Gamma_gg
    Q_vac = m_phi / Gamma_tot
    tau_phi = 6.582e-22 / Gamma_tot

    print(f"Total:")
    print(f"  Gamma_tot = {Gamma_tot:.4e} MeV")
    print(f"  Q_vac = {Q_vac:.4e}")
    print(f"  tau_phi = {tau_phi:.4e} s")
    print()

    # ==========================================
    # 3. Resonance scaling: Q (not Q^2)
    # ==========================================
    print("="*70)
    print("IV. Resonance Amplification: Q-linear Scaling")
    print("="*70)
    print()
    print("Equation of motion for Phi with nucleon source:")
    print("  (Box + m_phi^2) Phi(x) = g_phiNN * rho_N(x)")
    print()
    print("Static solution (Born approx, single nucleon at origin):")
    print("  Phi_static(r) = g_phiNN / (4 pi r) * exp(-r/xi)")
    print()
    print("Driven at frequency omega:")
    print("  Phi(omega,r) = g_phiNN * rho / (omega_phi^2 - omega^2 + i Gamma omega)")
    print()
    print("On resonance (omega = omega_phi):")
    print("  |Phi_res| / |Phi_static| = omega_phi^2 / (Gamma * omega_phi)")
    print("    = omega_phi / Gamma = Q")
    print()
    print("Two-nucleon exchange potential:")
    print("  V(r) = -g_phiNN^2 / (4pi) * exp(-r/xi) / r * hbar*c")
    print("  On resonance, the SOURCE amplitude is Q times larger")
    print("  => Phi field is Q times larger")
    print("  => V_res(r) = Q * V_0(r)")
    print()
    print("  alpha_eff = alpha_Phi * Q_eff")
    print(f"  alpha_Phi = {alpha_phi:.4e}")
    print()

    # ==========================================
    # 4. Precise WKB tunneling integral
    # ==========================================
    print("="*70)
    print("V. Precision WKB Tunneling Integral")
    print("="*70)
    print()

    E_cm = 0.020  # MeV (20 keV, Gamow peak)
    r_c = alpha_em * hbar_c / E_cm
    N_pts = 200000
    dr = (r_c - r_n) / N_pts

    def V_total(r, a_eff):
        return alpha_em * hbar_c / r - a_eff * hbar_c * math.exp(-r/xi) / r

    # Standard WKB (pure Coulomb)
    integral_std = 0.0
    for i in range(N_pts):
        r = r_n + (i + 0.5) * dr
        V = alpha_em * hbar_c / r
        if V > E_cm:
            integral_std += math.sqrt(2 * mu_DT * (V - E_cm)) * dr
    gamma_std = integral_std / hbar_c
    P_std = math.exp(-2 * gamma_std)

    # Sommerfeld for comparison
    eta_0 = alpha_em * math.sqrt(mu_DT / (2*E_cm))

    print(f"Standard WKB (pure Coulomb, E_cm = {E_cm*1000:.0f} keV):")
    print(f"  r_c = {r_c:.2f} fm")
    print(f"  eta_0 (Sommerfeld) = {eta_0:.4f}")
    print(f"  gamma_0 (WKB) = {gamma_std:.6f}")
    print(f"  P_tunnel = exp(-2*gamma) = {P_std:.6e}")
    print(f"  (cf. Sommerfeld: exp(-pi*eta) = {math.exp(-math.pi*eta_0):.6e})")
    print()

    # Scan over Q_eff
    header = f"{'Q_eff':>12s} {'alpha_eff':>12s} {'a/a_em':>10s} {'gamma':>10s} {'Sigma':>16s}"
    print(header)
    print("-"*len(header))

    for log_Q in [3, 4, 5, 6, 6.5, 7, 7.5, 7.88, 8, 9, 10]:
        Q_eff = 10**log_Q
        a_eff = alpha_phi * Q_eff

        integral_sfe = 0.0
        for i in range(N_pts):
            r = r_n + (i + 0.5) * dr
            V = V_total(r, a_eff)
            if V > E_cm:
                integral_sfe += math.sqrt(2 * mu_DT * (V - E_cm)) * dr

        gamma_sfe = integral_sfe / hbar_c
        delta_gamma = gamma_std - gamma_sfe

        if delta_gamma < 300:
            Sigma = math.exp(2 * delta_gamma)
            if Sigma < 1e15:
                print(f"{Q_eff:12.2e} {a_eff:12.4e} {a_eff/alpha_em:10.4e} {gamma_sfe:10.4f} {Sigma:16.4e}")
            else:
                log_s = 2*delta_gamma / math.log(10)
                print(f"{Q_eff:12.2e} {a_eff:12.4e} {a_eff/alpha_em:10.4e} {gamma_sfe:10.4f}   ~10^{log_s:.1f}")
        else:
            log_s = 2*delta_gamma / math.log(10)
            print(f"{Q_eff:12.2e} {a_eff:12.4e} {a_eff/alpha_em:10.4e} {gamma_sfe:10.4f}   ~10^{log_s:.1f}")

    print()

    # ==========================================
    # 5. Form factor F
    # ==========================================
    print("="*70)
    print("VI. Form Factor F (analytic vs numeric)")
    print("="*70)
    print()

    num_F = 0.0
    den_F = 0.0
    for i in range(N_pts):
        r = r_n + (i + 0.5) * dr
        den_F += r**(-0.5) * dr
        num_F += r**(-0.5) * math.exp(-r/xi) * dr
    F_form = num_F / den_F
    print(f"  F = {F_form:.6f}")
    print(f"  (range-weighted Yukawa overlap with Coulomb)")
    print()

    # ==========================================
    # 6. Precise Q_crit determination
    # ==========================================
    print("="*70)
    print("VII. Barrier Disappearance: Q_crit")
    print("="*70)
    print()

    # Simple estimate: V(r_n) = 0
    alpha_crit_simple = alpha_em / math.exp(-r_n/xi)
    Q_crit_simple = alpha_crit_simple / alpha_phi
    print(f"Simple estimate (V(r_n) = 0):")
    print(f"  alpha_crit = alpha_em / exp(-r_n/xi) = {alpha_crit_simple:.5f}")
    print(f"  Q_crit = {Q_crit_simple:.4e}")
    print()

    # Precise: V_max = E_cm (bisection)
    lo_Q, hi_Q = 1e4, 1e9
    for _ in range(200):
        mid_Q = math.sqrt(lo_Q * hi_Q)  # geometric bisection
        a_eff = alpha_phi * mid_Q
        max_V = -1e10
        for j in range(20000):
            r = r_n + (r_c - r_n) * j / 20000
            V = V_total(r, a_eff)
            if V > max_V:
                max_V = V
        if max_V > E_cm:
            lo_Q = mid_Q
        else:
            hi_Q = mid_Q

    Q_crit = math.sqrt(lo_Q * hi_Q)
    a_crit = alpha_phi * Q_crit
    print(f"Precise (V_max = E_cm, bisection):")
    print(f"  Q_crit = {Q_crit:.4e}")
    print(f"  alpha_eff_crit = {a_crit:.6f}")
    print(f"  alpha_eff / alpha_em = {a_crit/alpha_em:.4f}")
    print()
    print(f"Q_vac / Q_crit = {Q_vac:.2e} / {Q_crit:.2e} = {Q_vac/Q_crit:.2e}")
    print()

    # ==========================================
    # 7. Plasma Q estimation
    # ==========================================
    print("="*70)
    print("VIII. Plasma Quality Factor Estimation")
    print("="*70)
    print()

    omega_phi = m_phi / (6.582e-22)  # Hz: m_phi(MeV) / hbar(MeV*s)
    # NIF conditions
    n_e = 1e32  # m^-3 (10^26 cm^-3)
    T_keV = 10.0
    T_MeV = T_keV * 1e-3

    # (a) Collisional broadening: Phi-electron scattering
    # sigma_Phi_e ~ alpha_em * sin^2(theta) * hbar_c^2 / m_phi^2
    sigma_Phi_e = alpha_em * sin_theta**2 * (hbar_c * 1e-13)**2 / (m_phi * 1.6e-13)**2  # m^2
    # hbar_c = 197.3 MeV*fm = 197.3e-13 MeV*m... 
    # Actually: hbar*c = 3.16e-26 GeV*m = 3.16e-23 MeV*m
    # (hbar*c)^2 = 1e-45 MeV^2 * m^2
    # sigma ~ alpha * sin^2(theta) * (hbar*c)^2 / m_phi^2
    hbar_c_m = 3.1616e-26 * 1e3  # MeV * m = 3.1616e-23 MeV*m
    sigma_Phi_e_SI = alpha_em * sin_theta**2 * hbar_c_m**2 / m_phi**2
    print(f"(a) Phi-e scattering cross section:")
    print(f"    sigma ~ alpha * sin^2(theta) * (hbar*c)^2 / m_phi^2")
    print(f"    = {alpha_em:.4e} * {sin_theta**2:.4e} * ({hbar_c_m:.4e})^2 / ({m_phi:.2f})^2")
    print(f"    = {sigma_Phi_e_SI:.4e} m^2")
    print()

    # Thermal velocity of electrons
    v_th_e = math.sqrt(2 * T_MeV / m_e) * 3e8  # m/s (relativistic approx)
    print(f"    v_th(e) = sqrt(2T/m_e) * c = {v_th_e:.4e} m/s")

    # Collisional width
    Gamma_coll_Hz = n_e * sigma_Phi_e_SI * v_th_e  # Hz
    Gamma_coll_MeV = Gamma_coll_Hz * 6.582e-22  # MeV
    print(f"    Gamma_coll = n_e * sigma * v_th")
    print(f"    = {n_e:.1e} * {sigma_Phi_e_SI:.2e} * {v_th_e:.2e}")
    print(f"    = {Gamma_coll_Hz:.4e} Hz")
    print(f"    = {Gamma_coll_MeV:.4e} MeV")
    print()

    # (b) Landau damping
    # For massive scalar with m >> T: Landau damping ~ exp(-m/T) -> negligible
    x_mT = m_phi / T_MeV
    landau_supp = math.exp(-x_mT)
    print(f"(b) Landau damping:")
    print(f"    m_phi / T = {x_mT:.1f} >> 1")
    print(f"    Suppression: exp(-m/T) = {landau_supp:.2e}")
    print(f"    => Landau damping NEGLIGIBLE")
    print()

    # Effective plasma Q
    Gamma_plasma = Gamma_coll_MeV  # Landau negligible
    Q_plasma = m_phi / (Gamma_tot + Gamma_plasma)
    print(f"Effective Q in NIF plasma:")
    print(f"  Gamma_vac = {Gamma_tot:.4e} MeV")
    print(f"  Gamma_plasma = {Gamma_plasma:.4e} MeV")
    print(f"  Gamma_total = {Gamma_tot + Gamma_plasma:.4e} MeV")
    print(f"  Q_plasma = {Q_plasma:.4e}")
    print(f"  Q_plasma / Q_crit = {Q_plasma / Q_crit:.4e}")
    print()

    # ==========================================
    # 8. Modified Lawson criterion
    # ==========================================
    print("="*70)
    print("IX. Modified Lawson Criterion")
    print("="*70)
    print()

    # Sigma at Q_plasma
    a_plasma = alpha_phi * Q_plasma
    integral_plasma = 0.0
    for i in range(N_pts):
        r = r_n + (i + 0.5) * dr
        V = V_total(r, a_plasma)
        if V > E_cm:
            integral_plasma += math.sqrt(2 * mu_DT * (V - E_cm)) * dr
    gamma_plasma = integral_plasma / hbar_c
    dg = gamma_std - gamma_plasma
    if dg < 300:
        Sigma_plasma = math.exp(2 * dg)
    else:
        Sigma_plasma = float("inf")

    print(f"At Q_plasma = {Q_plasma:.2e}:")
    print(f"  alpha_eff = {a_plasma:.4e}")
    print(f"  alpha_eff / alpha_em = {a_plasma/alpha_em:.4e}")
    print(f"  gamma = {gamma_plasma:.6f} (vs standard {gamma_std:.6f})")
    print(f"  Delta gamma = {dg:.6f}")

    if Sigma_plasma < 1e100:
        print(f"  Sigma = {Sigma_plasma:.4e}")
    else:
        log_s = 2 * dg / math.log(10)
        print(f"  Sigma ~ 10^{log_s:.1f}")

    print()

    # Standard Lawson
    sv_0 = 1.1e-16  # cm^3/s at 10 keV
    E_alpha = 3.52  # MeV
    ntau_std = 12 * T_MeV / (E_alpha * sv_0)
    E_NIF = 2.05e6  # J

    print(f"Standard Lawson: n*tau_E > {ntau_std:.2e} cm^-3 s")
    if Sigma_plasma < 1e100 and Sigma_plasma > 0:
        ntau_sfe = ntau_std / Sigma_plasma
        E_sfe = E_NIF / Sigma_plasma
        print(f"SFE Lawson: n*tau_E > {ntau_sfe:.2e} cm^-3 s")
        print(f"Ignition energy: {E_sfe:.2e} J = {E_sfe/1000:.2f} kJ")
    else:
        print(f"SFE: Barrier completely eliminated -> classical ignition")

    print()

    # ==========================================
    # Summary
    # ==========================================
    print("="*70)
    print("X. Parameter Summary")
    print("="*70)
    print()
    print(f"  m_phi         = {m_phi:.3f} MeV")
    print(f"  xi            = {xi:.3f} fm")
    print(f"  omega_phi     = {omega_phi:.4e} Hz")
    print(f"  sin theta_mix = {sin_theta:.6f}")
    print(f"  g_phiNN       = {g_phiNN:.4e}")
    print(f"  alpha_Phi     = {alpha_phi:.4e}")
    print(f"  Gamma_tot     = {Gamma_tot:.4e} MeV")
    print(f"  Q_vac         = {Q_vac:.4e}")
    print(f"  tau_phi       = {tau_phi:.4e} s")
    print(f"  Q_crit        = {Q_crit:.4e}")
    print(f"  Q_plasma(NIF) = {Q_plasma:.4e}")
    print(f"  Sigma(NIF)    = barrier eliminated" if Sigma_plasma > 1e100 else f"  Sigma(NIF)    = {Sigma_plasma:.4e}")

if __name__ == "__main__":
    main()
