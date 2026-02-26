import math

def main():
    # Basic parameters
    hbar_c = 197.327  # MeV*fm
    alpha_em = 1.0/137.036
    m_p = 938.272
    m_phi = m_p * 0.17776**2  # 29.648 MeV
    xi = hbar_c / m_phi
    omega_phi = m_phi / 6.582e-22  # Hz
    Gamma_phi = 9.588e-15  # MeV
    Q_vac = m_phi / Gamma_phi
    alpha_phi = 1.966e-10
    g_phiNN = 4.971e-5
    sin_theta = 0.04344
    Q_crit = 1e9

    print("="*70)
    print("SFE Resonance Drive: Feasibility Analysis")
    print("="*70)
    print()

    # === Problem statement ===
    print("A. Problem: Required Energy Resolution")
    print("-"*70)
    print(f"  omega_phi = {omega_phi:.3e} Hz")
    print(f"  E_phi = {m_phi:.3f} MeV")
    print(f"  Q_vac = {Q_vac:.2e}")
    print(f"  Resonance width: dE = E/Q = {m_phi/Q_vac:.2e} MeV")
    print(f"    = {m_phi/Q_vac*1e15:.2f} feV (femto-eV)")
    print()

    # Q_eff to achieve various Sigma
    print("  Required Q_eff for given Sigma (from WKB table):")
    print(f"    Sigma = 1.01: Q_eff ~ 1e6  -> dE/E = 1e-6")
    print(f"    Sigma = 1.13: Q_eff ~ 1e7  -> dE/E = 1e-7")
    print(f"    Sigma = 5.4:  Q_eff ~ 1e8  -> dE/E = 1e-8")
    print(f"    Sigma = 278:  Q_eff ~ 1e9  -> dE/E = 1e-9")
    print()

    # === Current technology ===
    print("="*70)
    print("B. Current Gamma-ray Technology")
    print("="*70)
    print()

    sources = [
        ("ELI-NP (ICS)", "10-20 MeV", 0.005, 1e8),
        ("HIgammaS (ICS)", "1-100 MeV", 0.03, 1e8),
        ("VEGA-3 (laser Compton)", "1-19 MeV", 0.01, 1e6),
        ("NRF (nuclear resonance)", "1-10 MeV", 1e-6, 1e6),
        ("Channeling radiation", "10-100 MeV", 0.10, 1e10),
    ]

    print(f"{'Source':<25s} {'Energy':>12s} {'dE/E':>10s} {'Flux':>12s} {'Q_eff':>10s}")
    print("-"*70)
    for name, Erange, dEE, flux in sources:
        Q_eff = 1.0/dEE
        print(f"{name:<25s} {Erange:>12s} {dEE:>10.1e} {flux:>12.0e} {Q_eff:>10.0e}")
    print()

    # === Key insight: two-stage approach ===
    print("="*70)
    print("C. Two-stage Approach: ICS + phi Self-filtering")
    print("="*70)
    print()
    print("Core insight: We do NOT need delta E/E = 1/Q_vac for the SOURCE.")
    print("The phi boson itself acts as an ultra-narrow bandpass filter.")
    print()
    print("Mechanism:")
    print("  1) Broad-band gamma beam (dE/E ~ 0.5%) irradiates plasma")
    print("  2) gamma + N -> phi + N  (photo-production)")
    print("     Only gammas within dE = Gamma_phi of m_phi contribute")
    print("  3) Produced phi has lifetime tau = 69 ns")
    print("  4) During this time, phi oscillates coherently at omega_phi")
    print("  5) This coherent phi field modifies Coulomb barrier")
    print()
    print("The effective Q is determined by phi's NATURAL linewidth,")
    print("not by the source bandwidth!")
    print()

    # Quantitative analysis
    print("-"*70)
    print("Quantitative Analysis:")
    print()

    # Photo-production cross section
    # gamma + N -> phi + N via Higgs-portal mixing
    # At E_gamma = m_phi: threshold production
    # sigma ~ sin^2(theta_mix) * alpha_em * (hbar_c/m_phi)^2
    lambda_c = hbar_c / m_phi  # fm
    sigma_prod = sin_theta**2 * alpha_em * lambda_c**2 * 1e-26  # cm^2
    print(f"  phi photo-production cross section:")
    print(f"    sigma ~ sin^2(theta) * alpha * (hbar*c/m_phi)^2")
    print(f"    = {sin_theta**2:.4f} * {alpha_em:.4f} * ({lambda_c:.2f} fm)^2")
    print(f"    = {sigma_prod:.2e} cm^2")
    print()

    # Fraction of beam in resonance window
    dE_E_beam = 0.005  # 0.5% bandwidth
    f_res = (Gamma_phi / m_phi) / dE_E_beam
    print(f"  Fraction of beam in resonance window:")
    print(f"    f = (Gamma/m) / (dE/E)_beam")
    print(f"    = ({Gamma_phi:.2e}/{m_phi:.2f}) / {dE_E_beam}")
    print(f"    = {f_res:.2e}")
    print()

    # Effective production cross section (beam-averaged)
    sigma_eff = sigma_prod * f_res
    print(f"  Beam-averaged production cross section:")
    print(f"    sigma_eff = sigma_prod * f_res = {sigma_eff:.2e} cm^2")
    print()

    # NIF plasma conditions
    n_N = 6e29  # cm^-3
    R_cap = 0.005  # cm (50 um compressed capsule)
    l_path = 2 * R_cap
    print(f"  NIF compressed capsule:")
    print(f"    n_N = {n_N:.0e} cm^-3")
    print(f"    path length = {l_path*1e4:.0f} um")
    print()

    # Number of phi produced per gamma
    N_phi_per_gamma = n_N * sigma_prod * l_path
    print(f"  phi produced per incident gamma:")
    print(f"    P = n_N * sigma * l = {N_phi_per_gamma:.2e}")
    print()

    # With ELI-NP flux
    N_gamma = 1e8  # photons/s (at 29 MeV)
    # Only resonant fraction contributes to phi production
    # But actually, the cross section already accounts for the resonance
    # The resonant enhancement factor is built into sigma_prod
    
    # More careful: The photo-production sigma is peaked at E = m_phi
    # with width Gamma_phi. For a broad beam, the integrated cross section:
    # integral sigma(E) dE = sigma_peak * pi * Gamma_phi / 2
    # And the beam provides photons at rate dN/dE = N_total / (E * dE/E)
    # So: Rate = (dN/dE)|_{m_phi} * integral sigma dE
    #          = N_total / (m_phi * dE_E_beam) * sigma_peak * pi * Gamma / 2

    dNdE = N_gamma / (m_phi * dE_E_beam)  # photons / MeV / s
    sigma_peak = 4 * math.pi * lambda_c**2 * 1e-26  # cm^2 (geometric, spin-0)
    # Actual peak = sigma_geom * BR_in * BR_out
    # BR_in(gamma + N -> phi) ~ sin^2(theta) * alpha
    # This is getting complicated. Use simpler estimate:
    
    # sigma_Breit-Wigner at peak for gamma + N -> phi -> anything:
    # sigma_BW = pi * (hbar*c/p)^2 * Gamma_gammaN / Gamma_tot
    # where Gamma_gammaN is the partial width for phi -> gamma + N channel
    # This channel doesn't really exist for a scalar...
    
    # Better approach: phi is produced via Primakoff-like process
    # gamma + Z -> phi + Z (coherent nuclear production)
    # or via inverse decay in medium
    
    # Actually the cleanest production channel:
    # In a dense medium, the photon can convert to phi via mixing
    # The conversion probability per unit length:
    # P/L = (sin theta_mix)^2 * Im(self-energy) ~ (sin theta)^2 * omega / (2 * l_abs)
    
    # Photon-phi oscillation in medium (like neutrino oscillation):
    # theta_eff = theta_mix * m_phi^2 / (m_phi^2 - omega_plasma^2)
    # omega_plasma for NIF: omega_p^2 = 4*pi*alpha*n_e/m_e
    
    n_e = 1e26  # cm^-3
    # omega_p^2 in MeV^2:
    # omega_p = sqrt(4*pi*alpha*n_e*(hbar*c)^3 / (m_e*c^2))
    # (hbar*c)^3 = (197.3 MeV*fm)^3 = 7.69e6 MeV^3*fm^3
    # fm^3 = 1e-39 cm^3
    omega_p2 = 4*math.pi*alpha_em * n_e * 1e-39 * 197.327**3 / 0.511  # MeV^2
    omega_p = math.sqrt(omega_p2) if omega_p2 > 0 else 0
    print(f"  Plasma frequency:")
    print(f"    omega_p^2 = {omega_p2:.2e} MeV^2")
    print(f"    omega_p = {omega_p:.4f} MeV = {omega_p*1e3:.1f} keV")
    print(f"    (cf. m_phi = {m_phi:.2f} MeV -> omega_p << m_phi)")
    print()

    # Photon-phi conversion probability (Primakoff in medium):
    # P(gamma -> phi) = sin^2(2*theta_eff) * sin^2(Delta * L / 2)
    # Delta = |m_phi^2 - omega_p^2| / (2*E)
    # For E ~ m_phi: Delta ~ m_phi / 2

    Delta = abs(m_phi**2 - omega_p2) / (2 * m_phi)  # MeV
    L_osc = 2 * math.pi / Delta  # fm (oscillation length in natural units)
    L_osc_cm = L_osc * 1e-13  # cm
    print(f"  Photon-phi oscillation:")
    print(f"    Delta = (m_phi^2 - omega_p^2)/(2E) = {Delta:.2f} MeV")
    print(f"    L_osc = 2*pi*hbar*c/Delta = {2*math.pi*hbar_c/Delta:.2f} fm")
    print(f"    = {2*math.pi*hbar_c/Delta*1e-13:.2e} cm")
    print()
    
    # Conversion probability over capsule path:
    sin2_theta_eff = sin_theta**2  # omega_p << m_phi so theta_eff ~ theta_mix
    Delta_L = Delta * l_path / (hbar_c * 1e-13)  # dimensionless
    P_conv = sin2_theta_eff * math.sin(min(Delta_L/2, math.pi/2))**2
    # Actually for Delta*L >> 1, average over oscillations: <sin^2> = 1/2
    print(f"    Delta * L / (hbar*c) = {Delta_L:.2e}")
    if Delta_L > 10:
        P_conv = sin2_theta_eff * 0.5
        print(f"    Delta*L >> 1 -> average sin^2 = 1/2")
    print(f"    P(gamma -> phi) = sin^2(theta) * <sin^2> = {P_conv:.4e}")
    print()

    # phi production rate
    R_phi = N_gamma * P_conv
    print(f"  phi production rate:")
    print(f"    R = N_gamma * P_conv")
    print(f"    = {N_gamma:.0e} * {P_conv:.2e}")
    print(f"    = {R_phi:.2e} phi/s")
    print()

    # Steady-state phi number in capsule
    tau_phi = 6.865e-8  # s
    N_phi = R_phi * tau_phi
    print(f"  Steady-state phi count:")
    print(f"    N_phi = R * tau = {N_phi:.2e}")
    print()

    # phi field energy density
    # Each phi carries energy m_phi, confined to volume ~ xi^3
    # Energy density from N_phi in capsule:
    V_cap = 4/3*math.pi*R_cap**3  # cm^3
    rho_phi = N_phi * m_phi / (V_cap * 6.242e18)  # convert MeV to eV, then... 
    # 1 MeV = 1.602e-13 J
    E_phi_total = N_phi * m_phi * 1.602e-13  # J
    print(f"  Total phi energy in capsule:")
    print(f"    E_phi = {E_phi_total:.2e} J")
    print()

    # === Key comparison ===
    print("="*70)
    print("D. Critical Assessment")
    print("="*70)
    print()
    print("The phi field produced this way is INCOHERENT:")
    print("  - Each phi is produced at a random phase")
    print("  - No coherent buildup occurs")
    print("  - The field averages to ~sqrt(N_phi), not N_phi")
    print()
    print("For coherent effects (Coulomb barrier modification),")
    print("we need a COHERENT phi field, which requires:")
    print("  - Source at precisely omega_phi (dE < Gamma_phi)")
    print("  - Or stimulated emission cascades")
    print()

    # === Stimulated emission (phi laser) ===
    print("="*70)
    print("E. Stimulated Emission: phi Boson Laser")
    print("="*70)
    print()
    print("If spontaneous phi production creates a background,")
    print("stimulated emission can amplify it coherently.")
    print()
    print("Gain condition (like laser threshold):")
    print("  g_stimulated > g_loss")
    print()

    # Stimulated emission cross section
    sigma_stim = math.pi * lambda_c**2 * 1e-26  # cm^2 (resonant)
    # Times branching ratio and coupling
    sigma_stim_eff = sigma_stim * sin_theta**2 * alpha_em
    print(f"  sigma_stimulated ~ pi * lambda^2 * sin^2(theta) * alpha")
    print(f"    = {sigma_stim_eff:.2e} cm^2")
    print()

    # Gain per unit length
    gain = n_N * sigma_stim_eff  # cm^-1
    print(f"  Gain coefficient:")
    print(f"    g = n_N * sigma = {gain:.2e} cm^-1")
    print(f"    Gain over capsule: g*L = {gain * l_path:.2e}")
    print()

    # Loss (phi decay)
    loss = 1.0 / (3e10 * tau_phi)  # cm^-1 (c * tau)
    print(f"  Loss coefficient:")
    print(f"    loss = 1/(c*tau) = {loss:.2e} cm^-1")
    print()
    print(f"  Gain/Loss = {gain/loss:.2e}")
    if gain > loss:
        print(f"  => ABOVE THRESHOLD: phi lasing possible!")
    else:
        print(f"  => Below threshold by factor {loss/gain:.0e}")
    print()

    # === Self-consistent resonance ===
    print("="*70)
    print("F. Self-consistent Resonance (Bootstrap Mechanism)")
    print("="*70)
    print()
    print("During D-T fusion in NIF:")
    print("  1) Fusion reactions produce 14.1 MeV neutrons + 3.5 MeV alphas")
    print("  2) These create secondary nuclear reactions")
    print("  3) Nuclear Bremsstrahlung produces continuous gamma spectrum")
    print()

    # Nuclear Bremsstrahlung from 14.1 MeV neutrons
    # Each n-N collision: ~1e-3 probability of gamma emission
    # Spectrum: quasi-thermal, extending to E_n = 14.1 MeV
    # At 29.65 MeV: exponentially suppressed (E > E_n)
    print("  14.1 MeV neutron Bremsstrahlung at 29.65 MeV:")
    print(f"    Exponentially suppressed (E_phi > E_neutron)")
    print(f"    => This channel is CLOSED")
    print()

    # However: secondary fusion reactions can produce higher energy
    # D + He-3 -> He-4 + p + 18.35 MeV (if T has been produced)
    # p + B-11 -> 3 He-4 + 8.68 MeV
    # None reach 29.65 MeV

    print("  No nuclear reaction in D-T plasma reaches 29.65 MeV in gamma")
    print()

    print("="*70)
    print("G. Conclusion: Practical Path")
    print("="*70)
    print()
    print("Current bottleneck: Creating coherent 29.65 MeV phi field")
    print()
    print("Most promising approach: INVERSE COMPTON + NUCLEAR TARGET")
    print()
    print("Step 1: Generate ~30 MeV gamma beam via ICS (dE/E ~ 0.5%)")
    print("Step 2: Irradiate D-T target")
    print("Step 3: Measure fusion yield vs gamma energy")
    print("        Scan E_gamma from 25 to 35 MeV")
    print("Step 4: Look for resonance enhancement at E = m_phi")
    print()
    print("Even with dE/E = 0.5% (Q_eff ~ 200):")
    print(f"  alpha_eff = alpha_phi * Q = {alpha_phi * 200:.2e}")
    print(f"  Sigma ~ 1.0000 (unmeasurable)")
    print()
    print("But if phi exists, it acts as its own filter:")
    print("  The produced phi bosons have natural width Gamma_phi")
    print("  Each phi modifies the barrier for tau = 69 ns")
    print("  The question is: how many phi in the target at any time?")
    print()

    # More careful: production rate with broad beam
    # Resonant cross section integrated over energy:
    # integral sigma_BW dE = 2*pi^2 * (hbar*c/m_phi)^2 * Gamma_in
    # (Breit-Wigner integrated cross section)
    #
    # For phi production via mixing:
    # Gamma_in(gamma -> phi) ~ sin^2(theta) * (alpha * m_phi^3) / (8*v^2)
    # This is actually the same as Gamma(phi -> gamma gamma) 
    
    # Use the conversion probability approach:
    print(f"  Photon-phi conversion probability = {P_conv:.4e}")
    print(f"  With {N_gamma:.0e} gamma/s on target:")
    print(f"  phi production = {R_phi:.2e} /s")
    print(f"  Steady-state N_phi = {N_phi:.2e}")
    print()
    
    if N_phi > 1:
        print(f"  With N_phi = {N_phi:.0f} phi bosons in the target,")
        print(f"  each modifying the barrier within r < xi = {xi:.1f} fm,")
        V_xi = 4/3*math.pi*(xi*1e-13)**3  # cm^3
        n_phi_local = N_phi / V_cap  # average phi density
        print(f"  average phi density = {n_phi_local:.2e} cm^-3")
        # Compare to nuclear density
        print(f"  nuclear density = {n_N:.0e} cm^-3")
        print(f"  phi/nucleon ratio = {n_phi_local/n_N:.2e}")
        print()
        print(f"  Each phi creates a local potential well of depth:")
        V_well = alpha_phi * hbar_c / xi  # MeV (at r = xi)
        print(f"    V = alpha_phi * hbar*c / xi = {V_well:.4e} MeV = {V_well*1e6:.4f} eV")
        print(f"  -> Negligible compared to Coulomb barrier (~450 keV at r_n)")
    else:
        print(f"  N_phi < 1: insufficient phi production with current technology")
    
    print()
    print("="*70)
    print("H. Technology Requirements Summary")
    print("="*70)
    print()
    print(f"  For Sigma > 1.01: Q_eff > 1e6  -> dE/E < 1e-6  [NRF reachable]")
    print(f"  For Sigma > 1.1:  Q_eff > 1e7  -> dE/E < 1e-7  [R&D needed]")
    print(f"  For Sigma > 5:    Q_eff > 1e8  -> dE/E < 1e-8  [breakthrough needed]")
    print(f"  For Sigma = 278:  Q_eff > 1e9  -> dE/E < 1e-9  [far future]")
    print()
    print("  Alternative: If phi boson is discovered (PADME/NA64),")
    print("  its actual parameters may differ from SFE predictions.")
    print("  A lighter phi (lower m_phi) would be easier to drive.")
    print()
    print("  Scaling: m_phi -> m_phi/k reduces Q_crit by k")
    print("  (because alpha_phi scales as sin^2(theta) ~ v_phi^2 / m_H^2 * m_N^2/v^2)")
    print("  and Gamma ~ m_phi (so Q_vac ~ constant)")

if __name__ == "__main__":
    main()
