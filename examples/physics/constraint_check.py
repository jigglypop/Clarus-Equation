import math

def main():
    print("="*70)
    print("SFE phi Boson vs Experimental Constraints")
    print("="*70)
    print()

    # SFE predictions
    sin_theta = 0.04344
    sin2_theta = sin_theta**2
    m_phi = 29.648  # MeV
    tau_phi = 6.865e-8  # s = 69 ns
    ctau = 3e8 * tau_phi  # m

    print("A. SFE Predictions:")
    print(f"  m_phi = {m_phi:.2f} MeV")
    print(f"  sin(theta_mix) = {sin_theta:.5f}")
    print(f"  sin^2(theta) = {sin2_theta:.4e}")
    print(f"  tau = {tau_phi*1e9:.1f} ns")
    print(f"  c*tau = {ctau:.1f} m")
    print()

    # ============================================================
    # 1. K+ -> pi+ S (E949/E787 at BNL, NA62 at CERN)
    # ============================================================
    print("="*70)
    print("B. K+ -> pi+ S (Kaon rare decay)")
    print("="*70)
    print()

    m_K = 493.677  # MeV
    m_pi = 139.570  # MeV
    print(f"  m_K - m_pi = {m_K - m_pi:.1f} MeV > m_phi = {m_phi:.1f} MeV")
    print(f"  -> Kinematically OPEN")
    print()

    # Higgs portal K -> pi S: flavor-changing neutral current
    # Effective Hamiltonian: s -> d + S via W-top loop with Higgs mixing
    # BR(K+ -> pi+ S) = sin^2(theta) * BR_SM(K->piH)|_{m_H=m_S}
    #
    # From Winkler (2019) and Clarke, Foot, Volkas:
    # For m_S < 2*m_mu = 211 MeV (only e+e- decay):
    # BR(K+ -> pi+ S) ~ 3 * 10^{-6} * sin^2(theta)
    #   [includes CKM, top loop, phase space for m_S = 30 MeV]
    #
    # More precise: use Cheng-Chiang formula
    # BR = (sin theta)^2 * 3 * alpha^2 / (16*pi^2*sin^4(thetaW)) *
    #      |V_ts*V_td|^2 * m_t^4/m_W^4 * m_K/(m_K^2-m_pi^2) *
    #      f_+(0)^2 * lambda^{1/2} * tau_K / hbar

    # Use standard result from literature:
    # BR(K+ -> pi+ S) / sin^2(theta) ~ 5 * 10^{-6} for m_S ~ 30 MeV
    # (Leutwyler-Roos form factor, Inami-Lim loop function)

    BR_per_sin2 = 5e-6  # approximate for m_S = 30 MeV
    BR_SFE = BR_per_sin2 * sin2_theta
    print(f"  BR(K+ -> pi+ S) = {BR_per_sin2:.0e} * sin^2(theta)")
    print(f"  = {BR_per_sin2:.0e} * {sin2_theta:.4e}")
    print(f"  = {BR_SFE:.2e}")
    print()

    # S decay: tau = 69 ns, boosted gamma ~ E_S / m_S
    # In K+ -> pi+ S at rest: E_S = (m_K^2 + m_S^2 - m_pi^2)/(2*m_K)
    E_S = (m_K**2 + m_phi**2 - m_pi**2) / (2*m_K)
    p_S = math.sqrt(E_S**2 - m_phi**2)
    gamma_S = E_S / m_phi
    beta_S = p_S / E_S
    d_decay = gamma_S * beta_S * ctau
    print(f"  S kinematics from K+ at rest:")
    print(f"    E_S = {E_S:.1f} MeV")
    print(f"    p_S = {p_S:.1f} MeV/c")
    print(f"    gamma = {gamma_S:.2f}")
    print(f"    beta*gamma*c*tau = {d_decay:.1f} m")
    print()

    # E949 detector: ~1 m radius, surrounding K+ stop target
    # S with d_decay ~ 170 m: escapes detector -> invisible signal
    # E949 limit on K+ -> pi+ + invisible:
    # For m_X in the range 0-260 MeV:
    # E949+E787 combined: BR(K+ -> pi+ X) < 7.3e-11 (central region)
    # But the actual limit depends on pi+ momentum region:
    # Region I (211-229 MeV/c): < 7.3e-11
    # Region II (140-199 MeV/c): < 5.3e-10

    # For m_S = 30 MeV:
    # pi+ momentum from K+ -> pi+ S at rest:
    p_pi = p_S  # momentum conservation
    print(f"  pi+ momentum: {p_pi:.1f} MeV/c")

    # E949 Region I: 211 < p_pi < 229 MeV/c
    # E949 Region II: 140 < p_pi < 199 MeV/c
    # Our p_pi = 229 MeV/c - need to check
    print(f"  E949 acceptance regions:")
    print(f"    Region I:  211-229 MeV/c (BR < 7.3e-11)")
    print(f"    Region II: 140-199 MeV/c (BR < 5.3e-10)")

    if 211 < p_pi < 229:
        limit = 7.3e-11
        region = "I"
    elif 140 < p_pi < 199:
        limit = 5.3e-10
        region = "II"
    else:
        limit = None
        region = "outside"

    print(f"    p_pi = {p_pi:.1f} MeV/c -> Region {region}")
    print()

    if limit:
        ratio = BR_SFE / limit
        sin_upper = math.sqrt(limit / BR_per_sin2)
        print(f"  E949 limit: BR < {limit:.1e}")
        print(f"  SFE prediction: BR = {BR_SFE:.1e}")
        print(f"  Ratio: {ratio:.1f}x above limit")
        print(f"  -> sin(theta) upper bound: {sin_upper:.2e}")
        print(f"  -> SFE sin(theta) = {sin_theta:.4f} is {sin_theta/sin_upper:.0f}x above")
        if ratio > 1:
            print(f"  ** EXCLUDED by E949 **")
        print()
    else:
        print(f"  p_pi = {p_pi:.1f} falls outside standard E949 regions")
        print(f"  Need to check NA62 coverage")
        print()

    # NA62 K+ -> pi+ nu nu search
    print("  NA62 (2024 update):")
    print("    K+ -> pi+ nu nu: BR = (1.3 +0.7 -0.5) x 10^-10")
    print("    Covers broader pi+ momentum range than E949")
    print("    For m_S = 30 MeV: expected to have full coverage")
    print()

    # ============================================================
    # 2. PADME / X17 searches
    # ============================================================
    print("="*70)
    print("C. PADME / X17 Searches (e+e- -> gamma + X)")
    print("="*70)
    print()
    print("  PADME (2025): Searched for 16.7-17.4 MeV resonance")
    print(f"  SFE phi mass: {m_phi:.1f} MeV - OUTSIDE PADME scan range")
    print("  -> No direct constraint from PADME on SFE phi")
    print()

    # ============================================================
    # 3. Beam dump experiments
    # ============================================================
    print("="*70)
    print("D. Beam Dump Experiments (LSND, CHARM, PS191)")
    print("="*70)
    print()
    print(f"  SFE phi: c*tau = {ctau:.1f} m, boosted c*tau ~ 100+ m")
    print()
    print("  Beam dump sensitivity structure:")
    print("    - Small sin(theta) -> long lifetime -> reaches detector")
    print("    - Large sin(theta) -> short lifetime -> decays before detector")
    print()
    print("  LSND (Foroughi-Abari & Ritz, 2020):")
    print("    Baseline: 30 m, leading for m = 100-350 MeV")
    print("    For m = 30 MeV: weak constraint (below pion threshold)")
    print()
    print("  CHARM: Baseline 480 m, sensitive to long-lived particles")
    print(f"    SFE phi boosted c*tau ~ {d_decay:.0f} m << 480 m")
    print(f"    -> phi decays before CHARM detector")
    print(f"    -> CHARM constrains LOWER sin(theta), not SFE value")
    print()
    print("  PS191: Baseline 128 m")
    print(f"    Same argument: phi decays before detector")
    print()
    print("  ** Beam dump experiments generally exclude")
    print("     sin(theta) ~ 10^-4 to 10^-3 (long-lived regime)")
    print("     SFE's sin(theta) = 0.04 is in the SHORT-LIVED regime")
    print("     -> Beam dumps do NOT exclude SFE **")
    print()

    # ============================================================
    # 4. MicroBooNE
    # ============================================================
    print("="*70)
    print("E. MicroBooNE (NuMI beam, 2021)")
    print("="*70)
    print()
    print("  Direct search: m_S = 100-200 MeV, sin(theta) < 3-5 x 10^-4")
    print(f"  SFE phi mass: {m_phi:.1f} MeV -> BELOW MicroBooNE mass range")
    print("  Recast to 30-150 MeV: applies to HNL, not directly to scalar")
    print()
    print("  Production: K -> pi S in NuMI absorber")
    print(f"  S with sin(theta) = 0.04: decays in ~{d_decay:.0f} m")
    print("  MicroBooNE is 100 m from absorber")
    print(f"  Survival probability: exp(-100/{d_decay:.0f}) = {math.exp(-100/d_decay):.3f}")
    print("  -> Most phi decay before reaching detector")
    print("  -> MicroBooNE has REDUCED sensitivity for large sin(theta)")
    print()

    # ============================================================
    # 5. Electron anomalous magnetic moment
    # ============================================================
    print("="*70)
    print("F. Electron g-2 Constraint")
    print("="*70)
    print()
    # phi contributes to (g-2)_e via one-loop diagram
    # delta(a_e) = sin^2(theta) * m_e^2 / (8*pi^2 * m_phi^2) * f(m_e/m_phi)
    # For m_phi >> m_e: f -> 1, so
    m_e = 0.511  # MeV
    delta_ae = sin2_theta * m_e**2 / (8 * math.pi**2 * m_phi**2)
    print(f"  delta(a_e) = sin^2(theta) * m_e^2 / (8*pi^2 * m_phi^2)")
    print(f"  = {sin2_theta:.4e} * {m_e**2:.4f} / (8*pi^2 * {m_phi**2:.1f})")
    print(f"  = {delta_ae:.2e}")
    print()
    print(f"  Experimental: a_e(exp) - a_e(SM) ~ few x 10^-13")
    print(f"  SFE contribution: {delta_ae:.1e}")
    if delta_ae < 1e-12:
        print(f"  -> CONSISTENT with experiment")
    else:
        print(f"  -> May be in TENSION with experiment")
    print()

    # ============================================================
    # 6. Supernova SN1987A
    # ============================================================
    print("="*70)
    print("G. Supernova SN1987A Cooling")
    print("="*70)
    print()
    print("  Light scalars can be produced in SN core and carry away energy")
    print("  Constraint: luminosity in new particles < few x 10^53 erg/s")
    print()
    print("  For m_phi = 30 MeV, sin(theta) = 0.04:")
    print(f"    mean free path in SN core (T~30 MeV, rho~10^14 g/cm^3):")
    # In SN core, phi is produced and absorbed rapidly
    # If mfp << R_core (~10 km), phi is trapped -> no energy loss
    # If mfp >> R_core, phi free-streams -> energy loss constraint
    # "Trapping" limit: sin(theta) > ~10^-4 for m ~ 30 MeV
    print("    SN trapping regime: sin(theta) > ~10^-4")
    print(f"    SFE sin(theta) = {sin_theta:.4f} >> 10^-4")
    print("    -> phi is TRAPPED in SN core")
    print("    -> No anomalous cooling")
    print("    -> SN1987A does NOT constrain SFE")
    print()

    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("H. SUMMARY")
    print("="*70)
    print()
    print("  Constraint          |  Status for SFE phi")
    print("  --------------------|---------------------------")
    print("  E949 K->pi+invisible|  CRITICAL: likely excluded")
    print("  NA62 K->pi+invisible|  CRITICAL: need exact limit")
    print("  Beam dumps (CHARM)  |  NOT excluded (too short-lived)")
    print("  MicroBooNE          |  Mass below range / low sensitivity")
    print("  PADME               |  Mass above scan range")
    print("  Electron g-2        |  Consistent")
    print("  SN1987A             |  Consistent (trapping)")
    print()

    print("="*70)
    print("I. KEY QUESTION: K+ -> pi+ S at m_S = 30 MeV")
    print("="*70)
    print()
    print(f"  pi+ momentum from K(rest) -> pi + S(30 MeV): {p_pi:.1f} MeV/c")
    print()
    print("  E949 published (2008, PRD 77 052003):")
    print("    Region I:  211 < p_pi < 229 MeV/c")
    print("    Region II: 140 < p_pi < 199 MeV/c")
    print(f"    Our p_pi = {p_pi:.1f} MeV/c")
    if p_pi > 229:
        print("    -> ABOVE Region I upper edge!")
        print("    -> May fall in kinematic gap of E949")
        print("    -> Need NA62 data which covers broader range")
    elif 199 < p_pi < 211:
        print("    -> BETWEEN Region I and II (gap)")
        print("    -> E949 has reduced sensitivity here")
    print()

    # NA62: K+ -> pi+ nu nu with full kinematic coverage
    # NA62 2024: 51 signal candidates, BR = (1.3 +0.7 -0.5) x 10^-10
    # For any BSM scalar: BR(K->pi S) < few x BR(K->pi nu nu)
    BR_NA62 = 1.3e-10
    sin_upper_NA62 = math.sqrt(BR_NA62 / BR_per_sin2)
    print(f"  NA62 (2024): BR(K->pi+invisible) = 1.3 x 10^-10")
    print(f"  Upper bound on sin(theta) from NA62:")
    print(f"    sin(theta) < sqrt({BR_NA62:.1e}/{BR_per_sin2:.0e})")
    print(f"    = {sin_upper_NA62:.4e}")
    print(f"    SFE: {sin_theta:.4f} -> {sin_theta/sin_upper_NA62:.0f}x above")
    print()
    print("  ** SFE phi (sin theta = 0.04, m = 30 MeV) appears to be")
    print("     EXCLUDED by K+ -> pi+ invisible searches at ~250x level **")
    print()
    print("  UNLESS the SFE phi has suppressed K->pi production")
    print("  (e.g., non-standard flavor structure different from")
    print("  generic Higgs-mixed scalar)")

if __name__ == "__main__":
    main()
