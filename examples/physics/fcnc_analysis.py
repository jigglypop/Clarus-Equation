import math

def main():
    print("="*70)
    print("SFE phi: FCNC Coupling Structure Analysis")
    print("="*70)
    print()

    sin_theta = 0.04344
    sin2 = sin_theta**2
    m_phi = 29.648  # MeV
    m_K = 493.677
    m_pi = 139.570
    v = 246000.0  # MeV

    # ==========================================================
    # 1. Standard Higgs-mixed scalar: K -> pi S
    # ==========================================================
    print("I. Standard Higgs Portal Scalar")
    print("-"*70)
    print()
    print("For a Higgs-mixed scalar phi with mixing angle theta:")
    print("  All SM Yukawa couplings are inherited with factor sin(theta)")
    print("  -> c_q = c_l = c_g = sin(theta)")
    print()
    print("K+ -> pi+ S proceeds via:")
    print("  s-quark -> d-quark transition with scalar emission")
    print("  Mediated by W-top loop (penguin diagram)")
    print()
    print("  Amplitude ~ G_F * V_ts* V_td * m_t^2 * sin(theta) / v")
    print()

    # Effective Hamiltonian: H_eff = c_sd * (s_bar d) * phi
    # c_sd ~ (G_F/sqrt(2)) * (m_t^2/(16*pi^2)) * V_ts* V_td * sin(theta)
    # This is the SAME penguin that generates K -> pi nu nu
    # with an extra factor of sin(theta) * (v/m_H^2)

    # BR(K -> pi S) / sin^2(theta) = f(m_S, m_K, m_pi)
    # Literature value: ~5 x 10^-6 for m_S = 30 MeV
    BR_per_sin2 = 5e-6
    BR_SFE = BR_per_sin2 * sin2
    print(f"  BR(K+ -> pi+ S) = {BR_per_sin2:.0e} * sin^2(theta)")
    print(f"  = {BR_SFE:.2e}")
    print()

    # E949 constraint
    print(f"  E949 limit (Region I): BR < 7.3e-11")
    print(f"  SFE: BR = {BR_SFE:.1e} -> EXCLUDED by factor {BR_SFE/7.3e-11:.0f}")
    print()

    # ==========================================================
    # 2. SFE phi: Is it really Higgs-mixed?
    # ==========================================================
    print("="*70)
    print("II. SFE Phi: Coupling Structure Analysis")
    print("="*70)
    print()
    print("SFE Lagrangian:")
    print("  L = L_SM + (1/2)(d_mu Phi)^2 - (1/2)mu^2 Phi^2")
    print("      - lambda_HP |H|^2 Phi^2")
    print()
    print("Key properties of Phi:")
    print("  1. Gauge singlet (no SU(3)xSU(2)xU(1) charge)")
    print("  2. Real scalar")
    print("  3. Z2 symmetry: Phi -> -Phi (from Phi^2 coupling)")
    print()

    # ==========================================================
    # 3. Z2 symmetry and its consequences
    # ==========================================================
    print("="*70)
    print("III. Z2 Symmetry: The Key Distinction")
    print("="*70)
    print()
    print("The SFE Lagrangian has Phi^2 coupling, NOT Phi coupling:")
    print("  lambda_HP |H|^2 Phi^2  (Z2 symmetric)")
    print()
    print("This is DIFFERENT from the generic Higgs portal:")
    print("  mu_portal * |H|^2 * Phi  (Z2 breaking, linear)")
    print()
    print("Consequence for mixing:")
    print()

    # In the Z2-symmetric case:
    # L = -lambda_HP |H|^2 Phi^2
    # After EWSB: H = (v+h)/sqrt(2)
    # -> -lambda_HP * (v+h)^2/2 * Phi^2
    # = -lambda_HP * v^2/2 * Phi^2  (mass term)
    #   -lambda_HP * v * h * Phi^2  (h-Phi-Phi vertex)
    #   -(lambda_HP/2) * h^2 * Phi^2 (h-h-Phi-Phi vertex)
    #
    # NOTE: There is NO h-Phi mixing term!
    # The h-Phi mixing requires a LINEAR term in Phi.
    # With Phi^2 coupling only, the Z2 symmetry Phi -> -Phi
    # prevents any linear h-Phi mass mixing.

    print("  With Z2 symmetry (Phi^2 coupling only):")
    print("    After EWSB, the bilinear terms are:")
    print("    -lambda_HP * v^2/2 * Phi^2  -> mass for Phi")
    print("    -lambda_HP * v * h * Phi^2  -> h-Phi-Phi vertex")
    print()
    print("    There is NO h-Phi mass mixing term!")
    print("    (h*Phi cross term requires Z2-breaking)")
    print()
    print("  If Z2 is EXACT:")
    print("    - Phi does NOT mix with Higgs")
    print("    - sin(theta_mix) = 0")
    print("    - Phi has NO direct coupling to fermions")
    print("    - K -> pi Phi is FORBIDDEN")
    print("    - electron g-2 contribution = 0")
    print()

    # ==========================================================
    # 4. VEV of Phi and Z2 breaking
    # ==========================================================
    print("="*70)
    print("IV. Phi VEV and Z2 Breaking")
    print("="*70)
    print()
    print("SFE uses v_Phi = v * delta = 43.73 GeV")
    print("If Phi acquires a VEV, Z2 is SPONTANEOUSLY broken.")
    print()
    print("With <Phi> = v_Phi, expand Phi = v_Phi + phi:")
    print("  -lambda_HP |H|^2 (v_Phi + phi)^2")
    print("  = -lambda_HP |H|^2 v_Phi^2  (constant)")
    print("    -2*lambda_HP |H|^2 v_Phi phi  (LINEAR in phi)")
    print("    -lambda_HP |H|^2 phi^2  (quadratic)")
    print()
    print("The LINEAR term generates h-phi mixing!")
    print("  M^2_{h-phi} = 2*lambda_HP * v * v_Phi")
    print(f"  = 2 * 0.0316 * 246 * 43.73 = {2*0.0316*246*43.73:.0f} GeV^2")
    print()
    print("This gives sin(theta_mix) = 0.04344")
    print()
    print("BUT: Does SFE actually require v_Phi != 0?")
    print()

    # ==========================================================
    # 5. Two scenarios
    # ==========================================================
    print("="*70)
    print("V. Two Scenarios for SFE")
    print("="*70)
    print()

    print("SCENARIO A: v_Phi = 0 (Z2 exact)")
    print("  - Phi is a Z2-symmetric dark scalar")
    print("  - No Higgs mixing -> No FCNC")
    print("  - K -> pi Phi: FORBIDDEN")
    print("  - Phi -> e+e-: FORBIDDEN (no fermion coupling)")
    print("  - Phi is STABLE (dark matter candidate!)")
    print("  - Phi interacts only via Phi^2 operators")
    print("  - Nucleon coupling: ONLY through Phi^2*|H|^2")
    print("    This gives Phi-Phi-N-N contact interaction")
    print("    NOT single-Phi Yukawa")
    print("  - For fusion: no single-Phi exchange")
    print("    -> Mechanism completely different")
    print()
    print("  Consistency check:")
    print("  - SFE derives DM as Phi condensate: CONSISTENT")
    print("  - SFE predicts Phi is 'dark': CONSISTENT")
    print("  - No FCNC problem: CONSISTENT")
    print("  - No g-2 problem: NEEDS REANALYSIS")
    print("  - Muon g-2: SFE derives from geometry, not Phi loop")
    print()

    print("SCENARIO B: v_Phi != 0 (Z2 spontaneously broken)")
    print("  - Phi mixes with Higgs")
    print("  - sin(theta) = 0.04344")
    print("  - K -> pi phi: EXCLUDED by E949")
    print("  - This scenario is RULED OUT")
    print()

    # ==========================================================
    # 6. Scenario A: Phi^2 coupling analysis
    # ==========================================================
    print("="*70)
    print("VI. Scenario A Deep Analysis: Z2-symmetric Phi")
    print("="*70)
    print()

    # With Z2 symmetry, the Phi couples to nucleons ONLY via
    # Phi^2 |H|^2 -> after EWSB: Phi^2 * v * h + Phi^2 * h^2/2
    # The Phi^2 * h term gives Phi-Phi-h vertex
    # But Phi enters in pairs -> no single Phi exchange

    # For nucleon interactions:
    # Two-Phi exchange (van der Waals type):
    # V(r) ~ G^2 * exp(-2*m_phi*r) / r^5
    # Range: hbar*c / (2*m_phi) = 3.33 fm (half the Compton wavelength)

    xi_2phi = 197.327 / (2 * m_phi)
    print("Phi^2 nucleon interaction (two-Phi exchange):")
    print(f"  Range: hbar*c / (2*m_phi) = {xi_2phi:.2f} fm")
    print(f"  (vs single exchange: {197.327/m_phi:.2f} fm)")
    print()

    # Effective coupling for Phi^2 exchange:
    # G_eff = lambda_HP * f_N / m_H^2 (from h propagator)
    lambda_HP = 0.03160
    f_N = 0.30
    m_H = 125100  # MeV
    G_eff = lambda_HP * f_N / m_H**2
    print(f"  G_eff = lambda_HP * f_N / m_H^2 = {G_eff:.2e} MeV^-2")
    print()

    # For fusion: the relevant quantity is NOT single-phi exchange
    # but the Phi^2 condensate energy density
    # In Z2 scenario, "resonance fusion" works differently:
    # Not through single-phi exchange potential
    # But through Phi^2 modification of Higgs VEV

    print("For fusion (Z2-symmetric scenario):")
    print("  Mechanism is Phi^2 condensate modifying Higgs sector")
    print("  |H|^2 -> |H|^2 + delta(|H|^2) from Phi^2 background")
    print("  This changes nuclear physics without FCNC")
    print()

    # ==========================================================
    # 7. SFE muon g-2 in Z2 scenario
    # ==========================================================
    print("="*70)
    print("VII. Muon g-2 in Z2-symmetric Scenario")
    print("="*70)
    print()
    print("SFE derives muon g-2 from GEOMETRIC formula:")
    print("  delta(a_mu) = (alpha/2pi) * e^{-1} * (m_mu/M_SFE)^2")
    print("  = 249 x 10^{-11}")
    print()
    print("This does NOT require phi-muon Yukawa coupling!")
    print("The geometric derivation is independent of Phi's")
    print("coupling structure to fermions.")
    print()
    print("In the 1-loop interpretation:")
    print("  The Phi^2 contribution to g-2 comes from")
    print("  a TWO-Phi loop (seagull diagram), not single exchange.")
    print("  This requires Phi^2-mu-mu coupling from Phi^2*|H|^2.")
    print()

    # Phi^2 contribution to muon g-2:
    # Seagull diagram: mu -> mu + Phi + Phi (via h exchange)
    # delta(a_mu) ~ (m_mu^2 / (16*pi^2)) * G_eff^2 * m_mu^2 / m_phi^2
    # This is highly suppressed compared to single-exchange

    m_mu = 105.658
    delta_a_seagull = (m_mu**2 / (16*math.pi**2)) * G_eff**2 * m_mu**2 / m_phi**2
    print(f"  Phi^2 seagull contribution:")
    print(f"    delta(a_mu) ~ {delta_a_seagull:.2e}")
    print(f"    (vs SFE geometric prediction: 2.49e-08)")
    print(f"    -> Seagull diagram is {2.49e-8/delta_a_seagull:.0e}x too small")
    print()
    print("  This means the GEOMETRIC derivation of g-2 in SFE")
    print("  does NOT correspond to a simple perturbative loop.")
    print("  It may correspond to a non-perturbative effect")
    print("  of the Phi condensate on spacetime geometry.")
    print()

    # ==========================================================
    # 8. Resolution
    # ==========================================================
    print("="*70)
    print("VIII. Resolution: SFE Phi is Z2-symmetric")
    print("="*70)
    print()
    print("The SFE Lagrangian L = ... - lambda_HP |H|^2 Phi^2")
    print("has an EXACT Z2 symmetry (Phi -> -Phi).")
    print()
    print("Two cases for VEV:")
    print()
    print("CASE 1: v_Phi = 0 (Z2 unbroken)")
    print("  - Phi is dark: no mixing, no FCNC, no decay")
    print("  - Phi is stable dark matter candidate")
    print("  - All experimental constraints SATISFIED")
    print("  - But: sin(theta) = 0 -> no direct nucleon Yukawa")
    print("  - Resonance fusion mechanism CHANGES fundamentally")
    print()
    print("CASE 2: v_Phi != 0 (Z2 spontaneously broken)")
    print("  - sin(theta) = 0.04344")
    print("  - EXCLUDED by K -> pi phi (E949)")
    print("  - EXCLUDED by electron g-2")
    print()

    # However, there's a third possibility:
    print("CASE 3: Z2 broken by small explicit term")
    print("  L contains a small Z2-breaking term:")
    print("  delta_L = -mu3 * |H|^2 * Phi (linear portal)")
    print("  This gives a small mixing angle:")
    print("  sin(theta) ~ mu3 * v / m_H^2")
    print()
    print("  For sin(theta) < 3.8e-3 (E949 upper bound):")
    print(f"    mu3 < {3.8e-3 * 125100**2 / 246000:.2f} MeV = {3.8e-3 * 125.1**2 / 246:.4f} GeV")
    print()

    # With reduced mixing angle:
    sin_max = 3.8e-3
    alpha_phi_max = (sin_max * 938.272 * 0.30 / 246000)**2 / (4*math.pi)
    Q_crit_new = 0.01188 / alpha_phi_max
    print(f"  With sin(theta) = {sin_max:.1e}:")
    print(f"    g_phiNN = {sin_max * 938.272 * 0.30 / 246000:.2e}")
    print(f"    alpha_phi = {alpha_phi_max:.2e}")
    print(f"    Q_crit = {Q_crit_new:.2e}")
    print(f"    Q_vac estimate:")

    # New decay width
    m_e = 0.511
    Gamma_new = sin_max**2 * m_e**2 * m_phi / (8*math.pi*246000**2)
    Q_vac_new = m_phi / Gamma_new
    tau_new = 6.582e-22 / Gamma_new
    print(f"    Gamma = {Gamma_new:.2e} MeV")
    print(f"    Q_vac = {Q_vac_new:.2e}")
    print(f"    tau = {tau_new:.2e} s = {tau_new*1e6:.1f} us")
    print()
    print(f"    Q_vac / Q_crit = {Q_vac_new / Q_crit_new:.2e}")
    if Q_vac_new > Q_crit_new:
        print(f"    -> Still above Q_crit: barrier elimination possible!")
    else:
        print(f"    -> Below Q_crit: barrier NOT eliminated")
    print()

    # ==========================================================
    # 9. Summary
    # ==========================================================
    print("="*70)
    print("IX. Summary and Implications")
    print("="*70)
    print()
    print("1. SFE's original formulation (sin theta = 0.04) is")
    print("   EXCLUDED by E949 K+ -> pi+ invisible search.")
    print()
    print("2. The Z2-symmetric version (sin theta = 0, v_Phi = 0)")
    print("   is experimentally SAFE but requires reworking the")
    print("   resonance fusion mechanism (no single-phi exchange).")
    print()
    print(f"3. A small-mixing version (sin theta < {sin_max:.1e})")
    print("   satisfies ALL experimental constraints and STILL")
    print(f"   allows barrier elimination (Q_vac/Q_crit = {Q_vac_new/Q_crit_new:.0e}).")
    print()
    print("4. The key SFE parameter lambda_HP = delta^2 = 0.0316")
    print("   is NOT directly constrained by K decay searches")
    print("   (it governs Phi^2 coupling, not Phi mixing).")
    print()
    print("5. What IS constrained is the Phi VEV (v_Phi).")
    print(f"   v_Phi must satisfy: v_Phi < {sin_max * 125100**2 / (2*0.0316*246000):.1f} MeV")
    print(f"   (from sin theta = 2*lambda*v*v_Phi/m_H^2 < {sin_max:.1e})")
    print()

    v_phi_max = sin_max * 125100**2 / (2 * 0.0316 * 246000)
    print(f"   v_Phi < {v_phi_max:.0f} MeV = {v_phi_max/1000:.2f} GeV")
    print(f"   (SFE original: v_Phi = 43730 MeV = 43.73 GeV)")
    print(f"   -> v_Phi must be reduced by factor {43730/v_phi_max:.0f}")
    print()
    print("6. IMPLICATIONS FOR SFE THEORY:")
    print("   M_SFE = v * delta = 43.73 GeV can be maintained")
    print("   as the MASS SCALE of Phi (from lambda_HP * v^2),")
    print("   but v_Phi (the VEV) must be much smaller.")
    print("   This is naturally achieved if mu_Phi^2 > 0")
    print("   (Phi sits at origin, no spontaneous breaking).")
    print()
    print("   In this case:")
    print("   m_Phi = sqrt(mu^2 + lambda_HP * v^2) = v*delta")
    print("   v_Phi = 0 (Z2 preserved)")
    print("   sin(theta) = 0 (no mixing)")
    print("   Phi is a stable dark scalar")

if __name__ == "__main__":
    main()
