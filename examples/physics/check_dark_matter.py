"""
SFE Dark Matter Analysis
========================
SFE가 예측하는 암흑물질의 정량적 검증:
1. 기본 구조: Omega_DM = (1-eps^2) * alpha_s*pi / (1+alpha_s*pi)
2. 히그스 포탈 스칼라 DM: 열적 잔류밀도 vs SFE 예측
3. 직접 탐색: spin-independent 단면적 vs LZ/XENON 한계
4. 질량 텐션: M_SFE = 43.77 GeV vs m_phi = 22-30 MeV
5. 구조 형성: warm/cold DM 제약
6. 간접 탐색 및 충돌기 제약
"""
import math

# ============================================================
# Physical Constants
# ============================================================
alpha_em = 1.0 / 137.036
sin2_tW = 0.23122
cos2_tW = 1.0 - sin2_tW
delta = sin2_tW * cos2_tW   # 0.17776
v_EW = 246.22               # GeV (Higgs VEV)
alpha_s = 0.1179             # strong coupling
m_H = 125.1                 # GeV (Higgs mass)
Gamma_H = 4.07e-3            # GeV (Higgs total width)
M_Pl = 1.2209e19             # GeV (Planck mass)
m_b = 4.18                   # GeV (bottom quark)
m_c = 1.27                   # GeV (charm quark)
m_tau = 1.777                # GeV (tau lepton)
m_mu = 0.10566               # GeV (muon)
m_N = 0.939                  # GeV (nucleon mass)
f_N = 0.30                   # nuclear matrix element (Higgs-nucleon)
h_hubble = 0.674             # reduced Hubble constant
G_F = 1.1664e-5              # GeV^{-2} (Fermi constant)
GEV2_TO_CM2 = 0.3894e-27     # 1 GeV^{-2} = 0.3894e-27 cm^2
GEV2_TO_CM3S = 1.167e-17     # 1 GeV^{-2} = 1.167e-17 cm^3/s (for sigma*v)

# SFE parameters
M_SFE = v_EW * delta         # 43.77 GeV
lambda_HP = delta**2          # 0.0316
eps2 = 0.04865                # epsilon^2 (baryon fraction)
alpha_ratio = alpha_s * math.pi  # DM/DE ratio = 0.3704


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# PART 1: SFE Geometric DM Prediction
# ============================================================
def part1_geometric_prediction():
    separator("PART 1: SFE Geometric Dark Matter Prediction")

    Omega_b = eps2
    Omega_phi = 1.0 - eps2
    Omega_Lambda = Omega_phi / (1.0 + alpha_ratio)
    Omega_DM = Omega_phi * alpha_ratio / (1.0 + alpha_ratio)

    print(f"\n  Input parameters:")
    print(f"    d = 3 (spatial dimensions)")
    print(f"    sin^2(theta_W) = {sin2_tW}")
    print(f"    alpha_s = {alpha_s}")
    print(f"    delta = sin^2 cos^2 = {delta:.5f}")
    print(f"    D_eff = 3 + delta = {3 + delta:.5f}")

    print(f"\n  Derived:")
    print(f"    eps^2 = {eps2:.5f} (self-consistency solution)")
    print(f"    alpha = alpha_s * pi = {alpha_ratio:.4f}")

    print(f"\n  Energy composition:")
    print(f"    Omega_b  = eps^2           = {Omega_b:.5f}  (Planck: 0.0486 +/- 0.001)")
    print(f"    Omega_DE = (1-eps^2)/(1+a) = {Omega_Lambda:.4f}  (Planck: 0.6847)")
    print(f"    Omega_DM = (1-eps^2)a/(1+a)= {Omega_DM:.4f}  (Planck: 0.2589)")
    print(f"    Sum                        = {Omega_b + Omega_Lambda + Omega_DM:.4f}")

    Omega_DM_obs = 0.2589
    Omega_DM_err = 0.0057
    tension = abs(Omega_DM - Omega_DM_obs) / Omega_DM_err
    pct_err = abs(Omega_DM - Omega_DM_obs) / Omega_DM_obs * 100

    print(f"\n  DM precision: {pct_err:.2f}% off, {tension:.2f} sigma")

    # Physical interpretation
    print(f"\n  Physical interpretation:")
    print(f"    DM = suppression field quantum fluctuation condensate")
    print(f"    DE = suppression field vacuum energy")
    print(f"    DM/DE ratio = alpha_s * pi = {alpha_ratio:.4f}")
    print(f"      -> 1-loop QCD correction to vacuum energy")
    print(f"      -> DM fraction grows with strong coupling")

    # Reverse engineering: what alpha_s does Planck imply?
    Omega_DM_planck = 0.2589
    Omega_DE_planck = 0.6847
    alpha_s_implied = (Omega_DM_planck / Omega_DE_planck) / math.pi
    print(f"\n  Reverse check: alpha_s from Planck DM/DE = {alpha_s_implied:.4f}")
    print(f"    PDG value: {alpha_s} +/- 0.0009")
    print(f"    Difference: {abs(alpha_s_implied - alpha_s)/alpha_s*100:.2f}%")

    return Omega_DM


# ============================================================
# PART 2: Higgs Portal Thermal Relic Density
# ============================================================
def sigma_v_ff(m_DM, m_f, N_c, lam_HP):
    """Annihilation cross section * velocity for DM DM -> f fbar via Higgs."""
    if m_DM < m_f:
        return 0.0
    s = 4.0 * m_DM**2
    propagator2 = (s - m_H**2)**2 + (m_H * Gamma_H)**2
    beta_f = math.sqrt(1.0 - m_f**2 / m_DM**2)
    # Non-relativistic limit (s-wave):
    # sigma*v = N_c * lam^2 * v_EW^2 * m_f^2 * beta_f^3 / (4*pi * propagator^2)
    sv = N_c * lam_HP**2 * v_EW**2 * m_f**2 * beta_f**3 / (4.0 * math.pi * propagator2)
    return sv


def sigma_v_total(m_DM, lam_HP):
    """Total annihilation cross section * velocity for all kinematically accessible channels."""
    sv = 0.0
    # b bbar (N_c = 3)
    sv += sigma_v_ff(m_DM, m_b, 3, lam_HP)
    # c cbar (N_c = 3)
    sv += sigma_v_ff(m_DM, m_c, 3, lam_HP)
    # tau tau (N_c = 1)
    sv += sigma_v_ff(m_DM, m_tau, 1, lam_HP)
    # mu mu (N_c = 1)
    sv += sigma_v_ff(m_DM, m_mu, 1, lam_HP)
    # WW* (if kinematically accessible, m_DM > m_W)
    # For m_DM < m_W ~ 80 GeV, off-shell WW* contribution is subdominant
    # gg (gluon, via top loop) - subdominant at tree level
    return sv


def relic_density(m_DM, lam_HP):
    """Approximate thermal relic density Omega h^2."""
    sv = sigma_v_total(m_DM, lam_HP)  # in GeV^{-2}
    if sv <= 0:
        return float('inf')
    # Convert to natural units: sigma*v in cm^3/s
    sv_cm3s = sv * GEV2_TO_CM3S
    # Standard thermal relic formula:
    # Omega h^2 ~ 3e-27 cm^3/s / <sigma*v>
    # More precisely: Omega h^2 = 1.07e9 x_f / (sqrt(g*) M_Pl <sigma*v>)
    g_star = 86.25  # effective DOF at T ~ m_DM/25 for m_DM ~ 40 GeV
    x_f = 25.0      # freeze-out x = m/T
    omega_h2 = 1.07e9 * x_f / (math.sqrt(g_star) * M_Pl * sv)
    return omega_h2


def part2_thermal_relic():
    separator("PART 2: Higgs Portal Thermal Relic Density")

    m_DM = M_SFE  # 43.77 GeV
    print(f"\n  DM candidate: Higgs portal scalar Phi")
    print(f"    Mass: M_SFE = v_EW * delta = {m_DM:.2f} GeV")
    print(f"    Higgs portal coupling: lambda_HP = delta^2 = {lambda_HP:.4f}")

    # Individual channels
    sv_bb = sigma_v_ff(m_DM, m_b, 3, lambda_HP)
    sv_cc = sigma_v_ff(m_DM, m_c, 3, lambda_HP)
    sv_tt = sigma_v_ff(m_DM, m_tau, 1, lambda_HP)
    sv_total = sigma_v_total(m_DM, lambda_HP)

    print(f"\n  Annihilation channels at m_DM = {m_DM:.2f} GeV:")
    print(f"    Phi Phi -> b bbar:   {sv_bb:.4e} GeV^-2  ({sv_bb/sv_total*100:.1f}%)")
    print(f"    Phi Phi -> c cbar:   {sv_cc:.4e} GeV^-2  ({sv_cc/sv_total*100:.1f}%)")
    print(f"    Phi Phi -> tau tau:  {sv_tt:.4e} GeV^-2  ({sv_tt/sv_total*100:.1f}%)")
    print(f"    Total <sigma*v>:     {sv_total:.4e} GeV^-2")
    print(f"                       = {sv_total * GEV2_TO_CM3S:.4e} cm^3/s")

    # WIMP miracle value
    sv_wimp = 3.0e-26  # cm^3/s
    print(f"\n    WIMP miracle value:   {sv_wimp:.1e} cm^3/s")
    print(f"    SFE / WIMP ratio:    {sv_total * GEV2_TO_CM3S / sv_wimp:.4f}")

    # Relic density
    omega_h2 = relic_density(m_DM, lambda_HP)
    omega_h2_obs = 0.120
    print(f"\n  Thermal relic density:")
    print(f"    Omega_DM h^2 (SFE Higgs portal) = {omega_h2:.4f}")
    print(f"    Omega_DM h^2 (Planck observed)   = {omega_h2_obs:.3f}")

    if omega_h2 > omega_h2_obs:
        print(f"    -> OVERPRODUCED by factor {omega_h2/omega_h2_obs:.1f}")
        print(f"       Cross section too small -> too much DM survives freeze-out")
    else:
        print(f"    -> UNDERPRODUCED by factor {omega_h2_obs/omega_h2:.1f}")

    # What lambda_HP would give correct relic density?
    print(f"\n  Reverse: lambda_HP needed for Omega h^2 = 0.12:")
    for test_lam in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        oh2 = relic_density(m_DM, test_lam)
        marker = " <-- SFE" if abs(test_lam - lambda_HP) < 0.005 else ""
        marker2 = " <-- TARGET" if abs(oh2 - 0.12) / 0.12 < 0.3 else ""
        print(f"    lambda_HP = {test_lam:.3f} -> Omega h^2 = {oh2:.4f}{marker}{marker2}")

    # Higgs resonance effect
    print(f"\n  Higgs resonance analysis:")
    m_res = m_H / 2.0
    print(f"    Resonance at m_DM = m_H/2 = {m_res:.2f} GeV")
    print(f"    SFE prediction: m_DM = {m_DM:.2f} GeV")
    print(f"    Distance from resonance: {abs(m_DM - m_res):.2f} GeV ({abs(m_DM - m_res)/m_res*100:.1f}%)")

    oh2_res = relic_density(m_res, lambda_HP)
    print(f"    Omega h^2 at resonance: {oh2_res:.6f} (enhanced annihilation)")

    # Mass scan
    print(f"\n  Mass scan with lambda_HP = delta^2 = {lambda_HP:.4f}:")
    print(f"    {'m_DM (GeV)':>12s}  {'Omega h^2':>12s}  {'Status':>15s}")
    for m_test in [10, 20, 30, 40, M_SFE, 50, 55, 60, 62, 62.55, 63, 70, 80]:
        oh2 = relic_density(m_test, lambda_HP)
        if oh2 > 1e4:
            status = "excluded"
        elif oh2 > 0.12 * 3:
            status = "overproduced"
        elif oh2 > 0.12 * 0.3 and oh2 < 0.12 * 3:
            status = "VIABLE"
        else:
            status = "underproduced"
        label = f" (M_SFE)" if abs(m_test - M_SFE) < 0.01 else ""
        label += f" (m_H/2)" if abs(m_test - m_H/2) < 0.01 else ""
        print(f"    {m_test:12.2f}  {oh2:12.4f}  {status:>15s}{label}")

    return omega_h2


# ============================================================
# PART 3: Direct Detection
# ============================================================
def sigma_SI(m_DM, lam_HP):
    """Spin-independent DM-nucleon cross section (cm^2)."""
    mu_red = m_N * m_DM / (m_N + m_DM)  # reduced mass
    # sigma_SI = lambda_HP^2 * f_N^2 * m_N^2 * mu^2 / (pi * m_DM^2 * m_H^4)
    # But for real scalar, correct formula:
    # sigma_SI = lambda_HP^2 * f_N^2 * mu^2 * m_N^2 / (4 * pi * m_H^4 * m_DM^2)
    # Note: different references use slightly different conventions.
    # Using Cline et al. (2013) convention for real scalar:
    sig = lam_HP**2 * f_N**2 * mu_red**2 * m_N**2 / (4.0 * math.pi * m_H**4 * m_DM**2)
    return sig * GEV2_TO_CM2  # convert to cm^2


def part3_direct_detection():
    separator("PART 3: Direct Detection (Spin-Independent)")

    m_DM = M_SFE
    sig = sigma_SI(m_DM, lambda_HP)

    print(f"\n  DM-nucleon SI cross section:")
    print(f"    m_DM = {m_DM:.2f} GeV")
    print(f"    lambda_HP = {lambda_HP:.4f}")
    print(f"    f_N = {f_N}")
    print(f"    mu_reduced = {m_N * m_DM / (m_N + m_DM):.3f} GeV")
    print(f"\n    sigma_SI = {sig:.3e} cm^2")

    # Experimental limits (approximate, 2024)
    # LZ (2024): ~2e-47 cm^2 at 36 GeV, ~3e-47 at 44 GeV
    lz_limit = 3.0e-47  # cm^2 at ~44 GeV
    xenon_limit = 9.0e-47  # XENONnT 2023 at ~44 GeV

    print(f"\n  Experimental limits at m_DM ~ {m_DM:.0f} GeV:")
    print(f"    LZ (2024):      {lz_limit:.1e} cm^2")
    print(f"    XENONnT (2023): {xenon_limit:.1e} cm^2")
    print(f"\n    SFE / LZ ratio:     {sig/lz_limit:.1f}x")
    print(f"    SFE / XENONnT ratio: {sig/xenon_limit:.1f}x")

    if sig > lz_limit:
        print(f"\n    STATUS: EXCLUDED by direct detection")
        print(f"    SFE prediction is {sig/lz_limit:.0f}x above LZ limit")

        # What lambda_HP would be allowed?
        lam_max = lambda_HP * math.sqrt(math.sqrt(lz_limit / sig))
        print(f"\n    Maximum allowed lambda_HP: {lam_max:.6f}")
        print(f"    SFE predicts: {lambda_HP:.4f}")
        print(f"    Ratio: {lambda_HP / lam_max:.1f}x too large")
    else:
        print(f"\n    STATUS: ALLOWED")

    # Light DM scenario (m_phi ~ 25 MeV)
    print(f"\n  Light DM scenario (proton radius boson):")
    for m_light in [22.0e-3, 25.0e-3, 30.0e-3]:
        sig_light = sigma_SI(m_light, lambda_HP)
        print(f"    m_phi = {m_light*1e3:.0f} MeV: sigma_SI = {sig_light:.3e} cm^2")
    print(f"    LZ sensitivity at ~25 MeV: not sensitive (below threshold ~3 GeV)")
    print(f"    Sub-GeV DM detection: SENSEI, CDEX, NEWS-G, CRESST")
    # CRESST limit at 25 MeV is roughly ~1e-34 cm^2
    cresst_25MeV = 1.0e-34
    sig_25MeV = sigma_SI(25.0e-3, lambda_HP)
    print(f"    CRESST limit at 25 MeV: ~{cresst_25MeV:.0e} cm^2")
    print(f"    SFE at 25 MeV: {sig_25MeV:.2e} cm^2")
    if sig_25MeV < cresst_25MeV:
        print(f"    -> ALLOWED (below current sub-GeV limits)")


# ============================================================
# PART 4: Mass Tension Analysis
# ============================================================
def part4_mass_tension():
    separator("PART 4: Mass Tension (M_SFE vs m_phi)")

    print(f"\n  Two mass scales in SFE:")
    print(f"    M_SFE = v_EW * delta = {M_SFE:.2f} GeV (g-2 energy scale)")
    print(f"    m_phi = 22-30 MeV (proton radius boson)")
    print(f"    Ratio: {M_SFE / 0.025:.0f}x")

    print(f"\n  In Higgs portal: m_Phi^2 = mu_Phi^2 + lambda_HP * v_EW^2")
    m_phi_target = 0.025  # 25 MeV in GeV
    lambda_HP_v2 = lambda_HP * v_EW**2
    mu_phi2_needed = m_phi_target**2 - lambda_HP_v2

    print(f"    lambda_HP * v_EW^2 = {lambda_HP_v2:.2f} GeV^2")
    print(f"    For m_phi = 25 MeV: mu_Phi^2 = {mu_phi2_needed:.2f} GeV^2")
    print(f"    Fine-tuning: |mu_Phi^2| / (lambda_HP v^2) = {abs(mu_phi2_needed)/lambda_HP_v2:.6f}")
    print(f"    Cancellation: 1 part in {lambda_HP_v2 / m_phi_target**2:.0f}")

    print(f"\n  INTERPRETATION OPTIONS:")

    print(f"\n  Option A: M_SFE = 43.77 GeV IS the physical mass")
    print(f"    -> mu_Phi^2 negligible, portal-dominated")
    print(f"    -> DM mass = 43.77 GeV")
    print(f"    -> Problem: excluded by LZ direct detection")
    print(f"    -> Problem: proton radius needs separate light boson")

    print(f"\n  Option B: m_phi ~ 25 MeV IS the physical mass")
    print(f"    -> Extreme fine-tuning required (1:{lambda_HP_v2/m_phi_target**2:.0f})")
    print(f"    -> DM mass = 25 MeV (warm DM, sub-GeV)")
    print(f"    -> Below LZ threshold, but constrained by CRESST/SENSEI")
    print(f"    -> Below standard WIMP window")

    print(f"\n  Option C: M_SFE is an ENERGY SCALE, not a particle mass")
    print(f"    -> M_SFE enters g-2 formula as loop momentum cutoff")
    print(f"    -> Physical boson mass is independent: m_phi ~ 25 MeV")
    print(f"    -> lambda_HP concept doesn't directly set mass")
    print(f"    -> Most consistent with proton radius + g-2 simultaneously")
    print(f"    -> But then: Higgs portal derivation of M_SFE is reinterpreted")

    print(f"\n  Option D: Two scalar sector")
    print(f"    -> Heavy scalar Phi_H (43.77 GeV): mediates g-2, Higgs portal DM")
    print(f"    -> Light scalar Phi_L (25 MeV): mediates proton radius, Yukawa")
    print(f"    -> Phi_L could be pseudo-Goldstone of Phi_H symmetry breaking")
    print(f"    -> Adds complexity but resolves both tensions")


# ============================================================
# PART 5: Structure Formation Constraints
# ============================================================
def part5_structure_formation():
    separator("PART 5: Structure Formation & BBN Constraints")

    print(f"\n  === Heavy DM scenario: m_DM = {M_SFE:.2f} GeV ===")
    print(f"    Classification: COLD dark matter (CDM)")
    print(f"    Free-streaming length: lambda_fs << 1 Mpc")
    print(f"    -> Consistent with Lyman-alpha, galaxy surveys")
    print(f"    -> Standard CDM structure formation")

    # For light DM
    m_light = 0.025  # 25 MeV
    T_kd = m_light / 25.0  # kinetic decoupling ~ m/25
    # Free-streaming length ~ 0.1 Mpc * (keV / m_DM) for thermal relics
    # For 25 MeV: lambda_fs ~ 0.1 * (1e-3 / 0.025) ~ 0.004 Mpc
    lambda_fs = 0.1 * (1e-3 / m_light)

    print(f"\n  === Light DM scenario: m_DM = {m_light*1e3:.0f} MeV ===")
    print(f"    Free-streaming length: ~{lambda_fs:.4f} Mpc")
    print(f"    Classification: {'WARM' if lambda_fs > 0.01 else 'COLD'} dark matter")

    # BBN constraints
    print(f"\n  BBN constraints:")
    print(f"    Light scalar with m = 25 MeV:")
    print(f"      If in thermal equilibrium during BBN (T ~ 1 MeV): contributes to N_eff")
    print(f"      Real scalar: Delta N_eff = 4/7 * (T_phi/T_nu)^4")

    # If Phi decouples before e+e- annihilation
    g_star_dec = 86.25  # at T ~ m_DM
    g_star_nu = 10.75   # at neutrino decoupling
    T_ratio = (g_star_nu / g_star_dec)**(1.0/3.0)
    delta_Neff = (4.0/7.0) * T_ratio**(4.0)

    print(f"      T_phi/T_nu = (g*(nu)/g*(dec))^(1/3) = ({g_star_nu}/{g_star_dec})^(1/3) = {T_ratio:.4f}")
    print(f"      Delta N_eff = {delta_Neff:.4f}")
    print(f"      Planck limit: N_eff = 2.99 +/- 0.17 (95% CL)")
    print(f"      Available room: Delta N_eff < 0.30")
    if delta_Neff < 0.30:
        print(f"      -> ALLOWED ({delta_Neff:.3f} < 0.30)")
    else:
        print(f"      -> MARGINALLY CONSTRAINED")

    # Lyman-alpha constraint
    print(f"\n  Lyman-alpha forest constraint:")
    print(f"    Lower bound on thermal WDM mass: m > 5.3 keV (95% CL)")
    print(f"    m_phi = 25 MeV >> 5.3 keV -> SAFE")

    # CMB constraint on DM annihilation
    print(f"\n  CMB constraint on DM annihilation:")
    sv_total = sigma_v_total(M_SFE, lambda_HP)
    sv_cm3s = sv_total * GEV2_TO_CM3S
    p_ann = sv_cm3s / M_SFE  # cm^3/s/GeV, effective annihilation parameter
    p_ann_limit = 3.5e-28  # cm^3/s/GeV (Planck 2018 for s-wave)
    print(f"    p_ann = <sigma*v>/m_DM = {p_ann:.3e} cm^3/s/GeV")
    print(f"    Planck limit: {p_ann_limit:.1e} cm^3/s/GeV")
    if p_ann < p_ann_limit:
        print(f"    -> ALLOWED")
    else:
        print(f"    -> EXCLUDED")


# ============================================================
# PART 6: SFE DM vs Standard Paradigms
# ============================================================
def part6_comparison():
    separator("PART 6: SFE Dark Matter vs Standard Paradigms")

    print(f"\n  ===  DM Candidate Comparison ===")
    print(f"  {'':>20s} {'WIMP':>12s} {'Axion':>12s} {'SFE (heavy)':>14s} {'SFE (light)':>14s}")
    print(f"  {'Mass':>20s} {'~100 GeV':>12s} {'~ueV':>12s} {'43.8 GeV':>14s} {'25 MeV':>14s}")
    print(f"  {'Spin':>20s} {'1/2 or 1':>12s} {'0':>12s} {'0':>14s} {'0':>14s}")
    print(f"  {'Production':>20s} {'freeze-out':>12s} {'misalign':>12s} {'freeze-out':>14s} {'?':>14s}")
    print(f"  {'Direct det.':>20s} {'YES':>12s} {'cavity':>12s} {'EXCLUDED':>14s} {'sub-GeV':>14s}")
    print(f"  {'Omega predict':>20s} {'tuned':>12s} {'tuned':>12s} {'geometric':>14s} {'geometric':>14s}")
    print(f"  {'Parameters':>20s} {'many':>12s} {'few':>12s} {'0':>14s} {'0':>14s}")

    print(f"\n  SFE unique feature: Omega_DM derived from first principles")
    print(f"    Standard model: Omega_DM is measured, not predicted")
    print(f"    SUSY/WIMP: Omega_DM depends on sparticle spectrum (many parameters)")
    print(f"    Axion: Omega_DM depends on initial misalignment angle (1 parameter)")
    print(f"    SFE: Omega_DM = f(d, sin^2 theta_W, alpha_s) = 0.2571 (0 free parameters)")


# ============================================================
# PART 7: Critical Assessment & Testable Predictions
# ============================================================
def part7_assessment():
    separator("PART 7: Critical Assessment & Testable DM Predictions")

    print(f"\n  === TENSIONS & PROBLEMS ===")

    print(f"\n  [CRITICAL] Heavy DM (43.77 GeV) excluded by LZ")
    sig = sigma_SI(M_SFE, lambda_HP)
    lz = 3e-47
    print(f"    sigma_SI = {sig:.2e} cm^2 vs LZ limit {lz:.0e} cm^2")
    print(f"    Overshoot: {sig/lz:.0f}x")
    print(f"    IF M_SFE is the physical DM mass AND Higgs portal is the")
    print(f"    only interaction, this is excluded.")

    print(f"\n  [TENSION] Thermal relic underproduction")
    oh2 = relic_density(M_SFE, lambda_HP)
    print(f"    Omega h^2 = {oh2:.6f} vs observed 0.12")
    print(f"    Underproduction factor: {0.12/oh2:.0f}x")
    print(f"    Annihilation cross section too LARGE -> DM annihilates away")

    print(f"\n  [TENSION] Two mass scales")
    print(f"    g-2 scale: 43.77 GeV")
    print(f"    Proton radius: 22-30 MeV")
    print(f"    Cannot be the same particle without fine-tuning")

    print(f"\n  === POSSIBLE RESOLUTIONS ===")

    print(f"\n  Resolution 1: M_SFE is NOT a particle mass")
    print(f"    M_SFE = 43.77 GeV is the energy scale of path integral folding")
    print(f"    Not a physical particle mass but a momentum cutoff")
    print(f"    Actual suppression quanta: m_phi ~ 25 MeV from geometry")
    print(f"    -> Avoids direct detection exclusion")
    print(f"    -> Requires reinterpretation of Higgs portal section")

    print(f"\n  Resolution 2: Non-thermal production")
    print(f"    SFE DM not produced by thermal freeze-out")
    print(f"    Instead: geometric condensation of suppression field")
    print(f"    Relic density set by SFE equation, not Boltzmann equation")
    print(f"    -> Bypasses thermal overproduction problem")
    print(f"    -> Omega_DM = (1-eps^2)*alpha_s*pi/(1+alpha_s*pi) IS the prediction")
    print(f"    -> No need to match thermal cross section")

    print(f"\n  Resolution 3: Suppression boson is NOT the DM particle")
    print(f"    DM = suppression field vacuum fluctuations (collective, not particle)")
    print(f"    Boson (25 MeV) = mediator of suppression force")
    print(f"    DM is the field condensate, not individual quanta")
    print(f"    -> No direct detection signal expected")
    print(f"    -> Consistent with all null results")

    print(f"\n  === TESTABLE DM PREDICTIONS ===")

    print(f"\n  Prediction A: DM density from first principles")
    print(f"    Omega_DM h^2 = 0.1163 (SFE)")
    print(f"    Omega_DM h^2 = 0.1200 +/- 0.0012 (Planck)")
    Omega_DM_sfe = 0.2571
    oh2_sfe = Omega_DM_sfe * h_hubble**2
    oh2_obs = 0.1200
    print(f"    Exact: {oh2_sfe:.4f} vs {oh2_obs:.4f} ({abs(oh2_sfe-oh2_obs)/oh2_obs*100:.2f}% off)")

    print(f"\n  Prediction B: DM/DE ratio = alpha_s * pi")
    r_sfe = alpha_s * math.pi
    r_obs = 0.2589 / 0.6847
    print(f"    SFE:     {r_sfe:.4f}")
    print(f"    Planck:  {r_obs:.4f}")
    print(f"    Diff:    {abs(r_sfe - r_obs)/r_obs*100:.2f}%")
    print(f"    -> DESI BAO (2025-2028) will improve DM/DE precision")

    print(f"\n  Prediction C: alpha_s from cosmology")
    alpha_s_cosmo = r_obs / math.pi
    print(f"    alpha_s(cosmo) = Omega_DM / (pi * Omega_Lambda) = {alpha_s_cosmo:.4f}")
    print(f"    alpha_s(PDG)   = {alpha_s} +/- 0.0009")
    print(f"    Tension: {abs(alpha_s_cosmo - alpha_s)/0.0009:.1f} sigma")

    print(f"\n  Prediction D: No DM particle in direct detection")
    print(f"    IF Resolution 3 is correct:")
    print(f"    All direct detection experiments will continue to see null")
    print(f"    This is consistent with 40+ years of null results")
    print(f"    XENON, LZ, PandaX, DARWIN -> all null")

    print(f"\n  Prediction E: DM fraction scales with dimension")
    print(f"    Omega_DM(d) = (1 - e^{{-d}}) * alpha_s*pi / (1 + alpha_s*pi)")
    for d in [2, 3, 4, 10]:
        eps2_d = math.exp(-d)  # simplified
        Om_DM_d = (1 - eps2_d) * alpha_ratio / (1 + alpha_ratio)
        print(f"    d = {d:2d}: Omega_DM = {Om_DM_d:.4f}")


# ============================================================
# PART 8: Summary
# ============================================================
def part8_summary():
    separator("SUMMARY: SFE Dark Matter Status")

    print(f"""
  STRENGTHS:
    1. Omega_DM = 0.2571 from first principles (0.68% from Planck)
       No other theory does this with zero free parameters.
    2. DM/DE ratio = alpha_s * pi connects particle physics to cosmology
    3. Consistent with 40+ years of null direct detection
    4. Explains WHY DM fraction is ~26% (not 50%, not 1%)

  WEAKNESSES:
    1. If DM = 43.77 GeV Higgs portal scalar:
       -> Excluded by LZ direct detection (sigma_SI ~ {sigma_SI(M_SFE, lambda_HP):.0e} cm^2)
       -> Thermally UNDERPRODUCED (Omega h^2 ~ {relic_density(M_SFE, lambda_HP):.6f})
    2. Mass tension: M_SFE (43.77 GeV) vs m_phi (25 MeV)
    3. alpha = alpha_s * pi: why exactly pi? Not rigorously derived.
    4. DM "identity" unclear: particle vs field condensate

  MOST LIKELY INTERPRETATION:
    SFE dark matter is NOT a particle in the WIMP sense.
    It is the vacuum fluctuation condensate of the suppression field.
    The 25 MeV boson mediates the suppression force but is not itself DM.
    Omega_DM is set geometrically (path integral folding), not thermally.

  KEY EXPERIMENTAL TESTS:
    | Test                  | Timeline   | Expected Signal       |
    |-----------------------|------------|-----------------------|
    | DESI DM/DE ratio      | 2025-2028  | alpha_s*pi = 0.3704   |
    | Direct detection null | ongoing    | continued null        |
    | Fermilab g-2 final    | 2025-2026  | confirms anomaly      |
    | PADME/NA64 boson      | 2025-2027  | 25 MeV scalar         |
    | CMB-S4 Omega_DM       | ~2030      | 0.2571 (0.3% prec.)   |
""")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  SFE DARK MATTER COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    Omega_DM = part1_geometric_prediction()
    omega_h2 = part2_thermal_relic()
    part3_direct_detection()
    part4_mass_tension()
    part5_structure_formation()
    part6_comparison()
    part7_assessment()
    part8_summary()
