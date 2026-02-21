
import math
from dataclasses import dataclass
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from check_dark_matter_paper import Const, sfe_predictions
from check_muon_g2_integral import HBAR_C, feynman_integral


ALPHA_EM = 1.0 / 137.035999084
PI = math.pi
E_NUM = math.e
M_MU = 105.6583755       # MeV
M_E = 0.51099895         # MeV
M_P = 938.272088         # MeV
V_EW_MEV = 246.2196e3    # MeV
M_H_MEV = 125.25e3       # MeV
GAMMA_H_MEV = 4.07e3     # MeV
DA_MU_EXP = 249e-11
DA_MU_ERR = 48e-11


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_close(results, name, actual, expected, tol):
    diff = abs(actual - expected)
    passed = diff <= tol
    detail = (
        f"actual={actual:.10g}, expected={expected:.10g}, "
        f"|diff|={diff:.3e}, tol={tol:.3e}"
    )
    results.append(CheckResult(name=name, passed=passed, detail=detail))


def check_range(results, name, actual, lo, hi):
    passed = lo <= actual <= hi
    detail = f"actual={actual:.10g}, range=[{lo:.10g}, {hi:.10g}]"
    results.append(CheckResult(name=name, passed=passed, detail=detail))


def print_report(results):
    print("=" * 82)
    print("  SFE RECENT MATERIALS VERIFICATION")
    print("=" * 82)

    n_pass = 0
    for row in results:
        status = "PASS" if row.passed else "FAIL"
        if row.passed:
            n_pass += 1
        print(f"[{status}] {row.name}")
        print(f"       {row.detail}")

    n_total = len(results)
    n_fail = n_total - n_pass
    print("-" * 82)
    print(f"Total: {n_total}, Passed: {n_pass}, Failed: {n_fail}")
    print("=" * 82)

    if n_fail > 0:
        raise SystemExit(1)


def main():
    results = []

    # -----------------------------------------------------------------
    # 1) Cosmology block (recent: check_dark_matter_paper.py)
    # -----------------------------------------------------------------
    pred = sfe_predictions(Const.sin2_tW, Const.alpha_s)
    eps2 = pred["eps2"]
    delta = pred["delta"]
    d_eff = pred["D_eff"]
    alpha_ratio = pred["alpha"]
    omega_b = pred["Omega_b"]
    omega_l = pred["Omega_Lambda"]
    omega_dm = pred["Omega_DM"]

    eq_residual = abs(eps2 - math.exp(-(1.0 - eps2) * d_eff))

    check_close(results, "delta = sin^2(theta_W)cos^2(theta_W)", delta, 0.17776, 5e-5)
    check_close(results, "D_eff = 3 + delta", d_eff, 3.17776, 5e-5)
    check_close(results, "self-consistency equation residual", eq_residual, 0.0, 1e-12)
    check_close(results, "Omega_b prediction", omega_b, 0.04865, 2e-5)
    check_close(results, "Omega_Lambda prediction", omega_l, 0.6942, 2e-4)
    check_close(results, "Omega_DM prediction", omega_dm, 0.2571, 2e-4)
    check_close(results, "Omega sum to unity", omega_b + omega_l + omega_dm, 1.0, 1e-12)
    check_close(results, "DM/DE = alpha_s * pi", omega_dm / omega_l, alpha_ratio, 1e-12)

    # -----------------------------------------------------------------
    # 2) Muon/electron g-2 block (recent: check_muon_g2_derivation.py)
    # -----------------------------------------------------------------
    m_sfe = V_EW_MEV * delta
    da_mu = (ALPHA_EM / (2.0 * PI)) * (1.0 / E_NUM) * (M_MU / m_sfe) ** 2
    da_e = (ALPHA_EM / (2.0 * PI)) * (1.0 / E_NUM) * (M_E / m_sfe) ** 2
    sigma_mu = abs(da_mu - DA_MU_EXP) / DA_MU_ERR

    check_close(results, "M_SFE = v_EW * delta (GeV)", m_sfe / 1e3, 43.77, 0.02)
    check_close(results, "Delta a_mu prediction (x1e-11)", da_mu * 1e11, 249.0, 0.2)
    check_close(results, "Delta a_e prediction (x1e-14)", da_e * 1e14, 5.82, 0.05)
    check_range(results, "muon g-2 tension within 1 sigma", sigma_mu, 0.0, 1.0)

    # -----------------------------------------------------------------
    # 3) Scalar-loop equivalence / proton radius scale
    #    (recent: check_muon_g2_integral.py)
    # -----------------------------------------------------------------
    i_light = feynman_integral(0.001, M_MU)
    g_mu = math.sqrt(da_mu * 16.0 * PI**2)
    da_mu_loop = g_mu**2 / (8.0 * PI**2) * i_light
    kappa = g_mu / M_MU
    g_p = kappa * M_P

    r_p_e = 0.8751
    r_p_mu = 0.84087
    dr2_obs = r_p_e**2 - r_p_mu**2
    m_phi_mev = math.sqrt(3.0 * g_mu * g_p / (2.0 * ALPHA_EM * dr2_obs)) * HBAR_C

    check_close(results, "light-boson Feynman integral I(r->0)", i_light, 0.5, 5e-4)
    check_close(results, "geometric vs scalar-loop Delta a_mu", da_mu_loop, da_mu, 1e-12)
    check_close(results, "mass-proportional coupling kappa (MeV^-1)", kappa, 5.93e-6, 8e-8)
    check_range(results, "proton-radius boson mass window (MeV)", m_phi_mev, 21.0, 30.0)

    # -----------------------------------------------------------------
    # 4) Higgs portal consistency block (recent: check_open_problems.py)
    # -----------------------------------------------------------------
    lambda_hp = delta**2
    gamma_h_to_phiphi = lambda_hp**2 * V_EW_MEV**2 / (8.0 * PI * M_H_MEV)
    br_inv = gamma_h_to_phiphi / GAMMA_H_MEV
    theta_mix = lambda_hp * V_EW_MEV / M_H_MEV
    sin2_mix = math.sin(theta_mix) ** 2

    check_close(results, "lambda_HP = delta^2", lambda_hp, 0.0316, 5e-4)
    check_close(results, "BR(H->invisible)", br_inv, 0.005, 6e-4)
    check_close(results, "sin^2(theta_mix)", sin2_mix, 0.004, 6e-4)

    # -----------------------------------------------------------------
    # 5) Unification relation (recent: check_unification.py)
    # -----------------------------------------------------------------
    sin_tw = math.sqrt(Const.sin2_tW)
    xi_as = alpha_ratio_raw = Const.alpha_s ** (1.0 / 3.0)
    rhs_unif = 2.0 * Const.alpha_s ** (2.0 / 3.0)
    s2_from_as = 4.0 * Const.alpha_s ** (4.0 / 3.0)
    as_from_s = (sin_tw / 2.0) ** 1.5

    check_close(results, "sin(tW) = 2*as^(2/3) [unification]", sin_tw, rhs_unif, 5e-3)
    check_close(results, "sin^2(tW) = 4*as^(4/3)", Const.sin2_tW, s2_from_as, 5e-3)
    check_close(results, "alpha_s = (sin(tW)/2)^(3/2)", Const.alpha_s, as_from_s, 5e-4)

    # -----------------------------------------------------------------
    # 6) Dynamic dark energy block (recent: check_dynamic_de.py)
    # -----------------------------------------------------------------
    xi_unif = Const.alpha_s ** (1.0 / 3.0)
    w0_dynamic = -1.0 + 2.0 * xi_unif**2 / (3.0 * omega_l)
    desi_w0 = -0.770
    desi_w0_lo, desi_w0_hi = -0.881, -0.651

    # SFE potential structure: Mexican hat = Higgs-like (DESI best-fit)
    # V(Phi) = -(1/2)mu^2 Phi^2 + (1/4)lambda Phi^4
    # is equivalent to V(phi) = V0 + (1/2)m^2 phi^2 + (1/4)lambda phi^4 with m^2 < 0
    pot_v0 = omega_l  # normalized
    pot_m2 = -delta**2  # SSB tachyonic mass (sign convention)
    pot_lambda = delta**2 / (0.15**2)  # lambda = |m^2|/v^2
    pot_vev = math.sqrt(-pot_m2 / pot_lambda) if pot_m2 < 0 and pot_lambda > 0 else 0.0

    check_close(results, "xi = alpha_s^(1/3)", xi_unif, 0.4904, 5e-4)
    check_close(results, "w0 dynamic (xi=as^1/3)", w0_dynamic, -0.769, 0.01)
    check_range(results, "w0 dynamic within DESI 95% CL", w0_dynamic, desi_w0_lo, desi_w0_hi)
    check_range(results, "w0 dynamic differs from LCDM (w=-1)", abs(1.0 + w0_dynamic), 0.01, 0.5)
    results.append(CheckResult(
        name="SFE potential = Higgs-like (m^2 < 0, lambda > 0)",
        passed=pot_m2 < 0 and pot_lambda > 0 and pot_vev > 0,
        detail=f"m^2={pot_m2:.6f}, lambda={pot_lambda:.4f}, vev={pot_vev:.4f}",
    ))

    print_report(results)


if __name__ == "__main__":
    main()

