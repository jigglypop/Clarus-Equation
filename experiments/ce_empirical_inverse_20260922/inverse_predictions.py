"""Recovered CE empirical relations: forward, inverse and cross predictions.

These are conditional formula tests, not a derivation of nature's action.
All energies in eV, except explicitly named MeV/GeV outputs. See the paper.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = Path(__file__).resolve().parent
mp.mp.dps = 55
PI = math.pi
ALPHA_S = 0.11789
ALPHA_S_SIGMA = 0.0009  # historical input uncertainty, not a new determination
ALPHA_ZERO = 1 / 137.035999084
ALPHA_EW = 1 / 129.0  # the separate historical electroweak-scale input
V = 246.2196e9
M_MU = 105.6583755e6
M_E = 0.51099895e6
M_TAU = 1776.86e6
M_P = 938.272088e6
M_PL = 1.220910e28  # historical non-reduced Planck mass
HBAR_EV_S = 6.582119569e-16
MPC_KM = 3.0856775814913673e19
K_MU = ALPHA_ZERO / (2 * PI * math.e) * (M_MU / V) ** 2
NG = 12
PHASE_AREA = PI**2 / 2
DELTA = (4 * ALPHA_S ** (4 / 3)) * (1 - 4 * ALPHA_S ** (4 / 3))
EW_WINDOW = (0.12, 0.24)  # explicit reporting window around the measured weak angle


def q_of(delta: float) -> float:
    d = mp.mpf(3) + mp.mpf(delta)
    return float(-mp.lambertw(-d * mp.exp(-d), 0).real / d)


def s_of(delta: float) -> float:
    if not 0 < delta <= 0.25:
        raise ValueError("real low-angle branch requires 0 < delta <= 1/4")
    return 2 * delta / (1 + math.sqrt(max(0.0, 1 - 4 * delta)))


def alpha_of(delta: float) -> float:
    return (s_of(delta) / 4) ** 0.75


def loop(r: float, kind: str = "vector_shape") -> float:
    """Archived numerator is vector-shaped; pure scalar has x^2(2-x)."""
    rr = mp.mpf(r) ** 2
    if kind not in {"vector_shape", "scalar"}:
        raise ValueError(kind)
    n = 1 if kind == "vector_shape" else 2
    if r == 0:
        return n - 0.5
    return float(mp.quad(lambda x: x*x*(n-x)/(x*x+(1-x)*rr), [0, .01, .1, .5, 1]))


def forward(delta: float, include_loops: bool = False) -> dict[str, float]:
    s, a, q = s_of(delta), alpha_of(delta), q_of(delta)
    d = 3 + delta
    # The full three-layer sum, not the inaccurate one-factor abbreviation.
    gauge_sum = 2 * PI * (a + ALPHA_EW / delta)
    r = a * (d + q * (gauge_sum + delta**2))
    de = (1 - q) / (1 + r)
    dm = (1 - q) * r / (1 + r)
    ne = 18 * d
    logs = PHASE_AREA * ne - PI * delta * (1 - q)
    h = math.sqrt(PI) * M_PL * math.exp(-logs / 2) / HBAR_EV_S * MPC_KM
    rho = 1e3 * math.exp((math.log(de * 3 / 8) + 4 * math.log(M_PL) - logs) / 4)
    out = {
        "delta": delta, "sin2_theta_w": s, "alpha_s": a, "q": q,
        "gauge_weight_sum": gauge_sum, "R_3layer": r,
        "R_LO": a * d, "R_abbreviation": a*d*(1+q*delta),
        "R_normalized_weights": a*(d+q*(1+delta**2)),
        "omega_b": q, "omega_dm": dm, "omega_de": de, "omega_m": q+dm,
        "Ne": ne, "log_S": logs, "H_phase": h,
        "H0_if_true_deSitter_horizon": h / math.sqrt(de),
        "rho_de_quarter_meV": rho, "M_CE_GeV": V*delta/1e9,
        "mediator_MeV": M_P*delta**2/1e6,
        "mu_contact_1e11": K_MU/delta**2*1e11,
        "electron_contact": K_MU/delta**2*(M_E/M_MU)**2,
        "tau_contact": K_MU/delta**2*(M_TAU/M_MU)**2,
        "omega_b_h2": q*(h/100)**2, "omega_dm_h2": dm*(h/100)**2,
        "scalar_index_candidate": 1-2/ne,
        "w0_alternative_DE_branch": -1+2*a**(2/3)/(3*de),
        "wa_alternative_DE_branch": -2*a**(2/3)*(1-de)/de,
    }
    if include_loops:
        for kind, key in [("vector_shape", "historical_finite"), ("scalar", "scalar_matched")]:
            norm = loop(0, kind)
            for label, mass in [("mu", M_MU), ("electron", M_E), ("tau", M_TAU)]:
                factor = loop(M_P*delta**2/mass, kind)/norm
                value = K_MU/delta**2*(mass/M_MU)**2*factor
                out[f"{label}_{key}"] = value
                if label == "mu":
                    out[f"{label}_{key}_1e11"] = value*1e11
                    out[f"{key}_factor"] = factor
    return out


def all_roots(key: str, target: float) -> list[float]:
    """Numerical root enumeration, not a proof of root completeness.

    The dark ratio diverges at delta -> 0 because alpha_EW is held fixed;
    retaining the tiny-delta solution prevents an unjustified uniqueness claim.
    """
    grid = np.unique(np.r_[np.geomspace(1e-24, .25, 500), np.linspace(.01, .25, 301)])
    fun = lambda x: forward(float(x))[key] - target
    roots = []
    old, f_old = float(grid[0]), fun(grid[0])
    for point in grid[1:]:
        point = float(point)
        val = fun(point)
        if val * f_old < 0:
            root = brentq(fun, old, point, xtol=1e-30, rtol=1e-13)
            if not roots or abs(root-roots[-1]) > 1e-12*max(root, 1e-24):
                roots.append(root)
        if val == 0:
            roots.append(point)
        old, f_old = point, val
    return roots


def invert(key: str, target: float) -> dict:
    if key == "mu_contact_1e11":
        raw = math.sqrt(K_MU / (target*1e-11)) if target > 0 else None
        roots = [raw] if raw is not None and 0 < raw <= .25 else []
    elif key == "omega_b":
        raw = -math.log(target)/(1-target)-3 if 0 < target < 1 else None
        roots = [raw] if raw is not None and 0 < raw <= .25 else []
    else:
        raw, roots = None, all_roots(key, target)
    physical = [r for r in roots if EW_WINDOW[0] <= r <= EW_WINDOW[1]]
    return {"input_key": key, "input_value": target, "unrestricted_algebraic_delta": raw,
            "low_angle_roots": roots, "root_search_interval": [1e-24, .25],
            "reporting_EW_window": list(EW_WINDOW),
            "cross_predictions": [forward(r) for r in physical],
            "note": "retrospective conditional cross prediction; not blind validation"}


def sensitivity(delta: float) -> dict:
    keys = ["sin2_theta_w", "omega_b", "omega_dm", "omega_de", "H_phase", "rho_de_quarter_meV", "mu_contact_1e11"]
    eps = 1e-6
    lo, hi = forward(delta-eps), forward(delta+eps)
    jac_delta = np.array([(hi[k]-lo[k])/(2*eps) for k in keys])
    s = s_of(delta)
    d_delta_d_alpha = (1-2*s)*(16/3)*ALPHA_S**(1/3)
    jac_alpha = jac_delta*d_delta_d_alpha
    cov = np.outer(jac_alpha, jac_alpha)*ALPHA_S_SIGMA**2
    return {"keys": keys, "d_y_d_delta": jac_delta.tolist(),
            "d_delta_d_alpha_s": d_delta_d_alpha,
            "alpha_s_sigma": ALPHA_S_SIGMA,
            "conditional_sigma": np.sqrt(np.diag(cov)).tolist(),
            "input_covariance": cov.tolist(), "rank": int(np.linalg.matrix_rank(cov)),
            "scope": "one common alpha_s input only; no model error; other historical inputs fixed"}


def desi_comparison(base: dict) -> dict:
    # Published Eq. (21) + correlation, DESI DR2 II, flat LCDM.
    mean = np.array([.3027, 68.17])
    sig = np.array([.0036, .28])
    corr = -.975
    cov = np.outer(sig, sig)*np.array([[1, corr], [corr, 1]])
    inv = np.linalg.inv(cov)

    def chi2(d):
        p = forward(d)
        residual = np.array([p["omega_m"], p["H_phase"]])-mean
        return float(residual @ inv @ residual)

    fit = minimize_scalar(chi2, bounds=(.15, .20), method="bounded", options={"xatol": 1e-14})
    residual = np.array([base["omega_m"], base["H_phase"]])-mean
    eps = 1e-6
    lo, hi = forward(DELTA-eps), forward(DELTA+eps)
    dd_da = sensitivity(DELTA)["d_delta_d_alpha_s"]
    jac = np.array([(hi[k]-lo[k])/(2*eps)*dd_da for k in ["omega_m", "H_phase"]])
    total_cov = cov + np.outer(jac, jac)*ALPHA_S_SIGMA**2
    return {"source": "https://arxiv.org/html/2503.14738v3#S6.E21",
            "data_role": "already seen, compressed flat-LCDM posterior consistency only",
            "mean_omega_m_H0": mean.tolist(), "covariance": cov.tolist(),
            "chi2_fixed_inputs": chi2(DELTA), "marginal_pulls": (residual/sig).tolist(),
            "chi2_with_linear_input_covariance": float(residual @ np.linalg.inv(total_cov) @ residual),
            "best_delta_retrospective_fit": float(fit.x), "best_chi2_retrospective_fit": float(fit.fun),
            "best_fit_prediction": forward(float(fit.x)),
            "matter_mapping":"For this diagnostic omega_b+omega_dm is identified with total nonrelativistic matter, including massive neutrinos. The CDM-only inverse rows use a different identification and are not combined with this score.",
            "scope": "2-dimensional Gaussian diagnostic, not full likelihood, not a discovery significance; fit not adopted"}


def inverse_bridge_requirements() -> list[dict]:
    """Exact coefficient requirements from paired central values, without fitting them in."""
    rows = []
    for label, h, de in [("Planck_base",67.36,.6847),("Planck_BAO",67.66,.6889),("DESI_DR2_CMB",68.17,.6973)]:
        dh = all_roots("H_phase",h)[0]
        ph = forward(dh)
        required_r = (1-ph["q"]-de)/de
        required_weight = (required_r/ph["alpha_s"]-(3+dh))/ph["q"]-dh**2
        density_roots = all_roots("omega_de",de)
        dd = next(x for x in density_roots if EW_WINDOW[0]<x<EW_WINDOW[1])
        pd = forward(dd)
        entropy_offset = 2*math.log(pd["H_phase"]/h)
        rows.append({"label":label,"H_input":h,"DE_input":de,"delta_from_H":dh,
            "delta_from_DE":dd,"required_R_at_delta_from_H":required_r,
            "required_gauge_weight_sum_at_delta_from_H":required_weight,
            "nonnegative_total_feedback_possible_at_central_values":required_r >= ph["R_LO"],
            "DE_upper_bound_for_nonnegative_feedback_at_delta_from_H":(1-ph["q"])/(1+ph["R_LO"]),
            "required_log_entropy_offset_at_delta_from_DE":entropy_offset,
            "required_phase_area_at_delta_from_DE":PHASE_AREA+entropy_offset/pd["Ne"],
            "fractional_phase_area_change":entropy_offset/pd["Ne"]/PHASE_AREA,
            "status":"paired central-value inverse constraints, covariance not available for the Planck pairs; no new coefficients adopted"})
    return rows


def checks(base: dict) -> dict:
    q, d, delta = base["q"], 3+DELTA, DELTA
    # Independent fixed-point iteration versus Lambert W and analytic inverses.
    iterate = .05
    for _ in range(200):
        iterate = math.exp(-(1-iterate)*d)
    q_prime = -q*(1-q)/(1-d*q)
    eps = 1e-6
    numeric_q_prime = (q_of(delta+eps)-q_of(delta-eps))/(2*eps)
    inverse_mu = math.sqrt(K_MU/(base["mu_contact_1e11"]*1e-11))
    inverse_q = -math.log(q)/(1-q)-3
    # Reconstruct the old three-layer expression without using its simplification.
    a1 = ALPHA_EW/(1-base["sin2_theta_w"])
    aw = ALPHA_EW/base["sin2_theta_w"]
    r_direct = ALPHA_S*sum(1+q*a/(1/(2*PI)) for a in [a1, aw, ALPHA_S]) + ALPHA_S*delta*(1+q*delta)
    # Exact proposed low-energy vector completion, with coupling normalization fixed.
    g_v = math.sqrt(4*PI*ALPHA_ZERO/math.e)*M_MU/(V*delta)
    vector_value = g_v*g_v/(4*PI*PI)*loop(M_P*delta*delta/M_MU)
    g_s = math.sqrt(8*PI*ALPHA_ZERO/(3*math.e))*M_MU/(V*delta)
    scalar_value = g_s*g_s/(8*PI*PI)*loop(M_P*delta*delta/M_MU, "scalar")
    pairs = {
        "lambert_vs_iteration": (q, iterate, 1e-14),
        "fixed_point_residual": (q-math.exp(-d*(1-q)), 0, 1e-14),
        "fraction_conservation": (q+base["omega_dm"]+base["omega_de"], 1, 1e-14),
        "three_layer_algebra": (base["R_3layer"], r_direct, 1e-14),
        "inverse_mu_roundtrip": (inverse_mu, delta, 1e-14),
        "inverse_q_roundtrip": (inverse_q, delta, 1e-14),
        "q_derivative": (q_prime, numeric_q_prime, 2e-10),
        "vector_completion_normalization": (vector_value, base["mu_historical_finite"], 1e-22),
        "scalar_completion_normalization": (scalar_value, base["mu_scalar_matched"], 1e-22),
        "vector_massless_integral": (loop(0), .5, 1e-14),
        "scalar_massless_integral": (loop(0, "scalar"), 1.5, 1e-14),
    }
    result = {name: {"left": a, "right": b, "error": abs(a-b), "tolerance": tol,
                     "pass": abs(a-b) <= tol} for name, (a,b,tol) in pairs.items()}
    assert all(x["pass"] for x in result.values()), result
    return result


def main() -> None:
    sources = json.loads((HERE/"sources.json").read_text(encoding="utf-8"))
    for source in sources:
        assert hashlib.sha256((HERE/source["saved_path"]).read_bytes()).hexdigest() == source["saved_sha256"]
    base = forward(DELTA, True)
    scenarios = [
        ("historical_muon_249", "mu_contact_1e11", 249.0),
        ("WP25_plus_final_experiment_central", "mu_contact_1e11", 38.5),
        ("BMW_DMZ_2026_v2_plus_final_experiment_central", "mu_contact_1e11", 19.5),
        ("Planck2018_base_H0", "H_phase", 67.36),
        ("Planck2018_plus_BAO_H0", "H_phase", 67.66),
        ("DESI_DR2_CMB_H0", "H_phase", 68.17),
        ("SH0ES2022_H0", "H_phase", 73.04),
        ("Planck2018_base_DE", "omega_de", .6847),
        ("Planck2018_plus_BAO_DE", "omega_de", .6889),
        ("DESI_DR2_CMB_DE", "omega_de", 1-.3027),
        ("Planck2018_base_CDM_central_ratio", "omega_dm", .1200/.6736**2),
        ("Planck2018_plus_BAO_CDM_central_ratio", "omega_dm", .11933/.6766**2),
        ("Planck2018_plus_BAO_baryon_central_ratio", "omega_b", .02242/.6766**2),
        ("historical_w0_target", "w0_alternative_DE_branch", -.770),
    ]
    inverse = {name: invert(key, val) for name,key,val in scenarios}
    # A positive finite-kernel response decreases strictly with mediator mass.
    ratio = 38.5/base["mu_contact_1e11"]
    mediator_ratio = brentq(lambda r: loop(r)/.5-ratio, 1e-5, 100, xtol=1e-12)
    intervals = []
    for target, err, label in [(249,48,"historical"),(38.5,math.hypot(62,14.5),"WP25_final"),
                             (19.5,math.hypot(36,14.5),"BMW_DMZ_2026_v2_final")]:
        lower = max(target-err, 16*K_MU*1e11)
        upper = target+err
        physical = None if upper < lower else [math.sqrt(K_MU/(upper*1e-11)), math.sqrt(K_MU/(lower*1e-11))]
        intervals.append({"reference": label, "mean": target, "sigma": err,
                          "one_sigma_real_delta_intersection": physical,
                          "H_phase_range_from_reference_interval": None if physical is None else
                          [forward(physical[1])["H_phase"],forward(physical[0])["H_phase"]]})
    result = {
        "status": "conditional empirical recovery and inverse predictions; no fundamental completion claim",
        "inputs": {"alpha_s": ALPHA_S,"alpha_s_sigma":ALPHA_S_SIGMA,"alpha_zero":ALPHA_ZERO,
                   "alpha_EW":ALPHA_EW,"v_eV":V,"M_Pl_eV":M_PL,"Ng":NG,"phase_area":PHASE_AREA,
                   "muon_mass_eV":M_MU,"electron_mass_eV":M_E,"tau_mass_eV":M_TAU,"proton_mass_eV":M_P,
                   "hbar_eV_s":HBAR_EV_S,"Mpc_km":MPC_KM,
                   "historical_rounded_ratios":[.0487,.2623,.6891],"rounded_ratio_sum":.0487+.2623+.6891},
        "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "forward": base, "inverse": inverse, "sensitivity": sensitivity(DELTA),
        "muon_inverse_domain": {"minimum_contact_1e11":16*K_MU*1e11,"reference_intervals":intervals},
        "muon_reference_update": {"SM_WP25_1e11":116592033,"SM_sigma_1e11":62,
              "experiment_2025_1e11":116592071.5,"experiment_sigma_1e11":14.5,
              "difference_1e11":38.5,"difference_sigma_1e11":math.hypot(62,14.5),
              "contact_fixed_input_pull":(base["mu_contact_1e11"]-38.5)/math.hypot(62,14.5),
              "finite_fixed_input_pull":(base["mu_historical_finite_1e11"]-38.5)/math.hypot(62,14.5)},
        "muon_reference_2026": {
              "source":"https://arxiv.org/html/2407.10913v2",
              "version_date":"2026-04-28",
              "SM_1e11":116592052,"SM_sigma_1e11":36,
              "experiment_1e11":116592071.5,"experiment_sigma_1e11":14.5,
              "difference_1e11":19.5,"difference_sigma_1e11":math.hypot(36,14.5),
              "contact_fixed_input_pull":(base["mu_contact_1e11"]-19.5)/math.hypot(36,14.5),
              "finite_fixed_input_pull":(base["mu_historical_finite_1e11"]-19.5)/math.hypot(36,14.5),
              "scalar_matched_fixed_input_pull":(base["mu_scalar_matched_1e11"]-19.5)/math.hypot(36,14.5),
              "required_contact_multiplier":19.5/base["mu_contact_1e11"],
              "required_finite_multiplier":19.5/base["mu_historical_finite_1e11"],
              "required_vector_mediator_MeV_at_fixed_delta":brentq(
                    lambda r:loop(r)/.5-19.5/base["mu_contact_1e11"],1e-5,100,xtol=1e-12)*M_MU/1e6,
              "scope":"alternative SM reference, not statistically independent of WP25; not combined; observed uncertainty only"},
        "new_muon_inverse_constraints": {
              "required_contact_multiplier_at_fixed_delta":ratio,
              "required_finite_multiplier_at_fixed_delta":38.5/base["mu_historical_finite_1e11"],
              "mediator_MeV_at_fixed_delta_and_vector_normalization":mediator_ratio*M_MU/1e6,
              "note":"inverse requirements, NOT adopted or fit into the frozen forward branch"},
        "desi_joint_compressed":desi_comparison(base), "verification":checks(base),
        "paired_inverse_bridge_requirements":inverse_bridge_requirements(),
        "eos_branch": {"source_role":"historical formula selected by a scan; not a blind result",
              "equation":"1+w0=2*alpha_s^(2/3)/(3*Omega_DE); wa=-3*(1+w0)*(1-Omega_DE)",
              "historical_target_w0":-.770,
              "DESI_DR2_CMB_DESY5_w0":-.752,"DESI_DR2_CMB_DESY5_wa":-.86,
              "xi_from_historical_w0_and_frozen_DE":math.sqrt(3*base["omega_de"]*(1-.770)/2),
              "alpha_s_from_historical_w0_and_frozen_DE":(3*base["omega_de"]*(1-.770)/2)**1.5,
              "scope":"alternative dynamical-DE readout, not inserted into flat-LCDM posterior test; no joint w0-wa score without covariance"},
        "source_ids":[s["id"] for s in sources],
        "primary_references": {
            "Planck2018_table2":"https://arxiv.org/html/1807.06209v4",
            "DESI_DR2_II":"https://arxiv.org/html/2503.14738v3",
            "muon_WP25":"https://muon-gm2-theory.illinois.edu/white-paper-25/",
            "muon_final2025":"https://muon-g-2.fnal.gov/result2025.pdf",
            "muon_BMW_DMZ_2026_v2":"https://arxiv.org/html/2407.10913v2",
            "vector_loop_Eq3":"https://arxiv.org/pdf/hep-ph/0102222",
            "scalar_loop_Eq49":"https://arxiv.org/html/1712.10022v2",
        },
    }
    (HERE/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"forward":base,"checks":len(result["verification"]),
                      "desi_chi2":result["desi_joint_compressed"]["chi2_fixed_inputs"],
                      "muon_constraints":result["new_muon_inverse_constraints"]},indent=2))


if __name__ == "__main__":
    main()
