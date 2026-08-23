"""BA-V3-1 (a)+(c-analytic): closed-form gate-set consistency.

Independent re-derivation from definitions. Produces budget.json.
"""
import json, math
import numpy as np

SQ2 = math.sqrt(2.0)
def Phi(x):  return 0.5 * (1.0 + math.erf(x / SQ2))
def Phi_inv(p):
    lo, hi = -12.0, 12.0
    for _ in range(300):
        m = 0.5 * (lo + hi)
        if Phi(m) < p: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

out = {}

def budget(s, s_top, theta_G=0.2):
    rho = (1.0 - s) / (1.0 - s_top)
    lo_ratio = (1.0 - s_top) / (1.0 - s)
    hi_ratio = s * (1.0 - s_top) / (s_top * (1.0 - s)) if s_top > 0 else float("inf")
    res = {"s": s, "s_top": s_top, "theta_G": theta_G, "rho": rho,
           "ratio_window_ftop_over_thetaG": [lo_ratio, hi_ratio],
           "ftop_window": [lo_ratio * theta_G, min(1.0, hi_ratio * theta_G)],
           "ftop_window_raw_hi": hi_ratio * theta_G}
    for tag, ft in (("lo", res["ftop_window"][0]), ("hi", res["ftop_window"][1])):
        psi = theta_G / ft
        g = (1.0 - rho) / (rho - psi) if rho > psi else float("inf")
        res["gamma_at_ftop_" + tag] = g
        res["c_at_ftop_" + tag] = 1.0 / ((1.0 + g) * (1.0 - s)) if math.isfinite(g) else 0.0
    res["gamma_min_for_c_lt_1"] = s / (1.0 - s)
    z80 = Phi_inv(0.8)
    sw = []
    for ft in res["ftop_window"]:
        ft = min(max(ft, 1e-9), 1 - 1e-9)
        sw.append(Phi_inv(ft) + z80)
    res["z80"] = z80
    res["sigma_logw_window"] = sw
    return res

out["B1_target"] = budget(0.18, 0.05)
out["B1_corners"] = {"s%s_stop%s" % (s, st): budget(s, st)
                     for s in (0.10, 0.18, 0.25) for st in (0.005, 0.02, 0.05)}
out["B1_thetaG_scan"] = {"theta_G=%.2f" % tg: budget(0.18, 0.05, tg)["ftop_window"]
                         for tg in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}

def lam0_min(s, s_top, theta_G=0.2):
    a = theta_G * (1.0 - s_top) / (1.0 - s)
    return (s - a * s_top) / (1.0 - a)
out["B2_lambda0_min"] = {"s%s_stop%s" % (s, st): lam0_min(s, st)
                         for s in (0.10, 0.18, 0.25) for st in (0.0, 0.02, 0.05)}
def s_bot(s, s_top, f_top): return (s - f_top * s_top) / (1.0 - f_top)
out["B2_s_bot"] = {"f_top=%s" % ft: s_bot(0.18, 0.05, ft)
                   for ft in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)}
out["B2_knee"] = {}
for lam0 in (0.3, 0.5, 0.7, 0.9):
    row = {}
    for ft in (0.4, 0.5, 0.6):
        sb = s_bot(0.18, 0.05, ft)
        if sb >= lam0:
            row["f_top=%s" % ft] = "infeasible (s_bot>=lambda_0)"
            continue
        xt = math.sqrt(max(lam0 / 0.05 - 1.0, 0.0))
        xb = math.sqrt(max(lam0 / sb - 1.0, 0.0))
        row["f_top=%s" % ft] = {"s_bot": sb, "w_top_over_kappa_min": xt,
                                "w_bot_over_kappa": xb,
                                "required_wtop_over_wbot": xt / xb if xb > 0 else float("inf"),
                                "lognormal_wtop_over_wbot": (5 * ft) / ((1 - ft) / 0.8)}
    out["B2_knee"]["lambda_0=%s" % lam0] = row

def r1_readings(tau_p, p_p):
    A = 30.0 / tau_p
    B = 30.0 / (p_p * tau_p)
    return {"A_removal_only_persistent_numer": A,
            "B_removal_only_all_numer": B,
            "A_pm_removal_plus_formation": 2 * A,
            "B_pm_removal_plus_formation": 2 * B}
out["B3_R1_readings"] = {}
for tau_p in (375.0, 750.0, 1500.0):
    for p_p in (0.60, 0.73, 0.85):
        out["B3_R1_readings"]["tau_p=%s,p_p=%s" % (tau_p, p_p)] = r1_readings(tau_p, p_p)
out["B3_tau_p_band"] = {
    "reading_A": [30 / 0.08, 30 / 0.02],
    "reading_B_p_p_0.73": [30 / (0.08 * 0.73), 30 / (0.02 * 0.73)],
    "grutzendler_halflife_gt_13mo_implies_tau_p_gt": 13 * 30.0 / math.log(2)}
band = (0.02, 0.08)
grid = []
for tau_p in np.linspace(200, 6000, 600):
    for p_p in np.linspace(0.60, 0.85, 26):
        r = r1_readings(tau_p, p_p)
        grid.append((tau_p, p_p, sum(band[0] <= v <= band[1] for v in r.values())))
grid = np.array(grid)
out["B3_max_readings_in_band"] = int(grid[:, 2].max())
best = grid[grid[:, 2] == grid[:, 2].max()]
out["B3_argmax_example"] = {"tau_p": float(best[0, 0]), "p_p": float(best[0, 1]),
                            "readings": r1_readings(float(best[0, 0]), float(best[0, 1]))}
out["B3_at_target_A_0.04"] = {"p_p=%.2f" % p: r1_readings(750.0, p) for p in (0.60, 0.73, 0.85)}

def n_a(q_p, tau_t, tau_p):
    return q_p * math.exp(-8 / tau_p) + (1 - q_p) * math.exp(-8 / tau_t)
def n_b(q_p, tau_t, tau_p, W):
    cP, cT = q_p * (W + tau_p), (1 - q_p) * (W + tau_t)
    return (cP * math.exp(-8 / tau_p) + cT * math.exp(-8 / tau_t)) / (cP + cT)
def solve_qp(target, f, *a):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if f(m, *a) < target: lo = m
        else: hi = m
    return 0.5 * (lo + hi)
out["B4"] = {}
for tau_t in (1.5, 3.0):
    d = {"dev_N_a_qp_for_0.35": solve_qp(0.35, n_a, tau_t, 750.0),
         "dev_N_b_qp_for_0.35_W50": solve_qp(0.35, n_b, tau_t, 750.0, 50.0),
         "adult_N_a_qp_for_0.73": solve_qp(0.73, n_a, tau_t, 750.0),
         "adult_N_b_qp_for_0.73_W200": solve_qp(0.73, n_b, tau_t, 750.0, 200.0)}
    d["maturation_contrast_N_a"] = d["adult_N_a_qp_for_0.73"] / d["dev_N_a_qp_for_0.35"]
    d["maturation_contrast_N_b"] = d["adult_N_b_qp_for_0.73_W200"] / d["dev_N_b_qp_for_0.35_W50"]
    out["B4"]["tau_t=%s" % tau_t] = d

def Lmax_num(lam0, kappa):
    w = np.linspace(1e-6, 200 * kappa, 400001)
    L = lam0 * w / (1 + (w / kappa) ** 2)
    return float(L.max()), float(w[int(L.argmax())])
out["B5_absolute_loss_cap"] = {}
for lam0, kap in ((0.5, 8.0), (0.9, 3.0), (0.22, 30.0)):
    num, arg = Lmax_num(lam0, kap)
    out["B5_absolute_loss_cap"]["lam0=%s,kappa=%s" % (lam0, kap)] = {
        "numeric_max": num, "argmax": arg, "closed_form_lam0_kappa_over_2": lam0 * kap / 2,
        "rel_err": abs(num - lam0 * kap / 2) / (lam0 * kap / 2)}
def Sprime_min(lam0, kappa):
    v = np.linspace(0, 50 * kappa, 200001)
    x = (v / kappa) ** 2
    return float((1 - lam0 * (1 - x) / (1 + x) ** 2).min())
out["B5_S_monotone_min_deriv"] = {"lam0=%s" % l: Sprime_min(l, 5.0) for l in (0.2, 0.5, 0.9, 0.99)}
out["B5_c1_divergence_condition"] = ("Delta_bar > lam0*kappa/2 implies "
    "E[w_next] >= w + (Delta_bar - lam0*kappa/2): no cyclic steady state for ANY (lam0,kappa)")

out["B6_c_contrast"] = {"gamma_ad=%s" % g: (1 + g) / (1 + 1.5 * g)
                        for g in (0.22, 0.3, 0.5, 1.0, 2.0, 5.0, 20.0)}
out["B6_c_contrast_limit"] = 2.0 / 3.0
out["B6_required_R2_contrast"] = {"target": 0.73 / 0.35, "band_min": 0.60 / 0.45,
                                  "band_max": 0.85 / 0.25}
out["B6_hazard_gap_per_day"] = {"ratio=%.3f" % r: math.log(r) / 8.0
                                for r in (0.73 / 0.35, 0.60 / 0.45, 0.85 / 0.25)}

with open(__file__.replace("budget.py", "budget.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
