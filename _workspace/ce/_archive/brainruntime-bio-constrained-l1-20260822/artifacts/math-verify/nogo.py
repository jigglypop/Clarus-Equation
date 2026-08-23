"""Theorem V3-1 (iii) counterexample: lambda(w)>0 everywhere is NOT sufficient
for a cyclic steady state.  With homeostasis switched off (beta=0, c==1) and
mean daily gain Delta_bar > lambda_0*kappa/2, total strength diverges."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
H = os.path.dirname(os.path.abspath(__file__))
P = {k: float(v) for k, v in
     json.load(open(os.path.join(H, "search3.json")))["best"][0].items()
     if k in ("eta", "lam0", "kappa", "rho_inf", "kappa_m", "T_m", "Sstar", "g1g0")}
out = {"params": P, "Delta_bar": P["eta"],
       "loss_cap_lam0_kappa_over_2": P["lam0"] * P["kappa"] / 2,
       "condition_Delta_gt_cap": P["eta"] > P["lam0"] * P["kappa"] / 2, "runs": {}}
for beta in (0.0, 0.05, 0.2, 1.0):
    m = S.run(P, 119001, beta=beta)
    out["runs"]["beta=%.2f" % beta] = {
        k: float(m[k]) for k in ("R3a", "R3b", "R4", "R1_A", "R2ad_Na", "R5",
                                 "wbar_adult", "N_adult", "c_hom", "gamma", "f_top")}
# explicit divergence: mean strength vs time with beta=0 (kappa small, lam0 large)
for tag, q in (("base", P), ("small_kappa", dict(P, kappa=0.5)),
               ("large_kappa", dict(P, kappa=200.0))):
    m = S.run(q, 119001, beta=0.0)
    out["runs"]["beta=0,%s" % tag] = {"wbar_adult": float(m["wbar_adult"]),
                                      "R3a": float(m["R3a"]), "R4": float(m["R4"]),
                                      "loss_cap": q["lam0"] * q["kappa"] / 2,
                                      "Delta_bar": q["eta"]}
json.dump(out, open(os.path.join(H, "nogo.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
