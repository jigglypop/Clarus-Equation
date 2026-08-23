"""Gauge check: the day map is 1-homogeneous in (w, w_min, w0, eta, kappa, Sstar).
Fixing w_min == 1 removes exactly one dof; no residual scale ridge should remain
among the free 8.  Test: scale ALL of (w_min, w0, eta, kappa, Sstar) by c and
verify every dimensionless gate is invariant."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
H = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(H, "base_point.json")))["params"]
KEYS = ["R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R3b", "R4", "R5", "R6",
        "sigma_logw", "E1_skew_logw", "f_top", "theta_G", "gamma", "c_hom"]
ref = S.run(P, 119001)
out = {"ref": {k: float(ref[k]) for k in KEYS}, "scaled": {}, "max_rel_dev": {}}
for c in (0.37, 3.7):
    S.WMIN = 1.0 * c
    q = dict(P, eta=P["eta"] * c, kappa=P["kappa"] * c, Sstar=P["Sstar"] * c)
    m = S.run(q, 119001, w0=1.2 * c)
    out["scaled"]["c=%.2f" % c] = {k: float(m[k]) for k in KEYS}
    dev = max(abs(float(m[k]) - float(ref[k])) / max(abs(float(ref[k])), 1e-12)
              for k in KEYS if np.isfinite(ref[k]) and np.isfinite(m[k]))
    out["max_rel_dev"]["c=%.2f" % c] = dev
S.WMIN = 1.0
json.dump(out, open(os.path.join(H, "gauge.json"), "w"), indent=1)
print(json.dumps(out["max_rel_dev"], indent=1))
print(json.dumps(out["ref"], indent=1))
