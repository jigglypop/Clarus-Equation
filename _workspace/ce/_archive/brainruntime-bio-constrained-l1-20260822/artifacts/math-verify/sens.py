"""(e) design-constant sensitivity + scale-ridge (identifiability) test.
Design constants declared 'fit-forbidden' in contract 3.5: w0, tau_e, 16:8,
K constants, tau_el, delay law, NE/NI, judgement windows.  Undeclared: the
homeostatic gain beta and the eligibility dispersion (K shape)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S

H = os.path.dirname(os.path.abspath(__file__))
KEYS = ["R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R3b", "R4", "R5", "R6",
        "sigma_logw", "E1_skew_logw", "E1_skew_w", "E2_skew_rate_proxy",
        "E2_cv_rate_proxy", "f_top", "theta_G", "gamma", "c_hom", "N_adult",
        "wbar_adult"]
BASE = json.load(open(os.path.join(H, "base_point.json")))
P = BASE["params"]
SEED = 119001


def one(**kw):
    m = S.run(dict(P, **kw.pop("p", {})), SEED, **kw)
    return {k: float(m.get(k, float("nan"))) for k in KEYS} | {
        "npass": int(sum(S.gates_pass(m).values()))}

out = {"base_params": P, "base": one()}
out["w0"] = {str(v): one(w0=v) for v in (1.05, 1.2, 1.5, 2.0, 3.0)}
out["tau_e"] = {str(v): one(tau_e=v) for v in (1, 2, 3, 5)}
# 16:8 enters only as a scale on the daily eligibility mass -> eta*f
out["wake_fraction_16_8"] = {"%.2f" % f: one(p={"eta": P["eta"] * f})
                             for f in (0.5, 0.75, 1.0, 1.5)}
# K-kernel dispersion (A+-,tau+-,tau_el fix the eligibility distribution shape)
out["gain_shape"] = {str(v): one(gain_shape=v) for v in (0.5, 1.0, 2.0, 5.0, 20.0)}
# homeostatic gain beta: NOT declared anywhere in contract 3.4
out["beta_homeostatic_gain"] = {str(v): one(beta=v) for v in (0.2, 0.5, 0.8, 1.0)}
# network size
out["NE"] = {str(v): one(NE=v) for v in (48, 64, 90)}
# judgement windows (contract section 6, declared design constants)
out["adult_window"] = {"400-600": one(adult=(400, 600)), "500-700": one(adult=(500, 700))}
out["dev_window"] = {"20-70": one(dev=(20, 70)), "30-80": one(dev=(30, 80)),
                     "10-60": one(dev=(10, 60))}
# ---- scale ridge: (eta, kappa, Sstar) -> c*(...) --------------------------
out["scale_ridge"] = {}
for c in (0.5, 1.0, 2.0, 4.0):
    q = dict(P, eta=P["eta"] * c, kappa=P["kappa"] * c, Sstar=P["Sstar"] * c)
    m = S.run(q, SEED)
    out["scale_ridge"]["c=%.2f" % c] = {k: float(m.get(k, float("nan"))) for k in KEYS}
json.dump(out, open(os.path.join(H, "sens.json"), "w"), indent=1, default=float)
def row(tag, d):
    print("%-22s " % tag + "  ".join("%s=%.4g" % (k, d[k]) for k in
          ("R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R3b", "R4", "R5", "R6", "npass")
          if k in d))
print("BASE"); row("base", out["base"])
for grp in ("w0", "tau_e", "wake_fraction_16_8", "gain_shape",
            "beta_homeostatic_gain", "NE", "dev_window", "adult_window"):
    print("== " + grp)
    for k, v in out[grp].items(): row("  " + k, v)
print("== scale_ridge (eta,kappa,Sstar) x c")
for k, v in out["scale_ridge"].items(): row("  " + k, v)
