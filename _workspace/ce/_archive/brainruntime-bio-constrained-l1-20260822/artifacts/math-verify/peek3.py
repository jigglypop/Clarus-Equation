import json, os
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, "search3.json")))
print("n_eval", d["n_eval"], "min_V", round(d["min_V"], 5), "max_npass", d["max_npass"])
for k, v in d["miss_log"].items():
    print(" %-10s value=%.4g band=%s log_miss=%.4f" % (k, v["value"], v["band"], v["log_miss"]))
print("--- best point params/metrics ---")
b = d["best"][0]
KEYS = ["eta", "lam0", "kappa", "rho_inf", "kappa_m", "T_m", "Sstar", "g1g0"]
print({k: round(b[k], 5) for k in KEYS})
for k in ("R1_A", "R1_B", "R1_Apm", "R1_Bpm", "R2dev_Na", "R2ad_Na", "R2dev_Nb",
          "R2ad_Nb", "R3a", "R3b", "R4", "R5", "R6", "n_R6_event", "f_top",
          "theta_G", "gamma", "c_hom", "sigma_logw", "E1_skew_w", "E1_skew_logw",
          "E2_skew_rate_proxy", "E2_cv_rate_proxy", "N_adult", "wbar_adult"):
    print("  %-20s %.5g" % (k, b[k]))
print("gates", b["gates"])
print("--- npass across 6 refined ---", [(round(r["V"], 4), r["npass"]) for r in d["best"]])
