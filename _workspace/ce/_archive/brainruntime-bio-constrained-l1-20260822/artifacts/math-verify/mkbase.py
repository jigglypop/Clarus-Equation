import json, os
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, "search2.json")))
KEYS = ["eta", "lam0", "kappa", "rho_inf", "kappa_m", "T_m", "Sstar", "g1g0"]
b = d["best"][0]
json.dump({"params": {k: float(b[k]) for k in KEYS},
           "source": "search2.json best[0] (min V), calibration seed 119001",
           "V": b["V"], "npass": b["npass"],
           "npass_all_refined": max(r["npass"] for r in d["best"]),
           "gates": b["gates"]},
          open(os.path.join(H, "base_point.json"), "w"), indent=1)
print(json.dumps(json.load(open(os.path.join(H, "base_point.json"))), indent=1))
print("top-10 (V, npass):", [(round(r["V"], 4), r["npass"]) for r in d["best"]])
