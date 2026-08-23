"""Quenched wake-gain heterogeneity sigma_h (surrogate extension, NOT a contract
parameter): tests the diagnosis that the blocking quantity is the emergent
width sigma_logw of the weight distribution."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
from search import KEYS, to_p, clip, SPEC, SEED_SEARCH
from search2 import V
H = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(H, "search3.json")))["best"][0]
P = {k: float(P[k]) for k in KEYS}
out = {"base_params": P, "scan": {}}
for sh in (0.0, 0.5, 1.0, 1.5, 2.0):
    m = S.run(P, SEED_SEARCH, sigma_h=sh)
    out["scan"]["sigma_h=%.1f" % sh] = {
        k: float(m[k]) for k in ("R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R3b", "R4",
                                 "R5", "R6", "sigma_logw", "E1_skew_logw", "E1_skew_w",
                                 "f_top", "gamma", "c_hom")} | {
        "npass": int(sum(S.gates_pass(m).values())), "V": V(m)}
# short local search at sigma_h = 1.0
rng = np.random.default_rng(20260826)
best = None
for st in range(6):
    cur = clip(np.array([P[k] for k in KEYS]) * np.exp(rng.normal(0, 0.25, len(KEYS))))
    m = S.run(to_p(cur), SEED_SEARCH, sigma_h=1.0)
    cv, step = V(m), 0.3
    for it in range(90):
        y = cur.copy()
        for jj in rng.choice(len(KEYS), size=int(rng.integers(1, 4)), replace=False):
            y[jj] = cur[jj] * np.exp(rng.normal(0, step)) if SPEC[jj][3] else \
                cur[jj] + rng.normal(0, step * (SPEC[jj][2] - SPEC[jj][1]) * 0.2)
        y = clip(y)
        m2 = S.run(to_p(y), SEED_SEARCH, sigma_h=1.0)
        if V(m2) < cv: cur, cv, m = y, V(m2), m2
        if it % 25 == 24: step *= 0.7
    rec = {k: float(v) for k, v in zip(KEYS, cur)}
    rec.update({k: float(m[k]) for k in ("R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R3b",
                                         "R4", "R5", "R6", "sigma_logw", "E1_skew_logw",
                                         "E1_skew_w", "f_top", "gamma", "c_hom")})
    rec["V"] = cv; rec["npass"] = int(sum(S.gates_pass(m).values()))
    rec["gates"] = {k: bool(v) for k, v in S.gates_pass(m).items()}
    print("het start %d V %.4f npass %d" % (st, cv, rec["npass"]), flush=True)
    if best is None or cv < best["V"]: best = rec
out["local_search_sigma_h_1.0"] = best
json.dump(out, open(os.path.join(H, "het.json"), "w"), indent=1, default=float)
print(json.dumps(out["scan"], indent=1, default=float))
print("BEST at sigma_h=1.0:", json.dumps(best, indent=1, default=float))
