"""(d) local identifiability at the stage-3 best point: common-random-number
central-difference Jacobian of log-gate-values w.r.t. log-parameters, with
seed-noise normalisation (calibration seeds 119001..119006 only)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
from search import KEYS, SEED_SEARCH, SEEDS_NOISE, SPEC
H = os.path.dirname(os.path.abspath(__file__))
b = json.load(open(os.path.join(H, "search3.json")))["best"][0]
P = {k: float(b[k]) for k in KEYS}
G = ["R1_A", "R2dev_Na", "R2ad_Na", "R3a", "R3b", "R4", "R5", "R6"]
SEED = SEED_SEARCH
h = 0.08


def vals(q, seed=SEED):
    m = S.run(q, seed)
    return np.array([np.log(max(m[k], 1e-9)) for k in G])


J = np.zeros((len(G), len(KEYS)))
for j, k in enumerate(KEYS):
    qp, qm = dict(P), dict(P)
    qp[k] = P[k] * np.exp(h); qm[k] = P[k] * np.exp(-h)
    J[:, j] = (vals(qp) - vals(qm)) / (2 * h)
bad = ~np.isfinite(J).all(axis=1)
G2 = [g for g, bb in zip(G, bad) if not bb]
J = J[~bad]
noise_all = np.std(np.array([vals(P, sd) for sd in [SEED] + SEEDS_NOISE]), axis=0, ddof=1)
noise = noise_all[~bad]
Jn = J / np.maximum(noise, 1e-9)[:, None]
u, s, vt = np.linalg.svd(J)
un, sn, vtn = np.linalg.svd(Jn)
out = {"point": P, "gates_used": G2, "gates_dropped_nonfinite": [g for g, bb in zip(G, bad) if bb], "params": KEYS,
       "seed_noise_log_sd": {k: float(v) for k, v in zip(G2, noise)},
       "sv_raw": s.tolist(), "cond_raw": float(s[0] / s[-1]),
       "sv_noise_normalised": sn.tolist(),
       "rank_1sigma": int((sn > 1.0).sum()),
       "weakest_direction": {k: float(v) for k, v in zip(KEYS, vtn[-1])},
       "second_weakest": {k: float(v) for k, v in zip(KEYS, vtn[-2])},
       "third_weakest": {k: float(v) for k, v in zip(KEYS, vtn[-3])},
       "J_log": {g: {k: float(J[i, j]) for j, k in enumerate(KEYS)}
                 for i, g in enumerate(G2)}}
json.dump(out, open(os.path.join(H, "jac.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("seed_noise_log_sd", "sv_raw", "cond_raw",
                                      "sv_noise_normalised", "rank_1sigma",
                                      "weakest_direction", "second_weakest",
                                      "third_weakest")}, indent=1))
