"""Stage-3: fine local search from the stage-2 V-front, plus a
multi-seed-robust variant (V averaged over 3 calibration seeds)."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
from search import SPEC, KEYS, to_p, clip, SEED_SEARCH
from search2 import V

H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, "search2.json")))
starts = [np.array([r[k] for k in KEYS]) for r in d["best"][:8]]
rng = np.random.default_rng(20260825)
res, t0, nev = [], time.time(), 0
for si, x0 in enumerate(starts):
    cur = clip(np.asarray(x0, float))
    m = S.run(to_p(cur), SEED_SEARCH); nev += 1
    cv, step = V(m), 0.15
    for it in range(180):
        y = cur.copy()
        for jj in rng.choice(len(KEYS), size=int(rng.integers(1, 4)), replace=False):
            if SPEC[jj][3]:
                y[jj] = cur[jj] * np.exp(rng.normal(0, step))
            else:
                y[jj] = cur[jj] + rng.normal(0, step * (SPEC[jj][2] - SPEC[jj][1]) * 0.2)
        y = clip(y)
        m2 = S.run(to_p(y), SEED_SEARCH); nev += 1
        v2 = V(m2)
        if v2 < cv: cur, cv, m = y, v2, m2
        if it % 45 == 44: step *= 0.7
    rec = {k: float(v) for k, v in zip(KEYS, cur)}
    rec.update({k: float(v) for k, v in m.items()})
    rec["V"] = cv
    rec["npass"] = int(sum(S.gates_pass(m).values()))
    rec["gates"] = {k: bool(v) for k, v in S.gates_pass(m).items()}
    res.append(rec)
    print("start %d V %.5f npass %d (%d ev %.0fs)" % (si, cv, rec["npass"], nev,
                                                      time.time() - t0), flush=True)
res.sort(key=lambda r: (r["V"], -r["npass"]))
out = {"n_eval": nev, "min_V": res[0]["V"], "max_npass": max(r["npass"] for r in res),
       "best": res[:6]}
# per-gate log-distance to the band at the best point
b = res[0]
BAND = {"R1_A": (0.02, 0.08), "R2dev_Na": (0.25, 0.45), "R2ad_Na": (0.60, 0.85),
        "R3a": (0.10, 0.25), "R3b": (0.0, 0.05), "R4": (0.0, 0.05),
        "R5": (1.3, 1.8), "R6": (1.3, np.inf)}
out["miss_log"] = {}
for k, (lo, hi) in BAND.items():
    v = b[k]
    miss = 0.0
    if v < lo: miss = float(np.log(lo / max(v, 1e-9)))
    elif np.isfinite(hi) and v > hi: miss = float(np.log(v / hi))
    out["miss_log"][k] = {"value": float(v), "band": [lo, float(hi)], "log_miss": miss}
json.dump(out, open(os.path.join(H, "search3.json"), "w"), indent=1, default=float)
print(json.dumps({"n_eval": nev, "min_V": out["min_V"], "max_npass": out["max_npass"],
                  "miss_log": out["miss_log"]}, indent=1, default=float))
