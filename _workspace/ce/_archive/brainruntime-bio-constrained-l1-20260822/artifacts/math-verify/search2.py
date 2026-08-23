"""Stage-2 targeted search: multistart (1+1)-ES on a smooth band-violation
score V (0 iff all 8 L1 gates + the R2 monotonicity clause hold).
Calibration seed 119001 only."""
import os, sys, json, time, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
from search import SPEC, KEYS, to_p, clip, lhs, SEED_SEARCH, SEEDS_NOISE

H = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6
BAND = {"R1_A": (0.02, 0.08), "R2dev_Na": (0.25, 0.45), "R2ad_Na": (0.60, 0.85),
        "R3a": (0.10, 0.25), "R3b": (0.0, 0.05), "R4": (0.0, 0.05),
        "R5": (1.3, 1.8), "R6": (1.3, np.inf)}


def V(m):
    tot = 0.0
    for k, (lo, hi) in BAND.items():
        v = m.get(k, np.nan)
        if not np.isfinite(v): return 1e9
        if v < lo: tot += np.log((lo + EPS) / (v + EPS)) ** 2
        elif np.isfinite(hi) and v > hi: tot += np.log((v + EPS) / (hi + EPS)) ** 2
    a, b = m.get("R2ad_Na", np.nan), m.get("R2dev_Na", np.nan)
    if not (np.isfinite(a) and np.isfinite(b)): return 1e9
    if a <= b: tot += np.log((b + EPS) / (a + EPS)) ** 2
    return float(tot)


def main(nstart=60, iters=50):
    rows = [{k: float(v) for k, v in r.items()}
            for r in csv.DictReader(open(os.path.join(H, "lhs.csv")))]
    for r in rows:
        r["_V"] = V(r)
    rows.sort(key=lambda r: r["_V"])
    starts = [np.array([r[k] for k in KEYS]) for r in rows[:nstart - 20]]
    rng = np.random.default_rng(20260824)
    starts += [x for x in lhs(20, rng)]
    best, t0, nev = [], time.time(), 0
    for si, x0 in enumerate(starts):
        cur = clip(np.asarray(x0, float))
        m = S.run(to_p(cur), SEED_SEARCH); nev += 1
        cv, step = V(m), 0.45
        for it in range(iters):
            y = cur.copy()
            for jj in rng.choice(len(KEYS), size=int(rng.integers(1, 4)), replace=False):
                if SPEC[jj][3]:
                    y[jj] = cur[jj] * np.exp(rng.normal(0, step))
                else:
                    y[jj] = cur[jj] + rng.normal(0, step * (SPEC[jj][2] - SPEC[jj][1]) * 0.3)
            y = clip(y)
            m2 = S.run(to_p(y), SEED_SEARCH); nev += 1
            v2 = V(m2)
            if v2 < cv: cur, cv, m = y, v2, m2
            if it % 12 == 11: step *= 0.65
        rec = {k: float(v) for k, v in zip(KEYS, cur)}
        rec.update({k: float(v) for k, v in m.items()})
        rec["V"] = cv
        rec["npass"] = int(sum(S.gates_pass(m).values()))
        rec["gates"] = {k: bool(v) for k, v in S.gates_pass(m).items()}
        best.append(rec)
        if si % 5 == 0:
            print("start %d/%d V=%.4f npass=%d (%d ev, %.0fs)"
                  % (si, len(starts), cv, rec["npass"], nev, time.time() - t0), flush=True)
    best.sort(key=lambda r: (r["V"], -r["npass"]))
    out = {"n_eval": nev, "n_start": len(starts), "iters": iters,
           "best": best[:10],
           "min_V": best[0]["V"], "max_npass": max(r["npass"] for r in best),
           "V_by_gate_at_best": {}}
    b0 = best[0]
    for k, (lo, hi) in BAND.items():
        v = b0.get(k, np.nan)
        out["V_by_gate_at_best"][k] = {"value": v, "band": [lo, None if not np.isfinite(hi) else hi],
                                       "in": bool(lo <= v <= hi)}
    out["noise"] = []
    for rec in best[:3]:
        p = {k: rec[k] for k in KEYS}
        per = []
        for sd in [SEED_SEARCH] + SEEDS_NOISE:
            m = S.run(p, sd)
            per.append({"seed": sd, "V": V(m), "npass": int(sum(S.gates_pass(m).values())),
                        **{k: float(m[k]) for k in ("R1_A", "R1_B", "R1_Apm", "R1_Bpm",
                                                    "R2dev_Na", "R2ad_Na", "R2dev_Nb",
                                                    "R2ad_Nb", "R3a", "R3b", "R4", "R5",
                                                    "R6", "n_R6_event", "sigma_logw",
                                                    "E1_skew_w", "E1_skew_logw",
                                                    "E2_skew_rate_proxy", "E2_cv_rate_proxy",
                                                    "f_top", "theta_G", "gamma", "c_hom")}})
        out["noise"].append({"params": p, "per_seed": per})
    json.dump(out, open(os.path.join(H, "search2.json"), "w"), indent=1, default=float)
    print(json.dumps({"n_eval": nev, "min_V": out["min_V"], "max_npass": out["max_npass"],
                      "V_by_gate_at_best": out["V_by_gate_at_best"]}, indent=1, default=float))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60,
         int(sys.argv[2]) if len(sys.argv) > 2 else 50)
