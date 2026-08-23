"""(1) validate closed-form identity (I) inside its stated preconditions,
(2) local refinement from the LHS front, (3) noise-seed re-evaluation."""
import os, sys, json, time, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
from search import SPEC, KEYS, to_p, clip, SEED_SEARCH, SEEDS_NOISE

H = os.path.dirname(os.path.abspath(__file__))
rows = []
with open(os.path.join(H, "lhs.csv")) as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

out = {}
# ---- (1) identity (I): (1+gamma_top)(1-s_top) = (1+gamma)(1-s) -----------
# preconditions: cyclic stationarity (small mass/shape drift) AND negligible
# birth-death mass flux (low turnover) AND low top-20% churn.
def ident_err(r):
    s, st, ft, tg, gm = r["R3a"], r["R3b"], r["f_top"], r["theta_G"], r["gamma"]
    rho = (1 - s) / (1 - st); psi = tg / ft
    if rho <= psi or gm <= 0: return None
    return abs((1 - rho) / (rho - psi) - gm) / gm
for tag, cond in (("all", lambda r: True),
                  ("turnover<0.2", lambda r: r["R1_A"] < 0.2),
                  ("turnover<0.08", lambda r: r["R1_A"] < 0.08),
                  ("turnover<0.08 & |drift_ftop|<0.5",
                   lambda r: r["R1_A"] < 0.08 and abs(r["drift_ftop"]) < 0.5)):
    e = [ident_err(r) for r in rows
         if all(np.isfinite([r["R3a"], r["R3b"], r["f_top"], r["theta_G"],
                             r["gamma"], r["R1_A"], r["drift_ftop"]])) and cond(r)]
    e = [x for x in e if x is not None]
    out.setdefault("identity", {})[tag] = {
        "n": len(e), "median": float(np.median(e)) if e else None,
        "p90": float(np.quantile(e, 0.9)) if e else None}
# ---- (2) local refinement -----------------------------------------------
def score(m):
    ok = S.gates_pass(m)
    return (sum(ok.values()), -S.loss(m))
front = sorted(rows, key=lambda r: (-r["npass"], r["loss"]))[:20]
rng = np.random.default_rng(20260823)
best = []
t0 = time.time(); nev = 0
for st in front:
    x = np.array([st[k] for k in KEYS])
    m = S.run(to_p(x), SEED_SEARCH); nev += 1
    cur, cs = x, score(m)
    step = 0.35
    for it in range(40):
        y = cur.copy()
        j = rng.integers(0, len(KEYS), size=rng.integers(1, 4))
        for jj in j:
            lg = SPEC[jj][3]
            y[jj] = cur[jj] * np.exp(rng.normal(0, step)) if lg else \
                cur[jj] + rng.normal(0, step * (SPEC[jj][2] - SPEC[jj][1]) * 0.25)
        y = clip(y)
        m2 = S.run(to_p(y), SEED_SEARCH); nev += 1
        s2 = score(m2)
        if s2 > cs:
            cur, cs, m = y, s2, m2
        if it % 15 == 14: step *= 0.6
    mm = S.run(to_p(cur), SEED_SEARCH); nev += 1
    rec = {k: float(v) for k, v in zip(KEYS, cur)}
    rec.update({k: float(v) for k, v in mm.items()})
    rec["npass"] = int(sum(S.gates_pass(mm).values()))
    rec["loss"] = S.loss(mm)
    rec["gates"] = {k: bool(v) for k, v in S.gates_pass(mm).items()}
    best.append(rec)
    print("start npass=%d -> %d  loss %.3f  (%d ev, %.0fs)"
          % (st["npass"], rec["npass"], rec["loss"], nev, time.time() - t0), flush=True)
best.sort(key=lambda r: (-r["npass"], r["loss"]))
out["refine_evals"] = nev
out["refine_best"] = best[:8]
out["refine_max_npass"] = max(r["npass"] for r in best)
# ---- (3) noise seeds on the top 3 ---------------------------------------
out["noise"] = []
for rec in best[:3]:
    p = {k: rec[k] for k in KEYS}
    per = []
    for sd in [SEED_SEARCH] + SEEDS_NOISE:
        m = S.run(p, sd)
        per.append({"seed": sd, "npass": int(sum(S.gates_pass(m).values())),
                    **{k: float(m[k]) for k in ("R1_A", "R2dev_Na", "R2ad_Na", "R3a",
                                                "R3b", "R4", "R5", "R6", "sigma_logw",
                                                "E1_skew_logw", "E1_skew_w",
                                                "E2_skew_rate_proxy")}})
    out["noise"].append({"params": p, "per_seed": per})
with open(os.path.join(H, "refine.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps({"identity": out["identity"], "refine_max_npass": out["refine_max_npass"],
                  "refine_evals": nev}, indent=1))
