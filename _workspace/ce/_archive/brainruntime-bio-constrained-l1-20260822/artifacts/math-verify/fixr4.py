"""UNIT CORRECTION: R4 is measured as (slope*100/mean) = relative drift over
100 days expressed as a FRACTION, so the section-5 band '<5%/100d' is 0.05,
not 5.0.  surrogate.BANDS / search2.BAND used 5.0.  This script (a) patches the
constant in both files, (b) recomputes every gate statistic from the stored
lhs.csv without re-simulating."""
import os, re, csv, json, itertools
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
for fn, old, new in (("surrogate.py", '"R4": (0.0, 5.0)', '"R4": (0.0, 0.05)'),
                     ("surrogate.py", '"R4": ("le", 5.0)', '"R4": ("le", 0.05)'),
                     ("search2.py", '"R4": (0.0, 5.0)', '"R4": (0.0, 0.05)'),
                     ("search3.py", '"R4": (0.0, 5.0)', '"R4": (0.0, 0.05)')):
    p = os.path.join(H, fn); s = open(p).read()
    if old in s:
        open(p, "w").write(s.replace(old, new))
        print("patched", fn, old)
rows = [{k: float(v) for k, v in r.items()}
        for r in csv.DictReader(open(os.path.join(H, "lhs.csv")))]
BAND = {"R1_A": (0.02, 0.08), "R2dev_Na": (0.25, 0.45), "R2ad_Na": (0.60, 0.85),
        "R3a": (0.10, 0.25), "R3b": (1e-12, 0.05), "R4": (0.0, 0.05),
        "R5": (1.3, 1.8), "R6": (1.3, np.inf)}
def gp(r):
    g = {k: bool(np.isfinite(r[k]) and lo <= r[k] <= hi) for k, (lo, hi) in BAND.items()}
    g["R2_monotone"] = bool(np.isfinite(r["R2ad_Na"]) and np.isfinite(r["R2dev_Na"])
                            and r["R2ad_Na"] > r["R2dev_Na"])
    return g
G = [gp(r) for r in rows]
ks = list(G[0])
out = {"n": len(rows), "per_gate": {k: int(sum(g[k] for g in G)) for k in ks},
       "all_L1": int(sum(all(g.values()) for g in G)),
       "npass_hist": {str(k): int(sum(1 for g in G if sum(g.values()) == k))
                      for k in range(10)},
       "R4": {"max": float(np.nanmax([r["R4"] for r in rows])),
              "median": float(np.nanmedian([r["R4"] for r in rows])),
              "frac_pass_0.05": float(np.mean([r["R4"] < 0.05 for r in rows])),
              "band_fraction": 0.05}}
out["pairs_zero"] = [("%s^%s" % (a, b)) for a, b in itertools.combinations(ks, 2)
                     if sum(g[a] and g[b] for g in G) == 0]
out["pairs_tight"] = dict(sorted({("%s^%s" % (a, b)): int(sum(g[a] and g[b] for g in G))
                                  for a, b in itertools.combinations(ks, 2)}.items(),
                                 key=lambda x: x[1])[:12])
json.dump(out, open(os.path.join(H, "gate_stats_corrected.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
