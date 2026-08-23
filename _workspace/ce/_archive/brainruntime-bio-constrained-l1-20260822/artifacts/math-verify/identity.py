"""Residual validation of the two mass-balance identities used in 11-math (a)/(c).
   res_all = (1+gamma) c (1-R3a) - 1
   res_top = (1+gamma_top) c (1-R3b) - 1,  gamma_top = (theta_G/f_top) gamma
Stated preconditions: cyclic stationarity, negligible birth-death mass flux."""
import os, csv, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
rows = [{k: float(v) for k, v in r.items()}
        for r in csv.DictReader(open(os.path.join(H, "lhs.csv")))]
def col(n): return np.array([r[n] for r in rows])
g, c, s, st, ft, tg = (col("gamma"), col("c_hom"), col("R3a"), col("R3b"),
                       col("f_top"), col("theta_G"))
r1, dft = col("R1_A"), col("drift_ftop")
res_all = (1 + g) * c * (1 - s) - 1.0
res_top = (1 + (tg / ft) * g) * c * (1 - st) - 1.0
ok = np.isfinite(res_all) & np.isfinite(res_top)
out = {"theta_G_stats": {"median": float(np.nanmedian(tg)),
                         "q10": float(np.nanquantile(tg, .1)),
                         "q90": float(np.nanquantile(tg, .9))}}
for tag, m in (("all", ok),
               ("turnover R1_A<0.08", ok & (r1 < 0.08)),
               ("turnover<0.08 and |drift_ftop|<0.5", ok & (r1 < 0.08) & (np.abs(dft) < 0.5)),
               ("turnover<0.08 and gamma<1", ok & (r1 < 0.08) & (g < 1.0))):
    out[tag] = {"n": int(m.sum()),
                "res_all_med": float(np.median(np.abs(res_all[m]))),
                "res_all_p90": float(np.quantile(np.abs(res_all[m]), .9)),
                "res_top_med": float(np.median(np.abs(res_top[m]))),
                "res_top_p90": float(np.quantile(np.abs(res_top[m]), .9))}
# R6 ceiling check:  R6 <= 1/R2ad_Na  (both use the birth-cohort reading)
r6, r2a = col("R6"), col("R2ad_Na")
mm = np.isfinite(r6) & np.isfinite(r2a) & (r2a > 0)
out["R6_ceiling"] = {"n": int(mm.sum()),
                     "max_violation_of_R6_le_1_over_R2ad": float(np.max(r6[mm] * r2a[mm])),
                     "frac_within": float(np.mean(r6[mm] * r2a[mm] <= 1.0 + 1e-9)),
                     "implied_R2ad_max_for_R6_ge_1.3": 1.0 / 1.3,
                     "R6_ceiling_at_R2ad_target_0.73": 1.0 / 0.73}
json.dump(out, open(os.path.join(H, "identity.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
