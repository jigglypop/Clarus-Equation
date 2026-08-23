import os, csv, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(H, "lhs.csv")))]
def col(n): return np.array([r[n] for r in rows])
g, c, s, st, ft, tg = col("gamma"), col("c_hom"), col("R3a"), col("R3b"), col("f_top"), col("theta_G")
r1, dft, N = col("R1_A"), col("drift_ftop"), col("N_adult")
res_all = (1 + g) * c * (1 - s) - 1.0
gt = (tg / ft) * g
res_top = (1 + gt) * c * (1 - st) - 1.0
ok = np.isfinite(res_all) & np.isfinite(res_top)
for tag, m in (("all", ok), ("turnover<0.08", ok & (r1 < 0.08)),
               ("turnover<0.08,|drift_ftop|<0.5", ok & (r1 < 0.08) & (np.abs(dft) < 0.5)),
               ("turnover<0.08,drift<0.5,N>300", ok & (r1 < 0.08) & (np.abs(dft) < 0.5) & (N > 300)),
               ("gamma<1", ok & (g < 1.0) & (r1 < 0.08))):
    print("%-34s n=%4d  |res_all| med %.4f p90 %.4f | |res_top| med %.4f p90 %.4f"
          % (tag, m.sum(), np.median(np.abs(res_all[m])), np.quantile(np.abs(res_all[m]), .9),
             np.median(np.abs(res_top[m])), np.quantile(np.abs(res_top[m]), .9)))
m = ok & (r1 < 0.08) & (np.abs(dft) < 0.5) & (g < 1.0)
print("\nsample rows (turnover<0.08, gamma<1):")
idx = np.flatnonzero(m)[:12]
for i in idx:
    print(" g=%8.4f c=%.4f s=%.4f st=%.5f ft=%.3f tG=%.3f | res_all=%+.4f res_top=%+.4f"
          % (g[i], c[i], s[i], st[i], ft[i], tg[i], res_all[i], res_top[i]))
