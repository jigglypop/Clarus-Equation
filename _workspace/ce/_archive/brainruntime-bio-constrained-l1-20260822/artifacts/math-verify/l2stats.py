"""(g) L2 decidability in the surrogate: E1 (log-normal weights) and
E2 (right-skewed rates)."""
import os, csv, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
rows = [{k: float(v) for k, v in r.items()}
        for r in csv.DictReader(open(os.path.join(H, "lhs.csv")))]
def c(n): return np.array([r[n] for r in rows])
sl, sw, sg, cv, N = (c("E1_skew_logw"), c("E1_skew_w"), c("E2_skew_rate_proxy"),
                     c("E2_cv_rate_proxy"), c("N_adult"))
m = np.isfinite(sl) & np.isfinite(sw) & (N > 200)
out = {"n_used": int(m.sum())}
out["E1_skew_logw"] = {q: float(np.quantile(sl[m], p)) for q, p in
                       (("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95))}
out["E1_skew_w"] = {q: float(np.quantile(sw[m], p)) for q, p in
                    (("p05", .05), ("median", .5), ("p95", .95))}
out["E1_frac_abs_logskew_lt_0.5"] = float(np.mean(np.abs(sl[m]) < 0.5))
out["E1_frac_both"] = float(np.mean((np.abs(sl[m]) < 0.5) & (sw[m] > 1.0)))
mm = np.isfinite(sg) & np.isfinite(cv) & (N > 200)
out["E2_cv_rate_proxy"] = {q: float(np.quantile(cv[mm], p)) for q, p in
                           (("p05", .05), ("median", .5), ("p95", .95), ("max", 1.0))}
out["E2_skew_rate_proxy"] = {q: float(np.quantile(sg[mm], p)) for q, p in
                             (("p05", .05), ("median", .5), ("p95", .95))}
out["E2_frac_skew_gt_0.5"] = float(np.mean(sg[mm] > 0.5))
out["E2_frac_skew_gt_0.5_and_cv_gt_0.05"] = float(np.mean((sg[mm] > 0.5) & (cv[mm] > 0.05)))
out["note"] = ("E2 rate proxy = per-postsynaptic-neuron input mass. Under full "
               "per-neuron homeostasis (beta=1) this quantity is pinned by "
               "construction: its CV is the residual after one daily correction.")
json.dump(out, open(os.path.join(H, "l2stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
