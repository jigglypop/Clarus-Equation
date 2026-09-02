"""adversary attempt-04 audit part 3:
(a) chi^2 test of the one-constant kernel law eps^2 = eps_star^2 D/n^2 across all 15 measured RMS points;
(b) independent-Delta power check of the K4 consistency windows (500 fresh Delta samples, seeds 800000+).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
A04 = HERE.parent
ROOT = HERE.parents[3]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
import check_modes as cm  # noqa: E402
import driver_numbers as dn  # noqa: E402

sizes = list(cm.SIZES)
z = np.load(A04 / "trial_values.npz")
her = {n: z["her_" + str(n)] for n in sizes}
heriid = {n: z["heriid_" + str(n)] for n in sizes}
iid = {n: z["iid_" + str(n)] for n in sizes}
ex = {n: dn.cayley_exact(n)["E_D"] for n in sizes}
out = {}


def logrms_and_se(arr):
    a = np.asarray(arr, float) ** 2
    T = len(a)
    jk = np.log(np.sqrt((a.sum() - a) / (T - 1)))
    se = math.sqrt((T - 1) / T * float(np.sum((jk - jk.mean()) ** 2)))
    return math.log(math.sqrt(a.mean())), se


pts = []
for n in sizes:
    y, s = logrms_and_se(her[n])
    pts.append(("her", n, y - 0.5 * math.log(ex[n]) + math.log(n), s))
    y, s = logrms_and_se(heriid[n])
    pts.append(("heriid", n, y - math.log(math.sqrt(n - 1) / n), s))
    y, s = logrms_and_se(iid[n])
    pts.append(("iid", n, y - math.log(math.sqrt(n - 1) / n), s))
w = np.array([1 / p[3] ** 2 for p in pts])
v = np.array([p[2] for p in pts])
mu = float((w * v).sum() / w.sum())
chi2 = float((w * (v - mu) ** 2).sum())
out["eps_star_fit"] = {"log_eps_star": mu, "eps_star": math.exp(mu), "chi2": chi2, "dof": len(pts) - 1,
                       "chi2_over_dof": chi2 / (len(pts) - 1),
                       "points": [{"mode": p[0], "n": p[1], "log_eps_star_hat": p[2], "se": p[3],
                                   "z": (p[2] - mu) / p[3]} for p in pts]}
# her-only vs iid-only eps_star (does the heritable branch need a different constant?)
for tag in ("her", "heriid", "iid"):
    sel = [p for p in pts if p[0] == tag]
    ww = np.array([1 / p[3] ** 2 for p in sel])
    vv = np.array([p[2] for p in sel])
    m = float((ww * vv).sum() / ww.sum())
    out["eps_star_fit"][tag + "_only"] = {"eps_star": math.exp(m),
                                          "se_log": float(1 / math.sqrt(ww.sum())),
                                          "chi2": float((ww * (vv - m) ** 2).sum()), "dof": len(sel) - 1}
a = out["eps_star_fit"]["her_only"]
b = out["eps_star_fit"]["iid_only"]
out["eps_star_fit"]["her_vs_iid_z"] = (math.log(a["eps_star"]) - math.log(b["eps_star"])) / math.sqrt(
    a["se_log"] ** 2 + b["se_log"] ** 2)

# ---- (b) K4 independent-Delta power
N = 500
rat, sl = [], []
for k in range(N):
    d = cm.run_defect(cm.DEFECT_GRID, 800000 + k)
    rat.append(d["ratio_64_over_8"])
    sl.append(d["slope"])
rat = np.array(rat)
sl = np.array(sl)
lo_r, hi_r = cm.WINDOWS["defect_ratio_64_over_8"]
lo_s, hi_s = cm.WINDOWS["defect_slope"]
out["k4_independent_delta"] = {
    "N": N, "ratio_mean": float(rat.mean()), "ratio_sd": float(rat.std(ddof=1)),
    "slope_mean": float(sl.mean()), "slope_sd": float(sl.std(ddof=1)),
    "P_ratio_outside": float(np.mean((rat < lo_r) | (rat > hi_r))),
    "P_slope_outside": float(np.mean((sl < lo_s) | (sl > hi_s))),
    "P_either_outside": float(np.mean((rat < lo_r) | (rat > hi_r) | (sl < lo_s) | (sl > hi_s))),
    "exact_limit_ratio": (63 / 64 ** 2) / (7 / 64), "exact_limit_slope": dn.slope(cm.DEFECT_GRID, [(n - 1) / n ** 2 for n in cm.DEFECT_GRID]),
    "observed_ratio": 0.14306936148581373, "observed_slope": -0.8981828134067237,
}
json.dump(out, open(HERE / "audit_law_defect.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
