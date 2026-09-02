"""adversary attempt-04 audit part 2: trial-level statistics, alternative SE, verify-block arithmetic."""
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

out = {}
sizes = list(cm.SIZES)
ex = {n: dn.cayley_exact(n) for n in sizes}
z = np.load(A04 / "trial_values.npz")
r = json.loads((A04 / "F-02_result_snapshot.json").read_text(encoding="utf-8"))
her = {n: z["her_" + str(n)] for n in sizes}
heriid = {n: z["heriid_" + str(n)] for n in sizes}
iid = {n: z["iid_" + str(n)] for n in sizes}
mix = z["mix"]
out["npz_shapes"] = {"her": [int(her[n].shape[0]) for n in sizes], "mix": [int(x) for x in mix.shape]}
out["npz_rms_minus_result"] = {
    "her": [cm.rms(her[n]) - v for n, v in zip(sizes, r["her"]["rms_her"])],
    "heriid": [cm.rms(heriid[n]) - v for n, v in zip(sizes, r["her"]["rms_iid"])],
    "iid": [cm.rms(iid[n]) - v for n, v in zip(sizes, r["iid"]["rms"])],
    "mix": [cm.rms(mix[:, 0]) - r["mix"]["rms_iid"], cm.rms(mix[:, 1]) - r["mix"]["rms_her"],
            cm.rms(mix[:, 2]) - r["mix"]["rms_mix"]],
}


def tail(arr):
    x = np.asarray(arr, float) ** 2
    return {"cv_eps2": float(x.std(ddof=1) / x.mean()),
            "kurt_eps2": float(((x - x.mean()) ** 4).mean() / x.var() ** 2),
            "ess": float(x.sum() ** 2 / np.sum(x * x)),
            "max_share": float(x.max() / x.sum())}


out["tails_her"] = {str(n): tail(her[n]) for n in sizes}
out["tails_iid"] = {str(n): tail(iid[n]) for n in sizes}
out["tails_mix"] = {"iid": tail(mix[:, 0]), "her": tail(mix[:, 1]), "mix": tail(mix[:, 2])}
out["eps_star_implied"] = {
    "her": [float(cm.rms(her[n]) * n / math.sqrt(ex[n]["E_D"])) for n in sizes],
    "heriid": [float(cm.rms(heriid[n]) / (math.sqrt(n - 1) / n)) for n in sizes],
    "iid": [float(cm.rms(iid[n]) / (math.sqrt(n - 1) / n)) for n in sizes],
}
out["ratio_obs_over_exact"] = [float(cm.rms(her[n]) / cm.rms(heriid[n]) / math.sqrt(ex[n]["E_D"] / (n - 1)))
                               for n in sizes]

xs = np.log(np.array(sizes, float))
w = (xs - xs.mean()) / np.sum((xs - xs.mean()) ** 2)


def dm_var_logrms(arr):
    x = np.asarray(arr, float) ** 2
    return float(x.var(ddof=1) / (len(x) * x.mean() ** 2) / 4.0)


out["delta_method_se"] = {
    "her_slope": float(math.sqrt(sum(w[i] ** 2 * dm_var_logrms(her[n]) for i, n in enumerate(sizes)))),
    "iid_slope": float(math.sqrt(sum(w[i] ** 2 * dm_var_logrms(iid[n]) for i, n in enumerate(sizes)))),
}
rel = math.sqrt(dm_var_logrms(her[128]) + dm_var_logrms(heriid[128]))
out["delta_method_se"]["her_ratio_128"] = rel * cm.rms(her[128]) / cm.rms(heriid[128])


def jack_slope_se(dic):
    var = 0.0
    per = []
    for i, n in enumerate(sizes):
        a = np.asarray(dic[n], float) ** 2
        T = len(a)
        rms_jk = np.sqrt((a.sum() - a) / (T - 1))
        ll = np.log(rms_jk)
        v = (T - 1) / T * float(np.sum((ll - ll.mean()) ** 2))
        per.append(math.sqrt(v))
        var += w[i] ** 2 * v
    return math.sqrt(var), per


jh, per_h = jack_slope_se(her)
ji, per_i = jack_slope_se(iid)
out["jackknife_se"] = {"her_slope": jh, "iid_slope": ji,
                       "per_n_se_log_rms_her": per_h, "per_n_se_log_rms_iid": per_i}

brng = np.random.default_rng(987654321)
T = cm.TRIALS
B = 4000
sl_h, rt, sl_i = [], [], []
for _ in range(B):
    rh = [cm.rms(her[n][brng.integers(0, T, T)]) for n in sizes]
    ri = [cm.rms(heriid[n][brng.integers(0, T, T)]) for n in sizes]
    sl_h.append(cm.fit_slope(sizes, rh))
    rt.append(rh[-1] / ri[-1])
    sl_i.append(cm.fit_slope(sizes, [cm.rms(iid[n][brng.integers(0, T, T)]) for n in sizes]))
out["bootstrap_indep_seed"] = {
    "B": B, "her_slope_se": float(np.std(sl_h, ddof=1)), "her_ratio_128_se": float(np.std(rt, ddof=1)),
    "iid_slope_se": float(np.std(sl_i, ddof=1)),
    "her_slope_ci95": [float(np.percentile(sl_h, 2.5)), float(np.percentile(sl_h, 97.5))],
    "P_her_slope_boot_gt_0.63": float(np.mean(np.array(sl_h) > 0.63)),
}
M = mix.shape[0]
Xc, Xn = [], []
for _ in range(B):
    idx = brng.integers(0, M, M)
    a, b, c = cm.rms(mix[idx, 0]), cm.rms(mix[idx, 1]), cm.rms(mix[idx, 2])
    Xc.append((c * c - a * a - b * b) / (a * b))
    a2 = cm.rms(mix[brng.integers(0, M, M), 0])
    b2 = cm.rms(mix[brng.integers(0, M, M), 1])
    c2 = cm.rms(mix[brng.integers(0, M, M), 2])
    Xn.append((c2 * c2 - a2 * a2 - b2 * b2) / (a2 * b2))
Xc = np.array(Xc)
out["mix_bootstrap"] = {"crn_se": float(Xc.std(ddof=1)), "no_crn_se": float(np.std(Xn, ddof=1)),
                        "crn_ci95": [float(np.percentile(Xc, 2.5)), float(np.percentile(Xc, 97.5))],
                        "P_boot_above_hi": float(np.mean(Xc > 0.99)),
                        "corr_eps2_iid_her": float(np.corrcoef(mix[:, 0] ** 2, mix[:, 1] ** 2)[0, 1]),
                        "corr_eps2_mix_her": float(np.corrcoef(mix[:, 2] ** 2, mix[:, 1] ** 2)[0, 1])}

out["verify_block_arith"] = {
    "X32_from_raw_minus_stat": (r["mix"]["rms_mix"] ** 2 - r["mix"]["rms_iid"] ** 2 - r["mix"]["rms_her"] ** 2)
    / (r["mix"]["rms_iid"] * r["mix"]["rms_her"]) - r["stats"]["mix_X_32"],
    "ratio128_from_raw_minus_stat": r["her"]["rms_her"][-1] / r["her"]["rms_iid"][-1] - r["stats"]["her_ratio_128"],
    "defect_ratio_from_raw_minus_stat": r["defect"]["eps"][4] / r["defect"]["eps"][1] - r["stats"]["defect_ratio_64_over_8"],
    "r48": r["defect"]["eps"][1] / r["defect"]["eps"][0],
    "sqrt_EDC128_over_127": math.sqrt(134587 / 127),
    "identity_9_over_64": ((64 - 1) / 64 ** 2) / ((8 - 1) / 8 ** 2) - 9 / 64,
    "her_slope_refit_minus_stat": cm.fit_slope(sizes, r["her"]["rms_her"]) - r["stats"]["her_slope"],
    "iid_slope_refit_minus_stat": cm.fit_slope(sizes, r["iid"]["rms"]) - r["stats"]["iid_slope"],
    "defect_slope_refit_minus_stat": cm.fit_slope(r["defect"]["grid"], r["defect"]["eps"]) - r["stats"]["defect_slope"],
    "defect_eps_over_exact_shape": [r["defect"]["eps"][i] / ((n - 1) / n ** 2) for i, n in enumerate(r["defect"]["grid"])],
}
json.dump(out, open(HERE / "audit_trials.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("npz_rms_minus_result", "eps_star_implied", "ratio_obs_over_exact",
                                      "delta_method_se", "jackknife_se", "bootstrap_indep_seed", "mix_bootstrap",
                                      "verify_block_arith")}, ensure_ascii=False, indent=1))
