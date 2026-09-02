"""Q-0008 attempt-04: standard errors for the pre-registered F-02 statistics (K1, K2, K5; K4 consistency).

Reproduces the *same* random streams as verify/Q-0008/F-02/check_modes.py (seed 20260902, identical
sampling order, identical helper functions imported from the untouched script) so that per-trial residuals
are available, checks that the reproduced RMS equal the values in verify/Q-0008/F-02/result.json, and
bootstraps trials (B = 2000, bootstrap seed 20260902) to obtain standard errors and the distance of each
statistic to its pre-registered window edge in sigma units.  Nothing in the card or in check_modes.py is
modified; no constant is redefined here (all are imported from check_modes).

defect mode is deterministic given the seed (one Delta sample): no trial variance, se = null.

Usage: python verify/Q-0008/attempt-04/check_se.py
Writes verify/Q-0008/attempt-04/se_bootstrap.json and trial_values.npz
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))

import check_modes as cm  # noqa: E402  (untouched pre-registered script)
from driver_numbers import uniform_rooted_tree  # noqa: E402

B_BOOT = 2000
BOOT_SEED = 20260902


def trials_her(sizes, trials, delta, seed):
    rng_h = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)
    her, iid = {}, {}
    for n in sizes:
        her[n] = np.array([cm.sample_her(n, rng_h, delta) for _ in range(trials)])
        iid[n] = np.array([cm.sample_iid(n, rng_i, delta) for _ in range(trials)])
    return her, iid


def trials_iid(sizes, trials, delta, seed):
    rng = np.random.default_rng(seed)
    return {n: np.array([cm.sample_iid(n, rng, delta) for _ in range(trials)]) for n in sizes}


def trials_mix(n, trials, delta, seed):
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < trials:
        parent = uniform_rooted_tree(n, rng)
        xi = rng.normal(size=(n, 4, 4))
        zeta = rng.normal(size=(n, 4, 4))
        her = cm.heritable_labels(parent, zeta)
        vi, vh, vm = cm.block_residual(xi, delta), cm.block_residual(her, delta), cm.block_residual(xi + her, delta)
        if not (math.isfinite(vi) and math.isfinite(vh) and math.isfinite(vm)):
            continue
        rows.append((vi, vh, vm))
    return np.array(rows)


def boot_stats(arr):
    arr = np.asarray(arr, dtype=float)
    return {"se": float(np.std(arr, ddof=1)), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}


def sigma_to_window(value, se, window):
    lo, hi = window
    if se is None or se == 0:
        return None
    return {"to_low": (value - lo) / se, "to_high": (hi - value) / se, "nearest": min(value - lo, hi - value) / se}


def main() -> int:
    t0 = time.perf_counter()
    sizes = list(cm.SIZES)
    res = json.loads((F02 / "result.json").read_text(encoding="utf-8"))
    out = {"bootstrap_B": B_BOOT, "bootstrap_seed": BOOT_SEED, "seed": cm.SEED, "delta": cm.DELTA}
    brng = np.random.default_rng(BOOT_SEED)
    T = cm.TRIALS

    # ---- her (K1)
    her, iid_h = trials_her(sizes, cm.TRIALS, cm.DELTA, cm.SEED)
    rms_her = [cm.rms(her[n]) for n in sizes]
    rms_iid_h = [cm.rms(iid_h[n]) for n in sizes]
    repro_her = max(abs(a - b) for a, b in zip(rms_her, res["her"]["rms_her"]))
    repro_iid_h = max(abs(a - b) for a, b in zip(rms_iid_h, res["her"]["rms_iid"]))
    slopes, ratios = [], []
    for _ in range(B_BOOT):
        rh = [cm.rms(her[n][brng.integers(0, T, T)]) for n in sizes]
        ri = [cm.rms(iid_h[n][brng.integers(0, T, T)]) for n in sizes]
        slopes.append(cm.fit_slope(sizes, rh))
        ratios.append(rh[-1] / ri[-1])
    out["her"] = {
        "rms_her": rms_her,
        "rms_iid_seed20260903": rms_iid_h,
        "reproduction_max_abs_diff": {"rms_her": repro_her, "rms_iid": repro_iid_h},
        "her_slope": {"value": cm.fit_slope(sizes, rms_her), **boot_stats(slopes)},
        "her_ratio_128": {"value": rms_her[-1] / rms_iid_h[-1], **boot_stats(ratios)},
        "ratio_her_over_iid_all": [h / i for h, i in zip(rms_her, rms_iid_h)],
        "local_slopes": [
            float(math.log(rms_her[k + 1] / rms_her[k]) / math.log(sizes[k + 1] / sizes[k]))
            for k in range(len(sizes) - 1)
        ],
    }

    # ---- mix (K2)
    mix = trials_mix(cm.MIX_N, cm.MIX_TRIALS, cm.DELTA, cm.SEED)
    r_i, r_h, r_m = cm.rms(mix[:, 0]), cm.rms(mix[:, 1]), cm.rms(mix[:, 2])
    X = (r_m * r_m - r_i * r_i - r_h * r_h) / (r_i * r_h)
    repro_mix = max(abs(r_i - res["mix"]["rms_iid"]), abs(r_h - res["mix"]["rms_her"]), abs(r_m - res["mix"]["rms_mix"]))
    Xs = []
    M = cm.MIX_TRIALS
    for _ in range(B_BOOT):
        idx = brng.integers(0, M, M)
        a, b, c = cm.rms(mix[idx, 0]), cm.rms(mix[idx, 1]), cm.rms(mix[idx, 2])
        Xs.append((c * c - a * a - b * b) / (a * b))
    out["mix"] = {
        "rms_iid": r_i,
        "rms_her": r_h,
        "rms_mix": r_m,
        "reproduction_max_abs_diff": repro_mix,
        "mix_X_32": {"value": X, **boot_stats(Xs)},
    }

    # ---- iid (K5)
    iid = trials_iid(sizes, cm.TRIALS, cm.DELTA, cm.SEED)
    rms_iid = [cm.rms(iid[n]) for n in sizes]
    repro_iid = max(abs(a - b) for a, b in zip(rms_iid, res["iid"]["rms"]))
    sl = []
    for _ in range(B_BOOT):
        sl.append(cm.fit_slope(sizes, [cm.rms(iid[n][brng.integers(0, T, T)]) for n in sizes]))
    out["iid"] = {
        "rms": rms_iid,
        "reproduction_max_abs_diff": repro_iid,
        "iid_slope": {"value": cm.fit_slope(sizes, rms_iid), **boot_stats(sl)},
        "exact_prediction_slope": res["iid"]["exact_prediction_slope"],
        "rms_over_sqrt_nm1_over_n": [r / (math.sqrt(n - 1) / n) for r, n in zip(rms_iid, sizes)],
    }

    # ---- defect (K4 consistency): deterministic, no trial variance
    out["defect"] = {
        "eps": res["defect"]["eps"],
        "grid": res["defect"]["grid"],
        "defect_ratio_64_over_8": {"value": res["defect"]["ratio_64_over_8"], "se": None},
        "defect_slope": {"value": res["defect"]["slope"], "se": None},
        "r48": res["defect"]["eps"][1] / res["defect"]["eps"][0],
        "note": "single pre-registered Delta sample (seed 20260902); no trial variance",
    }

    # ---- sigma distances to the pre-registered windows
    sig = {}
    for key, block in (("her_slope", out["her"]), ("her_ratio_128", out["her"]), ("mix_X_32", out["mix"]), ("iid_slope", out["iid"])):
        v, se = block[key]["value"], block[key]["se"]
        sig[key] = {
            "value": v,
            "se": se,
            "window": list(cm.WINDOWS[key]),
            "preregistered": cm.PREREGISTERED[key],
            "in_window": bool(cm.WINDOWS[key][0] <= v <= cm.WINDOWS[key][1]),
            "sigma": sigma_to_window(v, se, cm.WINDOWS[key]),
            "dev_from_prereg_over_se": (v - cm.PREREGISTERED[key]) / se,
        }
    for key in ("defect_ratio_64_over_8", "defect_slope"):
        v = out["defect"][key]["value"]
        sig[key] = {
            "value": v,
            "se": None,
            "window": list(cm.WINDOWS[key]),
            "preregistered": cm.PREREGISTERED[key],
            "in_window": bool(cm.WINDOWS[key][0] <= v <= cm.WINDOWS[key][1]),
            "sigma": None,
        }
    out["sigma_to_window"] = sig
    out["reproduction_ok"] = bool(max(repro_her, repro_iid_h, repro_mix, repro_iid) < 1e-12)
    out["elapsed_s"] = time.perf_counter() - t0
    np.savez(
        HERE / "trial_values.npz",
        **{f"her_{n}": her[n] for n in sizes},
        **{f"heriid_{n}": iid_h[n] for n in sizes},
        **{f"iid_{n}": iid[n] for n in sizes},
        mix=mix,
    )
    (HERE / "se_bootstrap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reproduction_ok": out["reproduction_ok"], "sigma_to_window": sig, "elapsed_s": out["elapsed_s"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
