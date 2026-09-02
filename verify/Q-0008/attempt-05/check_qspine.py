"""Q-0008 attempt-05 (ladder step 7, prediction test K3, card F-02).

Replicates `verify/Q-0008/F-02/check_modes.py --mode qspine` with the SAME random stream
(seed 20260902; iid n=36 stream seed 20260903), storing per-trial residuals so that standard
errors can be attached to the two pre-registered statistics:

    qspine_slope_vs_En          window [0.42, 0.59]   preregistered 0.5047 +- 0.085
    qspine_ratio_b8_over_iid36  window [6.01, 7.65]   preregistered 6.832  +- 0.82

Nothing here changes the verdict: the verdict is the official script result.json stats.*.
This file (a) reproduces those numbers bit-for-bit as a consistency check,
(b) gives bootstrap (B = 2000, bootstrap seed 20260902) and delta-method standard errors,
(c) reports the distance of each statistic to the window edges in units of its SE, and
(d) tabulates the pre-registered alternatives (chain-like 1.0 / 23.11, mean-field 0.533 / 8.245,
Cayley(36) 9.064).  Per-depth ratios RMS_Q(b)/RMS_iid(36) vs sqrt(E[D/n_b^2]) 36/sqrt(35) from the
card tree-only table are reported as ACCOMPANYING diagnostics (not a kill; not pre-registered
as a window).

Constants are copied from the card / official script and are frozen.
Usage: python verify/Q-0008/attempt-05/check_qspine.py
Writes verify/Q-0008/attempt-05/result.json
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

import check_modes as cm  # noqa: E402  (official kill script; constants and residual function)
from driver_numbers import qspine_block  # noqa: E402

# ---- frozen constants (must equal card F-02 / check_modes.py) ------------------------------
SEED = 20260902
DELTA = 0.005
DEPTHS = (2, 3, 4, 5, 6, 7, 8)
TRIALS = 512
IID_N = 36
BOOT_B = 2000
BOOT_SEED = 20260902
PREREG = {"qspine_slope_vs_En": (0.5047, 0.085), "qspine_ratio_b8_over_iid36": (6.832, 0.82)}
WINDOWS = {"qspine_slope_vs_En": (0.42, 0.59), "qspine_ratio_b8_over_iid36": (6.01, 7.65)}
# alternatives (card K3 baseline fields)
ALT = {
    "chain_like": {"slope": 1.0, "ratio": 23.11},
    "mean_field_sqrtED_over_En": {"slope": 0.533, "ratio": 8.245},
    "uniform_cayley_36": {"slope": None, "ratio": 9.064},
}
# card note table (tree-only MC, seed 20261902): E[D/n_b^2], b = 2..8  -> accompanying per-b diagnostics
CARD_E_D_OVER_N2 = [0.1017, 0.2126, 0.3558, 0.5327, 0.7411, 0.9842, 1.2607]


def assert_constants() -> dict:
    checks = {
        "seed": cm.SEED == SEED,
        "delta": cm.DELTA == DELTA,
        "depths": tuple(cm.QSPINE_DEPTHS) == DEPTHS,
        "trials": cm.QSPINE_TRIALS == TRIALS,
        "iid_n": cm.QSPINE_IID_N == IID_N,
        "prereg_slope": cm.PREREGISTERED["qspine_slope_vs_En"] == PREREG["qspine_slope_vs_En"][0],
        "prereg_ratio": cm.PREREGISTERED["qspine_ratio_b8_over_iid36"] == PREREG["qspine_ratio_b8_over_iid36"][0],
        "window_slope": tuple(cm.WINDOWS["qspine_slope_vs_En"]) == WINDOWS["qspine_slope_vs_En"],
        "window_ratio": tuple(cm.WINDOWS["qspine_ratio_b8_over_iid36"]) == WINDOWS["qspine_ratio_b8_over_iid36"],
        "min_det": cm.MIN_DET == 0.05,
    }
    if not all(checks.values()):
        raise SystemExit(f"constant mismatch vs official script: {checks}")
    return checks


def replicate() -> dict:
    """Same call order as check_modes.run_qspine, but keeps every trial."""
    rng = np.random.default_rng(SEED)
    rng_i = np.random.default_rng(SEED + 1)
    per_b_vals: list[list[float]] = []
    per_b_ns: list[list[int]] = []
    rejections = 0
    for b in DEPTHS:
        vals, ns = [], []
        while len(vals) < TRIALS:
            parent = qspine_block(b, rng)
            n = len(parent)
            value = cm.block_residual(cm.heritable_labels(parent, rng.normal(size=(n, 4, 4))), DELTA)
            if math.isfinite(value):
                vals.append(value)
                ns.append(n)
            else:
                rejections += 1
        per_b_vals.append(vals)
        per_b_ns.append(ns)
    iid_vals = [cm.sample_iid(IID_N, rng_i, DELTA) for _ in range(TRIALS)]
    return {"vals": per_b_vals, "ns": per_b_ns, "iid": iid_vals, "rejections": rejections}


def slope_from_rms(rms_list) -> float:
    return cm.fit_slope([b * (b + 1) // 2 for b in DEPTHS], rms_list)


def main() -> int:
    t0 = time.perf_counter()
    const = assert_constants()
    rep = replicate()
    runtime = time.perf_counter() - t0

    V = [np.asarray(v, dtype=float) for v in rep["vals"]]
    I = np.asarray(rep["iid"], dtype=float)
    rms_b = [cm.rms(v) for v in V]
    rms_iid = cm.rms(I)
    E_n = [b * (b + 1) // 2 for b in DEPTHS]
    slope = slope_from_rms(rms_b)
    slope_b = cm.fit_slope(DEPTHS, rms_b)
    ratio = rms_b[DEPTHS.index(8)] / rms_iid

    # ---- delta-method SE: var(log rms) = var(v^2) / (4 T mean(v^2)^2)
    def var_log_rms(v: np.ndarray) -> float:
        m2 = np.mean(v * v)
        return float(np.var(v * v, ddof=1) / (4 * len(v) * m2 * m2))

    x = np.log(np.asarray(E_n, dtype=float))
    c = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    se_slope_delta = float(np.sqrt(np.sum(c * c * np.array([var_log_rms(v) for v in V]))))
    se_ratio_delta = float(ratio * math.sqrt(var_log_rms(V[-1]) + var_log_rms(I)))

    # ---- bootstrap SE (resample trials within each depth and within the iid block)
    brng = np.random.default_rng(BOOT_SEED)
    bs_slope, bs_ratio = [], []
    for _ in range(BOOT_B):
        r = []
        for v in V:
            idx = brng.integers(0, len(v), size=len(v))
            r.append(float(np.sqrt(np.mean(v[idx] ** 2))))
        idx = brng.integers(0, len(I), size=len(I))
        ri = float(np.sqrt(np.mean(I[idx] ** 2)))
        bs_slope.append(slope_from_rms(r))
        bs_ratio.append(r[-1] / ri)
    se_slope_boot = float(np.std(bs_slope, ddof=1))
    se_ratio_boot = float(np.std(bs_ratio, ddof=1))
    ci_slope = [float(np.percentile(bs_slope, 2.5)), float(np.percentile(bs_slope, 97.5))]
    ci_ratio = [float(np.percentile(bs_ratio, 2.5)), float(np.percentile(bs_ratio, 97.5))]

    def window_report(key, value, se):
        lo, hi = WINDOWS[key]
        pre, unc = PREREG[key]
        return {
            "value": value,
            "se": se,
            "window": [lo, hi],
            "pass": bool(lo <= value <= hi),
            "preregistered": pre,
            "preregistered_uncertainty": unc,
            "z_vs_preregistered": (value - pre) / se if se > 0 else None,
            "sigma_to_lower_edge": (value - lo) / se if se > 0 else None,
            "sigma_to_upper_edge": (hi - value) / se if se > 0 else None,
        }

    stats = {
        "qspine_slope_vs_En": window_report("qspine_slope_vs_En", slope, se_slope_boot),
        "qspine_ratio_b8_over_iid36": window_report("qspine_ratio_b8_over_iid36", ratio, se_ratio_boot),
    }
    stats["qspine_slope_vs_En"]["se_delta_method"] = se_slope_delta
    stats["qspine_slope_vs_En"]["ci95_bootstrap"] = ci_slope
    stats["qspine_ratio_b8_over_iid36"]["se_delta_method"] = se_ratio_delta
    stats["qspine_ratio_b8_over_iid36"]["ci95_bootstrap"] = ci_ratio

    # ---- alternatives table (sigma distances use bootstrap SE)
    alt_table = {}
    for name, a in ALT.items():
        row = {}
        if a["slope"] is not None:
            row["slope"] = a["slope"]
            row["slope_sigma_from_observed"] = (a["slope"] - slope) / se_slope_boot
        row["ratio"] = a["ratio"]
        row["ratio_sigma_from_observed"] = (a["ratio"] - ratio) / se_ratio_boot
        alt_table[name] = row

    # ---- accompanying per-depth diagnostics (not pre-registered as kill)
    per_b = []
    for k, b in enumerate(DEPTHS):
        pred_ratio = math.sqrt(CARD_E_D_OVER_N2[k]) * IID_N / math.sqrt(IID_N - 1)
        obs_ratio = rms_b[k] / rms_iid
        se_obs = obs_ratio * math.sqrt(var_log_rms(V[k]) + var_log_rms(I))
        per_b.append({
            "b": b, "E_n_exact": E_n[k], "mean_n_observed": float(np.mean(rep["ns"][k])),
            "rms": rms_b[k], "ratio_over_iid36_observed": obs_ratio, "ratio_over_iid36_card_table": pred_ratio,
            "ratio_se_delta": se_obs, "obs_over_card": obs_ratio / pred_ratio,
            "rms_over_delta2": rms_b[k] / DELTA**2,
        })

    # ---- compare with the official run if its result.json exists (bit-for-bit reproduction)
    official = None
    off_path = F02 / "result.json"
    if off_path.is_file():
        try:
            off = json.loads(off_path.read_text(encoding="utf-8"))
            if "qspine" in off:
                q = off["qspine"]
                official = {
                    "stats": {k: off.get("stats", {}).get(k) for k in WINDOWS},
                    "verdict": {k: off.get("verdict", {}).get(k) for k in WINDOWS},
                    "rms": q.get("rms"), "mean_n": q.get("mean_n"), "rms_iid_36": q.get("rms_iid_36"),
                    "slope_vs_b": q.get("slope_vs_b"),
                }
                official["reproduced_bitwise"] = {
                    "rms": bool(np.allclose(q["rms"], rms_b, rtol=0, atol=0)),
                    "rms_iid_36": q["rms_iid_36"] == rms_iid,
                    "slope": off["stats"]["qspine_slope_vs_En"] == slope,
                    "ratio": off["stats"]["qspine_ratio_b8_over_iid36"] == ratio,
                }
                official["max_abs_diff_rms"] = float(np.max(np.abs(np.asarray(q["rms"]) - np.asarray(rms_b))))
        except Exception as exc:  # pragma: no cover
            official = {"error": repr(exc)}

    out = {
        "question": "Q-0008", "card": "F-02", "attempt": 5, "ladder_step": 7, "mode": "qspine",
        "seed": SEED, "iid_seed": SEED + 1, "delta": DELTA, "depths": list(DEPTHS), "trials_per_depth": TRIALS,
        "iid_n": IID_N, "min_det": cm.MIN_DET, "rejections": rep["rejections"],
        "constants_match_official_script": const,
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED},
        "E_n_exact": E_n, "mean_n_observed": [float(np.mean(ns)) for ns in rep["ns"]],
        "rms": rms_b, "rms_iid_36": rms_iid, "slope_vs_b": slope_b,
        "stats": stats,
        "kills_fired": [k for k, s in stats.items() if not s["pass"]],
        "alternatives": alt_table,
        "per_depth_accompanying": per_b,
        "official_run": official,
        "runtime_s_replication": runtime,
    }
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stats": {k: {kk: s[kk] for kk in ("value", "se", "window", "pass")} for k, s in stats.items()},
                      "kills_fired": out["kills_fired"], "rejections": rep["rejections"],
                      "official_reproduced": (official or {}).get("reproduced_bitwise"),
                      "runtime_s": runtime}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
