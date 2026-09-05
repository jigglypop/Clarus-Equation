"""Q-0008 F-02 kill script: centered-kernel law eps_block^2 = eps_star^2 ||H kappa H||_F^2 / n^2.

Pre-registered (card derivations/Q-0008/F-02.formula.md, 2026-09-02).  Seed, sizes, delta, trial
counts, windows are frozen; do not edit after seeing results.  Derived from F-01's script
(verify/Q-0008/F-01/check_modes.py, untouched) with the modes replaced:

  iid     : e_v = I + delta xi_v, xi_v iid N(0,1)^{4x4}; uniform sizes grid          -> control (no floor)
  her     : uniform rooted Cayley tree (Pruefer), label_v = sum of xi on root path    -> gamma_her, ratio(128)
  mix     : n=32 Cayley, label_v = xi_v + (heritable zeta path sum), common random numbers
            X = (RMS_mix^2 - RMS_iid^2 - RMS_her^2) / (RMS_iid RMS_her)              -> cross term (not quadrature)
  qspine  : depth-b Q-spine block (11.6: spine + Poisson(1) side GW trees, cut at depth b-1),
            heritable labels; RMS_Q(b) vs E[n_b]=b(b+1)/2 and RMS_Q(8)/RMS_iid(36)   -> Q-spine universality
  defect  : (n-1) copies of Sigma(I) + one aligned mismatched cell Sigma(I+0.35 G);
            eps(64)/eps(8) and slope over n in {4..64}                              -> exact identity (p=1/n)
            CONSISTENCY CHECK, NOT KILL (card revision 2, 2026-09-02): the prover disclosed
            r48 = eps(8)/eps(4) = 0.5867 at the pre-registered seed, which is correlated 0.997/0.999
            with the two K4 statistics (adversary b7), and the windows are centred on the delta_c -> 0
            limit so that a true identity fires 34.5% of the time (b6/b7).  PREREGISTERED values and
            WINDOWS below are left untouched; a window miss is recorded as "inconsistent", not as a
            card kill.  Independent kills are K1, K2, K3, K5.

All residuals are simplicity_residual (12.4, normalized traceless Plebanski gram) of the polar-aligned
block sum; statistic is RMS over trials; delta = 0.005 for every stochastic mode (delta^2 regime).
MIN_DET: a configuration is resampled if any cell has det(I + delta*label) <= MIN_DET (declared; at
delta = 0.005 the expected rejection rate is 0).

Usage: python verify/Q-0008/F-02/check_modes.py --mode {iid,her,mix,qspine,defect,all} [--smoke]
Writes verify/Q-0008/F-02/result.json (unless --smoke).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from driver_numbers import qspine_block, tree_arrays, uniform_rooted_tree  # noqa: E402
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    simplicity_residual,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

SEED = 20260902
DELTA = 0.005
MIN_DET = 0.05
SIZES = (8, 16, 32, 64, 128)
TRIALS = 256
MIX_N = 32
MIX_TRIALS = 1024
QSPINE_DEPTHS = (2, 3, 4, 5, 6, 7, 8)
QSPINE_TRIALS = 512
QSPINE_IID_N = 36
DEFECT_GRID = (4, 8, 16, 32, 64)
DEFECT_PERTURBATION = 0.35
DEFECT_MIN_DET = 0.2

PREREGISTERED = {
    "iid_slope": -0.4783,
    "her_slope": 0.5302,
    "her_ratio_128": 32.554,
    "mix_X_32": 0.7406,
    "qspine_slope_vs_En": 0.5047,
    "qspine_ratio_b8_over_iid36": 6.832,
    "defect_ratio_64_over_8": 0.140625,
    "defect_slope": -0.9069,
}
WINDOWS = {
    "iid_slope": (-0.58, -0.38),
    "her_slope": (0.43, 0.63),
    "her_ratio_128": (26.0, 39.1),
    "mix_X_32": (0.49, 0.99),
    "qspine_slope_vs_En": (0.42, 0.59),
    "qspine_ratio_b8_over_iid36": (6.01, 7.65),
    "defect_ratio_64_over_8": (0.124, 0.158),
    "defect_slope": (-0.96, -0.86),
}

REFERENCE = geometric_self_dual_triple(np.eye(4))


# ---------------------------------------------------------------- block residual
def block_residual(labels: np.ndarray, delta: float) -> float:
    blocked = np.zeros_like(REFERENCE)
    for lab in labels:
        tetrad = np.eye(4) + delta * lab
        if float(np.linalg.det(tetrad)) <= MIN_DET:
            return math.nan
        candidate = geometric_self_dual_triple(tetrad)
        blocked += optimal_internal_alignment(REFERENCE, candidate).aligned_candidate
    return simplicity_residual(blocked)


def heritable_labels(parent: list[int], xi: np.ndarray) -> np.ndarray:
    order, _, _, _ = tree_arrays(parent)
    labels = np.zeros_like(xi)
    for v in order:
        p = parent[v]
        labels[v] = xi[v] + (labels[p] if p >= 0 else 0.0)
    return labels


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def fit_slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, dtype=float)), np.log(np.asarray(ys, dtype=float)), 1)[0])


# ---------------------------------------------------------------- modes
def sample_iid(n: int, rng: np.random.Generator, delta: float) -> float:
    while True:
        value = block_residual(rng.normal(size=(n, 4, 4)), delta)
        if math.isfinite(value):
            return value


def sample_her(n: int, rng: np.random.Generator, delta: float) -> float:
    while True:
        parent = uniform_rooted_tree(n, rng)
        value = block_residual(heritable_labels(parent, rng.normal(size=(n, 4, 4))), delta)
        if math.isfinite(value):
            return value


def run_iid(sizes, trials, delta, seed) -> dict:
    rng = np.random.default_rng(seed)
    out = {"sizes": list(sizes), "rms": [], "trials": trials, "delta": delta, "seed": seed, "rejections": 0}
    for n in sizes:
        out["rms"].append(rms([sample_iid(n, rng, delta) for _ in range(trials)]))
    out["slope"] = fit_slope(sizes, out["rms"])
    out["exact_prediction_slope"] = fit_slope(sizes, [math.sqrt(n - 1) / n for n in sizes])
    return out


def run_her(sizes, trials, delta, seed) -> dict:
    rng_h = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)
    out = {"sizes": list(sizes), "rms_her": [], "rms_iid": [], "trials": trials, "delta": delta, "seed": seed}
    for n in sizes:
        out["rms_her"].append(rms([sample_her(n, rng_h, delta) for _ in range(trials)]))
        out["rms_iid"].append(rms([sample_iid(n, rng_i, delta) for _ in range(trials)]))
    out["slope"] = fit_slope(sizes, out["rms_her"])
    out["ratio_her_over_iid"] = [h / i for h, i in zip(out["rms_her"], out["rms_iid"])]
    out["ratio_128"] = out["ratio_her_over_iid"][list(sizes).index(128)] if 128 in sizes else None
    return out


def run_mix(n: int, trials: int, delta: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    e_i, e_h, e_m = [], [], []
    while len(e_m) < trials:
        parent = uniform_rooted_tree(n, rng)
        xi = rng.normal(size=(n, 4, 4))
        zeta = rng.normal(size=(n, 4, 4))
        her = heritable_labels(parent, zeta)
        vi, vh, vm = block_residual(xi, delta), block_residual(her, delta), block_residual(xi + her, delta)
        if not (math.isfinite(vi) and math.isfinite(vh) and math.isfinite(vm)):
            continue
        e_i.append(vi)
        e_h.append(vh)
        e_m.append(vm)
    r_i, r_h, r_m = rms(e_i), rms(e_h), rms(e_m)
    return {
        "n": n,
        "trials": trials,
        "delta": delta,
        "seed": seed,
        "rms_iid": r_i,
        "rms_her": r_h,
        "rms_mix": r_m,
        "X": (r_m * r_m - r_i * r_i - r_h * r_h) / (r_i * r_h),
    }


def run_qspine(depths, trials, iid_n, delta, seed) -> dict:
    rng = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)
    out = {"depths": list(depths), "E_n_exact": [b * (b + 1) // 2 for b in depths], "rms": [], "mean_n": [],
           "trials": trials, "delta": delta, "seed": seed}
    for b in depths:
        vals, ns = [], []
        while len(vals) < trials:
            parent = qspine_block(b, rng)
            n = len(parent)
            value = block_residual(heritable_labels(parent, rng.normal(size=(n, 4, 4))), delta)
            if math.isfinite(value):
                vals.append(value)
                ns.append(n)
        out["rms"].append(rms(vals))
        out["mean_n"].append(float(np.mean(ns)))
    out["slope_vs_En"] = fit_slope(out["E_n_exact"], out["rms"])
    out["slope_vs_b"] = fit_slope(depths, out["rms"])
    out["rms_iid_36"] = rms([sample_iid(iid_n, rng_i, delta) for _ in range(trials)])
    out["ratio_b8_over_iid36"] = out["rms"][list(depths).index(8)] / out["rms_iid_36"] if 8 in depths else None
    return out


def run_defect(grid, seed) -> dict:
    rng = np.random.default_rng(seed)
    while True:
        tetrad = np.eye(4) + DEFECT_PERTURBATION * rng.normal(size=(4, 4))
        if float(np.linalg.det(tetrad)) > DEFECT_MIN_DET:
            break
    aligned_c = optimal_internal_alignment(REFERENCE, geometric_self_dual_triple(tetrad)).aligned_candidate
    eps = [simplicity_residual((n - 1) * REFERENCE + aligned_c) for n in grid]
    out = {"grid": list(grid), "eps": eps, "seed": seed, "perturbation": DEFECT_PERTURBATION}
    out["slope"] = fit_slope(grid, eps)
    out["ratio_64_over_8"] = eps[list(grid).index(64)] / eps[list(grid).index(8)] if (64 in grid and 8 in grid) else None
    out["exact_identity_prediction_ratio"] = (63 / 64**2) / (7 / 64)
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=("iid", "her", "mix", "qspine", "defect", "all"))
    parser.add_argument("--smoke", action="store_true", help="execution check only (tiny sizes, no verdict)")
    args = parser.parse_args()

    if args.smoke:
        a = run_iid((4, 6), 2, DELTA, SEED)
        b = run_her((4, 6), 2, DELTA, SEED)
        c = run_mix(5, 3, DELTA, SEED)
        d = run_qspine((2, 3), 2, 4, DELTA, SEED)
        e = run_defect((4, 8), SEED)
        assert all(math.isfinite(x) for x in a["rms"] + b["rms_her"] + d["rms"] + e["eps"])
        assert math.isfinite(c["X"])
        print(json.dumps({"smoke": "ok", "iid": a["rms"], "her": b["rms_her"], "mixX": c["X"], "qspine": d["rms"],
                          "defect_eps": e["eps"]}))
        return 0

    result: dict = {"card": "F-02", "question": "Q-0008", "seed": SEED, "delta": DELTA, "min_det": MIN_DET,
                    "preregistered": PREREGISTERED, "windows": WINDOWS}
    modes = ("iid", "her", "mix", "qspine", "defect") if args.mode == "all" else (args.mode,)
    stats: dict[str, float] = {}
    for mode in modes:
        if mode == "iid":
            block = run_iid(SIZES, TRIALS, DELTA, SEED)
            stats["iid_slope"] = block["slope"]
        elif mode == "her":
            block = run_her(SIZES, TRIALS, DELTA, SEED)
            stats["her_slope"] = block["slope"]
            stats["her_ratio_128"] = block["ratio_128"]
        elif mode == "mix":
            block = run_mix(MIX_N, MIX_TRIALS, DELTA, SEED)
            stats["mix_X_32"] = block["X"]
        elif mode == "qspine":
            block = run_qspine(QSPINE_DEPTHS, QSPINE_TRIALS, QSPINE_IID_N, DELTA, SEED)
            stats["qspine_slope_vs_En"] = block["slope_vs_En"]
            stats["qspine_ratio_b8_over_iid36"] = block["ratio_b8_over_iid36"]
        else:
            block = run_defect(DEFECT_GRID, SEED)
            stats["defect_ratio_64_over_8"] = block["ratio_64_over_8"]
            stats["defect_slope"] = block["slope"]
        result[mode] = block
    verdict = {}
    for key, value in stats.items():
        lo, hi = WINDOWS[key]
        verdict[key] = "KILL" if not (lo <= value <= hi) else "survive"
    result["stats"] = stats
    result["verdict"] = verdict
    out = HERE / "result.json"
    existing = {}
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    for key, value in result.items():
        if key in ("stats", "verdict"):
            existing.setdefault(key, {}).update(value)
        else:
            existing[key] = value
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "stats": stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
