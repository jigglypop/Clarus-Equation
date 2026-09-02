"""Q-0016 F-01 kill script: split-conserving inheritance kernel (competitor of Q-0008 F-02).

Pre-registered (card derivations/Q-0016/F-01.formula.md, 2026-09-03).  Seed, sizes, delta, trial counts and
windows are frozen; do not edit after seeing results.  Physics machinery (block_residual = polar-aligned
block sum -> simplicity_residual, MIN_DET resampling, delta = 0.005) is imported unchanged from the F-02
kill script; only the LABEL SAMPLER differs:

  F-02 (heritable_labels): label_v = sum of i.i.d. xi_u over the root path of v.
  F-01 here (split_labels):  same, but at every split (k >= 2 children of one parent) the children's
        increments are eta_c = sqrt(k/(k-1)) (xi_c - mean_ch xi): sum_c eta_c = 0 exactly, unit marginal
        variance, sibling correlation -1/(k-1).  Only children and the root keep eta = xi.

Modes (all: split sampler, seed 20260902; i.i.d. comparison blocks seed 20260903):
  qspine_split : depth-b Q-spine block (11.6), b in 2..8, 512 trials/b; RMS_Q(b) vs E[n_b] = b(b+1)/2 and
                 RMS_Q(8)/RMS_iid(36)                                         -> K_A1
  binary_split : complete binary tree n in {7,15,31,63}, 512 trials/n; RMS_bin(15)/RMS_iid(15) and the
                 log-log slope of RMS_bin(n) over the grid                        -> K_A2
  cayley_split : uniform rooted Cayley tree (Pruefer) n in {8,16,32,64,128}, 256 trials/n; slope and
                 RMS_split(8)/RMS_iid(8)                                          -> K_A3

Windows: predicted +- max(3 * a-priori SE, 5%).  A-priori SEs are the bootstrap SEs F-02 observed for
the same statistic type at the same trial count (E-20260902-015/016): slope 0.0114 (512 trials) and
0.019 (256 trials, Cayley grid); ratio 2.9% (Q-spine b=8, 512), 3.5% (= 4.9%/sqrt2, n=15 at 512),
1.4% (Cayley n=8 at 256, adversary b3).

Usage: python verify/Q-0016/F-01/check_split_modes.py --mode {qspine_split,binary_split,cayley_split,all} [--smoke]
Writes verify/Q-0016/F-01/result.json (unless --smoke).
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))

from check_modes import DELTA, MIN_DET, block_residual, fit_slope, rms, sample_iid  # noqa: E402
from driver_numbers import qspine_block, uniform_rooted_tree  # noqa: E402
from predict_split_kernel import complete_binary_parent, split_labels  # noqa: E402

SEED = 20260902
QSPINE_DEPTHS = (2, 3, 4, 5, 6, 7, 8)
QSPINE_TRIALS = 512
QSPINE_IID_N = 36
BINARY_SIZES = (7, 15, 31, 63)
BINARY_TRIALS = 512
CAYLEY_SIZES = (8, 16, 32, 64, 128)
CAYLEY_TRIALS = 256

PREREGISTERED = {
    "qspine_split_slope_vs_En": 0.3695,
    "qspine_split_ratio_b8_over_iid36": 7.814,
    "binary_split_ratio_15": 4.504,
    "binary_split_slope_7_63": 0.1454,
    "cayley_split_slope": 0.4434,
    "cayley_split_ratio_8": 2.6035,
}
WINDOWS = {
    "qspine_split_slope_vs_En": (0.335, 0.404),
    "qspine_split_ratio_b8_over_iid36": (7.134, 8.494),
    "binary_split_ratio_15": (4.031, 4.977),
    "binary_split_slope_7_63": (0.111, 0.180),
    "cayley_split_slope": (0.386, 0.500),
    "cayley_split_ratio_8": (2.473, 2.734),
}
F02_ALTERNATIVE = {  # what Q-0008 F-02 (i.i.d. increments) predicts for the same statistic (all outside the windows)
    "qspine_split_slope_vs_En": 0.5047,
    "qspine_split_ratio_b8_over_iid36": 6.832,
    "binary_split_ratio_15": 3.164,
    "binary_split_slope_7_63": 0.2927,
    "cayley_split_slope": 0.5302,
    "cayley_split_ratio_8": 1.9877,
}


def sample_split(parent: list[int], rng: np.random.Generator, delta: float) -> float:
    n = len(parent)
    while True:
        value = block_residual(split_labels(parent, rng.normal(size=(n, 4, 4))), delta)
        if math.isfinite(value):
            return value


def run_qspine_split(depths, trials, iid_n, delta, seed) -> dict:
    rng = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)
    out = {"depths": list(depths), "E_n_exact": [b * (b + 1) // 2 for b in depths], "rms": [], "mean_n": [],
           "trials": trials, "delta": delta, "seed": seed}
    for b in depths:
        vals, ns = [], []
        while len(vals) < trials:
            parent = qspine_block(b, rng)
            vals.append(sample_split(parent, rng, delta))
            ns.append(len(parent))
        out["rms"].append(rms(vals))
        out["mean_n"].append(float(np.mean(ns)))
    out["slope_vs_En"] = fit_slope(out["E_n_exact"], out["rms"])
    out["rms_iid_36"] = rms([sample_iid(iid_n, rng_i, delta) for _ in range(trials)])
    out["ratio_b8_over_iid36"] = out["rms"][list(depths).index(8)] / out["rms_iid_36"] if 8 in depths else None
    return out


def run_binary_split(sizes, trials, delta, seed) -> dict:
    rng = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)
    out = {"sizes": list(sizes), "rms_split": [], "rms_iid": [], "trials": trials, "delta": delta, "seed": seed}
    for n in sizes:
        depth = int(round(math.log2(n + 1))) - 1
        parent = complete_binary_parent(depth)
        assert len(parent) == n
        out["rms_split"].append(rms([sample_split(parent, rng, delta) for _ in range(trials)]))
        out["rms_iid"].append(rms([sample_iid(n, rng_i, delta) for _ in range(trials)]))
    out["ratio_split_over_iid"] = [s / i for s, i in zip(out["rms_split"], out["rms_iid"])]
    out["ratio_15"] = out["ratio_split_over_iid"][list(sizes).index(15)] if 15 in sizes else None
    out["slope"] = fit_slope(sizes, out["rms_split"])
    return out


def run_cayley_split(sizes, trials, delta, seed) -> dict:
    rng = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)
    out = {"sizes": list(sizes), "rms_split": [], "rms_iid": [], "trials": trials, "delta": delta, "seed": seed}
    for n in sizes:
        vals = []
        for _ in range(trials):
            parent = uniform_rooted_tree(n, rng)
            vals.append(sample_split(parent, rng, delta))
        out["rms_split"].append(rms(vals))
        out["rms_iid"].append(rms([sample_iid(n, rng_i, delta) for _ in range(trials)]))
    out["ratio_split_over_iid"] = [s / i for s, i in zip(out["rms_split"], out["rms_iid"])]
    out["ratio_8"] = out["ratio_split_over_iid"][list(sizes).index(8)] if 8 in sizes else None
    out["slope"] = fit_slope(sizes, out["rms_split"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=("qspine_split", "binary_split", "cayley_split", "all"))
    parser.add_argument("--smoke", action="store_true", help="execution check only (tiny sizes, no verdict)")
    args = parser.parse_args()

    if args.smoke:
        a = run_qspine_split((2, 3), 2, 4, DELTA, SEED)
        b = run_binary_split((3, 7), 2, DELTA, SEED)
        c = run_cayley_split((4, 6), 2, DELTA, SEED)
        assert all(math.isfinite(x) for x in a["rms"] + b["rms_split"] + c["rms_split"])
        print(json.dumps({"smoke": "ok", "qspine": a["rms"], "binary": b["rms_split"], "cayley": c["rms_split"]}))
        return 0

    result: dict = {"card": "F-01", "question": "Q-0016", "seed": SEED, "delta": DELTA, "min_det": MIN_DET,
                    "preregistered": PREREGISTERED, "windows": WINDOWS, "f02_alternative": F02_ALTERNATIVE}
    modes = ("qspine_split", "binary_split", "cayley_split") if args.mode == "all" else (args.mode,)
    stats: dict[str, float] = {}
    for mode in modes:
        t0 = time.time()
        if mode == "qspine_split":
            block = run_qspine_split(QSPINE_DEPTHS, QSPINE_TRIALS, QSPINE_IID_N, DELTA, SEED)
            stats["qspine_split_slope_vs_En"] = block["slope_vs_En"]
            stats["qspine_split_ratio_b8_over_iid36"] = block["ratio_b8_over_iid36"]
        elif mode == "binary_split":
            block = run_binary_split(BINARY_SIZES, BINARY_TRIALS, DELTA, SEED)
            stats["binary_split_ratio_15"] = block["ratio_15"]
            stats["binary_split_slope_7_63"] = block["slope"]
        else:
            block = run_cayley_split(CAYLEY_SIZES, CAYLEY_TRIALS, DELTA, SEED)
            stats["cayley_split_slope"] = block["slope"]
            stats["cayley_split_ratio_8"] = block["ratio_8"]
        block["wall_seconds"] = time.time() - t0
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
