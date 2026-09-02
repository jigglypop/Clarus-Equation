"""Q-0017 F-01 kill script (physical MC on the F-02 machine): merge-averaging DAG labels.

Pre-registered (card derivations/Q-0017/F-01.formula.md, 2026-09-03).  Seeds, sizes, delta, trial
counts and windows are frozen; do not edit after seeing results.  Re-uses Q-0008 F-02 block_residual
(polar-aligned block sum, 12.4 normalized traceless Plebanski residual, MIN_DET=0.05, delta=0.005) and
the same Pruefer Cayley generator; labels follow the merge rule of predict_merge_gamma.py:

  label_v = xi_v + (label_{p(v)} + label_{r(v)})/2   if v is a merge event (u_v < q and a partner exists)
  label_v = xi_v +  label_{p(v)}                     otherwise
  q = 0 is F-02 heritable mode (E-015 measured her_slope 0.5576 on this grid).

Modes (each a separate pre-registered statistic; see PREREGISTERED / WINDOWS):
  k1  : Cayley, F-02 grid n in {8,16,32,64,128}, q = 1, 256 trials/size, seed 20260902
        -> stats.k1_slope_q1_f02grid      (marginality on the F-02 grid: no sign change in q)
  k2  : Cayley, plateau grid n in {128,256,512,1024}, q = 1, 128 trials/size, seed 20260902;
        i.i.d. n = 1024, 128 trials, seed 20260903
        -> stats.k2_slope_q1_plateaugrid, stats.k2_ratio_1024_q1_over_iid
  k3  : layered a = 2 (W_d = (d+1)^2, d_tree = 3), h in {8,12,16} (n = 204, 650, 1496), q = 1,
        192 trials/size, seed 20260902 -> stats.k3_slope_L2_q1 (negative: attractor for d_tree > 2)
  k5  : (control, not a kill) layered a = 1 cone (d_tree = 2), h in {16,23,32,45} (n = 136..1035),
        q = 1, 128 trials/size -> stats.k5_slope_L1_q1 (marginal, ~0)
K4 is tree-only (predict_merge_gamma.py --stage k4).

Usage: python verify/Q-0017/F-01/check_merge.py --mode {k1,k2,k3,k5,all} [--smoke]
Writes verify/Q-0017/F-01/result.json (unless --smoke).
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
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(HERE))

from check_modes import DELTA, MIN_DET, block_residual, fit_slope, rms, sample_iid  # noqa: E402
from driver_numbers import tree_arrays, uniform_rooted_tree  # noqa: E402
from predict_merge_gamma import layered_n, layered_parent, merge_draws  # noqa: E402

SEED = 20260902
K1_SIZES = (8, 16, 32, 64, 128)
K1_TRIALS = 256
K2_SIZES = (128, 256, 512, 1024)
K2_TRIALS = 128
K2_IID_N = 1024
K3_H = (8, 12, 16)
K3_TRIALS = 192
K5_H = (16, 23, 32, 45)
K5_TRIALS = 128

# ---- pre-registered values (tree-only kernel recursion, predictions.json; seed 20260902) and windows
PREREGISTERED = {
    "k1_slope_q1_f02grid": 0.1973,  # predictions.json grid_stage.gamma_grid[q=1.0], SE 0.006
    "k2_slope_q1_plateaugrid": 0.1544,  # plateau_stage.gamma_K2grid[q=1.0], SE 0.025
    "k2_ratio_1024_q1_over_iid": 34.516,  # sqrt(E D_1(1024)/1023), SE 1.7
    "k2_ratio_1024_q05_over_q1": 1.667,  # sqrt(E D_0.5(1024)/E D_1(1024)), SE <= 0.11
    "k3_slope_L2_q1": -0.3789,  # layered_stage.L2.gamma_K3grid[q=1.0], SE 0.0008
    "k5_slope_L1_q1": -0.0474,  # cone h in {16,23,32,45} least-squares of tree-only table (control)
}
WINDOWS = {
    "k1_slope_q1_f02grid": (0.10, 0.30),
    "k2_slope_q1_plateaugrid": (0.05, 0.25),
    "k2_ratio_1024_q1_over_iid": (27.6, 41.4),
    "k2_ratio_1024_q05_over_q1": (1.42, 1.92),
    "k3_slope_L2_q1": (-0.48, -0.28),
    "k5_slope_L1_q1": (-0.15, 0.05),
}


def _load_card_numbers() -> None:
    missing = [k for k, v in WINDOWS.items() if v is None]
    if missing:
        raise SystemExit(f"windows not frozen for {missing}: fill from the card before running")


# ---------------------------------------------------------------- labels
def merge_labels(parent: list[int], level_list, u: np.ndarray, r: np.ndarray, q: float, xi: np.ndarray) -> np.ndarray:
    par = np.asarray(parent, dtype=np.int64)
    labels = np.zeros_like(xi)
    labels[level_list[0]] = xi[level_list[0]]
    for lv in level_list[1:]:
        merged = (u[lv] < q) & (r[lv] >= 0)
        single = lv[~merged]
        both = lv[merged]
        if single.size:
            labels[single] = xi[single] + labels[par[single]]
        if both.size:
            labels[both] = xi[both] + 0.5 * (labels[par[both]] + labels[r[both]])
    return labels


def sample_merge(parent_fn, q: float, rng: np.random.Generator, delta: float) -> float:
    while True:
        parent = parent_fn(rng)
        level_list, _, _, u, r = merge_draws(parent, rng)
        n = len(parent)
        value = block_residual(merge_labels(parent, level_list, u, r, q, rng.normal(size=(n, 4, 4))), delta)
        if math.isfinite(value):
            return value


def run_cayley(sizes, trials, q, delta, seed) -> dict:
    rng = np.random.default_rng(seed)
    out = {"sizes": list(sizes), "q": q, "rms": [], "trials": trials, "delta": delta, "seed": seed}
    for n in sizes:
        out["rms"].append(rms([sample_merge(lambda g, n=n: uniform_rooted_tree(n, g), q, rng, delta) for _ in range(trials)]))
    out["slope"] = fit_slope(sizes, out["rms"])
    return out


def run_layered(a, hs, trials, q, delta, seed) -> dict:
    rng = np.random.default_rng(seed)
    sizes = [layered_n(a, h) for h in hs]
    out = {"a": a, "h": list(hs), "sizes": sizes, "q": q, "rms": [], "trials": trials, "delta": delta, "seed": seed}
    for h in hs:
        out["rms"].append(rms([sample_merge(lambda g, h=h: layered_parent(a, h, g), q, rng, delta) for _ in range(trials)]))
    out["slope"] = fit_slope(sizes, out["rms"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=("k1", "k2", "k3", "k5", "all"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        a = run_cayley((6, 9), 2, 1.0, DELTA, SEED)
        b = run_layered(2, (3, 4), 2, 1.0, DELTA, SEED)
        assert all(math.isfinite(x) for x in a["rms"] + b["rms"])
        print(json.dumps({"smoke": "ok", "cayley_q1": a["rms"], "L2_q1": b["rms"]}))
        return 0
    _load_card_numbers()
    result: dict = {"card": "F-01", "question": "Q-0017", "seed": SEED, "delta": DELTA, "min_det": MIN_DET,
                    "preregistered": PREREGISTERED, "windows": WINDOWS}
    modes = ("k1", "k2", "k3", "k5") if args.mode == "all" else (args.mode,)
    stats: dict[str, float] = {}
    for mode in modes:
        if mode == "k1":
            block = run_cayley(K1_SIZES, K1_TRIALS, 1.0, DELTA, SEED)
            stats["k1_slope_q1_f02grid"] = block["slope"]
        elif mode == "k2":
            block = run_cayley(K2_SIZES, K2_TRIALS, 1.0, DELTA, SEED)
            rng_i = np.random.default_rng(SEED + 1)
            block["rms_iid_1024"] = rms([sample_iid(K2_IID_N, rng_i, DELTA) for _ in range(K2_TRIALS)])
            block["ratio_1024"] = block["rms"][list(K2_SIZES).index(1024)] / block["rms_iid_1024"]
            half = run_cayley((1024,), K2_TRIALS, 0.5, DELTA, SEED + 2)
            block["rms_q05_1024"] = half["rms"][0]
            block["ratio_q05_over_q1_1024"] = half["rms"][0] / block["rms"][list(K2_SIZES).index(1024)]
            stats["k2_slope_q1_plateaugrid"] = block["slope"]
            stats["k2_ratio_1024_q1_over_iid"] = block["ratio_1024"]
            stats["k2_ratio_1024_q05_over_q1"] = block["ratio_q05_over_q1_1024"]
        elif mode == "k3":
            block = run_layered(2, K3_H, K3_TRIALS, 1.0, DELTA, SEED)
            stats["k3_slope_L2_q1"] = block["slope"]
        else:
            block = run_layered(1, K5_H, K5_TRIALS, 1.0, DELTA, SEED)
            stats["k5_slope_L1_q1"] = block["slope"]
        result[mode] = block
    verdict = {}
    for key, value in stats.items():
        lo, hi = WINDOWS[key]
        tag = "KILL" if not (lo <= value <= hi) else "survive"
        if key == "k5_slope_L1_q1":
            tag = "inconsistent" if tag == "KILL" else "consistent"  # control, not a kill
        verdict[key] = tag
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
