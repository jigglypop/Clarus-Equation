"""Q-0008 F-01 kill script: three mismatch modes of one coarse block.

Pre-registered (card derivations/Q-0008/F-01.formula.md, 2026-09-02). Windows, seed,
sizes and perturbation scale are frozen; do not edit after seeing results.

Modes
  iid    : each cell tetrad = I + delta * xi_v, xi_v i.i.d. N(0,1)^{4x4}      -> slope pre-registered -0.5
  her    : uniform rooted Cayley tree (Pruefer), delta e_v = delta * sum_{u on root..v} xi_u
                                                                           -> slope pre-registered 0.2261 (+-0.1)
  coh    : one fixed mismatched candidate repeated r times (13.5)           -> slope pre-registered 0.0 (+-0.1)
  ratio  : RMS_her / RMS_iid at n=32, same delta                            -> pre-registered 11.1528 (+-20%)

Usage: python verify/Q-0008/F-01/check_modes.py --mode {iid,her,coh,ratio,all} [--smoke]
Writes verify/Q-0008/F-01/result.json (unless --smoke).
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    simplicity_residual,
)
from examples.physics.gravity.urbantke_shape_matching_rg import (  # noqa: E402
    optimal_internal_alignment,
    repeated_coherent_mismatch_residual,
)

SEED = 20260902
SIZES = (8, 16, 32, 64, 128)
TRIALS = 64
DELTA = 0.02
RATIO_N = 32
RATIO_TRIALS = 256
COH_REPEATS = (1, 4, 16, 64, 256)
COH_PERTURBATION = 0.35
MIN_DET = 0.05

WINDOWS = {
    "iid": (-0.6, -0.4),
    "her": (0.126, 0.326),
    "coh": (-0.1, 0.1),
    "ratio": (8.92, 13.38),
}
PREREGISTERED = {"iid": -0.5, "her": 0.2261, "coh": 0.0, "ratio": 11.1528}

REFERENCE = geometric_self_dual_triple(np.eye(4))


def expected_w2(n: int) -> float:
    """Exact E[sum_u |sub(u)|^2] for a uniform rooted labelled tree on n vertices."""
    total = Fraction(n * n)
    for k in range(1, n):
        total += Fraction(math.comb(n, k) * k ** (k + 1) * (n - k) ** (n - k), n ** (n - 1))
    return float(total)


def uniform_rooted_tree(n: int, rng: np.random.Generator) -> list[int]:
    """Return parent array (parent[root] = -1) of a uniform rooted labelled tree via Pruefer."""
    if n == 1:
        return [-1]
    if n == 2:
        root = int(rng.integers(0, 2))
        return [-1, 0] if root == 0 else [1, -1]
    seq = rng.integers(0, n, size=n - 2)
    degree = np.ones(n, dtype=int)
    for s in seq:
        degree[s] += 1
    adjacency: list[list[int]] = [[] for _ in range(n)]
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    for s in seq:
        leaf = heapq.heappop(leaves)
        adjacency[leaf].append(int(s))
        adjacency[int(s)].append(leaf)
        degree[s] -= 1
        if degree[s] == 1:
            heapq.heappush(leaves, int(s))
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    adjacency[u].append(v)
    adjacency[v].append(u)
    root = int(rng.integers(0, n))
    parent = [-2] * n
    parent[root] = -1
    stack = [root]
    while stack:
        x = stack.pop()
        for y in adjacency[x]:
            if parent[y] == -2:
                parent[y] = x
                stack.append(y)
    return parent


def block_residual_from_tetrad_perturbations(perturbations: np.ndarray) -> float:
    blocked = np.zeros_like(REFERENCE)
    for de in perturbations:
        tetrad = np.eye(4) + de
        if float(np.linalg.det(tetrad)) <= MIN_DET:
            return math.nan
        candidate = geometric_self_dual_triple(tetrad)
        blocked += optimal_internal_alignment(REFERENCE, candidate).aligned_candidate
    return simplicity_residual(blocked)


def sample_iid(n: int, rng: np.random.Generator, delta: float) -> float:
    while True:
        xi = rng.normal(size=(n, 4, 4))
        value = block_residual_from_tetrad_perturbations(delta * xi)
        if math.isfinite(value):
            return value


def sample_her(n: int, rng: np.random.Generator, delta: float) -> float:
    while True:
        parent = uniform_rooted_tree(n, rng)
        xi = rng.normal(size=(n, 4, 4))
        labels = np.zeros((n, 4, 4))
        children: list[list[int]] = [[] for _ in range(n)]
        for v, p in enumerate(parent):
            if p >= 0:
                children[p].append(v)
        root = parent.index(-1)
        order = [root]
        idx = 0
        while idx < len(order):
            x = order[idx]
            idx += 1
            order.extend(children[x])
        for v in order:
            p = parent[v]
            labels[v] = xi[v] + (labels[p] if p >= 0 else 0.0)
        value = block_residual_from_tetrad_perturbations(delta * labels)
        if math.isfinite(value):
            return value


def rms(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def fit_slope(sizes, values) -> float:
    return float(np.polyfit(np.log(np.asarray(sizes, dtype=float)), np.log(np.asarray(values, dtype=float)), 1)[0])


def run_mode(mode: str, sizes, trials: int, delta: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sampler = sample_iid if mode == "iid" else sample_her
    out = {"sizes": list(sizes), "rms": [], "mean": [], "trials": trials, "delta": delta, "seed": seed}
    for n in sizes:
        vals = [sampler(n, rng, delta) for _ in range(trials)]
        out["rms"].append(rms(vals))
        out["mean"].append(float(np.mean(vals)))
    out["slope"] = fit_slope(sizes, out["rms"])
    out["slope_mean_based"] = fit_slope(sizes, out["mean"])
    if mode == "her":
        out["exact_linear_prediction_slope"] = fit_slope(sizes, [math.sqrt(expected_w2(n)) / n for n in sizes])
    return out


def run_coh(seed: int, repeats) -> dict:
    rng = np.random.default_rng(seed)
    while True:
        tetrad = np.eye(4) + COH_PERTURBATION * rng.normal(size=(4, 4))
        if float(np.linalg.det(tetrad)) > 0.2:
            break
    candidate = geometric_self_dual_triple(tetrad)
    residuals = [repeated_coherent_mismatch_residual(REFERENCE, candidate, repeats=r) for r in repeats]
    return {"repeats": list(repeats), "residuals": residuals, "slope": fit_slope(repeats, residuals), "seed": seed}


def run_ratio(n: int, trials: int, delta: float, seed: int) -> dict:
    rng_i = np.random.default_rng(seed)
    rng_h = np.random.default_rng(seed + 1)
    iid = [sample_iid(n, rng_i, delta) for _ in range(trials)]
    her = [sample_her(n, rng_h, delta) for _ in range(trials)]
    r_i, r_h = rms(iid), rms(her)
    return {
        "n": n,
        "trials": trials,
        "delta": delta,
        "rms_iid": r_i,
        "rms_her": r_h,
        "rms_her_over_iid": r_h / r_i,
        "exact_linear_prediction": math.sqrt(expected_w2(n) / n),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=("iid", "her", "coh", "ratio", "all"))
    parser.add_argument("--smoke", action="store_true", help="execution check only (tiny sizes, no verdict)")
    args = parser.parse_args()

    if args.smoke:
        run_mode("iid", (4, 6), 2, DELTA, SEED)
        run_mode("her", (4, 6), 2, DELTA, SEED)
        run_coh(SEED, (1, 2))
        run_ratio(4, 2, DELTA, SEED)
        assert abs(expected_w2(3) - 13.0) < 1e-12
        print("smoke ok")
        return 0

    result: dict = {"card": "F-01", "question": "Q-0008", "seed": SEED, "preregistered": PREREGISTERED, "windows": WINDOWS}
    modes = ("iid", "her", "coh", "ratio") if args.mode == "all" else (args.mode,)
    verdict = {}
    for mode in modes:
        if mode in ("iid", "her"):
            block = run_mode(mode, SIZES, TRIALS, DELTA, SEED)
            stat = block["slope"]
        elif mode == "coh":
            block = run_coh(SEED, COH_REPEATS)
            stat = block["slope"]
        else:
            block = run_ratio(RATIO_N, RATIO_TRIALS, DELTA, SEED)
            stat = block["rms_her_over_iid"]
        lo, hi = WINDOWS[mode]
        block["window"] = [lo, hi]
        block["killed"] = not (lo <= stat <= hi)
        verdict[mode] = "KILL" if block["killed"] else "survive"
        result[mode] = block
    result["verdict"] = verdict
    out = Path(__file__).resolve().parent / "result.json"
    existing = {}
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update({k: v for k, v in result.items() if k != "verdict"})
    existing.setdefault("verdict", {}).update(verdict)
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {m: (result[m].get("slope", result[m].get("rms_her_over_iid"))) for m in modes}
    print(json.dumps({"verdict": verdict, **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
