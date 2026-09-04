"""a9: independent replicate of kill K4 on a FRESH seed stream (offset 179, not the pre-registered 78).
Measures how much kill power K4 actually has, given that the card already disclosed a 4-tree smoke
(offset 77) returning (-0.007, -0.040, -0.027).  Memory-lean D (block accumulation) so h=90 fits.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
import predict_merge_gamma as P  # noqa

H_LIST = (45, 64, 90)
QS = (0.25, 0.5, 1.0)
TREES = 24
OFFSET = 179


def D_lean(parent, level_list, u, r, q, block=512):
    n = len(parent)
    par = np.asarray(parent, dtype=np.int64)
    A = np.zeros((n, n))
    root = level_list[0]
    A[root, root] = 1.0
    for lv in level_list[1:]:
        merged = (u[lv] < q) & (r[lv] >= 0)
        single = lv[~merged]
        both = lv[merged]
        if single.size:
            A[single] = A[par[single]]
        if both.size:
            A[both] = 0.5 * (A[par[both]] + A[r[both]])
        A[lv, lv] = 1.0
    A -= A.mean(axis=0, keepdims=True)
    tot = 0.0
    for j0 in range(0, n, block):
        Gb = A.T @ A[:, j0:j0 + block]
        tot += float(np.sum(Gb * Gb))
    del A
    return tot


def main():
    out = {"offset": OFFSET, "trees": TREES, "h": list(H_LIST), "q": list(QS), "rows": []}
    for h in H_LIST:
        n = P.layered_n(1, h)
        rng = np.random.default_rng(P.SEED + 1000 * n + OFFSET)
        acc = np.zeros((TREES, len(QS)))
        t0 = time.time()
        for t in range(TREES):
            parent = P.layered_parent(1, h, rng)
            level_list, widths, depth, u, r = P.merge_draws(parent, rng)
            for j, q in enumerate(QS):
                acc[t, j] = D_lean(parent, level_list, u, r, q) / n ** 2
        out["rows"].append({"h": h, "n": n, "E_D_over_n2": acc.mean(axis=0).tolist(),
                            "se": (acc.std(axis=0, ddof=1) / math.sqrt(TREES)).tolist(),
                            "wall_s": time.time() - t0})
        print(json.dumps(out["rows"][-1]), flush=True)
    ns = [r["n"] for r in out["rows"]]
    gam = []
    for j in range(len(QS)):
        ys = [r["E_D_over_n2"][j] for r in out["rows"]]
        gam.append(float(np.polyfit(np.log(ns), 0.5 * np.log(ys), 1)[0]))
    out["gamma_fit_all_sizes"] = gam
    out["window"] = list(P.K4_WINDOW)
    out["verdict"] = {str(q): ("KILL" if not (P.K4_WINDOW[0] <= g <= P.K4_WINDOW[1]) else "survive")
                      for q, g in zip(QS, gam)}
    out["card_smoke_offset77"] = [-0.007, -0.040, -0.027]
    print(json.dumps({k: out[k] for k in ("gamma_fit_all_sizes", "window", "verdict")}, indent=1))
    (HERE / "a9_k4_replicate.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
