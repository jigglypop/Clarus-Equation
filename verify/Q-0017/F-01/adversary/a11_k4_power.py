"""a11: kill power of K4.  Fresh stream (offset 179), full pre-registered grid h in {45,64,90},
q in {0.25,0.5,1.0}, 24 trees, float32 D (relative error ~1e-6, irrelevant for a slope).
Also bootstraps the 3-point slope to estimate the false-kill probability of the +-0.15 window.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
import predict_merge_gamma as P  # noqa

H_LIST, QS, TREES, OFFSET = (45, 64, 90), (0.25, 0.5, 1.0), 24, 179


def D_lean(parent, level_list, u, r, q, block=768):
    n = len(parent)
    par = np.asarray(parent, dtype=np.int64)
    A = np.zeros((n, n), dtype=np.float32)
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
        tot += float(np.sum(Gb.astype(np.float64) ** 2))
    del A
    return tot


def main():
    per = {}
    ns = []
    for h in H_LIST:
        n = P.layered_n(1, h)
        ns.append(n)
        rng = np.random.default_rng(P.SEED + 1000 * n + OFFSET)
        acc = np.zeros((TREES, len(QS)))
        t0 = time.time()
        for t in range(TREES):
            parent = P.layered_parent(1, h, rng)
            level_list, widths, depth, u, r = P.merge_draws(parent, rng)
            for j, q in enumerate(QS):
                acc[t, j] = D_lean(parent, level_list, u, r, q) / n ** 2
        per[n] = acc
        print(json.dumps({"h": h, "n": n, "mean": acc.mean(axis=0).tolist(),
                          "se": (acc.std(axis=0, ddof=1) / math.sqrt(TREES)).tolist(),
                          "wall_s": time.time() - t0}), flush=True)
    logn = np.log(np.asarray(ns, float))
    X = np.vstack([logn, np.ones_like(logn)]).T
    gam = [float(np.linalg.lstsq(X, 0.5 * np.log([per[n][:, j].mean() for n in ns]), rcond=None)[0][0])
           for j in range(len(QS))]
    rng = np.random.default_rng(20260903)
    B = 2000
    gb = np.zeros((B, len(QS)))
    for b in range(B):
        ys = []
        for n in ns:
            idx = rng.integers(0, TREES, size=TREES)
            ys.append(0.5 * np.log(per[n][idx].mean(axis=0)))
        gb[b] = np.linalg.lstsq(X, np.stack(ys), rcond=None)[0][0]
    lo, hi = P.K4_WINDOW
    out = {"offset": OFFSET, "sizes": ns, "q": list(QS), "trees": TREES,
           "gamma_fit_all_sizes": gam, "boot_se": gb.std(axis=0, ddof=1).tolist(),
           "window": [lo, hi],
           "verdict": {str(q): ("KILL" if not (lo <= g <= hi) else "survive") for q, g in zip(QS, gam)},
           "P_outside_window_bootstrap": {str(q): float(np.mean((gb[:, j] < lo) | (gb[:, j] > hi)))
                                          for j, q in enumerate(QS)},
           "card_smoke_offset77": [-0.007, -0.040, -0.027],
           "E_D_over_n2": {str(n): per[n].mean(axis=0).tolist() for n in ns}}
    print(json.dumps({k: out[k] for k in ("gamma_fit_all_sizes", "boot_se", "verdict",
                                          "P_outside_window_bootstrap")}, indent=1))
    (HERE / "a11_k4_power.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
