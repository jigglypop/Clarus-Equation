"""Simulated annealing over parent arrays (every rooted shape reachable): max and min of c."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a_core import stats_fast  # noqa: E402
import a_fam as F  # noqa: E402


def c_canonical(parent):
    """parent[v] < v for v >= 1, so topological order is 0..n-1."""
    n = len(parent)
    s = np.ones(n)
    par = np.asarray(parent)
    for v in range(n - 1, 0, -1):
        s[par[v]] += s[v]
    w = 1.0 - s / n
    P = np.zeros(n)
    Q = np.zeros(n)
    A2 = np.zeros(n)
    for v in range(1, n):
        p = par[v]
        P[v] = P[p] + w[p]
        Q[v] = Q[p] + w[p] ** 2
        A2[v] = A2[p] + s[p] ** 2
    diag = float(np.sum(s * s * w * w))
    A = diag + 2.0 * float(np.sum(s * s * Q))
    At = diag + 2.0 * float(np.sum(s * s * w * P))
    tot2 = float(np.sum(s * s))
    B = (tot2 * tot2 - float(np.sum(s ** 4)) - 2.0 * float(np.sum(s * s * A2))) / (n * n)
    return (A + B) / At


def anneal(n, sign, iters, seed, start=None):
    rng = np.random.default_rng(seed)
    parent = list(start) if start is not None else [-1] + [i - 1 for i in range(1, n)]
    cur = c_canonical(parent)
    best, best_p = cur, list(parent)
    T0, T1 = 0.05, 1e-4
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        v = int(rng.integers(1, n))
        old = parent[v]
        new = int(rng.integers(0, v))
        if new == old:
            continue
        parent[v] = new
        val = c_canonical(parent)
        if sign * (val - cur) > 0 or rng.random() < np.exp(sign * (val - cur) / T):
            cur = val
            if sign * (cur - best) > 0:
                best, best_p = cur, list(parent)
        else:
            parent[v] = old
    return best, best_p


out = {}
for n in (16, 40, 100, 250):
    starts = {
        "chain": F.chain(n),
        "split2": [-1] + [0, 0] + [1 + (i % 2) for i in range(n - 3)],
        "power": None,
    }
    for sign, tag in ((+1, "max"), (-1, "min")):
        best = (-10.0 if sign > 0 else 10.0, None)
        for name, st in starts.items():
            for seed in (1, 2, 3):
                if st is not None and (max(st[1:]) >= n or len(st) != n):
                    st = None
                v, p = anneal(n, sign, 8000, seed * 977 + n, start=st)
                if sign * (v - best[0]) > 0:
                    best = (v, p)
        out[f"n{n}_{tag}"] = {"c": best[0], "parent_head": best[1][:24] if best[1] else None,
                              "check_stats_fast": stats_fast(best[1])["c"] if best[1] else None}
        print(f"n={n:<4} {tag}: c = {best[0]:.6f}", flush=True)
(HERE / "a9_anneal.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
