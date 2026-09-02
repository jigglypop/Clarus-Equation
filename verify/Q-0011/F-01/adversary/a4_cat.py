"""Exact O(m) evaluator for 'spine + hung leaves' trees, given the spine subtree-size list.

Tree: root (size n) -> v_1 -> ... -> v_m with prescribed subtree sizes s_1 > ... > s_m,
every size drop realised by LEAVES hung at the corresponding spine vertex.
Validated against the O(n) driver, the literal matrix definition and the card's tree_stats
(a4_check.py: max |dc| = 5e-15).
"""
from __future__ import annotations

import numpy as np


def cat_stats(n: int, sizes) -> dict:
    s = np.asarray(sizes, dtype=np.int64)
    s = np.unique(s)[::-1].astype(np.float64)
    s = s[(s >= 1) & (s <= n - 1)]
    m = len(s)
    if m == 0:
        raise ValueError("empty spine")
    w = 1.0 - s / n
    wL = 1.0 - 1.0 / n
    L0 = float(n - 1 - s[0])
    Li = s - np.concatenate([s[1:], [0.0]]) - 1.0
    P = np.concatenate([[0.0], np.cumsum(w)[:-1]])
    Q = np.concatenate([[0.0], np.cumsum(w ** 2)[:-1]])
    diag = float(np.sum(s ** 2 * w ** 2)) + (float(np.sum(Li)) + L0) * wL ** 2
    A = diag + 2.0 * (float(np.sum(s ** 2 * Q)) + float(np.sum(Li * (Q + w ** 2))))
    At = diag + 2.0 * (float(np.sum(s ** 2 * w * P)) + wL * float(np.sum(Li * (P + w))))
    Ltot = float(np.sum(Li)) + L0
    s2 = s ** 2
    tail = np.concatenate([np.cumsum(s2[::-1])[::-1][1:], [0.0]])
    B = (2.0 / n ** 2) * (Ltot * (Ltot - 1.0) / 2.0 + L0 * float(np.sum(s2)) + float(np.sum(Li * tail)))
    D = A + B
    return {"n": n, "m": m, "D": D, "A": A, "B": B, "At": At, "c": D / At, "L_tot": Ltot}


def cat_parent(n: int, sizes):
    """Explicit parent array of the same tree (only for small/medium n)."""
    s = sorted({int(x) for x in sizes if 1 <= int(x) <= n - 1}, reverse=True)
    parent = [-1]
    spine = []
    prev = 0
    for sz in s:
        parent.append(prev)
        v = len(parent) - 1
        spine.append(v)
        prev = v
    for i, sz in enumerate(s):
        nxt = s[i + 1] if i + 1 < len(s) else 0
        for _ in range(sz - nxt - 1):
            parent.append(spine[i])
    while len(parent) < n:
        parent.append(0)
    if len(parent) != n:
        raise ValueError(f"built {len(parent)} != {n}")
    return parent
