"""Adversary core for Q-0011 F-01: INDEPENDENT reimplementation of D, A, B, A_tilde, c.

Nothing here imports the card's check_families.tree_stats; the O(n) driver is re-derived
from the definitions and cross-checked against the literal matrix definition
    kappa = A A^T,  A[v,u] = 1 iff u ancestor-or-self of v,   D = ||H kappa H||_F^2
    kappa_eff = A W A^T,  W = diag(1 - s_u/n),   n^2 mu2_eff = ||kappa_eff||_F^2
    c = D / (n^2 mu2_eff)
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def tree_arrays(parent):
    n = len(parent)
    children = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        if p >= 0:
            children[p].append(v)
        else:
            root = v
    order = [root]
    i = 0
    while i < len(order):
        order.extend(children[order[i]])
        i += 1
    depth = np.zeros(n, dtype=np.int64)
    sub = np.ones(n, dtype=np.int64)
    for v in order[1:]:
        depth[v] = depth[parent[v]] + 1
    for v in reversed(order):
        if parent[v] >= 0:
            sub[parent[v]] += sub[v]
    return order, depth, sub


def stats_matrix(parent):
    n = len(parent)
    order, _, sub = tree_arrays(parent)
    Amat = np.zeros((n, n))
    for v in order:
        if parent[v] >= 0:
            Amat[v] = Amat[parent[v]]
        Amat[v, v] = 1.0
    kappa = Amat @ Amat.T
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ kappa @ H
    D = float(np.sum(K * K))
    w = 1.0 - sub.astype(float) / n
    keff = (Amat * w) @ Amat.T
    at = float(np.sum(keff * keff))
    return {"n": n, "D": D, "n2_mu2_eff": at, "c": D / at if at > 0 else float("nan")}


def stats_fast(parent):
    n = len(parent)
    order, depth, sub = tree_arrays(parent)
    s = sub.astype(np.float64)
    w = 1.0 - s / n
    p_sum = np.zeros(n)
    q_sum = np.zeros(n)
    anc2 = np.zeros(n)
    for v in order:
        par = parent[v]
        if par >= 0:
            p_sum[v] = p_sum[par] + w[par]
            q_sum[v] = q_sum[par] + w[par] ** 2
            anc2[v] = anc2[par] + s[par] ** 2
    diag = float(np.sum(s * s * w * w))
    A = diag + 2.0 * float(np.sum(s * s * q_sum))
    At = diag + 2.0 * float(np.sum(s * s * w * p_sum))
    tot2 = float(np.sum(s * s))
    sum4 = float(np.sum(s ** 4))
    nested_cross = 2.0 * float(np.sum(s * s * anc2))
    B = (tot2 * tot2 - sum4 - nested_cross) / (n * n)
    D = A + B
    return {"n": float(n), "D": D, "A": A, "B": B, "n2_mu2_eff": At,
            "mu2_eff": At / n ** 2, "c": D / At if At > 0 else float("nan"),
            "max_depth": float(depth.max()), "sum_s2": tot2}


def stats_exact(parent):
    n = len(parent)
    order, _, sub = tree_arrays(parent)
    s = [Fraction(int(x)) for x in sub]
    N = Fraction(n)
    w = [1 - si / N for si in s]
    p_sum = [Fraction(0)] * n
    q_sum = [Fraction(0)] * n
    a2 = [Fraction(0)] * n
    for v in order:
        par = parent[v]
        if par >= 0:
            p_sum[v] = p_sum[par] + w[par]
            q_sum[v] = q_sum[par] + w[par] ** 2
            a2[v] = a2[par] + s[par] ** 2
    diag = sum(s[v] ** 2 * w[v] ** 2 for v in range(n))
    A = diag + 2 * sum(s[v] ** 2 * q_sum[v] for v in range(n))
    At = diag + 2 * sum(s[v] ** 2 * w[v] * p_sum[v] for v in range(n))
    tot2 = sum(si ** 2 for si in s)
    sum4 = sum(si ** 4 for si in s)
    nested = 2 * sum(s[v] ** 2 * a2[v] for v in range(n))
    B = (tot2 ** 2 - sum4 - nested) / N ** 2
    D = A + B
    return {"D": D, "A": A, "B": B, "At": At, "c": D / At}


def c_of(parent):
    return stats_fast(parent)["c"]
